"""E4b review, check 2: the headline numbers recomputed from results/preds with the reviewer's own scoring (numpy only: no flood.evaluate), for the four new sets and the
three stored sets, seed means and every single seed: pooled onset and all-day warning tables at trigger 1.0, median NSE, leads 1-3, test years; the seed ranges and the seed
rule; the per-basin picture and the Wilcoxon p of the gain rule; the paired onsets; the closure of the gap to wqaP; the validation years (trigger chosen by the CSI rule on two
grids and by the 30%-false rule, replicated from code/warning_policy.py and the script, and scored once on the test years); the verdict words; the little-rain slice; the
seed means against the single seeds; the stored sets' own results files. Every number compared with results/within_day_inputs.json. Needs OUT/my_columns.npz (r3). About a minute."""
from r_common import *
from scipy.stats import binomtest, wilcoxon

t0 = time.time()
R = jload(RESULTS)
S = R["summary_test"]["sets"]
C = Compare(tol=1e-9)
MY = np.load(OUT / "my_columns.npz", allow_pickle=False)
ids = [str(b) for b in MY["basins"]]

# ---- the data: dates, observed flow (identical across sets), thresholds from the training years ---------------------------------------------------------------
dates, basins, OBS, _, _ = load_npz("lstm_wqa")
assert basins == ids and dates.equals(pd.DatetimeIndex(MY["days"]))
TR, VA, TE = (within(dates, *SPLIT[p]) for p in ("train", "val", "test"))
THR = np.array([float(np.quantile(OBS[i][TR], 0.95)) for i in range(len(ids))])
for i, g in enumerate(ids):
    C.add(f"thresholds {g}", R["thresholds"][g], THR[i])
PRED, OBS_OF = {}, {}
for n in SHOWN:
    for name in [f"lstm_{n}"] + [f"lstm_{n}_s{s}" for s in SEEDS[n]]:
        d, b, o, p, _ = load_npz(name)
        assert b == ids and d.equals(dates), name
        PRED[name], OBS_OF[name] = p, o
obs_same = all(np.array_equal(OBS_OF[k], OBS) for k in OBS_OF)
C.add("observed flow identical in every set and seed file", True, obs_same)


def tables(p, part, trigger=1.0):
    """Per lead: per-basin NSE and the pooled onset / all-day tables at one trigger on the target days of one split part (own code)."""
    part_mask = {"train": TR, "val": VA, "test": TE}[part]
    res = {}
    for j, k in enumerate(LEADS):
        H = dict(hits=0, misses=0, false_alarms=0, correct_neg=0)
        Ho = dict(H)
        nse, pod_b, far_b = [], [], []
        for i in range(len(ids)):
            o, f = OBS[i], p[i, :, j]
            issue = np.concatenate([np.full(k, np.nan), o[:-k]])                                   # the observed flow k days before the target day
            ok = part_mask & np.isfinite(o) & np.isfinite(f)
            oo, ff, ii = o[ok], f[ok], issue[ok]
            nse.append(1.0 - np.sum((ff - oo) ** 2) / np.sum((oo - oo.mean()) ** 2))
            hi, warn, low = oo > THR[i], ff > trigger * THR[i], ii <= THR[i]                      # NaN issue flow: not an onset (as flood.flood_scores)
            for T, m in ((H, np.ones(len(oo), bool)), (Ho, low)):
                T["hits"] += int((hi & warn & m).sum()); T["misses"] += int((hi & ~warn & m).sum()); T["false_alarms"] += int((~hi & warn & m).sum()); T["correct_neg"] += int((~hi & ~warn & m).sum())
            h_, m_, f_ = int((hi & warn & low).sum()), int((hi & ~warn & low).sum()), int((~hi & warn & low).sum())
            pod_b.append(h_ / (h_ + m_) if h_ + m_ else np.nan); far_b.append(f_ / (h_ + f_) if h_ + f_ else np.nan)
        rates = lambda T: dict(T, pod=T["hits"] / (T["hits"] + T["misses"]) if T["hits"] + T["misses"] else np.nan, far=T["false_alarms"] / (T["hits"] + T["false_alarms"]) if T["hits"] + T["false_alarms"] else np.nan,
                               csi=T["hits"] / (T["hits"] + T["misses"] + T["false_alarms"]) if T["hits"] + T["misses"] + T["false_alarms"] else np.nan)
        res[f"l{k}"] = dict(nse=np.array(nse), nse_median=float(np.median(nse)), nse_mean=float(np.mean(nse)), onset=rates(Ho), all=rates(H), onset_pod_by_basin=pod_b, onset_far_by_basin=far_b)
    return res


EV = {name: tables(PRED[name], "test") for name in PRED}
EV_VAL = {f"lstm_{n}": tables(PRED[f"lstm_{n}"], "val") for n in SHOWN}
# 1. summary_test (seed means), validation summary, per basin
for n in SHOWN:
    for k in LEADS:
        m, f = EV[f"lstm_{n}"][f"l{k}"], S[n][f"l{k}"]
        C.add(f"{n} l{k} test nse_median", f["nse_median"], m["nse_median"]); C.add(f"{n} l{k} test nse_mean", f["nse_mean"], m["nse_mean"])
        for kind in ("onset", "all"):
            for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                C.add(f"{n} l{k} test {kind} {c}", f[kind][c], m[kind][c])
        mv, fv = EV_VAL[f"lstm_{n}"][f"l{k}"], R["validation_used_for_no_choice"]["summary"][n][f"l{k}"]
        C.add(f"{n} l{k} val nse_median", fv["nse_median"], mv["nse_median"])
        for c in ("hits", "pod", "far", "csi"):
            C.add(f"{n} l{k} val onset {c}", fv["onset"][c], mv["onset"][c])
        for i, g in enumerate(ids):
            C.add(f"{n} l{k} per_basin nse {g}", R["per_basin"]["nse"][n][f"l{k}"][g], m["nse"][i])
            C.add(f"{n} l{k} per_basin onset_warned {g}", R["per_basin"]["onset_warned"][n][f"l{k}"][g], m["onset_pod_by_basin"][i])
            C.add(f"{n} l{k} per_basin onset_false {g}", R["per_basin"]["onset_false"][n][f"l{k}"][g], m["onset_far_by_basin"][i])
# 2. seed ranges and the seed means against the single seeds
seed_mean_check = {}
for n in SHOWN:
    rg = R["seed_ranges"]["sets"][n]
    for k in LEADS:
        vals = {c: [] for c in ("nse_median", "onset_pod", "onset_far", "onset_csi", "all_pod", "all_far", "all_csi")}
        for s in SEEDS[n]:
            e = EV[f"lstm_{n}_s{s}"][f"l{k}"]
            vals["nse_median"].append(e["nse_median"])
            for kind in ("onset", "all"):
                for c in ("pod", "far", "csi"):
                    vals[f"{kind}_{c}"].append(e[kind][c])
        C.add(f"{n} l{k} n_seeds", rg[f"l{k}"]["n_seeds"], len(SEEDS[n]))
        for c, v in vals.items():
            C.add(f"{n} l{k} seed_range {c} lo", rg[f"l{k}"][c][0], min(v)); C.add(f"{n} l{k} seed_range {c} hi", rg[f"l{k}"][c][1], max(v))
            for j, s in enumerate(SEEDS[n]):
                C.add(f"{n} l{k} by_seed {c} s{s}", rg[f"l{k}"]["by_seed"][c][j], v[j])
    mean = np.mean([PRED[f"lstm_{n}_s{s}"] for s in SEEDS[n]], axis=0)
    a = PRED[f"lstm_{n}"]
    fin = np.isfinite(a)
    seed_mean_check[n] = dict(same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(mean))), max_abs_diff=float(np.abs(a[fin] - mean[fin]).max()), n_values=int(fin.sum()))
C.add("seed means equal the day-by-day mean of the seed files (max abs diff over 7 sets)", 0.0, max(v["max_abs_diff"] for v in seed_mean_check.values()), tol=1e-12)
# 3. the seed rule, the gain rule (NSE: every seed above, most basins, Wilcoxon p < 0.05), the readings
above = lambda a, b: min(a) > max(b)
pods = lambda n, k: [EV[f"lstm_{n}_s{s}"][f"l{k}"]["onset"]["pod"] for s in SEEDS[n]]
nses = lambda n, k: [EV[f"lstm_{n}_s{s}"][f"l{k}"]["nse_median"] for s in SEEDS[n]]
PAIRS = [("within_day_columns", "wqaW", "wqa"), ("within_day_against_the_older_control", "wqaW", "wqaW_older"), ("older_control_against_wqa", "wqaW_older", "wqa"), ("q_rel_alone", "wqaQ", "wqa"),
         ("within_day_against_q_rel", "wqaW", "wqaQ"), ("how_far_from_perfect_rain", "wqaW", "wqaP"), ("within_day_with_the_forecast", "wqaWF_sub", "wqaF_sub"), ("forecast_on_top_of_within_day", "wqaWF_sub", "wqaW")]
rule_mine = {}
for key, T, Cc in PAIRS:
    for k in LEADS:
        rd = R["readings"][key]["by_lead"][f"l{k}"]
        a, b = EV[f"lstm_{T}"][f"l{k}"]["nse"], EV[f"lstm_{Cc}"][f"l{k}"]["nse"]
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
        up, down = above(nses(T, k), nses(Cc, k)), above(nses(Cc, k), nses(T, k))
        n_hi, n_lo = int((d > 0).sum()), int((d < 0).sum())
        gain = up and n_hi > 6 and p < 0.05
        loss = down and n_lo > 6 and p < 0.05
        o_up, o_down = above(pods(T, k), pods(Cc, k)), above(pods(Cc, k), pods(T, k))
        mine = dict(nse_reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference", onset_reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference",
                    n_basins_higher=n_hi, median_basin_difference=float(np.median(d)), wilcoxon_p=p, every_seed_above_nse=up, every_seed_above_onsets=o_up, every_seed_below_onsets=o_down)
        rule_mine[f"{key} l{k}"] = mine
        C.add(f"readings {key} l{k} onsets every_seed_above_every_seed", rd["onsets"]["every_seed_above_every_seed"], o_up)
        C.add(f"readings {key} l{k} onsets every_seed_below_every_seed", rd["onsets"]["every_seed_below_every_seed"], o_down)
        C.add(f"readings {key} l{k} onsets reading", rd["onsets"]["reading"], mine["onset_reading"])
        C.add(f"readings {key} l{k} onsets difference_points", rd["onsets"]["difference_points"], 100 * (EV[f"lstm_{T}"][f"l{k}"]["onset"]["pod"] - EV[f"lstm_{Cc}"][f"l{k}"]["onset"]["pod"]))
        C.add(f"readings {key} l{k} onsets hits_difference", rd["onsets"]["hits_difference"], EV[f"lstm_{T}"][f"l{k}"]["onset"]["hits"] - EV[f"lstm_{Cc}"][f"l{k}"]["onset"]["hits"])
        C.add(f"readings {key} l{k} nse gain", rd["nse"]["gain"], EV[f"lstm_{T}"][f"l{k}"]["nse_median"] - EV[f"lstm_{Cc}"][f"l{k}"]["nse_median"])
        C.add(f"readings {key} l{k} nse n_basins_higher", rd["nse"]["n_basins_higher"], n_hi)
        C.add(f"readings {key} l{k} nse median_basin_difference", rd["nse"]["median_basin_difference"], float(np.median(d)))
        C.add(f"readings {key} l{k} nse wilcoxon_p", rd["nse"]["wilcoxon_p"], p)
        C.add(f"readings {key} l{k} nse reading", rd["nse"]["reading"], mine["nse_reading"])
        C.add(f"readings {key} l{k} nse every_seed_above_every_seed", rd["nse"]["every_seed_above_every_seed"], up)
        # the seed-by-seed difference lists: how many entries do they hold (zip truncation to the shorter seed list)?
        C.add(f"readings {key} l{k} difference_by_seed_points length", len(rd["onsets"]["difference_by_seed_points"]), min(len(SEEDS[T]), len(SEEDS[Cc])))
        if T in R["gain_rule"]["sets"] and f"against_{Cc}" in R["gain_rule"]["sets"][T]:
            gr = R["gain_rule"]["sets"][T][f"against_{Cc}"][f"l{k}"]
            C.add(f"gain_rule {T} against {Cc} l{k} nse clear_gain", gr["nse"]["clear_gain"], gain); C.add(f"gain_rule {T} against {Cc} l{k} nse reading", gr["nse"]["reading"], mine["nse_reading"])
            C.add(f"gain_rule {T} against {Cc} l{k} onset reading", gr["onset"]["reading"], mine["onset_reading"]); C.add(f"gain_rule {T} against {Cc} l{k} nse wilcoxon_p", gr["nse"]["wilcoxon_p"], p)
# the E4b success record
E = R["e4b_success"]
r1 = {x: pods(x, 1) for x in ("wqaW", "wqa", "wqaW_older", "wqaQ", "wqaP")}
mine_success = dict(every_seed_of_wqaW_above_every_seed_of_wqa=above(r1["wqaW"], r1["wqa"]), every_seed_of_wqaW_above_every_seed_of_the_older_control=above(r1["wqaW"], r1["wqaW_older"]),
                    control_flat=not above(r1["wqaW_older"], r1["wqa"]), control_below_wqa_in_every_seed=above(r1["wqa"], r1["wqaW_older"]), every_seed_of_wqaQ_above_every_seed_of_wqa=above(r1["wqaQ"], r1["wqa"]),
                    every_seed_of_wqaW_above_every_seed_of_wqaQ=above(r1["wqaW"], r1["wqaQ"]))
mine_success["met"] = mine_success["every_seed_of_wqaW_above_every_seed_of_wqa"] and mine_success["every_seed_of_wqaW_above_every_seed_of_the_older_control"]
for k, v in mine_success.items():
    C.add(f"e4b_success {k}", E[k], v)
for x in r1:
    C.add(f"e4b_success seed_ranges {x} lo", E["seed_ranges"][x][0], min(r1[x])); C.add(f"e4b_success seed_ranges {x} hi", E["seed_ranges"][x][1], max(r1[x]))
    C.add(f"e4b_success onset_warned {x}", E["onset_warned"][x], EV[f"lstm_{x}"]["l1"]["onset"]["pod"]); C.add(f"e4b_success n_seeds {x}", E["n_seeds"][x], len(SEEDS[x]))
# the rule with the first three seeds of wqaW and of the control only (the seed count of the stored base), as a reading
three = dict(wqaW_seeds_0_2=[round(v, 4) for v in r1["wqaW"][:3]], met_with_three_seeds=above(r1["wqaW"][:3], r1["wqa"]) and above(r1["wqaW"][:3], r1["wqaW_older"][:3]),
             wqaW_above_wqa_with_three_seeds=above(r1["wqaW"][:3], r1["wqa"]), wqaW_above_older_with_three_seeds=above(r1["wqaW"][:3], r1["wqaW_older"][:3]),
             seeds_of_wqaW_below_the_best_seed_of_wqa=[s for s, v in zip(SEEDS["wqaW"], r1["wqaW"]) if v <= max(r1["wqa"])], best_seed_of_wqa=float(max(r1["wqa"])),
             seeds_of_wqaW_below_the_best_seed_of_the_control=[s for s, v in zip(SEEDS["wqaW"], r1["wqaW"]) if v <= max(r1["wqaW_older"])], best_seed_of_control=float(max(r1["wqaW_older"])))
# 4. paired onsets (flags per onset, basin after basin, target days of the test years)
def flags(p, k, trigger=1.0):
    out = []
    j = k - 1
    for i in range(len(ids)):
        o, f = OBS[i], p[i, :, j]
        issue = np.concatenate([np.full(k, np.nan), o[:-k]])
        ok = TE & np.isfinite(o) & np.isfinite(f)
        m = ok & (o > THR[i]) & (issue <= THR[i])
        out.append((f > trigger * THR[i])[m])
    return np.concatenate(out)
for key, T, Cc in PAIRS:
    for k in LEADS:
        fa, fc = flags(PRED[f"lstm_{T}"], k), flags(PRED[f"lstm_{Cc}"], k)
        both, oa, oc, nei = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum()), int((~fa & ~fc).sum())
        pp = R["paired_onsets"]["pairs"][key][f"l{k}"]
        for name, v in (("n_onsets", len(fa)), ("hits_first", int(fa.sum())), ("hits_second", int(fc.sum())), ("warned_by_both", both), ("warned_only_by_first", oa), ("warned_only_by_second", oc), ("warned_by_neither", nei),
                        ("sign_test_p", float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))):
            C.add(f"paired {key} l{k} {name}", pp[name], v)
# 5. closure of the gap to wqaP
def closure(name, base, ceil):
    rows = []
    for k in LEADS:
        Eb, Ec, En = EV[f"lstm_{base}"][f"l{k}"], EV[f"lstm_{ceil}"][f"l{k}"], EV[f"lstm_{name}"][f"l{k}"]
        b, c, n = Eb["nse_median"], Ec["nse_median"], En["nse_median"]
        ob, oc, on = Eb["onset"]["pod"], Ec["onset"]["pod"], En["onset"]["pod"]
        rows.append(dict(gap=c - b, gain=n - b, share_of_gap_closed=(n - b) / (c - b) if c - b else np.nan, basins_better_than_base=int((En["nse"] > Eb["nse"]).sum()), median_basin_gain=float(np.median(En["nse"] - Eb["nse"])),
                         onset_gain_points=100 * (on - ob), onset_share_of_gap_closed=(on - ob) / (oc - ob) if oc - ob else np.nan, onset_warned_ceiling=oc, onset_warned_base=ob, onset_warned_with=on))
    return rows
for key, (name, base) in {"wqaQ": ("wqaQ", "wqa"), "wqaW": ("wqaW", "wqa"), "wqaW_older": ("wqaW_older", "wqa"), "wqaWF_sub": ("wqaWF_sub", "wqa"), "wqaWF_sub_from_the_forecast_model": ("wqaWF_sub", "wqaF_sub"), "wqaF_sub": ("wqaF_sub", "wqa")}.items():
    rows = closure(name, base, "wqaP")
    for j, k in enumerate(LEADS):
        fr = R["closure"]["sets"][key]["by_lead"][j]
        for c in ("gap", "gain", "share_of_gap_closed", "basins_better_than_base", "median_basin_gain", "onset_gain_points", "onset_share_of_gap_closed", "onset_warned_ceiling", "onset_warned_base", "onset_warned_with"):
            C.add(f"closure {key} l{k} {c}", fr[c], rows[j][c])
# 6. the validation years choose the triggers (replicated from code/warning_policy.py and the script); the test years are scored once at them
GRID = [round(float(x), 2) for x in np.arange(0.50, 2.0001, 0.05)]
LEVELS = [round(float(x), 2) for x in np.arange(0.30, 1.501, 0.05)]
POD_GO, FAR_GO, CSI_CAUTION = 0.70, 0.30, 0.40
verdict = lambda t: "NO EVENTS" if t["hits"] + t["misses"] == 0 else "GO" if (t["pod"] >= POD_GO and t["far"] <= FAR_GO) else "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"
_f = lambda x, bad=-1.0: bad if x is None or (isinstance(x, float) and np.isnan(x)) else x


def arrays(p, part, k):
    """Pooled over the basins on the target days of a part: obs_hi, ratio = forecast / Q95, was_low (as code/warning_policy.arrays)."""
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


def choose_csi(A, grid=GRID):
    Sg = {t: score_at(A, t) for t in grid}
    return max(grid, key=lambda t: (_f(Sg[t]["onset"]["csi"]), -abs(t - 1.0))), Sg


def choose_far_cap(A, cap=0.30, grid=GRID):
    Sg = {t: score_at(A, t) for t in grid}
    ok = [t for t in grid if _f(Sg[t]["onset"]["far"], 9) <= cap]
    if ok:
        return max(ok, key=lambda t: (_f(Sg[t]["onset"]["pod"]), _f(Sg[t]["onset"]["csi"]))), True, Sg
    return min(grid, key=lambda t: (_f(Sg[t]["onset"]["far"], 9), -t)), False, Sg


def choose_levels(A, kind):
    """The script's own chooser over its level grid 0.30..1.50 (best pooled CSI of `kind`, ties to the level nearest 1.0, the lower one on a tie)."""
    Sg = [score_at(A, m)[kind] for m in LEVELS]
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in Sg])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]


ops_mine = {}
for n in SHOWN:
    OP, VD = R["operating_points"]["sets"][n], R["verdicts"]["sets"][n]
    for k in LEADS:
        A_val, A_test = arrays(PRED[f"lstm_{n}"], "val", k), arrays(PRED[f"lstm_{n}"], "test", k)
        C.add(f"operating_points {n} l{k} n_test_onsets", OP[f"l{k}"]["n_test_onsets"], int(A_test[0][A_test[2]].sum()))
        t_csi, Sg = choose_csi(A_val)
        t_far, reached, _ = choose_far_cap(A_val)
        for rule, t, rch in (("csi", t_csi, True), ("far_cap", t_far, reached)):
            fr = OP[f"l{k}"]["rules"][rule]
            st, sv = score_at(A_test, t), Sg[t]
            C.add(f"operating_points {n} l{k} {rule} trigger", fr["trigger"], t); C.add(f"operating_points {n} l{k} {rule} reached", fr["reached"], rch)
            for kind in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    C.add(f"operating_points {n} l{k} {rule} validation {kind} {c}", fr["validation"][kind][c], sv[kind][c]); C.add(f"operating_points {n} l{k} {rule} test {kind} {c}", fr["test"][kind][c], st[kind][c])
                C.add(f"operating_points {n} l{k} {rule} test_verdict {kind}", fr["test_verdict"][kind], verdict(st[kind]))
            seeds = [score_at(arrays(PRED[f"lstm_{n}_s{s}"], "test", k), t)["onset"] for s in SEEDS[n]]
            C.add(f"operating_points {n} l{k} {rule} single_seeds pod lo", fr["single_seeds_at_this_trigger"]["onset_pod"][0], min(s["pod"] for s in seeds))
            C.add(f"operating_points {n} l{k} {rule} single_seeds pod hi", fr["single_seeds_at_this_trigger"]["onset_pod"][1], max(s["pod"] for s in seeds))
            C.add(f"operating_points {n} l{k} {rule} single_seeds far lo", fr["single_seeds_at_this_trigger"]["onset_far"][0], min(s["far"] for s in seeds))
            ops_mine[f"{n} l{k} {rule}"] = dict(trigger=t, reached=rch, test_onset=st["onset"], validation_onset=sv["onset"])
        # the verdict records: plain (1.0) and the two levels chosen on the validation years over the script's level grid
        vd = VD[f"l{k}"]
        for kind in ("onset", "all"):
            C.add(f"verdicts {n} l{k} plain {kind} verdict", vd["plain"][kind]["verdict"], verdict(score_at(A_test, 1.0)[kind]))
        for label, kind in (("chosen_on_validation_onsets", "onset"), ("chosen_on_validation_all_days", "all")):
            t = choose_levels(A_val, kind)
            st = score_at(A_test, t)
            C.add(f"verdicts {n} l{k} {label} trigger", vd[label]["trigger"], t)
            for kind2 in ("onset", "all"):
                C.add(f"verdicts {n} l{k} {label} {kind2} verdict", vd[label][kind2]["verdict"], verdict(st[kind2]))
                for c in ("hits", "false_alarms", "pod", "far", "csi"):
                    C.add(f"verdicts {n} l{k} {label} {kind2} {c}", vd[label][kind2][c], st[kind2][c])
        for s in SEEDS[n]:
            for kind in ("onset", "all"):
                C.add(f"verdicts {n} l{k} single_seeds_plain {kind} s{s}", vd["single_seeds_plain"][kind][str(s)], verdict(score_at(arrays(PRED[f"lstm_{n}_s{s}"], "test", k), 1.0)[kind]))
# 7. the little-rain slice (lead 1): onsets by the AORC rain of the target day (my column), the Daymet rain of the target day, the AORC rain of the issue day, the last 6 hours
import flood as F
D = F.load_data()
aorc, last6 = MY["aorc_to_midnight"], MY["aorc_last6h"]
rows = []
for i, g in enumerate(ids):
    o = OBS[i]
    issue = np.concatenate([[np.nan], o[:-1]])
    m = TE & np.isfinite(o) & (o > THR[i]) & (issue <= THR[i])
    for t in np.flatnonzero(m):
        rows.append(dict(gauge=g, t=t, aorc_target=aorc[i, t], daymet_target=float(D[g].prcp.values[t]), aorc_issue=aorc[i, t - 1], last6_issue=last6[i, t - 1],
                         **{n: bool(PRED[f"lstm_{n}"][i, t, 0] > THR[i]) for n in SHOWN}, **{f"{n}_s{s}": bool(PRED[f"lstm_{n}_s{s}"][i, t, 0] > THR[i]) for n in SHOWN for s in SEEDS[n]}))
E_ = pd.DataFrame(rows)
assert all(np.isfinite(PRED[f"lstm_{n}"][i, t, 0]) for n in SHOWN for i, t in zip([ids.index(g) for g in E_.gauge], E_.t))
C.add("little_rain n_onsets", R["little_rain_onsets"]["n_onsets"], len(E_))
SL = {"aorc_target_below_5mm": E_.aorc_target < 5, "aorc_target_at_least_5mm": E_.aorc_target >= 5, "daymet_target_below_5mm": E_.daymet_target < 5, "daymet_target_at_least_5mm": E_.daymet_target >= 5,
      "aorc_issue_day_below_1mm": E_.aorc_issue < 1, "aorc_issue_day_at_least_1mm": E_.aorc_issue >= 1, "aorc_last6h_of_issue_day_at_least_5mm": E_.last6_issue >= 5, "aorc_last6h_of_issue_day_below_5mm": E_.last6_issue < 5}
for key, mask in SL.items():
    fr = R["little_rain_onsets"]["slices"][key]
    Sx = E_[mask.values]
    C.add(f"little_rain {key} n", fr["n"], int(mask.sum()))
    for n in SHOWN:
        C.add(f"little_rain {key} warned {n}", fr["warned"][n], float(Sx[n].mean())); C.add(f"little_rain {key} hits {n}", fr["hits"][n], int(Sx[n].sum()))
        C.add(f"little_rain {key} single_seeds {n} lo", fr["single_seeds"][n][0], min(float(Sx[f"{n}_s{s}"].mean()) for s in SEEDS[n]))
        C.add(f"little_rain {key} single_seeds {n} hi", fr["single_seeds"][n][1], max(float(Sx[f"{n}_s{s}"].mean()) for s in SEEDS[n]))
    for T, Cc in (("wqaW", "wqa"), ("wqaW", "wqaW_older"), ("wqaQ", "wqa"), ("wqaWF_sub", "wqaF_sub")):
        pr = fr[f"paired_{T}_vs_{Cc}"]
        C.add(f"little_rain {key} paired {T} vs {Cc} only_first", pr["only_first"], int((Sx[T] & ~Sx[Cc]).sum())); C.add(f"little_rain {key} paired {T} vs {Cc} only_second", pr["only_second"], int((~Sx[T] & Sx[Cc]).sum()))
        C.add(f"little_rain {key} paired {T} vs {Cc} both", pr["both"], int((Sx[T] & Sx[Cc]).sum())); C.add(f"little_rain {key} paired {T} vs {Cc} gain_points", pr["gain_points"], 100 * (Sx[T].mean() - Sx[Cc].mean()))
sm, wet = E_[E_.aorc_target < 5], E_[E_.aorc_target >= 5]
W = R["little_rain_onsets"]["where_the_gain_sits"]
C.add("where_the_gain_sits hits_gained_in_the_slice", W["hits_gained_in_the_slice"], int(sm.wqaW.sum() - sm.wqa.sum())); C.add("where_the_gain_sits hits_gained_outside", W["hits_gained_outside"], int(wet.wqaW.sum() - wet.wqa.sum()))
C.add("where_the_gain_sits points_in_the_slice", W["points_in_the_slice"], 100 * (sm.wqaW.mean() - sm.wqa.mean())); C.add("where_the_gain_sits points_outside", W["points_outside"], 100 * (wet.wqaW.mean() - wet.wqa.mean()))
r18 = jload("results/forecast_input.json")["lead1_facts"]["little_rain_on_the_target_day"]
C.add("little_rain slice n equals script 18", r18["n"], int((E_.aorc_target < 5).sum())); C.add("little_rain slice wqa share equals script 18 no_slots", r18["no_slots"], float(sm.wqa.mean()))
# 8. the stored sets' own results files against my numbers
for n, f in (("wqa", "results/aorc_rain_to_midnight.json"), ("wqaF_sub", "results/forecast_input.json"), ("wqaP", "results/forecast_input.json")):
    s = jload(f)["summary_test"]["sets"][n]
    for k in LEADS:
        m = EV[f"lstm_{n}"][f"l{k}"]
        C.add(f"stored {n} own file l{k} nse_median", s[f"l{k}"]["nse_median"], m["nse_median"]); C.add(f"stored {n} own file l{k} onset pod", s[f"l{k}"]["onset"]["pod"], m["onset"]["pod"])
        C.add(f"stored {n} own file l{k} onset hits", s[f"l{k}"]["onset"]["hits"], m["onset"]["hits"]); C.add(f"stored {n} own file l{k} all csi", s[f"l{k}"]["all"]["csi"], m["all"]["csi"])
# 9. a cross-check of my scoring against the engine (flood.evaluate) on one set, as the project's reference
thr_d = {g: THR[i] for i, g in enumerate(ids)}
Pw = F.load_preds("lstm_wqaW")
e = F.evaluate(Pw, thr_d)["summary"]
engine = {f"l{k}": dict(nse_median=e[f"l{k}"]["nse_median"], onset_pod=e[f"l{k}"]["onset"]["pod"], onset_far=e[f"l{k}"]["onset"]["far"], all_csi=e[f"l{k}"]["all"]["csi"]) for k in LEADS}
engine_ok = all(abs(engine[f"l{k}"]["nse_median"] - EV["lstm_wqaW"][f"l{k}"]["nse_median"]) < 1e-12 and abs(engine[f"l{k}"]["onset_pod"] - EV["lstm_wqaW"][f"l{k}"]["onset"]["pod"]) < 1e-12 for k in LEADS)
summ = C.summary()
headline = {n: {f"l{k}": dict(nse_median=EV[f"lstm_{n}"][f"l{k}"]["nse_median"], onset=EV[f"lstm_{n}"][f"l{k}"]["onset"], all=EV[f"lstm_{n}"][f"l{k}"]["all"], seed_onset_pod=pods(n, k), seed_nse_median=nses(n, k)) for k in LEADS} for n in SHOWN}
jsave("r2_headline.json", dict(summary=summ, rows=C.rows, headline_mine=headline, seed_mean_check=seed_mean_check, success_mine=mine_success, success_with_three_seeds=three, rule_mine=rule_mine, operating_points_mine=ops_mine,
                               engine_cross_check=dict(set="lstm_wqaW", my_scoring_equals_flood_evaluate=engine_ok, engine=engine), thresholds=dict(zip(ids, THR.tolist())), seconds=round(time.time() - t0, 1)))
print(f"compared {summ['entries']} values: {summ['matching']} match, {len(summ['not_matching'])} differ" + (f": {summ['not_matching'][:20]}" if summ["not_matching"] else ""))
print("headline (test, trigger 1.0, lead 1): " + "; ".join(f"{n} {100 * headline[n]['l1']['onset']['pod']:.1f}% warned, {100 * headline[n]['l1']['onset']['far']:.1f}% false, NSE {headline[n]['l1']['nse_median']:.3f}" for n in SHOWN))
print("success rule (mine):", mine_success, "| with three seeds:", three)
print("my scoring equals flood.evaluate on lstm_wqaW:", engine_ok, "| seconds:", round(time.time() - t0, 1))
