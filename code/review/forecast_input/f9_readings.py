"""E2 review, check (h): the arithmetic of every derived block of results/forecast_input.json, recomputed with the reviewer's own code from results/preds (own NSE per basin, own onset counts on each set's own split,
own Wilcoxon test, own paired-onset counts and sign tests):
 1. the readings (value_of_the_forecast, forecast_vs_climatology_control, value_of_a_misleading_input, retrained_vs_substituted [4 comparisons], forecast_without_rain_to_midnight [3], how_far_from_perfect [2 + the shares of the gap],
    raw_vs_bias_corrected [2], substituted_network_given_perfect_rain [2], stopping_point_chosen_with_the_forecast): 3 leads each, field by field, with the overlaps of the warned onsets;
 2. gain_rule (12 comparisons x 3 leads) and e2_success; 3. closure (13 closures x 3 leads); 4. the verdict records of 23 sets (plain level and the two levels chosen on the validation years of the set's own split, the verdicts of the
    single seeds), with the warning level chosen by the reviewer's own sweep; 5. outcome, outcome_caveat (seven clauses) and outcome_full rebuilt from the reviewer's numbers, including the yardstick clause (the reading of
    the LSTM against ridge and forest at 1 day, by the NSE rule of report section 4.2 and by a paired sign test of the onsets) and the clock clause (the margin of the last needed file at the issue time, recomputed from the
    posting times of results/data_access.json). Numbers to 1e-9, flags and words exactly. About 1 minute."""
import sys; sys.dont_write_bytecode = True
import re
from scipy.stats import wilcoxon, binomtest
from f_common import *
import flood as F
from f_scoring import *

R = jload("results/forecast_input.json"); G = jload("results/gefs_basins.json")
D = F.load_data(); ids = list(D)
SH = SETS_E2 + ["wq", "wqa", "wqf", "wqaF_sub_rawpred", "wqaF_sub_perfectpred", "wqaP_forecastpred"]
SIMPLE = ["base_persistence", "base_linreg_wq", "base_rf_wq", "base_linreg_wqa", "base_rf_wqa", "base_linreg_wqaF", "base_rf_wqaF"]
arm = lambda n: ARM.get(n, "main")
POD, NSE_S, NSE_B, CNT, FLAGS = {}, {}, {}, {}, {}
for n in SH:
    for k in KS:
        POD[n, k] = [counts(f"lstm_{n}_s{s}", k)["pod"] for s in SEEDS]
        NSE_S[n, k] = [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", k).values()))) for s in SEEDS]
        nb = nse_per_basin(f"lstm_{n}", k); NSE_B[n, k] = np.array([nb[g] for g in ids]); CNT[n, k] = counts(f"lstm_{n}", k)
        t = onset_table(f"lstm_{n}", k)
        FLAGS[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
for n in SIMPLE:
    for k in KS:
        nb = nse_per_basin(n, k); NSE_B[n, k] = np.array([nb[g] for g in ids]); CNT[n, k] = counts(n, k) if n != "base_persistence" else None
        t = onset_table(n, k); FLAGS[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
above = lambda a, b: min(a) > max(b)
rng = lambda v: [min(v), max(v)]
n_b = len(ids)


def nse_reading(T, C, k):
    n_hi = int((NSE_B[T, k] > NSE_B[C, k]).sum()); n_lo = int((NSE_B[T, k] < NSE_B[C, k]).sum())
    try: p = float(wilcoxon(NSE_B[T, k], NSE_B[C, k]).pvalue)
    except ValueError: p = float("nan")
    up, down = above(NSE_S[T, k], NSE_S[C, k]), above(NSE_S[C, k], NSE_S[T, k]); small = p == p and p < 0.05
    gain, loss = up and n_hi > n_b / 2 and small, down and n_lo > n_b / 2 and small
    return n_hi, n_lo, p, up, down, small, gain, loss


def reading(T, C, k):
    assert arm(T) == arm(C)
    cT, cC = CNT[T, k], CNT[C, k]
    n_hi, n_lo, p, up, down, small, gain, loss = nse_reading(T, C, k)
    o_up, o_down = above(POD[T, k], POD[C, k]), above(POD[C, k], POD[T, k])
    medT, medC = float(np.median(NSE_B[T, k])), float(np.median(NSE_B[C, k]))
    fa, fc = FLAGS[T, k], FLAGS[C, k]
    both, oa, oc = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    return dict(with_set=f"lstm_{T}", without_set=f"lstm_{C}", arm=arm(T),
                onsets=dict(warned_with=cT["pod"], warned_without=cC["pod"], difference_points=100 * (cT["pod"] - cC["pod"]), hits_with=cT["hits"], hits_without=cC["hits"], onsets_in_the_test_years=cT["n_onsets"], hits_difference=cT["hits"] - cC["hits"],
                            seed_range_with=rng(POD[T, k]), seed_range_without=rng(POD[C, k]), every_seed_above_every_seed=o_up, every_seed_below_every_seed=o_down, difference_by_seed_points=[100 * (a - b) for a, b in zip(POD[T, k], POD[C, k])],
                            false_warnings_with=cT["far"], false_warnings_without=cC["far"], reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"),
                nse=dict(median_with=medT, median_without=medC, gain=medT - medC, seed_range_with=rng(NSE_S[T, k]), seed_range_without=rng(NSE_S[C, k]), every_seed_above_every_seed=up, difference_by_seed=[a - b for a, b in zip(NSE_S[T, k], NSE_S[C, k])],
                         n_basins_higher=n_hi, n_basins=n_b, median_basin_difference=float(np.median(NSE_B[T, k] - NSE_B[C, k])), wilcoxon_p=p, reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                overlap_of_warned_onsets=dict(first=T if False else None, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=oa, warned_only_by_second=oc, warned_by_neither=int((~fa & ~fc).sum()),
                                              warned_by_at_least_one=both + oa + oc, sign_test_p=float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan")),
                gain_rule=dict(nse=dict(single_seeds_all_above=up, single_seeds_all_below=down, n_basins_lower=n_lo, higher_in_most_basins=bool(n_hi > n_b / 2), p_below_cut=bool(small), clear_gain=bool(gain), clear_loss=bool(loss)),
                               onset=dict(gain=cT["pod"] - cC["pod"], false_with=cT["far"], false_without=cC["far"], single_seeds_all_above=o_up, single_seeds_all_below=o_down, clear_gain=o_up, clear_loss=o_down)))


def cmp(mine, theirs, path=""):
    worst, bad = 0.0, []
    if isinstance(mine, dict):
        for k, v in mine.items():
            if k not in theirs: bad.append((path + "/" + k, "missing in the JSON", None)); continue
            w, b = cmp(v, theirs[k], path + "/" + k); worst = max(worst, w); bad += b
    elif isinstance(mine, (list, tuple)):
        if len(mine) != len(theirs): return worst, [(path, f"length {len(mine)}", f"length {len(theirs)}")]
        for i, (a, b_) in enumerate(zip(mine, theirs)):
            w, b = cmp(a, b_, path + f"[{i}]"); worst = max(worst, w); bad += b
    elif mine is None: pass
    elif isinstance(mine, (bool, str)):
        if mine != theirs: bad.append((path, mine, theirs))
    else:
        if theirs is None and mine != mine: pass
        elif theirs is None or abs(mine - theirs) > 1e-9 * max(1.0, abs(theirs)): bad.append((path, mine, theirs)); worst = max(worst, abs(mine - (theirs if theirs is not None else 0)))
        else: worst = max(worst, abs(mine - theirs))
    return worst, bad


out = {}
# ---- 1. the readings ------------------------------------------------------------------------------------------------------------------------------------------------------
PATHS = {("value_of_the_forecast", "by_lead"): ("wqaF_sub", "wqa"), ("forecast_vs_climatology_control", "by_lead"): ("wqaF_sub", "wqaF_clim"), ("value_of_a_misleading_input", "by_lead"): ("wqaF_clim", "wqa"),
         ("retrained_vs_substituted", "retrained"): ("wqaF_s", "wqa_s"), ("retrained_vs_substituted", "substituted"): ("wqaF_sub", "wqa"), ("retrained_vs_substituted", "retrained_against_its_control"): ("wqaF_s", "wqaF_clim_s"),
         ("retrained_vs_substituted", "retrained_against_perfect_rain"): ("wqaP_s", "wqaF_s"), ("forecast_without_rain_to_midnight", "by_lead"): ("wqF_sub", "wq"),
         ("forecast_without_rain_to_midnight", "rain_to_midnight_given_the_forecast"): ("wqaF_sub", "wqF_sub"), ("forecast_without_rain_to_midnight", "forecast_alone_against_the_rain_to_midnight_alone"): ("wqF_sub", "wqa"),
         ("how_far_from_perfect", "perfect_rain_minus_substituted_forecast"): ("wqaP", "wqaF_sub"), ("how_far_from_perfect", "perfect_rain_minus_rain_to_midnight"): ("wqaP", "wqa"),
         ("raw_vs_bias_corrected", "at_prediction_time"): ("wqaF_sub", "wqaF_sub_rawpred"), ("substituted_network_given_perfect_rain", "perfect_rain_given_minus_forecast_given"): ("wqaF_sub_perfectpred", "wqaF_sub"),
         ("substituted_network_given_perfect_rain", "against_wqaP"): ("wqaP", "wqaF_sub_perfectpred"), ("stopping_point_chosen_with_the_forecast", "by_lead"): ("wqaF_sub", "wqaP_forecastpred")}
if "retrained_on_the_raw_forecast" in R["raw_vs_bias_corrected"]: PATHS[("raw_vs_bias_corrected", "retrained_on_the_raw_forecast")] = ("wqaF_sub", "wqaF_sub_raw")
bad_all, n_cmp, blocks = [], 0, {}
print("1. readings (3 leads each): own numbers against the JSON, field by field")
for (sec, sub), (T, C) in PATHS.items():
    node = R[sec][sub]
    if "by_lead" in node and "l1" not in node: node = node["by_lead"]
    worst = 0
    for k in KS:
        m = reading(T, C, k); j = node[f"l{k}"]
        mine = {key: m[key] for key in ("with_set", "without_set", "arm", "onsets", "nse")}; mine["overlap_of_warned_onsets"] = {kk: vv for kk, vv in m["overlap_of_warned_onsets"].items() if kk != "first"}
        w, b = cmp(mine, j); worst = max(worst, w); bad_all += [(sec, sub, k) + x for x in b]; n_cmp += sum(len(v) if isinstance(v, dict) else 1 for v in mine.values())
    l1 = reading(T, C, 1); blocks[f"{sec}/{sub}"] = dict(pair=f"lstm_{T} minus lstm_{C}", max_abs_deviation=worst)
    print(f"   {sec + '/' + sub:92s} lstm_{T} - lstm_{C}: lead 1 onsets {100 * l1['onsets']['warned_with']:.1f}% vs {100 * l1['onsets']['warned_without']:.1f}% ({l1['onsets']['difference_points']:+.1f}), NSE {l1['nse']['gain']:+.3f} ({l1['nse']['reading']}); max dev {worst:.1e}")
print(f"   {len(PATHS)} comparisons x 3 leads compared; mismatches: {len(bad_all)}", bad_all[:3])
out["readings"] = dict(blocks=blocks, mismatches=len(bad_all), n_comparisons=len(PATHS))

# ---- 2. gain_rule and e2_success ------------------------------------------------------------------------------------------------------------------------------------------
GR = {"wqaF_sub_against_wqa": ("wqaF_sub", "wqa"), "wqaF_sub_against_wqaF_clim": ("wqaF_sub", "wqaF_clim"), "wqaF_sub_against_wqaP": ("wqaF_sub", "wqaP"), "wqF_sub_against_wq": ("wqF_sub", "wq"), "wqaF_clim_against_wqa": ("wqaF_clim", "wqa"),
      "wqaP_against_wqa": ("wqaP", "wqa"), "wqaF_s_against_wqa_s": ("wqaF_s", "wqa_s"), "wqaF_s_against_wqaF_clim_s": ("wqaF_s", "wqaF_clim_s"), "wqaF_s_against_wqaP_s": ("wqaF_s", "wqaP_s"), "wqa_s_against_wq_s": ("wqa_s", "wq_s"),
      "wqaF_clim_s_against_wqa_s": ("wqaF_clim_s", "wqa_s"), "wqaF_sub_against_wqaF_sub_rawpred": ("wqaF_sub", "wqaF_sub_rawpred")}
gr_bad, gr_n = [], 0
for key, (T, C) in GR.items():
    for k in KS:
        m = reading(T, C, k); j = R["gain_rule"][key][f"l{k}"]
        mine = dict(against=f"lstm_{C}", lead=k, nse=dict(median_with=m["nse"]["median_with"], median_without=m["nse"]["median_without"], gain=m["nse"]["gain"], seed_range_with=m["nse"]["seed_range_with"], seed_range_without=m["nse"]["seed_range_without"], n_basins_higher=m["nse"]["n_basins_higher"], n_basins=n_b,
                                                         median_basin_difference=m["nse"]["median_basin_difference"], wilcoxon_p=m["nse"]["wilcoxon_p"], reading=m["nse"]["reading"], **m["gain_rule"]["nse"]),
                    onset=dict(warned_with=m["onsets"]["warned_with"], warned_without=m["onsets"]["warned_without"], seed_range_with=m["onsets"]["seed_range_with"], seed_range_without=m["onsets"]["seed_range_without"], reading=m["onsets"]["reading"], **m["gain_rule"]["onset"]))
        w, b = cmp(mine, j); gr_n += sum(len(v) for v in mine.values() if isinstance(v, dict)) + 2; gr_bad += [(key, k) + x for x in b]
print(f"2. gain_rule: {len(GR)} comparisons x 3 leads, {gr_n} values compared, {len(gr_bad)} differ", gr_bad[:3])
out["gain_rule"] = dict(n=gr_n, differ=len(gr_bad), details=gr_bad[:5])
e2 = {f"l{k}": dict(onset_warned=CNT["wqaF_sub", k]["pod"], onset_warned_wqa=CNT["wqa", k]["pod"], onset_warned_control=CNT["wqaF_clim", k]["pod"], seed_range=rng(POD["wqaF_sub", k]), every_seed_above_wqa=above(POD["wqaF_sub", k], POD["wqa", k]),
                    every_seed_above_the_control=above(POD["wqaF_sub", k], POD["wqaF_clim", k])) for k in KS}
for k in KS: e2[f"l{k}"]["met"] = bool(e2[f"l{k}"]["every_seed_above_wqa"] and e2[f"l{k}"]["every_seed_above_the_control"])
w, b = cmp(e2, R["e2_success"]["by_lead"]); met23 = bool(e2["l2"]["met"] and e2["l3"]["met"])
out["e2_success"] = dict(mine=e2, mismatches=b, met_at_2_and_3_days=met23, json_equal=(met23 == R["e2_success"]["met_at_2_and_3_days"]))
print(f"   e2_success: mismatches {b}; met at 2 and 3 days ahead (mine): {met23} (JSON {R['e2_success']['met_at_2_and_3_days']}); lead 1: every seed above wqa {e2['l1']['every_seed_above_wqa']}, above the control {e2['l1']['every_seed_above_the_control']}")

# ---- 3. closure --------------------------------------------------------------------------------------------------------------------------------------------------------------------
med = lambda n, k: float(np.median(NSE_B[n, k]))
cl_bad, cl_n, cl_share = [], 0, {}
for key, c in R["closure"].items():
    m_ = re.match(r"(.+)__base_(.+)__ceiling_(.+)", key); S_, b_, c_ = m_.groups()
    for k in KS:
        row = c["by_lead"][k - 1]
        bb, cc, nn = med(b_, k), med(c_, k), med(S_, k); ob, oc, on = CNT[b_, k]["pod"], CNT[c_, k]["pod"], CNT[S_, k]["pod"]
        mine = dict(lead=k, nse_base=bb, nse_with=nn, nse_ceiling=cc, gap=cc - bb, gain=nn - bb, share_of_gap_closed=(nn - bb) / (cc - bb) if cc != bb else float("nan"), basins_better_than_base=int((NSE_B[S_, k] > NSE_B[b_, k]).sum()), median_basin_gain=float(np.median(NSE_B[S_, k] - NSE_B[b_, k])),
                    onset_warned_base=ob, onset_warned_with=on, onset_warned_ceiling=oc, onset_false_base=CNT[b_, k]["far"], onset_false_with=CNT[S_, k]["far"], onset_false_ceiling=CNT[c_, k]["far"], onset_share_of_gap_closed=(on - ob) / (oc - ob) if oc != ob else float("nan"),
                    seed_range_nse_with=rng(NSE_S[S_, k]), seed_range_nse_base=rng(NSE_S[b_, k]), onset_seed_range_with=rng(POD[S_, k]), onset_seed_range_base=rng(POD[b_, k]))
        w, b = cmp(mine, row); cl_n += len(mine); cl_bad += [(key, k) + x for x in b]
        cl_share[(S_, b_, c_, k)] = (mine["onset_share_of_gap_closed"], mine["share_of_gap_closed"])
print(f"3. closure: {len(R['closure'])} closures x 3 leads, {cl_n} values compared, {len(cl_bad)} differ", cl_bad[:3])
out["closure"] = dict(n=cl_n, differ=len(cl_bad), details=cl_bad[:5])

# ---- 4. verdicts -----------------------------------------------------------------------------------------------------------------------------------------------------------------
crit = R["verdicts"]["criteria"]
pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
_cache = {}
def tbl(name, lead, which, level, kind):
    if name not in _cache: _cache[name] = load_npz(name)
    dates, basins, obs, pred = _cache[name]; sp = split_of(name); thr = thresholds(basins, dates, sp); a, b = sp[which]; sel = part(dates, a, b)
    H = M = Fa = 0
    for i, g in enumerate(basins):
        o, f = obs[i], pred[i, :, lead - 1]; issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = sel & np.isfinite(o) & np.isfinite(f)
        if kind == "onset": ok = ok & (issue <= thr[g])
        high, warn = (o > thr[g]), (f > level * thr[g])
        H += int((high & warn & ok).sum()); M += int((high & ~warn & ok).sum()); Fa += int((~high & warn & ok).sum())
    return dict(hits=H, misses=M, false_alarms=Fa, pod=H / (H + M) if H + M else float("nan"), far=Fa / (H + Fa) if H + Fa else float("nan"), csi=H / (H + M + Fa) if H + M + Fa else float("nan"))
def verdict(t):
    if t["hits"] + t["misses"] == 0: return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go: return "GO"
    return "CAUTION" if t["csi"] >= csi_c else "NOT YET"
vb, vn, vrows = [], 0, {}
for name in list(R["verdicts"]["sets"]):
    jv = R["verdicts"]["sets"][name]; pn = f"lstm_{name}" if name in SH else name
    for k in KS:
        for label in ("plain", "chosen_on_validation_all_days", "chosen_on_validation_onsets"):
            if label == "plain": lvl = 1.0
            else:
                kind0 = "all" if label.endswith("all_days") else "onset"
                csi = np.array([c if c == c else -1.0 for c in (tbl(pn, k, "val", m, kind0)["csi"] for m in LEVELS)]); best = np.flatnonzero(csi >= csi.max() - 1e-12)
                lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
            j = jv[f"l{k}"][label]; vn += 1; vb += [(name, k, label, "trigger", lvl, j["trigger"])] if abs(lvl - j["trigger"]) > 1e-9 else []
            for kk in ("all", "onset"):
                t = tbl(pn, k, "test", lvl, kk); jj = j[kk]
                for fld in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    a_, b_ = t[fld], jj[fld]; vn += 1
                    if not ((b_ is None and a_ != a_) or (b_ is not None and abs(a_ - b_) < 1e-9)): vb.append((name, k, label, kk, fld, a_, b_))
                vn += 1; vb += [(name, k, label, kk, "verdict", verdict(t), jj["verdict"])] if verdict(t) != jj["verdict"] else []
            vrows[f"{name}:l{k}:{label}"] = dict(trigger=lvl, onset_pod=tbl(pn, k, "test", lvl, "onset")["pod"], onset_verdict=verdict(tbl(pn, k, "test", lvl, "onset")), all_verdict=verdict(tbl(pn, k, "test", lvl, "all")))
    # single seeds, plain level
    if "single_seeds_plain" in jv["l1"]:
        for k in KS:
            for kind in ("all", "onset"):
                mine_s = {str(s): verdict(counts_ := (lambda t: dict(t, hits=t["hits"], misses=t["misses"]))(tbl(f"lstm_{name}_s{s}", k, "test", 1.0, kind))) for s in SEEDS}
                vn += 3; vb += [(name, k, "single_seeds_plain", kind, s_, mine_s[s_], jv[f"l{k}"]["single_seeds_plain"][kind][s_]) for s_ in mine_s if mine_s[s_] != jv[f"l{k}"]["single_seeds_plain"][kind][s_]]
print(f"4. verdict records ({len(R['verdicts']['sets'])} sets x 3 leads x 3 ways of choosing the warning level, plus the single seeds): {vn} values recomputed with my own validation choice (each set's own validation years) and test counting, {len(vb)} differ", vb[:3])
print("   onset verdicts at the plain level, leads 1/2/3 (mine):", {n: [vrows[f'{n}:l{k}:plain']['onset_verdict'] for k in KS] for n in ("wqa", "wqaF_sub", "wqaF_clim", "wqaP", "wqF_sub", "wqaF_s", "wqa_s", "wqaP_s", "wqf", "base_linreg_wqaF", "base_rf_wqaF")})
print("   all-high-flow-days verdicts at the plain level, lead 1 (mine):", {n: vrows[f'{n}:l1:plain']['all_verdict'] for n in ("wqa", "wqaF_sub", "wqaP", "wqaF_s", "base_linreg_wqaF", "base_rf_wqaF")}, "| chosen on the validation onsets, 'all days' verdict, lead 1, wqaF_sub:", vrows["wqaF_sub:l1:chosen_on_validation_onsets"]["all_verdict"], "at level", vrows["wqaF_sub:l1:chosen_on_validation_onsets"]["trigger"])
out["verdicts"] = dict(n=vn, differ=len(vb), details=vb[:5], onset_verdicts_plain={n: [vrows[f"{n}:l{k}:plain"]["onset_verdict"] for k in KS] for n in R["verdicts"]["sets"]}, all_verdicts_plain_lead1={n: vrows[f"{n}:l1:plain"]["all_verdict"] for n in R["verdicts"]["sets"]},
                        any_go_for_onsets=bool(any(v["onset_verdict"] == "GO" for v in vrows.values())), onset_verdict_set=sorted({v["onset_verdict"] for v in vrows.values() if True}))
out["total_mismatches"] = len(bad_all) + len(gr_bad) + len(cl_bad) + len(vb) + len(out["e2_success"]["mismatches"])
print(f"TOTAL mismatches in readings / gain_rule / e2_success / closure / verdicts: {out['total_mismatches']}")
jsave("f9_readings.json", out)
