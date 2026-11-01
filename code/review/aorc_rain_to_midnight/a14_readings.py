"""AORC review, check (a)/(g) extension: the arithmetic of every derived block of results/aorc_rain_to_midnight.json, recomputed with the reviewer's own code from results/preds (own NSE per basin, own onset counts,
own Wilcoxon test, own overlap counts): (1) the nine reading blocks (value_of_hours_09_to_24 ... rain_table_with_the_right_clock), 3 leads each, field by field; (2) the overlaps of warned onsets and their sign tests;
(3) gain_rule (12 comparisons x 3 leads) and e1b_success; (4) closure (4 sets x 3 leads, with its references); (5) the verdict records (6 sets x 3 leads x 3 ways of choosing the warning level), the warning level
chosen on the VALIDATION years with the reviewer's own choice. Numbers are compared to 1e-9, flags and words exactly. About a minute."""
import sys; sys.dont_write_bytecode = True
import re
from scipy.stats import wilcoxon, binomtest
from a_common import *
import flood as F
from a_scoring import *

R = jload("results/aorc_rain_to_midnight.json")
ids = list(F.load_data())
SEEDS = (0, 1, 2)
NAMES = ["wq", "wqn", "wqh", "wqnh", "wqf", "wqa", "wqa_to09", "wqa_utc", "wqa_rep"]
POD, NSE_S, NSE_B, CNT, FLAGS = {}, {}, {}, {}, {}
for n in NAMES:
    for k in (1, 2, 3):
        POD[n, k] = [counts(f"lstm_{n}_s{s}", k)["pod"] for s in SEEDS]
        NSE_S[n, k] = [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", k).values()))) for s in SEEDS]
        nb = nse_per_basin(f"lstm_{n}", k); NSE_B[n, k] = np.array([nb[g] for g in ids]); CNT[n, k] = counts(f"lstm_{n}", k)
        t = onset_table(f"lstm_{n}", k)
        FLAGS[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
above = lambda a, b: min(a) > max(b)
rng = lambda v: [min(v), max(v)]


def reading(T, C, k):
    cT, cC = CNT[T, k], CNT[C, k]
    n_hi = int((NSE_B[T, k] > NSE_B[C, k]).sum()); n_lo = int((NSE_B[T, k] < NSE_B[C, k]).sum()); n_b = len(ids)
    try: p = float(wilcoxon(NSE_B[T, k], NSE_B[C, k]).pvalue)
    except ValueError: p = float("nan")
    up, down = above(NSE_S[T, k], NSE_S[C, k]), above(NSE_S[C, k], NSE_S[T, k])
    small = p == p and p < 0.05
    gain, loss = up and n_hi > n_b / 2 and small, down and n_lo > n_b / 2 and small
    o_up, o_down = above(POD[T, k], POD[C, k]), above(POD[C, k], POD[T, k])
    medT, medC = float(np.median(NSE_B[T, k])), float(np.median(NSE_B[C, k]))
    return dict(onsets=dict(warned_with=cT["pod"], warned_without=cC["pod"], difference_points=100 * (cT["pod"] - cC["pod"]), hits_with=cT["hits"], hits_without=cC["hits"], onsets_in_the_test_years=cT["n_onsets"], hits_difference=cT["hits"] - cC["hits"],
                     seed_range_with=rng(POD[T, k]), seed_range_without=rng(POD[C, k]), every_seed_above_every_seed=o_up, every_seed_below_every_seed=o_down, difference_by_seed_points=[100 * (a - b) for a, b in zip(POD[T, k], POD[C, k])],
                     false_warnings_with=cT["far"], false_warnings_without=cC["far"], reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"),
                nse=dict(median_with=medT, median_without=medC, gain=medT - medC, seed_range_with=rng(NSE_S[T, k]), seed_range_without=rng(NSE_S[C, k]), every_seed_above_every_seed=up, difference_by_seed=[a - b for a, b in zip(NSE_S[T, k], NSE_S[C, k])],
                         n_basins_higher=n_hi, n_basins=n_b, median_basin_difference=float(np.median(NSE_B[T, k] - NSE_B[C, k])), wilcoxon_p=p, reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                gain_rule=dict(nse=dict(single_seeds_all_above=up, single_seeds_all_below=down, n_basins_lower=n_lo, higher_in_most_basins=bool(n_hi > n_b / 2), p_below_cut=bool(small), clear_gain=bool(gain), clear_loss=bool(loss)),
                               onset=dict(gain=cT["pod"] - cC["pod"], false_with=cT["far"], false_without=cC["far"], single_seeds_all_above=o_up, single_seeds_all_below=o_down, clear_gain=o_up, clear_loss=o_down)))


def overlap(A, C, k=1):
    fa, fc = FLAGS[A, k], FLAGS[C, k]
    both, oa, oc = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    return dict(n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=oa, warned_only_by_second=oc, warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + oa + oc,
                sign_test_p=float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))


def cmp(mine, theirs, path=""):
    """largest absolute deviation of the numbers and every mismatch of a flag or a text, recursively (only the keys that are in mine)"""
    worst, bad = 0.0, []
    if isinstance(mine, dict):
        for k, v in mine.items():
            if k not in theirs: bad.append((path + "/" + k, "missing in the JSON", None)); continue
            w, b = cmp(v, theirs[k], path + "/" + k); worst = max(worst, w); bad += b
    elif isinstance(mine, (list, tuple)):
        if len(mine) != len(theirs): return worst, [(path, f"length {len(mine)}", f"length {len(theirs)}")]
        for i, (a, b_) in enumerate(zip(mine, theirs)):
            w, b = cmp(a, b_, path + f"[{i}]"); worst = max(worst, w); bad += b
    elif isinstance(mine, (bool, str)):
        if mine != theirs: bad.append((path, mine, theirs))
    else:
        if theirs is None and mine != mine: pass
        elif theirs is None or abs(mine - theirs) > 1e-9: bad.append((path, mine, theirs)); worst = max(worst, abs(mine - (theirs if theirs is not None else 0)))
        else: worst = max(worst, abs(mine - theirs))
    return worst, bad


out = {}
tot_values = 0
# ---- 1. the nine reading blocks ----------------------------------------------------------------------------------------------------------------------------------------------------
BLOCKS = {"value_of_hours_09_to_24": ("wqa", "wqa_to09"), "value_of_evening_hours_beyond_utc_day": ("wqa", "wqa_utc"), "value_of_hours_09_to_utc_day_end": ("wqa_utc", "wqa_to09"), "second_rain_product_alone": ("wqa_to09", "wq"),
          "aorc_against_era5": ("wqa", "wqh"), "aorc_against_nldas": ("wqa", "wqn"), "nldas_rain_against_aorc_over_the_utc_day": ("wqn", "wqa_utc"), "aorc_against_nldas_and_era5_together": ("wqa", "wqnh")}
bad_all = []
print("1. reading blocks (3 leads each): own numbers against the JSON, field by field")
for key, (T, C) in BLOCKS.items():
    worst = 0
    for k in (1, 2, 3):
        m = reading(T, C, k); j = R[key]["by_lead"][f"l{k}"]
        assert (j["with_set"], j["without_set"]) == (f"lstm_{T}", f"lstm_{C}"), (key, j["with_set"], j["without_set"])
        w, b = cmp({"onsets": m["onsets"], "nse": m["nse"]}, {"onsets": j["onsets"], "nse": j["nse"]}); worst = max(worst, w); bad_all += [(key, k) + x for x in b]
    for sub, mm in (("against_rain_table_alone", ("wqa_rep", "wq")), ("against_aorc_as_an_extra_column", ("wqa_rep", "wqa"))):
        pass
    l1 = reading(T, C, 1); out.setdefault("blocks", {})[key] = dict(pair=f"lstm_{T} minus lstm_{C}", max_abs_deviation=worst)
    print(f"  {key:44s} lstm_{T} - lstm_{C}: lead 1 onsets {100 * l1['onsets']['warned_with']:.1f}% vs {100 * l1['onsets']['warned_without']:.1f}% ({l1['onsets']['difference_points']:+.1f}), NSE {l1['nse']['gain']:+.3f}; max deviation from the JSON {worst:.1e}")
for sub, (T, C) in (("against_rain_table_alone", ("wqa_rep", "wq")), ("against_aorc_as_an_extra_column", ("wqa_rep", "wqa"))):
    worst = 0
    for k in (1, 2, 3):
        m = reading(T, C, k); j = R["rain_table_with_the_right_clock"][sub][f"l{k}"]
        w, b = cmp({"onsets": m["onsets"], "nse": m["nse"]}, {"onsets": j["onsets"], "nse": j["nse"]}); worst = max(worst, w); bad_all += [("rain_table_with_the_right_clock/" + sub, k) + x for x in b]
    l1 = reading(T, C, 1); out["blocks"]["rain_table_with_the_right_clock/" + sub] = dict(pair=f"lstm_{T} minus lstm_{C}", max_abs_deviation=worst)
    print(f"  rain_table_with_the_right_clock/{sub:36s} lstm_{T} - lstm_{C}: lead 1 onsets {100 * l1['onsets']['warned_with']:.1f}% vs {100 * l1['onsets']['warned_without']:.1f}% ({l1['onsets']['difference_points']:+.1f}), NSE {l1['nse']['gain']:+.3f}; max deviation {worst:.1e}")
print("   mismatches:", len(bad_all), bad_all[:3])
out["blocks_mismatches"] = len(bad_all)

# ---- 2. overlaps -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
OV = {("value_of_hours_09_to_24", "overlap_of_warned_onsets"): ("wqa", "wqa_to09"), ("value_of_evening_hours_beyond_utc_day", "overlap_of_warned_onsets"): ("wqa", "wqa_utc"), ("aorc_against_era5", "overlap_of_warned_onsets"): ("wqa", "wqh"),
      ("aorc_against_nldas", "overlap_of_warned_onsets"): ("wqa", "wqn"), ("nldas_rain_against_aorc_over_the_utc_day", "overlap_of_warned_onsets"): ("wqn", "wqa_utc"), ("aorc_against_nldas_and_era5_together", "overlap_of_warned_onsets"): ("wqa", "wqnh")}
ov_bad, out["overlaps"] = [], {}
for (key, sub), (A, C) in OV.items():
    m = overlap(A, C); j = R[key][sub]
    d = [(k, m[k], j[k]) for k in m if abs(m[k] - j[k]) > 1e-9 * max(1, abs(j[k])) and not (m[k] != m[k])]
    # sign test p: exact comparison to a relative 1e-9 (tiny p-values)
    ov_bad += [(key,) + x for x in d]; out["overlaps"][key] = dict(mine=m, differ=len(d))
for sub, (A, C) in (("replaced_against_rain_table_alone", ("wqa_rep", "wq")), ("replaced_against_aorc_as_an_extra_column", ("wqa_rep", "wqa"))):
    if sub in R["rain_table_with_the_right_clock"]["overlap_of_warned_onsets"]:
        m = overlap(A, C); j = R["rain_table_with_the_right_clock"]["overlap_of_warned_onsets"][sub]
        d = [(k, m[k], j[k]) for k in m if abs(m[k] - j[k]) > 1e-9 * max(1, abs(j[k]))]; ov_bad += [(sub,) + x for x in d]; out["overlaps"]["rain_table_with_the_right_clock/" + sub] = dict(mine=m, differ=len(d))
print(f"\n2. overlaps of warned onsets ({len(out['overlaps'])} pairs; 9 numbers each, the sign-test p to 1e-9 relative): {len(ov_bad)} differ", ov_bad[:3])
for k, v in out["overlaps"].items():
    m = v["mine"]; print(f"   {k:62s} both {m['warned_by_both']}, only first {m['warned_only_by_first']}, only second {m['warned_only_by_second']}, neither {m['warned_by_neither']}, sign test p {m['sign_test_p']:.2g}")
out["overlaps_mismatches"] = len(ov_bad)

# ---- 3. gain_rule and e1b_success -------------------------------------------------------------------------------------------------------------------------------------------------------
GR = {("wqa", "against_control"): ("wqa", "wqa_to09"), ("wqa", "against_utc_day"): ("wqa", "wqa_utc"), ("wqa", "against_rain_table_alone"): ("wqa", "wq"), ("wqa", "against_nldas_rain"): ("wqa", "wqn"), ("wqa", "against_era5_rain_to_midnight"): ("wqa", "wqh"),
      ("wqa", "control_against_rain_table_alone"): ("wqa_to09", "wq"), ("wqa", "utc_day_against_rain_table_alone"): ("wqa_utc", "wq"), ("wqa", "utc_day_against_matched_window_to_09"): ("wqa_utc", "wqa_to09"),
      ("wqa_rep", "against_rain_table_alone"): ("wqa_rep", "wq"), ("wqa_rep", "against_aorc_as_an_extra_column"): ("wqa_rep", "wqa"), ("wqa_rep", "against_nldas_rain"): ("wqa_rep", "wqn"), ("wqa_rep", "against_era5_rain_to_midnight"): ("wqa_rep", "wqh")}
gr_bad, gr_n = [], 0
for (fam, key), (T, C) in GR.items():
    for k in (1, 2, 3):
        m = reading(T, C, k); j = R["gain_rule"][fam][key][f"l{k}"]
        mine = dict(nse=dict(median_with=m["nse"]["median_with"], median_without=m["nse"]["median_without"], gain=m["nse"]["gain"], seed_range_with=m["nse"]["seed_range_with"], seed_range_without=m["nse"]["seed_range_without"], n_basins_higher=m["nse"]["n_basins_higher"], n_basins=m["nse"]["n_basins"],
                                median_basin_difference=m["nse"]["median_basin_difference"], wilcoxon_p=m["nse"]["wilcoxon_p"], reading=m["nse"]["reading"], **m["gain_rule"]["nse"]),
                    onset=dict(warned_with=m["onsets"]["warned_with"], warned_without=m["onsets"]["warned_without"], seed_range_with=m["onsets"]["seed_range_with"], seed_range_without=m["onsets"]["seed_range_without"], reading=m["onsets"]["reading"], **m["gain_rule"]["onset"]))
        w, b = cmp(mine, j); gr_n += sum(len(v) for v in mine.values()); gr_bad += [(fam, key, k) + x for x in b]
print(f"\n3. gain_rule: {len(GR)} comparisons x 3 leads, {gr_n} values compared, {len(gr_bad)} differ", gr_bad[:3])
out["gain_rule"] = dict(n=gr_n, differ=len(gr_bad), details=gr_bad[:5])
E = R["e1b_success"]["wqa"]
mine_e = dict(every_seed_above_every_seed_with_nldas_rain=above(POD["wqa", 1], POD["wqn", 1]), every_seed_above_every_seed_of_the_control=above(POD["wqa", 1], POD["wqa_to09", 1]), every_seed_above_every_seed_with_the_rain_table_alone=above(POD["wqa", 1], POD["wq", 1]),
              control_flat=not above(POD["wqa_to09", 1], POD["wq", 1]), every_seed_above_every_seed_with_era5_rain_to_midnight=above(POD["wqa", 1], POD["wqh", 1]), every_seed_of_era5_rain_to_midnight_above_every_seed_with_aorc=above(POD["wqh", 1], POD["wqa", 1]),
              control_utc_onset_warned=CNT["wqa_utc", 1]["pod"], control_utc_seed_range=rng(POD["wqa_utc", 1]), control_utc_flat=not above(POD["wqa_utc", 1], POD["wq", 1]), every_seed_above_every_seed_of_the_control_utc=above(POD["wqa", 1], POD["wqa_utc", 1]))
mine_e["met"] = all([mine_e["every_seed_above_every_seed_with_nldas_rain"], mine_e["every_seed_above_every_seed_of_the_control"], mine_e["every_seed_above_every_seed_with_the_rain_table_alone"], mine_e["control_flat"]])
w, b = cmp(mine_e, E); out["e1b_success"] = dict(mine=mine_e, max_dev=w, mismatches=b)
S1b = E["roadmap_step_1b"]
holds = mine_e["every_seed_above_every_seed_of_the_control"] and mine_e["every_seed_above_every_seed_with_the_rain_table_alone"] and not mine_e["every_seed_of_era5_rain_to_midnight_above_every_seed_with_aorc"]
grows = holds and mine_e["every_seed_above_every_seed_with_era5_rain_to_midnight"]
w2, b2 = cmp(dict(gain_holds=bool(holds), gain_grows=bool(grows), not_clearly_below_era5=not mine_e["every_seed_of_era5_rain_to_midnight_above_every_seed_with_aorc"]), S1b); out["roadmap_step_1b"] = dict(max_dev=w2, mismatches=b2)
print(f"   e1b_success (10 fields + met): max deviation {w:.1e}, mismatches {b}; roadmap_step_1b flags (holds {holds}, grows {grows}): mismatches {b2}")

# ---- 4. closure -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cl_bad, cl_n = [], 0
REFS = {"control_window_09_to_09": "wqa_to09", "utc_day": "wqa_utc", "era5_rain_to_midnight": "wqh", "nldas_rain": "wqn", "nldas_and_era5_together": "wqnh", "aorc_replacing_the_rain_table": "wqa_rep", "aorc_to_midnight": "wqa"}
med = lambda n, k: float(np.median(NSE_B[n, k]))
for name in ("wqa", "wqa_to09", "wqa_utc", "wqa_rep"):
    for k in (1, 2, 3):
        row = R["closure"][name]["by_lead"][k - 1]
        b0, c0, n_ = med("wq", k), med("wqf", k), med(name, k); ob, oc, on = CNT["wq", k]["pod"], CNT["wqf", k]["pod"], CNT[name, k]["pod"]
        mine = dict(lead=k, nse_base=b0, nse_with=n_, nse_perfect_forecast=c0, gap=c0 - b0, gain=n_ - b0, share_of_gap_closed=(n_ - b0) / (c0 - b0), basins_better=int((NSE_B[name, k] > NSE_B["wq", k]).sum()), median_basin_gain=float(np.median(NSE_B[name, k] - NSE_B["wq", k])),
                    onset_warned_base=ob, onset_warned_with=on, onset_warned_perfect_forecast=oc, onset_false_base=CNT["wq", k]["far"], onset_false_with=CNT[name, k]["far"], onset_false_perfect_forecast=CNT["wqf", k]["far"], onset_share_of_gap_closed=(on - ob) / (oc - ob),
                    seed_range_nse_with=rng(NSE_S[name, k]), seed_range_nse_base=rng(NSE_S["wq", k]), onset_seed_range_with=rng(POD[name, k]), onset_seed_range_base=rng(POD["wq", k]), references={})
        for rk, rj in row["references"].items():
            Cc = REFS[rk]
            mine["references"][rk] = dict(set=f"lstm_{Cc}", nse_median=med(Cc, k), nse_seed_range=rng(NSE_S[Cc, k]), nse_gain_over_it=n_ - med(Cc, k), basins_better_than_it=int((NSE_B[name, k] > NSE_B[Cc, k]).sum()), median_basin_gain_over_it=float(np.median(NSE_B[name, k] - NSE_B[Cc, k])),
                                          share_of_nse_gap_closed_beyond_it=(n_ - med(Cc, k)) / (c0 - b0), onset_warned=CNT[Cc, k]["pod"], onset_seed_range=rng(POD[Cc, k]), onset_false=CNT[Cc, k]["far"], onset_gain_over_it=on - CNT[Cc, k]["pod"],
                                          onset_share_of_gap_closed_beyond_it=(on - CNT[Cc, k]["pod"]) / (oc - ob), own_nse_gain_over_base=med(Cc, k) - b0, own_share_of_nse_gap_closed=(med(Cc, k) - b0) / (c0 - b0), own_onset_gain_over_base=CNT[Cc, k]["pod"] - ob,
                                          own_onset_share_of_gap_closed=(CNT[Cc, k]["pod"] - ob) / (oc - ob))
        w, b = cmp(mine, row); cl_n += 20 + sum(len(v) for v in mine["references"].values()); cl_bad += [(name, k) + x for x in b]
        if k == 1: print(f"   closure {name:9s} lead 1: onset gap closed {100 * mine['onset_share_of_gap_closed']:.1f}% (JSON {100 * row['onset_share_of_gap_closed']:.1f}%), NSE gap closed {100 * mine['share_of_gap_closed']:.1f}% (JSON {100 * row['share_of_gap_closed']:.1f}%)")
print(f"\n4. closure: 4 sets x 3 leads with their references, {cl_n} values compared, {len(cl_bad)} differ", cl_bad[:3])
out["closure"] = dict(n=cl_n, differ=len(cl_bad), details=cl_bad[:5])
l2 = {k: dict(nse=(med("wqa", k), med("wq", k)), onsets=(CNT["wqa", k]["pod"], CNT["wq", k]["pod"])) for k in (1, 2, 3)}
print("   leads 2 and 3 (the claim 'nearly unchanged'): " + "; ".join(f"lead {k}: NSE {v['nse'][1]:.3f} -> {v['nse'][0]:.3f} ({v['nse'][0] - v['nse'][1]:+.3f}), onsets {100 * v['onsets'][1]:.1f}% -> {100 * v['onsets'][0]:.1f}% ({100 * (v['onsets'][0] - v['onsets'][1]):+.1f} points, x{v['onsets'][0] / v['onsets'][1]:.2f})" for k, v in l2.items()))
out["leads_wqa_vs_wq"] = {f"l{k}": dict(nse_with=v["nse"][0], nse_without=v["nse"][1], onsets_with=v["onsets"][0], onsets_without=v["onsets"][1], hits_with=CNT["wqa", k]["hits"], hits_without=CNT["wq", k]["hits"]) for k, v in l2.items()}

# ---- 5. verdict records ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
crit = R["verdicts"]["criteria"]
pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
VAL = tuple(R["split"]["val"])
_cache = {}
def tbl(name, lead, a, b, level, kind):
    if name not in _cache: _cache[name] = load_npz(name)
    dates, basins, obs, pred = _cache[name]; thr = thresholds(basins, dates); sel = part(dates, a, b)
    H = M = Fa = Cc = 0
    for i, g in enumerate(basins):
        o, f = obs[i], pred[i, :, lead - 1]
        issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = sel & np.isfinite(o) & np.isfinite(f)
        if kind == "onset": ok = ok & (issue <= thr[g])
        high, warn = (o > thr[g]), (f > level * thr[g])
        H += int((high & warn & ok).sum()); M += int((high & ~warn & ok).sum()); Fa += int((~high & warn & ok).sum()); Cc += int((~high & ~warn & ok).sum())
    return dict(hits=H, misses=M, false_alarms=Fa, pod=H / (H + M) if H + M else float("nan"), far=Fa / (H + Fa) if H + Fa else float("nan"), csi=H / (H + M + Fa) if H + M + Fa else float("nan"))
def verdict(t):
    if t["hits"] + t["misses"] == 0: return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go: return "GO"
    return "CAUTION" if t["csi"] >= csi_c else "NOT YET"
vb, vn, rows = [], 0, {}
for name in ("lstm_wqa", "lstm_wqa_to09", "lstm_wqa_utc", "lstm_wqa_rep", "base_linreg_wqa", "base_rf_wqa"):
    jv = R["verdicts"]["sets"][name[5:] if name.startswith("lstm_") else name]
    for k in (1, 2, 3):
        for label in ("plain", "chosen_on_validation_all_days", "chosen_on_validation_onsets"):
            if label == "plain": lvl = 1.0
            else:
                kind0 = "all" if label.endswith("all_days") else "onset"
                csi = np.array([c if c == c else -1.0 for c in (tbl(name, k, *VAL, m, kind0)["csi"] for m in LEVELS)]); best = np.flatnonzero(csi >= csi.max() - 1e-12)
                lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
            j = jv[f"l{k}"][label]
            vn += 1; vb += [(name, k, label, "trigger", lvl, j["trigger"])] if abs(lvl - j["trigger"]) > 1e-9 else []
            for kk in ("all", "onset"):
                t = tbl(name, k, *TEST, lvl, kk); jj = j[kk]
                for fld in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    a, b = t[fld], jj[fld]; vn += 1
                    if not ((b is None and a != a) or abs(a - b) < 1e-9): vb.append((name, k, label, kk, fld, a, b))
                vn += 1; vb += [(name, k, label, kk, "verdict", verdict(t), jj["verdict"])] if verdict(t) != jj["verdict"] else []
            rows[f"{name}:l{k}:{label}"] = dict(trigger=lvl, onset_pod=tbl(name, k, *TEST, lvl, "onset")["pod"], onset_verdict=verdict(tbl(name, k, *TEST, lvl, "onset")), all_verdict=verdict(tbl(name, k, *TEST, lvl, "all")))
print(f"\n5. verdict records (6 sets x 3 leads x 3 choices of the warning level: trigger, 2 tables x 6 numbers and a verdict): {vn} values recomputed with my own validation choice and own test counting, {len(vb)} differ", vb[:3])
for key in ("lstm_wqa:l1:chosen_on_validation_onsets", "lstm_wqa_to09:l1:chosen_on_validation_onsets", "lstm_wqa_utc:l1:chosen_on_validation_onsets", "lstm_wqa_rep:l1:chosen_on_validation_onsets", "base_linreg_wqa:l1:chosen_on_validation_onsets", "base_rf_wqa:l1:chosen_on_validation_onsets"):
    r = rows[key]; print(f"   {key}: warning level {r['trigger']:.2f} of the high-flow level, onsets warned on the test years {100 * r['onset_pod']:.1f}%, onset verdict {r['onset_verdict']}")
print("   onset verdicts over all records:", sorted({v['onset_verdict'] for v in rows.values()}), "| 'all high-flow days' verdicts at lead 1, plain:", {k.split(':')[0]: v['all_verdict'] for k, v in rows.items() if k.endswith('l1:plain')})
out["verdicts"] = dict(n=vn, differ=len(vb), details=vb[:5], onset_verdicts=sorted({v["onset_verdict"] for v in rows.values()}), rows=rows)
out["total_mismatches"] = len(bad_all) + len(ov_bad) + len(gr_bad) + len(cl_bad) + len(vb) + len(out["e1b_success"]["mismatches"]) + len(out["roadmap_step_1b"]["mismatches"])
print("\nTOTAL mismatches in readings / overlaps / gain_rule / e1b_success / closure / verdicts:", out["total_mismatches"])
jsave("a14_readings.json", out)
