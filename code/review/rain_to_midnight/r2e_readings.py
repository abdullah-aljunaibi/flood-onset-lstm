"""ROUND 2, check (e): the arithmetic of the new reading fields, recomputed with the reviewer's own code from results/preds: differences, seed by seed, clear / not clear flags, Wilcoxon p, overlap of warned onsets,
the matched-control flags of e1_success, the gain-rule and closure entries of the new sets, and every clause of outcome_caveat."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from scipy.stats import wilcoxon, binomtest
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json")
ids = list(data())
SETS = ["wq", "wqn", "wqh", "wqh_older", "wqh_to09", "wqh_utc", "wqnh", "wqf"]
SEEDS = (0, 1, 2)
POD, NSE_S, NSE_B, CNT, FLAGS = {}, {}, {}, {}, {}
for n in SETS:
    for k in (1, 2, 3):
        POD[n, k] = [counts(f"lstm_{n}_s{s}", k)["pod"] for s in SEEDS]
        NSE_S[n, k] = [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", k).values()))) for s in SEEDS]
        nb = nse_per_basin(f"lstm_{n}", k); NSE_B[n, k] = np.array([nb[g] for g in ids]); CNT[n, k] = counts(f"lstm_{n}", k)
        t = onset_table(f"lstm_{n}", k)
        FLAGS[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])             # one True/False per onset, basin after basin
above = lambda a, b: min(a) > max(b)

def reading(T, C, k):
    cT, cC = CNT[T, k], CNT[C, k]
    n_hi = int((NSE_B[T, k] > NSE_B[C, k]).sum()); n_lo = int((NSE_B[T, k] < NSE_B[C, k]).sum())
    try: p = float(wilcoxon(NSE_B[T, k], NSE_B[C, k]).pvalue)
    except ValueError: p = float("nan")
    up, down = above(NSE_S[T, k], NSE_S[C, k]), above(NSE_S[C, k], NSE_S[T, k])
    small = p == p and p < 0.05
    gain, loss = up and n_hi > 6 and small, down and n_lo > 6 and small
    o_up, o_down = above(POD[T, k], POD[C, k]), above(POD[C, k], POD[T, k])
    return dict(onsets=dict(warned_with=cT["pod"], warned_without=cC["pod"], difference_points=100 * (cT["pod"] - cC["pod"]), hits_with=cT["hits"], hits_without=cC["hits"], onsets_in_the_test_years=cT["n_onsets"], hits_difference=cT["hits"] - cC["hits"],
                     seed_range_with=[min(POD[T, k]), max(POD[T, k])], seed_range_without=[min(POD[C, k]), max(POD[C, k])], every_seed_above_every_seed=o_up, every_seed_below_every_seed=o_down,
                     difference_by_seed_points=[100 * (a - b) for a, b in zip(POD[T, k], POD[C, k])], false_warnings_with=cT["far"], false_warnings_without=cC["far"], reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"),
                nse=dict(median_with=float(np.median(NSE_B[T, k])), median_without=float(np.median(NSE_B[C, k])), gain=float(np.median(NSE_B[T, k]) - np.median(NSE_B[C, k])), seed_range_with=[min(NSE_S[T, k]), max(NSE_S[T, k])], seed_range_without=[min(NSE_S[C, k]), max(NSE_S[C, k])],
                         every_seed_above_every_seed=up, difference_by_seed=[a - b for a, b in zip(NSE_S[T, k], NSE_S[C, k])], n_basins_higher=n_hi, median_basin_difference=float(np.median(NSE_B[T, k] - NSE_B[C, k])), wilcoxon_p=p,
                         reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"))

def overlap(A, C, k=1):
    fa, fc = FLAGS[A, k], FLAGS[C, k]
    both, oa, oc = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    return dict(n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=oa, warned_only_by_second=oc, warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + oa + oc,
                sign_test_p=float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))

def cmp_dict(mine, theirs, path=""):
    """Largest absolute deviation of every number and every mismatch of a flag or text, recursively."""
    worst, bad = 0.0, []
    if isinstance(mine, dict):
        for k, v in mine.items():
            w, b = cmp_dict(v, theirs[k], path + "/" + k); worst = max(worst, w); bad += b
    elif isinstance(mine, (list, tuple)):
        assert len(mine) == len(theirs), path
        for i, (a, b_) in enumerate(zip(mine, theirs)):
            w, b = cmp_dict(a, b_, path + f"[{i}]"); worst = max(worst, w); bad += b
    elif isinstance(mine, (bool, str)):
        if mine != theirs: bad.append((path, mine, theirs))
    else:
        if theirs is None and mine != mine: pass
        elif abs(mine - theirs) > 1e-9: bad.append((path, mine, theirs)); worst = max(worst, abs(mine - theirs))
        else: worst = max(worst, abs(mine - theirs))
    return worst, bad

out = {"readings": {}}
READ = {"value_of_hours_09_to_24": ("wqh", "wqh_to09"), "value_of_evening_hours_beyond_utc_day": ("wqh", "wqh_utc"), "value_of_hours_09_to_utc_day_end": ("wqh_utc", "wqh_to09"),
        "second_rain_product_alone": ("wqh_to09", "wq"), "nldas_rain_against_era5_over_the_utc_day": ("wqn", "wqh_utc")}
tot_bad = []
print("pair readings (3 leads each): my own numbers against the JSON field by field (numbers to 1e-9, flags and words exactly)")
for key, (T, C) in READ.items():
    worst = 0
    for k in (1, 2, 3):
        mine = reading(T, C, k)
        w, bad = cmp_dict(mine, {"onsets": R[key]["by_lead"][f"l{k}"]["onsets"], "nse": R[key]["by_lead"][f"l{k}"]["nse"]}); worst = max(worst, w); tot_bad += [(key, k) + b for b in bad]
    out["readings"][key] = dict(pair=f"lstm_{T} minus lstm_{C}", max_abs_deviation=worst, mismatches=[b for b in tot_bad if b[0] == key])
    l1 = reading(T, C, 1)
    print(f"  {key:44s} lstm_{T} - lstm_{C}: lead 1 onsets {100*l1['onsets']['warned_with']:.1f}% vs {100*l1['onsets']['warned_without']:.1f}% ({l1['onsets']['difference_points']:+.1f}, {l1['onsets']['hits_with']} vs {l1['onsets']['hits_without']} hits, by seed {[round(x,1) for x in l1['onsets']['difference_by_seed_points']]}) every above {l1['onsets']['every_seed_above_every_seed']} -> {l1['onsets']['reading']}; NSE {l1['nse']['gain']:+.3f} ({l1['nse']['n_basins_higher']}/12, p {l1['nse']['wilcoxon_p']:.4f}) -> {l1['nse']['reading']};  max dev {worst:.1e}")
BOTH = {"against_rain_to_midnight": ("wqnh", "wqh"), "against_nldas_rain": ("wqnh", "wqn"), "against_rain_table_alone": ("wqnh", "wq")}
for key, (T, C) in BOTH.items():
    worst = 0
    for k in (1, 2, 3):
        mine = reading(T, C, k)
        w, bad = cmp_dict(mine, {"onsets": R["both_products"][key][f"l{k}"]["onsets"], "nse": R["both_products"][key][f"l{k}"]["nse"]}); worst = max(worst, w); tot_bad += [("both_products/" + key, k) + b for b in bad]
    out["readings"]["both_products/" + key] = dict(pair=f"lstm_{T} minus lstm_{C}", max_abs_deviation=worst)
    l1 = reading(T, C, 1)
    print(f"  both_products {key:26s} lstm_{T} - lstm_{C}: lead 1 onsets {100*l1['onsets']['warned_with']:.1f}% vs {100*l1['onsets']['warned_without']:.1f}% ({l1['onsets']['difference_points']:+.1f}, {l1['onsets']['hits_with']} vs {l1['onsets']['hits_without']} hits) -> {l1['onsets']['reading']}; NSE {l1['nse']['gain']:+.3f} ({l1['nse']['n_basins_higher']}/12, p {l1['nse']['wilcoxon_p']:.4f}) -> {l1['nse']['reading']};  max dev {worst:.1e}")
# overlaps
OV = {("value_of_hours_09_to_24", "overlap_of_warned_onsets"): ("wqh", "wqh_to09"), ("value_of_evening_hours_beyond_utc_day", "overlap_of_warned_onsets"): ("wqh", "wqh_utc")}
ov_dev = []
for (key, sub), (A, C) in OV.items():
    m = overlap(A, C); j = R[key][sub]
    ov_dev.append(max(abs(m[k] - j[k]) for k in m if isinstance(m[k], (int, float)) and m[k] == m[k])); out.setdefault("overlaps", {})[key] = dict(mine=m, json_equal=ov_dev[-1] < 1e-9)
for sub, (A, C) in {"rain_to_midnight_against_nldas_rain": ("wqh", "wqn"), "both_against_rain_to_midnight": ("wqnh", "wqh"), "both_against_nldas_rain": ("wqnh", "wqn")}.items():
    m = overlap(A, C); j = R["both_products"]["overlap_of_warned_onsets"][sub]
    ov_dev.append(max(abs(m[k] - j[k]) for k in m if isinstance(m[k], (int, float)) and m[k] == m[k])); out.setdefault("overlaps", {})["both_products/" + sub] = dict(mine=m, json_equal=ov_dev[-1] < 1e-9)
print("\noverlap of warned onsets (5 pairs): own loops against the JSON, max deviation", max(ov_dev))
for k, v in out["overlaps"].items():
    m = v["mine"]; print(f"  {k:60s} both {m['warned_by_both']}, only first {m['warned_only_by_first']}, only second {m['warned_only_by_second']}, neither {m['warned_by_neither']}, sign test p {m['sign_test_p']:.3g}")
out["max_overlap_dev"] = max(ov_dev)

# e1_success: the matched-control flags
print("\ne1_success (with the gauge): matched-window flags")
e1 = R["e1_success"]["wqh"]
r1 = {x: [min(POD[x, 1]), max(POD[x, 1])] for x in ("wqh", "wqh_older", "wq", "wqn", "wqh_to09", "wqh_utc")}
mine_e1 = dict(every_seed_above_every_seed_with_nldas_rain=above(POD["wqh", 1], POD["wqn", 1]), every_seed_above_every_seed_of_the_control=above(POD["wqh", 1], POD["wqh_older", 1]), every_seed_above_every_seed_with_the_rain_table_alone=above(POD["wqh", 1], POD["wq", 1]),
               control_flat=not above(POD["wqh_older", 1], POD["wq", 1]))
mine_e1["met"] = all(mine_e1.values())
for c, key in (("wqh_to09", "to09"), ("wqh_utc", "utc")):
    mine_e1[f"control_{key}_onset_warned"] = CNT[c, 1]["pod"]; mine_e1[f"control_{key}_seed_range"] = r1[c]; mine_e1[f"control_{key}_flat"] = not above(POD[c, 1], POD["wq", 1]); mine_e1[f"every_seed_above_every_seed_of_the_control_{key}"] = above(POD["wqh", 1], POD[c, 1])
w, bad = cmp_dict(mine_e1, {k: e1[k] for k in mine_e1}); out["e1_success"] = dict(mine=mine_e1, max_dev=w, mismatches=bad)
print("  mine:", {k: (v if not isinstance(v, list) else [round(x, 4) for x in v]) for k, v in mine_e1.items() if k != "control_to09_onset_warned"})
print("  deviation from the JSON:", w, bad)

# gain_rule and closure of the new sets (key fields)
print("\ngain_rule entries for the new sets: n_basins_higher, p, readings, seed flags against the JSON")
GR = {("wqh", "against_matched_window_to_09"): ("wqh", "wqh_to09"), ("wqh", "against_utc_day"): ("wqh", "wqh_utc"), ("wqh", "matched_window_to_09_against_rain_table_alone"): ("wqh_to09", "wq"), ("wqh", "utc_day_against_rain_table_alone"): ("wqh_utc", "wq"),
      ("wqh", "utc_day_against_matched_window_to_09"): ("wqh_utc", "wqh_to09"), ("wqnh", "against_rain_to_midnight"): ("wqnh", "wqh"), ("wqnh", "against_nldas_rain"): ("wqnh", "wqn"), ("wqnh", "against_rain_table_alone"): ("wqnh", "wq")}
gr_bad = []; gr_n = 0
for (fam, key), (T, C) in GR.items():
    for k in (1, 2, 3):
        m = reading(T, C, k); j = R["gain_rule"][fam][key][f"l{k}"]
        chk = [(m["nse"]["reading"], j["nse"]["reading"]), (m["onsets"]["reading"], j["onset"]["reading"]), (m["nse"]["n_basins_higher"], j["nse"]["n_basins_higher"]), (m["nse"]["every_seed_above_every_seed"], j["nse"]["single_seeds_all_above"]),
               (m["onsets"]["every_seed_above_every_seed"], j["onset"]["single_seeds_all_above"]), (m["onsets"]["every_seed_below_every_seed"], j["onset"]["single_seeds_all_below"])]
        nums = [(m["nse"]["wilcoxon_p"], j["nse"]["wilcoxon_p"]), (m["onsets"]["warned_with"], j["onset"]["warned_with"]), (m["onsets"]["warned_without"], j["onset"]["warned_without"]), (m["nse"]["median_with"], j["nse"]["median_with"]),
                (m["nse"]["median_without"], j["nse"]["median_without"]), (m["nse"]["median_basin_difference"], j["nse"]["median_basin_difference"])]
        gr_n += len(chk) + len(nums)
        gr_bad += [(fam, key, k, a, b) for a, b in chk if a != b] + [(fam, key, k, a, b) for a, b in nums if abs(a - b) > 1e-9]
print(f"  8 comparisons x 3 leads x 12 fields = {gr_n} values compared, {len(gr_bad)} differ", gr_bad[:3])
out["gain_rule_new"] = dict(n=gr_n, differ=len(gr_bad), details=gr_bad[:5])
print("\nclosure of the new sets (share of the gap to wqf closed, NSE and onsets), 3 sets x 3 leads, and the 'references' versus entries")
cl_bad = []; cl_n = 0
for name, base in (("wqh_to09", "wq"), ("wqh_utc", "wq"), ("wqnh", "wq")):
    for k in (1, 2, 3):
        row = R["closure"][name]["by_lead"][k - 1]
        b, c, n_ = float(np.median(NSE_B[base, k])), float(np.median(NSE_B["wqf", k])), float(np.median(NSE_B[name, k]))
        ob, oc, on = CNT[base, k]["pod"], CNT["wqf", k]["pod"], CNT[name, k]["pod"]
        pairs = [(row["nse_base"], b), (row["nse_with"], n_), (row["gap"], c - b), (row["gain"], n_ - b), (row["share_of_gap_closed"], (n_ - b) / (c - b)), (row["onset_warned_base"], ob), (row["onset_warned_with"], on), (row["onset_share_of_gap_closed"], (on - ob) / (oc - ob)),
                 (row["basins_better"], int((NSE_B[name, k] > NSE_B[base, k]).sum())), (row["median_basin_gain"], float(np.median(NSE_B[name, k] - NSE_B[base, k])))]
        for ref_label, Cc in row["references"].items() and {"rain_to_midnight": "wqh", "nldas_rain": "wqn"}.items():
            if ref_label not in row["references"]: continue
            v = row["references"][ref_label]
            pairs += [(v["nse_gain_over_it"], n_ - float(np.median(NSE_B[Cc, k]))), (v["onset_gain_over_it"], on - CNT[Cc, k]["pod"]), (v["onset_share_of_gap_closed_beyond_it"], (on - CNT[Cc, k]["pod"]) / (oc - ob)),
                      (v["basins_better_than_it"], int((NSE_B[name, k] > NSE_B[Cc, k]).sum())), (v["own_onset_share_of_gap_closed"], (CNT[Cc, k]["pod"] - ob) / (oc - ob))]
        cl_n += len(pairs); cl_bad += [(name, k, a, b) for a, b in pairs if abs(a - b) > 1e-9]
        if k == 1: print(f"  {name:9s} lead 1: onset gap closed {100*(on-ob)/(oc-ob):.1f}% (JSON {100*row['onset_share_of_gap_closed']:.1f}%), NSE gap closed {100*(n_-b)/(c-b):.1f}% (JSON {100*row['share_of_gap_closed']:.1f}%)")
print(f"  {cl_n} values compared, {len(cl_bad)} differ", cl_bad[:3])
out["closure_new"] = dict(n=cl_n, differ=len(cl_bad), details=cl_bad[:5])

# outcome_caveat, clause by clause
print("\noutcome_caveat clause by clause")
th, tn = CNT["wqh", 1], CNT["wqn", 1]
claims = {"same share as NLDAS, 183 hits of 611 in both": (th["hits"], tn["hits"], th["misses"], tn["misses"]) == (183, 183, 428, 428) and th["n_onsets"] == 611,
          "the three single trainings span 27.5% to 31.6%": (round(100 * min(POD['wqh', 1]), 1), round(100 * max(POD['wqh', 1]), 1)) == (27.5, 31.6),
          "control 21% against 19% with the rain table alone": (round(100 * CNT["wqh_older", 1]["pod"]), round(100 * CNT["wq", 1]["pod"])) == (21, 19),
          "the control is not flat (every control seed above every rain-table seed)": above(POD["wqh_older", 1], POD["wq", 1]),
          "rule not met (not every wqh seed above every wqn seed, control not flat)": (not above(POD["wqh", 1], POD["wqn", 1])) and above(POD["wqh_older", 1], POD["wq", 1])}
for k, v in claims.items(): print(f"  {'TRUE ' if v else 'FALSE'}  {k}")
out["outcome_caveat_clauses_true"] = all(claims.values())
out["outcome_caveat_text_equals_rebuilt"] = None
rebuilt = (f"The NLDAS rain up to the evening gives the same share ({round(100*th['pod'])}%, {th['hits']} hits of {th['n_onsets']} in both); the three single trainings with the rain to midnight span {100*min(POD['wqh',1]):.1f}% to {100*max(POD['wqh',1]):.1f}%; "
           f"the control is not flat ({round(100*CNT['wqh_older',1]['pod'])}% against {round(100*CNT['wq',1]['pod'])}% with the rain table alone); the success rule of the work order (clearly above the NLDAS result in every training, control flat) is not met.")
out["outcome_caveat_text_equals_rebuilt"] = (rebuilt == R["outcome_caveat"])
out["outcome_caveat_starts_with_rebuilt"] = R["outcome_caveat"].startswith(rebuilt)          # round 3: two clauses were appended after round 2 (checked by r3c_clauses.py), so the five older clauses are the beginning of the text
print("  my rebuilt sentence (the five older clauses) equals the JSON's outcome_caveat:", out["outcome_caveat_text_equals_rebuilt"], "| is its beginning:", out["outcome_caveat_starts_with_rebuilt"], "(the two clauses appended later: r3c_clauses.py)")
out["total_mismatches"] = len(tot_bad) + len(gr_bad) + len(cl_bad)
print("\nTOTAL mismatches in readings / gain_rule / closure:", out["total_mismatches"])
jsave("r2e_readings.json", out)
