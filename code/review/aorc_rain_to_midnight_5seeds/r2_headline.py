"""Round 2, checks h1-h6: the five-seed numbers of results/aorc_rain_to_midnight.json recomputed from results/preds with my own numpy scoring (r_common.score_arrays):
h1 summary_test of every set (13 sets x 3 leads), h2 seed averages = the mean of the single seeds and seed_ranges (by_seed and ranges), h3 the simple models (yardstick.table_test),
h4 the reading blocks (value_of_*, second_rain_product_alone, aorc_against_*, nldas_rain_against_*, rain_table_with_the_right_clock; overlaps and sign tests) and gain_rule,
h5 e1b_success, h6 the yardstick lines of the LSTM against the simple models. One prediction array in memory at a time."""
from r_common import *
from scipy.stats import wilcoxon, binomtest

J = jload(RESULTS)
C = Compare()
mine = {}
METRICS = ("nse_median", "onset_pod", "onset_far", "onset_csi", "all_pod", "all_far", "all_csi")


def metric(r, k, m):
    if m == "nse_median":
        return r[k]["nse_median"]
    part, x = m.split("_")
    return r[k][part][x]


# ---- h1 summary_test -------------------------------------------------------------------------------------------------------------------------------
ST = J["summary_test"]["sets"]
ref_keys = None
for n, blk in ST.items():
    r = score(f"lstm_{n}")
    if ref_keys is None:
        ref_keys = r[1]["onset_keys"]
    assert r[1]["onset_keys"] == ref_keys, f"{n}: the onsets differ (observed flow not the same)"
    for k in LEADS:
        lk = f"l{k}"
        C.add(f"h1 {n} {lk} nse_median", blk[lk]["nse_median"], r[k]["nse_median"])
        C.add(f"h1 {n} {lk} nse_mean", blk[lk]["nse_mean"], r[k]["nse_mean"])
        for part in ("onset", "all"):
            for x in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                C.add(f"h1 {n} {lk} {part} {x}", blk[lk][part][x], r[k][part][x])
mine["n_onsets_l1"] = len(ref_keys)

# ---- h2 seed averages and seed ranges ---------------------------------------------------------------------------------------------------------------
SR = J["seed_ranges"]["sets"]
SINGLE = {}
for n in SR:
    seeds = seeds_of(n)
    acc = None
    for s in seeds:
        d, b, o, p, meta = load_npz(f"lstm_{n}_s{s}")
        acc = p.copy() if acc is None else acc + p
        SINGLE[(n, s)] = score_arrays(d, b, o, p)
        del p
    d, b, o, pa, meta = load_npz(f"lstm_{n}")
    mx = float(np.nanmax(np.abs(acc / len(seeds) - pa)))
    C.add(f"h2 {n} average = mean of {len(seeds)} single seeds (max abs diff < 1e-6)", True, mx < 1e-6)
    C.add(f"h2 {n} meta seeds of the average", meta.get("seeds"), seeds)
    mine.setdefault("average_max_abs_diff", {})[n] = mx
    del acc, pa
    for k in LEADS:
        lk = f"l{k}"
        C.add(f"h2 {n} {lk} n_seeds", SR[n][lk]["n_seeds"], len(seeds))
        for m in METRICS:
            v = [metric(SINGLE[(n, s)], k, m) for s in seeds]
            fs = SR[n][lk]["by_seed"][m]
            for i, s in enumerate(seeds):
                C.add(f"h2 {n} {lk} {m} seed {s}", fs[i] if i < len(fs) else None, v[i])
            C.add(f"h2 {n} {lk} {m} range low", SR[n][lk][m][0], min(v))
            C.add(f"h2 {n} {lk} {m} range high", SR[n][lk][m][1], max(v))
mine["seeds"] = {n: seeds_of(n) for n in SR}
mine["wqa_l1_single_seed_onset_hits"] = [SINGLE[("wqa", s)][1]["onset"]["hits"] for s in seeds_of("wqa")]
mine["wqa_l1_single_seed_nse_median"] = [SINGLE[("wqa", s)][1]["nse_median"] for s in seeds_of("wqa")]

# ---- h3 the simple models ---------------------------------------------------------------------------------------------------------------------------
TT = J["yardstick"]["table_test"]
for n in ("base_linreg_wqa", "base_rf_wqa", "base_linreg_wq", "base_rf_wq"):
    r = score(n)
    for k in LEADS:
        blk = TT[f"l{k}"][n]
        C.add(f"h3 {n} l{k} nse_median", blk["nse_median"], r[k]["nse_median"])
        for part in ("onset", "all"):
            for x in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                C.add(f"h3 {n} l{k} {part} {x}", blk[part][x], r[k][part][x])


# ---- h4 reading blocks ------------------------------------------------------------------------------------------------------------------------------
def singles(setname):
    n = setname.replace("lstm_", "")
    if (n, seeds_of(n)[0]) not in SINGLE:
        for s in seeds_of(n):
            d, b, o, p, meta = load_npz(f"lstm_{n}_s{s}")
            SINGLE[(n, s)] = score_arrays(d, b, o, p)
            del p
    return [SINGLE[(n, s)] for s in seeds_of(n)]


def pair(w, wo, k):
    rw, ro = score(w), score(wo)
    sw, so = singles(w), singles(wo)
    ow = [x[k]["onset"]["pod"] for x in sw]
    oo = [x[k]["onset"]["pod"] for x in so]
    nw = [x[k]["nse_median"] for x in sw]
    no = [x[k]["nse_median"] for x in so]
    dif = rw[k]["nse"] - ro[k]["nse"]
    return dict(on=dict(warned_with=rw[k]["onset"]["pod"], warned_without=ro[k]["onset"]["pod"], difference_points=100 * (rw[k]["onset"]["pod"] - ro[k]["onset"]["pod"]),
                        hits_with=rw[k]["onset"]["hits"], hits_without=ro[k]["onset"]["hits"], onsets_in_the_test_years=rw[k]["onset"]["hits"] + rw[k]["onset"]["misses"],
                        hits_difference=rw[k]["onset"]["hits"] - ro[k]["onset"]["hits"], seed_range_with=[min(ow), max(ow)], seed_range_without=[min(oo), max(oo)],
                        every_seed_above_every_seed=min(ow) > max(oo), every_seed_below_every_seed=max(ow) < min(oo),
                        difference_by_seed_points=[100 * (a - b) for a, b in zip(ow, oo)], false_warnings_with=rw[k]["onset"]["far"], false_warnings_without=ro[k]["onset"]["far"]),
                nse=dict(median_with=rw[k]["nse_median"], median_without=ro[k]["nse_median"], gain=rw[k]["nse_median"] - ro[k]["nse_median"], seed_range_with=[min(nw), max(nw)],
                         seed_range_without=[min(no), max(no)], every_seed_above_every_seed=min(nw) > max(no), difference_by_seed=[a - b for a, b in zip(nw, no)],
                         n_basins_higher=int((dif > 0).sum()), n_basins_lower=int((dif < 0).sum()), n_basins=len(dif), median_basin_difference=float(np.median(dif)),
                         wilcoxon_p=float(wilcoxon(dif).pvalue)))


def cmp_list(key, fv, mv):
    if isinstance(fv, list) and fv and isinstance(fv[0], (int, float)) and not isinstance(fv[0], bool):
        C.add(key + " length", len(fv), len(mv))
        for i, (a, b) in enumerate(zip(fv, mv)):
            C.add(f"{key}[{i}]", a, b)
    else:
        C.add(key, fv, mv)


READ_BLOCKS = ["value_of_hours_09_to_24", "value_of_evening_hours_beyond_utc_day", "value_of_hours_09_to_utc_day_end", "second_rain_product_alone", "aorc_against_era5", "aorc_against_nldas",
               "nldas_rain_against_aorc_over_the_utc_day", "aorc_against_nldas_and_era5_together"]
ON_KEYS = ["warned_with", "warned_without", "difference_points", "hits_with", "hits_without", "onsets_in_the_test_years", "hits_difference", "seed_range_with", "seed_range_without",
           "every_seed_above_every_seed", "every_seed_below_every_seed", "difference_by_seed_points", "false_warnings_with", "false_warnings_without"]
NSE_KEYS = ["median_with", "median_without", "gain", "seed_range_with", "seed_range_without", "every_seed_above_every_seed", "difference_by_seed", "n_basins_higher", "n_basins",
            "median_basin_difference", "wilcoxon_p"]


def leadblocks(name, blk):
    for lk, v in blk.items():
        if not lk.startswith("l"):
            continue
        k = int(lk[1])
        p = pair(v["with_set"], v["without_set"], k)
        for x in ON_KEYS:
            cmp_list(f"h4 {name} {lk} onsets {x}", v["onsets"][x], p["on"][x])
        for x in NSE_KEYS:
            cmp_list(f"h4 {name} {lk} nse {x}", v["nse"][x], p["nse"][x])


def overlap(name, ov):
    a, b = score(ov["first"])[ov["lead"]]["onset_hit"], score(ov["second"])[ov["lead"]]["onset_hit"]
    m = dict(n_onsets=len(a), hits_first=int(a.sum()), hits_second=int(b.sum()), warned_by_both=int((a & b).sum()), warned_only_by_first=int((a & ~b).sum()),
             warned_only_by_second=int((~a & b).sum()), warned_by_neither=int((~a & ~b).sum()), warned_by_at_least_one=int((a | b).sum()))
    n10, n01 = m["warned_only_by_first"], m["warned_only_by_second"]
    m["sign_test_p"] = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else float("nan")
    for x, y in m.items():
        if x == "sign_test_p":
            C.add(f"h4 {name} overlap {x} (relative 1e-9)", True, abs(ov[x] - y) <= 1e-9 * max(abs(y), 1e-300))
        else:
            C.add(f"h4 {name} overlap {x}", ov[x], y)
    return m


OVER = {}
for name in READ_BLOCKS:
    leadblocks(name, J[name]["by_lead"])
    if "overlap_of_warned_onsets" in J[name]:
        OVER[name] = overlap(name, J[name]["overlap_of_warned_onsets"])
RT = J["rain_table_with_the_right_clock"]
for sub in ("against_rain_table_alone", "against_aorc_as_an_extra_column"):
    leadblocks(f"rain_table_with_the_right_clock.{sub}", RT[sub])
for sub, ov in RT["overlap_of_warned_onsets"].items():
    OVER[f"rain_table_with_the_right_clock.{sub}"] = overlap(f"rain_table_with_the_right_clock.{sub}", ov)
mine["overlaps"] = OVER

# gain_rule
GK_NSE = {"median_with": "median_with", "median_without": "median_without", "gain": "gain", "seed_range_with": "seed_range_with", "seed_range_without": "seed_range_without",
          "single_seeds_all_above": "every_seed_above_every_seed", "n_basins_higher": "n_basins_higher", "n_basins_lower": "n_basins_lower", "n_basins": "n_basins",
          "median_basin_difference": "median_basin_difference", "wilcoxon_p": "wilcoxon_p"}
GK_ON = {"warned_with": "warned_with", "warned_without": "warned_without", "seed_range_with": "seed_range_with", "seed_range_without": "seed_range_without", "false_with": "false_warnings_with",
         "false_without": "false_warnings_without", "single_seeds_all_above": "every_seed_above_every_seed", "single_seeds_all_below": "every_seed_below_every_seed"}
for s, blocks in J["gain_rule"].items():
    for bn, bl in blocks.items():
        for lk, v in bl.items():
            if not lk.startswith("l"):
                continue
            k = int(lk[1])
            with_set = {"control_against_rain_table_alone": "lstm_wqa_to09", "utc_day_against_rain_table_alone": "lstm_wqa_utc",
                        "utc_day_against_matched_window_to_09": "lstm_wqa_utc"}.get(bn, f"lstm_{s}")          # the blocks named '<control>_against_...' read the control set
            p = pair(with_set, v["against"], k)
            for fk, mk in GK_NSE.items():
                cmp_list(f"h4 gain_rule {s} {bn} {lk} nse {fk}", v["nse"][fk], p["nse"][mk])
            for fk, mk in GK_ON.items():
                cmp_list(f"h4 gain_rule {s} {bn} {lk} onset {fk}", v["onset"][fk], p["on"][mk])
            C.add(f"h4 gain_rule {s} {bn} {lk} onset gain", v["onset"]["gain"], p["on"]["warned_with"] - p["on"]["warned_without"])
            n = p["nse"]
            cg = n["every_seed_above_every_seed"] and n["n_basins_higher"] > n["n_basins"] / 2 and n["wilcoxon_p"] < 0.05
            C.add(f"h4 gain_rule {s} {bn} {lk} nse clear_gain", v["nse"]["clear_gain"], bool(cg))
            C.add(f"h4 gain_rule {s} {bn} {lk} onset clear_gain", v["onset"]["clear_gain"], bool(p["on"]["every_seed_above_every_seed"]))

# ---- h5 e1b_success ---------------------------------------------------------------------------------------------------------------------------------
E = J["e1b_success"]["wqa"]
R = {n: [min(x[1]["onset"]["pod"] for x in singles(n)), max(x[1]["onset"]["pod"] for x in singles(n))] for n in ("wqa", "wqa_to09", "wq", "wqn", "wqh", "wqa_utc", "wqa_rep")}


def pod(n):
    return score(f"lstm_{n}")[1]["onset"]["pod"]


for fk, n in dict(with_aorc_rain_to_midnight="wqa", nldas="wqn", era5_rain_to_midnight="wqh", control="wqa_to09", rain_table_alone="wq").items():
    C.add(f"h5 onset_warned {fk}", E["onset_warned"][fk], pod(n))
for n in ("wqa", "wqa_to09", "wq", "wqn", "wqh"):
    cmp_list(f"h5 seed_ranges {n}", E["seed_ranges"][n], R[n])


def above(a, b):
    return R[a][0] > R[b][1]


my = dict(every_seed_above_every_seed_with_nldas_rain=above("wqa", "wqn"), every_seed_above_every_seed_of_the_control=above("wqa", "wqa_to09"),
          every_seed_above_every_seed_with_the_rain_table_alone=above("wqa", "wq"), control_flat=not above("wqa_to09", "wq"),
          every_seed_above_every_seed_with_era5_rain_to_midnight=above("wqa", "wqh"), every_seed_of_era5_rain_to_midnight_above_every_seed_with_aorc=above("wqh", "wqa"),
          control_utc_flat=not above("wqa_utc", "wq"), every_seed_above_every_seed_of_the_control_utc=above("wqa", "wqa_utc"))
my["met"] = my["every_seed_above_every_seed_with_nldas_rain"] and my["control_flat"]
for x, y in my.items():
    C.add(f"h5 {x}", E[x], y)
C.add("h5 control_utc_onset_warned", E["control_utc_onset_warned"], pod("wqa_utc"))
cmp_list("h5 control_utc_seed_range", E["control_utc_seed_range"], R["wqa_utc"])
RB = E["roadmap_step_1b"]
C.add("h5 roadmap gain_holds", RB["gain_holds"], above("wqa", "wq") and above("wqa", "wqa_to09") and not above("wqh", "wqa"))
C.add("h5 roadmap gain_grows", RB["gain_grows"], above("wqa", "wq") and above("wqa", "wqa_to09") and not above("wqh", "wqa") and above("wqa", "wqh"))
SV = RB["second_variant"]
C.add("h5 second_variant replaced", SV["onset_warned"]["replaced"], pod("wqa_rep"))
cmp_list("h5 second_variant seed range replaced", SV["seed_ranges"]["replaced"], R["wqa_rep"])
C.add("h5 second_variant every_seed_above", SV["every_seed_above_every_seed_of_the_rain_table_alone"], above("wqa_rep", "wq"))
mine["e1b_success"] = dict(my, seed_ranges=R)

# ---- h6 LSTM against the simple models (yardstick) --------------------------------------------------------------------------------------------------
LA = J["yardstick"]["lstm_against_simple"]
for key, blk in LA.items():
    lstm, base = key.split("__")
    for lk, v in blk.items():
        if not lk.startswith("l"):
            continue
        k = int(lk[1])
        rl, rb = score(lstm)[k], score(base)[k]
        sl = singles(lstm)
        dif = rl["nse"] - rb["nse"]
        nse_s = [x[k]["nse_median"] for x in sl]
        pods = [x[k]["onset"]["pod"] for x in sl]
        m = dict(lstm_nse_median=rl["nse_median"], baseline_nse_median=rb["nse_median"], difference_of_medians=rl["nse_median"] - rb["nse_median"], n_basins_lstm_higher=int((dif > 0).sum()),
                 n_basins=len(dif), median_nse_difference=float(np.median(dif)), wilcoxon_p=float(wilcoxon(dif).pvalue), lstm_seed_median_min=min(nse_s), lstm_seed_median_max=max(nse_s),
                 n_basins_lstm_higher_by_seed=[int(((x[k]["nse"] - rb["nse"]) > 0).sum()) for x in sl], lstm_onset_pod=rl["onset"]["pod"], baseline_onset_pod=rb["onset"]["pod"],
                 lstm_onset_far=rl["onset"]["far"], baseline_onset_far=rb["onset"]["far"], lstm_onset_pod_seed_range=[min(pods), max(pods)])
        for x, y in m.items():
            if x in v:
                cmp_list(f"h6 {key} {lk} {x}", v[x], y)
        if k == 1:
            mine.setdefault("lstm_trainings_above_simple_onsets_l1", {})[base] = int(sum(p_ > rb["onset"]["pod"] for p_ in pods))
NI = J["yardstick"]["new_input_given_to_each_model"]
for mdl, blk in NI.items():
    if mdl == "lstm":
        w, wo = "lstm_wqa", "lstm_wq"
    else:
        kind = "linreg" if ("linreg" in mdl or "ridge" in mdl) else "rf"
        w, wo = f"base_{kind}_wqa", f"base_{kind}_wq"
    for lk, v in blk["by_lead"].items():
        k = int(lk[1])
        a, b = score(w)[k], score(wo)[k]
        C.add(f"h6 new_input {mdl} {lk} onset_gain_points", v["onset_gain_points"], 100 * (a["onset"]["pod"] - b["onset"]["pod"]))
        C.add(f"h6 new_input {mdl} {lk} nse_gain", v["nse_gain"], a["nse_median"] - b["nse_median"])
mine["new_input_models"] = list(NI)

out = dict(compare=C.rows, summary=C.summary(), mine=mine, n_values=len(C.rows), files_read=READ)
jsave("r2_headline.json", out)
print(json.dumps(C.summary(), indent=1)[:3000])
w = score("lstm_wqa")[1]
print("wqa l1:", w["onset"], "nse_median", w["nse_median"], "single hits", mine["wqa_l1_single_seed_onset_hits"])
print("trainings above simple:", mine.get("lstm_trainings_above_simple_onsets_l1"))
