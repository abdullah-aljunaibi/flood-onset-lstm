"""Check 2: the five-seed numbers of results/rain_to_midnight.json against my scores (OUT/scores.json): summary_test, the simple models' test table, per_basin,
seed_ranges, models.<set>.n_seeds_scored and per_seed, every seed rule of gain_rule (single seeds above / below, basins won, Wilcoxon p, readings), the readings
value_of_hours_09_to_24, value_of_evening_hours_beyond_utc_day, value_of_hours_09_to_utc_day_end, second_rain_product_alone, nldas_rain_against_era5_over_the_utc_day,
both_products (with the paired onset counts), e1_success, closure, ladder, yardstick lstm_against_simple and new_input_given_to_each_model. Writes OUT/compare.json."""
from r_common import *
from scipy.stats import binomtest, wilcoxon

R = jload(PROJ / RESULTS)
M = jread("scores.json")
SC = M["scores"]
nb = len(M["ids"])
C = Compare(tol=1e-9)
BLOCK = {}


def walk(prefix, mine, theirs):
    """Compare every leaf of my dict with the file's dict at the same keys (keys I computed only)."""
    if isinstance(mine, dict):
        for k, v in mine.items():
            if not isinstance(theirs, dict) or k not in theirs:
                C.add(f"{prefix}.{k} (missing in file)", None, "present")
                continue
            walk(f"{prefix}.{k}", v, theirs[k])
    else:
        C.add(prefix, theirs, mine)


def block(name, fn):
    before = set(C.rows)
    fn()
    keys = [k for k in C.rows if k not in before]
    BLOCK[name] = dict(entries=len(keys), matching=sum(C.rows[k]["ok"] for k in keys))


E = lambda n: SC[n] if n.startswith("base_") else SC[f"lstm_{n}"]
seeds = lambda n: [SC[f"lstm_{n}_s{s}"] for s in SEEDS[n]]
med = lambda n, k: E(n)[f"l{k}"]["nse_median"]
ons = lambda n, k, c="pod": E(n)[f"l{k}"]["onset"][c]
nse_arr = lambda n, k: np.array(E(n)[f"l{k}"]["nse"])
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]
div = lambda a, b: a / b if b else float("nan")
CUT = ("hits", "misses", "false_alarms", "pod", "far", "csi")


def rng(n, k):
    S = [s[f"l{k}"] for s in seeds(n)]
    out = dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]))
    by = dict(nse_median=[s["nse_median"] for s in S])
    for kind in ("onset", "all"):
        for c in ("pod", "far", "csi"):
            out[f"{kind}_{c}"] = lo_hi([s[kind][c] for s in S])
            by[f"{kind}_{c}"] = [s[kind][c] for s in S]
    out["by_seed"] = by
    return out


RNG = {n: {f"l{k}": rng(n, k) for k in LEADS} for n in SETS}


def b_summary():
    for n in SETS:
        for k in LEADS:
            m, f = E(n)[f"l{k}"], R["summary_test"]["sets"][n][f"l{k}"]
            C.add(f"summary_test {n} l{k} nse_median", f["nse_median"], m["nse_median"])
            C.add(f"summary_test {n} l{k} nse_mean", f["nse_mean"], m["nse_mean"])
            for kind in ("onset", "all"):
                for c in CUT:
                    C.add(f"summary_test {n} l{k} {kind} {c}", f[kind][c], m[kind][c])
    for n in SIMPLE:
        for k in LEADS:
            f, m = R["yardstick"]["table_test"][f"l{k}"][n], E(n)[f"l{k}"]
            C.add(f"yardstick.table_test {n} l{k} nse_median", f["nse_median"], m["nse_median"])
            for kind in ("onset", "all"):
                for c in CUT:
                    C.add(f"yardstick.table_test {n} l{k} {kind} {c}", f[kind][c], m[kind][c])


def b_per_basin():
    for n in SETS:
        for k in LEADS:
            for i, g in enumerate(M["ids"]):
                C.add(f"per_basin nse {n} l{k} {g}", R["per_basin"]["nse"][n][f"l{k}"][g], E(n)[f"l{k}"]["nse"][i])
                C.add(f"per_basin onset_warned {n} l{k} {g}", R["per_basin"]["onset_warned"][n][f"l{k}"][g], E(n)[f"l{k}"]["onset_pod_by_basin"][i])


def b_ranges():
    for n in SETS:
        walk(f"seed_ranges.{n}", RNG[n], R["seed_ranges"]["sets"][n])
        C.add(f"models.{n}.n_seeds_scored", R["models"][n]["n_seeds_scored"], len(SEEDS[n]))
    for n in OWN:
        for j, s in enumerate(SEEDS[n]):
            ps = R["models"][n]["per_seed"][j]
            C.add(f"models.{n}.per_seed[{j}].seed", ps["seed"], s)
            C.add(f"models.{n}.per_seed s{s} test_nse_median", ps["test_nse_median"], [SC[f"lstm_{n}_s{s}"][f"l{k}"]["nse_median"] for k in LEADS])
            C.add(f"models.{n}.per_seed s{s} test_onset_warned", ps["test_onset_warned"], [SC[f"lstm_{n}_s{s}"][f"l{k}"]["onset"]["pod"] for k in LEADS])
            C.add(f"models.{n}.per_seed s{s} test_onset_false", ps["test_onset_false"], [SC[f"lstm_{n}_s{s}"][f"l{k}"]["onset"]["far"] for k in LEADS])
        C.add(f"training rows of {n}", sorted(t["seed"] for t in R["training"] if t["set"] == n), SEEDS[n])


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return int((d > 0).sum()), int((d < 0).sum()), float(np.median(d)), p


def rule(T, Cc, k):
    hi, lo, md, p = paired(nse_arr(T, k), nse_arr(Cc, k))
    sT, sC = RNG[T][f"l{k}"], RNG[Cc][f"l{k}"]
    up, down = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"])
    small = bool(p < P_CUT)
    gain, loss = up and hi > nb / 2 and small, down and lo > nb / 2 and small
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    return dict(against=f"lstm_{Cc}", lead=k,
                nse=dict(median_with=med(T, k), median_without=med(Cc, k), gain=med(T, k) - med(Cc, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                         single_seeds_all_above=up, single_seeds_all_below=down, n_basins_higher=hi, n_basins_lower=lo, n_basins=nb, higher_in_most_basins=hi > nb / 2,
                         median_basin_difference=md, wilcoxon_p=p, p_below_cut=small, clear_gain=bool(gain), clear_loss=bool(loss),
                         reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=ons(T, k), warned_without=ons(Cc, k), gain=ons(T, k) - ons(Cc, k), seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                           false_with=ons(T, k, "far"), false_without=ons(Cc, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down,
                           clear_gain=bool(o_up), clear_loss=bool(o_down), reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"))


MY_RULES = {}


def b_gain_rule():
    first = {("wqh", "control_against_rain_table_alone"): "wqh_older", ("wh", "control_against_rain_table_alone"): "wh_older",
             ("wqh", "matched_window_to_09_against_rain_table_alone"): "wqh_to09", ("wqh", "utc_day_against_rain_table_alone"): "wqh_utc",
             ("wqh", "utc_day_against_matched_window_to_09"): "wqh_utc"}          # the pairs whose first set is not the set they are filed under (as the script builds them)
    for name, keys in R["gain_rule"].items():
        for key, by in keys.items():
            T = first.get((name, key), name)
            for lk_, entry in by.items():
                Cc = entry["against"][len("lstm_"):]
                mine = rule(T, Cc, int(lk_[1:]))
                MY_RULES[f"{name}|{key}|{lk_}"] = mine
                walk(f"gain_rule.{name}.{key}.{lk_}", mine, entry)
    expected = {"wqh": {"against_control", "against_rain_table_alone", "against_nldas_rain", "control_against_rain_table_alone", "against_matched_window_to_09", "against_utc_day",
                        "matched_window_to_09_against_rain_table_alone", "utc_day_against_rain_table_alone", "utc_day_against_matched_window_to_09"},
                "wh": {"against_control", "against_rain_table_alone", "against_nldas_rain", "control_against_rain_table_alone"},
                "wqnh": {"against_rain_to_midnight", "against_nldas_rain", "against_rain_table_alone"}}
    C.add("gain_rule pairs present", {k: sorted(v) for k, v in R["gain_rule"].items()}, {k: sorted(v) for k, v in expected.items()})


def flags(n, k):
    return np.array(E(n)[f"l{k}"]["onset_flags"], bool)


def overlap(A, B_, k=1):
    fa, fc = flags(A, k), flags(B_, k)
    both, oa, oc = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    lab = lambda n: n if n.startswith("base_") else f"lstm_{n}"
    return dict(first=lab(A), second=lab(B_), lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=oa,
                warned_only_by_second=oc, warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + oa + oc,
                sign_test_p=float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))


def pair_reading(T, Cc):
    out = {}
    for k in LEADS:
        r = rule(T, Cc, k)
        tT, tC, sT, sC = E(T)[f"l{k}"]["onset"], E(Cc)[f"l{k}"]["onset"], RNG[T][f"l{k}"], RNG[Cc][f"l{k}"]
        out[f"l{k}"] = dict(with_set=f"lstm_{T}", without_set=f"lstm_{Cc}",
                            onsets=dict(warned_with=tT["pod"], warned_without=tC["pod"], difference_points=100 * (tT["pod"] - tC["pod"]), hits_with=tT["hits"], hits_without=tC["hits"],
                                        onsets_in_the_test_years=tT["hits"] + tT["misses"], hits_difference=tT["hits"] - tC["hits"], seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                                        every_seed_above_every_seed=r["onset"]["single_seeds_all_above"], every_seed_below_every_seed=r["onset"]["single_seeds_all_below"],
                                        difference_by_seed_points=[100 * (x - y) for x, y in zip(sT["by_seed"]["onset_pod"], sC["by_seed"]["onset_pod"])],
                                        false_warnings_with=tT["far"], false_warnings_without=tC["far"], reading=r["onset"]["reading"]),
                            nse=dict(median_with=r["nse"]["median_with"], median_without=r["nse"]["median_without"], gain=r["nse"]["gain"], seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                                     every_seed_above_every_seed=r["nse"]["single_seeds_all_above"], difference_by_seed=[x - y for x, y in zip(sT["by_seed"]["nse_median"], sC["by_seed"]["nse_median"])],
                                     n_basins_higher=r["nse"]["n_basins_higher"], n_basins=nb, median_basin_difference=r["nse"]["median_basin_difference"], wilcoxon_p=r["nse"]["wilcoxon_p"], reading=r["nse"]["reading"]))
    return out


READ_PAIRS = {"value_of_hours_09_to_24": ("wqh", "wqh_to09"), "value_of_evening_hours_beyond_utc_day": ("wqh", "wqh_utc"), "value_of_hours_09_to_utc_day_end": ("wqh_utc", "wqh_to09"),
              "second_rain_product_alone": ("wqh_to09", "wq"), "nldas_rain_against_era5_over_the_utc_day": ("wqn", "wqh_utc")}
MY_READ = {}


def b_readings():
    for key, (T, Cc) in READ_PAIRS.items():
        MY_READ[key] = pair_reading(T, Cc)
        walk(f"{key}.by_lead", MY_READ[key], R[key]["by_lead"])
    MY_READ["overlap wqh wqh_to09"] = overlap("wqh", "wqh_to09")
    MY_READ["overlap wqh wqh_utc"] = overlap("wqh", "wqh_utc")
    walk("value_of_hours_09_to_24.overlap_of_warned_onsets", MY_READ["overlap wqh wqh_to09"], R["value_of_hours_09_to_24"]["overlap_of_warned_onsets"])
    walk("value_of_evening_hours_beyond_utc_day.overlap_of_warned_onsets", MY_READ["overlap wqh wqh_utc"], R["value_of_evening_hours_beyond_utc_day"]["overlap_of_warned_onsets"])
    bp = R["both_products"]
    for key, Cc in (("against_rain_to_midnight", "wqh"), ("against_nldas_rain", "wqn"), ("against_rain_table_alone", "wq")):
        MY_READ[f"both_products.{key}"] = pair_reading("wqnh", Cc)
        walk(f"both_products.{key}", MY_READ[f"both_products.{key}"], bp[key])
    for key, (A, B_) in (("rain_to_midnight_against_nldas_rain", ("wqh", "wqn")), ("both_against_rain_to_midnight", ("wqnh", "wqh")), ("both_against_nldas_rain", ("wqnh", "wqn"))):
        MY_READ[f"both_products.overlap.{key}"] = overlap(A, B_)
        walk(f"both_products.overlap_of_warned_onsets.{key}", MY_READ[f"both_products.overlap.{key}"], bp["overlap_of_warned_onsets"][key])


MY_E1 = {}


def b_e1():
    fam = {"wqh": ("wqh_older", "wq", "wqn"), "wh": ("wh_older", "w", "wn")}
    for name, (ctrl, base, nl) in fam.items():
        r1 = {x: RNG[x]["l1"]["onset_pod"] for x in (name, ctrl, base, nl)}
        parts = dict(every_seed_above_every_seed_with_nldas_rain=above(r1[name], r1[nl]), every_seed_above_every_seed_of_the_control=above(r1[name], r1[ctrl]),
                     every_seed_above_every_seed_with_the_rain_table_alone=above(r1[name], r1[base]), control_flat=not above(r1[ctrl], r1[base]))
        mine = dict(lead=1, onset_warned=dict(with_rain_to_midnight=ons(name, 1), nldas=ons(nl, 1), control=ons(ctrl, 1), rain_table_alone=ons(base, 1)), seed_ranges=r1, **parts, met=bool(all(parts.values())))
        if name == "wqh":
            for c, key in (("wqh_to09", "to09"), ("wqh_utc", "utc")):
                rc = RNG[c]["l1"]["onset_pod"]
                mine.update({f"control_{key}_onset_warned": ons(c, 1), f"control_{key}_seed_range": rc, f"control_{key}_flat": not above(rc, r1[base]),
                             f"every_seed_above_every_seed_of_the_control_{key}": above(r1[name], rc)})
        MY_E1[name] = mine
        walk(f"e1_success.{name}", mine, R["e1_success"][name])
        C.add(f"e1_success.{name} keys", sorted(R["e1_success"][name]), sorted(list(mine) + ["rule"]))


def versus(T, Cc, base, ceil_, k):
    return dict(set=f"lstm_{Cc}", nse_median=med(Cc, k), nse_seed_range=RNG[Cc][f"l{k}"]["nse_median"], nse_gain_over_it=med(T, k) - med(Cc, k), basins_better_than_it=int((nse_arr(T, k) > nse_arr(Cc, k)).sum()),
                median_basin_gain_over_it=float(np.median(nse_arr(T, k) - nse_arr(Cc, k))), share_of_nse_gap_closed_beyond_it=div(med(T, k) - med(Cc, k), med(ceil_, k) - med(base, k)),
                onset_warned=ons(Cc, k), onset_seed_range=RNG[Cc][f"l{k}"]["onset_pod"], onset_false=ons(Cc, k, "far"), onset_gain_over_it=ons(T, k) - ons(Cc, k),
                onset_share_of_gap_closed_beyond_it=div(ons(T, k) - ons(Cc, k), ons(ceil_, k) - ons(base, k)), own_nse_gain_over_base=med(Cc, k) - med(base, k),
                own_share_of_nse_gap_closed=div(med(Cc, k) - med(base, k), med(ceil_, k) - med(base, k)), own_onset_gain_over_base=ons(Cc, k) - ons(base, k),
                own_onset_share_of_gap_closed=div(ons(Cc, k) - ons(base, k), ons(ceil_, k) - ons(base, k)))


def b_closure():
    for name, (base, ceil_, ctrl, nl) in {"wqh": ("wq", "wqf", "wqh_older", "wqn"), "wh": ("w", "wf", "wh_older", "wn")}.items():
        for j, k in enumerate(LEADS):
            b, c, n, o = med(base, k), med(ceil_, k), med(name, k), med(ctrl, k)
            ob, oc, on, oo = ons(base, k), ons(ceil_, k), ons(name, k), ons(ctrl, k)
            r = dict(lead=k, nse_base=b, nse_with_rain_to_midnight=n, nse_perfect_forecast=c, gap=c - b, gain=n - b, share_of_gap_closed=div(n - b, c - b), nse_control_one_day_older=o, gain_control=o - b,
                     share_of_gap_closed_beyond_control=div(n - o, c - b), basins_better=int((nse_arr(name, k) > nse_arr(base, k)).sum()), basins_better_control=int((nse_arr(ctrl, k) > nse_arr(base, k)).sum()),
                     basins_better_than_control=int((nse_arr(name, k) > nse_arr(ctrl, k)).sum()), median_basin_gain=float(np.median(nse_arr(name, k) - nse_arr(base, k))),
                     median_basin_gain_over_control=float(np.median(nse_arr(name, k) - nse_arr(ctrl, k))), onset_warned_base=ob, onset_warned_with_rain_to_midnight=on, onset_warned_control=oo,
                     onset_warned_perfect_forecast=oc, onset_false_base=ons(base, k, "far"), onset_false_with_rain_to_midnight=ons(name, k, "far"), onset_false_control=ons(ctrl, k, "far"),
                     onset_false_perfect_forecast=ons(ceil_, k, "far"), onset_share_of_gap_closed=div(on - ob, oc - ob), onset_share_of_gap_closed_beyond_control=div(on - oo, oc - ob),
                     seed_range_nse_with=RNG[name][f"l{k}"]["nse_median"], seed_range_nse_control=RNG[ctrl][f"l{k}"]["nse_median"], seed_range_nse_base=RNG[base][f"l{k}"]["nse_median"],
                     onset_seed_range_with=RNG[name][f"l{k}"]["onset_pod"], onset_seed_range_control=RNG[ctrl][f"l{k}"]["onset_pod"], onset_seed_range_base=RNG[base][f"l{k}"]["onset_pod"],
                     basins_perfect_forecast_below_base=int((nse_arr(ceil_, k) < nse_arr(base, k)).sum()),
                     nse_nldas=med(nl, k), share_of_gap_closed_by_nldas=div(med(nl, k) - b, c - b), onset_warned_nldas=ons(nl, k), onset_false_nldas=ons(nl, k, "far"),
                     onset_share_of_gap_closed_by_nldas=div(ons(nl, k) - ob, oc - ob), onset_seed_range_nldas=RNG[nl][f"l{k}"]["onset_pod"], seed_range_nse_nldas=RNG[nl][f"l{k}"]["nse_median"])
            ctrls = {"one_day_older": ctrl, **({"to09": "wqh_to09", "utc": "wqh_utc"} if name == "wqh" else {})}
            r["controls"] = {lab: versus(name, cc, base, ceil_, k) for lab, cc in ctrls.items()}
            walk(f"closure.{name}.l{k}", r, R["closure"][name]["by_lead"][j])
    for name, refs in {"wqh_to09": {"rain_to_midnight": "wqh"}, "wqh_utc": {"rain_to_midnight": "wqh"}, "wqnh": {"rain_to_midnight": "wqh", "nldas_rain": "wqn"}}.items():
        base, ceil_ = "wq", "wqf"
        for j, k in enumerate(LEADS):
            b, c, n = med(base, k), med(ceil_, k), med(name, k)
            ob, oc, on = ons(base, k), ons(ceil_, k), ons(name, k)
            r = dict(lead=k, nse_base=b, nse_with=n, nse_perfect_forecast=c, gap=c - b, gain=n - b, share_of_gap_closed=div(n - b, c - b), basins_better=int((nse_arr(name, k) > nse_arr(base, k)).sum()),
                     median_basin_gain=float(np.median(nse_arr(name, k) - nse_arr(base, k))), onset_warned_base=ob, onset_warned_with=on, onset_warned_perfect_forecast=oc,
                     onset_false_base=ons(base, k, "far"), onset_false_with=ons(name, k, "far"), onset_false_perfect_forecast=ons(ceil_, k, "far"), onset_share_of_gap_closed=div(on - ob, oc - ob),
                     seed_range_nse_with=RNG[name][f"l{k}"]["nse_median"], seed_range_nse_base=RNG[base][f"l{k}"]["nse_median"], onset_seed_range_with=RNG[name][f"l{k}"]["onset_pod"],
                     onset_seed_range_base=RNG[base][f"l{k}"]["onset_pod"], references={lab: versus(name, cc, base, ceil_, k) for lab, cc in refs.items()})
            walk(f"closure.{name}.l{k}", r, R["closure"][name]["by_lead"][j])


def b_ladder():
    for fam, rungs in R["ladder"].items():
        for rec in rungs["rungs"]:
            n = rec["set"][len("lstm_"):]
            ctrl = rec["control_set"][len("lstm_"):] if rec["control_set"] else None
            for k in LEADS:
                e = dict(nse_median=med(n, k), nse_seed_range=RNG[n][f"l{k}"]["nse_median"], onset_warned=ons(n, k), onset_seed_range=RNG[n][f"l{k}"]["onset_pod"], onset_false=ons(n, k, "far"),
                         n_seeds=RNG[n][f"l{k}"]["n_seeds"])
                if ctrl:
                    e.update(control_nse_median=med(ctrl, k), control_nse_seed_range=RNG[ctrl][f"l{k}"]["nse_median"], control_onset_warned=ons(ctrl, k),
                             control_onset_seed_range=RNG[ctrl][f"l{k}"]["onset_pod"], control_onset_false=ons(ctrl, k, "far"))
                walk(f"ladder.{fam}.{n}.l{k}", e, rec["by_lead"][f"l{k}"])
            for cset, by in rec.get("matched_controls", {}).items():
                c = cset[len("lstm_"):]
                for k in LEADS:
                    walk(f"ladder.{fam}.{n}.matched.{c}.l{k}", dict(nse_median=med(c, k), nse_seed_range=RNG[c][f"l{k}"]["nse_median"], onset_warned=ons(c, k), onset_seed_range=RNG[c][f"l{k}"]["onset_pod"],
                                                                     onset_false=ons(c, k, "far"), n_seeds=RNG[c][f"l{k}"]["n_seeds"]), by[f"l{k}"])


MY_YARD = {}


def b_yardstick():
    for T, n in (("wqh", "base_linreg_wqh"), ("wqh", "base_rf_wqh"), ("wqh", "base_persistence"), ("wq", "base_linreg_wq"), ("wq", "base_rf_wq")):
        for k in LEADS:
            a, b = nse_arr(T, k), nse_arr(n, k)
            sm = RNG[T][f"l{k}"]["by_seed"]["nse_median"]
            bm = med(n, k)
            hi, lo, md, p = paired(a, b)
            c = dict(lstm_nse_median=med(T, k), baseline_nse_median=bm, difference_of_medians=med(T, k) - bm, n_basins_lstm_higher=hi, n_basins=nb, median_nse_difference=md, wilcoxon_p=p,
                     lstm_seed_median_min=min(sm), lstm_seed_median_max=max(sm), baseline_against_seed_spread="below every seed" if bm < min(sm) else "above every seed" if bm > max(sm) else "inside the seed spread",
                     n_basins_lstm_higher_by_seed=[int((np.array(SC[f"lstm_{T}_s{s}"][f"l{k}"]["nse"]) > b).sum()) for s in SEEDS[T]],
                     lstm_onset_pod=ons(T, k), baseline_onset_pod=ons(n, k), lstm_onset_far=ons(T, k, "far"), baseline_onset_far=ons(n, k, "far"), lstm_onset_pod_seed_range=RNG[T][f"l{k}"]["onset_pod"],
                     lstm_csi=E(T)[f"l{k}"]["all"]["csi"], baseline_csi=E(n)[f"l{k}"]["all"]["csi"])
            most, few = hi > nb / 2, hi < nb / 2
            small, below = p < P_CUT, c["baseline_against_seed_spread"] == "below every seed"
            c["reading"] = ("lstm" if below else "lstm_avg") if (most and small) else "simple" if (few and small) else ("lstm_med" if most and below else "tie")
            sp = RNG[T][f"l{k}"]["by_seed"]["onset_pod"]
            ov = overlap(T, n, k)
            pl, pb = np.array(E(T)[f"l{k}"]["onset_pod_by_basin"], float), np.array(E(n)[f"l{k}"]["onset_pod_by_basin"], float)
            ok = np.isfinite(pl) & np.isfinite(pb)
            h2, _, md2, p2 = paired(pl[ok], pb[ok])
            plain = dict(lstm_warned=ons(T, k), simple_warned=ons(n, k), lstm_hits=E(T)[f"l{k}"]["onset"]["hits"], simple_hits=E(n)[f"l{k}"]["onset"]["hits"],
                         onsets=E(T)[f"l{k}"]["onset"]["hits"] + E(T)[f"l{k}"]["onset"]["misses"], lstm_seed_range=RNG[T][f"l{k}"]["onset_pod"], n_lstm_trainings=len(sp),
                         lstm_trainings_above_the_simple_model=int(sum(x > ons(n, k) for x in sp)), every_lstm_training_above=bool(min(sp) > ons(n, k)), every_lstm_training_below=bool(max(sp) < ons(n, k)),
                         paired_onsets=dict(warned_by_both=ov["warned_by_both"], only_the_lstm=ov["warned_only_by_first"], only_the_simple_model=ov["warned_only_by_second"], sign_test_p=ov["sign_test_p"]),
                         basins=dict(n_basins_lstm_higher=h2, n_basins_compared=int(ok.sum()), median_difference_points=100 * md2, wilcoxon_p=p2))
            small2 = ov["sign_test_p"] < P_CUT
            on_read = "lstm" if plain["every_lstm_training_above"] and small2 else "simple" if plain["every_lstm_training_below"] and small2 else "tie"
            c["onsets"] = dict(plain_level=plain, reading={"lstm": "the LSTM warns of more onsets", "simple": "the simple model warns of more onsets", "tie": "no clear difference in the onsets"}[on_read])
            MY_YARD[f"lstm_{T}__{n} l{k}"] = c
            walk(f"yardstick.lstm_against_simple.lstm_{T}__{n}.l{k}", c, R["yardstick"]["lstm_against_simple"][f"lstm_{T}__{n}"][f"l{k}"])
    for lab, (W, Wo) in {"lstm": ("wqh", "wq"), "ridge regression": ("base_linreg_wqh", "base_linreg_wq"), "random forest": ("base_rf_wqh", "base_rf_wq")}.items():
        for k in LEADS:
            hi, lo, md, p = paired(nse_arr(W, k), nse_arr(Wo, k))
            tW, tO = E(W)[f"l{k}"]["onset"], E(Wo)[f"l{k}"]["onset"]
            m = dict(nse_median_with=med(W, k), nse_median_without=med(Wo, k), nse_gain=med(W, k) - med(Wo, k), n_basins_higher=hi, n_basins=nb, median_basin_difference=md, wilcoxon_p=p,
                     onset_warned_with=tW["pod"], onset_warned_without=tO["pod"], onset_gain_points=100 * (tW["pod"] - tO["pod"]), onset_hits_with=tW["hits"], onset_hits_without=tO["hits"],
                     onsets_in_the_test_years=tW["hits"] + tW["misses"], onset_false_with=tW["far"], onset_false_without=tO["far"])
            MY_YARD[f"new_input {lab} l{k}"] = m
            walk(f"yardstick.new_input_given_to_each_model.{lab}.l{k}", m, R["yardstick"]["new_input_given_to_each_model"][lab]["by_lead"][f"l{k}"])


for name, fn in (("summary_test and the simple models' test table", b_summary), ("per_basin", b_per_basin), ("seed_ranges, models.n_seeds_scored, per_seed, training rows", b_ranges),
                 ("gain_rule (every pair, every lead)", b_gain_rule), ("readings (value_of_*, second_rain_product_alone, nldas vs utc, both_products, paired onsets)", b_readings),
                 ("e1_success", b_e1), ("closure", b_closure), ("ladder", b_ladder), ("yardstick (lstm_against_simple, new_input_given_to_each_model)", b_yardstick)):
    block(name, fn)


def b_hashes():
    for n, v in M["seed_average_check"].items():
        C.add(f"seed average lstm_{n} equals the mean of its {v['n_seeds']} seed files (max abs diff)", 0.0, v["max_abs_diff"], tol=0.0)
    for p, h in M["sha256"].items():
        if p in R["integrity"]["sha256_of_saved_files"]:
            C.add(f"integrity sha256 {p}", R["integrity"]["sha256_of_saved_files"][p], h)


block("seed averages and integrity hashes", b_hashes)
S = C.summary()
jsave("compare.json", dict(summary=S, by_block=BLOCK, rows=C.rows, my_rules=MY_RULES, my_readings=MY_READ, my_e1=MY_E1, my_yardstick=MY_YARD, ranges=RNG))
print(json.dumps(BLOCK, indent=1))
print(S["entries"], S["matching"], S["not_matching"][:40])
