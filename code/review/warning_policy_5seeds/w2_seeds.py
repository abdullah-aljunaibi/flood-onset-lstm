"""w2: the single-seed numbers. For every set and its seed files lstm_<set>_s<i>: the seed set's own validation-chosen trigger (both rules), its test
scores at that trigger and at the seed-mean set's trigger (mine), the ranges over seeds; each against the record's per_seed and seed_range. Also:
is the seed-mean set the day-by-day average of its seed files (max |difference|)? And the ranges quoted in outcome_caveat for lstm_wqa lead 1.
Output: OUT/w2_seeds.json."""
from w_common import *

R = record()
out = dict(sets={}, mismatches=[], n_compared=0)
brief = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}


def compare(path, mine, theirs, kind="share"):
    out["n_compared"] += 1
    d = diff(mine, theirs)
    if d is None or abs(d) > (0 if kind == "count" else 1e-9):
        out["mismatches"].append(dict(where=path, mine=mine, file=theirs))


def rng_of(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(x)]
    return [min(v), max(v)] if v else None


for name, arm in SETS.items():
    S, sp = Preds(name), SPLITS[arm]
    thr = own_thresholds(S, sp)
    seeds = {s: Preds(s) for s in seeds_of(name)}
    rec = R["sets"][name]
    # the seed-mean set against the mean of the seed files, day by day (NaN where any is NaN)
    if seeds:
        M = np.nanmean(np.stack([Q.pred for Q in seeds.values()]), axis=0)
        both = np.isfinite(M) & np.isfinite(S.pred)
        mx = float(np.max(np.abs(M[both] - S.pred[both]))) if both.any() else None
        same_nan = bool(np.array_equal(np.isfinite(M), np.isfinite(S.pred)))
        obs_same = all(bool(np.array_equal(Q.obs, S.obs)) for Q in seeds.values())
    else:
        mx, same_nan, obs_same = None, None, None
    srec = dict(n_seeds=len(seeds), seed_mean_minus_average_of_seeds_max_abs=mx, same_nan_pattern=same_nan, obs_identical_in_every_seed_file=obs_same, leads={})
    compare(f"{name}.max_abs_difference_seed_mean_vs_average_of_seed_sets", mx, rec["max_abs_difference_seed_mean_vs_average_of_seed_sets"])
    for k in LEADS:
        Rv, Rt = rows(S, thr, k, sp["val"]), rows(S, thr, k, sp["test"])
        row = {}
        for rule in RULES:
            t = choose(Rv, rule)["trigger"]
            fr = rec["leads"][lk(k)]["rules"][rule]
            assert close(t, fr["trigger"]), f"{name} {k} {rule}: my trigger {t} differs from the file's {fr['trigger']} (see w1)"
            per = {}
            for s, Q in seeds.items():
                tag = s.split("_")[-1]                       # s0, s1, s2
                Qv, Qt = rows(Q, thr, k, sp["val"]), rows(Q, thr, k, sp["test"])
                cs = choose(Qv, rule)
                per[tag] = dict(own_trigger=cs["trigger"], own_reached=cs["reached"], test_at_own_trigger=brief(tables(Qt, cs["trigger"])["onset"]), test_at_set_trigger=brief(tables(Qt, t)["onset"]))
                fp = fr["per_seed"][tag]
                compare(f"{name}.{lk(k)}.{rule}.{tag}.own_trigger", cs["trigger"], fp["own_trigger"], "count")
                if cs["reached"] != fp["own_reached"]:
                    out["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.{tag}.own_reached", mine=cs["reached"], file=fp["own_reached"]))
                for where in ("test_at_own_trigger", "test_at_set_trigger"):
                    for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                        compare(f"{name}.{lk(k)}.{rule}.{tag}.{where}.{c}", per[tag][where][c], fp[where][c], "count" if c in ("hits", "misses", "false_alarms") else "share")
            sr = {where: {m: rng_of(v[where][m] for v in per.values()) for m in ("pod", "far", "csi")} for where in ("test_at_own_trigger", "test_at_set_trigger")}
            for where in sr:
                for m in sr[where]:
                    mine, theirs = sr[where][m], fr["seed_range"][where][m]
                    out["n_compared"] += 1
                    if not ((mine is None and theirs is None) or (mine and theirs and close(mine[0], theirs[0]) and close(mine[1], theirs[1]))):
                        out["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.seed_range.{where}.{m}", mine=mine, file=theirs))
            row[rule] = dict(set_trigger=t, per_seed=per, seed_range=sr)
        srec["leads"][lk(k)] = row
    out["sets"][name] = srec
    r1 = srec["leads"]["l1"]
    print(f"{name:14s} l1 single seeds at the set's trigger: csi level warned {pc(r1['csi']['seed_range']['test_at_set_trigger']['pod'][0])}-{pc(r1['csi']['seed_range']['test_at_set_trigger']['pod'][1])} "
          f"false {pc(r1['csi']['seed_range']['test_at_set_trigger']['far'][0])}-{pc(r1['csi']['seed_range']['test_at_set_trigger']['far'][1])} | far_cap level warned "
          f"{pc(r1['far_cap']['seed_range']['test_at_set_trigger']['pod'][0])}-{pc(r1['far_cap']['seed_range']['test_at_set_trigger']['pod'][1])} false "
          f"{pc(r1['far_cap']['seed_range']['test_at_set_trigger']['far'][0])}-{pc(r1['far_cap']['seed_range']['test_at_set_trigger']['far'][1])} | seed-mean vs mean of seeds max |diff| {srec['seed_mean_minus_average_of_seeds_max_abs']}", flush=True)

# the four ranges quoted in outcome_caveat (lstm_wqa, lead 1, single networks at the set's trigger), as the record prints them
a = out["sets"]["lstm_wqa"]["leads"]["l1"]
q = lambda v: f"{pc(v[0])}-{pc(v[1])}"
out["outcome_caveat_ranges_wqa_l1"] = dict(csi_warned=q(a["csi"]["seed_range"]["test_at_set_trigger"]["pod"]), csi_false=q(a["csi"]["seed_range"]["test_at_set_trigger"]["far"]),
                                           far_cap_warned=q(a["far_cap"]["seed_range"]["test_at_set_trigger"]["pod"]), far_cap_false=q(a["far_cap"]["seed_range"]["test_at_set_trigger"]["far"]))
# whether any single network meets GO at the set's trigger, anywhere (for the 'no single training meets it' clause)
out["single_networks_meeting_go_at_set_trigger"] = [f"{n} {lk(k)} {rule} {tag}" for n in SETS for k in LEADS for rule in RULES for tag, v in out["sets"][n]["leads"][lk(k)][rule]["per_seed"].items()
                                                   if verdict(dict(v["test_at_set_trigger"], correct_neg=0)) == "GO"]
out["single_networks_meeting_go_at_own_trigger"] = [f"{n} {lk(k)} {rule} {tag}" for n in SETS for k in LEADS for rule in RULES for tag, v in out["sets"][n]["leads"][lk(k)][rule]["per_seed"].items()
                                                   if verdict(dict(v["test_at_own_trigger"], correct_neg=0)) == "GO"]
jsave("w2_seeds.json", out)
print(f"\ncompared {out['n_compared']} values; mismatches {len(out['mismatches'])}; outcome_caveat ranges (wqa l1): {out['outcome_caveat_ranges_wqa_l1']}; "
      f"single networks meeting GO at the set's trigger: {out['single_networks_meeting_go_at_set_trigger']}; at their own trigger: {out['single_networks_meeting_go_at_own_trigger']}")
