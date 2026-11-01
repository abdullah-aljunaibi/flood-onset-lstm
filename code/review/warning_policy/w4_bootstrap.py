"""w4: the block bootstrap re-run with my own code and a different seed. Fixed 30-day blocks of TEST target days from the start of the test period
(the block of a row = days since the test start // 30, the same block for every basin, so a storm's basins move together), drawn with replacement
(as many blocks as there are), 500 draws, seed 20261029; the pooled onset POD and FAR at the record's triggers, 2.5th-97.5th percentiles; for every set,
lead and rule. Against the record's 2000-draw intervals: do they overlap, and how far apart are the endpoints? Also (a) the record's own procedure
repeated with its seed 0 and 2000 draws, to see whether the stored intervals come out again from the stated settings, and (b) a moving-block variant
for lstm_wqa lead 1 (blocks start on any day) as a check that the fixed-block choice does not drive the interval. Output: OUT/w4_bootstrap.json."""
from w_common import *

R = record()
MY_DRAWS, MY_SEED = 500, 20261029
out = dict(my_draws=MY_DRAWS, my_seed=MY_SEED, block_days=BLOCK_DAYS, sets={}, non_overlapping=[], max_endpoint_gap_points=0.0, n_intervals=0, record_procedure_repeated=dict(mismatches=[], n_compared=0))


def per_block(Rt, w, nb):
    """Per block: onsets that were high (hits + misses), hits, false alarms (on the onset rows)."""
    blk = Rt["day"] // BLOCK_DAYS
    on, hi = Rt["low"], Rt["hi"]
    return np.stack([np.bincount(blk[on & hi], minlength=nb), np.bincount(blk[on & hi & w], minlength=nb), np.bincount(blk[on & ~hi & w], minlength=nb)], 1).astype(float)


def interval(ev, cnt):
    S = cnt @ ev
    with np.errstate(invalid="ignore", divide="ignore"):
        pod, far = S[:, 1] / S[:, 0], S[:, 2] / (S[:, 1] + S[:, 2])
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return dict(pod_ci95=ci(pod), far_ci95=ci(far), draws=int(cnt.shape[0]), blocks=int(cnt.shape[1]), draws_without_a_warning=int(np.isnan(far).sum()))


def my_counts(nb, B, seed):
    """My draw: B times, nb block indices with replacement -> how often each block is drawn."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.choice(nb, size=nb, replace=True), minlength=nb) for _ in range(B)]).astype(float)


def record_counts(nb, B, seed=0):
    """The record's stated procedure: rng.integers(0, nb, size=(B, nb)) then bincount per row."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(r, minlength=nb) for r in rng.integers(0, nb, size=(B, nb))]).astype(float)


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


cnt_mine, cnt_rec = {}, {}
for arm, sp in SPLITS.items():
    a, b = (pd.Timestamp(x) for x in sp["test"])
    nb = (b - a).days // BLOCK_DAYS + 1
    cnt_mine[arm], cnt_rec[arm] = my_counts(nb, MY_DRAWS, MY_SEED), record_counts(nb, 2000, 0)

for name, arm in SETS.items():
    S, sp = Preds(name), SPLITS[arm]
    thr = own_thresholds(S, sp)
    srec = {}
    for k in LEADS:
        Rv, Rt = rows(S, thr, k, sp["val"]), rows(S, thr, k, sp["test"])
        nb = cnt_mine[arm].shape[1]
        row = {}
        for rule in RULES:
            t = choose(Rv, rule)["trigger"]
            fr = R["sets"][name]["leads"][lk(k)]["rules"][rule]
            assert close(t, fr["trigger"]), f"{name} {k} {rule}: trigger differs (see w1)"
            ev = per_block(Rt, warn(Rt, t), nb)
            mine, rep = interval(ev, cnt_mine[arm]), interval(ev, cnt_rec[arm])
            theirs = fr["bootstrap_test_onsets"]
            mine_vs = dict(pod_overlap=overlap(mine["pod_ci95"], theirs["pod_ci95"]), far_overlap=(overlap(mine["far_ci95"], theirs["far_ci95"]) if None not in theirs["far_ci95"] else None),
                           pod_endpoint_gap_points=[100 * abs(mine["pod_ci95"][i] - theirs["pod_ci95"][i]) for i in range(2)],
                           far_endpoint_gap_points=([100 * abs(mine["far_ci95"][i] - theirs["far_ci95"][i]) for i in range(2)] if None not in theirs["far_ci95"] else None))
            out["n_intervals"] += 1
            if not mine_vs["pod_overlap"] or mine_vs["far_overlap"] is False:
                out["non_overlapping"].append(dict(set=name, lead=k, rule=rule, mine=mine, file=theirs))
            out["max_endpoint_gap_points"] = max([out["max_endpoint_gap_points"]] + mine_vs["pod_endpoint_gap_points"] + (mine_vs["far_endpoint_gap_points"] or []))
            # the record's procedure repeated: must give the stored interval exactly
            for key in ("pod_ci95", "far_ci95"):
                for i in range(2):
                    out["record_procedure_repeated"]["n_compared"] += 1
                    if not close(rep[key][i], theirs[key][i], 1e-12):
                        out["record_procedure_repeated"]["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.{key}[{i}]", repeated=rep[key][i], file=theirs[key][i]))
            if rep["blocks"] != theirs["blocks"] or rep["draws_without_a_warning"] != theirs["draws_without_a_warning"]:
                out["record_procedure_repeated"]["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.blocks_or_draws_without_a_warning", repeated=(rep["blocks"], rep["draws_without_a_warning"]), file=(theirs["blocks"], theirs["draws_without_a_warning"])))
            row[rule] = dict(trigger=t, mine=mine, file=theirs, record_procedure_repeated=rep, comparison=mine_vs)
        srec[lk(k)] = row
    out["sets"][name] = srec
    r1 = srec["l1"]
    print(f"{name:14s} l1 csi: mine POD {pc(r1['csi']['mine']['pod_ci95'][0])}-{pc(r1['csi']['mine']['pod_ci95'][1])} file {pc(r1['csi']['file']['pod_ci95'][0])}-{pc(r1['csi']['file']['pod_ci95'][1])} | "
          f"far_cap: mine POD {pc(r1['far_cap']['mine']['pod_ci95'][0])}-{pc(r1['far_cap']['mine']['pod_ci95'][1])} file {pc(r1['far_cap']['file']['pod_ci95'][0])}-{pc(r1['far_cap']['file']['pod_ci95'][1])}", flush=True)

# (b) moving blocks for lstm_wqa lead 1: blocks of 30 consecutive days starting on any day of the test period, as many as needed to cover its length
S, sp = Preds("lstm_wqa"), SPLIT_MAIN
thr = own_thresholds(S, sp)
Rv, Rt = rows(S, thr, 1, sp["val"]), rows(S, thr, 1, sp["test"])
n_days = int(Rt["day"].max()) + 1
rng = np.random.default_rng(MY_SEED + 1)
mb = {}
for rule in RULES:
    t = choose(Rv, rule)["trigger"]
    w = warn(Rt, t)
    on, hi = Rt["low"], Rt["hi"]
    day_on, day_hit, day_fa = np.bincount(Rt["day"][on & hi], minlength=n_days), np.bincount(Rt["day"][on & hi & w], minlength=n_days), np.bincount(Rt["day"][on & ~hi & w], minlength=n_days)
    cum = lambda x: np.r_[0, np.cumsum(x)]
    C_on, C_hit, C_fa = cum(day_on), cum(day_hit), cum(day_fa)
    n_blocks = math.ceil(n_days / BLOCK_DAYS)
    pods, fars = [], []
    for _ in range(MY_DRAWS):
        st = rng.integers(0, n_days - BLOCK_DAYS + 1, size=n_blocks)
        en = st + BLOCK_DAYS
        on_, hit_, fa_ = (C_on[en] - C_on[st]).sum(), (C_hit[en] - C_hit[st]).sum(), (C_fa[en] - C_fa[st]).sum()
        pods.append(hit_ / on_ if on_ else np.nan)
        fars.append(fa_ / (hit_ + fa_) if hit_ + fa_ else np.nan)
    mb[rule] = dict(trigger=t, pod_ci95=[float(np.nanpercentile(pods, 2.5)), float(np.nanpercentile(pods, 97.5))], far_ci95=[float(np.nanpercentile(fars, 2.5)), float(np.nanpercentile(fars, 97.5))],
                    blocks_per_draw=n_blocks, file=R["sets"]["lstm_wqa"]["leads"]["l1"]["rules"][rule]["bootstrap_test_onsets"])
out["moving_block_wqa_l1"] = mb
a = out["sets"]["lstm_wqa"]["l1"]
out["outcome_caveat_bootstrap_wqa_l1"] = dict(csi_warned_mine=f"{pc(a['csi']['mine']['pod_ci95'][0])}-{pc(a['csi']['mine']['pod_ci95'][1])}", far_cap_warned_mine=f"{pc(a['far_cap']['mine']['pod_ci95'][0])}-{pc(a['far_cap']['mine']['pod_ci95'][1])}",
                                              csi_warned_file=f"{pc(a['csi']['file']['pod_ci95'][0])}-{pc(a['csi']['file']['pod_ci95'][1])}", far_cap_warned_file=f"{pc(a['far_cap']['file']['pod_ci95'][0])}-{pc(a['far_cap']['file']['pod_ci95'][1])}")
jsave("w4_bootstrap.json", out)
print(f"\n{out['n_intervals']} intervals: non-overlapping {len(out['non_overlapping'])}; max endpoint gap {out['max_endpoint_gap_points']:.2f} points; record's procedure repeated: "
      f"{len(out['record_procedure_repeated']['mismatches'])} mismatches of {out['record_procedure_repeated']['n_compared']}; moving blocks wqa l1: {json.dumps({r: (mb[r]['pod_ci95'], mb[r]['far_ci95']) for r in mb})}")
print("outcome_caveat bootstrap (wqa l1):", out["outcome_caveat_bootstrap_wqa_l1"])
