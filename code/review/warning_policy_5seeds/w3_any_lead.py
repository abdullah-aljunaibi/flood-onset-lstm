"""w3: the any-lead window row with my own implementation. On the TEST target days of each set's split, a target day is kept when obs and all three
forecasts are finite; the onsets are by the lead-1 rule (the flow the day before at or below Q95); a day counts as warned by lead k when the forecast
of lead k exceeds that lead's trigger (the triggers chosen on validation by the rule, mine); 'any lead' = warned by lead 1, 2 or 3; 'early' = by lead 2
or 3; and how many lead-1 misses were warned early. Each table against the record's any_lead_test. Output: OUT/w3_any_lead.json."""
from w_common import *

R = record()
out = dict(sets={}, mismatches=[], n_compared=0)


def compare(path, mine, theirs, kind="share"):
    out["n_compared"] += 1
    d = diff(mine, theirs)
    if d is None or abs(d) > (0 if kind == "count" else 1e-9):
        out["mismatches"].append(dict(where=path, mine=mine, file=theirs))


def any_lead(S, thr, triggers, period):
    a, b = period
    m = mask(S.dates, a, b)
    hi, low, w = [], [], {k: [] for k in LEADS}
    for i, g in enumerate(S.basins):
        o = S.obs[i]
        before1 = np.concatenate([[np.nan], o[:-1]])
        ok = m & np.isfinite(o) & np.isfinite(S.pred[i]).all(axis=1)
        with np.errstate(invalid="ignore"):
            low.append((before1 <= thr[g])[ok])
        hi.append((o > thr[g])[ok])
        for k in LEADS:
            w[k].append((S.pred[i, :, k - 1] > triggers[k] * thr[g])[ok])
    hi, low = np.concatenate(hi), np.concatenate(low)
    w = {k: np.concatenate(v) for k, v in w.items()}
    on = low
    H, W1, W2, W3 = hi[on], w[1][on], w[2][on], w[3][on]
    early = W2 | W3
    return dict(triggers={str(k): float(triggers[k]) for k in LEADS}, onsets=int(H.sum()), n_onset_rows=int(on.sum()), lead1_only=table(H, W1), any_lead=table(H, W1 | early),
                two_or_three_days_before=table(H, early), lead1_misses_warned_two_or_three_days_before=int((H & ~W1 & early).sum()), lead1_misses=int((H & ~W1).sum()),
                lead2_only=table(H, W2), lead3_only=table(H, W3))


for name, arm in SETS.items():
    S, sp = Preds(name), SPLITS[arm]
    thr = own_thresholds(S, sp)
    srec = {}
    for rule in RULES:
        trig = {k: choose(rows(S, thr, k, sp["val"]), rule)["trigger"] for k in LEADS}
        mine = any_lead(S, thr, trig, sp["test"])
        mine["verdict_any_lead"] = verdict(mine["any_lead"])
        fr = R["sets"][name]["any_lead_test"][rule]
        for k in LEADS:
            compare(f"{name}.{rule}.trigger.l{k}", trig[k], float(fr["triggers"][str(k)]), "count")
        compare(f"{name}.{rule}.onsets", mine["onsets"], fr["onsets"], "count")
        compare(f"{name}.{rule}.lead1_misses", mine["lead1_misses"], fr["lead1_misses"], "count")
        compare(f"{name}.{rule}.lead1_misses_warned_two_or_three_days_before", mine["lead1_misses_warned_two_or_three_days_before"], fr["lead1_misses_warned_two_or_three_days_before"], "count")
        for tab in ("lead1_only", "any_lead", "two_or_three_days_before"):
            for c in ("hits", "misses", "false_alarms", "correct_neg", "pod", "far", "csi"):
                compare(f"{name}.{rule}.{tab}.{c}", mine[tab][c], fr[tab][c], "count" if c in ("hits", "misses", "false_alarms", "correct_neg") else "share")
        out["n_compared"] += 1
        if mine["verdict_any_lead"] != fr["verdict_any_lead"]:
            out["mismatches"].append(dict(where=f"{name}.{rule}.verdict_any_lead", mine=mine["verdict_any_lead"], file=fr["verdict_any_lead"]))
        srec[rule] = mine
    out["sets"][name] = srec
    w = srec["csi"]
    print(f"{name:14s} csi triggers {w['triggers']}: lead-1 warned {pc(w['lead1_only']['pod'])} -> any lead {pc(w['any_lead']['pod'])} at {pc(w['any_lead']['far'])} false; "
          f"{w['lead1_misses_warned_two_or_three_days_before']} of {w['lead1_misses']} lead-1 misses warned 2-3 days before", flush=True)

w = out["sets"]["lstm_wqa"]["csi"]
out["outcome_caveat_any_lead_wqa"] = dict(any_lead_warned=pc(w["any_lead"]["pod"]), any_lead_false=pc(w["any_lead"]["far"]), lead1_misses_warned_early=w["lead1_misses_warned_two_or_three_days_before"], lead1_misses=w["lead1_misses"])
jsave("w3_any_lead.json", out)
print(f"\ncompared {out['n_compared']} values; mismatches {len(out['mismatches'])}; outcome_caveat (wqa, csi triggers): {out['outcome_caveat_any_lead_wqa']}")
