"""w1: the headline numbers recomputed with my own code. For every stored set (the 8 the brief names + lstm_wqn) and lead 1-3:
the trigger chosen on the VALIDATION years by the 'csi' rule and by the 'far_cap' rule (my own choose()), the validation table at it, the TEST
years scored once at it (pooled onset and all-days tables), the verdicts, the trigger-free average precision; each against the record.
Also: my Q95 thresholds (from the npz obs, training years of the set's split) against flood.thresholds(flood.load_data()); my pooled test
tables against flood.evaluate at the chosen triggers (the project's reference); whether the extra 'issue-day flow finite' condition of
warning_policy.arrays changes any count; the CSI triggers against those stored by script 18 (results/forecast_input.json) where both exist.
Output: OUT/w1_headline.json."""
from w_common import *
import flood as F                                      # the project's reference, used only to compare against

R = record()
D = F.load_data()
ref_thr = {"main": F.thresholds(D), "sat": F.thresholds(D, split=SPLIT_SAT)}
out = dict(thresholds={}, sets={}, mismatches=[], n_compared=0, max_abs_diff_shares=0.0, max_abs_diff_counts=0)
cmp_fields_tab = ("hits", "misses", "false_alarms", "correct_neg", "pod", "far", "csi")


def compare(path, mine, theirs, kind="share"):
    """Record one comparison; a mismatch is a count that differs or a share that differs by more than 1e-9."""
    out["n_compared"] += 1
    d = diff(mine, theirs)
    bad = d is None or abs(d) > (0 if kind == "count" else 1e-9)
    if kind == "count" and d is not None:
        out["max_abs_diff_counts"] = max(out["max_abs_diff_counts"], abs(int(d)))
    elif d is not None:
        out["max_abs_diff_shares"] = max(out["max_abs_diff_shares"], abs(d))
    if bad:
        out["mismatches"].append(dict(where=path, mine=mine, file=theirs))
    return not bad


def compare_table(path, mine, theirs):
    return all(compare(f"{path}.{c}", mine[c], theirs[c], "count" if c in ("hits", "misses", "false_alarms", "correct_neg") else "share") for c in cmp_fields_tab)


# forecast_input.json's validation-chosen onset triggers (script 18), to see whether the CSI rule here chose the same levels
fi_trig = {}
try:
    FI = json.loads((PROJ / "results/forecast_input.json").read_text(encoding="utf-8"))
    V = FI.get("verdicts", {}).get("sets", {})
    for n in V:
        for k in LEADS:
            try:
                fi_trig[(f"lstm_{n}", k)] = float(V[n][lk(k)]["chosen_on_validation_onsets"]["trigger"])
            except Exception:
                pass
except Exception as e:
    out["forecast_input_triggers_note"] = f"not read: {e!r}"

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None

for name, arm in SETS.items():
    S, sp = Preds(name), SPLITS[arm]
    thr = own_thresholds(S, sp)
    out["thresholds"][name] = dict(max_abs_diff_vs_flood_thresholds=float(max(abs(thr[g] - ref_thr[arm][g]) for g in thr)), q95=thr)
    rec = R["sets"][name]
    srec = dict(leads={}, csi_trigger_vs_script18={})
    for k in LEADS:
        Rv, Rt = rows(S, thr, k, sp["val"]), rows(S, thr, k, sp["test"])
        Rt_strict = rows(S, thr, k, sp["test"], require_before_finite=True)
        Rv_strict = rows(S, thr, k, sp["val"], require_before_finite=True)
        frec = rec["leads"][lk(k)]
        on = Rt["low"]
        row = dict(n_test_rows=int(len(on)), n_test_onset_rows=int(on.sum()), n_test_onsets=int(Rt["hi"][on].sum()),
                   rows_dropped_by_the_issue_day_finite_condition=dict(val=int(len(Rv["f"]) - len(Rv_strict["f"])), test=int(len(Rt["f"]) - len(Rt_strict["f"]))), rules={})
        compare(f"{name}.{lk(k)}.n_test_rows", row["n_test_rows"], frec["n_test_rows"], "count")
        compare(f"{name}.{lk(k)}.n_test_onset_rows", row["n_test_onset_rows"], frec["n_test_onset_rows"], "count")
        compare(f"{name}.{lk(k)}.n_test_onsets", row["n_test_onsets"], frec["n_test_onsets"], "count")
        for rule in RULES:
            c = choose(Rv, rule)
            t = c["trigger"]
            test = tables(Rt, t)
            fr = frec["rules"][rule]
            r = dict(trigger=t, reached=c["reached"], candidates_tied=c["candidates_tied"], validation_onset=c["validation"], validation_all=tables(Rv, t)["all"], test=test,
                     verdict_onsets=verdict(test["onset"]), verdict_all_days=verdict(test["all"]),
                     file=dict(trigger=fr["trigger"], reached=fr["reached"], verdict_onsets=fr["verdict_onsets"], verdict_all_days=fr["verdict_all_days"]))
            ok_t = compare(f"{name}.{lk(k)}.{rule}.trigger", t, fr["trigger"], "count")
            if fr["reached"] != c["reached"]:
                out["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.reached", mine=c["reached"], file=fr["reached"]))
            compare_table(f"{name}.{lk(k)}.{rule}.validation.onset", c["validation"], fr["validation"]["onset"])
            compare_table(f"{name}.{lk(k)}.{rule}.validation.all", r["validation_all"], fr["validation"]["all"])
            compare_table(f"{name}.{lk(k)}.{rule}.test.onset", test["onset"], fr["test"]["onset"])
            compare_table(f"{name}.{lk(k)}.{rule}.test.all", test["all"], fr["test"]["all"])
            for v in ("verdict_onsets", "verdict_all_days"):
                out["n_compared"] += 1
                if r[v] != fr[v]:
                    out["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.{v}", mine=r[v], file=fr[v]))
            # the project's reference: flood.evaluate at the same trigger must give the same pooled tables
            ev = F.evaluate(F.load_preds(name), ref_thr[arm], "test", t, sp)["summary"][lk(k)]
            r["flood_evaluate_agrees"] = all(close(ev["onset"][c_], test["onset"][c_]) and close(ev["all"][c_], test["all"][c_]) for c_ in cmp_fields_tab)
            if not r["flood_evaluate_agrees"]:
                out["mismatches"].append(dict(where=f"{name}.{lk(k)}.{rule}.flood_evaluate", mine=test, reference=dict(onset=ev["onset"], all=ev["all"])))
            # the strict rows (issue-day flow finite) give the same tables?
            ts = tables(Rt_strict, t)
            r["strict_rows_same_tables"] = all(close(ts["onset"][c_], test["onset"][c_]) and close(ts["all"][c_], test["all"][c_]) for c_ in cmp_fields_tab)
            row["rules"][rule] = r
        ap = average_precision(Rt["hi"][on], Rt["f"][on] / Rt["t"][on])
        row["average_precision_test_onsets"] = ap
        compare(f"{name}.{lk(k)}.average_precision", ap, frec["trigger_free"]["average_precision_test_onsets"])
        if average_precision_score is not None:
            row["average_precision_sklearn"] = float(average_precision_score(Rt["hi"][on], Rt["f"][on] / Rt["t"][on]))
        if (name, k) in fi_trig:
            srec["csi_trigger_vs_script18"][lk(k)] = dict(here=row["rules"]["csi"]["trigger"], script18=fi_trig[(name, k)], equal=close(row["rules"]["csi"]["trigger"], fi_trig[(name, k)]))
        srec["leads"][lk(k)] = row
    out["sets"][name] = srec
    l1 = srec["leads"]["l1"]
    print(f"{name:14s} l1: onsets {l1['n_test_onsets']:4d} | csi trigger {l1['rules']['csi']['trigger']:.2f} warned {pc(l1['rules']['csi']['test']['onset']['pod'])} false {pc(l1['rules']['csi']['test']['onset']['far'])}"
          f" | far_cap trigger {l1['rules']['far_cap']['trigger']:.2f} warned {pc(l1['rules']['far_cap']['test']['onset']['pod'])} false {pc(l1['rules']['far_cap']['test']['onset']['far'])} reached {l1['rules']['far_cap']['reached']}"
          f" | AP {l1['average_precision_test_onsets']:.3f} | mismatches so far {len(out['mismatches'])}", flush=True)

out["thresholds_max_abs_diff_vs_flood"] = max(v["max_abs_diff_vs_flood_thresholds"] for v in out["thresholds"].values())
out["far_cap_not_reached"] = sorted(f"{n} {lk(k)}" for n in SETS for k in LEADS if not out["sets"][n]["leads"][lk(k)]["rules"]["far_cap"]["reached"])
out["far_cap_not_reached_matches_record"] = out["far_cap_not_reached"] == sorted(R["far_cap_not_reached_on_validation"])
out["ties_in_the_choice"] = [dict(set=n, lead=k, rule=rule, tied=out["sets"][n]["leads"][lk(k)]["rules"][rule]["candidates_tied"]) for n in SETS for k in LEADS for rule in RULES if out["sets"][n]["leads"][lk(k)]["rules"][rule]["candidates_tied"]]
out["csi_trigger_vs_script18"] = {n: out["sets"][n]["csi_trigger_vs_script18"] for n in SETS if out["sets"][n]["csi_trigger_vs_script18"]}
jsave("w1_headline.json", out)
print(f"\ncompared {out['n_compared']} values; mismatches {len(out['mismatches'])}; max |diff| shares {out['max_abs_diff_shares']:.2e}, counts {out['max_abs_diff_counts']}; "
      f"thresholds vs flood.thresholds max |diff| {out['thresholds_max_abs_diff_vs_flood']:.2e}; far_cap not reached: {out['far_cap_not_reached']} (matches record: {out['far_cap_not_reached_matches_record']}); ties: {out['ties_in_the_choice']}")
print("CSI triggers against script 18:", json.dumps(out["csi_trigger_vs_script18"]))
