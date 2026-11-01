"""w5: the logistic stacker and the leakage check.
(a) My own refit: the four features built from my own rows (log of the seed-mean forecast over Q95, log of the issue-day flow over Q95, the spread
    and the max over the seed files of the log forecast ratio; clipped at 1e-3), standardised with the mean and spread of the FIRST validation half,
    a logistic regression (scikit-learn, C=1, lbfgs) fitted on that half, the probability cut chosen on the SECOND half by the far_cap rule over the
    grid 0.02..0.98, the test years scored once; against the record's cut, coefficients, n_fit, n_choose and test tables (main-split sets).
(b) The made-up-values test on the code under review: the test-years forecasts AND flows of the seed-mean set and of its seed files are replaced by
    random numbers; warning_policy.choose_trigger (through warning_policy.arrays) and warning_policy.stack are run again; the chosen triggers, the
    stacker's coefficients and cut must not change, and the perturbed test scores must. Also the same test on my own choose().
(c) The periods: the halves lie inside the validation years, and the validation and test periods do not overlap.
Output: OUT/w5_stacker.json."""
from w_common import *
import flood as F
import warning_policy as W
from sklearn.linear_model import LogisticRegression

R = record()
out = dict(refit={}, mismatches=[], n_compared=0, leakage={}, periods={})


def compare(path, mine, theirs, kind="share", tol=1e-9):
    out["n_compared"] += 1
    d = diff(mine, theirs)
    if d is None or abs(d) > (0 if kind == "count" else tol):
        out["mismatches"].append(dict(where=path, mine=mine, file=theirs))


def features(S, seeds, thr, lead, period):
    Rm = rows(S, thr, lead, period, require_before_finite=True)
    lr = np.log(np.clip(Rm["f"] / Rm["t"], 1e-3, None))
    lq = np.log(np.clip(Rm["q_rel"], 1e-3, None))
    if seeds:
        L = np.stack([np.log(np.clip((lambda Q: Q["f"] / Q["t"])(rows(Q, thr, lead, period, require_before_finite=True)), 1e-3, None)) for Q in seeds])
        assert L.shape[1] == len(lr)
        sp_, mx = L.std(0), L.max(0)
    else:
        sp_, mx = np.zeros(len(lr)), lr
    return Rm, np.stack([lr, lq, sp_, mx], 1)


def far_cap_cut(Rm, p, cuts):
    T = {c: table(Rm["hi"][Rm["low"]], (p > c)[Rm["low"]]) for c in cuts}
    nz = lambda x, bad: bad if x is None or (isinstance(x, float) and math.isnan(x)) else x
    ok = [c for c in cuts if nz(T[c]["far"], 9.0) <= FAR_CAP]
    if ok:
        return max(ok, key=lambda c: (nz(T[c]["pod"], -1.0), nz(T[c]["csi"], -1.0))), True, T
    return min(cuts, key=lambda c: (nz(T[c]["far"], 9.0), -c)), False, T


cuts = [round(0.02 + 0.01 * i, 2) for i in range(97)]
for name, arm in SETS.items():
    if arm != "main":
        continue
    S, sp = Preds(name), SPLITS[arm]
    thr = own_thresholds(S, sp)
    seeds = [Preds(s) for s in seeds_of(name)]
    srec = {}
    for k in LEADS:
        R1, X1 = features(S, seeds, thr, k, HALVES[arm][0])
        R2, X2 = features(S, seeds, thr, k, HALVES[arm][1])
        Rt, Xt = features(S, seeds, thr, k, sp["test"])
        mu, sd = X1.mean(0), X1.std(0)
        sd[sd == 0] = 1.0                                   # StandardScaler leaves a constant column at 0 (scale 1)
        z = lambda X: (X - mu) / sd
        model = LogisticRegression(C=1.0, max_iter=2000).fit(z(X1), R1["hi"])
        p2 = model.predict_proba(z(X2))[:, 1]
        cut, reached, T2 = far_cap_cut(R2, p2, cuts)
        pt = model.predict_proba(z(Xt))[:, 1]
        test = dict(all=table(Rt["hi"], pt > cut), onset=table(Rt["hi"][Rt["low"]], (pt > cut)[Rt["low"]]))
        fr = R["sets"][name]["leads"][lk(k)]["rules"].get("calibrated_stack", {})
        mine = dict(cut=float(cut), reached=reached, coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]), n_fit=int(len(X1)), n_choose=int(len(X2)), n_test=int(len(Xt)),
                    validation_second_half_onset=T2[cut], test=test, verdict_onsets=verdict(test["onset"]), file=dict(cut=fr.get("cut"), reached=fr.get("reached"), coefficients=fr.get("standardized_coefficients"), n_fit=fr.get("n_fit"), n_choose=fr.get("n_choose")))
        if "test" in fr:
            compare(f"{name}.{lk(k)}.stack.cut", cut, fr["cut"], "count")
            compare(f"{name}.{lk(k)}.stack.n_fit", mine["n_fit"], fr["n_fit"], "count")
            compare(f"{name}.{lk(k)}.stack.n_choose", mine["n_choose"], fr["n_choose"], "count")
            for i, c in enumerate(fr["standardized_coefficients"]):
                compare(f"{name}.{lk(k)}.stack.coef[{i}]", mine["coefficients"][i], c, "share", tol=1e-4)
            for tab in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "correct_neg", "pod", "far", "csi"):
                    compare(f"{name}.{lk(k)}.stack.test.{tab}.{c}", test[tab][c], fr["test"][tab][c], "count" if c in ("hits", "misses", "false_alarms", "correct_neg") else "share")
            for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                compare(f"{name}.{lk(k)}.stack.validation_second_half.onset.{c}", T2[cut][c], fr["validation_second_half"]["onset"][c], "count" if c in ("hits", "misses", "false_alarms") else "share")
            out["n_compared"] += 1
            if mine["verdict_onsets"] != fr.get("verdict_onsets"):
                out["mismatches"].append(dict(where=f"{name}.{lk(k)}.stack.verdict_onsets", mine=mine["verdict_onsets"], file=fr.get("verdict_onsets")))
        else:
            mine["file"] = fr
        srec[lk(k)] = mine
    out["refit"][name] = srec
    m1 = srec["l1"]
    print(f"{name:14s} l1 stacker: cut {m1['cut']:.2f} (file {m1['file']['cut']}) warned {pc(m1['test']['onset']['pod'])} false {pc(m1['test']['onset']['far'])} | coefs {np.round(m1['coefficients'], 4).tolist()} (file {np.round(m1['file']['coefficients'] or [], 4).tolist()})", flush=True)

# (c) the periods
for arm in SPLITS:
    v0, v1 = (pd.Timestamp(x) for x in SPLITS[arm]["val"])
    t0, t1 = (pd.Timestamp(x) for x in SPLITS[arm]["test"])
    h = HALVES[arm]
    out["periods"][arm] = dict(halves_inside_validation=all(v0 <= pd.Timestamp(a) <= pd.Timestamp(b) <= v1 for a, b in h), halves_adjacent=(pd.Timestamp(h[0][1]) + pd.Timedelta(days=1) == pd.Timestamp(h[1][0])),
                               halves_cover_validation=(pd.Timestamp(h[0][0]) == v0 and pd.Timestamp(h[1][1]) == v1), validation_before_test=v1 < t0, test_starts_right_after_validation=(v1 + pd.Timedelta(days=1) == t0),
                               record_halves=R["design"]["halves_of_the_validation_years"][arm], record_split=R["design"]["splits"][arm])

# (b) the made-up-values test on the code under review (and on mine), for lstm_wqa and lstm_wqaF_sub, all leads
D = F.load_data()
thr_ref = F.thresholds(D)
rng = np.random.default_rng(7)
leak = {}
for name in ("lstm_wqa", "lstm_wqaF_sub"):
    S = Preds(name)
    seeds = [Preds(s) for s in seeds_of(name)]
    thr = own_thresholds(S, SPLIT_MAIN)
    Sp = S.perturbed(SPLIT_MAIN["test"], rng)
    seeds_p = [Q.perturbed(SPLIT_MAIN["test"], rng) for Q in seeds]
    to_frames = lambda X: {g: pd.DataFrame({"obs": X.obs[i], **{lk(k): X.pred[i, :, k - 1] for k in LEADS}}, index=X.dates) for i, g in enumerate(X.basins)}
    P, Pp = to_frames(S), to_frames(Sp)
    Qs, Qsp = [to_frames(Q) for Q in seeds], [to_frames(Q) for Q in seeds_p]
    rec = {}
    for k in LEADS:
        A_val, A_val_p = W.arrays(P, thr_ref, k, "val", F.SPLIT), W.arrays(Pp, thr_ref, k, "val", F.SPLIT)
        A_test, A_test_p = W.arrays(P, thr_ref, k, "test", F.SPLIT), W.arrays(Pp, thr_ref, k, "test", F.SPLIT)
        row = dict(validation_rows_identical=bool(np.array_equal(A_val["ratio"], A_val_p["ratio"]) and np.array_equal(A_val["obs_hi"], A_val_p["obs_hi"]) and np.array_equal(A_val["was_low"], A_val_p["was_low"])),
                   test_rows_changed=bool(not np.array_equal(A_test["ratio"], A_test_p["ratio"])), rules={})
        for rule in RULES:
            c, cp = W.choose_trigger(A_val, rule), W.choose_trigger(A_val_p, rule)
            mine_c, mine_cp = choose(rows(S, thr, k, SPLIT_MAIN["val"]), rule), choose(rows(Sp, thr, k, SPLIT_MAIN["val"]), rule)
            row["rules"][rule] = dict(project_trigger_unchanged=c["trigger"] == cp["trigger"] and c["validation"] == cp["validation"], my_trigger_unchanged=mine_c["trigger"] == mine_cp["trigger"],
                                      project_test_score_changed=W.score_at(A_test, c["trigger"])["onset"] != W.score_at(A_test_p, cp["trigger"])["onset"])
        st, stp = W.stack(P, Qs, thr_ref, k, W.HALVES["main"], F.SPLIT["test"]), W.stack(Pp, Qsp, thr_ref, k, W.HALVES["main"], F.SPLIT["test"])
        row["stack"] = dict(coefficients_unchanged=bool(np.allclose(st["standardized_coefficients"], stp["standardized_coefficients"], atol=0, rtol=0)), cut_unchanged=st["cut"] == stp["cut"],
                            validation_second_half_unchanged=st["validation_second_half"] == stp["validation_second_half"], test_changed=st["test"] != stp["test"], n_fit=st["n_fit"], n_choose=st["n_choose"])
        rec[lk(k)] = row
    leak[name] = rec
out["leakage"] = dict(made_up_values_test=leak,
                      all_choices_unchanged=all(r["rules"][rule]["project_trigger_unchanged"] and r["rules"][rule]["my_trigger_unchanged"] for rec in leak.values() for r in rec.values() for rule in RULES)
                      and all(r["stack"]["coefficients_unchanged"] and r["stack"]["cut_unchanged"] and r["stack"]["validation_second_half_unchanged"] for rec in leak.values() for r in rec.values()),
                      all_test_scores_changed=all(r["rules"][rule]["project_test_score_changed"] for rec in leak.values() for r in rec.values() for rule in RULES) and all(r["stack"]["test_changed"] for rec in leak.values() for r in rec.values()),
                      note="choose_trigger receives only the validation arrays (20_warning_policy.py line 236: W.choose_trigger(A_val, rule)); stack fits on halves[0], chooses the cut on halves[1] and scores the test period once "
                           "(warning_policy.py stack()); thresholds come from the training years of the set's split (flood.thresholds); the GO criteria are read from results/flood_analysis.json, written before this experiment")
jsave("w5_stacker.json", out)
print(f"\nrefit: compared {out['n_compared']} values; mismatches {len(out['mismatches'])}" + (f" (first: {out['mismatches'][:3]})" if out["mismatches"] else ""))
print("periods:", json.dumps(out["periods"]))
print("leakage: choices unchanged with made-up test values:", out["leakage"]["all_choices_unchanged"], "| test scores changed:", out["leakage"]["all_test_scores_changed"])
