"""Warning-level policies, a calibrated stacker and lead-window scoring on STORED forecasts (nothing is trained).

Written 2026-10-28, ported from the Flood_Cloud_Model package (handoff/extras/Flood_Cloud_Model/run.py: choose(), event_score(), the
Platt calibration of train()) to the conventions of code/flood.py: a prediction frame is indexed by TARGET day (obs, l1, l2, l3), a warning is
forecast > trigger * Q95, and the onsets of lead k are the target days whose flow k days earlier was at or below Q95 (flood.flood_scores).

  arrays(P, thr, lead, part)              pooled over the basins on the target days of a split part: obs_hi, ratio = forecast / Q95, was_low
  choose_trigger(A_val, rule)             the trigger chosen on the VALIDATION arrays by one rule:
        'csi'          the project's rule so far: best pooled onset CSI (as in scripts 16 and 18)
        'far_cap'      max pooled onset POD subject to pooled onset FAR <= 0.30 (the false share the GO rule allows)
        'persistence'  the Flood_Cloud_Model rule: all-day false alarms <= persistence's on the same part and lead, then max onset hits,
                       then max all-day hits, then fewer false alarms
  score_at(A, trigger)                    the pooled onset and all-day tables at one trigger (the same numbers as flood.evaluate)
  stack(P, seeds, thr, lead, halves)      a logistic stacker: P(flow above Q95 on the target day) from the seed-mean forecast ratio, the seed spread
                                          and today's flow over Q95, fitted on the FIRST half of the validation years; its probability cut is chosen on
                                          the SECOND half with the far_cap rule (the two-half design of the Flood_Cloud_Model package)
  any_lead(P, thr, triggers, part)        window scoring: a target day counts as warned if ANY earlier issue (lead 1, 2 or 3) warned of it;
                                          and how many lead-1 misses were warned two or three days before

Every choice is made on validation; the test years are scored once per rule. Running this file writes
results/diagnostics_2026-10-28/warning_policy_rescore.json: a PRELIMINARY diagnostic, not a reviewed experiment. Use it to decide what to train
(NEXT_EXPERIMENTS_2026-10-28.md); do not quote it in the documents before it has its own review record.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flood as F

GRID = tuple(float(x) for x in np.round(np.arange(0.50, 2.0001, 0.05), 2))      # the triggers tried, multiples of Q95 (the scripts so far chose among such steps)
SAT_SPLIT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}       # 18_forecast_input.py
HALVES = {"main": (("2000-10-01", "2003-03-31"), ("2003-04-01", "2005-09-30")),          # the halves of the validation years used by 18_forecast_input.py
          "sat": (("2008-10-01", "2009-09-30"), ("2009-10-01", "2010-09-30"))}
FAR_CAP = 0.30
OUT = F.RES_OUT / "diagnostics_2026-10-28"
SETS = [("lstm_wq", "main"), ("lstm_wqn", "main"), ("lstm_wqa", "main"), ("lstm_wqaF_sub", "main"), ("lstm_wqaP", "main"), ("lstm_wqf", "main"),
        ("lstm_wqa_s", "sat"), ("lstm_wqaF_s", "sat"), ("lstm_wqaP_s", "sat")]


def arrays(P, thr, lead, part=None, split=F.SPLIT, period=None):
    """Pooled over the basins, on the target days of `part` (or of the explicit (start, end) `period`): obs_hi, ratio = forecast / Q95, was_low,
    q_rel = today's flow (lead days before the target) over Q95."""
    a, b = period or split[part]
    O, R, L, Q = [], [], [], []
    for g, p in P.items():
        before = p.obs.shift(lead).loc[a:b]
        q = p.loc[a:b]
        ok = (q.obs.notna() & q[f"l{lead}"].notna() & before.notna()).values
        O.append((q.obs > thr[g]).values[ok])
        R.append((q[f"l{lead}"] / thr[g]).values[ok])
        L.append((before <= thr[g]).values[ok])
        Q.append((before / thr[g]).values[ok])
    return dict(obs_hi=np.concatenate(O), ratio=np.concatenate(R), was_low=np.concatenate(L), q_rel=np.concatenate(Q))


def tables(A, warn):
    warn = np.asarray(warn, bool)
    return dict(all=F.contingency(A["obs_hi"], warn), onset=F.contingency(A["obs_hi"][A["was_low"]], warn[A["was_low"]]))


def score_at(A, trigger):
    return tables(A, A["ratio"] > trigger)


def persistence_false_alarms(A):
    """Persistence warns exactly when the flow `lead` days earlier was above Q95 (the days that are not was_low)."""
    return int((~A["was_low"] & ~A["obs_hi"]).sum())


def _f(x, bad=-1.0):
    return bad if x is None or (isinstance(x, float) and np.isnan(x)) else x


def choose_trigger(A_val, rule="far_cap", far_cap=FAR_CAP, grid=GRID):
    S = {t: score_at(A_val, t) for t in grid}
    if rule == "csi":
        best = max(grid, key=lambda t: (_f(S[t]["onset"]["csi"]), -abs(t - 1.0)))
        return dict(rule=rule, trigger=best, validation=S[best], reached=True)
    if rule == "far_cap":
        ok = [t for t in grid if _f(S[t]["onset"]["far"], 9) <= far_cap]
        if ok:
            best = max(ok, key=lambda t: (_f(S[t]["onset"]["pod"]), _f(S[t]["onset"]["csi"])))
            return dict(rule=rule, far_cap=far_cap, trigger=best, validation=S[best], reached=True)
        best = min(grid, key=lambda t: (_f(S[t]["onset"]["far"], 9), -t))
        return dict(rule=rule, far_cap=far_cap, trigger=best, validation=S[best], reached=False,
                    note="no trigger of the grid keeps the onset false share at or below the cap on validation; the trigger with the lowest false share is given")
    if rule == "persistence":
        budget = persistence_false_alarms(A_val)
        ok = [t for t in grid if S[t]["all"]["false_alarms"] <= budget]
        if ok:
            best = max(ok, key=lambda t: (S[t]["onset"]["hits"], S[t]["all"]["hits"], -S[t]["all"]["false_alarms"]))
            return dict(rule=rule, false_alarm_budget=budget, trigger=best, validation=S[best], reached=True)
        return dict(rule=rule, false_alarm_budget=budget, trigger=max(grid), validation=S[max(grid)], reached=False)
    raise ValueError(rule)


def _features(P, seeds, thr, lead, period):
    """Rows for the stacker: log forecast ratio (seed mean), spread over seeds (0 without seeds), log of today's flow over Q95."""
    A = arrays(P, thr, lead, period=period)
    cols = [np.log(np.clip(A["ratio"], 1e-3, None)), np.log(np.clip(A["q_rel"], 1e-3, None))]
    if seeds:
        R = np.stack([np.log(np.clip(arrays(S, thr, lead, period=period)["ratio"], 1e-3, None)) for S in seeds])
        assert R.shape[1] == len(A["ratio"]), "the seed sets do not cover the same days as the seed-mean set"
        cols += [R.std(0), R.max(0)]
    else:
        cols += [np.zeros(len(A["ratio"])), cols[0]]
    return A, np.stack(cols, 1)


def stack(P, seeds, thr, lead, halves, test_period, far_cap=FAR_CAP):
    """Fit on the first half of the validation years, choose the probability cut on the second half (far_cap rule on onsets), score the test years once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    A1, X1 = _features(P, seeds, thr, lead, halves[0])
    A2, X2 = _features(P, seeds, thr, lead, halves[1])
    At, Xt = _features(P, seeds, thr, lead, test_period)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X1, A1["obs_hi"])
    p2 = model.predict_proba(X2)[:, 1]
    cuts = np.round(np.arange(0.02, 0.981, 0.01), 2)
    S = {c: tables(A2, p2 > c) for c in cuts}
    ok = [c for c in cuts if _f(S[c]["onset"]["far"], 9) <= far_cap]
    if ok:
        cut, reached = max(ok, key=lambda c: (_f(S[c]["onset"]["pod"]), _f(S[c]["onset"]["csi"]))), True
    else:
        cut, reached = min(cuts, key=lambda c: (_f(S[c]["onset"]["far"], 9), -c)), False
    pt = model.predict_proba(Xt)[:, 1]
    coef = model.named_steps["logisticregression"].coef_[0].tolist()
    return dict(features=["log_forecast_over_q95_seed_mean", "log_today_flow_over_q95", "seed_spread_of_log_ratio", "seed_max_of_log_ratio"], standardized_coefficients=coef,
                fitted_on=list(halves[0]), cut_chosen_on=list(halves[1]), cut=float(cut), reached=reached, validation_second_half=S[cut], test=tables(At, pt > cut),
                n_fit=int(len(X1)), n_choose=int(len(X2)), seeds_used=len(seeds))


def any_lead(P, thr, triggers, part, split=F.SPLIT):
    """Window scoring on the target days of `part` (onsets by the lead-1 rule: flow the day before at or below Q95)."""
    a, b = split[part]
    rows = []
    for g, p in P.items():
        q = p.loc[a:b]
        ok = q.obs.notna() & q.l1.notna() & q.l2.notna() & q.l3.notna()
        rows.append(pd.DataFrame(dict(obs_hi=(q.obs > thr[g])[ok], was_low=(p.obs.shift(1) <= thr[g]).loc[a:b][ok],
                                      **{f"w{k}": (q[f"l{k}"] > triggers[k] * thr[g])[ok] for k in F.LEADS})))
    T = pd.concat(rows)
    on = T[T.was_low]
    early = on.w2 | on.w3
    return dict(triggers={str(k): float(triggers[k]) for k in F.LEADS}, onsets=int(on.obs_hi.sum()),
                lead1_only=F.contingency(on.obs_hi, on.w1), any_lead=F.contingency(on.obs_hi, on.w1 | early), two_or_three_days_before=F.contingency(on.obs_hi, early),
                lead1_misses_warned_two_or_three_days_before=int((on.obs_hi & ~on.w1 & early).sum()), lead1_misses=int((on.obs_hi & ~on.w1).sum()))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def diagnose(sets=SETS, log=print):
    t0 = time.time()
    D = F.load_data()
    thr_of = {"main": F.thresholds(D), "sat": F.thresholds(D, split=SAT_SPLIT)}
    split_of = {"main": F.SPLIT, "sat": SAT_SPLIT}
    out = dict(what="Warning-level rules, a calibrated stacker and lead-window scoring applied to STORED forecasts (nothing trained). Written by code/warning_policy.py on "
                    + time.strftime("%Y-%m-%d") + ".",
               status="PRELIMINARY DIAGNOSTIC: not a reviewed experiment; every choice on validation years only, test years scored once per rule; no seed ranges, no intervals.",
               rules=dict(csi="best pooled onset CSI on validation (the project's rule in scripts 16 and 18)",
                          far_cap=f"max pooled onset POD on validation subject to onset FAR <= {FAR_CAP} (the false share the GO rule allows)",
                          persistence="all-day false alarms on validation <= persistence's false alarms, then max onset hits, then max all-day hits, then fewer false alarms (Flood_Cloud_Model run.py choose())",
                          calibrated_stack="logistic regression on [log forecast/Q95 (seed mean), log today's flow/Q95, seed spread, seed max], fitted on the first half of validation, "
                                           "probability cut chosen on the second half by the far_cap rule, test scored once"),
               trigger_grid=list(GRID), halves=HALVES, splits=dict(main=F.SPLIT, sat=SAT_SPLIT), sets={})
    for name, arm in sets:
        if not F.have_preds(name):
            log(f"{name}: no stored forecasts, skipped")
            continue
        P, thr, split = F.load_preds(name), thr_of[arm], split_of[arm]
        seeds = [F.load_preds(f"{name}_s{i}") for i in range(5) if F.have_preds(f"{name}_s{i}")]      # seeds 3-4 exist for the sets of scripts 15-18 since 30 October 2026
        rec = dict(arm=arm, meta_inputs=F.preds_meta(name).get("inputs"), leads={})
        trig = {"csi": {}, "far_cap": {}}
        for k in F.LEADS:
            A_val, A_test = arrays(P, thr, k, "val", split), arrays(P, thr, k, "test", split)
            r = dict(n_test_onsets=int(A_test["obs_hi"][A_test["was_low"]].sum()), persistence_false_alarms_validation=persistence_false_alarms(A_val), rules={})
            for rule in ("csi", "far_cap", "persistence"):
                c = choose_trigger(A_val, rule)
                r["rules"][rule] = dict(c, test=score_at(A_test, c["trigger"]))
                if rule in trig:
                    trig[rule][k] = c["trigger"]
            try:
                r["rules"]["calibrated_stack"] = stack(P, seeds, thr, k, HALVES[arm], split["test"])
            except Exception as e:                                    # a stacker that cannot be fitted must not stop the rest
                r["rules"]["calibrated_stack"] = dict(error=repr(e))
            rec["leads"][str(k)] = r
            s = {rule: r["rules"][rule]["test"]["onset"] for rule in ("csi", "far_cap", "persistence")}
            st = r["rules"]["calibrated_stack"].get("test", {}).get("onset", {})
            log(f"{name:14s} l{k}: onsets {r['n_test_onsets']:4d} | csi-trigger {trig['csi'][k]:.2f} pod {s['csi']['pod']:.3f} far {s['csi']['far']:.3f}"
                f" | far<=30% trigger {trig['far_cap'][k]:.2f} pod {s['far_cap']['pod']:.3f} far {s['far_cap']['far']:.3f}"
                f" | persistence-budget pod {s['persistence']['pod']:.3f} far {s['persistence']['far']:.3f}"
                + (f" | stack pod {st.get('pod', float('nan')):.3f} far {st.get('far', float('nan')):.3f}" if st else ""))
        rec["any_lead_test"] = {rule: any_lead(P, thr, trig[rule], "test", split) for rule in trig}
        w = rec["any_lead_test"]["csi"]
        log(f"{name:14s} window: lead-1 pod {w['lead1_only']['pod']:.3f} -> any-lead pod {w['any_lead']['pod']:.3f} far {w['any_lead']['far']:.3f}; "
            f"{w['lead1_misses_warned_two_or_three_days_before']} of {w['lead1_misses']} lead-1 misses were warned 2-3 days before")
        out["sets"][name] = rec
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "warning_policy_rescore.json").write_text(json.dumps(clean(out), indent=1))
    log(f"wrote {OUT / 'warning_policy_rescore.json'} ({out['runtime_s']} s)")
    return out


if __name__ == "__main__":
    diagnose()
