"""E4a: the warning operating point on STORED forecasts (nothing is trained). Written 2026-10-28 for the work order NEXT_EXPERIMENTS_2026-10-28.md, section E4a.

The scripts so far chose one warning level per set: the trigger (a multiple of the basin's Q95) with the best pooled onset CSI on the validation years
(scripts 16 and 18). The GO rule asks for at least 70% of the onsets warned AND at most 30% of the warnings false, so a second, honest operating point
belongs beside it: the trigger with the most onsets warned among those whose onset false share stays at or below 30% on the validation years
(warning_policy.choose_trigger, rule 'far_cap'). This script scores every stored LSTM set at both levels, once, on its test years, and says how sure the numbers are.
For every set of warning_policy.SETS (main split: lstm_wq, lstm_wqn, lstm_wqa, lstm_wqaF_sub, lstm_wqaP, lstm_wqf; satellite-era split: lstm_wqa_s, lstm_wqaF_s,
lstm_wqaP_s, comparable only inside that split) and every lead 1-3:
  1. the trigger chosen on the VALIDATION years by the 'csi' rule and by the 'far_cap' rule, the validation POD/FAR at it, and the TEST years scored once at it
     (pooled onset POD, FAR, CSI, hits, false alarms; the all-days table beside it; the GO / CAUTION / NOT YET verdict of results/flood_analysis.json);
     'reached' is False when no trigger of the grid kept the validation false share at or below the cap (the lowest-false trigger is reported then);
  2. the same for every single-seed set lstm_<set>_s<i> that exists (i in range(5)): its own validation-chosen trigger and its test scores, and its test scores
     at the seed-mean set's trigger, so that the ensemble number stands beside the range of the single networks that could be deployed;
  3. a block bootstrap over the test target days (blocks of 30 consecutive days drawn with replacement, seed 0, 2000 draws, the same blocks for every set of a
     split) of the test POD and FAR at both triggers of the seed-mean sets: 95% intervals;
  4. the logistic stacker of warning_policy.stack for the main-split sets (fitted on the first half of the validation years, probability cut chosen on the
     second half by the far_cap rule, test scored once; needs scikit-learn, recorded as not computed without it);
  5. the any-lead window row (a target day counts as warned when the issue of lead 1, 2 or 3 warned of it), at the triggers of each rule;
  6. the trigger-free average precision of each set on the test onsets (the two operating points are two readings of ONE skill, not two skills).
Reads results/preds/lstm_*.npz (the stored forecasts), data/derived/camels12_daymet.pkl through flood.load_data (the Q95 thresholds), results/flood_analysis.json
(the GO rule). Writes results/warning_policy.json and figures/warning_policy.png and prints outcome_full. Rerun from the project folder with the environment that
has torch (flood.py imports it):   python code/20_warning_policy.py [--draws N]      Syntax check without torch:  python -X utf8 -m py_compile code/20_warning_policy.py
(flood and warning_policy are imported inside main only). FLOOD_QUICK=1 writes to results/quick and figures/quick with 200 draws.
"""
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AQUA, AXIS, BLUE, INK, INK2, MUTED, ORANGE, RED, ROOT, SEQ, SURFACE, YELLOW, use_style

SCRIPT, OUT = "20_warning_policy.py", "warning_policy"
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
STATUS = "EXPERIMENT (28-29 Oct 2026): every trigger chosen on the validation years; test years scored once per rule"
RULES = ("csi", "far_cap")
NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
MAX_SEEDS, BLOCK_DAYS, BOOT_SEED, B_DEFAULT = 5, 30, 0, 2000
# fixed wordings, word for word from code/report_sections/s0_summary.py (not imported here: it needs torch through flood)
AORC_WORDING, UPPER = "an analysis made after the fact", "an upper bound that cannot be had in practice"
def _hindsight_wording():
    """'hindsight at most about N points' beside the AORC rain-to-midnight share: N = the loosest bound of the Stage IV route for the networks the stored
    lstm_wqa set averages (results/review_aorc_rain_to_midnight.json, the latest round with a five-seed bound; 30 October 2026: 3.6 points, five networks);
    the review of 7 October 2026 measured 'about 3 points' on the three networks of seeds 0-2, the fallback when no later round holds a bound."""
    import json as _json
    from pathlib import Path as _Path
    try:
        rec = _json.loads((_Path(__file__).resolve().parents[1] / "results" / "review_aorc_rain_to_midnight.json").read_text(encoding="utf-8"))
        rounds = sorted((k for k in rec if k.startswith("round_") and k[6:].isdigit()), key=lambda k: int(k[6:]))
        for k in reversed(rounds):
            pts = ((rec[k] or {}).get("hindsight") or {}).get("five_seed_loosest_bound_points")
            if pts is not None:
                return f"hindsight at most about {round(float(pts))} points"
    except (OSError, ValueError, TypeError):
        pass
    return "hindsight at most about 3 points"


HINDSIGHT_WORDING = _hindsight_wording()          # five-seed bound since 30 October 2026 (registry key fw_hindsight prints the same)
LABEL = {"lstm_wq": "rain table + gauge", "lstm_wqn": "+ NLDAS rain to 17:30", "lstm_wqa": "+ AORC rain to midnight", "lstm_wqaF_sub": "+ GEFS forecast (substituted)",
         "lstm_wqaP": "perfect rain, 3 days", "lstm_wqf": "perfect weather", "lstm_wqa_s": "+ AORC rain to midnight (own split)", "lstm_wqaF_s": "+ GEFS forecast, retrained (own split)",
         "lstm_wqaP_s": "perfect rain, 3 days (own split)"}
UPPER_SETS = {"lstm_wqaP", "lstm_wqf", "lstm_wqaP_s"}       # the sets given rain or weather that cannot be had at the issue time
COLOR = {"lstm_wq": MUTED, "lstm_wqn": YELLOW, "lstm_wqa": BLUE, "lstm_wqaF_sub": AQUA, "lstm_wqaP": ORANGE, "lstm_wqf": RED, "lstm_wqa_s": BLUE, "lstm_wqaF_s": AQUA, "lstm_wqaP_s": ORANGE}
PLACE = {"lstm_wqaP": (7, -9, "left"), "lstm_wqaP_s": (-9, -28, "right"), "lstm_wqf": (7, 6, "left")}      # label offsets (points) and side in the figure where labels would collide
lk = lambda k: f"l{k}"
pc = lambda x: "no value" if x is None or not math.isfinite(x) else f"{100 * x:.0f}%"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)


def number(text, key, op):
    """A number read from the stored criterion text ('pod >= 0.70 and far <= 0.30'), as script 16 reads it."""
    m = re.search(rf"{key} {op} ([0-9.]+)", text)
    assert m, f"cannot read '{key} {op}' in the stored criterion: {text}"
    return float(m.group(1))


def block_of(P, lead, part, split):
    """The 30-day block of every row of warning_policy.arrays(P, thr, lead, part, split), in the same order (basins of P, then target days)."""
    a, b = split[part]
    blk = []
    for g, p in P.items():
        before, q = p.obs.shift(lead).loc[a:b], p.loc[a:b]
        ok = (q.obs.notna() & q[lk(lead)].notna() & before.notna()).values
        blk.append(((q.index[ok] - pd.Timestamp(a)).days // BLOCK_DAYS).astype(int))
    return np.concatenate(blk)


def draws(split, B, seed=BOOT_SEED):
    """How often each block of the split's test period is drawn in each of B resamples with replacement: a [B, blocks] count matrix, one per split, shared by its sets."""
    a, b = (pd.Timestamp(x) for x in split["test"])
    nb = (b - a).days // BLOCK_DAYS + 1
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(r, minlength=nb) for r in rng.integers(0, nb, size=(B, nb))]).astype(float)


def bootstrap(A, blk, warn, cnt):
    """95% block-bootstrap interval of the pooled onset POD and FAR of the warnings `warn` on the rows of A (blocks of target days; cnt from draws())."""
    nb, on, hi = cnt.shape[1], A["was_low"], A["obs_hi"]
    ev = np.stack([np.bincount(blk[on & hi], minlength=nb), np.bincount(blk[on & hi & warn], minlength=nb), np.bincount(blk[on & ~hi & warn], minlength=nb)], 1).astype(float)
    S = cnt @ ev
    with np.errstate(invalid="ignore", divide="ignore"):
        pod, far = S[:, 1] / S[:, 0], S[:, 2] / (S[:, 1] + S[:, 2])
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return dict(pod_ci95=ci(pod), far_ci95=ci(far), draws=int(cnt.shape[0]), blocks=int(nb), draws_without_a_warning=int(np.isnan(far).sum()))


def average_precision(y, s):
    """Average precision of the scores s for the labels y: the sum over the distinct score thresholds of (recall step) x precision, the definition of
    sklearn.metrics.average_precision_score, in numpy only (no trigger: the ranking of the onset rows by forecast / Q95 is scored as a whole)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    o = np.argsort(-s, kind="stable")
    y, s = y[o], s[o]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    rec, prec = tp / y.sum(), tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


brief = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}


def rng_of(vals):
    v = [float(x) for x in vals if x is not None and math.isfinite(x)]
    return [min(v), max(v)] if v else None


def clean(o):
    """Strict JSON: numpy values become Python ones; a number that is not finite (a share whose denominator is zero) becomes None (null)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def draw(R, path, pod_go, far_go):
    """Lead 1, test years: onsets warned against false warnings for every set at the two operating points (circle = CSI rule, square = at most 30% false on validation), the GO box drawn."""
    rows = R["summary"]["l1"]
    seed_words = " or ".join(NUM_WORD.get(n, str(n)) for n in sorted({v["n_seeds"] for v in R["sets"].values()}))      # "three", "five" or "three or five" (seeds 3-4 added to some sets on 30 October 2026)
    ir = R["provenance"]["independent_review"]
    rnd = ir.get("round") if isinstance(ir, dict) and ir.get("covers_this_run") else None          # a later round of the review covers this run (2026-10-30)
    stale = isinstance(ir, dict) and ir.get("covers_this_run") is False          # the review checked other forecast files (seeds 3-4 added, 30 October 2026): one line here, the full status in the JSON
    status_fig = (f"NOT YET REVIEWED for this run: the review of {str(ir.get('who', '')).split(', ')[-1]} checked other forecast files ({len(ir.get('inputs_changed_since_the_run_it_covered') or [])} of these differ or are new); "
                  "numbers preliminary" if stale else
                  f"REVIEWED ({rnd.replace('_', ' ')}, {ir.get('date')}): every number recomputed from results/preds by an independent AI reviewer" if rnd else R["status"])
    review_fig = ("a new independent review is needed (results/warning_policy.json, status)" if stale else f"the review record is results/review_warning_policy.json ({rnd.replace('_', ' ')})" if rnd
                  else R["provenance"]["review_line"])
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.215)
    ax.add_patch(Rectangle((0, pod_go), far_go, 1 - pod_go, facecolor=SEQ[0], edgecolor=AXIS, lw=1, alpha=.55, zorder=0))
    ax.annotate(f"GO: at least {pc(pod_go)} warned, at most {pc(far_go)} false", (far_go / 2, 0.985), ha="center", va="top", fontsize=9, color=INK2)
    for row in rows:
        c, hollow = COLOR.get(row["set"], INK2), row["arm"] == "sat"
        pts = [(row["csi"], "o"), (row["far_cap"], "s")]
        xs, ys = [p["false"] for p, _ in pts], [p["warned"] for p, _ in pts]
        ax.plot(xs, ys, color=c, lw=1.0, alpha=.45, zorder=1)
        for p, m in pts:
            ci = p["bootstrap"]
            if ci and None not in ci["far_ci95"] and None not in ci["pod_ci95"]:
                ax.plot(ci["far_ci95"], [p["warned"]] * 2, color=c, lw=0.9, alpha=.35, zorder=2)
                ax.plot([p["false"]] * 2, ci["pod_ci95"], color=c, lw=0.9, alpha=.35, zorder=2)
            ax.plot([p["false"]], [p["warned"]], ls="none", marker=m, ms=9, markerfacecolor=SURFACE if hollow else c, markeredgecolor=c, markeredgewidth=1.7 if hollow else 1.0, zorder=4)
        if row["stack"] and row["stack"]["warned"] is not None:
            ax.plot([row["stack"]["false"]], [row["stack"]["warned"]], ls="none", marker="^", ms=8, markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=1.4, alpha=.9, zorder=3)
        dx, dy, ha = PLACE.get(row["set"], (7, -2 if hollow else 4, "left"))
        ax.annotate(row["label"], (xs[0], ys[0]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=8.3, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=0, shrinkB=5) if abs(dy) > 15 else None)       # a leader line when the label sits away from its point
    ax.set_xlim(0, 0.86)
    ax.set_ylim(0, 1.0)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("share of the onset warnings that were false (FAR), test years")
    ax.set_ylabel("share of the onsets warned (POD), 1 day ahead, test years")
    lg = lambda marker, label, **kw: Line2D([], [], ls="none", marker=marker, ms=8.5, label=label, **kw)
    h = [lg("o", "warning level with the best onset CSI on the validation years (the project's rule so far)", color=INK2),
         lg("s", "the most onsets warned with at most 30% false on the validation years", color=INK2),
         lg("^", "logistic stacker, cut at 30% false on the second validation half (main split)", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.4),
         lg("o", "hollow: satellite-era split (own scores, comparable only inside that split)", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.7),
         Line2D([], [], color=INK2, lw=0.9, alpha=.6, label="95% block-bootstrap interval (30-day blocks of test days)")]
    fig.legend(handles=h, loc="upper left", ncol=1, frameon=False, fontsize=8.6, bbox_to_anchor=(0.012, 0.955), handlelength=1.3, labelspacing=0.35, handletextpad=0.5)
    fig.suptitle("Where the warning sits: onsets warned against false warnings, 1 day ahead, at two levels chosen on the validation years", fontsize=11, fontweight="bold", color=INK, x=0.012, y=0.985, ha="left")
    fig.text(.012, .008, f"Filled: main split, test years 2005-2014 ({rows[0]['n_onsets']} onsets), seed-mean sets ({seed_words} trainings averaged day by day). A line joins the two levels of one set: moving along it changes\n"
                         f"the trade-off, not the skill. AORC rain to midnight is {AORC_WORDING}; the forecast is the bias-corrected GEFS control member; 'perfect rain' gives the AORC rain of the\n"
                         f"target days as the forecast, 'perfect weather' the whole weather of those days: both are {UPPER}.\n"
                         f"Every trigger (a multiple of Q95) and every stacker cut was chosen on the validation years; the test years were scored once per rule.\n"
                         f"{status_fig}.\n"
                         f"The team's own checks are not yet done; {review_fig}. Written by code/20_warning_policy.py from results/warning_policy.json.",
             fontsize=7.6, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=()):
    import flood as F                                       # imported here so that the module can be imported and compiled without torch
    import torch
    import warning_policy as W
    use_style()
    t0 = time.time()
    B = int(argv[argv.index("--draws") + 1]) if "--draws" in argv else (200 if F.QUICK else B_DEFAULT)
    D = F.load_data()
    crit = json.loads((F.RES / "flood_analysis.json").read_text())["verdict"]["criteria"]
    pod_go, far_go, csi_caution = number(crit["GO"], "pod", ">="), number(crit["GO"], "far", "<="), number(crit["CAUTION"], "csi", ">=")

    def verdict(t):
        if t["hits"] + t["misses"] == 0:
            return "NO EVENTS"
        if t["pod"] >= pod_go and t["far"] <= far_go:        # far is NaN without a warning: then this is False
            return "GO"
        return "CAUTION" if t["csi"] >= csi_caution else "NOT YET"

    ARMS = {"main": dict(split=F.SPLIT, thr=F.thresholds(D), name="main split", test_years="2005-2014"),
            "sat": dict(split=W.SAT_SPLIT, thr=F.thresholds(D, split=W.SAT_SPLIT), name="satellite-era split", test_years="2010-2014")}
    for a in ARMS.values():
        a["cnt"] = draws(a["split"], B)
    sets = list(W.SETS) + ([] if any(n == "lstm_wqn" for n, _ in W.SETS) else [("lstm_wqn", "main")])
    files, skipped, S = {}, [], {}
    try:
        import sklearn
        sk_version = sklearn.__version__
    except ImportError:
        sk_version = None
    for name, arm in sets:
        if not F.have_preds(name):
            skipped.append(name)
            continue
        split, thr, halves = ARMS[arm]["split"], ARMS[arm]["thr"], W.HALVES[arm]
        P = F.load_preds(name)
        seeds = {f"s{i}": F.load_preds(f"{name}_s{i}") for i in range(MAX_SEEDS) if F.have_preds(f"{name}_s{i}")}
        for n_ in [name] + [f"{name}_{s}" for s in seeds]:
            files[f"results/preds/{n_}.npz"] = hashlib.sha256(F._pred_file(n_).read_bytes()).hexdigest()
        E = F.ensemble(list(seeds.values())) if seeds else None
        rec = dict(arm=arm, split=ARMS[arm]["name"], test_years=ARMS[arm]["test_years"], label=LABEL.get(name, name), meta_inputs=F.preds_meta(name).get("inputs"), n_seeds=len(seeds), seed_sets=[f"{name}_{s}" for s in seeds],
                   max_abs_difference_seed_mean_vs_average_of_seed_sets=(float(max(np.nanmax(np.abs(E[g][[lk(k) for k in F.LEADS]].values - P[g][[lk(k) for k in F.LEADS]].values)) for g in P)) if E else None), leads={})
        trig = {rule: {} for rule in RULES}
        for k in F.LEADS:
            A_val, A_test = W.arrays(P, thr, k, "val", split), W.arrays(P, thr, k, "test", split)
            blk = block_of(P, k, "test", split)
            assert len(blk) == len(A_test["ratio"]), f"{name} lead {k}: the block index does not line up with the pooled rows"
            on = A_test["was_low"]
            As = {s: (W.arrays(Q, thr, k, "val", split), W.arrays(Q, thr, k, "test", split)) for s, Q in seeds.items()}
            row = dict(n_test_onsets=int(A_test["obs_hi"][on].sum()), n_test_onset_rows=int(on.sum()), n_test_rows=int(len(on)), rules={})
            for rule in RULES:
                c = W.choose_trigger(A_val, rule)
                t = c["trigger"]
                trig[rule][k] = t
                test = W.score_at(A_test, t)
                per_seed = {}
                for s, (Av, At) in As.items():
                    cs = W.choose_trigger(Av, rule)
                    per_seed[s] = dict(own_trigger=cs["trigger"], own_reached=cs["reached"], test_at_own_trigger=brief(W.score_at(At, cs["trigger"])["onset"]), test_at_set_trigger=brief(W.score_at(At, t)["onset"]))
                sr = {where: {m: rng_of(v[where][m] for v in per_seed.values()) for m in ("pod", "far", "csi")} for where in ("test_at_own_trigger", "test_at_set_trigger")}
                row["rules"][rule] = dict(trigger=t, reached=c["reached"], note=c.get("note"), far_cap=c.get("far_cap"), validation=c["validation"], test=test, verdict_onsets=verdict(test["onset"]), verdict_all_days=verdict(test["all"]),
                                          bootstrap_test_onsets=bootstrap(A_test, blk, A_test["ratio"] > t, ARMS[arm]["cnt"]), seed_range=sr, per_seed=per_seed)
            if arm == "main":
                try:
                    st = W.stack(P, list(seeds.values()), thr, k, halves, split["test"])
                    st["verdict_onsets"] = verdict(st["test"]["onset"])
                    row["rules"]["calibrated_stack"] = st
                except Exception as e:                      # a stacker that cannot be fitted (for example without scikit-learn) must not stop the rest
                    row["rules"]["calibrated_stack"] = dict(not_computed=repr(e))
            row["trigger_free"] = dict(average_precision_test_onsets=average_precision(A_test["obs_hi"][on], A_test["ratio"][on]), share_of_onset_rows_that_are_onsets=float(A_test["obs_hi"][on].mean()),
                                       seed_range=rng_of(average_precision(At["obs_hi"][At["was_low"]], At["ratio"][At["was_low"]]) for _, At in As.values()))
            rec["leads"][lk(k)] = row
        rec["any_lead_test"] = {}
        for rule in RULES:
            w = W.any_lead(P, thr, trig[rule], "test", split)
            w["verdict_any_lead"] = verdict(w["any_lead"])
            rec["any_lead_test"][rule] = w
        S[name] = rec
        r1 = rec["leads"]["l1"]["rules"]
        st = r1.get("calibrated_stack", {}).get("test", {}).get("onset")
        print(f"{name:14s} l1: onsets {rec['leads']['l1']['n_test_onsets']:4d} | csi trigger {r1['csi']['trigger']:.2f} warned {pc(r1['csi']['test']['onset']['pod'])} false {pc(r1['csi']['test']['onset']['far'])}"
              f" | far<=30% trigger {r1['far_cap']['trigger']:.2f} warned {pc(r1['far_cap']['test']['onset']['pod'])} false {pc(r1['far_cap']['test']['onset']['far'])} reached {r1['far_cap']['reached']}"
              + (f" | stack warned {pc(st['pod'])} false {pc(st['far'])}" if st else "") + f" | AP {rec['leads']['l1']['trigger_free']['average_precision_test_onsets']:.3f}", flush=True)

    # ---- summary tables, outcome, file -------------------------------------------------------------------------------------------------------------------
    def point(r):
        return dict(trigger=r["trigger"], reached=r["reached"], warned=r["test"]["onset"]["pod"], false=r["test"]["onset"]["far"], csi=r["test"]["onset"]["csi"], hits=r["test"]["onset"]["hits"], false_alarms=r["test"]["onset"]["false_alarms"],
                    validation_warned=r["validation"]["onset"]["pod"], validation_false=r["validation"]["onset"]["far"], verdict=r["verdict_onsets"], seed_range_warned=r["seed_range"]["test_at_set_trigger"]["pod"],
                    seed_range_false=r["seed_range"]["test_at_set_trigger"]["far"], seed_range_warned_own_trigger=r["seed_range"]["test_at_own_trigger"]["pod"], bootstrap=r["bootstrap_test_onsets"])
    summary = {}
    for k in F.LEADS:
        summary[lk(k)] = []
        for name, rec in S.items():
            row, st = rec["leads"][lk(k)], rec["leads"][lk(k)]["rules"].get("calibrated_stack")
            summary[lk(k)].append(dict(set=name, label=rec["label"], arm=rec["arm"], n_onsets=row["n_test_onsets"], csi=point(row["rules"]["csi"]), far_cap=point(row["rules"]["far_cap"]),
                                       stack=(dict(warned=st["test"]["onset"]["pod"], false=st["test"]["onset"]["far"], cut=st["cut"], reached=st["reached"], verdict=st["verdict_onsets"]) if st and "test" in st else None),
                                       average_precision=row["trigger_free"]["average_precision_test_onsets"],
                                       any_lead={rule: dict(warned=rec["any_lead_test"][rule]["any_lead"]["pod"], false=rec["any_lead_test"][rule]["any_lead"]["far"]) for rule in RULES}))
    go = [f"{n} {lk(k)} {rule}" for n, rec in S.items() for k in F.LEADS for rule in rec["leads"][lk(k)]["rules"] if rec["leads"][lk(k)]["rules"][rule].get("verdict_onsets") == "GO"]
    go_sets = {}                                             # which sets meet GO at which leads (either rule), and whether every single training does too at the set's trigger
    for n, rec in S.items():
        for k in F.LEADS:
            for rule in RULES:
                r = rec["leads"][lk(k)]["rules"][rule]
                if r["verdict_onsets"] == "GO":
                    v = go_sets.setdefault(n, dict(leads=[], single_networks_meet_it=True))
                    v["leads"] = sorted(set(v["leads"]) | {k})
                    v["single_networks_meet_it"] &= all(s["test_at_set_trigger"]["pod"] >= pod_go and s["test_at_set_trigger"]["far"] <= far_go for s in r["per_seed"].values())
    days = lambda ks: ", ".join(f"{k} day" + ("s" if k > 1 else "") for k in ks) + " ahead"
    if not go_sets:
        go_txt = "is met by none"
    else:
        go_txt = "is met only by " + "; ".join(f"the {LABEL.get(n, n)} model {days(v['leads'])}" + ("" if v["single_networks_meet_it"] else ", and there only by the seed-mean set (no single training meets it)") for n, v in go_sets.items())
        if set(go_sets) <= UPPER_SETS:
            go_txt += f"; {'that model is' if len(go_sets) == 1 else 'those models are'} {UPPER}, and no model whose inputs can be had in practice meets it"
    go_all = [(n, k) for n, rec in S.items() if n not in UPPER_SETS for k in F.LEADS if rec["leads"][lk(k)]["rules"]["csi"].get("verdict_all_days") == "GO"]     # review finding 1 (2026-10-29)
    and_list = lambda xs: (", ".join(xs[:-1]) + " and " + xs[-1]) if len(xs) > 1 else "".join(xs)
    all_days_txt = ("; on all high-flow days, most of them continuing floods that persistence also catches, it is met by " + and_list([
        f"the model with the {LABEL.get(n, n).lstrip('+ ')} {days([k])} at its CSI trigger ({pc(S[n]['leads'][lk(k)]['rules']['csi']['test']['all']['pod'])} warned, {pc(S[n]['leads'][lk(k)]['rules']['csi']['test']['all']['far'])} false)" for n, k in go_all])
                    if go_all else "")
    not_reached = sorted({f"{n} {lk(k)}" for n, rec in S.items() for k in F.LEADS if not rec["leads"][lk(k)]["rules"]["far_cap"]["reached"]})
    a, p_, f_ = (S.get(n, {}).get("leads", {}).get("l1", {}).get("rules") for n in ("lstm_wqa", "lstm_wqaP", "lstm_wqf"))
    sub = S.get("lstm_wqaF_sub", {}).get("leads", {}).get("l1", {}).get("rules")
    assert a and p_ and f_, "the outcome needs lstm_wqa, lstm_wqaP and lstm_wqf"
    n_on = S["lstm_wqa"]["leads"]["l1"]["n_test_onsets"]
    outcome = (f"At the warning level the project chose on the validation years (best CSI), the LSTM with AORC rain to midnight ({AORC_WORDING}) warns of {pc(a['csi']['test']['onset']['pod'])} of the {n_on} onsets 1 day ahead "
               f"({HINDSIGHT_WORDING}) with {pc(a['csi']['test']['onset']['far'])} of its warnings false; at the level that keeps the false share at or below 30% on the validation years it warns of {pc(a['far_cap']['test']['onset']['pod'])} "
               f"(test false share {pc(a['far_cap']['test']['onset']['far'])}); the perfect-rain model keeps {pc(p_['far_cap']['test']['onset']['pod'])} and the perfect-weather model {pc(f_['far_cap']['test']['onset']['pod'])} (both {UPPER}); "
               f"the GO rule for the onsets (at least {pc(pod_go)} of the onsets warned and at most {pc(far_go)} of the onset warnings false) {go_txt}{all_days_txt}; a decision layer re-arranges the trade-off and does not create skill.")
    bb = {rule: a[rule]["bootstrap_test_onsets"] for rule in RULES}
    sr = {rule: a[rule]["seed_range"]["test_at_set_trigger"] for rule in RULES}
    sta, stf = a.get("calibrated_stack", {}), (sub or {}).get("calibrated_stack", {})
    stack_txt = (f"The logistic stacker (fitted on the first half of the validation years, cut chosen on the second half at the 30% cap) warns of {pc(sta['test']['onset']['pod'])} at {pc(sta['test']['onset']['far'])} false for the rain-to-midnight set"
                 + (f" and {pc(stf['test']['onset']['pod'])} at {pc(stf['test']['onset']['far'])} false for the forecast set (against {pc(sub['far_cap']['test']['onset']['pod'])} at its 30%-false trigger)" if stf and "test" in stf else "") + ". "
                 if sta and "test" in sta else "The logistic stacker row was not computed in this run (" + str(sta.get("not_computed", "no stacker")) + "). ")
    w1 = S["lstm_wqa"]["any_lead_test"]["csi"]
    caveat = (f"Single networks (the {NUM_WORD.get(S['lstm_wqa']['n_seeds'], str(S['lstm_wqa']['n_seeds']))} seeds of the rain-to-midnight set, each at the set's trigger): {pc(sr['csi']['pod'][0])}-{pc(sr['csi']['pod'][1])} warned with {pc(sr['csi']['far'][0])}-{pc(sr['csi']['far'][1])} false at the CSI level, "
              f"{pc(sr['far_cap']['pod'][0])}-{pc(sr['far_cap']['pod'][1])} with {pc(sr['far_cap']['far'][0])}-{pc(sr['far_cap']['far'][1])} false at the 30%-false level; a block bootstrap over the test days (30-day blocks, {bb['csi']['draws']} draws) puts the seed-mean "
              f"set's share at {pc(bb['csi']['pod_ci95'][0])}-{pc(bb['csi']['pod_ci95'][1])} and {pc(bb['far_cap']['pod_ci95'][0])}-{pc(bb['far_cap']['pod_ci95'][1])} at the two levels. " + stack_txt
              + f"Warned by any lead (1, 2 or 3 days before) with the CSI triggers: {pc(w1['any_lead']['pod'])} of the onsets at {pc(w1['any_lead']['far'])} false; {w1['lead1_misses_warned_two_or_three_days_before']} of the {w1['lead1_misses']} lead-1 misses "
              f"of the rain-to-midnight set were warned two or three days before. " + (f"No trigger of the grid kept the validation false share at or below 30% for {', '.join(not_reached)}: the lowest-false trigger is reported there. " if not_reached else "")
              + "The satellite-era split's numbers are comparable only inside that split.")
    design = dict(
        what_is_chosen_where="every trigger (a multiple of the basin's Q95, from the grid) and every stacker cut is chosen on the validation years of the set's own split; the test years are scored once per rule; nothing is trained",
        rules=dict(csi="the trigger with the best pooled onset CSI on the validation years (the rule of scripts 16 and 18, which search the grid 0.30-1.50; this script searches 0.50-2.00, so triggers above 1.50 are reachable here; ties go to the trigger nearest 1.0)",
                   far_cap=f"the trigger with the most onsets warned on the validation years among those whose pooled onset false share is at or below {W.FAR_CAP} (the false share the GO rule allows), ties broken by the higher CSI; reached=False when none is, then the trigger with the lowest false share, ties to the larger trigger (warning_policy.choose_trigger)"),
        trigger_grid=list(W.GRID), far_cap=W.FAR_CAP, halves_of_the_validation_years=W.HALVES, splits=dict(main=F.SPLIT, sat=W.SAT_SPLIT),
        onset="a target day whose flow `lead` days earlier (the issue day) was at or below Q95 (flood.flood_scores); at leads 2 and 3 this means 'issued while the river was low', not 'first day above Q95'",
        pooled="hits, misses and false alarms summed over the 12 basins, then POD = hits / (hits + misses), FAR = false alarms / (hits + false alarms), CSI = hits / (hits + misses + false alarms) (flood.rates)",
        seed_sets=f"lstm_<set>_s<i>, i in range({MAX_SEEDS}), those that exist; the seed-mean set is the day-by-day average of their forecasts (checked: max_abs_difference_seed_mean_vs_average_of_seed_sets); each seed set is scored at its own validation-chosen trigger and at the seed-mean set's trigger",
        bootstrap=dict(kind="block bootstrap over the target days of the test years", block_days=BLOCK_DAYS, draws=B, seed=BOOT_SEED, interval="2.5th to 97.5th percentile of the pooled onset POD and FAR at the fixed trigger",
                       same_blocks_for_every_set_of_a_split=True, note="blocks keep the correlation of neighbouring days and of the basins of one storm; the pattern of 18_forecast_input.py"),
        average_precision="numpy, the definition of sklearn.metrics.average_precision_score: onset rows of the test years ranked by forecast / Q95, no trigger",
        stacker="warning_policy.stack: logistic regression on [log forecast/Q95 (seed mean), log today's flow/Q95, seed spread, seed max], fitted on the first half of the validation years, cut chosen on the second half by the far_cap rule; main-split sets only",
        any_lead="warning_policy.any_lead: a target day counts as warned when the issue of lead 1, 2 or 3 warned of it at that lead's trigger of the rule; onsets by the lead-1 rule",
        go_rule=dict(crit, source="results/flood_analysis.json, verdict.criteria"))
    REVIEW_FILE = F.RES / "review_warning_policy.json"                    # the independent review (2026-10-29): read, never written here
    review = json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.exists() and not F.QUICK else None
    rounds = sorted(int(k[6:]) for k in (review or {}) if re.fullmatch(r"round_\d+", k))      # 2026-10-30: the latest round of the record speaks (round_N with the highest N; the top level is round 1)
    rv, round_label = (review[f"round_{rounds[-1]}"], f"round {rounds[-1]}") if rounds else (review, None)
    rev_run = (re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", str((rv.get("dates") or {}).get("results_file_written", ""))) or ["(time not given)"])[0] if rounds else "2026-10-28 22:51"     # the run the review covered
    out_json, out_fig = F.RES_OUT / f"{OUT}.json", F.FIG_OUT / f"{OUT}.png"
    # The stale guard (the logic of scripts 21 and 22; added 2026-10-30): the review recomputed its numbers from the stored forecast files of the run it reviewed, so it covers a later
    # run only while every forecast file this run reads is byte-identical to that run's (inputs_sha256 of the file this run replaces, when that file was itself covered).
    prev = json.loads(out_json.read_text(encoding="utf-8")) if review and out_json.exists() else None            # the run this one replaces
    prev_review = ((prev or {}).get("provenance") or {}).get("independent_review")
    prev_covered = bool(prev) and (hashlib.sha256(out_json.read_bytes()).hexdigest() == ((review.get("files_reviewed") or {}).get(f"results/{OUT}.json") or {}).get("sha256")
                                   or (isinstance(prev_review, dict) and bool(prev_review.get("covers_this_run", True))))     # the reviewed record itself, or a later run still covered (runs before this flag existed applied the review's wording and changed no number)
    review_sha = hashlib.sha256(REVIEW_FILE.read_bytes()).hexdigest() if review else None
    carry = isinstance(prev_review, dict) and prev_review.get("covers_this_run") is False and prev_review.get("record_sha256", review_sha) == review_sha      # what an earlier run found under the same review record stays found
    prev_inputs = (prev or {}).get("inputs_sha256") or {}
    inputs_changed = sorted(set((prev_review.get("inputs_changed_since_the_run_it_covered") or []) if carry else []) | {p for p in set(files) | set(prev_inputs) if files.get(p) != prev_inputs.get(p)})
    seeds_changed = {**((prev_review.get("seed_counts_changed") or {}) if carry else {}),
                     **{n: [prev["sets"][n].get("n_seeds"), S[n]["n_seeds"]] for n in S if prev and n in (prev.get("sets") or {}) and prev["sets"][n].get("n_seeds") != S[n]["n_seeds"]}}
    if rounds:      # a later round covers this run while every forecast file this run reads is a file that round recomputed its numbers with (its stored_sets_reviewed)
        ssr = rv.get("stored_sets_reviewed") or {}
        n_sd = lambda fs, n: sum(1 for p in fs if re.fullmatch(rf"results/preds/{n}_s\d+\.npz", p))
        inputs_changed = sorted(p for p in set(files) | set(ssr) if files.get(p) != ssr.get(p))
        seeds_changed = {n: [n_sd(ssr, n), S[n]["n_seeds"]] for n in S if n_sd(ssr, n) != S[n]["n_seeds"]}
    review_covers = bool(review) and (bool(rounds) or prev_covered) and not inputs_changed
    if review:
        n_must = sum(1 for f in rv.get("findings", []) if str(f.get("severity", "")).startswith("must"))
        review_line = (f"independent review done ({rv.get('who', 'AI reviewer agent')}{'; ' + round_label if rounds else ''}): verdict '{rv.get('verdict', '')}', {n_must} must-fix finding(s), all wording, applied in this run; "
                       f"its numbers were recomputed from results/preds with its own code (results/review_warning_policy.json)")
        if not review_covers:
            review_line = (f"results/review_warning_policy.json ({rv.get('who', 'AI reviewer agent')}{'; ' + round_label if rounds else ''}) reviewed the run of {rev_run} on the stored forecasts of that day, not this run's: "
                           + ((f"{len(inputs_changed)} of the forecast files this run reads differ from that run's or are new"
                               + (" (seed counts " + ", ".join(f"{n} {a} -> {b}" for n, (a, b) in seeds_changed.items()) + ")" if seeds_changed else "")) if inputs_changed else "the run it covered was replaced by one it does not cover")
                           + "; the review predates these forecasts and a new review is needed before any number is quoted")
    else:
        review_line = "this file's independent review is not yet done"
    added5, kept = [n for n in S if S[n]["n_seeds"] == 5], [n for n in S if S[n]["n_seeds"] != 5]
    and_join = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)
    seed_choice = ([f"Seeds 3 and 4 of {and_join(added5)} were trained on 30 October 2026 on a second machine (scripts 16 and 18, see their SEED_NOTE), after the three-seed test scores of this file were known "
                    "(the rain-to-midnight set: 51% warned with 50% false at the CSI level, 25% at the 30%-false level); these sets are now five-seed means"
                    + (f"; {and_join(kept)} keep {and_join(sorted({NUM_WORD.get(S[n]['n_seeds'], str(S[n]['n_seeds'])) for n in kept}))} seeds and their numbers are unchanged" if kept else "") + "."]
                   if added5 else [])          # review round 2 (2026-10-30), must-fix: the disclosure HANDOFF.md section 7 asks for
    status = f"REVIEWED ({round_label}): {review_line}. The team's own checks are not yet done. {STATUS}" if rounds and review_covers else STATUS if review_covers or not review else f"NOT YET REVIEWED for this run: {review_line}; every number of this file is preliminary and must carry that label in any document (HANDOFF.md section 7). {STATUS}"
    R = dict(what=f"E4a: the warning operating point on STORED forecasts (nothing is trained). Written by code/{SCRIPT} on " + time.strftime("%Y-%m-%d") + ".",
             status=status, design=design, sets_read=[n for n, _ in sets if n in S], sets_skipped_no_stored_forecasts=skipped, sets=S, summary=summary, go_met_at=go, go_met_by=go_sets, far_cap_not_reached_on_validation=not_reached,
             outcome=outcome, outcome_caveat=caveat, outcome_full=outcome + " " + caveat,
             provenance=dict(written_and_run_by="Claude Code (Fable 5.1) wrote and ran this script under the team lead's direction",
                             independent_review=(dict(record="results/review_warning_policy.json", record_sha256=review_sha, who=rv.get("who"), verdict=rv.get("verdict"), findings=len(rv.get("findings", [])),
                                                      **({"round": f"round_{rounds[-1]}", "date": (rv.get("dates") or {}).get("review"), "record_script_sha256": rv.get("record_script_sha256"), "counts": rv.get("counts")} if rounds else {}),
                                                      must_fix=sum(1 for f in rv.get("findings", []) if str(f.get("severity", "")).startswith("must")),
                                                      covers_this_run=review_covers, inputs_changed_since_the_run_it_covered=inputs_changed, seed_counts_changed=seeds_changed,
                                                      note=(f"{round_label} of the review (made on the run of {rev_run}) recomputed this run's numbers from the same forecast files (its stored_sets_reviewed equals inputs_sha256); "
                                                            "this run applies its must-fix wording (the choices entry on seeds 3 and 4) and changes no number"
                                                            if review_covers and rounds else
                                                            "the review was made on the run of 2026-10-28 22:51; this run applies its wording fixes (GO clause for onsets with the all-days clause, both perfect sets labelled as the upper bound, the grid and tie rules stated, this line) and changes no number"
                                                            if review_covers else
                                                            f"the review{' (' + round_label + ')' if rounds else ''} was made on the run of {rev_run}, on the stored forecasts of that day{'' if rounds else ' (three seeds per set)'}; this run reads forecast files that differ from them or are new "
                                                            "(inputs_changed_since_the_run_it_covered), so its numbers do not cover this run: status says so, and a new review record is needed before any number of this file is quoted"))
                                                 if review else "not yet done: no results/review_warning_policy.json exists; HANDOFF.md section 7 asks for one that recomputes the headline numbers from results/preds"),
                             review_line=review_line,
                             team_check="The team's own checks are not yet done (team.json)"),
             choices_made_after_looking_at_test_scores=["The 30%-false rule and the trigger grid were fixed in NEXT_EXPERIMENTS_2026-10-28.md before this script ran; the test years were scored once per rule.",
                                                        "The preliminary diagnostic results/diagnostics_2026-10-28/warning_policy_rescore.json (code/warning_policy.py, 28 Oct 2026) had already printed the test scores of these rules for these sets before this script was written; "
                                                        "this script changes no rule, grid, cap, split or set relative to it and adds the seed ranges, the block-bootstrap intervals, the average precision, the verdicts and the figure."] + seed_choice,
             notes=["Quoting: documents quote outcome_full, never outcome alone. Percentages in the outcome are rounded to whole points; the tables hold the shares.",
                    "The seed-mean set (the ensemble) keeps obs exact and averages the forecasts; its warnings differ from any single deployable network: seed_range gives the single networks.",
                    "A null in this file is a share whose denominator is zero (for example the false share of a set that gave no warning).",
                    "Nothing from results/diagnostics_2026-10-28/ is a result; this file supersedes warning_policy_rescore.json for the numbers it repeats."],
             inputs_sha256=files, environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__, scikit_learn=sk_version),
             script=f"code/{SCRIPT}", script_sha256=HASH_NOW, runtime_s=round(time.time() - t0, 1), finished_local_time=time.strftime("%Y-%m-%d %H:%M"), quick_mode=F.QUICK)
    text = json.dumps(clean(R), indent=1, allow_nan=False)
    out_json.write_text(text)
    draw(json.loads(text), out_fig, pod_go, far_go)        # the figure reads exactly what the JSON holds
    print(f"\n{R['outcome_full']}\n")
    print(f"wrote {rel(out_json)} and {rel(out_fig)} ({R['runtime_s']} s; {B} bootstrap draws; stacker " + ("computed" if sk_version else "NOT computed: scikit-learn missing") + ")")
    return R


if __name__ == "__main__":
    main(sys.argv[1:])
