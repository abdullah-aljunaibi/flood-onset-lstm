"""More rivers in training: does one LSTM trained on 63 rivers forecast the same 12 rivers better than one trained on 12, and rivers it has never seen?

Experiment E3 of ROADMAP.md (step 1d). 04_lstm_all.py trains one LSTM on the 12 study basins. Here the same network, settings, inputs, seeds and years are used with more
rivers: the 63 south-eastern CAMELS basins with a complete record (the 12 and 51 others of results/basin_audit.csv, flood.candidate_ids()).
 (a) lstm63_wq and lstm63_w: trained on all 63 (the 12 included), scored on the 12 only (their prediction sets hold the 12), against the stored lstm_wq / lstm_w (12 rivers,
     the same settings). The trainer is unchanged, so it stops on the mean validation-years NSE of all 63 training rivers, the 12 included; the 12's own validation NSE at the
     chosen update is given beside it (information only). With 63 rivers each river gets about a fifth of the training sequences per update, so lstm63eq_wq repeats the gauge
     model with five times the update limit and the patience window (it asks whether a longer window moves the optimum; its kept networks turn out to lie where those of lstm63_wq do,
     so it is not equal training per river).
 (b) rivers never seen, at scale: unseen50_w_static and unseen50_w_nostatic, past weather only (a river without a gauge), trained on the 50 other rivers (the 51 minus 03471500,
     which lies inside 03473000: the outlines show it), with and without the ten basin properties; scalers, loss weights and the stopping point come from the 50 only, and one model
     forecasts all 12. Set against the leave-one-out models of 05_lobo.py (11 training rivers, one training each) and the in-training models. Probes, information only: the 12's basin
     properties replaced by the median of the 50, and a 100-update training with the 12's records spoiled against a clean one. The proximity of the 50 to the 12 is recomputed
     here (distance of the nearest training gauge, nearest training basin in property space), because it is what makes the two tests differ.
 (c) scores with flood.evaluate on the test years, leads 1 to 3, seed ranges, the gain rule of report section 4.2 (every single seed above every single seed, higher in most basins,
     Wilcoxon p below 0.05), the GO / CAUTION / NOT YET verdicts of results/flood_analysis.json, every training's record, the saved networks (results/models/) with a reload check.
Settings: results/lstm_all.json, key config (hidden size, learning rate, dropout, budget, early stopping), three seeds. Reads the prediction sets of 04_lstm_all.py (lstm_wq, lstm_w,
lstm_wqf and their seeds) and of 05_lobo.py, and results/lstm_lobo.json. Writes results/more_basins.json, figures/more_basins.png, results/models/*.pt and the prediction sets
lstm63_wq, lstm63_w, lstm63eq_wq, unseen50_w_static, unseen50_w_nostatic (+ one per seed each; the 12 basins only).
Without training again: python code/17_more_basins.py --rescore  recomputes every score, table, note and the figure from the saved prediction sets and networks. Training only what is
missing (after a crash, or the arm with five times the update limit after a pass with --no-equal): --extend. The plain run trains everything. FLOOD_QUICK=1 is the smoke test (tiny budgets, results/quick).
The test years are scored only after the trainings of the sets being scored have finished; the trainings print validation scores only. --note="text" adds a line to
choices_made_after_looking_at_test_scores (--rescore and --extend carry the lines of the existing results file over); --prior-pass='{json}' records a pass whose results file was never
written (timing.passes).
"""
import contextlib
import hashlib
import importlib
import json
import math
import pathlib
import re
import sys
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import binomtest, spearmanr, wilcoxon

import flood as F
from common import *

use_style()
SCRIPT = "17_more_basins.py"
t_start = time.time()
RESCORE = "--rescore" in sys.argv                          # no training: scores, notes and figure again from the saved prediction sets and networks
EXTEND = "--extend" in sys.argv                            # trains only the sets whose files are missing (after a crash), then scores
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
LOAD = RESCORE or EXTEND
SKIP_EQUAL = "--no-equal" in sys.argv                      # leave the arm with five times the update limit (lstm63eq_wq) out of this pass (its three trainings can be added later with --extend)
NOTES_CLI = [a[len("--note="):] for a in sys.argv if a.startswith("--note=")]                                          # lines for choices_made_after_looking_at_test_scores
PRIOR_PASS_CLI = [json.loads(a[len("--prior-pass="):]) for a in sys.argv if a.startswith("--prior-pass=")]            # passes whose results file was never written
SEEDS = [0, 1] if F.QUICK else [0, 1, 2, 3, 4]          # seeds 3-4 added 30 October 2026 on the Windows PC (disclosed through --note); two seeds in the smoke test: it walks the same code (seed ranges, the gain rule) as the full run
CTRL_SEEDS = (0, 1, 2)                                     # the stored controls of 04 have three seeds
SEED_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(len(SEEDS), str(len(SEEDS)))       # the seed count of this script's sets in words (seeds 3-4 added 30 October 2026)
SEED_NOTE = ('Seeds 3 and 4 of every set of 17_more_basins.py were added on 30 October 2026 on a second machine (Windows 11 Home, Intel Core i7-10750H, CPU only, Python 3.11.15, torch 2.2.2+cpu, numpy 1.26.4); seeds 0-2 were trained on the first machine (macOS, Intel Core i5) and were kept, not retrained. They were added after the test-year scores of seeds 0-2 and the independent review of them were known. Nothing else changed: the same settings (results/lstm_all.json, key config), inputs, split, stopping rule, warning level and success rule. Every seed average (<set>.npz) and every seed range of 17_more_basins.py now covers five trainings; the control sets of 04_lstm_all.py (lstm_wq, lstm_w, l'
             'stm_wqf) keep their three and the references of 05_lobo.py are single trainings. The two extra seeds were planned on 14 October 2026 (code/schedule.py, week 2: five trainings per model where the NSE decides; promised at the proposal talk of 15 October) for every set alike, whatever the direction of its result; the three-seed numbers remain in round 1 of the independent review record (results/review_more_basins.json) and in a copy of the three-seed results file, results/archive/more_basins_3seeds_2026-10-30.json (copied unchanged from the backup of 30 October 2026, 10:41); passes records only the trainings and times of each pass. five_seed_changes lists what changed with five seeds.')        # the --note of 30 October 2026 (fiveseeds/note17.txt) with its last clause corrected; replaces that line in choices
SEED_NOTE_START = "Seeds 3 and 4 of every set of 17_more_basins.py"       # the carried-over line this note replaces
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
P_CUT = 0.05                                               # 'p is small', as in section 4.2 of the report
EQ = 5                                                     # the 5x arm: five times the update limit and the patience window (63 / 12 = 5.25 would be equal training per river at a fixed budget)
NESTED, NESTED_IN = "03471500", "03473000"                 # S F Holston River at Riverside lies inside S F Holston River near Damascus (the outlines are checked below)
MODELS = F.RES_OUT / "models"                              # one file per trained network
lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)                         # a path as the project shows it
notes = []

# formatting helpers (as in 15_rain_to_midnight.py)
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"                                    # a share in per cent (never "-0%")
p1 = lambda x: f"{100 * x:.1f}%" if x == x else "no value"
n3 = lambda x: f"{x:.3f}"
n2 = lambda x: f"{x:.2f}"
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
yes = lambda b: "yes" if b else "no"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
div = lambda a, b: a / b if b else float("nan")
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None      # lowest and highest single seed (a seed with no value is left out)
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                            # every single seed of a above every single seed of b
digest = lambda ids: hashlib.sha256(",".join(ids).encode()).hexdigest()[:12]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p (as in 04_lstm_all.py)."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)                        # two-sided signed-rank test over the basins
    except ValueError:                                          # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


# ---- the figure (also redrawn alone from the results file: python code/17_more_basins.py --figure) --------------------------------------------------------
def draw(R):
    """2 x 2: the same 12 rivers (median NSE and onsets warned, 1 day ahead) when 12 or 63 rivers were in training, with and without the gauge; and rivers never seen (11 against 50 training rivers)."""
    import textwrap
    S, RG, UN, n_s = R["summary_test"]["sets"], R["seed_ranges"]["sets"], R["unseen"]["sets"], len(R["design"]["seeds"])
    GREY, BLUE2 = MUTED, SEQ[5]
    n_on = S["lstm_wq"]["l1"]["onset"]["hits"] + S["lstm_wq"]["l1"]["onset"]["misses"]
    seeds = lambda n, vals: vals if RG[n]["l1"]["n_seeds"] > 1 else None                  # a set of 05_lobo.py has one training: no range
    nse = lambda n: (S[n]["l1"]["nse_median"], seeds(n, RG[n]["l1"]["by_seed"]["nse_median"]))
    onset = lambda n: (S[n]["l1"]["onset"]["pod"], seeds(n, RG[n]["l1"]["by_seed"]["onset_pod"]))
    u_on = lambda n: (UN[n]["own_level"]["l1"]["onset"]["pod"], seeds(n, UN[n]["own_level"]["l1"]["by_seed"]["onset_pod"]))
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.5))

    def panel(ax, groups, fmt, title):
        """groups: [(group label, [(tick label, (value, single seeds or None), colour or None for an outline bar)])]."""
        x, xs, labs, centers, tops = 0.0, [], [], [], []
        for glabel, bars in groups:
            x0 = x
            for lab, (v, sd), col in bars:
                if col is None:                                                    # the same rivers when they were in training: an outline, not a fill
                    ax.bar(x, v, width=0.6, facecolor="none", edgecolor=INK2, linewidth=1.3, zorder=3)
                else:
                    ax.bar(x, v, width=0.6, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
                top = v
                if sd:
                    ax.plot([x, x], [min(sd), max(sd)], color=INK, lw=1.2, solid_capstyle="butt", zorder=4)
                    for y in (min(sd), max(sd)):
                        ax.plot([x - 0.1, x + 0.1], [y, y], color=INK, lw=1.2, zorder=4)
                    ax.scatter([x] * len(sd), sd, s=15, color=INK, edgecolors=SURFACE, linewidths=0.9, zorder=5)
                    top = max(max(sd), v)
                tops.append((x, top, fmt(v)))
                xs.append(x); labs.append(lab)
                x += 1.0
            centers.append((glabel, (x0 + x - 1.0) / 2))
            x += 0.8
        top_all = max(t for _, t, _ in tops)
        ax.set_ylim(0, top_all * 1.16)
        for xx, t, txt in tops:
            ax.text(xx, t + 0.012 * top_all, txt, ha="center", va="bottom", fontsize=8, color=INK, zorder=6)
        for glabel, c in centers:
            ax.text(c, -0.2, glabel, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.4, color=INK2)
        ax.set_xticks(xs, labs, fontsize=8); ax.tick_params(length=0, labelsize=8); ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.65, x - 0.8 + 0.15); ax.set_title(title, fontsize=9.2)

    has_eq = "lstm63eq_wq" in S
    eq = lambda f: [("63\nx5", f("lstm63eq_wq"), BLUE2)] if has_eq else []
    panel(axes[0, 0], [("with river gauge", [("12", nse("lstm_wq"), GREY), ("63", nse("lstm63_wq"), BLUE)] + eq(nse)),
                       ("past weather only", [("12", nse("lstm_w"), GREY), ("63", nse("lstm63_w"), BLUE)])], lambda v: f"{v:.2f}", "The same 12 rivers: median NSE, 1 day ahead")
    panel(axes[0, 1], [("with river gauge", [("12", onset("lstm_wq"), GREY), ("63", onset("lstm63_wq"), BLUE)] + eq(onset)),
                       ("past weather only", [("12", onset("lstm_w"), GREY), ("63", onset("lstm63_w"), BLUE)])], lambda v: f"{100 * v:.0f}%", f"The same 12 rivers: onsets warned, 1 day ahead")
    panel(axes[1, 0], [("with basin properties", [("11", nse("lobo_w_static"), GREY), ("50", nse("unseen50_w_static"), ORANGE), ("12\nseen", nse("lstm_w"), None)]),
                       ("weather only", [("11", nse("lobo_w_nostatic"), GREY), ("50", nse("unseen50_w_nostatic"), ORANGE), ("12\nseen", nse("lobo_ref_w_nostatic"), None)])],
          lambda v: f"{v:.2f}", "Rivers never seen: median NSE, 1 day ahead")
    panel(axes[1, 1], [("with basin properties", [("11", u_on("lobo_w_static"), GREY), ("50", u_on("unseen50_w_static"), ORANGE), ("12\nseen", u_on("lstm_w"), None)]),
                       ("weather only", [("11", u_on("lobo_w_nostatic"), GREY), ("50", u_on("unseen50_w_nostatic"), ORANGE), ("12\nseen", u_on("lobo_ref_w_nostatic"), None)])],
          lambda v: f"{100 * v:.0f}%", "Rivers never seen: onsets warned, 1 day ahead")
    for ax in axes[:, 0]:
        ax.set_ylabel("NSE in the test years (1 = perfect)", fontsize=8.3)
    for ax in axes[:, 1]:
        ax.set_ylabel(f"share of the {n_on} onsets warned", fontsize=8.3)
    h = [Line2D([], [], ls="none", marker="s", ms=8, color=GREY, label="12 rivers in training (04), or 11 (05)"),
         Line2D([], [], ls="none", marker="s", ms=8, color=BLUE, label="63 rivers in training"),
         *([Line2D([], [], ls="none", marker="s", ms=8, color=BLUE2, label=f"63 rivers, {EQ}x the update limit and patience window")] if has_eq else []),
         Line2D([], [], ls="none", marker="s", ms=8, color=ORANGE, label="50 other rivers in training, the 12 never seen"),
         Line2D([], [], ls="none", marker="s", ms=8, markerfacecolor="none", markeredgecolor=INK2, markeredgewidth=1.3, label="the same rivers when in training (12 rivers)"),
         Line2D([], [], color=INK, lw=1.2, marker="o", ms=4, markeredgecolor=SURFACE, label=f"single seeds and their range; bar = average of the {n_s} seeds")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 0.142), handlelength=1.6, labelspacing=0.45, columnspacing=1.8)
    g12 = R["gain_rule"]["lstm63_wq_against_lstm_wq"]
    n_w, n_ref = RG["lstm_w"]["l1"]["n_seeds"], RG["lobo_ref_w_nostatic"]["l1"]["n_seeds"]                  # trainings behind the two outline bars
    note = [f"Test years 2005-2014, the 12 study rivers; the number under a bar is the number of rivers in training. Gauge models warn when the forecast is above the observed high-flow level; "
            f"models without a gauge when it is above the 95th percentile of their own forecasts (the rule of 05_lobo.py).",
            f"One training, no seed range: the grey 11-river bars (05_lobo.py: one training per river). The outline bar of the weather-only group is {'one training' if n_ref == 1 else f'an average of {n_ref} trainings'} "
            f"({S['lobo_ref_w_nostatic']['l1']['nse_median']:.2f}), that of the group with properties {'one training' if n_w == 1 else f'an average of {n_w} seeds'} ({S['lstm_w']['l1']['nse_median']:.2f}).",
            f"2 and 3 days ahead, with the gauge, median NSE {g12['l2']['nse']['median_without']:.2f} -> {g12['l2']['nse']['median_with']:.2f} and {g12['l3']['nse']['median_without']:.2f} -> {g12['l3']['nse']['median_with']:.2f} (12 -> 63 rivers). "
            f"The 50 lie closer to the 12 than the 11 did: the nearest training gauge is {R['proximity']['summary']['50']['median_km']:.0f} against {R['proximity']['summary']['11_leave_one_out']['median_km']:.0f} km away (median)."]
    fig.text(0.5, 0.008, "\n".join(textwrap.fill(t, 128) for t in note), ha="center", va="bottom", fontsize=7.2, color=INK2, linespacing=1.45)
    fig.suptitle("More rivers in training: the same 12 rivers, and rivers never seen", fontsize=10.3, fontweight="bold", color=INK, x=0.012, y=0.992, ha="left")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.935, bottom=0.31, hspace=0.6, wspace=0.27)
    fig.savefig(F.FIG_OUT / "more_basins.png", dpi=220)
    plt.close(fig)


if "--figure" in sys.argv:                                  # only redraw the figure from the results file
    draw(json.loads((F.RES_OUT / "more_basins.json").read_text()))
    sys.exit(0)


# ---- 0. the rivers, the data, and what is checked about them ---------------------------------------------------------------------
ids12 = F.study_ids()
ids63 = ids12 + [g for g in F.candidate_ids() if g not in set(ids12)]          # the 12 first, then the 51 others in the audit's order
ids51 = ids63[len(ids12):]
ids50 = [g for g in ids51 if g != NESTED]
assert (len(ids12), len(ids63), len(ids51), len(ids50)) == (12, 63, 51, 50) and NESTED in ids51 and NESTED_IN in ids12, "the river sets are not the ones this script was written for"
D, B = F.load_data(ids=ids63), F.basins(ids63)                                  # the 63 tables (cache of their own) and their properties
thr = F.thresholds({g: D[g] for g in ids12})                                    # the high-flow level of each of the 12: the observed 95th percentile of its training years, used for scoring only
D12, B12 = F.load_data(), F.basins()                                            # the stored tables of the 12, for the checks below
AUDIT = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
ALL_DAYS = pd.date_range(*F.PERIOD)
DATA_CHECKS = dict(
    n_rivers=len(D), rows_per_table=sorted({len(d) for d in D.values()}), every_day_of_the_period_present=all(d.index.equals(ALL_DAYS) for d in D.values()),
    no_missing_value_in_any_table=all(not d.isna().any().any() for d in D.values()), no_negative_flow=all(bool((d.q_mm >= 0).all()) for d in D.values()),
    columns=list(D[ids63[0]].columns), static_properties=list(F.STATIC), static_properties_present_and_finite=bool(np.isfinite(B.loc[ids63, F.STATIC].values.astype(float)).all()),
    audit=dict(candidates=len(AUDIT), complete=int(AUDIT.complete.sum()), rule="no missing flow day and no missing forcing value over flood.PERIOD (01_data_audit.py), re-derived from results/basin_audit.csv by flood.candidate_ids()"),
    the_12_in_the_63_tables_equal_the_stored_12_tables=bool(list(D12) == ids12 and all(D[g].equals(D12[g]) for g in ids12)),
    properties_of_the_12_equal_flood_basins=bool(B.loc[ids12].equals(B12)), thresholds_of_the_12_equal_those_of_the_stored_12_tables=bool(thr == F.thresholds(D12)),
    landscapes_of_the_63=B.landscape.value_counts().to_dict(), landscapes_of_the_12=B.loc[ids12].landscape.value_counts().to_dict(), landscapes_of_the_50=B.loc[ids50].landscape.value_counts().to_dict(),
    cache_file=f"data/derived/camels63_daymet_{hashlib.sha256(','.join(sorted(ids63)).encode()).hexdigest()[:10]}.pkl")
assert DATA_CHECKS["every_day_of_the_period_present"] and DATA_CHECKS["no_missing_value_in_any_table"] and DATA_CHECKS["static_properties_present_and_finite"], "the 63 records are not complete"
assert DATA_CHECKS["the_12_in_the_63_tables_equal_the_stored_12_tables"] and DATA_CHECKS["thresholds_of_the_12_equal_those_of_the_stored_12_tables"], "the 12 inside the 63-basin tables are not the stored 12"


def outline_overlap():
    """Area shared by pairs of the study basins and the other 51, from the CAMELS outlines in an equal-area projection: (result, None), or (None, why) when the geometry packages are
    missing or the outline files cannot be read (the hand-in archive leaves the 69 MB folder out), so that --rescore still runs and keeps the stored result."""
    try:
        import shapefile
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely.ops import transform
    except ImportError as e:
        return None, f"the packages pyshp, pyproj and shapely were not found ({e})"
    try:
        r = shapefile.Reader(str(DATA / "camels_us" / "basin_set_full_res" / "HCDN_nhru_final_671.shp"))
        fields, want, polys = [f[0] for f in r.fields[1:]], set(ids63), {}
        to_area = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True).transform              # NAD83 -> CONUS Albers equal area
        for sr in r.iterShapeRecords():
            rec = dict(zip(fields, sr.record))
            g = str(rec.get("hru_id", rec.get("GAGEID"))).zfill(8)
            if g in want:
                polys[g] = transform(to_area, shape(sr.shape.__geo_interface__).buffer(0))
    except Exception as e:                                                                          # shapefile.ShapefileException, a missing file, a damaged one
        return None, f"the outline files could not be read ({type(e).__name__}: {str(e)[:120]})"
    assert set(polys) == want, "an outline is missing"
    rows = []
    for i, a in enumerate(ids12):
        for b in ids12[i + 1:] + ids51:
            if polys[a].intersects(polys[b]):
                x = polys[a].intersection(polys[b]).area / 1e6
                if x > 0:
                    rows.append(dict(a=a, b=b, shared_km2=float(x), share_of_a=float(x * 1e6 / polys[a].area), share_of_b=float(x * 1e6 / polys[b].area), area_a_km2=float(polys[a].area / 1e6), area_b_km2=float(polys[b].area / 1e6)))
    return dict(pairs_checked=len(ids12) * (len(ids12) - 1) // 2 + len(ids12) * len(ids51), pairs_with_shared_area=rows, projection="EPSG:5070 (CONUS Albers equal area) from the outlines' NAD83 coordinates",
                source="data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp"), None


OLD_FILE = F.RES_OUT / "more_basins.json"                                       # the results file this pass continues (--rescore, --extend): its passes, its outline check and its list of notes are carried over
old = json.loads(OLD_FILE.read_text()) if LOAD and OLD_FILE.exists() else None
OUTLINES, OUTLINES_WHY = outline_overlap()
if OUTLINES is not None:
    OUTLINES["re_checked_in_this_pass"] = True
elif old and old.get("outlines"):                                               # the outlines are not here (the hand-in archive leaves them out): keep the stored result
    OUTLINES = dict(old["outlines"], re_checked_in_this_pass=False, not_re_checked_because=OUTLINES_WHY)
    notes.append(f"The outlines were not re-checked in this pass ({OUTLINES_WHY}); the result of the earlier pass (outlines.pairs_with_shared_area) was kept.")
else:
    notes.append(f"The outlines were not checked ({OUTLINES_WHY}): that 03471500 lies inside 03473000 is the finding of the scoping work, not re-checked here.")
if OUTLINES is not None:
    nest = [r for r in OUTLINES["pairs_with_shared_area"] if {r["a"], r["b"]} == {NESTED, NESTED_IN}]
    assert nest and nest[0]["share_of_b"] > 0.99 and nest[0]["b"] == NESTED, "the outlines do not show the nested gauge this script drops"
    NEST_ROW = nest[0]
    OTHER_OVERLAPS = [r for r in OUTLINES["pairs_with_shared_area"] if r is not nest[0]]
else:
    NEST_ROW, OTHER_OVERLAPS = None, None


# ---- how close the training rivers lie to the 12 (recomputed here from the CAMELS attributes and gauge coordinates) -------------------
def haversine_km(la1, lo1, la2, lo2):
    a = np.sin(np.radians(la2 - la1) / 2) ** 2 + np.cos(np.radians(la1)) * np.cos(np.radians(la2)) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(a)))


LAT, LON = B.gauge_lat.astype(float), B.gauge_lon.astype(float)
SP = B[F.STATIC].astype(float)


def nearest_km(g, pool):
    """Distance in km from the gauge of g to the nearest gauge of the other basins in pool, and which one."""
    d = {h: haversine_km(LAT[g], LON[g], LAT[h], LON[h]) for h in pool if h != g}
    h = min(d, key=d.get)
    return d[h], h


def nearest_property(g, pool):
    """Nearest training basin in basin-property space as the model sees it: ten properties standardised with the mean and spread of the training basins (as 05_lobo.unlike_the_rest)."""
    o = SP.loc[[h for h in pool if h != g]]
    mu, sd = o.mean(), o.std(ddof=0)
    z = (SP.loc[g] - mu) / sd
    d = np.sqrt((((o - mu) / sd - z) ** 2).sum(axis=1))
    return dict(distance=float(d.min()), nearest=str(d.idxmin()), outside_range=[c for c in F.STATIC if SP.loc[g, c] > o[c].max() or SP.loc[g, c] < o[c].min()], largest_abs_z=float(z.abs().max()))


DESIGNS = {"11_leave_one_out": lambda g: [h for h in ids12 if h != g], "50": lambda g: ids50, "51": lambda g: ids51}        # the training rivers when g is the river never seen
PROX = {name: {g: dict(km=nearest_km(g, pool(g))[0], nearest_gauge=nearest_km(g, pool(g))[1], **{f"prop_{k}": v for k, v in nearest_property(g, pool(g)).items()}) for g in ids12} for name, pool in DESIGNS.items()}
prox_sum = {name: dict(median_km=float(np.median([r["km"] for r in t.values()])), min_km=float(min(r["km"] for r in t.values())), max_km=float(max(r["km"] for r in t.values())),
                       n_within_50_km=int(sum(r["km"] <= 50 for r in t.values())), n_within_100_km=int(sum(r["km"] <= 100 for r in t.values())),
                       median_property_distance=float(np.median([r["prop_distance"] for r in t.values()])), n_with_a_property_outside_the_training_range=int(sum(bool(r["prop_outside_range"]) for r in t.values())),
                       largest_abs_z=float(max(r["prop_largest_abs_z"] for r in t.values()))) for name, t in PROX.items()}
ZERO = {g: float((D[g].q_mm == 0).mean()) for g in ids63}                       # share of days without any flow, whole period
zero_sum = dict(share_of_zero_flow_days_of_the_12={g: ZERO[g] for g in ids12}, largest_share_among_the_50=float(max(ZERO[g] for g in ids50)), basin_with_it=max(ids50, key=ZERO.get),
                twelve_with_more_than_every_one_of_the_50=[g for g in ids12 if ZERO[g] > max(ZERO[h] for h in ids50)],
                n_of_the_50_with_any_zero_flow_day=int(sum(ZERO[g] > 0 for g in ids50)), n_of_the_12_with_any_zero_flow_day=int(sum(ZERO[g] > 0 for g in ids12)))
PROXIMITY = dict(how_to_read="for each of the 12 as the river never seen: the nearest training gauge (great-circle km between the CAMELS gauge coordinates) and the nearest training basin in the "
                             "standardised ten-property space of the model, when the training rivers are the other 11 (the leave-one-out test of 05_lobo.py), the 50 of this script, or the 51 (with the nested gauge)",
                 per_basin=PROX, summary=prox_sum, zero_flow=zero_sum,
                 landscape_counts=dict(twelve=DATA_CHECKS["landscapes_of_the_12"], fifty=DATA_CHECKS["landscapes_of_the_50"]))


# ---- 1. settings, the sets of this script, saved networks ------------------------------------------------------------------------
LA = json.loads((F.RES / "lstm_all.json").read_text())                          # the real file also in a smoke test
LOBO = json.loads((F.RES / "lstm_lobo.json").read_text())
FA = json.loads((F.RES / "flood_analysis.json").read_text())
conf = dict(LA["config"])                                  # hidden size, learning rate, dropout, window, batch, budget and stopping rule chosen by 04_lstm_all.py on the validation years
if F.QUICK:
    conf = dict(conf, updates=40, eval_every=20)
conf_eq = dict(conf, updates=EQ * conf["updates"], eval_every=EQ * conf["eval_every"])         # 5x the budget; patience stays 6 checks, each 5x as long: 5x the patience window in updates
ST = tuple(F.STATIC)
SET = {"lstm63_wq": dict(stage=1, train=ids63, use_flow=True, static=ST, conf=conf, inputs="wq", control="lstm_wq",
                         what="one LSTM trained on 63 rivers (the 12 and 51 others), past weather + river gauge + basin properties, scored on the 12 only"),
       "lstm63_w": dict(stage=1, train=ids63, use_flow=False, static=ST, conf=conf, inputs="w", control="lstm_w",
                        what="one LSTM trained on 63 rivers (the 12 and 51 others), past weather + basin properties (no gauge), scored on the 12 only"),
       "unseen50_w_static": dict(stage=2, train=ids50, use_flow=False, static=ST, conf=conf, inputs="w", control="lobo_w_static",
                                 what="one LSTM trained on the 50 other rivers, past weather + basin properties: the 12 are rivers never seen"),
       "unseen50_w_nostatic": dict(stage=2, train=ids50, use_flow=False, static=(), conf=conf, inputs="w", control="lobo_w_nostatic",
                                   what="one LSTM trained on the 50 other rivers, past weather only (no basin properties): the 12 are rivers never seen"),
       "lstm63eq_wq": dict(stage=3, train=ids63, use_flow=True, static=ST, conf=conf_eq, inputs="wq", control="lstm_wq",
                           what=f"as lstm63_wq with {EQ}x the update limit ({conf_eq['updates']}) and {EQ}x the patience window (a check every {conf_eq['eval_every']} updates, stop after {conf_eq['patience']} without gain)")}
if SKIP_EQUAL:
    SET.pop("lstm63eq_wq")
EQUAL = "lstm63eq_wq" in SET
for v in SET.values():
    v["n_train"], v["train_digest"] = len(v["train"]), digest(v["train"])
STORED = ["lstm_wq", "lstm_w", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w", "lstm_wqf"]      # sets of 04 and 05, read here, never made
REAL_PRED = F.RES / "preds"                                # the sets of 04 and 05 are read from here also in a smoke test (the quick folder holds their tiny smoke-test output)


def load_stored(name):
    """A prediction set written by another script (04_lstm_all.py, 05_lobo.py): always the real file."""
    keep, F.PRED = F.PRED, REAL_PRED
    try:
        return F.load_preds(name)
    finally:
        F.PRED = keep


for n in STORED + [f"{n}_s{s}" for n in ("lstm_wq", "lstm_w", "lstm_wqf") for s in CTRL_SEEDS]:
    assert (REAL_PRED / f"{n}.npz").exists(), f"the stored set {n} is missing: run 04_lstm_all.py, 05_lobo.py first"
assert conf == {k: LA["config"][k] for k in conf} or F.QUICK


def have_seed(name, seed):
    """True when this script has saved the forecasts and the network of that training with these settings and these training rivers."""
    f, m = F.PRED / f"{name}_s{seed}.npz", MODELS / f"{name}_s{seed}.pt"
    if not (f.exists() and m.exists()):
        return False
    meta = F.preds_meta(f"{name}_s{seed}")
    return meta.get("script") == SCRIPT and meta.get("config") == SET[name]["conf"] and meta.get("train_ids_digest") == SET[name]["train_digest"]


def save_mine(name, P, meta):
    """flood.save_preds, but never over a prediction set that another script wrote."""
    f = F.PRED / f"{name}.npz"
    assert not f.exists() or F.preds_meta(name).get("script") == SCRIPT, f"{f.name} belongs to another script: not overwritten"
    F.save_preds(name, P, dict(meta, script=SCRIPT))


def save_model(run, name, seed):
    """results/models/<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again: weights, settings,
    scalers (the properties' too), the training rivers in training order, the property values used, the split, the training record, the software."""
    f = MODELS / f"{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    cols = list(run.cfg.static)
    torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in run.net.state_dict().items()}, cfg=asdict(run.cfg),
                    sc={k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in run.sc.items()}, ids=list(run.ids), dyn=list(run.cfg.dyn), static=cols,
                    static_values_train={g: [float(x) for x in B.loc[g, cols]] for g in run.ids} if cols else {},
                    scored_ids=list(ids12), static_values_scored={g: [float(x) for x in B.loc[g, cols]] for g in ids12} if cols else {},
                    split=dict(run.split), best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update),
                    torch_version=str(torch.__version__), numpy_version=str(np.__version__), pandas_version=str(pd.__version__), torch_threads=int(torch.get_num_threads()), quick_mode=F.QUICK,
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to float64 arrays; "
                                  "flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, updates_run, best_update, best_val_nse, n_params, split).predict(D, scored_ids, static_df) "
                                  "with D = flood.load_data(ids=...) and static_df = a table of static_values_scored (rows = gauge ids, columns = static); the inputs are the six Daymet columns "
                                  "of flood.FORCING (+ the observed flow when cfg.use_flow), and the properties of flood.STATIC when cfg.static is set"), tmp)
    tmp.replace(f)
    return f


def load_run(name, seed):
    """Rebuild a trained network from its file alone: (flood.Run, the file's dict, the property table of the scored rivers or None)."""
    f = MODELS / f"{name}_s{seed}.pt"
    z = torch.load(f, map_location="cpu", weights_only=True)
    assert z["set"] == name and z["script"] == SCRIPT and z["seed"] == seed, f"{f.name} is not the network of {name}, seed {seed}"
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    assert z["state_dict"]["lstm.weight_ih_l0"].shape[1] == n_in and list(cfg.dyn) == z["dyn"] and list(cfg.static) == z["static"], f"{f.name}: the input columns do not fit the weights"
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    sdf = pd.DataFrame(z["static_values_scored"], index=list(cfg.static)).T if cfg.static else None
    return run, z, sdf


def record_of(name, seed):
    """The training record of one network, from its own file."""
    z = torch.load(MODELS / f"{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    return dict(set=name, seed=seed, **{k: z[k] for k in COST}, model_file=rel(MODELS / f"{name}_s{seed}.pt"), curve=[[h["update"], round(h["val_nse"], 4)] for h in z["history"]],
                history=z["history"], n_train_rivers=len(z["ids"]), torch_threads=z["torch_threads"], mean_load_average=float(np.mean([h.get("load", float("nan")) for h in z["history"]])))


val12 = lambda E: float(np.mean([[E["basins"][g][lk(k)]["nse"] for k in F.LEADS] for g in ids12]))    # the trainer's own score (mean NSE over rivers and leads), restricted to the 12


# ---- 2. training (under the CPU lock) -------------------------------------------------------------------------------------------------
missing = [(n, s) for n in SET for s in SEEDS if not have_seed(n, s)]
if RESCORE:
    assert not missing, "--rescore trains nothing, and these are missing: " + ", ".join(f"{n}_s{s}" for n, s in missing) + ": run --extend first"
assert old is not None or not RESCORE, "--rescore needs the results file of a training run"
old_written = time.strftime("%Y-%m-%d %H:%M", time.localtime(OLD_FILE.stat().st_mtime)) if old else None
todo = {n: ([] if RESCORE else [s for s in SEEDS if not EXTEND or (n, s) in set(missing)]) for n in SET}
probe_old = (old or {}).get("leak_probes", {}).get("training_ignores_the_12")
PROBE_NOW = not RESCORE and (not EXTEND or probe_old is None)           # an --extend pass makes the probe when the results file it continues does not hold it
log = lambda s: print(s, flush=True)


def probe_training():
    """Information only: does fit() read anything of the 12 when they are not in `ids`? Three short trainings on the 50 (properties as inputs): two on the clean tables, one with the
    12's records spoiled (flow missing, weather and properties replaced by large random numbers). Their curves must agree to within what the same training gives twice."""
    n_up = 20 if F.QUICK else 100
    cfg = F.Cfg(use_flow=False, static=ST, seed=0, **dict(conf, updates=n_up, eval_every=n_up, patience=1))
    rng = np.random.default_rng(123)
    Dp, Bp = dict(D), B.copy()
    for g in ids12:
        d = D[g].copy()
        d[F.FORCING] = rng.normal(size=(len(d), len(F.FORCING))) * 100.0
        d["q_mm"] = np.nan
        Dp[g] = d
        Bp.loc[g, F.STATIC] = rng.normal(size=len(F.STATIC)) * 100.0
    ra, rb, rc = (F.fit(cfg, D, ids50, static_df=B), F.fit(cfg, D, ids50, static_df=B), F.fit(cfg, Dp, ids50, static_df=Bp))
    curve = lambda r: [{k: h[k] for k in ("update", "train_loss", "val_nse")} | dict(zip(("val_l1", "val_l2", "val_l3"), h["val_nse_by_lead"])) for h in r.history]
    gap = lambda x, y: max(abs(p[k] - q[k]) for p, q in zip(curve(x), curve(y)) for k in p)
    wgap = lambda x, y: max(float((x.net.state_dict()[k] - y.net.state_dict()[k]).abs().max()) for k in x.net.state_dict())
    out = dict(updates=n_up, trained_on=len(ids50), same_training_twice_curve_gap=gap(ra, rb), same_training_twice_weight_gap=wgap(ra, rb), spoiled_12_curve_gap=gap(ra, rc), spoiled_12_weight_gap=wgap(ra, rc),
               same_ids_and_curve_updates=[h["update"] for h in ra.history], train_loss_clean=ra.history[-1]["train_loss"], train_loss_spoiled=rc.history[-1]["train_loss"])
    out["the_12_are_not_read"] = bool(out["spoiled_12_curve_gap"] <= out["same_training_twice_curve_gap"] and out["spoiled_12_weight_gap"] <= out["same_training_twice_weight_gap"])
    return out


def train_one(name, seed, tag):
    v = SET[name]
    cfg = F.Cfg(use_flow=v["use_flow"], static=v["static"], seed=seed, **v["conf"])
    run = F.fit(cfg, D, v["train"], static_df=B, log=log)
    P = run.predict(D, ids=ids12, static_df=B)                          # forecasts for the 12 ONLY, whatever the training set was
    mf = rel(save_model(run, name, seed))
    save_mine(f"{name}_s{seed}", P, dict(model="lstm", inputs=v["inputs"], seed=seed, seeds=[seed], config=v["conf"], trained_on=v["n_train"], train_ids_digest=v["train_digest"], static=list(v["static"]),
                                         scored_on="the 12 study basins only", model_file=mf, quick_mode=F.QUICK, what=v["what"]))
    e = F.evaluate(P, thr, "val")
    print(f"training {tag} | {name} seed {seed} | {v['n_train']} rivers | best at update {run.best_update} of {run.updates_run}, {run.seconds:.0f} s | validation NSE of the {v['n_train']} training rivers "
          f"{run.best_val_nse:.4f}, of the 12 {val12(e):.4f} (information only; no test score is looked at before every training has finished)", flush=True)
    return run


waited, trained_now, probe_new = 0.0, [], None
n_todo, k_done = sum(len(s) for s in todo.values()), 0
for stage in (1, 2, 3):
    names = [n for n in SET if SET[n]["stage"] == stage and todo[n]]
    do_probe = stage == 2 and PROBE_NOW
    if not names and not do_probe:
        continue
    t_ask = time.time()
    with F.cpu_lock(SCRIPT):
        waited += time.time() - t_ask
        if do_probe:
            probe_new = probe_training()
            print(f"probe: training on the 50 with the 12's records spoiled gives a curve {probe_new['spoiled_12_curve_gap']:.2e} from the clean one; the same training twice differs by "
                  f"{probe_new['same_training_twice_curve_gap']:.2e} -> the 12 are not read: {probe_new['the_12_are_not_read']}", flush=True)
        for name in names:
            for seed in todo[name]:
                k_done += 1
                run = train_one(name, seed, f"{k_done} of {n_todo}")
                trained_now.append((name, seed, run.seconds))
for name, v in SET.items():                                               # the seed average of each set, once every seed is there
    if all(have_seed(name, s) for s in SEEDS) and (any(n == name for n, _, _ in trained_now) or not (F.PRED / f"{name}.npz").exists()):
        save_mine(name, F.ensemble([F.load_preds(f"{name}_s{s}") for s in SEEDS]), dict(model="lstm", inputs=v["inputs"], seeds=SEEDS, config=v["conf"], trained_on=v["n_train"], train_ids_digest=v["train_digest"],
                                                                                     static=list(v["static"]), scored_on="the 12 study basins only", quick_mode=F.QUICK,
                                                                                     what=v["what"] + "; day-by-day average of the seeds"))
assert all(have_seed(n, s) for n in SET for s in SEEDS), "a training is missing: run --extend"


# ---- 3. every set scored the same way: the test years once, after every training; the validation years shown apart (they chose nothing here) ---------
b05 = importlib.import_module("05_lobo")                  # own_levels() and warnings_own_level(): the warning rule for a river without a gauge (nothing runs on import)
PRED, SEEDP, REC = {}, {}, {(n, s): record_of(n, s) for n in SET for s in SEEDS}
for n in SET:
    SEEDP[n], PRED[n] = [F.load_preds(f"{n}_s{s}") for s in SEEDS], F.load_preds(n)
    again = F.ensemble(SEEDP[n])
    assert all(np.array_equal(again[g].values, PRED[n][g].values, equal_nan=True) for g in ids12), f"{n} is not the average of its seeds"
SEED_AVERAGE_CHECK = {}
for n in ("lstm_wq", "lstm_w", "lstm_wqf"):
    SEEDP[n], PRED[n] = [load_stored(f"{n}_s{s}") for s in CTRL_SEEDS], load_stored(n)
    again = F.ensemble(SEEDP[n])
    SEED_AVERAGE_CHECK[n] = float(max(np.nanmax(np.abs(again[g].values - PRED[n][g].values)) for g in ids12))     # the stored seed average against the average of the stored seeds
for n in ("lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w"):
    PRED[n], SEEDP[n] = load_stored(n), [load_stored(n)]                                               # one training per river: there is no seed range
ALL = list(PRED)
assert all(set(PRED[n]) == set(ids12) for n in ALL) and all(np.allclose(PRED[n][g].obs.values, D[g].q_mm.values) for n in ALL for g in ids12), "a set does not hold the 12 rivers or their observed flow"
EV = {n: dict(test=F.evaluate(PRED[n], thr), val=F.evaluate(PRED[n], thr, "val"), seeds=[F.evaluate(p, thr) for p in SEEDP[n]], seeds_val=[F.evaluate(p, thr, "val") for p in SEEDP[n]]) for n in ALL}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in ALL for part in ("test", "val") for k in F.LEADS), "days without a forecast: the sets would not be compared on identical days"

med = lambda E, k: E["summary"][lk(k)]["nse_median"]
nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids12])
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]                                         # onset warnings pooled over the 12
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
row = lambda E, k: dict(nse_median=med(E, k), nse_mean=E["summary"][lk(k)]["nse_mean"], onset=CUT(E["summary"][lk(k)]["onset"]), all=CUT(E["summary"][lk(k)]["all"]))

# my scoring path reproduces what 04 and 05 stored for the same forecasts
CHECK = {}
for n, inp in (("lstm_wq", "wq"), ("lstm_w", "w"), ("lstm_wqf", "wqf")):
    CHECK[n] = dict(nse_median=[med(EV[n]["test"], k) - LA["sets"][inp]["test"]["summary"][lk(k)]["nse_median"] for k in F.LEADS],
                    onset_pod=[ons(EV[n]["test"], k) - LA["sets"][inp]["test"]["summary"][lk(k)]["onset"]["pod"] for k in F.LEADS])
for n, key in (("lobo_w_static", "held_out_with_properties"), ("lobo_w_nostatic", "held_out_without_properties"), ("lobo_ref_w_nostatic", "in_training_no_properties"), ("lobo_rf_w", "forest_held_out")):
    CHECK[n] = dict(nse_median=[med(EV[n]["test"], k) - LOBO["summary"][lk(k)]["nse_median"][key] for k in F.LEADS])
STORED_CHECKS = dict(what="difference between the score this script computes for a stored set and the score its own script stored (results/lstm_all.json, results/lstm_lobo.json); all must be zero to rounding",
                     differences=CHECK, max_abs_difference=float(max(abs(x) for c in CHECK.values() for v in c.values() for x in v)),
                     stored_seed_average_against_average_of_stored_seeds_max_abs_difference=SEED_AVERAGE_CHECK)
assert STORED_CHECKS["max_abs_difference"] < 1e-9, "this script does not reproduce the stored scores of 04 / 05 for the same forecasts"

# ---- the checks of report section 4.2 (copied from 15_rain_to_midnight.py) ----------------------------------------------------------
n_b = len(ids12)


def seed_range(n, k):
    """Lowest and highest single-seed value of a set at one lead, for the scores the rule and the figure use."""
    S = [e["summary"][lk(k)] for e in EV[n]["seeds"]]
    return dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]), **{f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")},
                by_seed=dict(nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")}))


ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in ALL}


def paired_onsets(T, C, k):
    """The onsets of the test years paired: an onset is a target day above the high-flow level whose issue day was at or below it (the rule of flood.flood_scores, trigger 1.0, the observed level),
    pooled over the 12 rivers. Which of them does each set warn of? Exact McNemar test (two-sided binomial test of the onsets only one of the two sets warns of)."""
    a, b = F.SPLIT["test"]
    wT, wC = [], []
    for g in ids12:
        pT, pC = PRED[T][g], PRED[C][g]
        low = (pT.obs.shift(k) <= thr[g]).loc[a:b].values
        qT, qC = pT.loc[a:b], pC.loc[a:b]
        ok = (qT.obs.notna() & qT[lk(k)].notna()).values
        assert np.array_equal(ok, (qC.obs.notna() & qC[lk(k)].notna()).values), "the two sets do not cover the same days"
        ev = ok & low & (qT.obs > thr[g]).values
        wT.append((qT[lk(k)] > thr[g]).values[ev]), wC.append((qC[lk(k)] > thr[g]).values[ev])
    wT, wC = np.concatenate(wT), np.concatenate(wC)
    only_T, only_C = int((wT & ~wC).sum()), int((~wT & wC).sum())
    assert int(wT.sum()) == EV[T]["test"]["summary"][lk(k)]["onset"]["hits"] and int(wC.sum()) == EV[C]["test"]["summary"][lk(k)]["onset"]["hits"], "the paired onsets do not give the hits of flood.evaluate"
    return dict(n_onsets=int(len(wT)), both=int((wT & wC).sum()), only_with=only_T, only_without=only_C, neither=int((~wT & ~wC).sum()),
                mcnemar_p=float(binomtest(min(only_T, only_C), only_T + only_C, 0.5).pvalue) if only_T + only_C else float("nan"))


def rule(T, C, k):
    """Set T against set C at lead k. NSE: every single seed of T above every single seed of C, higher in most basins, Wilcoxon p below P_CUT - all three make a clear gain (as in
    15_rain_to_midnight.py and 10_rain_timing.py). Onsets warned: pooled over the basins, so only the seeds can be compared. A set with one training only (05_lobo.py) has a
    'seed range' of one value: the rule then asks whether every seed of T is above that single training."""
    ET, EC = EV[T]["test"], EV[C]["test"]
    pr, n_low = paired(nse_of(ET, k), nse_of(EC, k)), int((nse_of(ET, k) < nse_of(EC, k)).sum())
    sT, sC = ranges[T][lk(k)], ranges[C][lk(k)]
    up, down = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"])
    small_p = bool(pr["wilcoxon_p"] < P_CUT)                       # False when p is not defined
    gain, loss = up and pr["n_basins_higher"] > n_b / 2 and small_p, down and n_low > n_b / 2 and small_p
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    sTo, sCo = ET["summary"][lk(k)]["onset"], EC["summary"][lk(k)]["onset"]
    return dict(against=C, lead=k, seeds_of_the_set=sT["n_seeds"], seeds_of_the_reference=sC["n_seeds"],
                nse=dict(median_with=med(ET, k), median_without=med(EC, k), gain=med(ET, k) - med(EC, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                         single_seeds_all_above=up, single_seeds_all_below=down, n_basins_higher=pr["n_basins_higher"], n_basins_lower=n_low, n_basins=n_b,
                         higher_in_most_basins=bool(pr["n_basins_higher"] > n_b / 2), median_basin_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"],
                         p_below_cut=small_p, clear_gain=bool(gain), clear_loss=bool(loss),
                         reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=ons(ET, k), warned_without=ons(EC, k), gain=ons(ET, k) - ons(EC, k), seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                           false_with=ons(ET, k, "far"), false_without=ons(EC, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down,
                           clear_gain=bool(o_up), clear_loss=bool(o_down), reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference",
                           reading_is="the seed rule on the pooled share of onsets warned at the fixed (observed) level: a flag, not a verdict on skill; the hits, false alarms, CSI and the paired test below say what moved",
                           hits_with=sTo["hits"], hits_without=sCo["hits"], misses_with=sTo["misses"], misses_without=sCo["misses"], false_alarms_with=sTo["false_alarms"], false_alarms_without=sCo["false_alarms"],
                           csi_with=sTo["csi"], csi_without=sCo["csi"], paired=paired_onsets(T, C, k)))


PAIRS_MAIN = [("lstm63_wq", "lstm_wq"), ("lstm63_w", "lstm_w")] + ([("lstm63eq_wq", "lstm_wq"), ("lstm63eq_wq", "lstm63_wq")] if EQUAL else [])
gain_rule = {f"{T}_against_{C}": {lk(k): rule(T, C, k) for k in F.LEADS} for T, C in PAIRS_MAIN}

# how much of the gap to the perfect-forecast model (an upper bound that cannot be had in practice) does training on more rivers close?
gap_to_upper_bound = {}
for T, C in [("lstm63_wq", "lstm_wq")] + ([("lstm63eq_wq", "lstm_wq")] if EQUAL else []):
    gap_to_upper_bound[f"{T}_against_{C}"] = {lk(k): dict(
        nse_gain=med(EV[T]["test"], k) - med(EV[C]["test"], k), nse_gap=med(EV["lstm_wqf"]["test"], k) - med(EV[C]["test"], k),
        nse_share_of_gap_closed=div(med(EV[T]["test"], k) - med(EV[C]["test"], k), med(EV["lstm_wqf"]["test"], k) - med(EV[C]["test"], k)),
        onset_gain=ons(EV[T]["test"], k) - ons(EV[C]["test"], k), onset_gap=ons(EV["lstm_wqf"]["test"], k) - ons(EV[C]["test"], k),
        onset_share_of_gap_closed=div(ons(EV[T]["test"], k) - ons(EV[C]["test"], k), ons(EV["lstm_wqf"]["test"], k) - ons(EV[C]["test"], k))) for k in F.LEADS}

# ---- the verdicts of results/flood_analysis.json, applied to every set (copied from 15_rain_to_midnight.py) ------------------------------------
CRIT = FA["verdict"]["criteria"]


def number(text, key, op):
    m = re.search(rf"{key} {op} ([0-9.]+)", text)
    assert m, f"cannot read '{key} {op}' in the stored criterion: {text}"
    return float(m.group(1))


POD_GO, FAR_GO, CSI_CAUTION = number(CRIT["GO"], "pod", ">="), number(CRIT["GO"], "far", "<="), number(CRIT["CAUTION"], "csi", ">=")        # the numbers come from the stored text
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.10 if F.QUICK else 0.05)]       # warning level as a multiple of the high-flow level (06_flood_analysis.py)
KINDS, COUNTS = ("all", "onset"), ("hits", "misses", "false_alarms", "correct_neg")
pool = lambda tabs: F.rates({c: int(sum(t[c] for t in tabs)) for c in COUNTS})


def verdict(t):
    """The project's rule (results/flood_analysis.json, verdict.criteria) applied to one warning table pooled over the basins."""
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= POD_GO and t["far"] <= FAR_GO:                # far is NaN when no warning was given: then pod is 0 and this is False
        return "GO"
    return "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"


def score(P, part, trig):
    """Warning tables pooled over the basins at one warning level per lead {lead: multiple}, as flood.evaluate pools them."""
    tabs = {g: {lk(k): F.flood_scores(P[g], thr[g], k, part, trig[k]) for k in F.LEADS} for g in P}
    return {lk(k): {kind: pool([tabs[g][lk(k)][kind] for g in P]) for kind in KINDS} for k in F.LEADS}


def sweep(P, part):
    """Pooled warning tables for every warning level of LEVELS: {lead: {kind: [table per level]}}, counted with flood.contingency on the days flood.flood_scores selects."""
    a, b = F.SPLIT[part]
    out = {}
    for k in F.LEADS:
        cells = []
        for g, p in P.items():
            low, q = (p.obs.shift(k) <= thr[g]).loc[a:b].values, p.loc[a:b]
            cells.append(((q.obs > thr[g]).values, q[lk(k)].values, low, thr[g]))
        out[lk(k)] = dict(all=[pool([F.contingency(o, f > m * t) for o, f, w, t in cells]) for m in LEVELS],
                          onset=[pool([F.contingency(o[w], (f > m * t)[w]) for o, f, w, t in cells]) for m in LEVELS])
    return out


def choose(tabs):
    """Index of the level with the best pooled CSI; a tie goes to the level closest to the high-flow level itself."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])


rec_of = lambda t: dict(verdict=verdict(t), **CUT(t))


def verdict_record(P, E_test, seed_evals):
    """The three verdict readings of one prediction set, per lead: at the plain level (trigger 1.0) and at the two levels chosen on the validation years; also the verdict of every single seed."""
    sv, plain = sweep(P, "val"), E_test["summary"]                                   # the validation years choose the level; the test years are only scored
    chosen = {}
    for label, crit in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
        trig = {k: LEVELS[choose(sv[lk(k)][crit])] for k in F.LEADS}
        chosen[label] = (trig, score(P, "test", trig))
    return {lk(k): dict(
        plain=dict(trigger=1.0, **{kind: rec_of(plain[lk(k)][kind]) for kind in KINDS}),
        **{label: dict(trigger=trig[k], chosen_by=f"best pooled validation-years CSI of {'all high-flow days' if 'all_days' in label else 'onsets'}", **{kind: rec_of(sc_[lk(k)][kind]) for kind in KINDS})
           for label, (trig, sc_) in chosen.items()},
        single_seeds_plain={kind: {str(i): verdict(e["summary"][lk(k)][kind]) for i, e in enumerate(seed_evals)} for kind in KINDS}) for k in F.LEADS}


verdicts = {n: verdict_record(PRED[n], EV[n]["test"], EV[n]["seeds"]) for n in ALL}
VERDICT_NOTE = {"plain": "trigger 1.0: a warning when the forecast is above the observed high-flow level itself (the level of every other score here; a river without a gauge has no such level: see unseen.own_level)",
                "chosen_on_validation_all_days": "the warning level chosen on the validation years by the best pooled CSI of all high-flow days: the level of the verdicts stored in results/flood_analysis.json",
                "chosen_on_validation_onsets": "the warning level chosen on the validation years by the best pooled CSI of the onsets"}

# ---- stopping points, updates run, and the 12's own validation score at the chosen update (information only) -------------------------------------
def costs_of(n):
    """The training records of the single seeds of a set (this script's own, or those of 04 for the stored controls)."""
    if n in SET:
        return [dict({c: REC[(n, s)][c] for c in COST}, seed=s) for s in SEEDS]
    return [dict(LA["sets"][{"lstm_wq": "wq", "lstm_w": "w"}[n]]["cost"][str(s)], seed=s) for s in CTRL_SEEDS]


def stop_row(n, n_train, conf_n):
    c = costs_of(n)
    v12 = [val12(e) for e in EV[n]["seeds_val"]]
    out = dict(n_train_rivers=n_train, updates_run=[x["updates_run"] for x in c], best_update=[x["best_update"] for x in c], seconds=[x["seconds"] for x in c],
               best_val_nse_of_the_training_rivers=[x["best_val_nse"] for x in c], validation_nse_of_the_12_at_that_update=v12,
               mean_updates_run=float(np.mean([x["updates_run"] for x in c])), mean_best_update=float(np.mean([x["best_update"] for x in c])),
               sequences_per_river_at_the_best_update=float(np.mean([x["best_update"] for x in c]) * conf_n["batch"] / n_train),
               sequences_per_river_in_all_updates_run=float(np.mean([x["updates_run"] for x in c]) * conf_n["batch"] / n_train),
               stopped_at_the_update_limit=[bool(x["updates_run"] >= conf_n["updates"]) for x in c], update_limit=conf_n["updates"], patience_window_in_updates=conf_n["patience"] * conf_n["eval_every"])
    return out


STOPPING = {n: stop_row(n, SET[n]["n_train"] if n in SET else 12, SET[n]["conf"] if n in SET else LA["config"]) for n in ("lstm_wq", "lstm_w", "lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic") if n in SET or n in ("lstm_wq", "lstm_w")}
STOPPING_CHECK = {n: float(max(abs(a - b) for a, b in zip(STOPPING[n]["best_val_nse_of_the_training_rivers"], STOPPING[n]["validation_nse_of_the_12_at_that_update"]))) for n in ("lstm_wq", "lstm_w")}
assert max(STOPPING_CHECK.values()) < 1e-4, "the 12's validation score does not reproduce the trainer's score for the stored 12-river models (stored rounded to 4 decimals)"
EQUAL_SEQUENCES_UPDATES = STOPPING["lstm_wq"]["sequences_per_river_at_the_best_update"] * len(ids63) / conf["batch"]      # updates at 63 rivers that give each river as many sequences as the 12-river control had at its best update
STOPPING_RATIO = {pair: dict(mean_updates_run=STOPPING[a]["mean_updates_run"] / STOPPING[b]["mean_updates_run"], mean_best_update=STOPPING[a]["mean_best_update"] / STOPPING[b]["mean_best_update"],
                             sequences_per_river_at_the_best_update=STOPPING[a]["sequences_per_river_at_the_best_update"] / STOPPING[b]["sequences_per_river_at_the_best_update"])
                  for pair, (a, b) in {"lstm63_wq_over_lstm_wq": ("lstm63_wq", "lstm_wq"), "lstm63_w_over_lstm_w": ("lstm63_w", "lstm_w"), "lstm63eq_wq_over_lstm_wq": ("lstm63eq_wq", "lstm_wq")}.items() if a in SET}


# ---- 4. rivers never seen, at scale: the 12 forecast by models that had 50 other rivers (against 11 in 05_lobo.py) -----------------------------
UN_SHOW = ["lstm_w", "lstm63_w", "lobo_ref_w_nostatic", "lobo_w_static", "lobo_w_nostatic", "lobo_rf_w", "unseen50_w_static", "unseen50_w_nostatic"]
UN_LABEL = {"lstm_w": "in training: 12 rivers, weather + properties (04)", "lstm63_w": "in training: 63 rivers, weather + properties (this script)",
            "lobo_ref_w_nostatic": "in training: 12 rivers, weather only (05)", "lobo_w_static": "never seen, 11 training rivers, weather + properties (05, one training per river)",
            "lobo_w_nostatic": "never seen, 11 training rivers, weather only (05, one training per river)", "lobo_rf_w": "never seen, 11 training rivers, random forest (05)",
            "unseen50_w_static": "never seen, 50 training rivers, weather + properties (this script)", "unseen50_w_nostatic": "never seen, 50 training rivers, weather only (this script)"}
UN_TRAIN = {"lstm_w": 12, "lstm63_w": 63, "lobo_ref_w_nostatic": 12, "lobo_w_static": 11, "lobo_w_nostatic": 11, "lobo_rf_w": 11, "unseen50_w_static": 50, "unseen50_w_nostatic": 50}
OWN = {n: b05.warnings_own_level(PRED[n], thr) for n in UN_SHOW}                                       # a warning when the forecast is above the model's OWN 95th percentile of its training-years forecasts
OWN_SEEDS = {n: [b05.warnings_own_level(p, thr) for p in SEEDP[n]] for n in UN_SHOW}
own = lambda n, k, kind="onset", c="pod": OWN[n]["summary"][lk(k)][kind][c]
OWN_CHECK = {n: max(abs(own(n, k, kind, c) - LOBO["summary"][lk(k)]["warnings_model_level"][n][kind][c]) for k in F.LEADS for kind in ("all", "onset") for c in ("pod", "far"))
             for n in ("lobo_w_static", "lobo_w_nostatic", "lobo_rf_w", "lobo_ref_w_nostatic")}
assert max(OWN_CHECK.values()) < 1e-9, "the warnings at the model's own level do not reproduce those stored by 05_lobo.py"


def own_seed_range(n, k):
    S = [o["summary"][lk(k)] for o in OWN_SEEDS[n]]
    return {f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")}


def own_rule(T, C, k):
    """The onsets warned at the model's own level, set T against set C: the single seeds of T against those of C (a set of 05 has one training only)."""
    a, b = own_seed_range(T, k), own_seed_range(C, k)
    return dict(warned_with=own(T, k), warned_without=own(C, k), gain=own(T, k) - own(C, k), seed_range_with=a["onset_pod"], seed_range_without=b["onset_pod"], false_with=own(T, k, "onset", "far"),
                false_without=own(C, k, "onset", "far"), single_seeds_all_above=above(a["onset_pod"], b["onset_pod"]), single_seeds_all_below=above(b["onset_pod"], a["onset_pod"]))


def loss_row(Es, Eu, k):
    """NSE of a river when it was in training (Es) minus when it was never seen (Eu): positive = worse when never seen; per river, then the median over the 12."""
    a, b = nse_of(Es, k), nse_of(Eu, k)
    d = a - b
    return dict(median_loss=float(np.median(d)), mean_loss=float(d.mean()), largest_loss=float(d.max()), n_rivers_lower_when_never_seen=int((d > 0).sum()), n_rivers=n_b, wilcoxon_p=paired(a, b)["wilcoxon_p"])


LOSS = {"leave_one_out_with_properties__05_recomputed": (EV["lstm_w"]["seeds"][0], EV["lobo_w_static"]["test"], "lstm_w seed 0 against lobo_w_static: what 05_lobo.py stored"),
        "leave_one_out_without_properties__05_recomputed": (EV["lobo_ref_w_nostatic"]["test"], EV["lobo_w_nostatic"]["test"], "lobo_ref_w_nostatic against lobo_w_nostatic: what 05_lobo.py stored"),
        "50_rivers_with_properties__against_12_river_seed_average": (EV["lstm_w"]["test"], EV["unseen50_w_static"]["test"], "lstm_w (seed average) against unseen50_w_static (seed average)"),
        "50_rivers_with_properties__single_seed_0_against_seed_0": (EV["lstm_w"]["seeds"][0], EV["unseen50_w_static"]["seeds"][0], "lstm_w seed 0 against unseen50_w_static seed 0: the pairing of 05_lobo.py"),
        "50_rivers_with_properties__against_63_river_seed_average": (EV["lstm63_w"]["test"], EV["unseen50_w_static"]["test"], "lstm63_w against unseen50_w_static: the same other 50 rivers, the 12 in training or not"),
        "50_rivers_without_properties__against_12_river_model": (EV["lobo_ref_w_nostatic"]["test"], EV["unseen50_w_nostatic"]["test"], "lobo_ref_w_nostatic (12 rivers, one training) against unseen50_w_nostatic (seed average)")}
loss = {name: dict(what=what, **{lk(k): loss_row(Es, Eu, k) for k in F.LEADS}) for name, (Es, Eu, what) in LOSS.items()}
LOSS_CHECK = dict(with_properties=max(abs(loss["leave_one_out_with_properties__05_recomputed"][lk(k)]["median_loss"] - LOBO["summary"][lk(k)]["loss_from_not_seeing_the_basin"]["with_properties"]["median_nse_loss"]) for k in F.LEADS),
                  without_properties=max(abs(loss["leave_one_out_without_properties__05_recomputed"][lk(k)]["median_loss"] - LOBO["summary"][lk(k)]["loss_from_not_seeing_the_basin"]["without_properties"]["median_nse_loss"]) for k in F.LEADS))
assert max(LOSS_CHECK.values()) < 1e-9, "the loss from not having seen a river does not reproduce the numbers stored by 05_lobo.py"

PAIRS_UNSEEN = [("unseen50_w_static", "lobo_w_static"), ("unseen50_w_nostatic", "lobo_w_nostatic"), ("unseen50_w_static", "unseen50_w_nostatic"), ("unseen50_w_static", "lobo_rf_w"),
                ("unseen50_w_nostatic", "lobo_rf_w"), ("unseen50_w_static", "lstm63_w")]


def single_trainings(T, C, k):
    """Each single training of T against the reference set C (a set of 05_lobo.py is one training per river): its median NSE, the difference of the medians, the rivers higher and the paired Wilcoxon p."""
    out = []
    for s_, e in zip(SEEDS, EV[T]["seeds"]):
        pr = paired(nse_of(e, k), nse_of(EV[C]["test"], k))
        out.append(dict(seed=s_, median_nse=med(e, k), difference_of_medians=med(e, k) - med(EV[C]["test"], k), n_rivers_higher=pr["n_basins_higher"], median_river_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"]))
    return out


gain_rule_unseen = {f"{T}_against_{C}": {lk(k): dict(rule(T, C, k), own_level=own_rule(T, C, k), single_trainings=single_trainings(T, C, k)) for k in F.LEADS} for T, C in PAIRS_UNSEEN}

# what averaging three seeds adds to a single training, and which single trainings the in-training partners of 05 are
SEED_AVERAGING = {n: {lk(k): dict(seed_average_median=med(EV[n]["test"], k), single_trainings=[med(e, k) for e in EV[n]["seeds"]], mean_of_the_single_trainings=float(np.mean([med(e, k) for e in EV[n]["seeds"]])),
                                  average_minus_mean_of_the_single_trainings=med(EV[n]["test"], k) - float(np.mean([med(e, k) for e in EV[n]["seeds"]]))) for k in F.LEADS}
                  for n in ["lstm_wq", "lstm_w", "lstm63_wq", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic"] + (["lstm63eq_wq"] if EQUAL else [])}
PARTNERS = dict(what="the in-training partners of the never-seen sets: lstm_w seed 0 is the partner 05_lobo.py used (with properties); the weather-only partner lobo_ref_w_nostatic is one training",
                lstm_w_single_trainings_median_nse_1_day=[med(e, 1) for e in EV["lstm_w"]["seeds"]], lstm_w_seed_0_is_the_best_of_its_seeds=bool(med(EV["lstm_w"]["seeds"][0], 1) == max(med(e, 1) for e in EV["lstm_w"]["seeds"])),
                lstm_w_seed_average_median_nse_1_day=med(EV["lstm_w"]["test"], 1), lobo_ref_w_nostatic_median_nse_1_day=med(EV["lobo_ref_w_nostatic"]["test"], 1), lobo_ref_w_nostatic_trainings=1)

# by river: how near the training rivers are, and the scores
by_river = {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g]), zero_flow_share=ZERO[g], nearest_training_gauge_km_50=PROX["50"][g]["km"], nearest_training_gauge_50=PROX["50"][g]["nearest_gauge"],
                    nearest_training_gauge_km_11=PROX["11_leave_one_out"][g]["km"], property_distance_50=PROX["50"][g]["prop_distance"], property_distance_11=PROX["11_leave_one_out"][g]["prop_distance"],
                    nse_1_day_ahead={n: EV[n]["test"]["basins"][g]["l1"]["nse"] for n in UN_SHOW}) for g in ids12}
by_distance = {}
for design, key, sets in (("50_training_rivers", "50", ("unseen50_w_static", "unseen50_w_nostatic")), ("11_training_rivers", "11_leave_one_out", ("lobo_w_static", "lobo_w_nostatic"))):
    near, far = [g for g in ids12 if PROX[key][g]["km"] <= 50], [g for g in ids12 if PROX[key][g]["km"] > 50]
    pairs_ = {"with_properties": ("lstm_w", sets[0]), "without_properties": ("lobo_ref_w_nostatic", sets[1])}
    by_distance[design] = dict(cutoff_km=50, rivers_with_a_training_gauge_within_the_cutoff=near, rivers_beyond=far,
                               **{label: {grp: dict(n_rivers=len(ids), median_nse_in_training=float(np.median([EV[a]["test"]["basins"][g]["l1"]["nse"] for g in ids])) if ids else None,
                                                    median_nse_never_seen=float(np.median([EV[b]["test"]["basins"][g]["l1"]["nse"] for g in ids])) if ids else None,
                                                    median_loss=float(np.median([EV[a]["test"]["basins"][g]["l1"]["nse"] - EV[b]["test"]["basins"][g]["l1"]["nse"] for g in ids])) if ids else None)
                                          for grp, ids in (("within_cutoff", near), ("beyond", far))} for label, (a, b) in pairs_.items()})
sp = lambda x, y: dict(zip(("rho", "p"), map(float, spearmanr(x, y))))
loss63 = np.array([EV["lstm63_w"]["test"]["basins"][g]["l1"]["nse"] - EV["unseen50_w_static"]["test"]["basins"][g]["l1"]["nse"] for g in ids12])
loss63n = np.array([EV["lobo_ref_w_nostatic"]["test"]["basins"][g]["l1"]["nse"] - EV["unseen50_w_nostatic"]["test"]["basins"][g]["l1"]["nse"] for g in ids12])
what_goes_with_the_loss = dict(what="Spearman rank correlation over the 12 rivers, 1 day ahead, between the NSE lost by not having seen the river (seen minus never seen, 50 training rivers) and its distance to the nearest of the 50 training "
                                    "gauges (km) or to the nearest of the 50 in property space; 12 points only",
                               with_properties_against_63_river_model=dict(distance_km=sp([PROX["50"][g]["km"] for g in ids12], loss63), property_distance=sp([PROX["50"][g]["prop_distance"] for g in ids12], loss63)),
                               without_properties_against_12_river_model=dict(distance_km=sp([PROX["50"][g]["km"] for g in ids12], loss63n), property_distance=sp([PROX["50"][g]["prop_distance"] for g in ids12], loss63n)))
GAIN_VS_DISTANCE = {}
for v_, (T_, C_) in {"with_properties": ("unseen50_w_static", "lobo_w_static"), "without_properties": ("unseen50_w_nostatic", "lobo_w_nostatic")}.items():
    gain_ = np.array([EV[T_]["test"]["basins"][g]["l1"]["nse"] - EV[C_]["test"]["basins"][g]["l1"]["nse"] for g in ids12])           # NSE of each river with 50 training rivers minus with 11, 1 day ahead
    km50_, km11_ = np.array([PROX["50"][g]["km"] for g in ids12]), np.array([PROX["11_leave_one_out"][g]["km"] for g in ids12])
    no_closer_ = [g for g in ids12 if PROX["50"][g]["km"] >= PROX["11_leave_one_out"][g]["km"] - 1e-9]
    GAIN_VS_DISTANCE[v_] = dict(set=T_, against=C_, spearman_gain_against_km_to_the_nearest_of_the_50=sp(gain_, km50_), spearman_gain_against_km_the_50_save_over_the_11=sp(gain_, km11_ - km50_),
                                rivers_whose_nearest_training_gauge_is_no_closer_with_the_50=no_closer_, median_gain_of_those_rivers=float(np.median(gain_[[ids12.index(g) for g in no_closer_]])) if no_closer_ else None,
                                median_gain_of_all_12=float(np.median(gain_)), gain_by_river={g: float(x) for g, x in zip(ids12, gain_)})
REVIEW_FILE = F.RES / "review_more_basins.json"                              # the independent review of this script, when it is there (a record written by an AI agent, not by a script)
REVIEW = json.loads(REVIEW_FILE.read_text()) if REVIEW_FILE.exists() else None
UNSEEN = dict(
    how_to_read="The 12 study rivers forecast by models that never read any of their flow (the 50 other rivers in training), past weather only. own_level: onsets and all-days warnings when the forecast exceeds the 95th percentile of the model's "
                "own training-years forecasts for that river (the rule of 05_lobo.py for a river without a gauge); the observed-level scores are in summary_test (flood.evaluate: the level a gauge would give).",
    sets={n: dict(label=UN_LABEL[n], saved_as=n, n_training_rivers=UN_TRAIN[n], trainings=len(SEEDP[n]), nse_median={lk(k): med(EV[n]["test"], k) for k in F.LEADS},
                  nse_seed_range={lk(k): ranges[n][lk(k)]["nse_median"] for k in F.LEADS}, n_rivers_with_positive_nse={lk(k): int((nse_of(EV[n]["test"], k) > 0).sum()) for k in F.LEADS},
                  own_level={lk(k): dict(onset=CUT(OWN[n]["summary"][lk(k)]["onset"]), all=CUT(OWN[n]["summary"][lk(k)]["all"]), seed_ranges=own_seed_range(n, k),
                                         by_seed=dict(onset_pod=[o["summary"][lk(k)]["onset"]["pod"] for o in OWN_SEEDS[n]], onset_far=[o["summary"][lk(k)]["onset"]["far"] for o in OWN_SEEDS[n]])) for k in F.LEADS})
          for n in UN_SHOW},
    loss_from_not_having_seen_the_river=loss, loss_stored_by_05_lobo=dict(with_properties={lk(k): LOBO["summary"][lk(k)]["loss_from_not_seeing_the_basin"]["with_properties"]["median_nse_loss"] for k in F.LEADS},
                                                                         without_properties={lk(k): LOBO["summary"][lk(k)]["loss_from_not_seeing_the_basin"]["without_properties"]["median_nse_loss"] for k in F.LEADS}),
    against_the_leave_one_out_models=gain_rule_unseen, by_river=by_river, by_distance=by_distance, what_goes_with_the_loss=what_goes_with_the_loss,
    gain_over_the_05_models_against_distance=dict(what="Spearman rank correlation over the 12 rivers, 1 day ahead, between each river's gain in NSE (50 training rivers minus 11, the 05 leave-one-out model of that river) and the distance of its "
                                                       "gauge to the nearest of the 50 training gauges (or the kilometres the 50 save over the 11); and the rivers whose nearest training gauge is no closer with the 50 than with the 11",
                                                  variants=GAIN_VS_DISTANCE),
    seed_averaging=dict(what="median NSE of the seed average against the mean of the single trainings' medians: what averaging the seeds of a set adds to one training (three seeds for lstm_wq and lstm_w of 04_lstm_all.py, " + SEED_WORD + " for the sets of this script)", sets=SEED_AVERAGING), in_training_partners=PARTNERS,
    checks_against_05=dict(own_level_warnings_max_abs_difference=OWN_CHECK, loss_max_abs_difference=LOSS_CHECK))


# ---- 5. probes (information only) and the reload check --------------------------------------------------------------------------------------
def probe_properties():
    """The 12's basin properties replaced by the median of the 50 (as the model sees them: every one of the 12 becomes the typical river of the 50), the forecasts made again from the saved networks.
    Expected: a model that reads properties forecasts differently, more so for a river whose properties lie further from the median; a model without properties does not change at all."""
    med50 = B.loc[ids50, F.STATIC].astype(float).median()
    mu50, sd50 = B.loc[ids50, F.STATIC].astype(float).mean(), B.loc[ids50, F.STATIC].astype(float).std(ddof=0)
    far_from_median = {g: float(np.sqrt((((B.loc[g, F.STATIC].astype(float) - med50) / sd50) ** 2).sum())) for g in ids12}      # distance from the typical river of the 50, in units of the 50's spread
    a, b = F.SPLIT["test"]
    out = {}
    for name in ("unseen50_w_static", "lstm63_w", "unseen50_w_nostatic"):
        P1s, P0s, rows = [], [], []
        for s in SEEDS:
            run, z, sdf = load_run(name, s)
            sdf2 = B.loc[ids12].copy()
            sdf2.loc[:, list(F.STATIC)] = med50.values
            if sdf is not None:
                sdf2 = sdf2.loc[:, list(F.STATIC)]
            P0, P1 = F.load_preds(f"{name}_s{s}"), run.predict(D, ids12, sdf2)
            P0s.append(P0), P1s.append(P1)
        E0, E1 = EV[name]["test"], F.evaluate(F.ensemble(P1s), thr)
        change = {g: float(np.mean([np.abs(P1s[i][g].l1.loc[a:b].values - P0s[i][g].l1.loc[a:b].values).mean() for i in range(len(SEEDS))])) for g in ids12}
        change_all = {g: float(np.mean([np.nanmax(np.abs(P1s[i][g][[lk(k) for k in F.LEADS]].values - P0s[i][g][[lk(k) for k in F.LEADS]].values)) for i in range(len(SEEDS))])) for g in ids12}
        largest_by_seed = [max(float(np.nanmax(np.abs(P1s[i][g][[lk(k) for k in F.LEADS]].values - P0s[i][g][[lk(k) for k in F.LEADS]].values))) for g in ids12) for i in range(len(SEEDS))]
        out[name] = dict(
            reads_basin_properties=bool(name != "unseen50_w_nostatic"),
            forecasts_identical_to_the_stored_ones=bool(all(np.array_equal(P1s[i][g][["obs", "l1", "l2", "l3"]].values, P0s[i][g][["obs", "l1", "l2", "l3"]].values, equal_nan=True) for i in range(len(SEEDS)) for g in ids12)),
            largest_forecast_change_mm_day=float(max(change_all.values())),
            largest_forecast_change_mm_day_means="the river whose largest change (any day, any lead), averaged over the seeds, is the largest",
            largest_forecast_change_of_one_seed_mm_day=float(max(largest_by_seed)), largest_forecast_change_by_seed_mm_day=largest_by_seed,
            mean_abs_change_1_day_ahead_mm_day_by_river=change,
            mean_abs_change_as_share_of_mean_flow_by_river={g: change[g] / float(D[g].q_mm.loc[a:b].mean()) for g in ids12},
            median_nse_with_true_properties={lk(k): med(E0, k) for k in F.LEADS}, median_nse_with_the_median_of_the_50={lk(k): med(E1, k) for k in F.LEADS},
            n_rivers_with_lower_nse_when_replaced={lk(k): int((nse_of(E1, k) < nse_of(E0, k)).sum()) for k in F.LEADS},
            median_change_in_nse={lk(k): float(np.median(nse_of(E1, k) - nse_of(E0, k))) for k in F.LEADS},
            change_against_distance_from_the_median_river=dict(distance_by_river=far_from_median, spearman_1_day_ahead=sp([far_from_median[g] for g in ids12], [change[g] for g in ids12]) if name != "unseen50_w_nostatic" else None))
    return out


PROBE_PROPS = probe_properties()


def reload_and_forecast(name, seed):
    """Rebuild the trained network from its file alone, forecast the 12 again on the SAME tables and compare with the stored forecasts of the training."""
    run, z, sdf = load_run(name, seed)
    new, was = run.predict(D, z["scored_ids"], sdf), F.load_preds(f"{name}_s{seed}")
    cols = ["obs"] + [lk(k) for k in F.LEADS]
    a, b = np.stack([was[g][cols].values for g in ids12]), np.stack([new[g][cols].values for g in ids12])
    same_nan, close = bool(np.array_equal(np.isnan(a), np.isnan(b))), bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    assert same_nan and close, f"{name}_s{seed}: the reloaded network does not reproduce the stored forecasts"
    f = MODELS / f"{name}_s{seed}.pt"
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_training_rivers=len(z["ids"]), n_params=int(z["n_params"]), n_values_compared=int(a.size),
                max_abs_difference=float(np.nanmax(np.abs(a - b))), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), same_missing_days=same_nan, allclose=close, passed=True)


RELOAD = dict(what="every trained network rebuilt from its .pt file alone (flood.LSTMNet + the saved weights, settings, scalers and the properties of the 12), run on the same tables, and compared with the forecasts stored by save_preds",
              tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal; bit_for_bit records exact equality",
              per_network={f"{n}_s{s}": reload_and_forecast(n, s) for n in SET for s in SEEDS})
RELOAD["all_bit_for_bit"] = all(r["bit_for_bit"] for r in RELOAD["per_network"].values())
SAVED = {}                                                                                   # sha256 of every file of this script's sets: prediction sets, single seeds, networks
for n in SET:
    for f in [F.PRED / f"{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"{n}_s{s}.npz", MODELS / f"{n}_s{s}.pt")]:
        SAVED[rel(f)] = sha(f)
earlier_hashes = (old or {}).get("integrity", {}).get("sha256_of_saved_files", {})
NEW_FILES = {rel(F.PRED / f"{n}_s{s}.npz") for n, s, _ in trained_now} | {rel(MODELS / f"{n}_s{s}.pt") for n, s, _ in trained_now}
REMADE = {rel(F.PRED / f"{n}.npz") for n in SET if any(n_ == n for n_, _, _ in trained_now)}          # the seed average of a set is rewritten when one of its seeds was trained in this pass
CMP = [f for f in SAVED if f in earlier_hashes and f not in NEW_FILES and f not in REMADE]
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script as they are when this file is written; a --rescore or --extend pass compares the files it only reads with the hashes of the earlier pass",
                 sha256_of_saved_files=SAVED, made_in_this_pass=sorted(NEW_FILES) + sorted(REMADE),
                 earlier_pass_compared=dict(files_compared=len(CMP), all_equal=all(earlier_hashes[f] == SAVED[f] for f in CMP), seed_averages_remade_in_this_pass=sorted(REMADE)))
assert INTEGRITY["earlier_pass_compared"]["all_equal"], "a saved file changed since the earlier pass recorded its hash"


# ---- 6. the one sentence, the findings, the notes -------------------------------------------------------------------------------------------
sr = lambda n, k, c="nse_median": ranges[n][lk(k)][c]
cap = lambda t: t[:1].upper() + t[1:]
CSI_SAME = 0.01                                            # two CSI values closer than this are called the same in the texts below (a word, not a test)
G = {key: gain_rule[key] for key in gain_rule}
r1, rw = G["lstm63_wq_against_lstm_wq"][lk(1)], G["lstm63_w_against_lstm_w"][lk(1)]
re_, rq = (G["lstm63eq_wq_against_lstm_wq"][lk(1)], G["lstm63eq_wq_against_lstm63_wq"][lk(1)]) if EQUAL else (None, None)
ru, rn = gain_rule_unseen["unseen50_w_static_against_lobo_w_static"][lk(1)], gain_rule_unseen["unseen50_w_nostatic_against_lobo_w_nostatic"][lk(1)]
A_, B_, C_, D_ = med(EV["lstm_wq"]["test"], 1), med(EV["lstm63_wq"]["test"], 1), ons(EV["lstm_wq"]["test"], 1), ons(EV["lstm63_wq"]["test"], 1)
E_, F_, E2_, F2_ = (med(EV[n]["test"], 1) for n in ("lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic"))
V_ = verdicts["lstm63_wq"][lk(1)]["plain"]["onset"]["verdict"]
outcome = (f"Given training on 63 rivers instead of 12, median NSE 1 day ahead for the same 12 rivers goes from {n2(A_)} to {n2(B_)} (onsets warned {pc(C_)} to {pc(D_)}); "
           f"for rivers never seen, from {n2(E_)} to {n2(F_)} with basin properties ({n2(E2_)} to {n2(F2_)} without them; 50 training rivers instead of 11); the verdict is {V_}.")
S63, S12, S_EQ = STOPPING["lstm63_wq"], STOPPING["lstm_wq"], STOPPING.get("lstm63eq_wq")


def nse_words(r):
    """The NSE comparison in words: the seed rule's reading and what it rests on."""
    return (f"{r['nse']['reading']} for NSE (every single seed above every single seed of the reference: {yes(r['nse']['single_seeds_all_above'])}; "
            f"higher in {r['nse']['n_basins_higher']} of {n_b} rivers, {pv(r['nse']['wilcoxon_p'])})")


def onset_words(r, first, second):
    """The onsets-warned comparison in words, every number from the tables: the share warned at the fixed (observed) high-flow level, the hits, false alarms and CSI behind it, the paired test and the flag of the
    seed rule. first = the set with more rivers (or the longer window), second = its reference. A flag of the seed rule is called a move of the operating point, not a change of skill, only when the CSI is the
    same (within CSI_SAME) and the paired onsets do not differ (McNemar p not below P_CUT); when the paired onsets differ the words say so even if the seed rule is silent."""
    o, pr = r["onset"], r["onset"]["paired"]
    rng = lambda r_, n_: f" ({span(r_, p1)})" if n_ > 1 else ""                                       # a single training has no seed range
    facts = (f"onsets warned {p1(o['warned_without'])}{rng(o['seed_range_without'], r['seeds_of_the_reference'])} -> {p1(o['warned_with'])}{rng(o['seed_range_with'], r['seeds_of_the_set'])} at the fixed high-flow level: hits {o['hits_without']} -> {o['hits_with']}, "
             f"false alarms {o['false_alarms_without']} -> {o['false_alarms_with']}, CSI {n3(o['csi_without'])} -> {n3(o['csi_with'])}; {pr['only_with']} onsets only {first} warns of and {pr['only_without']} only {second}: exact McNemar {pv(pr['mcnemar_p'])}")
    differ = pr["mcnemar_p"] < P_CUT
    more = lambda x, y: "more" if x > y else "fewer" if x < y else "as many"
    moved = f"{more(o['hits_with'], o['hits_without'])} onsets warned and {more(o['false_alarms_with'], o['false_alarms_without'])} false alarms; CSI {n3(o['csi_without'])} -> {n3(o['csi_with'])}"
    if o["reading"] == "no clear difference":
        if not differ:
            return facts + "; the seed rule finds no clear difference and the paired onsets do not differ"
        return facts + f"; the seed rule finds no clear difference, but the paired onsets do differ ({moved})"
    side = "below" if o["reading"] == "clear loss" else "above"
    flag = f"the seed rule flags a {o['reading']} in the share of onsets warned (every single seed of {first} {side} every single seed of {second})"
    if abs(o["csi_with"] - o["csi_without"]) < CSI_SAME and not differ:
        return (facts + f"; {flag}, but the CSI is the same and the paired onsets do not differ, so at the fixed level this is a move of the operating point ({more(o['hits_with'], o['hits_without'])} onsets warned and "
                f"{more(o['false_alarms_with'], o['false_alarms_without'])} false alarms), not a change of forecast skill")
    return facts + f"; {flag}; the paired onsets {'differ' if differ else 'do not differ'} ({moved})"


def levels_words(T, C, k):
    """The same two sets at the warning level chosen on the validation years (onsets), and on all high-flow days at the fixed level (numbers from verdicts and summary_test)."""
    vT, vC = verdicts[T][lk(k)]["chosen_on_validation_onsets"], verdicts[C][lk(k)]["chosen_on_validation_onsets"]
    aT, aC = EV[T]["test"]["summary"][lk(k)]["all"], EV[C]["test"]["summary"][lk(k)]["all"]
    d = aT["csi"] - aC["csi"]
    verdict_all = "better" if d >= CSI_SAME else "worse" if d <= -CSI_SAME else "the same"
    return (f"at the warning level chosen on the validation years (a multiple of the high-flow level: {vT['trigger']:.2f} against {vC['trigger']:.2f}) {vT['onset']['hits']} onsets ({p1(vT['onset']['pod'])}) against {vC['onset']['hits']} "
            f"({p1(vC['onset']['pod'])}), false alarms {vT['onset']['false_alarms']} against {vC['onset']['false_alarms']}, CSI {n3(vT['onset']['csi'])} against {n3(vC['onset']['csi'])}; on all high-flow days at the fixed level "
            f"CSI {n3(aT['csi'])} against {n3(aC['csi'])} ({verdict_all}) and false share {pc(aT['far'])} against {pc(aC['far'])}")


st_s, st_n = ru["single_trainings"], rn["single_trainings"]
join = lambda xs, f: ", ".join(f(x) for x in xs)
join_and = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)
avg_gain = [SEED_AVERAGING["unseen50_w_static"]["l1"]["average_minus_mean_of_the_single_trainings"], SEED_AVERAGING["unseen50_w_nostatic"]["l1"]["average_minus_mean_of_the_single_trainings"]]
gvd = GAIN_VS_DISTANCE
rv = ((REVIEW or {}).get("numbers") or {}).get("spearman_nse_vs_distance_27_rivers")
review_clause = (f"; the independent review (round 1, on the networks of seeds 0-2 and their three-seed averages) also scored the saved 50-river networks on the 12 and on {rv['unseen50_w_static']['n'] - n_b} further never-seen rivers whose flow record has gaps (n = {rv['unseen50_w_static']['n']}) and found that their NSE "
                 f"{'does not fall' if not any(rv[k_]['rho'] < 0 and rv[k_]['p'] < P_CUT for k_ in ('unseen50_w_static', 'unseen50_w_nostatic')) else 'falls'} with the distance to the nearest training gauge (Spearman rho {rv['unseen50_w_static']['rho']:+.2f}, {pv(rv['unseen50_w_static']['p'])} with properties; {rv['unseen50_w_nostatic']['rho']:+.2f}, {pv(rv['unseen50_w_nostatic']['p'])} without; "
                 "results/review_more_basins.json)") if rv else ""
n_equal_fixed = round(EQUAL_SEQUENCES_UPDATES, -2)
if EQUAL:
    eq_peaks = [(REC[("lstm63eq_wq", s)]["best_update"], REC[("lstm63eq_wq", s)]["best_val_nse"], REC[("lstm63eq_wq", s)]["curve"][-1]) for s in SEEDS]       # (best update, its validation NSE of the 63, the last check [update, NSE])
    eq_peak_words = "; ".join(f"seed {s}: {b:.3f} at {u} updates, {c[1]:.3f} at the last check ({c[0]})" for s, (u, b, c) in zip(SEEDS, eq_peaks))
    eq_falls = all(c[1] < b for _, b, c in eq_peaks)                                                  # the validation score of the 63 rivers is lower at the last check than at its peak, in every seed
    eq_moved = S_EQ["mean_best_update"] > 2 * S63["mean_best_update"]                                  # the kept networks lie at more than twice the update count of the 1x arm
    equal_clause = (f"Five times the update limit and the patience window (lstm63eq_wq: up to {conf_eq['updates']} updates, a check every {conf_eq['eval_every']}, stop after {conf_eq['patience']} checks without gain): it ran {S_EQ['mean_updates_run']:.0f} updates on average, "
                    f"but its kept networks are at {join_and([str(u) for u in S_EQ['best_update']])} updates (mean {S_EQ['mean_best_update']:.0f}; {S_EQ['sequences_per_river_at_the_best_update']:.0f} training sequences per river against "
                    f"{S12['sequences_per_river_at_the_best_update']:.0f} for the 12-river control), where those of lstm63_wq are (mean {S63['mean_best_update']:.0f}); the validation score of the 63 rivers peaks there{' and falls afterwards' if eq_falls else ''} ({eq_peak_words}), "
                    f"so a longer patience window {'moved' if eq_moved else 'did not move'} the optimum, and this arm did not test equal training per river (that needs about {n_equal_fixed:.0f} updates, fixed rather than early-stopped). "
                    f"Its scores: NSE {n3(med(EV['lstm63eq_wq']['test'], 1))} ({span(sr('lstm63eq_wq', 1), n3)}); against the 12-river control {nse_words(re_)}; {onset_words(re_, 'the five-times-limit model', 'the 12-river model')}; "
                    f"against lstm63_wq {nse_words(rq)}; {onset_words(rq, 'the five-times-limit model', 'lstm63_wq')}.")
else:
    equal_clause = ("The arm with five times the update limit and the patience window (lstm63eq_wq) was not run in this pass, so this comparison cannot yet tell 'more rivers do not help' from 'each river got a fifth of the training'.")
outcome_caveat = " ".join([
    f"With the river gauge, 1 day ahead (lowest to highest single training in brackets): median NSE {n3(A_)} ({span(sr('lstm_wq', 1), n3)}) with the 12 rivers in training, {n3(B_)} ({span(sr('lstm63_wq', 1), n3)}) with 63: {nse_words(r1)}. "
    f"{cap(onset_words(r1, 'the 63-river model', 'the 12-river model'))}; {levels_words('lstm63_wq', 'lstm_wq', 1)}.",
    f"Without the gauge: NSE {n3(med(EV['lstm_w']['test'], 1))} ({span(sr('lstm_w', 1), n3)}) against {n3(med(EV['lstm63_w']['test'], 1))} ({span(sr('lstm63_w', 1), n3)}): {nse_words(rw)}; {onset_words(rw, 'the 63-river model', 'the 12-river model')}. "
    f"At 2 days ahead with the gauge: {onset_words(G['lstm63_wq_against_lstm_wq'][lk(2)], 'the 63-river model', 'the 12-river model')}; at 3 days ahead: {onset_words(G['lstm63_wq_against_lstm_wq'][lk(3)], 'the 63-river model', 'the 12-river model')}.",
    f"The 63-river models ran {S63['mean_updates_run']:.0f} updates on average against {S12['mean_updates_run']:.0f} for the 12-river controls (best at {S63['mean_best_update']:.0f} against {S12['mean_best_update']:.0f}), "
    f"that is {S63['sequences_per_river_at_the_best_update']:.0f} against {S12['sequences_per_river_at_the_best_update']:.0f} training sequences per river at the best update: at equal updates each river gets about a fifth of the training. " + equal_clause,
    f"Rivers never seen (weather only), 1 day ahead, median NSE: with basin properties {n3(E_)} with 11 training rivers (05_lobo.py: one training per river) and {n3(F_)} with 50 ({SEED_WORD}-seed average; single trainings {join(st_s, lambda x: n3(x['median_nse']))}), "
    f"without properties {n3(E2_)} and {n3(F2_)} (single trainings {join(st_n, lambda x: n3(x['median_nse']))}). Against the 05 numbers the seed rule reads {ru['nse']['reading']} with properties ({pv(ru['nse']['wilcoxon_p'])}; every 50-river seed above the single 05 training: "
    f"{yes(ru['nse']['single_seeds_all_above'])}) and {rn['nse']['reading']} without ({pv(rn['nse']['wilcoxon_p'])}; every 50-river seed above the single 05 training: {yes(rn['nse']['single_seeds_all_above'])}); but each single 50-river training is {join(st_s, lambda x: format(x['difference_of_medians'], '+.3f'))} above the 05 number with properties "
    f"(higher in {join(st_s, lambda x: str(x['n_rivers_higher']))} of {n_b} rivers; Wilcoxon p = {join(st_s, lambda x: format(x['wilcoxon_p'], '.3f'))}) and {join(st_n, lambda x: format(x['difference_of_medians'], '+.3f'))} without "
    f"(p = {join(st_n, lambda x: format(x['wilcoxon_p'], '.3f'))}). Averaging {SEED_WORD} seeds adds {min(avg_gain):.3f} to {max(avg_gain):.3f} NSE to a single 50-river training, and the 05 numbers are single trainings. For scale, the same rivers when in training score "
    f"{n3(med(EV['lstm_w']['test'], 1))} with the 12-river model (a seed average; its seed 0, the in-training partner of 05_lobo.py, scores {n3(PARTNERS['lstm_w_single_trainings_median_nse_1_day'][0])}, "
    f"{'the best' if PARTNERS['lstm_w_seed_0_is_the_best_of_its_seeds'] else 'not the best'} of its three seeds), {n3(PARTNERS['lobo_ref_w_nostatic_median_nse_1_day'])} without properties (one training) and {n3(med(EV['lstm63_w']['test'], 1))} with the 63-river model. "
    f"Onsets of the never-seen rivers: at the model's own level (the rule of 05_lobo.py for a river without a gauge) {p1(ru['own_level']['warned_without'])} -> {p1(ru['own_level']['warned_with'])} with properties and "
    f"{p1(rn['own_level']['warned_without'])} -> {p1(rn['own_level']['warned_with'])} without; at the observed level, which such a river does not have, with properties: {onset_words(ru, 'the 50-river model', 'the 11-river model')}.",
    f"The number of training rivers, their variety and their nearness to the 12 all differ between the two designs and were not separated by it: the nearest training gauge is a median of {prox_sum['50']['median_km']:.0f} km away with the 50 against "
    f"{prox_sum['11_leave_one_out']['median_km']:.0f} km with the 11, the nearest training basin in property space {prox_sum['50']['median_property_distance']:.2f} against {prox_sum['11_leave_one_out']['median_property_distance']:.2f}. The gain over the 05 models "
    f"{'does not rank' if not any(v_['spearman_gain_against_km_to_the_nearest_of_the_50']['p'] < P_CUT for v_ in gvd.values()) else 'ranks'} with the distance to the nearest of the 50 training gauges (Spearman rho {gvd['with_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['rho']:+.2f}, {pv(gvd['with_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['p'])} with properties; "
    f"{gvd['without_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['rho']:+.2f}, {pv(gvd['without_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['p'])} without; {n_b} rivers), and the "
    f"{len(gvd['with_properties']['rivers_whose_nearest_training_gauge_is_no_closer_with_the_50'])} rivers whose nearest training gauge is no closer with the 50 gained a median of {gvd['with_properties']['median_gain_of_those_rivers']:+.3f} and "
    f"{gvd['without_properties']['median_gain_of_those_rivers']:+.3f} (all {n_b}: {gvd['with_properties']['median_gain_of_all_12']:+.3f} and {gvd['without_properties']['median_gain_of_all_12']:+.3f}){review_clause}; "
    "a buffered leave-out or a dose-response over random subsets of the 50 would separate scale from nearness (not done). One model scores all 12 rivers, so its seed noise is shared by all of them."])
outcome_full = outcome + " " + outcome_caveat

findings = []
for k in F.LEADS:
    for T, C, where in (("lstm63_wq", "lstm_wq", "with the river gauge"), ("lstm63_w", "lstm_w", "without the river gauge")):
        r = G[f"{T}_against_{C}"][lk(k)]
        findings.append(f"{days(k).capitalize()}, same 12 rivers, {where}: median test NSE {n3(r['nse']['median_without'])} ({span(r['nse']['seed_range_without'], n3)}) with 12 training rivers, {n3(r['nse']['median_with'])} ({span(r['nse']['seed_range_with'], n3)}) "
                        f"with 63 ({r['nse']['gain']:+.3f}); higher in {r['nse']['n_basins_higher']} of {n_b} rivers ({pv(r['nse']['wilcoxon_p'])}); {r['nse']['reading']} (NSE). "
                        f"{cap(onset_words(r, 'the 63-river model', 'the 12-river model'))}; {levels_words(T, C, k)}.")
    if EQUAL:
        r = G["lstm63eq_wq_against_lstm_wq"][lk(k)]
        findings.append(f"{days(k).capitalize()}, with the river gauge, five times the update limit and the patience window (lstm63eq_wq; kept networks at {', '.join(str(u) for u in S_EQ['best_update'])} updates): median test NSE {n3(r['nse']['median_with'])} "
                        f"({span(r['nse']['seed_range_with'], n3)}) against {n3(r['nse']['median_without'])} for 12 rivers; {r['nse']['reading']} (NSE). {cap(onset_words(r, 'the five-times-limit model', 'the 12-river model'))}.")
for k in F.LEADS:
    for name, lab in (("with_properties", "with basin properties"), ("without_properties", "weather only")):
        T, C = (("unseen50_w_static", "lobo_w_static") if name == "with_properties" else ("unseen50_w_nostatic", "lobo_w_nostatic"))
        r = gain_rule_unseen[f"{T}_against_{C}"][lk(k)]
        sg = r["single_trainings"]
        L_old = LOBO["summary"][lk(k)]["loss_from_not_seeing_the_basin"][name]["median_nse_loss"]
        L_new = loss["50_rivers_with_properties__against_63_river_seed_average" if name == "with_properties" else "50_rivers_without_properties__against_12_river_model"][lk(k)]
        findings.append(f"{days(k).capitalize()}, never-seen rivers, {lab}: median test NSE {n3(r['nse']['median_without'])} with 11 training rivers (one training per river), {n3(r['nse']['median_with'])} ({span(r['nse']['seed_range_with'], n3)}) with 50 (seed average); "
                        f"{r['nse']['n_basins_higher']} of {n_b} rivers higher ({pv(r['nse']['wilcoxon_p'])}); {r['nse']['reading']}. Single 50-river trainings: {join(sg, lambda x: format(x['difference_of_medians'], '+.3f'))} over the 05 number, "
                        f"higher in {join(sg, lambda x: str(x['n_rivers_higher']))} of {n_b} rivers, Wilcoxon p = {join(sg, lambda x: format(x['wilcoxon_p'], '.3f'))}. Median loss from not having seen the river: {L_old:+.3f} (11 training rivers) -> {L_new['median_loss']:+.3f} "
                        f"(50, against {'lstm63_w' if name == 'with_properties' else 'lobo_ref_w_nostatic'}); onsets warned at the model's own level (the rule for a river without a gauge) {p1(r['own_level']['warned_without'])} -> {p1(r['own_level']['warned_with'])}; "
                        f"at the observed level, which such a river does not have: {onset_words(r, 'the 50-river model', 'the 11-river model')}.")
pb = PROBE_PROPS
findings.append(f"Probe, 12 properties replaced by the median of the 50 (1 day ahead): unseen50_w_static median NSE {n3(pb['unseen50_w_static']['median_nse_with_true_properties']['l1'])} -> {n3(pb['unseen50_w_static']['median_nse_with_the_median_of_the_50']['l1'])} "
                f"({pb['unseen50_w_static']['n_rivers_with_lower_nse_when_replaced']['l1']} of {n_b} rivers lower; the river with the largest change changes by {pb['unseen50_w_static']['largest_forecast_change_mm_day']:.2f} mm/day on average over the seeds, "
                f"{pb['unseen50_w_static']['largest_forecast_change_of_one_seed_mm_day']:.1f} in one seed); the same for lstm63_w, in which the 12 are in training: "
                f"{n3(pb['lstm63_w']['median_nse_with_true_properties']['l1'])} -> {n3(pb['lstm63_w']['median_nse_with_the_median_of_the_50']['l1'])}; unseen50_w_nostatic cannot read properties: forecasts identical = "
                f"{pb['unseen50_w_nostatic']['forecasts_identical_to_the_stored_ones']}.")

probe_train = probe_new if probe_new is not None else probe_old
this_pass = dict(what="rescore" if RESCORE else "extension (--extend): the trainings whose files were missing" if EXTEND else "training run: every set from scratch", n_trainings=len(trained_now),
                 sets_trained=sorted({n for n, _, _ in trained_now}), seeds_trained=sorted({s for _, s, _ in trained_now}),
                 training_seconds=round(sum(x[2] for x in trained_now), 1), waited_for_cpu_s=round(waited, 1), finished_local_time=time.strftime("%Y-%m-%d %H:%M"))
passes = ((old or {}).get("passes") or []) + [this_pass]
for pp in PRIOR_PASS_CLI:                       # a pass whose results file was never written (the first run was stopped before it wrote one): its trainings are on disk, so their seconds are summed from the network files
    if not any(q.get("what") == pp["what"] for q in passes):
        done_ = [(n, s) for n in pp["sets"] for s in SEEDS]
        passes.insert(0, dict(what=pp["what"], sets=list(pp["sets"]), n_trainings=len(done_), training_seconds=round(sum(REC[k]["seconds"] for k in done_), 1),
                              **{k: v for k, v in pp.items() if k not in ("what", "sets")}))
# ---- the independent review in rounds, and what the two extra seeds changed (added 2026-10-30 after the five-seed reviews; texts only, no number of this file depends on it) ----
ARCHIVE_DIR = F.RES / "archive"                  # the three-seed results files, copied unchanged from the backup of 30 October 2026 (Project-Flood_backup_2026-10-30, 10:41) before seeds 3-4 were scored
CHANGE_KEYS = ("reading", "reading_in_words", "verdict", "met")              # the words of the seed rule, of the verdicts and of the success rules
CHANGE_SKIP = {"summary_test", "validation_used_for_no_choice", "per_basin", "per_river", "seed_ranges", "integrity", "passes", "notes", "what_was_not_done", "findings", "training", "models",
               "choices_made_after_looking_at_test_scores", "outcome", "outcome_caveat", "outcome_full", "independent_review", "five_seed_changes"}
COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def review_rounds(rv, script_rel, covers_default=None):
    """Every round of an independent review record, in order (round 1 = the top level of the record, round_N = a later round): when, the counts, what it covers and the script hash it read."""
    if not rv:
        return []
    later = sorted((k for k in rv if re.fullmatch(r"round_\d+", k) and isinstance(rv[k], dict)), key=lambda k: int(k.split("_")[1]))
    out = []
    for i, r in [(1, rv)] + [(int(k.split("_")[1]), rv[k]) for k in later]:
        c, fr = r.get("counts") or {}, (r.get("files_reviewed") or {}).get(script_rel)
        out.append(dict(round=i, when=str(r.get("when", "")).split(",")[0].strip(), when_full=r.get("when"), covers=r.get("covers") or (covers_default or {}).get(i) or "the three-seed numbers (seeds 0-2)",
                        verdict=r.get("verdict"), counts=c, script_sha256=r.get("record_script_sha256") or (fr.get("sha256") if isinstance(fr, dict) else None),
                        text=(f"{c.get('checks_listed')} checks listed, {c.get('checks_passed', c.get('checks_pass'))} passed, {c.get('failed')} failed; {c.get('must_fix')} must-fix and {c.get('should_fix')} should-fix items, "
                              f"{c.get('numerical_mismatches', c.get('numerical_discrepancies'))} numerical mismatches")))
    return out


def review_sentence(rows, record, n_seeds, hash_now):
    """One sentence on the rounds of the review: what each covers, which round covers the numbers of this file, and whether this version of the script is the one the last round read."""
    span_ = lambda xs: f"round {xs[0]}" if len(xs) == 1 else f"rounds {xs[0]}-{xs[-1]}" if xs == list(range(xs[0], xs[-1] + 1)) else "rounds " + ", ".join(map(str, xs))
    cap_ = lambda t: t[:1].upper() + t[1:]
    five_ = [r["round"] for r in rows if "five-seed" in r["covers"]]
    three_ = [r["round"] for r in rows if r["round"] not in five_]
    s_ = (f"The independent recomputation of the numbers by a second AI agent exists in {COUNT_WORD.get(len(rows), len(rows))} round{'s' * (len(rows) > 1)} ({record}): "
          + "; ".join(f"round {r['round']} ({r['when']}; {r['text']}) covers {r['covers']}" for r in rows) + ". ")
    if n_seeds > 3:
        if five_:
            s_ += (f"{cap_(span_(three_))} scored seeds 0-2 only; " if three_ else "") + f"the {COUNT_WORD.get(n_seeds, n_seeds)}-seed numbers of this file are covered by {span_(five_)}. "
        else:
            s_ += f"Every round scored seeds 0-2 only: the {COUNT_WORD.get(n_seeds, n_seeds)}-seed numbers of this file have not been recomputed by a second agent. "
    last_ = rows[-1]["script_sha256"]
    return s_ + (f"This version of the script has sha256 {hash_now[:12]}... (" + ("the version the last round read)." if hash_now == last_ else
                 "not the version the last round read: what it changed after that round was not recomputed by a second agent)."))


def five_seed_changes(new, name, label=lambda n: n):
    """The readings, verdict words and success rules, the lead-1 scores and the onset points of the value_of_* readings of this file that differ from the three-seed results file
    (results/archive/<name>_3seeds_2026-10-30.json), with one sentence that lists them; None in a smoke test or when that file is not there."""
    path = ARCHIVE_DIR / f"{name}_3seeds_2026-10-30.json"
    if F.QUICK or not path.exists():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    seeds_of = lambda J: J.get("seeds") or (J.get("design") or {}).get("seeds") or []
    w3, w5 = (COUNT_WORD.get(len(seeds_of(J)), len(seeds_of(J))) for J in (old, new))
    words, rhos = [], []

    def walk(where, a, b):
        if "rho" in a and "p" in a and "rho" in b and "p" in b and all(isinstance(v, (int, float)) and v == v for v in (a["p"], b["p"], a["rho"], b["rho"])):
            if (a["p"] < 0.05) != (b["p"] < 0.05):
                rhos.append(dict(where=".".join(where), three_seeds=dict(rho=a["rho"], p=a["p"]), five_seeds=dict(rho=b["rho"], p=b["p"])))
        for k, x in a.items():
            if k not in b:
                continue
            y = b[k]
            if isinstance(x, dict) and isinstance(y, dict):
                walk(where + [k], x, y)
            elif k in CHANGE_KEYS and not (k == "reading" and "reading_in_words" in a) and x != y and all(isinstance(v, (str, bool)) and len(str(v)) <= 80 for v in (x, y)):
                words.append(dict(where=".".join(where + [k]), three_seeds=x, five_seeds=y))
    for k in new:
        if k in old and k not in CHANGE_SKIP and isinstance(new[k], dict) and isinstance(old[k], dict):
            walk([k], old[k], new[k])
    so, sn = old["summary_test"]["sets"], new["summary_test"]["sets"]
    scores = [dict(set=label(n), lead=1, nse_median=[so[n]["l1"]["nse_median"], sn[n]["l1"]["nse_median"]], onset_pod=[so[n]["l1"]["onset"]["pod"], sn[n]["l1"]["onset"]["pod"]],
                   onset_hits=[so[n]["l1"]["onset"]["hits"], sn[n]["l1"]["onset"]["hits"]]) for n in sn if n in so and so[n] != sn[n]]
    pts_ = lambda J, k, l_: J[k]["by_lead"][l_]["onsets"]["difference_points"]
    points = [dict(reading=k, onset_difference_points={l_: [pts_(old, k, l_), pts_(new, k, l_)] for l_ in ("l1", "l2", "l3")})
              for k in new if k.startswith("value_of_") and k in old and all(isinstance(J[k], dict) and "by_lead" in J[k] for J in (old, new))
              and any(abs(pts_(old, k, l_) - pts_(new, k, l_)) > 1e-9 for l_ in ("l1", "l2", "l3"))]
    drop = {"by_lead", "sets", "pairs", "lstm_against_simple", "against_the_leave_one_out_models"} | set(CHANGE_KEYS)
    place = lambda w: ", ".join({"l1": "1 day", "l2": "2 days", "l3": "3 days", "nse": "NSE", "onset": "onsets"}.get(p, p.replace("__", " against ").replace("_", " ")) for p in w.split(".") if p not in drop)
    val = lambda v, k: ("met" if v else "not met") if k == "met" else str(v)
    key_ = lambda x: x["where"].rsplit(".", 1)[1]
    reads = [x for x in words if key_(x) not in ("verdict", "met")]
    verds = [x for x in words if key_(x) in ("verdict", "met")]
    p1_ = lambda x: f"{100 * x:.1f}%"
    rel_ = path.relative_to(F.RES.parent).as_posix()
    sentence = (f"With {w5} seeds instead of {w3} (the {w3}-seed results file is kept unchanged as {rel_}) these numbers and words changed, the {w3}-seed values in brackets. "
                + ("Lead 1, test years: " + "; ".join(f"{s['set']} onsets warned {p1_(s['onset_pod'][1])} ({p1_(s['onset_pod'][0])}; {s['onset_hits'][1]} hits against {s['onset_hits'][0]}), median NSE "
                                                      f"{s['nse_median'][1]:.3f} ({s['nse_median'][0]:.3f})" for s in scores) + ". " if scores else "No lead-1 score changed. ")
                + ("Onsets, test minus control in points at 1/2/3 days: " + "; ".join(f"{place(p['reading'])} " + "/".join(f"{p['onset_difference_points'][l_][1]:+.1f}" for l_ in ("l1", "l2", "l3"))
                                                                                       + " (" + "/".join(f"{p['onset_difference_points'][l_][0]:+.1f}" for l_ in ("l1", "l2", "l3")) + ")" for p in points) + ". " if points else "")
                + ("Readings of the seed rule and of report section 4.2: " + "; ".join(f"{place(x['where'])}: {val(x['five_seeds'], '')} ({val(x['three_seeds'], '')})" for x in reads) + ". "
                   if reads else "No reading of the seed rule changed. ")
                + ("Rank correlations whose p crossed 0.05: " + "; ".join(f"{place(x['where'])}: rho {x['five_seeds']['rho']:+.2f}, p {x['five_seeds']['p']:.3f} (rho {x['three_seeds']['rho']:+.2f}, p {x['three_seeds']['p']:.3f})"
                                                                         for x in rhos) + ". " if rhos else "")
                + ("Verdict words and success rules: " + "; ".join(f"{place(x['where'])}: {val(x['five_seeds'], key_(x))} ({val(x['three_seeds'], key_(x))})" for x in verds) + "."
                   if verds else "No verdict word and no success rule changed."))
    return dict(compared_with=rel_, seeds=dict(three_seed_file=seeds_of(old), this_file=seeds_of(new)),
                how_to_read=f"what changed when seeds 3 and 4 were added: this file against the {w3}-seed results file named in compared_with, leaf by leaf over the reading words (reading, reading_in_words), the verdict words "
                            "and the success rules (verdict, met) of every block but summary_test (its verdicts repeat those of verdicts), the lead-1 test scores of every set whose scores changed, the onset points of the value_of_* "
                            "readings and the rank correlations whose p crossed 0.05; each pair is [three seeds, five seeds]. Computed when this file is written.",
                scores_changed_lead_1=scores, value_of_onset_points_changed=points, readings_changed=reads, rank_correlations_crossing_p_005=rhos, verdicts_and_success_rules_changed=verds, sentence=sentence)


CHOICES = list((old or {}).get("choices_made_after_looking_at_test_scores") or []) if LOAD else []        # --rescore and --extend carry the lines of the existing file over; --note adds new ones
CHOICES += [t for t in NOTES_CLI if t not in CHOICES]
load_note = float(np.nanmean([REC[k]["mean_load_average"] for k in REC]))        # the Windows trainings of 30 October 2026 record no load average (NaN, flood.py:436): the mean is over the Mac trainings
waited_all = sum(p_["waited_for_cpu_s"] for p_ in passes)
rev_c = (REVIEW or {}).get("counts")
RR17 = review_rounds(REVIEW, "code/17_more_basins.py", {1: "seeds 0-2 (the three-seed numbers of 7 October)"})          # round 1 = the top level of the record; round_2 = the five-seed round of 30 October 2026
n34_ = sum(1 for k_ in REC if k_[1] >= 3)                                                     # the trainings of seeds 3 and 4 (added 30 October 2026)
written_by_ = "--rescore (no training)" if RESCORE else f"--extend ({len(trained_now)} trainings in this pass)" if EXTEND else "a training run"
ic = INTEGRITY["earlier_pass_compared"]
notes += [
    "Design (a). lstm63_wq and lstm63_w are one LSTM each, trained on the 63 south-eastern CAMELS basins with a complete record (the 12 study basins and the 51 others of results/basin_audit.csv; flood.candidate_ids(), checked when "
    "loaded: a row for every day of 1980-10-01 to 2014-09-30, no missing value, all ten properties present: data_checks). Network, inputs, settings (results/lstm_all.json, key config) and seeds are those of the stored controls lstm_wq and lstm_w; "
    "the only change is the list of training rivers. Scalers (weather, flow, the ten properties) and loss weights come from the training years of the training rivers. The networks are scored on the 12 only and the saved prediction sets hold "
    "the 12 (flood.evaluate, the same thresholds, the same test years as every set of the report); the 12 inside the 63-basin tables are bit-identical to the stored 12-basin tables (data_checks).",
    "Stopping. flood.fit is unchanged: at every check it scores the validation years (2000-2005) of ALL its training rivers and stops on their mean NSE over rivers and leads. With 63 rivers that mean includes the 12, so the stopping point is chosen "
    "on the validation years of 63 rivers, not of the 12 (and of the 50, never the 12, for the never-seen models). stopping.*.validation_nse_of_the_12_at_that_update gives the 12's own validation score at the chosen update, information only: "
    "nothing was chosen on it; for the stored 12-river controls it equals the trainer's own best_val_nse (checked).",
    f"Training per river. Each update draws {conf['batch']} sequences of one river each, uniformly over the training rivers, so with 63 rivers a river gets about a fifth of the sequences per update. The 12-river controls ran {S12['mean_updates_run']:.0f} updates on average "
    f"(best at {S12['mean_best_update']:.0f}): {S12['sequences_per_river_at_the_best_update']:.0f} sequences per river at the best update; the 63-river models ran {S63['mean_updates_run']:.0f} (best at {S63['mean_best_update']:.0f}): "
    f"{S63['sequences_per_river_at_the_best_update']:.0f} sequences per river, {STOPPING_RATIO['lstm63_wq_over_lstm_wq']['sequences_per_river_at_the_best_update']:.2f} times the control's. "
    f"{sum(S63['stopped_at_the_update_limit'])} of {len(SEEDS)} lstm63_wq trainings and {sum(STOPPING['lstm63_w']['stopped_at_the_update_limit'])} of {len(SEEDS)} lstm63_w trainings reached the update limit of {conf['updates']} before the stopping rule was met "
    f"(controls: {sum(S12['stopped_at_the_update_limit'])} of 3 and {sum(STOPPING['lstm_w']['stopped_at_the_update_limit'])} of 3). "
    + (f"lstm63eq_wq is the arm with five times the update limit ({conf_eq['updates']}) and the patience window (a check every {conf_eq['eval_every']} updates, the same patience of {conf_eq['patience']} checks): it ran {S_EQ['mean_updates_run']:.0f} updates on average, "
       f"but its kept networks are at {', '.join(str(u) for u in S_EQ['best_update'])} updates (mean {S_EQ['mean_best_update']:.0f}; {S_EQ['sequences_per_river_at_the_best_update']:.0f} sequences per river against {S12['sequences_per_river_at_the_best_update']:.0f} for the 12-river controls), "
       f"where those of lstm63_wq are (mean {S63['mean_best_update']:.0f}): {eq_peak_words}. A longer patience window {'moved' if eq_moved else 'did not move'} the optimum; it is not equal training per river, which would need about {n_equal_fixed:.0f} updates "
       f"({len(ids63)} x {S12['sequences_per_river_at_the_best_update']:.0f} / {conf['batch']}), fixed rather than early-stopped (stopping.updates_for_equal_sequences_per_river_at_63_rivers)." if EQUAL else
       f"The arm with five times the update limit and the patience window (lstm63eq_wq) was not run in this pass."),
    f"Design (b). unseen50_w_static and unseen50_w_nostatic: past weather only (a river without a gauge has no flow input), trained on the 50 other rivers - the 51 minus {NESTED}, which lies inside {NESTED_IN} (S F Holston River near Damascus): "
    + (f"the outlines show {NEST_ROW['shared_km2']:.0f} km2, {100 * NEST_ROW['share_of_b']:.0f}% of {NESTED} and {100 * NEST_ROW['share_of_a']:.0f}% of {NESTED_IN}'s basin, so for that river the 51-river design would not be a river never seen; no other pair of the "
       f"12 x 12 and 12 x 51 outlines shares any area ({len(OUTLINES['pairs_with_shared_area']) - 1} others found)" if NEST_ROW else "the outline check could not be run here") +
    ". Scalers, loss weights and the stopping point come from the 50 rivers only (the validation years of the 50); one model forecasts all 12; the high-flow level of each of the 12 (observed 95th percentile of its training years) is used for scoring only, "
    "as in 05_lobo.py. The 12's flow is never read in training (leak_probes.training_ignores_the_12). Same settings as lstm_w, " + SEED_WORD + " seeds per variant (05_lobo.py had one training per river).",
    f"Comparability with 05_lobo.py is limited, in this order. (1) Proximity and the rest of the design: the nearest training gauge is a median of {prox_sum['50']['median_km']:.1f} km from the 12 here and {prox_sum['11_leave_one_out']['median_km']:.1f} km in the leave-one-out test "
    f"({prox_sum['50']['n_within_50_km']} against {prox_sum['11_leave_one_out']['n_within_50_km']} of the 12 have a training gauge within 50 km); the nearest training basin in the ten-property space the model sees is {prox_sum['50']['median_property_distance']:.2f} against "
    f"{prox_sum['11_leave_one_out']['median_property_distance']:.2f} ({prox_sum['50']['n_with_a_property_outside_the_training_range']} against {prox_sum['11_leave_one_out']['n_with_a_property_outside_the_training_range']} of the 12 have a property outside the training range). "
    "Neighbouring rivers share storms and geology, so 'never seen' here means 'no flow record of this river', not 'far from any gauged river'. The number of training rivers, their variety and their nearness to the 12 change together between the two designs and were not "
    f"separated: the gain over the 05 models does not rank with the distance to the nearest of the 50 training gauges (Spearman rho {gvd['with_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['rho']:+.2f}, "
    f"{pv(gvd['with_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['p'])} with properties; {gvd['without_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['rho']:+.2f}, "
    f"{pv(gvd['without_properties']['spearman_gain_against_km_to_the_nearest_of_the_50']['p'])} without; {n_b} rivers), and the {len(gvd['with_properties']['rivers_whose_nearest_training_gauge_is_no_closer_with_the_50'])} rivers whose nearest training gauge is no closer "
    f"with the 50 gained a median of {gvd['with_properties']['median_gain_of_those_rivers']:+.3f} and {gvd['without_properties']['median_gain_of_those_rivers']:+.3f}{review_clause}. A buffered leave-out (train on the 50 minus every river within 50 and 100 km of the one held out) "
    "or a dose-response over random subsets of 11, 25 and 50 rivers would separate them; neither was done. (2) One model scores all 12 here against twelve different models in 05, so the seed noise is shared by all 12 scores and "
    f"the 12 scores are not independent draws. (3) The 50 are {DATA_CHECKS['landscapes_of_the_50'].get('Northern lowlands and Piedmont', 0)} northern lowland, {DATA_CHECKS['landscapes_of_the_50'].get('Southern lowlands (south of 32.5N)', 0)} southern lowland and "
    f"{DATA_CHECKS['landscapes_of_the_50'].get('Mountains (above 600 m)', 0)} mountain basins (the 11 of 05 were four per landscape less the one held out); "
    f"{' and '.join(F.SHORT[g] for g in zero_sum['twelve_with_more_than_every_one_of_the_50'])} have a larger share of days without any flow ({' and '.join(p1(ZERO[g]) for g in zero_sum['twelve_with_more_than_every_one_of_the_50'])}) than any of the 50 "
    f"(largest {p1(zero_sum['largest_share_among_the_50'])}), so they are rivers of a kind the 50 do not hold. (4) Budget and stopping: 05 capped its trainings at {LOBO['settings']['config']['updates']} updates and patience {LOBO['settings']['config']['patience']}, one seed; here {conf['updates']} / {conf['patience']}, {SEED_WORD} seeds. "
    f"(5) Single trainings against seed averages: the 05 numbers are single trainings, and the {SEED_WORD}-seed average of unseen50_w_static is {avg_gain[0]:.3f} above the mean of its single trainings ({avg_gain[1]:.3f} for unseen50_w_nostatic), so "
    "unseen.against_the_leave_one_out_models.<pair>.<lead>.single_trainings (each single 50-river training against the 05 number, with its Wilcoxon p) is the like-for-like comparison. "
    f"The in-training partner of 05 for the variant with properties, lstm_w seed 0 ({n3(PARTNERS['lstm_w_single_trainings_median_nse_1_day'][0])}), is {'the best' if PARTNERS['lstm_w_seed_0_is_the_best_of_its_seeds'] else 'not the best'} of lstm_w's three seeds "
    f"(seed average {n3(PARTNERS['lstm_w_seed_average_median_nse_1_day'])}); the weather-only partner lobo_ref_w_nostatic is one training ({n3(PARTNERS['lobo_ref_w_nostatic_median_nse_1_day'])}).",
    "Choices made on the 12. Hidden size and learning rate were chosen in 04_lstm_all.py on the validation years of all 12 rivers (a weak leak for the never-seen test: 04 found the candidates equal within seed noise, as 05_lobo.py states). "
    "Three of the ten properties (p_mean, aridity, p_seasonality) are CAMELS climate averages of 1989-2009, which overlap the validation years and four test years; no property is computed from the flow record.",
    "Warnings. The scores of the gauge sets use the observed high-flow level (flood.evaluate, trigger 1.0), as every other score of the report. A river without a gauge has no such level, so for the never-seen sets unseen.sets.<name>.own_level gives the warnings "
    "at the model's own level (the 95th percentile of its own training-years forecasts for that river: the rule of 05_lobo.py, reproduced here and checked against its stored numbers); the verdicts in verdicts use the observed level. "
    "The flag of the seed rule for onsets (gain_rule.<pair>.<lead>.onset.reading) is a flag on the pooled share warned at one warning level: read it with the hits, false alarms, CSI and paired test beside it (onset.paired), the levels chosen on the validation years "
    "(verdicts.sets.<set>.<lead>.chosen_on_validation_onsets) and the all-days scores (summary_test); a move of the share along the warning level is not a change of skill.",
    "Test years. The test years are scored only after the trainings of the sets being scored have finished (the trainings print validation scores only). The sets, seeds, settings, stopping rule and comparison pairs were fixed before any "
    "test score was computed; choices_made_after_looking_at_test_scores lists anything noted after test scores had been seen (--note=...; --rescore and --extend carry it over), and nothing was changed because of them. "
    f"The weight of any single comparison is small: Wilcoxon p describes {n_b} neighbouring, dependent rivers and is not corrected for the many comparisons made; with {n_b} rivers the smallest p is 0.0005.",
    f"Timing. The laptop was shared with other jobs (mean 1-minute load average at the validation checks of the Mac trainings: {load_note:.1f}; the Windows trainings of seeds 3 and 4, 30 October 2026, record no load average); every training ran under flood.cpu_lock ({waited_all:.0f} s of waiting over all passes). "
    "training seconds therefore describe this machine on this day, not the cost of a training. passes lists each pass, the first run (stopped by the harness at its 2-hour limit before it wrote a results file) included, "
    "but not the --extend run of 30 October 2026 that a low-memory guard stopped after its first training (lstm63_wq seed 3, saved complete; log results/logs/17_more_basins_extend_2026-10-30.log); "
    "runtime_s leaves out the waiting for the lock and the passes that only rescored.",
    f"Prediction sets: {', '.join(SET)} (seed averages) and one per seed (_s<k>); each holds the 12 study rivers only. Networks: results/models/<set>_s<k>.pt, plain dicts that torch.load(weights_only=True) reads "
    "(weights, settings, scalers including the properties', the training rivers in training order, the property values, the split, the training record); reload_check rebuilds each from its file alone.",
    f"Controls: lstm_wq and lstm_w (seed averages; their single seeds _s0 to _s2) of 04_lstm_all.py, and lobo_w_static, lobo_w_nostatic, lobo_ref_w_nostatic, lobo_rf_w of 05_lobo.py, were read from their prediction files and re-scored with the code used for the new sets; "
    f"checks_against_stored_results shows they reproduce the numbers their own scripts stored (largest difference {STORED_CHECKS['max_abs_difference']:.1e}).",
    ("Independent review. A second AI agent recomputed the numbers of this script with its own code (results/review_more_basins.json). "
     f"Round 1 ({RR17[0]['when']}; {RR17[0]['text']}) covers {RR17[0]['covers']}: its must-fix items were wording and records (the onsets of the 63-river gauge model were called a clear loss, and this list was not carried over "
     "by --rescore), and the file was rewritten after it on 7 October by --rescore with those items and the should-fix items applied (the wording of the onset result, of the arm with five times the update limit, of the never-seen "
     "comparison and of the proximity statement; the single trainings, the paired test, the first run in passes), with no training. "
     + (f"Seeds 3 and 4 of every set ({n34_} trainings) were added after round 1, on 30 October 2026 on the Windows PC, by --extend; " if n34_ else "")
     + ("; ".join(f"round {r_['round']} ({r_['when']}; {r_['text']}) covers {r_['covers']}" for r_ in RR17[1:]) + ". " if RR17[1:] else "the review covers seeds 0-2 only. " if n34_ else "")
     + f"This file was written by {written_by_}: {ic['files_compared']} saved prediction sets and networks have the sha256 recorded by the earlier pass (all equal: {ic['all_equal']}). "
     "It is the work of an AI agent, not of a team member: the team's own check (team.json) is still open.") if rev_c else
    "Independent review: results/review_more_basins.json was not found; the work order requires a recomputation of the headline numbers by a second agent (sections 3 and 6)."]
if F.QUICK:
    notes.append("SMOKE TEST (FLOOD_QUICK=1): tiny training budgets and two seeds; numbers are meaningless.")
if RESCORE:
    notes.append("This file was written by --rescore: every score, table, note and the figure were recomputed from the saved prediction sets and networks without training again; the training records come from the network files; "
                 "passes lists each pass.")
NOT_DONE = ([] if EQUAL else [
    f"The arm with five times the update limit and the patience window (lstm63eq_wq: {EQ}x, with the gauge, three seeds) was left out of this pass (--no-equal) because the laptop was too busy for its three long trainings: "
    "'python code/17_more_basins.py --extend' trains them and rewrites this file and the figure with the arm included."]) + [
    "The team's own check of E3 (team.json) is still open; the independent review by a second AI agent exists (see the note 'Independent review').",
    "The simple-model yardstick (ridge regression, random forest) at 63 rivers (and 50): only the stored leave-one-out forest lobo_rf_w (11 training rivers) is shown for scale; this experiment changes the training set, not the input.",
    "The 13_uncertainty.py ranges (test years and rivers drawn again) for the new sets; the seed ranges and the checks of gain_rule are all this file holds.",
    "The 63-river model without basin properties (lstm63n_w), the like-for-like in-training partner of the weather-only never-seen variant: the partner used is the 12-river lobo_ref_w_nostatic (one training), so that comparison changes the training set (12 -> 63 rivers) as well as the 12's presence.",
    "A buffered leave-out (train on the 50 minus every river within 50 and 100 km of the one held out) or a dose-response design (random subsets of 11, 25 and 50 rivers): they would separate the number of training rivers from their variety and nearness; "
    "this file holds only the split of the 12 by distance (unseen.by_distance) and the rank correlations (unseen.gain_over_the_05_models_against_distance).",
    f"Equal training per river at the kept network: the arm with five times the update limit did not test it (its kept networks lie where those of lstm63_wq do); it needs about {n_equal_fixed:.0f} updates, fixed rather than early-stopped.",
    "No more than 63 rivers: the 516 CAMELS basins with a complete flow record, and the gap handling the trainer would need for the others, are not tried (ROADMAP.md, step 3).",
    "Hidden size and learning rate were not tuned again for the larger training set; the arm with five times the update limit is the only change of budget and was run with the gauge only.",
    "The documents: no report subsection, no row in the key figure, no update of PLAN.md or README.md (NEXT_STEPS_FOR_AGENTS.md, sections 3 and 6): documents must quote outcome_full, not outcome alone.",
] + (["QUICK smoke test: two seeds, 40 updates per training: nothing here is a result."] if F.QUICK else [])


# ---- 7. the results file ---------------------------------------------------------------------------------------------------------------------
this_pass["runtime_s"] = round(time.time() - t_start - waited, 1)
training = [dict(set=n, seed=s, n_training_rivers=REC[(n, s)]["n_train_rivers"], **{c: REC[(n, s)][c] for c in COST}, validation_nse_of_the_12_at_that_update=val12(EV[n]["seeds_val"][i]),
                 reached_the_update_limit=bool(REC[(n, s)]["updates_run"] >= SET[n]["conf"]["updates"]), saved_forecasts=f"{n}_s{s}", model_file=REC[(n, s)]["model_file"],
                 mean_load_average=REC[(n, s)]["mean_load_average"], torch_threads=REC[(n, s)]["torch_threads"], curve=REC[(n, s)]["curve"]) for n in SET for i, s in enumerate(SEEDS)]
timing = dict(n_trainings=len(REC), training_seconds=round(sum(REC[k]["seconds"] for k in REC), 1), waited_for_cpu_s=round(sum(p["waited_for_cpu_s"] for p in passes), 1),
              runtime_s=round(sum(p["runtime_s"] for p in passes if p["what"] != "rescore"), 1), passes=passes)
if RESCORE:
    timing["rescore_runtime_s"] = round(time.time() - t_start, 1)
if sum(p["n_trainings"] for p in passes) < len(REC):
    notes.append(f"passes accounts for {sum(p['n_trainings'] for p in passes)} of the {len(REC)} trainings: the others were made by a run that was stopped before it wrote a results file, so their time is in training_seconds "
                 "(the sum over the network files) but not in runtime_s or waited_for_cpu_s; the logs results/logs/17_more_basins.log (the runs of 7 October) and results/logs/17_more_basins_extend_2026-10-30.log "
                 "(the --extend run of 30 October 2026 that was stopped after one training, lstm63_wq seed 3) hold their history.")
R = dict(
    script=SCRIPT, question="Does training on 63 rivers instead of 12 give better forecasts for the same 12 rivers, and for rivers never seen?", quick_mode=F.QUICK, split=F.SPLIT,
    outcome=outcome, outcome_caveat=outcome_caveat, outcome_full=outcome_full,
    design=dict(rivers=dict(twelve=ids12, sixty_three=ids63, other_51=ids51, other_50=ids50,
                            dropped_for_the_never_seen_test=dict(gauge_id=NESTED, inside=NESTED_IN, why="its basin lies inside the basin of " + NESTED_IN + " (outline overlap, see outlines): for that river the 51-river design is not a river never seen",
                                                                 shared_area=NEST_ROW, other_shared_areas=OTHER_OVERLAPS)),
                config=conf, config_equal_training=conf_eq, equal_training_arm_in_this_file=EQUAL, config_source="results/lstm_all.json, key config (chosen by 04_lstm_all.py on the validation years)", seeds=SEEDS, high_flow_level="observed 95th percentile of the training years of each of the 12 (flood.thresholds), scoring only",
                sets={n: dict(saved_as=n, label=v["what"], trained_on_rivers=v["n_train"], river_gauge=v["use_flow"], basin_properties=bool(v["static"]), config=v["conf"], control=v["control"], seeds=SEEDS,
                              scored_on="the 12 study rivers only", saved_networks=[rel(MODELS / f"{n}_s{s}.pt") for s in SEEDS]) for n, v in SET.items()},
                stored_sets_read=dict(zip(["lstm_wq", "lstm_w", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w", "lstm_wqf"],
                                          ["04: 12 rivers, gauge (control of lstm63_wq and lstm63eq_wq)", "04: 12 rivers, no gauge (control of lstm63_w)", "05: held out, 11 training rivers, properties (one training per river)",
                                           "05: held out, 11 training rivers, no properties", "05: 12 rivers in training, no properties", "05: held-out random forest", "04: perfect weather forecast, an upper bound that cannot be had in practice"]))),
    data_checks=DATA_CHECKS, outlines=OUTLINES, proximity=PROXIMITY, checks_against_stored_results=STORED_CHECKS,
    summary_test=dict(how_to_read="test years (2005-10-01 to 2014-09-30), the 12 rivers, seed average of each set (the stored seed average for the stored sets), warning level = the observed high-flow level; onset and all = warning tables pooled over the 12 rivers",
                      sets={n: {lk(k): row(EV[n]["test"], k) for k in F.LEADS} for n in ALL}),
    validation_used_for_no_choice=dict(label="USED FOR NO CHOICE. Validation years (2000-2005), shown for reference only: the settings came from 04_lstm_all.py and every training stopped on its own validation score "
                                             "(of its training rivers), so these scores flatter the models. The one thing chosen on the validation years here is the warning level of two of the three verdict readings.",
                                       summary={n: {lk(k): row(EV[n]["val"], k) for k in F.LEADS} for n in ALL}),
    seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per set and lead (by_seed: the single values, in the order of the seeds); a set of 05_lobo.py has one training only", sets=ranges),
    per_river=dict(how_to_read="test years, one value per river; the seed average of each set (flood.evaluate); gauge ids are text", rivers={g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids12},
                   nse={n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["nse"] for g in ids12} for k in F.LEADS} for n in ALL},
                   nse_by_seed={n: {lk(k): {str(s): {g: e["basins"][g][lk(k)]["nse"] for g in ids12} for s, e in zip(SEEDS, EV[n]["seeds"])} for k in F.LEADS} for n in SET},
                   onset_warned={n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids12} for k in F.LEADS} for n in ALL},
                   all_days_warned={n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["all"]["pod"] for g in ids12} for k in F.LEADS} for n in ALL}),
    gain_rule=gain_rule, gap_to_the_upper_bound=dict(how_to_read="share of the gap between lstm_wq and the perfect-weather-forecast model lstm_wqf (an upper bound that cannot be had in practice) that more rivers close", sets=gap_to_upper_bound),
    stopping=dict(how_to_read="updates run and the best update per training, the sequences each river got (updates x batch / training rivers), and the 12's validation NSE at the chosen update (information only)",
                  updates_for_equal_sequences_per_river_at_63_rivers=EQUAL_SEQUENCES_UPDATES, sets=STOPPING, ratios=STOPPING_RATIO, check_12_river_controls_against_the_trainers_score_max_abs_difference=STOPPING_CHECK),
    verdicts=dict(criteria=CRIT, criteria_source="results/flood_analysis.json, verdict.criteria", levels=VERDICT_NOTE, sets=verdicts),
    unseen=UNSEEN, leak_probes=dict(training_ignores_the_12=probe_train, properties_replaced_by_the_median_of_the_50=PROBE_PROPS,
                                   how_to_read="information only: training_ignores_the_12 compares three 100-update trainings on the 50 (two clean, one with the 12's records spoiled); properties_replaced_by_the_median_of_the_50 forecasts the 12 again "
                                               "from the saved networks with every one of the 12 given the median property values of the 50"),
    training=training, reload_check=RELOAD, integrity=INTEGRITY, findings=findings, notes=notes, what_was_not_done=NOT_DONE,
    independent_review=dict(file="results/review_more_basins.json", **{f"round_{r_['round']}": dict(when=r_["when_full"], covers=r_["covers"], counts=r_["counts"], verdict=r_["verdict"]) for r_ in RR17}) if REVIEW else None,
    choices_made_after_looking_at_test_scores=CHOICES, **timing)
_py = lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)


def clean(o, count=None):
    """Strict JSON: numpy values become Python ones, and a number that is not finite (NaN: a share whose denominator is zero, such as the share of false warnings of a model that gave no warning) becomes None."""
    if isinstance(o, dict):
        return {k: clean(v, count) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, count) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist(), count)
    if isinstance(o, (np.floating, float)):
        if math.isfinite(float(o)):
            return float(o)
        if count is not None:
            count[0] += 1
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


FIVE = five_seed_changes(clean(R), "more_basins")             # what the two extra seeds changed, against results/archive (E3 review round 2, should-fix 4)
R["choices_made_after_looking_at_test_scores"] = [(SEED_NOTE + (" " + FIVE["sentence"] if FIVE else "")) if c.startswith(SEED_NOTE_START) else c for c in R["choices_made_after_looking_at_test_scores"]]
if FIVE:
    R["five_seed_changes"] = FIVE
R["notes"].insert(0, "Quoting. The sentence in `outcome` is accurate arithmetic but incomplete as a statement of the result: any document (report, README, slides, Q&A, guide) must quote `outcome_full` (the outcome plus `outcome_caveat`), "
                     "never `outcome` alone. Both are computed from the tables of this file.")
nulls = [0]
clean(R, nulls)
R["notes"].append(f"{nulls[0]} values in this file are not numbers (NaN: shares whose denominator is zero, or a p-value that is not defined); they are written as null (strict JSON). A reader must treat null as 'no value'.")
JSON_TEXT = json.dumps(clean(R), indent=1, allow_nan=False, default=_py)
(F.RES_OUT / "more_basins.json").write_text(JSON_TEXT)
R = json.loads(JSON_TEXT)                                  # the figure reads exactly what the JSON holds


draw(R)

# ---- 9. what the run prints ----------------------------------------------------------------------------------------------------------------------
print("\n" + R["outcome"] + "\n")
hdr = f"{'set':22s} {'rivers':>6s}  " + "  ".join(f"NSE {k}d [seeds]       onsets {k}d [seeds]" for k in F.LEADS)
print(hdr)
for n in [n for n in ["lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w", "lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic"] if n in R["summary_test"]["sets"]]:
    r = R["summary_test"]["sets"][n]
    g = R["seed_ranges"]["sets"][n]
    nt = {**{k: v["n_train"] for k, v in SET.items()}, "lstm_wq": 12, "lstm_w": 12, "lobo_w_static": 11, "lobo_w_nostatic": 11}[n]
    print(f"{n:22s} {nt:6d}  " + "  ".join(f"{r[lk(k)]['nse_median']:.3f} [{g[lk(k)]['nse_median'][0]:.3f}-{g[lk(k)]['nse_median'][1]:.3f}]  {100 * r[lk(k)]['onset']['pod']:5.1f}% [{100 * g[lk(k)]['onset_pod'][0]:.1f}-{100 * g[lk(k)]['onset_pod'][1]:.1f}]" for k in F.LEADS))
print("\n" + "\n".join(R["findings"]))
print("\nstopping:", {n: (round(v["mean_updates_run"]), round(v["mean_best_update"]), round(v["sequences_per_river_at_the_best_update"])) for n, v in R["stopping"]["sets"].items()}, "(mean updates run, mean best update, sequences per river at the best update)")
print(f"\n{R['n_trainings']} networks, {R['training_seconds']:.0f} s of training, waited {R['waited_for_cpu_s']:.0f} s for the CPU, run time {R['runtime_s']:.0f} s")
print("notes:", *R["notes"], sep="\n  ")
