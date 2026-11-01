"""Week 4 - flood analysis: could the high-flow warnings be trusted, and where and why do they fail?

No training here: the script reads the forecasts saved by 02_baselines.py and 04_lstm_all.py and asks four questions.
A. How many high-flow days does each model warn of 1-3 days ahead, and how many of its warnings are false - for all high-flow
   days and for onsets (forecasts issued while the river was still low), over the 12 basins together and basin by basin?
B. Models trained on squared error forecast peaks too low. Does a lower warning level help? The level (a multiple of the
   high-flow level) is chosen on the validation years and then scored once on the test years.
C. Where and why do the warnings fail: by basin, season, storm type (tropical cyclones from the HURDAT2 best-track record,
   downloaded once to data/hurdat), by the day the rain is filed under, and by the size of the peak?
D. A verdict with criteria fixed in advance: GO, CAUTION or NOT YET.
Writes results/flood_analysis.json, results/flood_events.csv (one row per high-flow event of the test years) and
figures/flood_skill.png, flood_tradeoff.png, flood_failures.png, flood_event.png.
Redraw the figures from the saved files without recomputing: python code/06_flood_analysis.py --figure
"""
import hashlib
import json
import re
import sys
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

import flood as F
from common import *

use_style()
LABEL = {"base_persistence": "'Tomorrow = today'", "base_linreg_wq": "Linear regression, with river gauge", "base_rf_w": "Random forest, past weather only",
         "base_rf_wq": "Random forest, with river gauge", "base_rf_wqf": "Random forest, river gauge + perfect weather forecast",
         "lstm_w": "LSTM, past weather only (no gauge)", "lstm_wq": "LSTM, with river gauge", "lstm_wf": "LSTM, past weather + perfect weather forecast",
         "lstm_wqf": "LSTM, river gauge + perfect weather forecast"}
FAMILY = {n: ("persistence" if "persistence" in n else n.split("_")[1] if n.startswith("base") else "lstm") for n in LABEL}
SERVICE = {"lstm_wq": "what could run today with a river gauge", "lstm_w": "no river gauge", "lstm_wqf": "the ceiling: river gauge + a perfect weather forecast"}
SHOWN = ["base_persistence", "base_rf_wq", "lstm_wq", "lstm_wqf"]                    # the four models drawn in the figures
CSV_MODELS = ["lstm_wq", "lstm_wqf", "base_persistence"]
GRID = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.10 if F.QUICK else 0.05)]         # warning level as a multiple of the high-flow level
KINDS, COUNTS = ("all", "onset"), ("hits", "misses", "false_alarms", "correct_neg")
CRITERIA = {"GO": "pod >= 0.70 and far <= 0.30", "CAUTION": "csi >= 0.40 but not GO", "NOT YET": "otherwise",
            "applied_to": "test years, at the warning level chosen on the validation years; separately for all high-flow days and for onsets; pooled over "
                          "the 12 basins and per basin; lead 1, 2, 3; for lstm_wq, lstm_w and lstm_wqf",
            "definitions": "pod = share of high-flow days that were warned; far = share of warnings that were false; csi = hits / (hits + misses + false alarms)"}
SEASONS = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
STORMS = {"tropical": "Tropical cyclone nearby", "warm-season": "Other, May-Sep", "cool-season": "Other, Oct-Apr"}
WARM = (5, 6, 7, 8, 9)
RADIUS_KM, DAYS_BEFORE = 500, 3                                                    # a cyclone counts when it was this close on the onset day or the days before
RAIN_BINS = ["0-25%", "25-50%", "50-75%", "75-100%"]
DRY = "under 1 mm on both days"
HURDAT_URL = "https://www.nhc.noaa.gov/data/hurdat/"
lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"
pct = lambda x: "n/a" if x != x else f"{100 * x:.0f}%"
pval = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
vmin = lambda v: float(np.min([x for x in v if x == x], initial=np.inf)) if any(x == x for x in v) else float("nan")       # smallest value that is not NaN
vmax = lambda v: float(np.max([x for x in v if x == x], initial=-np.inf)) if any(x == x for x in v) else float("nan")
pool = lambda tabs: F.rates({c: int(sum(t[c] for t in tabs)) for c in COUNTS})
same = lambda x, y: all(x[c] == y[c] for c in COUNTS)                             # two warning tables hold the same counts (rates may be NaN)
t3 = lambda t: f"warned {pct(t['pod'])}, false {pct(t['far'])}, CSI {t['csi']:.2f}"


# ---- scoring helpers (all counts and rates come from flood.py) --------------------------------------------------------
def load(name):
    """flood.load_preds, but with each array of the file unpacked once instead of once per basin and lead (0.03 s, not 0.8 s)."""
    real = np.load
    np.load = lambda *a, **k: dict(real(*a, **k))
    try:
        return F.load_preds(name)
    finally:
        np.load = real


def score(P, thr, part, trig):
    """Warning tables at one warning level per lead {lead: multiple}: per basin and pooled, as flood.evaluate pools them."""
    B = {g: {lk(k): F.flood_scores(P[g], thr[g], k, part, trig[k]) for k in F.LEADS} for g in P}
    return dict(basins=B, summary={lk(k): {kind: pool([B[g][lk(k)][kind] for g in P]) for kind in KINDS} for k in F.LEADS})


def sweep(P, thr, part):
    """Pooled warning tables for every warning level of GRID: {lead: {kind: [table per level]}}.

    flood.flood_scores takes 2.6 ms per call and a sweep needs 27 000 of them, so the same days are selected once here
    (the selection lines of flood.flood_scores) and counted with flood.contingency; run() asserts that the tables at the
    plain and at the chosen level are identical to the ones flood.flood_scores gives."""
    a, b = F.SPLIT[part]
    out = {}
    for k in F.LEADS:
        cells = []
        for g, p in P.items():
            low, q = (p.obs.shift(k) <= thr[g]).loc[a:b].values, p.loc[a:b]
            assert q[lk(k)].notna().all() and q.obs.notna().all(), "a model has days without a forecast"
            cells.append(((q.obs > thr[g]).values, q[lk(k)].values, low, thr[g]))
        out[lk(k)] = dict(all=[pool([F.contingency(o, f > m * t) for o, f, w, t in cells]) for m in GRID],
                          onset=[pool([F.contingency(o[w], (f > m * t)[w]) for o, f, w, t in cells]) for m in GRID])
    return out


def choose(tabs):
    """Index of the level with the best pooled CSI; a tie goes to the level closest to the high-flow level itself."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(GRID)[best] - 1.0))])


def by_class(P, thr, lab, classes, lead, m):
    """Warning tables of the test years for each class of target day (lab: {gauge: labels by day}), pooled over basins.

    Days of other classes get no forecast, so flood.flood_scores leaves them out; the river level on the issue day is kept."""
    out = {}
    for c in classes:
        t = [F.flood_scores(P[g].assign(**{lk(lead): P[g][lk(lead)].where((lab[g] == c).values)}), thr[g], lead, "test", m) for g in P]
        out[c] = {kind: pool([x[kind] for x in t]) for kind in KINDS}
    return out


def verdict(t):
    """The project's rule (CRITERIA) applied to one warning table."""
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= 0.70 and t["far"] <= 0.30:                # far is NaN when no warning was given: then pod is 0 and this is False
        return "GO"
    return "CAUTION" if t["csi"] >= 0.40 else "NOT YET"


def analyse(P, thr, full=True):
    """One prediction set: plain rule, the validation-chosen warning level, and (full) the whole trade-off on the test years."""
    test = F.evaluate(P, thr, "test") if full else score(P, thr, "test", {k: 1.0 for k in F.LEADS})       # single seeds: warning tables only
    assert not full or all(test["summary"][lk(k)]["coverage_min"] == 1.0 for k in F.LEADS), "models are compared on identical days only"
    assert all(p.loc[F.SPLIT["val"][0]:F.SPLIT["test"][1]].notna().all().all() for p in P.values()), "a model has days without a forecast"
    sv = sweep(P, thr, "val")
    st = sweep(P, thr, "test") if full else None
    i1 = GRID.index(1.0)
    rec = dict(plain=dict(trigger=1.0, test=test, val={lk(k): {kind: sv[lk(k)][kind][i1] for kind in KINDS} for k in F.LEADS}))
    for crit, name in (("all", "tuned"), ("onset", "tuned_for_onsets"))[:2 if full else 1]:
        pick = {k: choose(sv[lk(k)][crit]) for k in F.LEADS}
        trig = {k: GRID[i] for k, i in pick.items()}
        s = score(P, thr, "test", trig)
        for k in F.LEADS if full else ():                   # the fast sweep must agree with the engine, count by count
            assert all(same(st[lk(k)][kind][pick[k]], s["summary"][lk(k)][kind]) for kind in KINDS), "sweep and flood.flood_scores disagree"
            assert all(same(st[lk(k)][kind][i1], test["summary"][lk(k)][kind]) for kind in KINDS), "sweep and flood.evaluate disagree"
        rec[name] = {lk(k): dict(trigger=trig[k], chosen_by=f"best pooled validation-years CSI of {'all high-flow days' if crit == 'all' else 'onsets'}",
                                 at_grid_edge=pick[k] in (0, len(GRID) - 1), val={kind: sv[lk(k)][kind][pick[k]] for kind in KINDS},
                                 test=s["summary"][lk(k)], basins={g: s["basins"][g][lk(k)] for g in P}) for k in F.LEADS}
    if full:
        rec["tradeoff_test"] = {lk(k): {kind: {c: [t[c] for t in st[lk(k)][kind]] for c in ("pod", "far", "csi")} for kind in KINDS} for k in F.LEADS}
        rec["csi_validation"] = {lk(k): {kind: [t["csi"] for t in sv[lk(k)][kind]] for kind in KINDS} for k in F.LEADS}
    return rec


# ---- tropical cyclones -------------------------------------------------------------------------------------------------
def hurdat(notes):
    """6-hourly positions of the Atlantic best-track record (HURDAT2), downloaded once and cached under data/hurdat."""
    folder = DATA / "hurdat"
    folder.mkdir(exist_ok=True)
    stamp = lambda n: (int(n[1]), int(n[2][-2:]) + 2000 if len(n[2]) == 6 else int(n[2][-4:]), int(n[2][:2]), int(n[2][2:4]))        # (last season, release y, m, d)
    rx = re.compile(r"(hurdat2-1851-(\d{4})-(\d{6,8})\.txt)")
    have = sorted({m.groups() for f in folder.glob("hurdat2-1851-*.txt") for m in [rx.search(f.name)] if m}, key=stamp)
    if not have:
        try:
            import requests
            names = sorted(set(rx.findall(requests.get(HURDAT_URL, timeout=60).text)), key=stamp)
            r = requests.get(HURDAT_URL + names[-1][0], timeout=180)
            r.raise_for_status()
            (folder / names[-1][0]).write_bytes(r.content)
            have = [names[-1]]
        except Exception as e:                              # no network: the season proxy alone
            notes.append(f"HURDAT2 could not be downloaded ({type(e).__name__}): no event is tagged 'tropical'; storm type is the season proxy alone.")
            return None, None
    f = folder / have[-1][0]
    rows, cur = [], None
    for line in f.read_text().splitlines():
        c = [x.strip() for x in line.split(",")]
        if re.fullmatch(r"AL\d{6}", c[0]):
            cur = (c[0], c[1])
        elif len(c) > 6:
            rows.append((cur[0], cur[1], pd.Timestamp(f"{c[0]} {c[1]}"), c[3], float(c[4][:-1]) * (1 if c[4][-1] == "N" else -1),
                         float(c[5][:-1]) * (-1 if c[5][-1] == "W" else 1), int(c[6])))
    H = pd.DataFrame(rows, columns=["storm_id", "name", "time_utc", "status", "lat", "lon", "wind_kt"])
    info = dict(file=f"data/hurdat/{f.name}", url=HURDAT_URL + f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), n_storms_in_file=int(H.storm_id.nunique()),
                last_season_in_file=int(H.time_utc.dt.year.max()), radius_km=RADIUS_KM, days_before_onset=DAYS_BEFORE)
    return H, info


def km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine, Earth radius 6371 km)."""
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2))


def cyclone_near(H, lat, lon, day, radius=RADIUS_KM):
    """The closest best-track position within the radius on `day` or the DAYS_BEFORE days before it (UTC dates), or None."""
    if H is None:
        return None
    h = H[(H.time_utc >= day - pd.Timedelta(days=DAYS_BEFORE)) & (H.time_utc < day + pd.Timedelta(days=1))]
    if h.empty:
        return None
    dist = km(lat, lon, h.lat.values, h.lon.values)
    i = int(np.argmin(dist))
    return dict(name=h.name.iloc[i], storm_id=h.storm_id.iloc[i], km=float(dist[i]), status=h.status.iloc[i]) if dist[i] <= radius else None


# ---- the analysis --------------------------------------------------------------------------------------------------------
def run():
    t0 = time.perf_counter()
    notes, D, B = [], F.load_data(), F.basins()
    ids, thr = list(D), F.thresholds(D)
    a, b = F.SPLIT["test"]
    names = [n for n in LABEL if F.have_preds(n)]
    if len(names) < len(LABEL):
        notes.append("Prediction sets not found and left out: " + ", ".join(n for n in LABEL if n not in names) + ".")
    P = {n: {g: p for g, p in load(n).items() if g in ids} for n in names}
    assert all(list(p) == ids for p in P.values()) and all(np.allclose(P[n][g].obs, D[g].q_mm) for n in names for g in ids), "every set must hold the same observed flow"
    print(f"{len(names)} prediction sets, {len(GRID)} warning levels from {GRID[0]} to {GRID[-1]} x the high-flow level", flush=True)

    # ---- A + B: warning skill at the plain level and at the validation-chosen level; seed spread ----------------------------
    S, seeds = {}, {}
    for n in names:
        S[n] = analyse(P[n], thr)
        if FAMILY[n] == "lstm":
            seeds[n] = [s for s in F.preds_meta(n).get("seeds", []) if F.have_preds(f"{n}_s{s}")]
            per = {s: analyse(load(f"{n}_s{s}"), thr, full=False) for s in seeds[n]}
            sp = {}
            for k in F.LEADS:
                rows = {"plain": {s: per[s]["plain"]["test"]["summary"][lk(k)] for s in per}, "tuned": {s: per[s]["tuned"][lk(k)]["test"] for s in per}}
                sp[lk(k)] = {rule: {kind: {f"{c}_{w}": fn([r[s][kind][c] for s in per]) for c in ("pod", "far", "csi") for w, fn in (("min", vmin), ("max", vmax))}
                                    | {"by_seed": {str(s): {c: r[s][kind][c] for c in ("pod", "far", "csi")} for s in per}} for kind in KINDS} for rule, r in rows.items()}
                sp[lk(k)]["tuned"]["trigger_by_seed"] = {str(s): per[s]["tuned"][lk(k)]["trigger"] for s in per}
                sp[lk(k)]["tuned"]["verdict_by_seed"] = {kind: {str(s): verdict(per[s]["tuned"][lk(k)]["test"][kind]) for s in per} for kind in KINDS}
            S[n]["seeds"] = dict(seeds=seeds[n], note="each single seed chooses its own warning level on the validation years", **sp)
        print(f"  {n:17s} 1 day ahead, test years: plain level {t3(S[n]['plain']['test']['summary']['l1']['all'])} | level x{S[n]['tuned']['l1']['trigger']:.2f} "
              f"{t3(S[n]['tuned']['l1']['test']['all'])} | onsets {t3(S[n]['tuned']['l1']['test']['onset'])}", flush=True)
    table = [dict(model=n, label=LABEL[n], lead=k, kind=kind, rule=rule, trigger=1.0 if rule == "plain" else S[n][rule][lk(k)]["trigger"],
                  **{c: (S[n]["plain"]["test"]["summary"] if rule == "plain" else {lk(k): S[n][rule][lk(k)]["test"]})[lk(k)][kind][c] for c in ("pod", "far", "csi", "bias", "hits", "misses", "false_alarms")})
             for n in names for k in F.LEADS for kind in KINDS for rule in ("plain", "tuned", "tuned_for_onsets")]

    # ---- event catalogue: consecutive test-year days above the high-flow level -------------------------------------------------
    H, hinfo = hurdat(notes)
    if H is not None:
        H = H[(H.time_utc >= pd.Timestamp(a) - pd.Timedelta(days=DAYS_BEFORE + 1)) & (H.time_utc <= pd.Timestamp(b) + pd.Timedelta(days=1))]
    trig = {n: {k: S[n]["tuned"][lk(k)]["trigger"] for k in F.LEADS} for n in names}
    ev, storm_lab, rain_lab, season_lab = [], {}, {}, {}
    for g in ids:
        d, lat, lon = D[g], B.gauge_lat[g], B.gauge_lon[g]
        hi = d.q_mm > thr[g]
        first = (hi & ~hi.shift(1, fill_value=False)).loc[a:b]
        q, h = d.q_mm.loc[a:b], hi.loc[a:b]
        run_id = (h != h.shift(1, fill_value=False)).cumsum()[h]
        share = d.prcp / (d.prcp + d.prcp.shift(1))                              # of the rain of issue day + target day, the part on the target day
        both = d.prcp + d.prcp.shift(1)
        rain_lab[g] = pd.Series(np.where(both < 1.0, DRY, np.array(RAIN_BINS)[np.clip((share.fillna(0) * 4).astype(int), 0, 3)]), index=d.index)
        season_lab[g] = pd.Series(d.index.month.map({m: s for s, ms in SEASONS.items() for m in ms}), index=d.index)
        lab = pd.Series(np.where(d.index.month.isin(WARM), "warm-season", "cool-season"), index=d.index)
        if H is not None:                                                         # days with a cyclone position within the radius, and the DAYS_BEFORE days after each
            close = H.time_utc[km(lat, lon, H.lat.values, H.lon.values) <= RADIUS_KM].dt.normalize().unique()
            lab.loc[lab.index.isin({t + pd.Timedelta(days=j) for t in close for j in range(DAYS_BEFORE + 1)})] = "tropical"
        for _, days_ in q[h].groupby(run_id):
            on, pk, end = days_.index[0], days_.idxmax(), days_.index[-1]
            tc = cyclone_near(H, lat, lon, on)
            kind = "tropical" if tc else ("warm-season" if on.month in WARM else "cool-season")
            lab.loc[on:end] = kind                                                # every day of an event carries the tag of its onset
            row = dict(gauge_id=g, basin=F.SHORT[g], landscape=F.LAND_SHORT[B.landscape[g]], onset_date=on.date().isoformat(), peak_date=pk.date().isoformat(),
                       end_date=end.date().isoformat(), duration_days=len(days_), peak_flow_mm_day=float(days_.max()), high_flow_level_mm_day=thr[g],
                       peak_over_level=float(days_.max() / thr[g]), started_before_test=not bool(first.loc[on]), season=season_lab[g][on], storm_type=kind,
                       cyclone_name=tc["name"] if tc else "", cyclone_id=tc["storm_id"] if tc else "", cyclone_km=round(tc["km"]) if tc else np.nan,
                       cyclone_status=tc["status"] if tc else "", cyclone_within_300km=bool(tc and cyclone_near(H, lat, lon, on, 300)),
                       rain_issue_day_mm=float(d.prcp.shift(1)[on]), rain_target_day_mm=float(d.prcp[on]),
                       rain_share_target_day=float(share[on]) if both[on] >= 1.0 else np.nan, rain_class=rain_lab[g][on])
            for k in F.LEADS:
                row[f"river_low_at_issue_l{k}"] = bool(d.q_mm.shift(k)[on] <= thr[g])
                for n in CSV_MODELS:
                    if n in P:
                        row[f"{n}_warned_l{k}"] = bool(P[n][g][lk(k)][on] > thr[g])
                        row[f"{n}_warned_l{k}_chosen_level"] = bool(P[n][g][lk(k)][on] > trig[n][k] * thr[g])
            for n in names:
                row[f"peak_ratio_l1_{n}"] = float(P[n][g].l1[pk] / days_.max())
            ev.append(row)
        storm_lab[g] = lab
    E = pd.DataFrame(ev)
    E.to_csv(F.RES_OUT / "flood_events.csv", index=False, float_format="%.4g")
    O = E[~E.started_before_test]                                               # real onsets: the river was low the day before
    for n in [m for m in CSV_MODELS if m in P]:                                 # the catalogue must agree with the engine's onset table
        t = S[n]["plain"]["test"]["summary"]["l1"]["onset"]
        assert len(O) == t["hits"] + t["misses"] and int(O[f"{n}_warned_l1"].sum()) == t["hits"], "event catalogue and flood.evaluate disagree"
    print(f"{len(E)} high-flow events in the test years ({len(O)} onsets), {int((E.storm_type == 'tropical').sum())} with a tropical cyclone nearby", flush=True)

    # ---- C: where and why it fails ---------------------------------------------------------------------------------------
    focus = [n for n in SERVICE if n in P]
    def split3(lab, classes):                                                    # plain level and validation-chosen level (computed once when they are the same)
        out = {}
        for n in focus:
            out[n] = dict(plain=by_class(P[n], thr, lab, classes, 1, 1.0))
            out[n]["tuned"] = out[n]["plain"] if trig[n][1] == 1.0 else by_class(P[n], thr, lab, classes, 1, trig[n][1])
        return out
    by_season, by_storm, by_rain = split3(season_lab, SEASONS), split3(storm_lab, STORMS), split3(rain_lab, RAIN_BINS + [DRY])
    for n in focus:                                                              # the classes split the engine's table exactly
        for part in (by_season, by_storm, by_rain):
            assert sum(t["onset"]["hits"] for t in part[n]["plain"].values()) == S[n]["plain"]["test"]["summary"]["l1"]["onset"]["hits"]
            assert sum(t["all"]["false_alarms"] for t in part[n]["plain"].values()) == S[n]["plain"]["test"]["summary"]["l1"]["all"]["false_alarms"]

    # what a perfect weather forecast would add, onset by onset (1 day ahead)
    rescue = {}
    if {"lstm_wq", "lstm_wqf"} <= set(P):
        for rule, suf in (("plain", ""), ("tuned", "_chosen_level")):
            x, y = O[f"lstm_wq_warned_l1{suf}"], O[f"lstm_wqf_warned_l1{suf}"]
            cnt = lambda m: dict(onsets=int(m.sum()), caught_with_gauge=int((x & m).sum()), caught_with_perfect_forecast=int((y & m).sum()),
                                 missed_with_gauge=int((~x & m).sum()), of_those_caught_with_perfect_forecast=int((~x & y & m).sum()),
                                 missed_by_both=int((~x & ~y & m).sum()), caught_with_gauge_but_not_with_forecast=int((x & ~y & m).sum()))
            rescue[rule] = dict(all_onsets=cnt(pd.Series(True, index=O.index)), by_rain_class={c: cnt(O.rain_class == c) for c in RAIN_BINS + [DRY]},
                                by_storm_type={c: cnt(O.storm_type == c) for c in STORMS})

    # by basin, 1 day ahead: scores against basin properties
    gis_file = F.RES / "basin_gis.json"
    gis = json.loads(gis_file.read_text())["basins"] if gis_file.exists() else {}
    if not gis:
        notes.append("results/basin_gis.json was not found: the basin comparison uses the CAMELS attributes only.")
    ATTR = {"area_km2": ("basin area (km2, CAMELS)", B.area_gages2), "slope_m_per_km": ("mean slope (m/km, CAMELS)", B.slope_mean),
            "baseflow_index": ("baseflow index (share of flow that is slow groundwater flow, CAMELS)", B.baseflow_index), "elev_mean_m": ("mean elevation (m, CAMELS)", B.elev_mean)}
    if gis:
        ATTR |= {"gis_slope_m_per_km": ("mean slope from the 30 m elevation grid (m/km, 07_basin_maps.py)", pd.Series({g: gis[g]["slope_mean_m_per_km"] for g in ids})),
                 "gis_forest_share": ("forest + woody wetland share (NLCD 2011, 07_basin_maps.py)", pd.Series({g: gis[g]["forest_plus_woody_wetland_share"] for g in ids})),
                 "gis_elongation_ratio": ("elongation ratio of the outline (1 = round, 07_basin_maps.py)", pd.Series({g: gis[g]["elongation_ratio"] for g in ids}))}
    ATTR["event_days_mean"] = ("mean length of a high-flow event in the test years (days; from the flow record itself)", E.groupby("gauge_id").duration_days.mean())
    by_basin = {}
    for n in [m for m in ("lstm_wq", "lstm_w", "lstm_wqf") if m in P]:
        rows = []
        for g in ids:
            r = dict(gauge_id=g, basin=F.SHORT[g], landscape=B.landscape[g], n_events=int((E.gauge_id == g).sum()), nse_l1=S[n]["plain"]["test"]["basins"][g]["l1"]["nse"],
                     **{k_: float(v[g]) for k_, (_, v) in ATTR.items()})
            for rule, t in (("plain", S[n]["plain"]["test"]["basins"][g]["l1"]), ("tuned", S[n]["tuned"]["l1"]["basins"][g])):
                r |= {f"{rule}_{kind}_{c}": t[kind][c] for kind in KINDS for c in ("pod", "far", "csi", "hits", "misses", "false_alarms")}
            rows.append(r)
        T = pd.DataFrame(rows)
        rho = {}
        for rule in ("plain", "tuned"):
            rho[rule] = {}
            for k_ in ATTR:
                rho[rule][k_] = {}
                for sc in [f"{kind}_{c}" for kind in KINDS for c in ("pod", "far", "csi")]:
                    ok = T[f"{rule}_{sc}"].notna()
                    r_ = spearmanr(T.loc[ok, k_], T.loc[ok, f"{rule}_{sc}"]) if T.loc[ok, f"{rule}_{sc}"].nunique() > 1 else None      # no rank correlation with a constant
                    rho[rule][k_][sc] = dict(rho=float(r_.statistic) if r_ else float("nan"), p=float(r_.pvalue) if r_ else float("nan"), n=int(ok.sum()))
        land = {rule: {F.LAND_SHORT[l]: {sc: float(grp[f"{rule}_{sc}"].median()) for sc in ("all_pod", "all_far", "all_csi", "onset_pod", "onset_far", "onset_csi")}
                       for l, grp in T.groupby("landscape")} for rule in ("plain", "tuned")}
        by_basin[n] = dict(lead=1, trigger_tuned=trig[n][1], table=rows, spearman=rho, median_by_landscape=land)
    n_tests = len(ATTR) * 6
    strong = {n: {rule: [f"{k_} ~ {sc}: rho {v['rho']:+.2f}" for k_, d_ in by_basin[n]["spearman"][rule].items() for sc, v in d_.items() if abs(v["rho"]) >= 0.587]
                  for rule in ("plain", "tuned")} for n in by_basin}

    # peak size: forecast for the peak day made the day before, as a share of the observed peak
    peak = {}
    for n in names:
        r, later = O[f"peak_ratio_l1_{n}"], O.peak_date != O.onset_date
        st = lambda x: dict(n=int(len(x)), median=float(x.median()), q25=float(x.quantile(.25)), q75=float(x.quantile(.75)), share_below_1=float((x < 1).mean()),
                            share_below_half=float((x < .5).mean())) if len(x) else dict(n=0)
        peak[n] = dict(all_events=st(r), peak_on_onset_day=st(r[~later]), peak_after_onset_day=st(r[later]),
                       by_storm_type={c: st(r[O.storm_type == c]) for c in STORMS},
                       forecast_below_high_flow_level_on_peak_day=float((r * O.peak_flow_mm_day < O.high_flow_level_mm_day).mean()))

    # ---- D: verdict ------------------------------------------------------------------------------------------------------
    V, vrows = {}, []
    for n in [m for m in SERVICE if m in P]:
        V[n] = dict(what=SERVICE[n], label=LABEL[n])
        for k in F.LEADS:
            t, s1 = S[n]["tuned"][lk(k)], S[n]["plain"]["test"]
            V[n][lk(k)] = dict(trigger=t["trigger"])
            for kind in KINDS:
                pb = {g: verdict(t["basins"][g][kind]) for g in ids}
                sd = S[n].get("seeds", {}).get(lk(k), {}).get("tuned", {}).get("verdict_by_seed", {}).get(kind, {})
                V[n][lk(k)][kind] = dict(pooled=verdict(t["test"][kind]), pod=t["test"][kind]["pod"], far=t["test"][kind]["far"], csi=t["test"][kind]["csi"],
                                         basins=pb, basin_count={v: sum(x == v for x in pb.values()) for v in ("GO", "CAUTION", "NOT YET", "NO EVENTS")},
                                         single_seeds=sd, at_plain_level=verdict(s1["summary"][lk(k)][kind]),
                                         at_level_chosen_for_onsets=verdict(S[n]["tuned_for_onsets"][lk(k)]["test"][kind]))
                vrows.append(dict(model=n, label=LABEL[n], lead=k, kind=kind, trigger=t["trigger"], pod=t["test"][kind]["pod"], far=t["test"][kind]["far"],
                                  csi=t["test"][kind]["csi"], verdict=V[n][lk(k)][kind]["pooled"], **{f"basins_{v.lower().replace(' ', '_')}": c for v, c in V[n][lk(k)][kind]["basin_count"].items()}))
    reference = {n: {lk(k): {kind: verdict(S[n]["tuned"][lk(k)]["test"][kind]) for kind in KINDS} for k in F.LEADS} for n in names if n not in V}

    # ---- the event shown in flood_event.png, picked by rule ----------------------------------------------------------------------
    shown = None
    if {"lstm_wq", "lstm_wqf"} <= set(P):
        pattern = ~O.lstm_wq_warned_l1_chosen_level & O.lstm_wqf_warned_l1_chosen_level      # the commonest way an onset goes: missed without, caught with the forecast
        c = O[pattern & (O.duration_days >= 3) & O.cyclone_within_300km]
        c = c if len(c) else O[pattern & (O.duration_days >= 3)]
        c = c if len(c) else O
        e = c.sort_values("peak_over_level", ascending=False).iloc[0]
        shown = dict(gauge_id=e.gauge_id, basin=e.basin, onset_date=e.onset_date, peak_date=e.peak_date, end_date=e.end_date, duration_days=int(e.duration_days), storm_type=e.storm_type,
                     cyclone_name=e.cyclone_name, cyclone_km=None if e.cyclone_km != e.cyclone_km else float(e.cyclone_km), cyclone_status=e.cyclone_status,
                     rain_issue_day_mm=float(e.rain_issue_day_mm), rain_target_day_mm=float(e.rain_target_day_mm), high_flow_level_mm_day=float(e.high_flow_level_mm_day),
                     trigger={n: trig[n][1] for n in ("lstm_wq", "lstm_wqf")},
                     rule="among the test-year onsets that the LSTM with river gauge missed 1 day ahead and the LSTM with a perfect weather forecast caught (validation-chosen warning levels), "
                          "lasting at least 3 days and with a tropical cyclone within 300 km: the one with the highest peak relative to the basin's high-flow level. An illustration picked by rule, "
                          "not a measure of typical skill.",
                     onsets_with_this_pattern=int(pattern.sum()), onsets=int(len(O)))

    # ---- findings in plain sentences (every number is read from the tables above) -------------------------------------------------
    find = [f"{len(E)} high-flow events in the test years over the 12 basins ({int((E.duration_days == 1).sum())} last a single day; median length {E.duration_days.median():.0f} days); "
            f"{int((E.storm_type == 'tropical').sum())} began with a tropical cyclone within {RADIUS_KM} km ({int(E.cyclone_within_300km.sum())} within 300 km), "
            f"{int((E.storm_type == 'warm-season').sum())} others in May-Sep and {int((E.storm_type == 'cool-season').sum())} in Oct-Apr."]
    for n in [m for m in ("base_persistence", "base_rf_wq", "lstm_w", "lstm_wq", "lstm_wqf") if m in P]:
        for kind, word in (("all", "all high-flow days"), ("onset", "onsets")):
            p_, t_ = S[n]["plain"]["test"]["summary"]["l1"][kind], S[n]["tuned"]["l1"]
            sd = S[n].get("seeds", {}).get("l1", {})
            spread = f" (single seeds: warned {pct(sd['plain'][kind]['pod_min'])} to {pct(sd['plain'][kind]['pod_max'])}, false {pct(sd['plain'][kind]['far_min'])} to {pct(sd['plain'][kind]['far_max'])})" if sd else ""
            find.append(f"{LABEL[n]}, 1 day ahead, {word}, test years: at the high-flow level itself {t3(p_)}{spread}; at the validation-chosen level "
                        f"(x{t_['trigger']:.2f}) {t3(t_['test'][kind])}.")
    for n in [m for m in ("lstm_wq", "lstm_w") if m in P]:
        tr, i7 = S[n]["tradeoff_test"]["l1"]["onset"], int(np.argmin(np.abs(np.array(GRID) - 0.7)))
        sd = S[n]["seeds"]["l1"]["tuned"]["trigger_by_seed"].values() if "seeds" in S[n] else []
        find.append(f"Does a lower warning level help? {LABEL[n]}: the validation years chose x" + " / x".join(f"{trig[n][k]:.2f}" for k in F.LEADS) + " for 1 / 2 / 3 days ahead"
                    + (f" (single seeds 1 day ahead: x{min(sd):.2f} to x{max(sd):.2f})" if sd else "") + f". On the test years (description only) a level of x{GRID[i7]:.2f} would warn of "
                    f"{pct(tr['pod'][i7])} of the onsets 1 day ahead instead of {pct(tr['pod'][GRID.index(1.0)])} at the high-flow level itself, with {pct(tr['far'][i7])} of those warnings false instead of "
                    f"{pct(tr['far'][GRID.index(1.0)])} (CSI of onsets {tr['csi'][i7]:.2f} against {tr['csi'][GRID.index(1.0)]:.2f}).")
    for n in V:
        agree = [all(v == V[n][lk(k)][kind]["pooled"] for v in V[n][lk(k)][kind]["single_seeds"].values()) for k in F.LEADS for kind in KINDS]
        find.append(f"Verdict for {LABEL[n]} ({SERVICE[n]}), test years at the validation-chosen level: " + "; ".join(
            f"{days(k)} - all high-flow days {V[n][lk(k)]['all']['pooled']} ({V[n][lk(k)]['all']['basin_count']['GO']} basins GO, {V[n][lk(k)]['all']['basin_count']['CAUTION']} CAUTION, "
            f"{V[n][lk(k)]['all']['basin_count']['NOT YET']} NOT YET), onsets {V[n][lk(k)]['onset']['pooled']} ({V[n][lk(k)]['onset']['basin_count']['GO']} GO, {V[n][lk(k)]['onset']['basin_count']['CAUTION']} CAUTION, "
            f"{V[n][lk(k)]['onset']['basin_count']['NOT YET']} NOT YET)" for k in F.LEADS) + f". Each single seed gives the same pooled verdict in {sum(agree)} of the {len(agree)} cases.")
    if "base_persistence" in reference and "lstm_wq" in V:
        x, y = S["base_persistence"]["tuned"]["l1"]["test"]["all"], S["lstm_wq"]["tuned"]["l1"]["test"]["all"]
        find.append(f"The same rule applied to 'tomorrow = today', 1 day ahead, all high-flow days: {reference['base_persistence']['l1']['all']} (CSI {x['csi']:.2f} against {y['csi']:.2f} for the LSTM with river gauge). "
                    f"{x['hits'] + x['misses'] - len(O)} of the {x['hits'] + x['misses']} high-flow days are the second or later day of an event, when the gauge already shows the river is high; "
                    f"on the {len(O)} onsets 'tomorrow = today' warns of {pct(S['base_persistence']['tuned']['l1']['test']['onset']['pod'])} and the LSTM with river gauge of {pct(S['lstm_wq']['tuned']['l1']['test']['onset']['pod'])}.")
    if "lstm_wq" in by_season:
        for part, word in ((by_season, "season"), (by_storm, "storm type"), (by_rain, "share of the two days' rain that the weather table files under the target day")):
            for n in [m for m in ("lstm_wq", "lstm_wqf") if m in part]:
                find.append(f"{LABEL[n]}, onsets 1 day ahead at the validation-chosen level, by {word}: " + "; ".join(
                    f"{c} {pct(v['onset']['pod'])} of {v['onset']['hits'] + v['onset']['misses']} warned ({pct(v['onset']['far'])} of the warnings false)" for c, v in part[n]["tuned"].items()) + ".")
    if "tuned" in rescue:
        x = rescue["tuned"]["all_onsets"]
        find.append(f"What a perfect weather forecast adds, 1 day ahead (validation-chosen levels): of {x['onsets']} onsets the LSTM with river gauge caught {x['caught_with_gauge']} and missed {x['missed_with_gauge']}; "
                    f"the LSTM that also reads the target days' weather caught {x['of_those_caught_with_perfect_forecast']} of the missed ones ({pct(x['of_those_caught_with_perfect_forecast'] / max(x['missed_with_gauge'], 1))}); "
                    f"{x['missed_by_both']} were missed by both and {x['caught_with_gauge_but_not_with_forecast']} were caught only without the forecast.")
    for n in [m for m in ("base_persistence", "lstm_wq", "lstm_wqf") if m in peak]:
        x = peak[n]["all_events"]
        find.append(f"Peak size, {LABEL[n]}: the forecast made the day before the peak is {pct(x['median'])} of the observed peak in the median event (middle half {pct(x['q25'])} to {pct(x['q75'])}; "
                    f"{pct(x['share_below_1'])} of the {x['n']} events are forecast too low; peak on the onset day: median {pct(peak[n]['peak_on_onset_day'].get('median', float('nan')))}, later: "
                    f"{pct(peak[n]['peak_after_onset_day'].get('median', float('nan')))}).")
    for n in [m for m in ("lstm_wq", "lstm_wqf") if m in by_basin]:
        r_ = by_basin[n]["spearman"]["tuned"]
        find.append(f"By basin, {LABEL[n]}, share of onsets warned 1 day ahead against basin properties (Spearman rank correlation, 12 basins): "
                    + "; ".join(f"{ATTR[k_][0].split(' (')[0]} {r_[k_]['onset_pod']['rho']:+.2f} ({pval(r_[k_]['onset_pod']['p'])})" for k_ in ("area_km2", "slope_m_per_km", "baseflow_index", "event_days_mean"))
                    + f". {len(strong[n]['tuned'])} of the {n_tests} correlations looked at for this model reach |rho| >= 0.587 (p < 0.05 in a single test); about {0.05 * n_tests:.0f} would by chance alone.")

    # ---- notes ---------------------------------------------------------------------------------------------------------------
    n_val, n_test = (sum(t[c] for c in ("hits", "misses")) for t in (S[names[0]]["plain"]["val"]["l1"]["all"], S[names[0]]["plain"]["test"]["summary"]["l1"]["all"]))
    edge = [f"{LABEL[n]} {days(k)} (x{S[n]['tuned'][lk(k)]['trigger']:.2f})" for n in names for k in F.LEADS if S[n]["tuned"][lk(k)]["at_grid_edge"]]
    notes += [
        "No model was trained here. All counts and rates come from flood.evaluate, flood.flood_scores, flood.contingency and flood.rates; test years 2005-10-01..2014-09-30, pooled = sums of hits, misses and false alarms over the 12 basins.",
        f"Warning level: a warning is given when the forecast exceeds m x the basin's high-flow level (95th percentile of the training years). m is one number per model and lead for all 12 basins, chosen on the grid "
        f"{GRID[0]:.2f}..{GRID[-1]:.2f} (step {GRID[1] - GRID[0]:.2f}) by the best pooled CSI of all high-flow days in the validation years 2000-2005 ('tuned'); a tie goes to the level closest to 1. 'plain' is m = 1. "
        "'tuned_for_onsets' is a variant in which the validation CSI of onsets chooses m. The test years chose nothing; tradeoff_test is the whole curve on the test years, as a description only.",
        f"The level is chosen on {n_val} high-flow days of 5 validation years (the 9 test years hold {n_test}), so the chosen level is itself uncertain: the single LSTM seeds choose their own level (seeds.l*.tuned.trigger_by_seed) and their range is given next to every LSTM score."
        + (" Chosen level at the edge of the grid (a value outside it might do better): " + "; ".join(edge) + "." if edge else " No chosen level is at the edge of the grid."),
        "The trade-off sweep over warning levels selects the scored days with the same lines as flood.flood_scores and counts them with flood.contingency (27 000 calls of flood.flood_scores would take over a minute); "
        "the script asserts that its tables are identical to flood.evaluate at the plain level and to flood.flood_scores at the chosen level.",
        "Onset = a forecast issued while the river was still at or below the high-flow level (flood.flood_scores). 1 day ahead that is exactly the first day of each event; 2 and 3 days ahead it also includes the second "
        "and third day of an event, so the number of 'onset' days grows with the lead. 'Tomorrow = today' can warn of an onset only when the warning level is below the high-flow level.",
        "Verdict criteria were fixed before the scores were looked at (verdict.criteria). The pooled verdict weights basins by their number of high-flow days; per-basin verdicts rest on few events "
        "(see by_basin.table n_events), so a single event can move a basin across a line.",
        "Event = consecutive test-year days above the high-flow level; a one-day dip below the level ends an event and the next day starts a new one. started_before_test marks an event already running on the first test day "
        f"(it is in the catalogue but not counted as an onset). {int((E.end_date == b).sum())} events were still running on the last test day ({b}): their length, and their peak if it came later, is cut there.",
        "The 'where and why' tables (failures: by basin, season, storm type, rain timing, peak size) are for forecasts made 1 day ahead only; the skill tables and the verdict cover 1, 2 and 3 days ahead.",
        f"Storm type: 'tropical' = a position of the Atlantic best-track record (HURDAT2, 6-hourly, every status including depressions and remnant or extratropical stages) lay within {RADIUS_KM} km of the river gauge "
        f"on the onset day or the {DAYS_BEFORE} days before (UTC dates; the flow day is a local day, a slack the {DAYS_BEFORE}-day window absorbs). That is nearness, not proof that the cyclone caused the rain. "
        "Otherwise 'warm-season' (onset May-Sep) or 'cool-season' (Oct-Apr) - a stated proxy for thunderstorm rain against frontal rain, not a weather classification. cyclone_within_300km in the CSV shows how many tags survive a tighter radius. "
        "For the day-by-day tables every day of an event carries the tag of its onset; days outside events are tagged by the same rule applied to the day itself.",
        "Rain timing, 1 day ahead: rain_share_target_day = basin-mean Daymet rain filed under the target day / (rain filed under the issue day + under the target day); days with under 1 mm in total are a class of their own. "
        "The models without a weather forecast read the weather table up to the issue day only, so a high share means the rise came with rain they could not read.",
        "Caution on 'rain of the target day': Daymet files rain by the rain-gauge observers' day, so part of the rain filed under the target day may already have fallen during the issue day (10_rain_timing.py measures this). "
        "'Not yet in the weather table' is therefore exact; 'not yet fallen' would overstate it, and part of the 1-day-ahead gap could be closed without any forecast by rain measured up to the evening of the issue day.",
        "The perfect weather forecast (lstm_wqf) reads the actual weather of the target days: an upper bound that cannot be had in practice. 'Of those caught with a perfect forecast' is therefore the most a real forecast could add, not what it would add.",
        f"Small groups: the {int((E.storm_type == 'tropical').sum())} onsets with a tropical cyclone nearby are few (one onset moves their warned share by about {100 / max(int((E.storm_type == 'tropical').sum()), 1):.0f} percentage points), "
        "and the cyclone tag overlaps with basin type and season, so the storm-type contrast is a description, not a tested difference.",
        "Peak size: forecast for the peak day issued the day before, divided by the observed peak flow; events already running on the first test day are left out.",
        f"By basin: Spearman rank correlations over 12 basins. With n = 12, |rho| must reach 0.587 for p < 0.05 in a single test, and {n_tests} correlations are looked at for each model and rule, so a few will pass by chance; "
        "basin properties are also tied to each other (the mountain basins are the steep ones with the highest baseflow index). The table can show a pattern worth a hypothesis; it cannot show a cause.",
        "The random forest has one seed (random_state=0), so its seed spread is not known. lstm_<set> is the average forecast of its seeds; its scores can lie outside the range of the single seeds (it tends to give fewer false warnings), so the seed range is given next to every LSTM score.",
        "flood_events.csv: gauge_id is text with a leading zero - read it with dtype=str. <model>_warned_l<k> = the forecast for the onset day issued k days before was above the high-flow level; "
        "..._chosen_level = above the validation-chosen level of that model and lead. river_low_at_issue_l<k> says whether the river was still low when that forecast was issued.",
    ]
    if F.QUICK:
        notes.append("QUICK smoke test: coarser grid of warning levels, and the LSTM prediction sets may be the tiny smoke-test models. Not a result.")
    R = dict(question="Could the high-flow warnings be trusted, and where and why do they fail?", quick_mode=F.QUICK, split=F.SPLIT, high_flow="daily flow above the basin's 95th percentile of the training years",
             thresholds_mm_day=thr, basins={g: dict(short=F.SHORT[g], landscape=B.landscape[g]) for g in ids}, models={n: dict(label=LABEL[n], family=FAMILY[n], seeds=seeds.get(n, [])) for n in names},
             warning_levels=GRID, skill=S, summary_table=table, events=dict(file="results/flood_events.csv", n_events=int(len(E)), n_onsets=int(len(O)), by_basin=E.groupby("basin").size().to_dict(),
             by_storm_type=E.storm_type.value_counts().to_dict(), by_season=E.season.value_counts().to_dict(), tropical_within_300km=int(E.cyclone_within_300km.sum()),
             cyclones=sorted(set(E.cyclone_name[E.cyclone_name != ""])), cyclone_status_at_closest_point=E.cyclone_status[E.cyclone_status != ""].value_counts().to_dict(),
             median_duration_days=float(E.duration_days.median()), one_day_events=int((E.duration_days == 1).sum()), hurdat=hinfo),
             failures=dict(lead=1, model_labels={n: LABEL[n] for n in focus}, by_basin=by_basin, by_season=by_season, by_storm_type=by_storm, by_rain_timing=by_rain,
                           perfect_forecast_rescue=rescue, peak_size=peak, spearman_tests_per_model_and_rule=n_tests, spearman_rho_needed_for_p05=0.587, strong_rank_correlations=strong),
             verdict=dict(criteria=CRITERIA, services=V, table=vrows, other_models_pooled=reference), event_shown=shown, findings=find, notes=notes, runtime_s=round(time.perf_counter() - t0, 1))
    (F.RES_OUT / "flood_analysis.json").write_text(json.dumps(R, indent=1))
    return R


# ---- figures (drawn from the saved JSON) -----------------------------------------------------------------------------------
LOOK = {"base_persistence": dict(color=F.MODEL_COL["persistence"], marker="D", ls="-", label="'Tomorrow = today' (river gauge only)"),
        "base_rf_wq": dict(color=F.MODEL_COL["rf"], marker="s", ls="-", label="Random forest (past weather + river gauge)"),
        "lstm_wq": dict(color=F.MODEL_COL["lstm"], marker="o", ls="-", label="LSTM (past weather + river gauge)"),
        "lstm_wqf": dict(color=F.MODEL_COL["lstm"], marker="o", ls=(0, (4, 2.2)), label="LSTM + perfect weather forecast (upper bound)")}
STATUS = {"HU": "hurricane", "TS": "tropical storm", "TD": "tropical depression", "EX": "former tropical cyclone", "LO": "remnant low", "SS": "subtropical storm", "SD": "subtropical depression"}
PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def finish(fig, name, title, note, handles, below, ncol=2, above=0.58, **adjust):
    """Title, legend and note of a figure, stacked from the bottom edge so that nothing overlaps; `below` = inches the axes need for their tick and axis labels."""
    H = fig.get_figheight()
    lines = textwrap.wrap(note, 134)
    fig.text(0.012, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    y = 0.012 + (len(lines) * 0.178 + 0.07) / H                                  # one note line is 0.178 inch high, one legend row 0.19
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y), columnspacing=1.6, handlelength=2.8, borderaxespad=0, borderpad=0.2)
    y += (-(-len(handles) // ncol) * 0.19 + 0.08) / H
    fig.suptitle(title, fontsize=10.2, fontweight="bold", color=INK, x=0.012, y=1 - 0.09 / H, ha="left", va="top")
    fig.subplots_adjust(bottom=y + below / H, top=1 - above / H, **adjust)
    fig.savefig(F.FIG_OUT / name, dpi=220)
    plt.close(fig)


def model_handles(show, filled=False):
    return [Line2D([], [], color=LOOK[n]["color"], ls=LOOK[n]["ls"], lw=1.8, marker=LOOK[n]["marker"], ms=6.5, markeredgewidth=1.3, markeredgecolor=LOOK[n]["color"] if n == "lstm_wqf" and not filled else SURFACE,
                   markerfacecolor=SURFACE if n == "lstm_wqf" and not filled else LOOK[n]["color"], label=LOOK[n]["label"]) for n in show]


def fig_skill(R):
    """Share of high-flow days warned and share of false warnings by lead, for all high-flow days and for onsets (plain level)."""
    S = R["skill"]
    show = [n for n in SHOWN if n in S]
    title = {("all", "pod"): "All high-flow days: share warned", ("all", "far"): "All high-flow days: share of warnings false",
             ("onset", "pod"): "Onsets only: share warned", ("onset", "far"): "Onsets only: share of warnings false"}
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.5), sharex=True, sharey=True)
    for i, kind in enumerate(KINDS):
        for j, c in enumerate(("pod", "far")):
            ax = axes[i, j]
            ax.axhspan(*((0.70, 1.0) if c == "pod" else (0.0, 0.30)), color=AQUA, alpha=.11, lw=0, zorder=1)
            for n in show:
                L = LOOK[n]
                y = [S[n]["plain"]["test"]["summary"][lk(k)][kind][c] for k in F.LEADS]
                ax.plot(F.LEADS, y, color=L["color"], ls=L["ls"], lw=1.8, marker=L["marker"], ms=7, markeredgewidth=1.3, markerfacecolor=SURFACE if n == "lstm_wqf" else L["color"],
                        markeredgecolor=L["color"] if n == "lstm_wqf" else SURFACE, zorder=4 if n.startswith("lstm") else 3)
                for k in F.LEADS if "seeds" in S[n] else ():                   # how far the single seeds lie apart, drawn beside the marker
                    sd = S[n]["seeds"][lk(k)]["plain"][kind]
                    ax.plot([k + 0.13] * 2, [sd[f"{c}_min"], sd[f"{c}_max"]], color=INK, lw=1.3, marker="_", ms=5, markeredgewidth=1.3, solid_capstyle="butt", zorder=6)
            ax.set_title(title[kind, c], fontsize=9); ax.tick_params(labelsize=8, length=0); ax.grid(axis="x", visible=False)
            ax.set_xlim(0.6, 3.4); ax.set_xticks(F.LEADS, [days(k) for k in F.LEADS]); ax.tick_params(axis="x", labelcolor=INK2)
            ax.set_ylim(-0.03, 1.03); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.yaxis.set_major_formatter(PCT); ax.tick_params(labelleft=True)
        axes[i, 0].set_ylabel("share of the high-flow days" if kind == "all" else "share of the onset days", fontsize=8.5)
        axes[i, 1].set_ylabel("share of the warnings given", fontsize=8.5)
    n_seeds = max(len(R["models"][n]["seeds"]) for n in show)
    h = model_handles(show) + [Line2D([], [], color=INK, lw=1.3, marker="_", ms=5, markeredgewidth=1.3, ls="none", label=f"Range of the {n_seeds} single LSTM seeds"),
                               Patch(facecolor=AQUA, alpha=.25, label="What the GO rule asks for (70% warned, 30% false)")]
    gap = max(abs(S[n]["tuned"][lk(k)]["test"][kind][c] - S[n]["plain"]["test"]["summary"][lk(k)][kind][c]) for n in show for k in F.LEADS for kind in KINDS for c in ("pod", "far")
              if S[n]["plain"]["test"]["summary"][lk(k)][kind][c] == S[n]["plain"]["test"]["summary"][lk(k)][kind][c])
    lev = sorted({S[n]["tuned"][lk(k)]["trigger"] for n in show for k in F.LEADS})
    warned = lambda n, kind, leads: ", ".join(pct(S[n]["plain"]["test"]["summary"][lk(k)][kind]["pod"]) for k in leads)
    note = (f"Test years 2005-2014, {len(R['basins'])} basins together. High flow = above the basin's top-5% level of the training years; a warning = a forecast above that level. "
            "Onset day = the forecast was issued while the river was still below the level. 'Tomorrow = today' can never warn of an onset, so it has no line at bottom right. "
            f"Where markers coincide the LSTM is drawn on top: 1 day ahead 'tomorrow = today' warns of {warned('base_persistence', 'all', [1])} of all high-flow days (LSTM with river gauge "
            f"{warned('lstm_wq', 'all', [1])}), and of the onsets the random forest warns of {warned('base_rf_wq', 'onset', F.LEADS)} (LSTM with river gauge {warned('lstm_wq', 'onset', F.LEADS)}). "
            f"The verdict is taken at the warning levels chosen on the validation years ({lev[0]:.2f} to {lev[-1]:.2f} x the level); with those no point moves by more than "
            f"{np.ceil(100 * gap):.0f} percentage points.")
    finish(fig, "flood_skill.png", "High-flow warnings 1 to 3 days ahead: how many days are warned, how many warnings are false", note, h, below=0.42,
           left=0.095, right=0.985, hspace=0.24, wspace=0.24)


def fig_tradeoff(R):
    """Warned share against false-warning share as the warning level varies, 1 day ahead, with the GO region."""
    S, G = R["skill"], R["warning_levels"]
    show, i1 = [n for n in SHOWN if n in S], R["warning_levels"].index(1.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 5.35), sharex=True, sharey=True)
    far = np.linspace(0, 0.6, 121)
    for ax, kind, ttl in zip(axes, KINDS, ("All high-flow days", "Onsets only")):
        ax.add_patch(Rectangle((0, 0.70), 0.30, 0.30, facecolor=AQUA, alpha=.16, edgecolor="none", zorder=1))
        ax.text(0.02, 0.975, "GO", fontsize=9, fontweight="bold", color=INK, va="top", ha="left", zorder=2)
        ax.plot(far, 1 / (1 / 0.40 + 1 - 1 / (1 - far)), color=INK2, lw=1.1, ls=(0, (1, 2)), zorder=2)        # CSI = 0.40 written as warned share against false share
        for n in show:
            L, t = LOOK[n], S[n]["tradeoff_test"]["l1"][kind]
            x, y, ic = np.array(t["far"], float), np.array(t["pod"], float), G.index(S[n]["tuned"]["l1"]["trigger"])
            ax.plot(x, y, color=L["color"], ls=L["ls"], lw=1.7, zorder=3)
            ax.plot(x[ic], y[ic], ls="none", marker=L["marker"], ms=8.5, color=L["color"], markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=5)
            ax.plot(x[i1], y[i1], ls="none", marker="o", ms=3.4, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=0.9, zorder=6)
            if n == "lstm_wq":                                                   # which end of a curve is which
                ax.text(x[0], y[0] + 0.035, f"{G[0]:.1f} x", fontsize=7.6, color=L["color"], ha="center", va="center", zorder=6)
                ax.text(x[-1] - 0.02, y[-1], f"{G[-1]:.1f} x", fontsize=7.6, color=L["color"], ha="right", va="center", zorder=6)
        ax.set_title(ttl, fontsize=9); ax.tick_params(labelsize=8, length=0)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
        ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("share of the warnings that were false", fontsize=8.5)
    axes[0].set_ylabel("share of the high-flow days that were warned", fontsize=8.5)
    h = model_handles(show, filled=True) + [Line2D([], [], ls="none", marker="o", ms=8, color=INK2, markeredgecolor=SURFACE, label="Warning level chosen on the validation years"),
                                            Line2D([], [], ls="none", marker="o", ms=3.4, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=0.9, label="The high-flow level itself (1.0 x)"),
                                            Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 2)), label="CAUTION needs a point above this line (CSI 0.40)"),
                                            Patch(facecolor=AQUA, alpha=.3, label="GO needs a point in this corner (70% warned, 30% false)")]
    lev = ", ".join(f"{S[n]['tuned']['l1']['trigger']:.2f}" for n in show)
    note = (f"1 day ahead, test years 2005-2014, {len(R['basins'])} basins together. Along each curve the warning level falls from {G[-1]:.1f} x the high-flow level (few warnings) to "
            f"{G[0]:.1f} x (many warnings). CSI = hits / (hits + misses + false warnings). The level chosen on the validation years (best CSI of all high-flow days) is {lev} x for the four models in the order of the legend; "
            "a white dot inside the marker means it is the high-flow level itself. The test-year curves are a description: they chose nothing.")
    finish(fig, "flood_tradeoff.png", "A lower warning level catches more high-flow days, but at the price of more false warnings", note, h, below=0.52,
           left=0.095, right=0.985, wspace=0.07)


def fig_failures(R):
    """Four views of the 1-day-ahead onset warnings: by season, storm type, basin size and rain timing."""
    X = R["failures"]
    pair = [(n, c) for n, c in (("lstm_wq", F.MODEL_COL["lstm"]), ("lstm_wqf", SEQ[1])) if n in X["by_season"]]
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 7.0), sharey=True)

    def bars(ax, T, classes, names):
        w = 0.37
        for j, (n, col) in enumerate(pair):
            t = [T[n]["tuned"][c]["onset"] for c in classes]
            x = np.arange(len(classes)) + (j - (len(pair) - 1) / 2) * w
            ax.bar(x, [v["pod"] for v in t], w * 0.9, color=col, zorder=3)
            ax.hlines([v["far"] for v in t], x - w * 0.45, x + w * 0.45, color=RED, lw=2.2, zorder=4)
        n_on = [T[pair[0][0]]["tuned"][c]["onset"]["hits"] + T[pair[0][0]]["tuned"][c]["onset"]["misses"] for c in classes]
        ax.set_xticks(range(len(classes)), [f"{nm}\n{k} onsets" for nm, k in zip(names, n_on)]); ax.grid(axis="x", visible=False); ax.set_xlim(-0.6, len(classes) - 0.4)

    bars(axes[0, 0], X["by_season"], list(SEASONS), list(SEASONS)); axes[0, 0].set_title("By season", fontsize=9)
    bars(axes[0, 1], X["by_storm_type"], list(STORMS), ["Tropical cyclone\nnearby", "Other rain,\nMay-Sep", "Other rain,\nOct-Apr"]); axes[0, 1].set_title("By storm type", fontsize=9)
    bars(axes[1, 1], X["by_rain_timing"], RAIN_BINS, RAIN_BINS); axes[1, 1].set_title("By rain timing", fontsize=9)
    axes[1, 1].set_xlabel("share of the two days' rain stamped on the target day\n(not yet in the rain table when the forecast is issued)", fontsize=8.5)

    ax, mark, rho = axes[1, 0], dict(zip(F.LAND_COL, ("^", "o", "s"))), []
    for n, col in pair:
        T = pd.DataFrame(X["by_basin"][n]["table"])
        for land, grp in T.groupby("landscape"):
            ax.plot(grp.area_km2, grp.tuned_onset_pod, ls="none", marker=mark[land], ms=7, color=col, markeredgecolor=F.MODEL_COL["lstm"], markeredgewidth=0.9, zorder=5 if n == "lstm_wq" else 4, clip_on=False)
        rho.append(X["by_basin"][n]["spearman"]["tuned"]["area_km2"]["onset_pod"]["rho"])
    if len(pair) == 2:
        A, Bq = (pd.DataFrame(X["by_basin"][n]["table"]).set_index("gauge_id") for n, _ in pair)
        ax.vlines(A.area_km2, A.tuned_onset_pod, Bq.tuned_onset_pod, color=AXIS, lw=1, zorder=2)
    ax.set_xscale("log"); ax.set_xticks([100, 200, 500, 1000, 2000], ["100", "200", "500", "1000", "2000"]); ax.minorticks_off(); ax.set_xlim(85, 2600)
    ax.set_xlabel("basin area (km2, log scale)", fontsize=8.5); ax.set_title("By basin size", fontsize=9)
    for a in axes.ravel():
        a.set_ylim(0, 1.0); a.set_yticks(np.arange(0, 1.01, 0.2)); a.yaxis.set_major_formatter(PCT); a.tick_params(labelsize=8, length=0); a.tick_params(axis="x", labelcolor=INK2)
    for a in axes[:, 0]:
        a.set_ylabel("onsets warned 1 day ahead", fontsize=8.5)
    h = [Patch(facecolor=pair[0][1], label="LSTM with river gauge: onsets warned")]
    h += [Patch(facecolor=pair[1][1], label="LSTM + perfect weather forecast: onsets warned")] if len(pair) == 2 else []
    h += [Line2D([], [], color=RED, lw=2.2, label="Share of that model's onset warnings that were false")]
    h += [Line2D([], [], ls="none", marker=m, ms=6.5, markerfacecolor=SURFACE, markeredgecolor=INK2, label=f"Basin in the {F.LAND_SHORT[l].lower()}") for l, m in mark.items()]
    dry = X["by_rain_timing"][pair[0][0]]["tuned"][DRY]["onset"]
    slope = X["by_basin"][pair[0][0]]["spearman"]["tuned"]["slope_m_per_km"]["onset_pod"]["rho"]
    note = (f"Test years 2005-2014, {R['events']['n_onsets']} onsets (first day of a high-flow event) in {len(R['basins'])} basins; 1 day ahead; warning levels chosen on the validation years. "
            f"Tropical cyclone nearby = a best-track position (HURDAT2) within {RADIUS_KM} km of the gauge on the onset day or the {DAYS_BEFORE} days before. "
            f"Rain timing leaves out {dry['hits'] + dry['misses']} onsets with under 1 mm on both days. Basin size: one marker per basin and model; rank correlation of the warned share with area "
            f"{rho[0]:+.2f} (river gauge)" + (f" and {rho[1]:+.2f} (perfect forecast)" if len(rho) == 2 else "") + f", with slope {slope:+.2f} (river gauge); only 12 basins.")
    finish(fig, "flood_failures.png", "Onset warnings fail when the rain behind the rise is not yet in the model's rain table", note, h, below=0.78,
           left=0.095, right=0.985, hspace=0.46, wspace=0.06)


def fig_event(R):
    """One test-year event, day by day: rain, observed flow, the two 1-day-ahead forecasts and how each day's warning turned out."""
    e = R["event_shown"]
    g, thr = e["gauge_id"], e["high_flow_level_mm_day"]
    on, end = pd.Timestamp(e["onset_date"]), pd.Timestamp(e["end_date"])
    w = slice(on - pd.Timedelta(days=11), end + pd.Timedelta(days=5))
    d = F.load_data()[g].loc[w]
    fc = {n: load(n)[g].l1.loc[w] for n in ("lstm_wq", "lstm_wqf")}
    hi, half = (d.q_mm > thr).values, pd.Timedelta(hours=12)
    fig = plt.figure(figsize=(6.7, 6.6))
    ax_r, ax_q, ax_s = fig.add_gridspec(3, 1, height_ratios=[0.85, 3.2, 0.95], hspace=0.13).subplots(sharex=True)
    ax_r.bar(d.index, d.prcp, width=0.8, color=SEQ[2], zorder=3)
    ax_r.set_ylabel("rain\n(mm per day)", fontsize=8.5); ax_r.set_ylim(0, max(d.prcp.max() * 1.15, 1))
    for ax in (ax_r, ax_q, ax_s):
        ax.axvspan(on - half, end + half, color=YELLOW, alpha=.13, lw=0, zorder=1)
        ax.tick_params(labelsize=8, length=0); ax.grid(axis="x", visible=False)
    ax_q.axhline(thr, color=RED, lw=1.2, ls=(0, (5, 2)), zorder=2)
    ax_q.text(d.index[-1] + half, thr, f"high-flow level\n{thr:.1f} mm per day ", fontsize=7.8, color=RED, ha="right", va="bottom", linespacing=1.1)
    for n in fc:
        if e["trigger"][n] != 1.0:                                                # a chosen warning level that is not the high-flow level itself
            ax_q.axhline(e["trigger"][n] * thr, color=LOOK[n]["color"], lw=0.9, ls=(0, (1, 2)), zorder=2)
    ax_q.plot(d.index, d.q_mm, color=INK, lw=2, marker="o", ms=3.6, zorder=5, label="Observed river flow")
    ax_q.plot(d.index, fc["lstm_wq"], color=LOOK["lstm_wq"]["color"], lw=1.7, marker="o", ms=4.2, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4,
              label="Forecast made the day before: LSTM with river gauge")
    ax_q.plot(d.index, fc["lstm_wqf"], color=LOOK["lstm_wqf"]["color"], lw=1.6, ls=LOOK["lstm_wqf"]["ls"], marker="o", ms=4.2, markerfacecolor=SURFACE, markeredgewidth=1.1, zorder=4,
              label="Forecast made the day before: LSTM + perfect weather forecast")
    top = max(d.q_mm.max(), fc["lstm_wq"].max(), fc["lstm_wqf"].max())
    ax_q.set_ylim(0, top * 1.36); ax_q.set_ylabel("river flow (mm per day over the basin)", fontsize=8.5)
    ax_q.legend(loc="upper right", fontsize=8, frameon=False, borderaxespad=0.2, handlelength=2.6)
    ax_q.annotate(f"Onset on {on.day} {on:%B}: {e['rain_target_day_mm']:.0f} mm of rain is\nstamped on that day - not yet in the inputs\nof the forecast made the evening before\n({e['rain_issue_day_mm']:.0f} mm the day before).",
                  xy=(on, d.q_mm[on]), xytext=(d.index[0] - half * 0.6, top * 0.62), fontsize=7.8, color=INK, ha="left", va="center", linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9, shrinkA=3, shrinkB=5, relpos=(1.0, 0.3)), zorder=7)
    LOOKS = {"hit": dict(marker="o", color=AQUA, ms=7), "miss": dict(marker="X", color=RED, ms=7.5), "false alarm": dict(marker="^", color=YELLOW, ms=7.5)}
    counts = {}
    for row, n in ((1, "lstm_wq"), (0, "lstm_wqf")):
        warn = (fc[n] > e["trigger"][n] * thr).values
        counts[n] = F.contingency(hi, warn)
        ax_s.plot(d.index, np.full(len(d), row), ls="none", marker="o", ms=1.8, color=AXIS, zorder=2)
        for what, m in (("hit", hi & warn), ("miss", hi & ~warn), ("false alarm", ~hi & warn)):
            ax_s.plot(d.index[m], np.full(m.sum(), row), ls="none", markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3, **LOOKS[what])
    ax_s.set_ylim(-0.6, 1.6); ax_s.set_yticks([1, 0], ["with river\ngauge", "+ perfect\nforecast"]); ax_s.tick_params(axis="y", labelsize=8, labelcolor=INK2); ax_s.grid(visible=False)
    ax_s.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3)); ax_s.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax_s.set_xlim(d.index[0] - half * 1.4, d.index[-1] + half * 1.4); ax_s.set_xlabel(f"{on.year}", fontsize=8.5)
    h = [Line2D([], [], ls="none", markeredgecolor=SURFACE, label={"hit": "Hit: high-flow day, warned", "miss": "Miss: high-flow day, no warning", "false alarm": "False warning: warned, flow stayed below"}[k], **v) for k, v in LOOKS.items()]
    h += [Patch(facecolor=YELLOW, alpha=.3, label="The high-flow event (days above the level)")]
    a_, b_ = counts["lstm_wq"], counts["lstm_wqf"]
    say = lambda t: f"{t['hits']} hit{'s' * (t['hits'] != 1)}, {t['misses']} miss{'es' * (t['misses'] != 1)} and {t['false_alarms']} false warning{'s' * (t['false_alarms'] != 1)}"
    rule = "the high-flow level" if set(e["trigger"].values()) == {1.0} else "the warning level chosen on the validation years (dotted lines: " + ", ".join(f"{m:.2f}" for m in e["trigger"].values()) + " x the high-flow level)"
    note = (f"Every forecast is for the day it is drawn on and was made the evening before. Warning = forecast above {rule}. In this window the LSTM with river gauge scores "
            f"{say(a_)}; with a perfect weather forecast {say(b_)}. An illustration picked by rule, not a typical score: {e['onsets_with_this_pattern']} of the {e['onsets']} onsets "
            "of the test years went this way (missed with the river gauge alone, caught with a perfect weather forecast).")
    storm = f" - {STATUS.get(e['cyclone_status'], 'tropical cyclone')} {e['cyclone_name'].title()}" if e["cyclone_name"] else ""
    finish(fig, "flood_event.png", f"One high-flow event day by day: {e['basin']}{storm}, {on:%B %Y}", note, h, below=0.5, above=0.4, left=0.115, right=0.985)


if __name__ == "__main__":
    R = json.loads((F.RES_OUT / "flood_analysis.json").read_text()) if "--figure" in sys.argv else run()
    R = json.loads(json.dumps(R))                                               # the figures read exactly what the JSON holds
    if all(n in R["skill"] for n in SHOWN):
        fig_skill(R); fig_tradeoff(R); fig_failures(R); fig_event(R)
    else:
        print("figures were not drawn: they need the prediction sets " + ", ".join(SHOWN))
    print("\n".join(R["findings"]))
    print("notes:", *R["notes"], sep="\n  ")
    print(f"runtime {R['runtime_s']} s")
