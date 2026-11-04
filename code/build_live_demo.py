"""The live demo: an interactive replay of the stored test-year forecasts, day by day, on one self-contained page -> live_demo/index.html
and live_demo/facts.json (the numbers it prints). Open the page in a browser; it needs no server and loads nothing from the network.

What the page shows:
  1. the numbers of the final results files: the four levels of information replayed from their stored test-year forecasts (each the
     average of its trainings, five or three as the results files record), the pooled scores of the whole test period, and the fixed
     wordings of the numbers registry (results/final_report_numbers.json: fw_aorc, fw_hindsight, fw_realtime_short, fw_upper) beside them;
  2. at every flood onset, 1 day ahead, with the rain to midnight: whether the same saved networks warned with the real-time radar
     product (Stage IV) in place of AORC (results/stage4_realtime.json -> per_onset);
  3. at every onset, when its rain fell relative to the issue time: in the rain table by the issue time, fallen before midnight but filed
     under the next day, still to come. The windows follow code/check_onset_rain_clock.py (results/onset_rain_clock.csv, whose 611
     onsets are asserted to be the page's own), except that midnight is the wall clock's, as the issue time and the model's rain to
     midnight are (code/15_rain_to_midnight.py); 09:00, the rain table's day end, stays in standard time. The rain is the hourly
     basin-mean AORC that the model reads, not that file's ERA5 at the basin centre, because the story of item 4 tells the night in
     AORC numbers and the two products differ much on some nights (Little River, 21 November 2006 before midnight: AORC 35.7 mm, ERA5
     11.0 mm);
  4. a guided story of one night (Little River, 21-22 November 2006), its numbers computed here and its choice disclosed on the page.

Nothing on the page is typed by hand. The build recounts every pooled number of the page from the stored forecasts with the rule the
page itself applies (a day is high when the flow is above the flood level, a warning when the forecast is; an onset is a high day while
the river was at or below the level `lead` days before; a false warning is a warning on a day that stays low while the river was low at
the issue), and asserts it against the results files. The page draws rounded values (hundredths of a mm), but it decides above / below
the flood level from flags computed here from the exact values (rounding alone would move a few borderline days); it joins the two
per-onset tables to the replay's own onsets and asserts that they are the same onsets with the same warned / missed; it asserts the facts
of the story; and it decodes the data it ships and compares them with what it checked. The fixed wordings and a few quoted figures come
from results/final_report_numbers.json. A changed results file fails the build instead of printing a stale number.

Inputs: results/*.json, results/onset_rain_clock.csv and results/flood_events.csv (the repository); results/preds/*.npz (release v1.0,
flood-onset-lstm_preds.zip);
data/derived/camels12_daymet.pkl, data/gefs/basins/*.csv and data/aorc/aorc_<gauge>.npz (release v1.0, flood-onset-lstm_data_caches.zip).
CAMELS-US itself is not needed. The build is tied to the released forecasts: after a retraining, its assertions (the story's night among
them) stop it on purpose.
Template: code/proposal_web/replay.html (the replay screen of the proposal talk, code/proposal_web/deck5.html, alone on the page).
Look at it: python code/proposal_web/shots_replay.py live_demo/index.html <outdir> [dark]

Run from the repository root:  python code/build_live_demo.py
Written 2026-11-04 by Claude Code (Opus 5.5) under the team lead's direction; reviewed by independent AI reviewer agents.
"""
import base64
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES, DATA, PRED = ROOT / "results", ROOT / "data", ROOT / "results" / "preds"
TEMPLATE, ASSETS = HERE / "proposal_web" / "replay.html", HERE / "proposal_web" / "assets"
OUT = ROOT / "live_demo" / "index.html"

TEST = ("2005-10-01", "2014-09-30")
TRAIN = ("1980-10-01", "2000-09-30")
FROM = "2005-09-28"                      # three days before the test years: the issue-day flow of the first targets (the page reads obs[t - lead])
LEADS = (1, 2, 3)
# the four levels of information of the replay: key, stored forecasts, results file and its set, label, caption ({...} = a registry wording)
LEVELS = [("wq", "lstm_wq", "aorc_rain_to_midnight", "wq", "Rain table + river gauge", "past weather, gauge, basin properties"),
          ("wqa", "lstm_wqa", "aorc_rain_to_midnight", "wqa", "+ rain to midnight", "AORC, {fw_aorc}"),
          ("wqaF", "lstm_wqaF_sub", "forecast_input", "wqaF_sub", "+ weather forecast of the next 3 days", "GEFSv12 at test time (trained on AORC rain)"),
          ("wqf", "lstm_wqf", "aorc_rain_to_midnight", "wqf", "+ perfect weather of the next 3 days", "{fw_upper}")]
SHORT = {"wq": "the rain table", "wqa": "the rain to midnight", "wqaF": "the weather forecast", "wqf": "perfect weather"}    # the levels inside a sentence
STORY = ("02128000", "2006-11-22")        # the onset of the talk's cover; asserted below to be the one the cover's rule picks
REVIEWERS = "independent AI reviewer agents"                                    # the start of the registry's fw_review
ICONS = {"i-db": "database", "i-grid": "table-2", "i-brain": "brain", "i-split": "split", "i-shield": "shield-check"}


def J(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def registry():
    """The fixed wordings and numbers of the final report and deck: results/final_report_numbers.json (code/final_report/numbers.py)."""
    R = J("final_report_numbers")

    def text(key):
        v = R[key]
        assert isinstance(v, dict) and v.get("text"), key
        return v["text"]
    return text


# ---- the same small helpers as code/build_proposal_web.py (the talk), kept here so the demo does not import the talk's deck modules
def b64_u16(a, scale=100.0):
    """A float array (mm/day) -> base64 of little-endian uint16 in hundredths, NaN -> 65535."""
    x = np.asarray(a, float)
    v = np.where(np.isnan(x), 65535, np.clip(np.round(x * scale), 0, 65534)).astype("<u2")
    return base64.b64encode(v.tobytes()).decode("ascii")


def b64_bits(mask):
    """A boolean array -> base64 of its bits, little-endian within each byte (day i = bit i % 8 of byte i // 8)."""
    return base64.b64encode(np.packbits(np.asarray(mask, bool).astype(np.uint8), bitorder="little").tobytes()).decode("ascii")


def above(x, thr):
    """x > thr on the exact values; NaN counts as not above (the page skips NaN days on its own)."""
    x = np.asarray(x, float)
    return np.where(np.isnan(x), False, x > thr)


def font_b64():
    """Inter (variable, latin subset; SIL Open Font License 1.1, assets/LICENSE-Inter.txt) as base64 woff2."""
    b = (ASSETS / "inter-latin-wght-normal.woff2").read_bytes()
    assert b[:4] == b"wOF2"
    return base64.b64encode(b).decode("ascii")


def icon_symbols():
    """The Lucide icons (ISC licence, assets/LICENSE-Lucide.txt) as <symbol> elements with the ids the template uses."""
    out = ["<!-- Icons: Lucide (https://lucide.dev), ISC licence; see code/proposal_web/assets/LICENSE-Lucide.txt -->", '<svg style="display:none" aria-hidden="true">']
    for sid, name in ICONS.items():
        t = (ASSETS / "icons" / f"{name}.svg").read_text(encoding="utf-8")
        inner = re.sub(r"\s+", " ", re.search(r"<svg[^>]*>(.*)</svg>", t, re.S).group(1).strip())
        out.append(f'  <symbol id="{sid}" viewBox="0 0 24 24">{inner}</symbol>')
    return "\n".join(out + ["</svg>"])


def colour_literals(tpl):
    """Colour literals of the template outside its two token blocks (:root and :root[data-theme="dark"]): every colour goes through a token."""
    css = tpl[tpl.index("<style>"):tpl.index("</style>")]
    dark = re.search(r':root\[data-theme="dark"\]\{[^}]*\}', css)
    assert dark, "the template has no dark token block"
    return re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)", css[dark.end():] + tpl[tpl.index("</style>"):])


# ---- the data of the replay ------------------------------------------------------------------------------------------------------
def stored():
    """Per basin, on the days of the replay (three days before the test years to their end): the observed flow, the rain table (Daymet),
    the AORC rain to midnight of the day, the GEFS forecast rain of the next three days, and the stored forecasts of the four levels at
    the three leads, by TARGET day as the engine stores them. Floats here; the page gets them in hundredths (b64_u16)."""
    FA, DA = J("flood_analysis"), J("data_audit")
    info = {s["gauge_id"]: s for s in DA["selected"]}
    Z = {k: np.load(PRED / f"{f}.npz", allow_pickle=False) for k, f, *_ in LEVELS}
    basins = [str(g) for g in Z["wq"]["basins"]]
    dates = pd.DatetimeIndex(Z["wq"]["dates"])
    assert all([str(g) for g in z["basins"]] == basins and (z["dates"] == Z["wq"]["dates"]).all() for z in Z.values()), "the stored sets do not share one index"
    sel = (dates >= FROM) & (dates <= TEST[1])
    days = dates[sel]
    assert len(days) == 3290 and str(days[3].date()) == TEST[0], len(days)
    DM = pickle.load(open(DATA / "derived" / "camels12_daymet.pkl", "rb"))     # the engine's daily tables, parsed once from CAMELS-US
    out = []
    for i, g in enumerate(basins):
        d = DM[g]
        assert d.index.equals(dates), g
        obs = Z["wq"]["obs"][i][sel].astype(float)
        assert np.allclose(obs, d.q_mm.values[sel], equal_nan=True), f"{g}: the stored obs differ from the daily table"
        assert all(np.allclose(z["obs"][i][sel], obs, equal_nan=True) for z in Z.values()), f"{g}: the stored sets disagree on the obs"
        gf = pd.read_csv(DATA / "gefs" / "basins" / f"gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
        # AORC rain of local day t = aorc_1 of the issue day t-1 (the table is by issue day: aorc_k = AORC rain of the local day t+k)
        aorc = gf["aorc_1"].shift(1, freq="D").reindex(days).values
        fc = [gf[f"fc_bc_{k}"].reindex(days).values for k in LEADS]       # the forecast rain of day t+k, issued on day t (bias corrected, the input of E2a)
        out.append(dict(id=g, name=FA["basins"][g]["short"], landscape=info[g]["landscape"], area=round(float(info[g]["area_gages2"])),
                        thr=float(FA["thresholds_mm_day"][g]), obs=obs, rain=d.prcp.values[sel].astype(float), aorc=aorc, fc=fc,
                        pred={k: Z[k]["pred"][i][sel].astype(float) for k, *_ in LEVELS}, train_obs=d.q_mm.loc[TRAIN[0]:TRAIN[1]].values))
        b = out[-1]
        b["hi"] = above(b["obs"], b["thr"])                                       # the river above its flood level
        b["wn"] = {k: [above(b["pred"][k][:, l], b["thr"]) for l in range(3)] for k, *_ in LEVELS}     # a warning, by target day and lead
    return days, out


def classify(obs, p, thr, L, test):
    """The page's rule, on one basin: masks of hits, misses and false warnings over the test target days."""
    was = np.r_[np.full(L, np.nan), obs[:-L]]
    ok = test & ~np.isnan(obs) & ~np.isnan(p) & ~np.isnan(was) & (was <= thr)
    hi, wn = obs > thr, p > thr
    return ok & hi & wn, ok & hi & ~wn, ok & ~hi & wn


def classify_flags(obs, p, hi, wn, L, test):
    """The same rule as the page applies it: validity from the values (NaN = missing), above / below from the flags."""
    vo, vp = ~np.isnan(obs), ~np.isnan(p)
    was_ok, was_hi = np.r_[np.zeros(L, bool), vo[:-L]], np.r_[np.zeros(L, bool), hi[:-L]]
    ok = test & vo & vp & was_ok & ~was_hi
    return ok & hi & wn, ok & hi & ~wn, ok & ~hi & wn


def check_shipped(data, B):
    """The page's DATA, decoded as the page decodes it, equals what the build checked: bits (np.packbits little = the JS bit()) and the
    hundredths (uint16, 65535 = missing) of every array."""
    u16 = lambda s: np.frombuffer(base64.b64decode(s), dtype="<u2")
    bits = lambda s, n: np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8), bitorder="little")[:n].astype(bool)
    hund = lambda x: np.where(np.isnan(np.asarray(x, float)), 65535, np.clip(np.round(np.asarray(x, float) * 100), 0, 65534)).astype(int)
    for s, b in zip(data["basins"], B):
        n = len(b["obs"])
        assert (bits(s["hi"], n) == b["hi"]).all(), (b["id"], "hi")
        for k in s["wn"]:
            for l in range(3):
                assert (bits(s["wn"][k][l], n) == b["wn"][k][l]).all(), (b["id"], k, l, "wn")
                assert (u16(s["pred"][k][l]).astype(int) == hund(b["pred"][k][:, l])).all(), (b["id"], k, l, "pred")
        for key in ("obs", "rain", "aorc"):
            assert (u16(s[key]).astype(int) == hund(b[key])).all(), (b["id"], key)
        for l in range(3):
            assert (u16(s["fc"][l]).astype(int) == hund(b["fc"][l])).all(), (b["id"], "fc", l)


def nse(obs, p, test):
    m = test & ~np.isnan(obs) & ~np.isnan(p)
    o, f = obs[m], p[m]
    return 1 - ((f - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def pooled(days, B):
    """The pooled scores of the whole test period per level and lead: read from the results files, and recounted twice from the stored
    forecasts: with the engine's rule on the exact floats, and with the page's rule on the flags it ships (hi, wn). They must agree. That the
    shipped base64 decodes back to these flags and values is asserted in main() (check_shipped)."""
    test = np.asarray((days >= TEST[0]) & (days <= TEST[1]))
    P, seeds = {}, {}
    for k, _f, src, key, *_ in LEVELS:
        R = J(src)
        S = R["summary_test"]["sets"][key]
        seeds[k] = int(R["seed_ranges"]["sets"][key]["l1"]["n_seeds"])
        P[k] = {}
        for L in LEADS:
            o = S[f"l{L}"]["onset"]
            rec = dict(hits=int(o["hits"]), misses=int(o["misses"]), false=int(o["false_alarms"]), pod=o["pod"], far=o["far"], csi=o["csi"], nse=S[f"l{L}"]["nse_median"])
            for how in ("floats", "page"):        # the engine's floats, and the page's rule (its flags) on the values the page reads
                h = m = fa = 0
                for b in B:
                    obs, p = b["obs"], b["pred"][k][:, L - 1]
                    a, c, e = classify(obs, p, b["thr"], L, test) if how == "floats" else classify_flags(obs, p, b["hi"], b["wn"][k][L - 1], L, test)
                    h, m, fa = h + a.sum(), m + c.sum(), fa + e.sum()
                assert (h, m, fa) == (rec["hits"], rec["misses"], rec["false"]), f"{k} l{L} ({how}): recount {(h, m, fa)} vs {src}.json {(rec['hits'], rec['misses'], rec['false'])}"
            med = float(np.median([nse(b["obs"], b["pred"][k][:, L - 1], test) for b in B]))
            assert abs(med - rec["nse"]) < 1e-9, f"{k} l{L}: median NSE recount {med:.12f} vs {src}.json {rec['nse']:.12f}"
            P[k][f"l{L}"] = rec
    return P, seeds


def aorc_clock(B, keys):
    """The rain clock of each onset from hourly AORC (basin mean); t = the issue day: table 09:00 (t-1) -> 09:00 (t), hidden 09:00 (t) ->
    midnight (t), future1 midnight -> 09:00 (t+1), future2 09:00 (t+1) -> midnight (t+1). 09:00 is local STANDARD time (the rain table's
    day end, results/day_windows.json); midnight is the WALL clock (the issue time; one hour earlier than standard midnight in summer).
    Hour-ending stamps: an hour belongs to the window that holds its end."""
    off = {g: int(v["utc_offset_h"]) for g, v in J("day_windows")["day_ends"]["daymet"]["per_basin"].items()}
    ZONE = {-5: "America/New_York", -6: "America/Chicago"}                       # as code/15_rain_to_midnight.py
    out, day_total = {}, {}
    for b in B:
        z = np.load(DATA / "aorc" / f"aorc_{b['id']}.npz", allow_pickle=False)
        st, r = z["stamp_utc_s"].astype(np.int64), z["rain_mm"].astype(float)
        assert (np.diff(st) == 3600).all() and not np.isnan(r).any(), b["id"]
        cum = np.r_[0.0, np.cumsum(r)]                                  # cum[k] = rain of the first k hours
        t0 = pd.Timestamp(st[0], unit="s")                              # the first hour-ending stamp, UTC

        o, zone = off[b["id"]], ZONE[off[b["id"]]]

        def upto(utc):                                                  # rain of the hours ending at or before this UTC time
            k = int((utc - t0) / pd.Timedelta(hours=1)) + 1
            return cum[max(0, min(k, len(r)))]
        std9 = lambda d: d + pd.Timedelta(hours=9 - o)                  # 09:00 local standard time of day d, in UTC
        wall24 = lambda d: (d + pd.Timedelta(days=1)).tz_localize(zone).tz_convert("UTC").tz_localize(None)    # the wall clock's midnight ending day d
        for g, day in keys:
            if g != b["id"]:
                continue
            t1 = pd.Timestamp(day); t = t1 - pd.Timedelta(days=1)
            edges = [std9(t - pd.Timedelta(days=1)), std9(t), wall24(t), std9(t1), wall24(t1)]
            assert all(x < y for x, y in zip(edges, edges[1:])), (g, day)
            out[(g, day)] = [round(upto(edges[i + 1]) - upto(edges[i]), 1) for i in range(4)]
            day_total[(g, day)] = upto(edges[2]) - upto(wall24(t - pd.Timedelta(days=1)))      # the issue day, midnight to midnight by the wall clock
    return out, day_total


def onset_tables(days, B, P):
    """The two per-onset tables (lead 1), joined to the replay's own onsets: {gauge: {day index: [Stage IV warned, Stage IV mm, AORC mm,
    forecast with Stage IV, rain clock (hourly AORC): table, hidden, future1, future2]}}. Asserts: the same onsets in all three, the same
    warned / missed."""
    test = np.asarray((days >= TEST[0]) & (days <= TEST[1]))
    idx = {str(d.date()): i for i, d in enumerate(days)}
    S4 = J("stage4_realtime")
    s4 = {(r["gauge"], r["target_day"]): r for r in S4["per_onset"]}
    with open(RES / "onset_rain_clock.csv", newline="", encoding="utf-8") as f:
        clock = {(r["gauge_id"], r["onset"]): r for r in csv.DictReader(f)}
    own = {}
    for b in B:
        hit, miss, _ = classify(b["obs"], b["pred"]["wqa"][:, 0], b["thr"], 1, test)
        for t in np.flatnonzero(hit | miss):
            own[(b["id"], str(days[t].date()))] = (b, t)
    n = len(own)
    assert n == P["wq"]["l1"]["hits"] + P["wq"]["l1"]["misses"], n
    assert set(own) == set(s4) == set(clock), f"onsets: replay {n}, stage4_realtime {len(s4)}, onset_rain_clock {len(clock)}"
    AC, AC_day = aorc_clock(B, list(own))
    out, warned_s4 = {}, 0
    for key, (b, t) in sorted(own.items()):
        r, c = s4[key], clock[key]
        assert r["tested"], key
        for k, field in (("wq", "forecast_mm_rain_table"), ("wqa", "forecast_mm_aorc")):
            p = b["pred"][k][t, 0]
            assert abs(p - r[field]) < 1e-4, f"{key}: stored {k} forecast {p:.5f} vs stage4_realtime {r[field]}"
            assert (p > b["thr"]) == r[f"warned_{'rain_table' if k == 'wq' else 'aorc'}"], f"{key}: warned with {k} differs"
        assert (c["missed_wq"] == "True") == (not r["warned_rain_table"]), f"{key}: onset_rain_clock.missed_wq differs"
        assert abs(float(r["aorc_mm"]) - b["aorc"][t - 1]) < 1e-3, f"{key}: AORC of the issue day {b['aorc'][t - 1]:.4f} vs {r['aorc_mm']}"
        assert abs(AC_day[key] - b["aorc"][t - 1]) < 5e-3, f"{key}: the clock's wall-clock day {AC_day[key]:.4f} vs the model's rain to midnight {b['aorc'][t - 1]:.4f}"
        assert (r["forecast_mm_stage4"] > b["thr"]) == bool(r["warned_stage4"]), f"{key}: the Stage IV forecast and its warned flag disagree"   # C5c
        warned_s4 += bool(r["warned_stage4"])
        out.setdefault(b["id"], {})[int(t)] = [int(bool(r["warned_stage4"])), round(r["stage4_mm"], 1), round(r["aorc_mm"], 1), round(r["forecast_mm_stage4"], 2)] \
            + AC[key]
    v = S4["models"]["wqa"]["variants"]["stage4"]
    assert warned_s4 == v["onsets_warned"], ("Stage IV warned", warned_s4, v["onsets_warned"])
    assert S4["models"]["wqa"]["aorc"] == P["wqa"]["l1"]["hits"], ("AORC warned", S4["models"]["wqa"]["aorc"], P["wqa"]["l1"]["hits"])
    return out, dict(hits=warned_s4, pod=warned_s4 / n, onsets=n)


def story(days, B, P, ONS, st4, T):
    """The night of 21-22 November 2006 at Little River, step by step; every number computed here."""
    g, onset_day = STORY
    test = np.asarray((days >= TEST[0]) & (days <= TEST[1]))
    # the cover's rule: of the onsets the rain table missed and the rain to midnight warned (1 day ahead), the one with the highest flow
    best, n_rule, n_other = None, 0, 0
    for b in B:
        hit_a, _, _ = classify(b["obs"], b["pred"]["wqa"][:, 0], b["thr"], 1, test)
        _, miss_q, _ = classify(b["obs"], b["pred"]["wq"][:, 0], b["thr"], 1, test)
        n_rule += int((hit_a & miss_q).sum())
        hit_q, _, _ = classify(b["obs"], b["pred"]["wq"][:, 0], b["thr"], 1, test)
        _, miss_a, _ = classify(b["obs"], b["pred"]["wqa"][:, 0], b["thr"], 1, test)
        n_other += int((hit_q & miss_a).sum())
        for t in np.flatnonzero(hit_a & miss_q):
            if best is None or b["obs"][t] > best[2]:
                best = (b["id"], str(days[t].date()), b["obs"][t])
    assert best[:2] == STORY, f"the cover's rule now picks {best[:2]}, not {STORY}: rewrite the story"
    bi = next(i for i, b in enumerate(B) if b["id"] == g)
    b = B[bi]
    t1 = int(np.flatnonzero(days == pd.Timestamp(onset_day))[0])
    t0 = t1 - 1                                                    # the issue day
    thr = b["thr"]
    rec = ONS[g][t1]
    flow0, flow1 = b["obs"][t0], b["obs"][t1]
    pct = float((b["train_obs"][~np.isnan(b["train_obs"])] < flow0).mean() * 100)
    fq, fa = b["pred"]["wq"][t1, 0], b["pred"]["wqa"][t1, 0]
    assert fq <= thr < fa and rec[0] == 1 and rec[3] > thr
    # the event: its peak over the flood level, ranked among the onsets of the test years (results/flood_events.csv, script 06)
    ev = pd.read_csv(RES / "flood_events.csv", dtype={"gauge_id": str})
    ev = ev[~ev.started_before_test.astype(bool)]
    assert {(r.gauge_id, r.onset_date) for r in ev.itertuples()} == {(gg, str(days[t].date())) for gg, recs in ONS.items() for t in recs}, "flood_events.csv vs the replay's onsets"
    e = ev[(ev.gauge_id == g) & (ev.onset_date == onset_day)].iloc[0]
    rank = int((ev.peak_over_level > e.peak_over_level).sum() + 1)
    med_ratio = float(ev.peak_over_level.median())
    assert rank == 1, rank
    # the hours of the issue day (hourly AORC, basin mean, hour-ending stamps in UTC -> local standard time; November: the wall clock too)
    z = np.load(DATA / "aorc" / f"aorc_{g}.npz", allow_pickle=False)
    off = int(J("day_windows")["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"])
    hr = pd.Series(z["rain_mm"], index=pd.to_datetime(z["stamp_utc_s"], unit="s") + pd.Timedelta(hours=off))
    day0 = pd.Timestamp(days[t0].date())
    win = lambda a, c: float(hr[(hr.index > day0 + pd.Timedelta(hours=a)) & (hr.index <= day0 + pd.Timedelta(hours=c))].sum())
    a_day, a_after9, a_next = win(0, 24), win(9, 24), win(24, 48)
    wet = hr[(hr.index > day0) & (hr.index <= day0 + pd.Timedelta(hours=24)) & (hr >= 0.1)]
    first = int((wet.index[0] - day0) / pd.Timedelta(hours=1)) - 1         # the hour the rain began (start of the first wet hour)
    assert abs(a_day - b["aorc"][t0]) < 1e-3, (a_day, b["aorc"][t0])
    assert a_day - a_after9 < 0.05, "the story says all the day's rain fell after 09:00"
    end_h = float(J("day_windows")["day_ends"]["daymet"]["per_basin"][g]["middle"])         # the rain table's day end measured at this river
    end_here = f"{int(round(end_h * 2) / 2):02d}:{'30' if round(end_h * 2) % 2 else '00'}"
    rain_table = b["rain"][t0]
    fmt_d = lambda t: f"{days[t].day} {days[t]:%B %Y}"
    mm = lambda v: f"{v:.1f} mm"
    md = lambda v: f"{v:.2f} mm/day"
    s4_share, a_share = round(100 * st4["pod"]), round(100 * P["wqa"]["l1"]["pod"])
    steps = [
        dict(set="wq", today=t0, title=f"{b['name']}, the evening of {fmt_d(t0)}",
             html=f"A small river of {b['area']:,} km². Its flow today, <b>{md(flow0)}</b>, is higher than on {pct:.0f}% of its days in the training "
                  f"years: an ordinary day, well under the flood level of <b>{md(thr)}</b>. Every night at midnight the model issues a forecast for the next three days."),
        dict(set="wq", today=t0, title="The usual inputs: no warning",
             html=f"With the rain table and the river gauge, the forecast for {days[t1].day} November is <b>{md(fq)}</b>: under the flood level, "
                  f"so no warning. The rain table holds only <b>{mm(rain_table)}</b> for {days[t0].day} November."),
        dict(set="wq", today=t0, ghost=True, title=f"The rain began at about {first:02d}:00",
             html=f"From about {first:02d}:00 to midnight, <b>{mm(a_after9)}</b> fell on the basin (AORC, {T('fw_aorc')}). The rain table's day closes "
                  f"in the morning or around midday ({T('day_end_daymet')} at the median river, about {end_here} at this one), so for {days[t0].day} "
                  f"November it holds only {mm(rain_table)} and files the rest under {days[t1].day} November: a forecast issued at midnight cannot see it."),
        dict(set="wqa", today=t0, title="Rain up to midnight: a warning",
             html=f"The same network design, trained with one more input, the issue day's rain up to midnight, forecasts <b>{md(fa)}</b>: over the "
                  f"flood level. A warning goes out."),
        dict(set="wqa", today=t1, title=f"{fmt_d(t1)}: the flood",
             html=(f"The river rose to <b>{flow1:.1f} mm/day</b>, {e.peak_over_level:.1f} times its flood level" if e.peak_date == onset_day else
                   f"The river rose to <b>{flow1:.1f} mm/day</b>, and the flood peaked on {pd.Timestamp(e.peak_date):%d %B} at {e.peak_over_level:.1f} times its flood level")
                  + f": the largest of the {st4['onsets']} flood onsets of the test years (a typical one peaks at {med_ratio:.1f} times). Another {mm(a_next)} fell on "
                  f"{days[t1].day} November, after the forecast. The forecast was far too low in size, but it crossed the flood level, and a warning needs only that."),
        dict(set="wqa", today=t1, title="Would real-time radar have seen it?",
             html=f"AORC is {T('fw_aorc')}. With the real-time radar product (Stage IV, <b>{mm(rec[1])}</b> on {days[t0].day} November) in its "
                  f"place, the same networks, trained on AORC, forecast <b>{md(rec[3])}</b>: still a warning (the <i class='d4'></i> under the tick). "
                  f"Stage IV comes about an hour late, which this check does not simulate: without the last hour of every issue day, "
                  f"{T('s4_stage4_no_last_hour_share')} of all onsets are warned instead of {s4_share}%. {T('fw_realtime_short').capitalize()}."),
        dict(set="wqa", today=t1, full=True, title="One night is not the rule",
             html=f"Chosen as the clearest case: of the {n_rule} onsets the rain table missed and the rain to midnight caught, it has the highest "
                  f"flow ({n_other} went the other way). Over all {st4['onsets']} onsets, 1 day ahead: "
                  f"<b>{round(100 * P['wq']['l1']['pod'])}%</b> warned with the rain table, <b>{a_share}%</b> with the rain to midnight ({T('fw_hindsight')}), "
                  f"<b>{s4_share}%</b> with real-time radar ({T('fw_realtime_short')}), <b>{round(100 * P['wqf']['l1']['pod'])}%</b> with perfect weather "
                  f"({T('fw_upper')}). Verdict: {T('kf_practice_all_not_yet')}; the radar check scores onsets only and has no verdict."),
    ]
    a = int(np.flatnonzero(days == day0 - pd.Timedelta(days=7))[0])
    zoom = [a, a + 41]                                             # six weeks from a week before the night: the story box covers the right half
    wy = days[t1].year + (1 if days[t1].month >= 10 else 0)
    facts = dict(gauge=g, basin=b["name"], issue_day=str(day0.date()), onset_day=onset_day, flow_issue_day=round(flow0, 3), percentile_train=round(pct, 1),
                 flood_level=round(thr, 4), forecast_rain_table=round(fq, 3), forecast_rain_to_midnight=round(fa, 3), rain_table_mm=round(rain_table, 2),
                 aorc_day_mm=round(a_day, 2), aorc_after_09_mm=round(a_after9, 2), aorc_first_wet_hour=first, aorc_next_day_mm=round(a_next, 2),
                 flow_onset_day=round(flow1, 3), peak_over_level=round(float(e.peak_over_level), 2), rank=rank, median_peak_over_level=round(med_ratio, 2),
                 stage4_mm=rec[1], forecast_stage4=rec[3], rain_table_day_end_here=end_here, onsets_of_the_selection_rule=n_rule,
                 onsets_the_other_way=n_other)
    return dict(gauge=g, basin=bi, wy=int(wy), zoom=zoom, ghost=dict(t=t0, mm=round(a_after9, 1), label=f"{mm(a_after9)} fell from {first:02d}:00 · the rain table shows {rain_table:.1f} mm"),
                steps=steps), facts


def main():
    T = registry()
    days, B = stored()
    P, seeds = pooled(days, B)
    ONS, st4 = onset_tables(days, B, P)
    assert st4["hits"] == int(T("s4_stage4_onsets_warned").replace(",", "")), (st4["hits"], T("s4_stage4_onsets_warned"))
    assert f"{round(100 * st4['pod'])}%" == T("s4_stage4_share"), T("s4_stage4_share")
    P["stage4"] = dict(hits=st4["hits"], pod=st4["pod"])
    ST, facts = story(days, B, P, ONS, st4, T)
    # the verdict rule of the page against the registry's verdicts of the key figure (1 day ahead)
    rule = lambda r: "GO" if r["pod"] >= .7 and r["far"] <= .3 else "CAUTION" if r["csi"] >= .4 else "NOT YET"
    for k in ("wq", "wqa", "wqf"):
        assert rule(P[k]["l1"]) == T(f"kf_level_{k}_onset_verdict"), (k, rule(P[k]["l1"]), T(f"kf_level_{k}_onset_verdict"))
    by = {n: " and ".join(SHORT[k] for k, *_ in LEVELS if seeds[k] == n) for n in (5, 3)}
    assert set(seeds.values()) == {3, 5}, seeds                                  # the sentences below speak of five and three
    mon = lambda d: pd.Timestamp(d).strftime("%b %Y")
    assert T("fw_review").startswith(REVIEWERS), T("fw_review")[:80]
    text = dict(
        pill=f"REPLAY of stored forecasts · test years {mon(TEST[0])} – {mon(TEST[1])}",
        models_sub="3 or 5 trainings · early stopping",
        models_tip=f"<b>3 Models.</b> One LSTM for all 12 basins (64 cells) against three simple baselines: tomorrow = today, ridge regression, random forest. "
                   f"Each forecast on this page is the average of the model's trainings: five for {by[5]}, three for {by[3]}; early stopping on the validation years.",
        score_tip="<b>5 Score.</b> NSE, KGE, onsets warned, false warnings, CSI. Verdicts fixed in advance: GO = at least 70% of onsets warned and at most 30% false; "
                  f"CAUTION = CSI at least 0.40; otherwise NOT YET. Headline numbers recomputed by {REVIEWERS}.",
        clock_src=f"Rain by the hour: AORC, the basin mean the model reads ({T('fw_aorc')}), from 09:00 the day before the issue day to midnight of the "
                  f"onset day. 09:00 stands for the end of the rain table's day: {T('day_end_daymet')} at the median river, {T('day_end_daymet_basin_range')} "
                  f"by river (standard time); midnight is the issue time (wall clock).",
        note=f"A replay on the test years of the stored forecasts, re-scored with the engine's own rule. The rain to midnight is AORC, {T('fw_aorc')} "
             f"({T('fw_hindsight')}); with the real-time radar product in its place {round(100 * st4['pod'])}% of the onsets are warned against "
             f"{round(100 * P['wqa']['l1']['pod'])}% ({T('s4_stage4_no_last_hour_share')} with the last hour of each issue day left out, since the one-hour "
             f"delay of Stage IV is not simulated): {T('fw_realtime_short')}. Nothing on this page is a live forecast.",
        realtime_short=T("fw_realtime_short"),
        gloss=f"<b>Words.</b> {T('fw_aorc_long')}. Stage IV: NOAA's real-time radar and gauge rain. GEFSv12: NOAA's archived weather forecast "
              f"(the reforecast, control member). Rain table: Daymet, the daily weather table the models read; its day ends near 09:00. "
              f"CSI = warned / (warned + missed + false warnings). Hindsight: how much of the AORC result may come from its after-the-fact daily anchoring.")
    tips = {"wq": "The usual inputs: the Daymet weather table (rain and five more columns), the river gauge and 10 basin properties.",
            "wqa": f"The usual inputs plus one column: the rain of the issue day up to midnight from AORC, {T('fw_aorc')}.",
            "wqaF": "The rain to midnight plus the GEFSv12 weather forecast of the next three days (control member, bias corrected). The networks were "
                    "trained with the AORC rain of those days in the forecast's place, and are run on the forecast. At 1 day ahead it warns fewer onsets "
                    f"than the rain to midnight alone: {T('fw_moved_point')}.",
            "wqf": f"The rain table, the gauge and the actual Daymet weather of the next 3 days, without the rain to midnight: {T('fw_upper')}."}
    assert P["wqaF"]["l1"]["hits"] < P["wqa"]["l1"]["hits"], "the third level's tooltip says it warns fewer onsets at 1 day ahead"
    assert "trained on the AORC rain of t+1..t+3" in J("forecast_input")["sets"]["wqaF_sub"]["what"], "the third level's tooltip describes E2a's training"
    levels = [dict(k=k, label=lab, sub=cap.format(fw_aorc=T("fw_aorc"), fw_upper=T("fw_upper")), seeds=seeds[k], tip=tips.get(k, ""))
              for k, _f, _s, _k, lab, cap in LEVELS]
    data = dict(start=str(days[0].date()), test_start=int(np.flatnonzero(days == pd.Timestamp(TEST[0]))[0]), n=len(days), sets=[k for k, *_ in LEVELS],
                basins=[dict(id=b["id"], name=b["name"], landscape=b["landscape"], area=b["area"], thr=round(b["thr"], 3), obs=b64_u16(b["obs"]),
                             rain=b64_u16(b["rain"]), aorc=b64_u16(b["aorc"]), fc=[b64_u16(x) for x in b["fc"]],
                             pred={k: [b64_u16(b["pred"][k][:, l]) for l in range(3)] for k, *_ in LEVELS},
                             hi=b64_bits(b["hi"]), wn={k: [b64_bits(m) for m in b["wn"][k]] for k, *_ in LEVELS}) for b in B])
    check_shipped(data, B)
    cfg = dict(levels=levels, pooled=P, text=text, onsets=ONS, story=ST, test=TEST)
    t = TEMPLATE.read_text(encoding="utf-8")
    lit = colour_literals(t)
    assert not lit, f"colour literals outside the CSS tokens in replay.html (use var(--token), add a token if needed): {sorted(set(lit))[:10]}"
    for key, val in (("__CFG__", json.dumps(cfg, separators=(",", ":"))), ("__DATA__", json.dumps(data, separators=(",", ":"))), ("__FONT__", font_b64()), ("__ICONS__", icon_symbols())):
        assert t.count(key) == 1, key
        t = t.replace(key, val)
    assert not re.search(r"__(CFG|DATA|FONT|ICONS)__", t), "a placeholder was left unfilled"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(t, encoding="utf-8", newline="\n")                     # LF on every system, so a rebuild gives the same bytes
    (OUT.parent / "facts.json").write_text(json.dumps(dict(what="the numbers the live demo prints, as computed by code/build_live_demo.py", story=facts,
                                                           pooled=P, seeds=seeds, text=text), indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) and {OUT.parent / 'facts.json'}; {len(B)} basins x {len(days)} days x {len(LEVELS)} levels; "
          f"{st4['onsets']} onsets checked against stage4_realtime.json and onset_rain_clock.csv; pooled numbers recounted for {len(LEVELS)} levels x 3 leads")


if __name__ == "__main__":
    main()
