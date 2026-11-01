"""Within-day inputs for the LSTM from data that is ALREADY in this project: the hourly AORC basin series (data/aorc) and threshold-relative flow.

Written 2026-10-28. Why: the Flood_Cloud_Model package (handoff/extras/Flood_Cloud_Model/preprocessing.py) builds its inputs from the last hours
before the issue time (rain of the last 1/3/6/12/24 h and its peak, flow relative to Q95). This project's LSTM reads one daily total of day t.
Basins of 112 to 1,928 km2 respond within hours, so the rain of the last hours of the issue day and its peak intensity are information the
daily total hides - and the lead-1 result turns on the onsets with little rain on the target day (results/review_forecast_input.json).

  within_day_table(g)  -> DataFrame by local calendar day (the wall-clock day of the gauge, summer time included, exactly as 16_aorc_rain_to_midnight.py):
      aorc_to_midnight   mm, the rain of the whole day (the column of script 16; this file proves it by a second route)
      aorc_last3h, aorc_last6h, aorc_last12h   mm, the rain of the hours ending 21:00-24:00, 18:00-24:00, 12:00-24:00 of the day
      aorc_max1h         mm, the wettest hour of the day (basin mean)
      aorc_maxcell1h     mm, the wettest 1 km cell in any hour of the day (NaN where every hour was filled from ERA5)
      aorc_wet_hours     hours of the day with at least 0.5 mm
      aorc_filled_hours  hours filled from ERA5 by code/fetch_aorc.py (a quality flag)
      hours              hours in the day: 24, or 23 / 25 when the clocks change
  add_inputs(D, thr)   -> {gauge: table} with those columns joined and q_rel = log1p(q_mm / Q95): flow relative to the basin's own high-flow level,
                          the Flood_Cloud_Model scaling (the LSTM scales flow with one global mean and spread; see NEXT_EXPERIMENTS_2026-10-28.md)
Causality: every value of day t uses hours that END at or before 24:00 of day t on the wall clock, i.e. at or before the issue time; the made-up-hours
check below proves it on the code. Run this file for the checks and for the timing statistic of the test-year onsets
(results/diagnostics_2026-10-28/subdaily_inputs_check.json). The hourly stamp is the END of the hour (fetch_aorc.py, script 16 note 'Hour stamps').
"""
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_aorc as AO
import flood as F

HOUR = pd.Timedelta(hours=1)
ZONE = {-5: "America/New_York", -6: "America/Chicago"}          # as in 16_aorc_rain_to_midnight.py
LAST = (3, 6, 12)
WET_MM = 0.5
OUT = F.RES_OUT / "diagnostics_2026-10-28"


def zones():
    """The time zone of every study basin, from the winter UTC offset measured in results/day_windows.json (the rule of script 16)."""
    dw = json.loads((F.RES / "day_windows.json").read_text())
    return {g: ZONE[int(v["utc_offset_h"])] for g, v in dw["day_ends"]["daymet"]["per_basin"].items()}


def wall_clock(t_utc, zone):
    return t_utc.tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)


def _table(stamp, x, maxcell, filled, zone, days):
    end = wall_clock(stamp, zone) - pd.Timedelta(minutes=1)      # an hour that ends at exactly 24:00 belongs to the day that ends then
    day = end.normalize()
    T = pd.DataFrame(dict(rain=np.asarray(x, float), maxcell=np.where(filled, np.nan, maxcell), filled=np.asarray(filled, bool), h_end=end.hour.values + 1), index=day)
    g = T.groupby(level=0)
    out = pd.DataFrame(dict(aorc_to_midnight=g.rain.sum(), aorc_max1h=g.rain.max(), aorc_maxcell1h=g.maxcell.max(), aorc_wet_hours=(T.rain >= WET_MM).groupby(level=0).sum(),
                            aorc_filled_hours=g.filled.sum(), hours=g.size()))
    for k in LAST:
        out[f"aorc_last{k}h"] = T.rain.where(T.h_end > 24 - k, 0.0).groupby(level=0).sum()
    return out.reindex(days)


def within_day_table(g, zone=None, days=None):
    zone = zone or zones()[g]
    days = pd.date_range(*F.PERIOD) if days is None else days
    stamp, x = AO.load_series(g)
    z = np.load(AO.basin_file(g), allow_pickle=False)
    return _table(stamp, x, z["max_cell_mm"].astype(float), z["filled"].astype(bool), zone, days)


def second_route(g, zone, days):
    """The daily total by another route: the UTC start and end of every wall-clock day from the time-zone rules, hours between them summed (script 16's proof)."""
    stamp, x = AO.load_series(g)
    start = stamp - HOUR
    zi = ZoneInfo(zone)
    a = days.tz_localize(zi).tz_convert("UTC").tz_localize(None)
    b = (days + pd.Timedelta(days=1)).tz_localize(zi).tz_convert("UTC").tz_localize(None)
    i0, i1 = np.asarray((a - start[0]) // HOUR, int), np.asarray((b - start[0]) // HOUR, int)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, float))])
    ok = (i0 >= 0) & (i1 <= len(x))
    s = np.full(len(days), np.nan)
    s[ok] = c[i1[ok]] - c[i0[ok]]
    return pd.Series(s, index=days)


def add_inputs(D, thr=None):
    thr = thr or F.thresholds(D)
    zs = zones()
    out = {}
    for g, d in D.items():
        W = within_day_table(g, zs[g], d.index).drop(columns=["hours"])
        out[g] = d.drop(columns=[c for c in W.columns if c in d.columns]).join(W).assign(q_rel=np.log1p(d.q_mm / thr[g]))
    return out


def checks(log=print):
    t0 = time.time()
    D = F.load_data()
    thr = F.thresholds(D)
    zs = zones()
    days = pd.date_range(*F.PERIOD)
    rec = dict(what="Checks of code/subdaily_inputs.py and the timing of the issue-day rain before the test-year onsets (written " + time.strftime("%Y-%m-%d") + ")",
               status="PRELIMINARY DIAGNOSTIC, not a reviewed experiment", basins={})
    W = {}
    for g in D:
        w = within_day_table(g, zs[g], days)
        W[g] = w
        r2 = second_route(g, zs[g], days)
        assert w.aorc_to_midnight.notna().all(), f"{g}: a day of the period has no AORC hours"
        assert np.allclose(w.aorc_to_midnight.values, r2.values, atol=1e-6), f"{g}: the two routes to the daily total disagree"
        assert ((w.aorc_last3h <= w.aorc_last6h + 1e-9) & (w.aorc_last6h <= w.aorc_last12h + 1e-9) & (w.aorc_last12h <= w.aorc_to_midnight + 1e-9)).all(), f"{g}: window sums out of order"
        assert set(w.hours.unique()) <= {23, 24, 25}, f"{g}: a day with {sorted(w.hours.unique())} hours"
        # made-up hours: change every hour AFTER the issue time of a cut day; no row up to the cut day may move
        stamp, x = AO.load_series(g)
        z = np.load(AO.basin_file(g), allow_pickle=False)
        cut = days[5000]
        issue_utc = (cut + pd.Timedelta(days=1)).tz_localize(ZoneInfo(zs[g])).tz_convert("UTC").tz_localize(None)       # 24:00 of the cut day, as UTC
        later = stamp > issue_utc                                                                                      # hours ending after the issue time
        x2 = np.where(later, x * 7 + 100, x)
        w2 = _table(stamp, x2, z["max_cell_mm"].astype(float), z["filled"].astype(bool), zs[g], days)
        same = w.loc[:cut].fillna(-1).equals(w2.loc[:cut].fillna(-1))
        moved = not w.loc[cut + pd.Timedelta(days=1)].fillna(-1).equals(w2.loc[cut + pd.Timedelta(days=1)].fillna(-1))
        assert same and moved, f"{g}: a within-day value read an hour after the issue time (same={same}, next day moved={moved})"
        rec["basins"][g] = dict(zone=zs[g], days=int(len(w)), days_23h=int((w.hours == 23).sum()), days_25h=int((w.hours == 25).sum()), filled_hours=int(w.aorc_filled_hours.sum()),
                                mean_mm_per_day=float(w.aorc_to_midnight.mean()), corr_with_daymet_prcp=float(np.corrcoef(w.aorc_to_midnight.values, D[g].prcp.values)[0, 1]),
                                share_of_rain_in_last6h_all_days=float(w.aorc_last6h.sum() / w.aorc_to_midnight.sum()))
        log(f"{g}: routes agree, causal; {rec['basins'][g]['days_23h']}+{rec['basins'][g]['days_25h']} clock-change days, {rec['basins'][g]['filled_hours']} filled hours, "
            f"corr with Daymet {rec['basins'][g]['corr_with_daymet_prcp']:.3f}")
    # timing statistic: lead-1 onsets of the test years (target day t+1 above Q95, day t at or below): where in day t did its rain fall?
    a, b = F.SPLIT["test"]
    rows = []
    for g, d in D.items():
        q = d.q_mm
        onset = ((q.shift(-1) > thr[g]) & (q <= thr[g])).loc[a:b]          # issue day t of a lead-1 onset
        w = W[g].loc[a:b]
        rows.append(pd.DataFrame(dict(onset=onset.values, total=w.aorc_to_midnight.values, last6=w.aorc_last6h.values, last12=w.aorc_last12h.values, max1h=w.aorc_max1h.values), index=onset.index))
    T = pd.concat(rows)
    wet = T[T.total >= 1.0]
    on, other = wet[wet.onset], wet[~wet.onset]
    def summary(S):
        return dict(days=int(len(S)), median_total_mm=float(S.total.median()), median_share_last6h=float((S.last6 / S.total).median()), median_share_last12h=float((S.last12 / S.total).median()),
                    share_of_days_with_half_the_rain_after_18h=float(((S.last6 / S.total) >= 0.5).mean()), median_max1h_mm=float(S.max1h.median()))
    rec["issue_day_rain_timing_test_years"] = dict(how_to_read="issue days t with at least 1 mm of AORC rain; 'onset' = the flow of day t+1 is above Q95 while day t is at or below (the lead-1 onsets); "
                                                               "the share of the day's rain that fell after 18:00 and after 12:00 local time, and the wettest hour",
                                                   onset_issue_days=summary(on), other_wet_days=summary(other),
                                                   onset_issue_days_dry=int((T.onset & (T.total < 1.0)).sum()), onsets_total=int(T.onset.sum()))
    rec["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subdaily_inputs_check.json").write_text(json.dumps(rec, indent=1))
    s = rec["issue_day_rain_timing_test_years"]
    log(f"test-year lead-1 onsets: {s['onsets_total']} (of which {s['onset_issue_days_dry']} with under 1 mm on the issue day); wet onset issue days: median share after 18:00 "
        f"{s['onset_issue_days']['median_share_last6h']:.2f} vs {s['other_wet_days']['median_share_last6h']:.2f} on other wet days; median wettest hour "
        f"{s['onset_issue_days']['median_max1h_mm']:.1f} vs {s['other_wet_days']['median_max1h_mm']:.1f} mm; days with half the rain after 18:00: "
        f"{s['onset_issue_days']['share_of_days_with_half_the_rain_after_18h']:.2f} vs {s['other_wet_days']['share_of_days_with_half_the_rain_after_18h']:.2f}")
    log(f"all checks passed; wrote {OUT / 'subdaily_inputs_check.json'} ({rec['runtime_s']} s)")
    return rec


if __name__ == "__main__":
    checks()
