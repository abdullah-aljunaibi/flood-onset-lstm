"""When did the rain of each flood onset fall, relative to the forecast issue time? No model is trained here.

For each flood onset of the test years (results/flood_events.csv, written by 06_flood_analysis.py) the rain of the 63 hours around the
forecast - from 09:00 of the day before the issue day t to midnight of the target day t+1 - is split by when it fell, using the cached hourly ERA5 rain at the basin centroid
(data/hourly/era5_*.json, written by check_day_windows.py). Windows, in local STANDARD time (the clocks measured in results/day_windows.json;
15_rain_to_midnight.py, which builds the input itself, uses the wall clock of the gauge):
  table   : 09:00 (t-1) -> 09:00 (t)   roughly what the Daymet rain table holds for day t: known at the issue time
  hidden  : 09:00 (t)   -> 24:00 (t)   fallen before the issue time but filed under day t+1: the model cannot see it (what E1 reveals)
  future1 : 00:00 (t+1) -> 09:00 (t+1) falls after the issue time, still inside the Daymet day t+1
  future2 : 09:00 (t+1) -> 24:00 (t+1) falls later on the target day (Daymet day t+2)
Writes results/onset_rain_clock.json and results/onset_rain_clock.csv (one row per onset). About 3 seconds.
Written 2026-10-07 by the session lead (Claude) as a diagnostic for ROADMAP.md.
"""
import json

import numpy as np
import pandas as pd

from common import RES, ROOT
OUT = RES
ev = pd.read_csv(RES / "flood_events.csv", dtype={"gauge_id": str}, parse_dates=["onset_date"])
ev = ev[~ev.started_before_test.astype(bool)]
dw = json.load(open(RES / "day_windows.json"))
off = {g: int(v["utc_offset_h"]) for g, v in dw["day_ends"]["daymet"]["per_basin"].items()}

def hourly(g):
    j = json.load(open(ROOT / "data" / "hourly" / f"era5_{g}.json"))["hourly"]
    t = pd.to_datetime(j["time"]); x = np.array(j["precipitation"], float)
    return pd.Series(x, index=t)          # value at T = rain of the hour ENDING at T (UTC)

H = {g: hourly(g) for g in ev.gauge_id.unique()}

def window_sum(s, start_local, end_local, utc_off):
    """Sum of hourly rain for local hours in [start_local, end_local): hour ending at T covers (T-1h, T]."""
    a = start_local - pd.Timedelta(hours=utc_off) + pd.Timedelta(hours=1)   # first hour-ending stamp inside the window, in UTC
    b = end_local - pd.Timedelta(hours=utc_off)                              # last hour-ending stamp inside the window
    return float(s.loc[a:b].sum())

rows = []
for r in ev.itertuples():
    g, t1 = r.gauge_id, r.onset_date; t = t1 - pd.Timedelta(days=1); u = off[g]
    L = lambda d, h: pd.Timestamp(d) + pd.Timedelta(hours=h)
    s = H[g]
    w = dict(table=window_sum(s, L(t - pd.Timedelta(days=1), 9), L(t, 9), u), hidden=window_sum(s, L(t, 9), L(t, 24), u),
             future1=window_sum(s, L(t1, 0), L(t1, 9), u), future2=window_sum(s, L(t1, 9), L(t1, 24), u))
    tot = sum(w.values())
    rows.append(dict(gauge_id=g, basin=r.basin, onset=t1.date().isoformat(), **w, total=tot,
                     known_by_table=(w["table"]) / tot if tot else np.nan, known_by_midnight=(w["table"] + w["hidden"]) / tot if tot else np.nan,
                     missed_wq=not bool(r.lstm_wq_warned_l1), warned_wqf=bool(r.lstm_wqf_warned_l1), rain_class=r.rain_class))
T = pd.DataFrame(rows)
def summary(df, label):
    q = lambda c: df[c].quantile([.25, .5, .75]).round(2).tolist()
    return dict(label=label, n=len(df), median_share_known_by_table=round(df.known_by_table.median(), 3), q_table=q("known_by_table"),
                median_share_known_by_midnight=round(df.known_by_midnight.median(), 3), q_midnight=q("known_by_midnight"),
                share_of_onsets_with_half_the_rain_known_by_table=round((df.known_by_table >= .5).mean(), 3),
                share_of_onsets_with_half_the_rain_known_by_midnight=round((df.known_by_midnight >= .5).mean(), 3),
                mean_mm=dict(table=round(df.table.mean(), 1), hidden=round(df.hidden.mean(), 1), future1=round(df.future1.mean(), 1), future2=round(df.future2.mean(), 1)))
res = dict(what=__doc__, n_onsets=len(T), all=summary(T, "all onsets"), missed_by_lstm_wq=summary(T[T.missed_wq], "onsets missed by LSTM with gauge, 1 day ahead"),
           missed_by_wq_but_warned_by_perfect=summary(T[T.missed_wq & T.warned_wqf], "missed by lstm_wq, warned by the perfect-forecast model"),
           by_rain_class={k: summary(v, k) for k, v in T.groupby("rain_class")})
json.dump(res, open(OUT / "onset_rain_clock.json", "w"), indent=1); T.to_csv(OUT / "onset_rain_clock.csv", index=False)
for k in ("all", "missed_by_lstm_wq", "missed_by_wq_but_warned_by_perfect"):
    s = res[k]; print(f"{s['label']}: n={s['n']}  median share of the 63-h rain known by the table {s['median_share_known_by_table']:.2f} | by midnight {s['median_share_known_by_midnight']:.2f} | onsets with >=50% known: table {s['share_of_onsets_with_half_the_rain_known_by_table']:.0%} -> midnight {s['share_of_onsets_with_half_the_rain_known_by_midnight']:.0%}  mean mm {s['mean_mm']}")
print("by Daymet rain class (share of the two days' rain filed under the target day):")
for k, s in res["by_rain_class"].items():
    print(f"  {k:12s} n={s['n']:3d}  known by table {s['median_share_known_by_table']:.2f} -> by midnight {s['median_share_known_by_midnight']:.2f}; >=50% known: {s['share_of_onsets_with_half_the_rain_known_by_table']:.0%} -> {s['share_of_onsets_with_half_the_rain_known_by_midnight']:.0%}")
