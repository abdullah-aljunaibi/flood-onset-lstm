"""d4: the rain clock of the flood onsets (code/check_onset_rain_clock.py -> results/onset_rain_clock.json), recomputed with the critic's own code from the ERA5 hourly cache.

What the builder's script states: for each of the 611 onsets of the test years (issue day t, target day t+1) the cached hourly ERA5 rain at the basin centroid (data/hourly/era5_<gauge>.json; the value at T is the rain of the hour ENDING
at T, UTC) is split by local STANDARD time into four windows - table: 09:00 (t-1) to 09:00 (t) (what the Daymet rain table holds for day t: known at the issue time); hidden: 09:00 (t) to 24:00 (t) (fallen, filed under day t+1);
future1: 00:00 (t+1) to 09:00 (t+1); future2: 09:00 (t+1) to 24:00 (t+1); mean rain 6.5 / 12.6 / 8.5 / 8.1 mm, shares 18 / 35 / 46%; of the 497 onsets that the LSTM with the river gauge misses the median share of that
rain known by the table is 3.8% and by midnight 50%; and it labels each onset by the share of the two days' Daymet rain that is filed under the target day (0-25%, ..., 75-100%, or under 1 mm on both days).

What this script does: reads the same hourly files with its own reader; takes the clock of each basin from its own reading of the audit file (Mississippi = Central, UTC-6 in winter; the other eleven Eastern, UTC-5); derives the 611 onsets
from the flow table and the training-year level itself; sums each window with a cumulative-sum route (an hour belongs to a window when its local end time lies in (start, end]); recomputes every summary field of the builder's file and the
rain classes from the Daymet rain column; compares. It also reports what the windows are: 24 + 15 + 9 + 15 = 63 hours, from 09:00 of the day before the issue day to midnight at the end of the target day (the documents call this 'the 48 hours around the onset').
Output: OUT/d4_rain_clock.json
"""
import json as _json
import time

from dcommon import *
import flood as F

t0 = time.time()
D = F.load_data()
ids = list(D)
info = study_basins()
assert ids == list(info)
OFF = {g: utc_offset_standard(time_zone(info[g])) for g in ids}
THR = high_flow_levels(D)
ON = test_onsets(D, THR)
assert len(ON) == 611
GI = {g: i for i, g in enumerate(ids)}

# ---- the hourly ERA5 files, over local standard time ----------------------------------------------------------------------------------------------------------------
CUM, LOCAL, RAIN = {}, {}, {}
for g in ids:
    j = _json.load(open(DATA / "hourly" / f"era5_{g}.json"))
    assert j["timezone"] == "GMT" and j["utc_offset_seconds"] == 0 and j["hourly_units"]["precipitation"] == "mm"
    stamp = pd.to_datetime(j["hourly"]["time"])
    x = np.asarray(j["hourly"]["precipitation"], dtype=float)
    assert np.isfinite(x).all() and (np.diff(stamp.values).astype("timedelta64[h]").astype(int) == 1).all()
    LOCAL[g] = (stamp + pd.Timedelta(hours=OFF[g])).values                       # the local standard time at which each hour ends
    CUM[g] = np.concatenate([[0.0], np.cumsum(x)])
    RAIN[g] = x


def window(g, start, end, route="direct"):
    """Rain of the hours whose local (standard-time) end lies in (start, end]: the hours from start to end. Two routes: the sum of the slice, and the difference of two cumulative sums
    (they differ by about 1e-11 mm, which matters only where a share sits exactly on a threshold: two onsets have exactly half of their rain in the table window)."""
    a = np.searchsorted(LOCAL[g], np.datetime64(start), side="right")
    b = np.searchsorted(LOCAL[g], np.datetime64(end), side="right")
    return float(RAIN[g][a:b].sum()) if route == "direct" else float(CUM[g][b] - CUM[g][a])


H = pd.Timedelta(hours=1)
rows = []
for g, tgt, iss in zip(ON.gauge_id, ON.target, ON.issue):
    w = dict(table=window(g, iss - pd.Timedelta(days=1) + 9 * H, iss + 9 * H), hidden=window(g, iss + 9 * H, iss + 24 * H),
             future1=window(g, tgt, tgt + 9 * H), future2=window(g, tgt + 9 * H, tgt + 24 * H))
    tot = sum(w.values())
    rows.append(dict(gauge_id=g, target=tgt, **w, total=tot, known_by_table=w["table"] / tot if tot else np.nan, known_by_midnight=(w["table"] + w["hidden"]) / tot if tot else np.nan))
T = pd.DataFrame(rows)
rows_cum = []
for g, tgt, iss in zip(ON.gauge_id, ON.target, ON.issue):
    rows_cum.append(dict(table=window(g, iss - pd.Timedelta(days=1) + 9 * H, iss + 9 * H, "cum"), hidden=window(g, iss + 9 * H, iss + 24 * H, "cum"),
                         future1=window(g, tgt, tgt + 9 * H, "cum"), future2=window(g, tgt + 9 * H, tgt + 24 * H, "cum")))
route_difference = float(np.max(np.abs(pd.DataFrame(rows_cum).values - T[["table", "hidden", "future1", "future2"]].values)))

# ---- who missed, who is warned by the perfect-forecast model, and the rain classes -----------------------------------------------------------------------------------
dates = pd.DatetimeIndex(np.load(RES / "preds" / "lstm_wq.npz")["dates"])
DI = {d: i for i, d in enumerate(dates)}


def warned_flags(name):
    z = np.load(RES / "preds" / f"{name}.npz", allow_pickle=False)
    assert list(z["basins"]) == ids
    return np.array([bool(z["pred"][GI[g], DI[t], 0] > THR[g]) for g, t in zip(ON.gauge_id, ON.target)])


T["warned_wq"], T["warned_wqf"] = warned_flags("lstm_wq"), warned_flags("lstm_wqf")
T["missed_wq"] = ~T.warned_wq
share, cls = [], []
for g, t in zip(ON.gauge_id, ON.target):
    p_t, p_i = float(D[g].prcp.loc[t]), float(D[g].prcp.loc[t - pd.Timedelta(days=1)])
    both = p_t + p_i
    s = p_t / both if both > 0 else 0.0
    share.append(s)
    cls.append("under 1 mm on both days" if both < 1.0 else ["0-25%", "25-50%", "50-75%", "75-100%"][min(int(np.floor(s * 4)), 3)])
T["rain_class"] = cls


def summary(df):
    q = lambda c: [round(float(v), 2) for v in df[c].quantile([.25, .5, .75])]
    return dict(n=int(len(df)), median_share_known_by_table=round(float(df.known_by_table.median()), 3), q_table=q("known_by_table"),
                median_share_known_by_midnight=round(float(df.known_by_midnight.median()), 3), q_midnight=q("known_by_midnight"),
                share_of_onsets_with_half_the_rain_known_by_table=round(float((df.known_by_table >= .5).mean()), 3),
                share_of_onsets_with_half_the_rain_known_by_midnight=round(float((df.known_by_midnight >= .5).mean()), 3),
                mean_mm=dict(table=round(float(df.table.mean()), 1), hidden=round(float(df.hidden.mean()), 1), future1=round(float(df.future1.mean()), 1), future2=round(float(df.future2.mean()), 1)))


mine = dict(all=summary(T), missed_by_lstm_wq=summary(T[T.missed_wq]), missed_by_wq_but_warned_by_perfect=summary(T[T.missed_wq & T.warned_wqf]), by_rain_class={k: summary(v) for k, v in T.groupby("rain_class")})
B = jload("onset_rain_clock")
cmp = []


def walk(a, b, path):
    if isinstance(b, dict):
        for k in b:
            if k == "label":
                continue
            walk(a[k], b[k], path + [k])
    elif isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [str(i)])
    else:
        cmp.append(dict(what=".".join(path), mine=a, builder=b, equal=bool(abs(a - b) < 1e-9)))


for k in ("all", "missed_by_lstm_wq", "missed_by_wq_but_warned_by_perfect"):
    walk(mine[k], {kk: vv for kk, vv in B[k].items() if kk != "label"}, [k])
for k, v in B["by_rain_class"].items():
    walk(mine["by_rain_class"][k], {kk: vv for kk, vv in v.items() if kk != "label"}, ["by_rain_class", k])
cmp.append(dict(what="n_onsets", mine=len(T), builder=B["n_onsets"], equal=len(T) == B["n_onsets"]))

# the per-onset rows against the builder's csv
C = pd.read_csv(RES / "onset_rain_clock.csv", dtype={"gauge_id": str}, parse_dates=["onset"]).sort_values(["gauge_id", "onset"]).reset_index(drop=True)
Tm = T.sort_values(["gauge_id", "target"]).reset_index(drop=True)
assert list(C.gauge_id) == list(Tm.gauge_id) and list(C.onset) == list(Tm.target)
row_diff = {c: float(np.max(np.abs(C[c].values - Tm[c].values))) for c in ("table", "hidden", "future1", "future2", "total")}
rain_class_equal = bool((C.rain_class.values == Tm.rain_class.values).all())
missed_equal = bool((C.missed_wq.values == Tm.missed_wq.values).all()) and bool((C.warned_wqf.values == Tm.warned_wqf.values).all())

# the shares quoted in the documents (report 4.6 and 4.8): of the mean rain of the four windows
m = {k: float(T[k].mean()) for k in ("table", "hidden", "future1", "future2")}
tot_m = sum(m.values())
quoted = dict(share_before_0900_of_the_issue_day_window_table=m["table"] / tot_m, share_0900_to_midnight_of_the_issue_day=m["hidden"] / tot_m, share_on_the_target_day=(m["future1"] + m["future2"]) / tot_m,
              mean_mm=m, hours_of_the_windows=dict(table=24, hidden=15, future1=9, future2=15, total=63))

# the rain-timing classes of the report (section 5): onsets per class and the share the LSTM with the river gauge warns of at the plain level
FA = jload("flood_analysis")["failures"]["by_rain_timing"]["lstm_wq"]["plain"]
cls_check = {}
for k in ("0-25%", "25-50%", "50-75%", "75-100%", "under 1 mm on both days"):
    sub = T[T.rain_class == k]
    h = int(sub.warned_wq.sum())
    theirs = FA[k]["onset"]
    cls_check[k] = dict(onsets=int(len(sub)), warned=h, share=h / len(sub) if len(sub) else None, builder_hits=theirs["hits"], builder_onsets=theirs["hits"] + theirs["misses"],
                        equal=bool(h == theirs["hits"] and len(sub) == theirs["hits"] + theirs["misses"]))
n_missed = int(T.missed_wq.sum())
res = dict(mine=mine, two_routes_largest_difference_mm=route_difference, compared=len(cmp), all_equal=all(c["equal"] for c in cmp), not_equal=[c for c in cmp if not c["equal"]], per_onset_rows_max_abs_difference_mm=row_diff,
           rain_class_of_every_onset_equal_to_the_csv=rain_class_equal, missed_and_perfect_flags_equal_to_the_csv=missed_equal, quoted_shares=quoted,
           rain_timing_classes_against_flood_analysis=cls_check, onsets_missed_by_lstm_wq=n_missed, onsets_missed_but_warned_by_the_perfect_forecast_model=int((T.missed_wq & T.warned_wqf).sum()),
           utc_offsets_used={g: OFF[g] for g in ids}, seconds=round(time.time() - t0, 1))
jsave("d4_rain_clock.json", res)
print(f"compared {len(cmp)} fields with results/onset_rain_clock.json: all equal {res['all_equal']}; not equal {res['not_equal']}")
print("per-onset rows, largest difference (mm):", row_diff, "| rain classes equal:", rain_class_equal, "| flags equal:", missed_equal)
print("shares:", {k: round(v, 4) for k, v in quoted.items() if k.startswith("share")}, "mean mm", {k: round(v, 2) for k, v in m.items()})
print("classes:", cls_check)
print("missed:", n_missed, "missed but warned by the perfect forecast:", res["onsets_missed_but_warned_by_the_perfect_forecast_model"])
