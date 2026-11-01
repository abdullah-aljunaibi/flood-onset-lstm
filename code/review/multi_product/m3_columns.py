"""E7 review, check 3: the input columns rebuilt by the reviewer's own route, and the clock.

  aorc_to_midnight, rain_to_midnight: for every day d the UTC instants of 00:00 local of d and of d+1 (zoneinfo, per basin), and the hours whose hour-ENDING stamp lies in
      (start, end] summed by cumulative sums with searchsorted (route A); the same day assignment summed by pandas groupby (route B, the summation the project uses, so
      the column can be compared BIT FOR BIT with the sha256 the results file records for every input column: inputs.sha256_of_input_columns)
  prcp_nldas: the PRCP column of the raw CAMELS NLDAS forcing file, parsed here (and compared with camels.load_basin)
  _older: the column moved one day later (day t reads day t-1; the first day keeps its own value), sha256 compared
  _shuf: the permutation rebuilt from the documented algorithm (default_rng(20261030); basins in study order, split parts train/val/test, months 1-12; one permutation per
      group shared by the two columns), sha256 of the source index and of both columns compared; the group constraints and counts checked
The clock: every hour summed into day t ends after 00:00 and at or before 24:00 local of day t (wall clock, summer time included); 23/24/25 hours; a made-up-hours test on
the reviewer's builder (hours ending after 24:00 local of a cut day x7+100: no day up to the cut moves, the next day moves) on random and clock-change days; the NLDAS
day measured against the AORC hours (which 24-hour window of AORC hours best matches the NLDAS daily total, 2002-2014) and the hour stamps of AORC against ERA5 (lag of
the best hourly correlation). Writes OUT/my_columns.npz (read by m2 and m4). About 30 s."""
from zoneinfo import ZoneInfo



from m_common import *
import flood as F

t0 = time.time()
R = jload(RESULTS)
C = Compare()
ids = F.study_ids()
C.add("study basins in the order of the results file", list(R["thresholds"]), ids)
Z = zones()
C.add("time zones (from day_windows utc offsets) = inputs.time_zones", R["inputs"]["time_zones"], {g: Z[g] for g in ids})
HOUR = pd.Timedelta(hours=1)
n_d = len(DAYS)


def utc_of_local_midnights(zone):
    """UTC instants (naive) of 00:00 local of every day of DAYS and of the day after the last."""
    days = pd.date_range(DAYS[0], DAYS[-1] + pd.Timedelta(days=1))
    return days.tz_localize(ZoneInfo(zone)).tz_convert("UTC").tz_localize(None)


def era5(g):
    j = json.loads((PROJ / f"data/hourly/era5_{g}.json").read_text())
    assert j["timezone"] == "GMT" and j["utc_offset_seconds"] == 0
    return pd.DatetimeIndex(pd.to_datetime(j["hourly"]["time"])), np.asarray(j["hourly"]["precipitation"], float)


def aorc(g):
    z = np.load(PROJ / f"data/aorc/aorc_{g}.npz", allow_pickle=False)
    return pd.DatetimeIndex(pd.to_datetime(z["stamp_utc_s"], unit="s")), z["rain_mm"].astype("float64")


def daily(stamp, x, zone):
    """Route A (cumsum + searchsorted) and route B (pandas groupby on the same day assignment); hours per day; the day index of every hour (-1 outside)."""
    mid = utc_of_local_midnights(zone)
    st = stamp.values
    i0 = np.searchsorted(st, mid.values[:-1], side="right")         # first hour whose end is after 00:00 local of d
    i1 = np.searchsorted(st, mid.values[1:], side="right")          # one past the last hour whose end is at or before 24:00 local of d
    assert i0[0] > 0 and i1[-1] < len(st), "the hourly series must cover every day"
    cs = np.concatenate([[0.0], np.cumsum(x)])
    a = cs[i1] - cs[i0]
    day_of_hour = np.full(len(st), -1)
    for d in range(n_d):
        day_of_hour[i0[d]:i1[d]] = d
    sel = day_of_hour >= 0
    b = pd.Series(x[sel]).groupby(day_of_hour[sel]).sum().reindex(range(n_d)).values
    return a, b, i1 - i0, day_of_hour, mid


def nldas_raw(g):
    f = next((PROJ / "data/camels_us/basin_dataset_public_v1p2/basin_mean_forcing/nldas").glob(f"*/{g}_lump_nldas_forcing_leap.txt"))
    with open(f) as fh:
        lines = fh.read().splitlines()
    hdr = lines[3].split()
    assert hdr[:4] == ["Year", "Mnth", "Day", "Hr"] and hdr[5].startswith("PRCP"), hdr
    rows = [ln.split() for ln in lines[4:] if ln.strip()]
    idx = pd.to_datetime(dict(year=[int(r[0]) for r in rows], month=[int(r[1]) for r in rows], day=[int(r[2]) for r in rows]))
    s = pd.Series([float(r[5]) for r in rows], index=pd.DatetimeIndex(idx))
    return s.loc[PERIOD[0]:PERIOD[1]], sorted({r[3] for r in rows})


cols = {c: {} for c in ("aorc_to_midnight", "rain_to_midnight", "prcp_nldas")}
clock, made_up, routes = {}, {}, {}
rng_cut = np.random.default_rng(11)
test_idx = np.flatnonzero(within(DAYS, *SPLIT["test"]))
from camels import load_basin
for g in ids:
    zone = Z[g]
    rec = {}
    for name, (stamp, x) in (("aorc_to_midnight", aorc(g)), ("rain_to_midnight", era5(g))):
        assert ((stamp[1:] - stamp[:-1]) == HOUR).all() and np.isfinite(x).all() and (x >= 0).all()
        a, b, nh, doh, mid = daily(stamp, x, zone)
        # every hour of day d ends after 00:00 and at or before 24:00 local of day d (wall clock)
        sel = doh >= 0
        end_local = stamp[sel].tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)
        start_local = (stamp[sel] - HOUR).tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)
        dd = DAYS[doh[sel]]
        on_day = bool(((end_local > dd) & (end_local <= dd + pd.Timedelta(days=1))).all())
        # an hour that ENDS exactly at 24:00 belongs to day d; its start lies on day d (no hour straddles two local days except across the clock change, where the
        # wall clock jumps: checked through the UTC instants above)
        last_hour_end = end_local[np.r_[doh[sel][1:] != doh[sel][:-1], True]]
        rec[name] = dict(routes_max_abs_diff=float(np.abs(a - b).max()), hours=sorted(set(map(int, np.unique(nh)))), n_23=int((nh == 23).sum()), n_25=int((nh == 25).sum()),
                         every_hour_on_its_local_day=on_day, last_hour_of_every_day_ends_at_24=bool((last_hour_end.hour == 0).all() and (last_hour_end.minute == 0).all()),
                         mean_mm_per_day=float(b.mean()))
        C.add(f"{g} {name} mean mm/day = data_side", R["inputs"]["data_side"][g][name]["mean_mm_per_day"], float(b.mean()), tol=1e-9)
        C.add(f"{g} {name} days with 23 / 25 hours = data_side", [R["inputs"]["data_side"][g][name]["days_with_23_hours"], R["inputs"]["data_side"][g][name]["days_with_25_hours"]],
              [int((nh == 23).sum()), int((nh == 25).sum())])
        # made-up hours: hours ENDING after 24:00 local of a cut day are changed; no day up to the cut may move, the next one must
        cuts = sorted({int(c) for c in rng_cut.choice(test_idx, 2, replace=False)} | {int(np.flatnonzero(nh == 23)[-1]), int(np.flatnonzero(nh == 25)[-1]),
                                                                                      int(np.flatnonzero(nh == 23)[-1]) - 1, int(np.flatnonzero(nh == 25)[-1]) - 1})
        ok = True
        for c in cuts:
            x2 = np.where(stamp.values > mid.values[c + 1], x * 7 + 100.0, x)
            b2 = daily(stamp, x2, zone)[1]
            ok &= bool(np.array_equal(b2[:c + 1], b[:c + 1]) and b2[c + 1] != b[c + 1])
        made_up[f"{g} {name}"] = dict(cuts=[str(DAYS[c].date()) for c in cuts], passed=ok)
        cols[name][g] = b
    nl, hrs = nldas_raw(g)
    assert nl.index.equals(DAYS) and nl.notna().all()
    nl_lb = load_basin(g, "nldas").loc[PERIOD[0]:PERIOD[1], "prcp"].astype(float)
    rec["prcp_nldas"] = dict(equal_to_camels_load_basin=bool(np.array_equal(nl.values, nl_lb.values)), hour_column_values=hrs, mean_mm_per_day=float(nl.mean()))
    C.add(f"{g} prcp_nldas mean mm/day = data_side", R["inputs"]["data_side"][g]["prcp_nldas"]["mean_mm_per_day"], float(nl.mean()))
    cols["prcp_nldas"][g] = nl.values
    clock[g] = rec

all_on_day = all(clock[g][n]["every_hour_on_its_local_day"] and clock[g][n]["last_hour_of_every_day_ends_at_24"] for g in ids for n in ("aorc_to_midnight", "rain_to_midnight"))
C.add("every hour summed into day t lies on local day t; the last hour of every day ends at 24:00 local (both hourly products, 12 basins)", True, all_on_day)
C.add("made-up-hours test on the reviewer's builder (12 basins x 2 products x 5-6 cut days)", True, all(v["passed"] for v in made_up.values()))
C.add("NLDAS parsed from the raw file = camels.load_basin (12 basins)", True, all(clock[g]["prcp_nldas"]["equal_to_camels_load_basin"] for g in ids))

# ---- older and shuffled columns ---------------------------------------------------------------------------------------------------------------------------------
for c in ("prcp_nldas", "rain_to_midnight"):
    cols[c + "_older"] = {g: np.concatenate([[cols[c][g][0]], cols[c][g][:-1]]) for g in ids}
tr_end, val_end = pd.Timestamp(SPLIT["train"][1]), pd.Timestamp(SPLIT["val"][1])
months = DAYS.month.values
part = np.where(DAYS <= tr_end, 0, np.where(DAYS <= val_end, 1, 2))
rng = np.random.default_rng(SHUF_SEED)
srcs, counts = [], dict(n=0, from_a_later_day=0, from_an_earlier_day=0, same_day=0, from_another_split_part=0, test_year_values_in_training_days=0, training_days=0)
group_ok = True
for g in ids:
    src = np.empty(n_d, int)
    for p in (0, 1, 2):
        for m in range(1, 13):
            pos = np.flatnonzero((months == m) & (part == p))
            src[pos] = pos[rng.permutation(len(pos))]
    group_ok &= bool(np.array_equal(np.sort(src), np.arange(n_d)) and (months[src] == months).all() and (part[src] == part).all())
    ar = np.arange(n_d)
    counts["n"] += n_d
    counts["from_a_later_day"] += int((src > ar).sum()); counts["from_an_earlier_day"] += int((src < ar).sum()); counts["same_day"] += int((src == ar).sum())
    counts["from_another_split_part"] += int((part[src] != part).sum()); counts["test_year_values_in_training_days"] += int(((part == 0) & (part[src] == 2)).sum())
    counts["training_days"] += int((part == 0).sum())
    srcs.append(src)
    for c in ("prcp_nldas", "rain_to_midnight"):
        cols.setdefault(c + "_shuf", {})[g] = cols[c][g][src]
SH = R["inputs"]["shuffle"]
C.add("shuffle: one permutation per basin x split part x month (values stay in their group)", True, group_ok)
C.add("shuffle counts = inputs.shuffle.counts", SH["counts"], counts)
src_sha = hashlib.sha256(np.ascontiguousarray(np.stack(srcs)).tobytes()).hexdigest()
C.add("shuffle source index sha256 = inputs.shuffle.source_index_sha256", SH["source_index_sha256"], src_sha)
# within-group multisets and the pairing of the two columns
pair_ok = all(np.array_equal(cols["prcp_nldas_shuf"][g], cols["prcp_nldas"][g][s]) and np.array_equal(cols["rain_to_midnight_shuf"][g], cols["rain_to_midnight"][g][s]) for g, s in zip(ids, srcs))
same_vals = all(np.array_equal(np.sort(cols[c + "_shuf"][g][(months == m) & (part == p)]), np.sort(cols[c][g][(months == m) & (part == p)]))
                for g in ids for c in ("prcp_nldas", "rain_to_midnight") for p in (0, 1, 2) for m in range(1, 13))
C.add("shuffled NLDAS and ERA5 moved together, same values per basin x part x month", True, pair_ok and same_vals)
# how far the shuffled values travel (days between the day a value was measured and the day it sits on), and same-year moves inside the test years
dist = np.concatenate([s - np.arange(n_d) for s in srcs])
test_sel = np.concatenate([part == 2 for _ in ids])
same_year_future_test = int(sum(((part == 2) & (s > np.arange(n_d)) & (s - np.arange(n_d) <= 31)).sum() for s in srcs))

# ---- sha256 of every input column against the results file ----------------------------------------------------------------------------------------------------
SHA = R["inputs"]["sha256_of_input_columns"]
my_sha = {c: hashlib.sha256(np.ascontiguousarray(np.stack([np.asarray(cols[c][g], dtype="float64") for g in ids])).tobytes()).hexdigest() for c in SHA}
for c in SHA:
    C.add(f"sha256 of input column {c} (all basins stacked, float64) = inputs.sha256_of_input_columns", SHA[c], my_sha[c])
route_a_vs_b = max(clock[g][n]["routes_max_abs_diff"] for g in ids for n in ("aorc_to_midnight", "rain_to_midnight"))

# ---- the NLDAS day measured against the AORC hours (2002-10-01 to 2014-09-30, the Stage IV era of AORC) -------------------------------------------------------
win = within(DAYS, "2002-10-01", "2014-09-30")
nldas_day = {}
for g in ids:
    stamp, x = aorc(g)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    st = stamp.values
    utc_mid = (DAYS + pd.Timedelta(days=1)).values                   # 00:00 UTC of day d+1
    best = None
    res = {}
    for e in range(-30, 19):                                         # window end = 00:00 UTC of d+1 + e hours
        end = utc_mid + np.timedelta64(e, "h")
        j1 = np.searchsorted(st, end, side="right")
        j0 = np.searchsorted(st, end - np.timedelta64(24, "h"), side="right")
        s = (cs[j1] - cs[j0])[win]
        r = float(np.corrcoef(s, cols["prcp_nldas"][g][win])[0, 1])
        res[e] = r
        if best is None or r > res[best]:
            best = e
    off = -5 if Z[g] == "America/New_York" else -6
    end_local_std = 24 + best + off                                  # hours after 00:00 local standard time of day d
    nldas_day[g] = dict(best_end_utc_offset_h=best, best_end_local_standard_hour=end_local_std, r_best=res[best], r_window_ending_at_local_midnight=res[-off],
                        before_issue_time=bool(end_local_std < 24))
C.add("NLDAS day ends before 24:00 local of day t in every basin (best-matching 24-h window of AORC hours)", True, all(v["before_issue_time"] for v in nldas_day.values()))

# ---- the hour stamps of AORC against ERA5 (both read as hour-ENDING): lag of the best hourly correlation, test years ----------------------------------------------
lag = {}
for g in ids[:4]:
    sa, xa = aorc(g)
    se, xe = era5(g)
    assert sa.equals(se)
    sel = within(sa, "2005-10-01", "2014-09-30")
    xa_, xe_ = xa[sel], xe[sel]
    rr = {L: float(np.corrcoef(xa_[max(0, -L):len(xa_) - max(0, L)], xe_[max(0, L):len(xe_) - max(0, -L)])[0, 1]) for L in range(-3, 4)}
    lag[g] = dict(best_lag_h=max(rr, key=rr.get), r=rr)

np.savez_compressed(OUT / "my_columns.npz", basins=np.array(ids), dates=DAYS.values.astype("datetime64[D]"),
                    **{c: np.stack([np.asarray(cols[c][g], float) for g in ids]) for c in cols}, shuffle_src=np.stack(srcs))
out = dict(compare=C.summary(), rows=C.rows, clock=clock, made_up_hours=made_up, n_made_up_cases=sum(len(v["cuts"]) for v in made_up.values()), route_a_vs_route_b_max_abs_diff_mm=route_a_vs_b,
           shuffle=dict(counts=counts, source_index_sha256=src_sha, group_constraint=group_ok, share_from_a_later_day=counts["from_a_later_day"] / counts["n"],
                        median_abs_days_moved=float(np.median(np.abs(dist))), share_moved_within_31_days=float((np.abs(dist) <= 31).mean()),
                        test_days_reading_a_value_measured_1_to_31_days_later=same_year_future_test, test_days=int(sum((part == 2).sum() for _ in ids))),
           sha256_mine=my_sha, nldas_day=nldas_day, aorc_vs_era5_lag=lag, seconds=round(time.time() - t0, 1))
jsave("m3_columns.json", out)
print(json.dumps(out["compare"]), "route A vs B", route_a_vs_b)
print(json.dumps(out["shuffle"]))
print(json.dumps({g: (v["best_end_local_standard_hour"], round(v["r_best"], 3), round(v["r_window_ending_at_local_midnight"], 3)) for g, v in nldas_day.items()}))
print(json.dumps({g: v["best_lag_h"] for g, v in lag.items()}), out["seconds"])
