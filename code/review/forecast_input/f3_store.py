"""E2 review, check (a) continued: the whole stored GEFS box (data/gefs/c00: 178 monthly npz files and their json sidecars) scanned with the reviewer's own code, numpy only (it does not call fetch_gefs.py: its --check rewrites
data/gefs/c00/done_apcp.txt). Shapes and dtypes, the 5392 init dates one per day 2000-01-01 .. 2014-10-05, the grid arrays (lat 25..38 south to north, lon -91..-75, step 0.25), step_end_hours 3..78, every flag 0, no 65535, the sidecars
(complete, counts, the per-day box totals against my own sum of the stored array, messages per day, twins, the most negative value before the clamp), done_apcp.txt, plus a look at the values themselves (zero days, extremes, the monthly
climate of the box). A few seconds."""
import sys; sys.dont_write_bytecode = True
import glob, hashlib
from f_common import *

D = f"{PROJ}/data/gefs/c00"
files = sorted(glob.glob(f"{D}/apcp_20*.npz"))
out = dict(n_months=len(files))
lat_exp, lon_exp, steps_exp = 25.0 + 0.25 * np.arange(53), -91.0 + 0.25 * np.arange(65), np.arange(3, 79, 3)
days_all = pd.date_range("2000-01-01", "2014-10-05")
inits, flags, tot_mm, max3h, wet_share, month_mean = [], [], [], [], [], {}
problems, side_problems = [], []
import collections
n_twin_days, msgs_hist = [], collections.Counter()
bytes_total = 0
for f in files:
    ym = os.path.basename(f)[5:11]
    z = np.load(f)
    a, init, fl = z["apcp"], z["init_date"], z["flag"]
    exp_days = days_all[(days_all.year * 100 + days_all.month) == int(ym)]
    if a.dtype != np.uint16 or a.shape != (len(exp_days), 26, 53, 65): problems.append((ym, "apcp shape/dtype", a.shape, a.dtype))
    if not np.array_equal(init, exp_days.strftime("%Y%m%d").astype(int)): problems.append((ym, "init_date"))
    if not (np.array_equal(z["lat"], lat_exp) and np.array_equal(z["lon"], lon_exp) and np.array_equal(z["step_end_hours"], steps_exp)): problems.append((ym, "grid or steps"))
    if not (fl == 0).all(): problems.append((ym, "flags", np.unique(fl)))
    if (a == 65535).any(): problems.append((ym, "no-data values"))
    mm = a.astype(np.float64) / 10.0
    box_total = mm.sum(axis=(1, 2, 3))
    inits.append(init); flags.append(fl); tot_mm.append(box_total); max3h.append(mm.max(axis=(1, 2, 3))); wet_share.append((a > 0).mean(axis=(1, 2, 3)))
    month_mean[ym] = float(mm[:, :12].mean())            # mean mm per 3 h over the first 36 hours, whole box
    js = json.load(open(f.replace(".npz", ".json")))
    if not js["complete"] or js["counts"]["0"] != len(exp_days) or js["counts"]["1"] or js["counts"]["2"] or js["counts"]["255"]: side_problems.append((ym, "counts", js["counts"], js["complete"]))
    bytes_total += js["bytes_fetched"]
    for k, d in enumerate(init):
        e = js["days"][str(d)]
        if abs(e["sum_mm_stored"] - box_total[k]) > 0.051: side_problems.append((ym, int(d), "sum_mm_stored", e["sum_mm_stored"], float(box_total[k])))
        if e["min_mm_before_clamp"] < -1.0: side_problems.append((ym, int(d), "min_mm_before_clamp", e["min_mm_before_clamp"]))
        if abs(e["max_mm_3h"] - mm[k].max()) > 0.051: side_problems.append((ym, int(d), "max_mm_3h", e["max_mm_3h"], float(mm[k].max())))
        msgs_hist[e["messages"]] += 1
        if e.get("twin_messages_ignored", 0): n_twin_days.append((int(d), e["messages"], e["twin_messages_ignored"]))
init = np.concatenate(inits); flag = np.concatenate(flags); tot = np.concatenate(tot_mm); mx = np.concatenate(max3h); wet = np.concatenate(wet_share)
done = [l.split() for l in open(f"{D}/done_apcp.txt").read().strip().split("\n")]
out.update(n_days=int(len(init)), first=int(init[0]), last=int(init[-1]), one_run_per_day=bool(np.array_equal(init, days_all.strftime("%Y%m%d").astype(int))), all_flags_zero=bool((flag == 0).all()), problems_in_the_arrays=problems[:5], n_problems_arrays=len(problems),
           problems_in_the_sidecars=side_problems[:5], n_problems_sidecars=len(side_problems), done_list=dict(n_lines=len(done), days_listed_twice={d: [x[1] for x in done if x[0] == d] for d in sorted({x[0] for x in done if sum(y[0] == x[0] for y in done) > 1})}, flag_of_the_last_line_per_day_all_zero=all(v == "0" for v in dict((x[0], x[1]) for x in done).values()),
                          days_equal_the_store=sorted({int(x[0]) for x in done}) == init.tolist(), note="the list is appended to during a run and rebuilt from the month files at the next start; the store itself holds flag 0 for every day"),
           bytes_fetched_total_gb=round(bytes_total / 1e9, 2), messages_per_day=dict(sorted(msgs_hist.items())), days_with_twin_messages_ignored=n_twin_days,
           values=dict(zero_total_days=int((tot == 0).sum()), box_total_mm_per_78h_median=float(np.median(tot)), box_total_mm_per_78h_p99=float(np.percentile(tot, 99)), largest_3h_window_mm=float(mx.max()), day_of_the_largest=int(init[mx.argmax()]),
                       median_share_of_boxes_with_rain=float(np.median(wet)), mean_rain_mm_per_day_whole_box_78h_over_3=float(tot.mean() / 3.25 / (53 * 65))),
           climate_first_36_hours_mm_per_3h_by_calendar_month={m: float(np.mean([v for k, v in month_mean.items() if int(k[4:]) == m])) for m in range(1, 13)})
print(f"{len(files)} monthly files, {len(init)} days ({int(init[0])} .. {int(init[-1])}), one run per day: {out['one_run_per_day']}, every flag 0: {out['all_flags_zero']}, array problems {len(problems)}, sidecar problems {len(side_problems)}, "
      f"done list {len(done)} lines, days listed twice {out['done_list']['days_listed_twice']}, days equal to the store {out['done_list']['days_equal_the_store']}, bytes fetched {out['bytes_fetched_total_gb']} GB, messages per day {out['messages_per_day']}, twin days {n_twin_days}")
print(f"values: zero days {out['values']['zero_total_days']}, median box total {out['values']['box_total_mm_per_78h_median']:.0f} mm per 78 h, largest 3-hour window {out['values']['largest_3h_window_mm']:.1f} mm on {out['values']['day_of_the_largest']}")
jsave("f3_store.json", out)
