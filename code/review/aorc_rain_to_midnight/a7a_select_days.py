"""AORC review, check (d): which extra days of the test years to fetch from Stage IV (a modest sample; in the event 924 + 140 files, about 1.3 GB: the files are 1.2 MB each, not the 1.1 MB assumed when the sample was planned).
The builder's sample (code/check_aorc_hindsight.py) is the 50 stormiest days by pooled ERA5 rain, at least 4 days apart. That sample holds little rain after midnight (mean Stage IV rain in the 7 hours after midnight:
1.1 mm against 21.8 mm before) and covers few of the onsets the forecast is scored on. This script adds days chosen WITHOUT any forecast: (1) the days before the most onsets (the issue days t of onsets
at lead 1: observed flow only, no forecast; days with at least 3 basins having an onset on day t+1), and (2) the days with the most AORC rain in the 7 hours after local midnight (night storms: where a leak of
rain after midnight into the hours before midnight would matter most), both at least 2 days apart from each other and not one of the builder's days. Writes a7a_days.json into the output folder. A few seconds."""
import sys; sys.dont_write_bytecode = True
import collections
from a_common import *
from a_scoring import *
import flood as F
import a_clock as C

N_ONSET, N_NIGHT, MIN_APART, MIN_FROM_BUILDER = int(os.environ.get("N_ONSET_DAYS", 17)), int(os.environ.get("N_NIGHT_DAYS", 11)), int(os.environ.get("MIN_DAYS_APART", 2)), 1
MIN_ONSETS = int(os.environ.get("MIN_ONSETS_PER_DAY", 3))          # defaults = the sample of the review (28 days). To cover EVERY onset (368 issue days, 56 of them already held): N_ONSET_DAYS=400 N_NIGHT_DAYS=0 MIN_ONSETS_PER_DAY=1 MIN_DAYS_APART=0
D = F.load_data(); ids = list(D)
DW = jload("results/day_windows.json")
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
ZONE = {g: "America/Chicago" if UTC_OFF[g] == -6 else "America/New_York" for g in ids}
test = np.arange(np.datetime64("2005-10-02"), np.datetime64("2014-09-29"))
H = jload("results/aorc_hindsight.json"); builder = [pd.Timestamp(d) for d in H["numbers"]["days"]]
ok = lambda d, chosen: all(abs((d - c).days) >= MIN_APART for c in chosen) and all(abs((d - b).days) >= MIN_FROM_BUILDER for b in builder)

# onsets of the test years (observed flow only): the issue day t of every onset at lead 1
t = onset_table("lstm_wq", 1)
cnt = collections.Counter(d for g in ids for d in pd.DatetimeIndex(t[g][0][t[g][1]] - pd.Timedelta(days=1)))
print("onsets at lead 1:", sum(cnt.values()), "on", len(cnt), "distinct issue days; builder's days cover", sum(c for d, c in cnt.items() if d in set(builder)), "of them")
chosen, why = [], {}
for d, c in cnt.most_common():
    if c >= MIN_ONSETS and ok(d, chosen) and len(chosen) < N_ONSET:
        chosen.append(d); why[str(d.date())] = f"onsets on t+1 in {c} basins"
# night storms: pooled AORC rain in the 7 hours after local midnight
S = {}
for g in ids:
    tt, x, fl, nv, nc = C.read_aorc(g)
    cs = np.concatenate([[0.0], np.cumsum(x)]); pos = lambda inst: int(((inst - tt[0]) / np.timedelta64(60, "m")))
    S[g] = np.array([cs[pos(C.local_instant_utc(C.FIRST_DAY + int(i) + 1, 7, ZONE[g])) + 1] - cs[pos(C.local_instant_utc(C.FIRST_DAY + int(i), 24, ZONE[g])) + 1] for i in (test - C.FIRST_DAY).astype(int)])
post = np.mean([S[g] for g in ids], 0)
n_before = len(chosen)
for k in np.argsort(-post):
    d = pd.Timestamp(test[k])
    if ok(d, chosen) and len(chosen) < n_before + N_NIGHT:
        chosen.append(d); why[str(d.date())] = f"night storm: pooled AORC rain in the 7 hours after midnight {post[k]:.1f} mm"
chosen = sorted(chosen)
files = len(chosen) * 33
print(f"{len(chosen)} days chosen ({N_ONSET} by onsets, {N_NIGHT} by night rain), {files} hourly Stage IV files (about {files * 1.23 / 1000:.2f} GB at the 1.2 MB the files measure, plus 5 files per day for the placebo hours after 12 UTC):")
for d in chosen: print("  ", d.date(), why[str(d.date())], "| onsets that day:", cnt.get(d, 0))
jsave("a7a_days.json", dict(days=[str(d.date()) for d in chosen], why=why, builder_days=[str(b.date()) for b in builder], n_files=files, onsets_covered=int(sum(cnt.get(d, 0) for d in chosen)), onsets_total=int(sum(cnt.values()))))
