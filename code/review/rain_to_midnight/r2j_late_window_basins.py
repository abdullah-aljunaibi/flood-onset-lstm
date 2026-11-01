"""ROUND 2, extra (check x1): how sensitive is value_of_hours_09_to_24 (lstm_wqh minus lstm_wqh_to09) to the basins whose Daymet window is not the 09:00 window? Leaves out the basins whose best-matching wall-clock
end hour, estimated by r2i_daymet_window.py from the hourly ERA5, is 14:00 or later (Little River, Charlie Creek, Myakka when r2i has not been run) and recounts the onsets warned. A few seconds."""
import sys; sys.dont_write_bytecode = True
from r2_common import *
import flood as F
from r2_scoring import *

ids = list(data())
if os.path.exists(f"{OUT}/r2i_daymet_window.json"):
    best = jread("r2i_daymet_window.json")["best_end_hour"]
    late = [g for g in ids if best[g] >= 14]
else:
    late = [g for g in ids if F.SHORT[g].split(",")[0] in ("Little River", "Charlie Creek", "Myakka")]
def hb(name):
    t = onset_table(name, 1); return {g: (int((h & w).sum()), int(h.sum())) for g, (_, h, w) in t.items()}
A, Bc = hb("lstm_wqh"), hb("lstm_wqh_to09")
keep = [g for g in ids if g not in late]
a, b, n = sum(A[g][0] for g in keep), sum(Bc[g][0] for g in keep), sum(A[g][1] for g in keep)
print(f"leaving out {[F.SHORT[g] for g in late]}: wqh {a} vs to09 {b} of {n} onsets: {100*(a-b)/n:+.1f} points (all 12 basins: {sum(A[g][0] for g in ids)} vs {sum(Bc[g][0] for g in ids)} of {sum(A[g][1] for g in ids)}: {100*(sum(A[g][0] for g in ids)-sum(Bc[g][0] for g in ids))/sum(A[g][1] for g in ids):+.1f})")
jsave("r2j_late_window_basins.json", dict(excluded=[F.SHORT[g] for g in late], wqh=a, to09=b, onsets=n, points=100 * (a - b) / n))
