"""Check s1: every prediction file of E3 scored with the reviewer's own numpy code, one file at a time.

Files: the five sets of script 17 (seed average + seeds 0-4), the controls lstm_wq, lstm_w, lstm_wqf (seed average + seeds 0-2) and the four single trainings of 05_lobo.py: 46 files.
Per file and lead: test and validation years, per-basin NSE, pooled warning tables (all days, onsets) at the observed level; for the seed averages and the 05 sets also the
per-basin shares warned, the onset vector (for the paired test), and the warning levels chosen on the validation years with their test tables; for the sets of the
never-seen comparison (and their seeds) the warnings at the model's own level (05_lobo.py's rule).
The high-flow level of each basin comes from lstm_wq's stored observed flow (every file holds the same obs: r1_integrity i4).
Output: $REVIEW_OUT/r2_scores.json. Run: python code/review/more_basins_5seeds/r2_score.py
"""
import time

from r_common import *

UN_SHOW = ["lstm_w", "lstm63_w", "lobo_ref_w_nostatic", "lobo_w_static", "lobo_w_nostatic", "lobo_rf_w", "unseen50_w_static", "unseen50_w_nostatic"]
t0 = time.time()
thr = thresholds(load_npz("lstm_wq"))
res = dict(thresholds=thr, files={})


def own_level(S):
    lev = S.own_levels()
    r = {}
    for k in LEADS:
        t, _ = S.cells(k, "test", level=lev)
        r[f"l{k}"] = dict(onset=t["onset"], all=t["all"])
    return r


for n in ALL_SETS:
    for f in files_of(n):
        S = Scored(f, thr)
        assert S.basins == sorted(S.basins) and len(S.basins) == 12
        rec = dict(meta_seeds=S.meta.get("seeds"), test={}, val={}, coverage={})
        for k in LEADS:
            for part in ("test", "val"):
                s = S.summary(k, part)
                rec[part][f"l{k}"] = dict(nse_median=s["nse_median"], nse_mean=s["nse_mean"], onset=s["onset"], all=s["all"], nse=s["nse"],
                                          onset_pod_by_basin={g: s["per"][g]["onset"]["pod"] for g in S.basins}, all_pod_by_basin={g: s["per"][g]["all"]["pod"] for g in S.basins})
                rec["coverage"][f"{part}_l{k}"] = S.coverage_min(k, part)
        if f == n:                                                                          # seed average or single 05 training: the validation-chosen levels and the onset vector
            rec["chosen"], rec["onset_vector"] = {}, {}
            for k in LEADS:
                v, keys = S.onset_vector(k)
                rec["onset_vector"][f"l{k}"] = dict(warned=v.astype(int).tolist(), keys=[f"{g}:{j}" for g, j in keys])
                for label, kind in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
                    m = S.choose_level(k, kind)
                    t, _ = S.cells(k, "test", trigger=m)
                    rec["chosen"].setdefault(label, {})[f"l{k}"] = dict(trigger=m, all=t["all"], onset=t["onset"])
        if n in UN_SHOW:
            rec["own_level"] = own_level(S)
        res["files"][f] = rec
        del S
    print(f"{n:22s} scored ({len(files_of(n))} files)  {time.time() - t0:5.0f} s", flush=True)
save("r2_scores", res)
