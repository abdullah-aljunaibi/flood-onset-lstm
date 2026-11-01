"""Check 1: every prediction set of the review scored with the reviewer's own numpy code, ONE FILE AT A TIME (low memory): the seed averages, every single seed,
the simple models. Per lead on the test years: per-basin NSE, pooled onset and all-day tables at trigger 1.0, per-basin onset share, the onset flags (for the
paired counts). The seed averages lstm_<set>.npz against the mean of their seed files. sha256 of every file read; the backup copies of seeds 0-2 against the
files now. Writes OUT/scores.json."""
from r_common import *
import time

t0 = time.time()
ref_dates, ids, OBS, _ = load_npz("lstm_wq")
TR, TE = within(ref_dates, *SPLIT["train"]), within(ref_dates, *SPLIT["test"])
THR = np.array([float(np.nanquantile(OBS[i][TR], 0.95)) for i in range(len(ids))])


def tables(p, obs):
    res = {}
    for j, k in enumerate(LEADS):
        H = dict(hits=0, misses=0, false_alarms=0)
        Ho = dict(H)
        nse, pod_b, flags = [], [], []
        for i in range(len(ids)):
            o, f = obs[i], p[i, :, j]
            issue = np.concatenate([np.full(k, np.nan), o[:-k]])
            ok = TE & np.isfinite(o) & np.isfinite(f)
            oo, ff, ii = o[ok], f[ok], issue[ok]
            nse.append(float(1.0 - np.sum((ff - oo) ** 2) / np.sum((oo - oo.mean()) ** 2)))
            with np.errstate(invalid="ignore"):
                hi, warn, low = oo > THR[i], ff > THR[i], ii <= THR[i]
            for T, m in ((H, np.ones(len(oo), bool)), (Ho, low)):
                T["hits"] += int((hi & warn & m).sum()); T["misses"] += int((hi & ~warn & m).sum()); T["false_alarms"] += int((~hi & warn & m).sum())
            h_, m_ = int((hi & warn & low).sum()), int((hi & ~warn & low).sum())
            pod_b.append(h_ / (h_ + m_) if h_ + m_ else float("nan"))
            flags.append(warn[hi & low])
        rate = lambda T: dict(T, pod=T["hits"] / (T["hits"] + T["misses"]) if T["hits"] + T["misses"] else float("nan"),
                              far=T["false_alarms"] / (T["hits"] + T["false_alarms"]) if T["hits"] + T["false_alarms"] else float("nan"),
                              csi=T["hits"] / (T["hits"] + T["misses"] + T["false_alarms"]) if T["hits"] + T["misses"] + T["false_alarms"] else float("nan"))
        res[f"l{k}"] = dict(nse=nse, nse_median=float(np.median(nse)), nse_mean=float(np.mean(nse)), onset=rate(Ho), all=rate(H), onset_pod_by_basin=pod_b,
                            onset_flags=np.concatenate(flags).astype(int).tolist())
    return res


SC, HASH, SEEDAVG, OBS_SAME, BACKUP_SAME = {}, {}, {}, {}, {}


def score(name):
    d, b, o, p = load_npz(name)
    assert b == ids and d.equals(ref_dates), name
    OBS_SAME[name] = bool(np.array_equal(o, OBS, equal_nan=True))
    HASH[f"results/preds/{name}.npz"] = sha256_of(PROJ / "results" / "preds" / f"{name}.npz")
    SC[name] = tables(p, OBS)
    return p


for n in SETS:
    acc = None
    for s in SEEDS[n]:
        p = score(f"lstm_{n}_s{s}")
        acc = p.copy() if acc is None else acc + p
        bk = BACKUP / "results" / "preds" / f"lstm_{n}_s{s}.npz"
        if bk.exists():
            BACKUP_SAME[f"lstm_{n}_s{s}"] = sha256_of(bk) == HASH[f"results/preds/lstm_{n}_s{s}.npz"]
        del p
    mean = acc / len(SEEDS[n])
    del acc
    a = score(f"lstm_{n}")
    fin = np.isfinite(a)
    SEEDAVG[n] = dict(n_seeds=len(SEEDS[n]), same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(mean))), max_abs_diff=float(np.abs(a[fin] - mean[fin]).max()), n_values=int(fin.sum()),
                      bitwise_equal=bool(np.array_equal(a, mean, equal_nan=True)))
    bk = BACKUP / "results" / "preds" / f"lstm_{n}.npz"
    SEEDAVG[n]["backup_average_sha256_equal_now"] = (sha256_of(bk) == HASH[f"results/preds/lstm_{n}.npz"]) if bk.exists() else None
    del a, mean
for n in SIMPLE:
    del_ = score(n)
    del del_
jsave("scores.json", dict(ids=ids, thresholds=THR.tolist(), scores=SC, sha256=HASH, seed_average_check=SEEDAVG, obs_identical=OBS_SAME, seeds_0_2_equal_to_backup=BACKUP_SAME,
                          seconds=round(time.time() - t0, 1)))
print("files scored", len(SC), "seconds", round(time.time() - t0, 1))
print("seed averages", {n: (v["max_abs_diff"], v["bitwise_equal"], v["backup_average_sha256_equal_now"]) for n, v in SEEDAVG.items()})
print("obs identical", all(OBS_SAME.values()), "seeds 0-2 equal to backup", all(BACKUP_SAME.values()), len(BACKUP_SAME))
