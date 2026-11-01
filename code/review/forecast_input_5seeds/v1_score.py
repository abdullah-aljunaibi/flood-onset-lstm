"""Round 2, check 1: every prediction file the five-seed numbers of results/forecast_input.json read (16 sets: their seed averages and 5 or 3 single seeds), scored with
the reviewer's own numpy code, ONE FILE AT A TIME (memory): Q95 of the training years of the set's own split (from the stored observed flow), onsets = target days above Q95
whose flow lead days earlier was at or below it, a warning = forecast > trigger x Q95; per lead: per-basin NSE on the test and validation days, pooled onset and all-day tables at
trigger 1.0, the sweep over the warning levels 0.30..1.50 (validation and test years), the onset rows (for the average precision, the ROC area and the paired onsets).
Also: every seed average equals the day-by-day mean of its seed files (summed in seed order, as flood.ensemble), its meta seed list, observed flow identical in every file,
and the sha256 of every file read. Output OUT/v1_scores.pkl."""
from v_common import *
import pickle

t0 = time.time()
ref_dates, ids, OBS, _, _ = load_npz("lstm_wqa")
assert np.all(np.diff(ref_dates.values).astype("timedelta64[D]").astype(int) == 1), "dates not contiguous"
THR = {arm: np.array([float(np.quantile(OBS[i][mask(ref_dates, *SPLITS[arm]["train"])], 0.95)) for i in range(len(ids))]) for arm in SPLITS}
PART = {arm: {p: mask(ref_dates, *SPLITS[arm][p]) for p in ("val", "test")} for arm in SPLITS}


def score_file(pred, arm, full):
    thr = THR[arm]
    res = {}
    for part in ("test", "val") if full else ("test",):
        pm = PART[arm][part]
        res[part] = {}
        for j, k in enumerate(LEADS):
            nse, ob, ab = [], [], []
            ON = np.zeros(3, int); AL = np.zeros(3, int)
            SW_on = np.zeros((len(LEVELS), 3), int); SW_al = np.zeros((len(LEVELS), 3), int)
            rows_y, rows_r, rows_b = [], [], []
            for i in range(len(ids)):
                o, f = OBS[i], pred[i, :, j]
                issue = np.r_[np.full(k, np.nan), o[:-k]]
                ok = pm & np.isfinite(o) & np.isfinite(f)
                oo, ff, ii = o[ok], f[ok], issue[ok]
                nse.append(1.0 - np.sum((ff - oo) ** 2) / np.sum((oo - oo.mean()) ** 2))
                hi, low = oo > thr[i], ii <= thr[i]
                for lev_i, m in enumerate(LEVELS if full else []):
                    w = ff > m * thr[i]
                    SW_al[lev_i] += [(hi & w).sum(), (hi & ~w).sum(), (~hi & w).sum()]
                    SW_on[lev_i] += [(hi & w & low).sum(), (hi & ~w & low).sum(), (~hi & w & low).sum()]
                w = ff > thr[i]
                a_ = np.array([(hi & w).sum(), (hi & ~w).sum(), (~hi & w).sum()]); o_ = np.array([(hi & w & low).sum(), (hi & ~w & low).sum(), (~hi & w & low).sum()])
                AL += a_; ON += o_; ab.append(a_); ob.append(o_)
                if full and part == "test":
                    rows_y.append(hi[low]); rows_r.append((ff / thr[i])[low]); rows_b.append(np.full(int(low.sum()), i))
            r = dict(nse=np.array(nse), onset=rates(*ON), all=rates(*AL), onset_b=[rates(*x) for x in ob], all_b=[rates(*x) for x in ab])
            if full:
                r["sweep_onset"] = [rates(*x) for x in SW_on]; r["sweep_all"] = [rates(*x) for x in SW_al]
                if part == "test":
                    r["rows_y"], r["rows_r"] = np.concatenate(rows_y), np.concatenate(rows_r)
            res[part][f"l{k}"] = r
    return res


SC, AVG_CHECK, SHA, META = {}, {}, {}, {}
for n in SHOWN:
    arm = ARM[n]
    acc = None
    for s in SEEDS[n]:
        name = f"lstm_{n}_s{s}"
        d, b, o, p, meta = load_npz(name)
        assert b == ids and d.equals(ref_dates), name
        assert np.array_equal(o, OBS), f"{name}: observed flow differs"
        SHA[f"results/preds/{name}.npz"] = sha256(f"results/preds/{name}.npz"); META[name] = meta
        SC[name] = score_file(p, arm, full=False)
        acc = (0 + p) if acc is None else acc + p
        del p
    mean = acc / len(SEEDS[n])
    d, b, o, p, meta = load_npz(f"lstm_{n}")
    assert b == ids and d.equals(ref_dates) and np.array_equal(o, OBS), n
    SHA[f"results/preds/lstm_{n}.npz"] = sha256(f"results/preds/lstm_{n}.npz"); META[f"lstm_{n}"] = meta
    fin = np.isfinite(p)
    AVG_CHECK[n] = dict(meta_seeds=meta.get("seeds"), expected_seeds=SEEDS[n], seeds_match=list(meta.get("seeds") or []) == SEEDS[n],
                        bit_for_bit=bool(np.array_equal(p, mean, equal_nan=True)), same_nan=bool(np.array_equal(np.isnan(p), np.isnan(mean))),
                        max_abs_diff=float(np.abs(p[fin] - mean[fin]).max()), n_values=int(fin.sum()), meta_script=meta.get("script"))
    SC[f"lstm_{n}"] = score_file(p, arm, full=True)
    del p, acc, mean
    print(f"{n:22s} {arm}: seeds {SEEDS[n]}, average bit for bit {AVG_CHECK[n]['bit_for_bit']} (max diff {AVG_CHECK[n]['max_abs_diff']:.1e}), meta seeds {AVG_CHECK[n]['meta_seeds']}; "
          f"l1/l2/l3 onsets warned " + "/".join(f"{SC[f'lstm_{n}']['test'][f'l{k}']['onset']['pod']:.4f}" for k in LEADS), flush=True)
pickle.dump(dict(SC=SC, AVG_CHECK=AVG_CHECK, SHA=SHA, META=META, THR={a: v.tolist() for a, v in THR.items()}, ids=ids), open(OUT / "v1_scores.pkl", "wb"))
jsave("v1_avg_check.json", dict(AVG_CHECK=AVG_CHECK, THR={a: dict(zip(ids, v.tolist())) for a, v in THR.items()}, n_files=len(SHA), seconds=round(time.time() - t0, 1)))
print(f"{len(SHA)} files scored in {time.time() - t0:.0f} s")
