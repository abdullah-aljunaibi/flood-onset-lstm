"""E2 review, check (i) and 'how much to trust': the same comparisons WITHOUT a warning trigger. On the onset subset of the test period (issue-day flow at or below the high-flow level; label = the flow of the target day
is above it) every set is scored by the average precision (area under the precision-recall curve: the right summary of a rare event) and the ROC area of its forecast flow divided by the basin's high-flow level (so
that the 12 basins can be pooled). A higher score means the forecast ranks the onset days better, whatever level one chooses to warn at. Difference of two sets with a 95% interval from a block bootstrap over target days
(blocks of 30 days, 1000 replicates, seed 20261008). Reads only the saved prediction sets. About 2 minutes."""
import sys; sys.dont_write_bytecode = True
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from f_common import *
from f_scoring import *

t0 = time.time()
SETS = {"main": ["lstm_wq", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqaF_sub_perfectpred", "lstm_wqaP", "lstm_wqaF_clim", "lstm_wqF_sub", "base_linreg_wqa", "base_linreg_wqaF", "base_rf_wqa", "base_rf_wqaF"],
        "sat": ["lstm_wq_s", "lstm_wqa_s", "lstm_wqaF_s", "lstm_wqaF_clim_s", "lstm_wqaP_s"]}
BLOCK, B, SEED = 30, 1000, 20261008
rng = np.random.default_rng(SEED)

def rows_of(names, lead):
    """The pooled rows of the test period shared by the sets: label, normalised score per set, target date."""
    lab, date, score = [], [], {n: [] for n in names}
    ref = None
    for n in names:
        dates, basins, obs, pred = load_npz(n)
        sp = split_of(n); a, b = sp["test"]; test = part(dates, a, b); thr = thresholds(basins, dates, sp)
        cur = []
        for i, g in enumerate(basins):
            o = obs[i]; f = pred[i, :, lead - 1]; issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
            ok = test & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g])
            cur.append((g, dates[ok], (o > thr[g])[ok], (f / thr[g])[ok]))
        if ref is None:
            ref = [(g, d, h) for g, d, h, _ in cur]
            lab = np.concatenate([h for _, _, h, _ in cur]); date = np.concatenate([d.values for _, d, _, _ in cur])
        else:
            assert all((r[1] == c[1]).all() and (r[2] == c[2]).all() for r, c in zip(ref, cur)), "sets must share rows"
        score[n] = np.concatenate([s for _, _, _, s in cur])
    return lab.astype(bool), pd.DatetimeIndex(date), score, sp

out = {}
for arm, names in SETS.items():
    a0 = pd.Timestamp(SPLITS[arm]["test"][0])
    print(f"\n{arm} split: average precision (AP) and ROC area of the forecast flow on the onset subset; onset share in the subset")
    for lead in KS:
        lab, date, score, sp = rows_of(names, lead)
        out.setdefault(arm, {})[f"l{lead}"] = dict(n_rows=int(len(lab)), n_onsets=int(lab.sum()), sets={n: dict(ap=float(average_precision_score(lab, score[n])), roc_auc=float(roc_auc_score(lab, score[n]))) for n in names})
        print(f"   lead {lead}: {int(lab.sum())} onsets of {len(lab)} days; " + " | ".join(f"{n.replace('lstm_', '').replace('wqaF_sub', 'F_sub')} AP {out[arm][f'l{lead}']['sets'][n]['ap']:.3f} AUC {out[arm][f'l{lead}']['sets'][n]['roc_auc']:.3f}" for n in names))
        blk = ((date - a0).days // BLOCK).values.astype(int); nb = blk.max() + 1
        idx_of = [np.flatnonzero(blk == k) for k in range(nb)]
        pairs = [("lstm_wqaF_sub", "lstm_wqa"), ("lstm_wqaF_sub", "lstm_wqaF_clim"), ("lstm_wqaP", "lstm_wqaF_sub"), ("base_rf_wqaF", "lstm_wqaF_sub"), ("base_linreg_wqaF", "lstm_wqaF_sub")] if arm == "main" else [("lstm_wqaF_s", "lstm_wqa_s"), ("lstm_wqaF_s", "lstm_wqaF_clim_s"), ("lstm_wqaP_s", "lstm_wqaF_s")]
        for T, C in pairs:
            d0 = out[arm][f"l{lead}"]["sets"][T]["ap"] - out[arm][f"l{lead}"]["sets"][C]["ap"]
            dd = []
            for _ in range(B):
                ii = np.concatenate([idx_of[k] for k in rng.integers(0, nb, nb)])
                l = lab[ii]
                dd.append(average_precision_score(l, score[T][ii]) - average_precision_score(l, score[C][ii]))
            ci = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            out[arm][f"l{lead}"].setdefault("ap_differences", {})[f"{T[5:] if T.startswith('lstm_') else T} - {C[5:] if C.startswith('lstm_') else C}"] = dict(diff=float(d0), ci95=ci, share_above_zero=float((np.array(dd) > 0).mean()))
        print("      AP differences (95% block bootstrap): " + " | ".join(f"{k}: {v['diff']:+.3f} ({v['ci95'][0]:+.3f} .. {v['ci95'][1]:+.3f})" for k, v in out[arm][f"l{lead}"]["ap_differences"].items()))
out["settings"] = dict(block_days=BLOCK, replicates=B, seed=SEED, seconds=round(time.time() - t0, 1))
jsave("f12d_trigger_free.json", out)
print(f"\n{time.time() - t0:.0f} s")
