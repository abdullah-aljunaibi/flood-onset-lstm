"""d2a: the critic's first, shorter bootstrap (10:3x), kept because the critique (CRITIQUE.md of 12:30) quotes its numbers: 41.5% to 61.7%, 27.9% to 48.4%, 27.2% to 46.0% for the AORC set and the forecast set.

It draws the 9 test water years and the 12 basins with replacement 4,000 times (seed 1) from the per-water-year-and-basin onset counts of the stored forecasts and prints the pooled share of onsets warned and the 2.5th and
97.5th percentiles, for the six sets whose ranges the report quotes. It reads the forecasts with the engine's loader (flood.load_preds) and the levels with the engine (flood.thresholds); d2_headline_and_ranges.py
repeats the work with the critic's own scoring from the .npz files (other random numbers, more statistics) and is the one the review record cites. Output: stdout and OUT/d2a_first_pass_bootstrap.json
"""
from dcommon import *
import flood as F

D = F.load_data()
thr = F.thresholds(D)
a, b = F.SPLIT['test']


def cells(name, lead):
    """hits[y, basin], onsets[y, basin] per water year of the TARGET day; onsets = a high-flow day forecast while the river was at or below the level `lead` days before."""
    P = F.load_preds(name)
    hits, ons = {}, {}
    gids = list(P.keys())
    for g in gids:
        p = P[g]
        q = p.loc[a:b]
        was_low = (p.obs.shift(lead) <= thr[g]).loc[a:b]
        ok = (q.obs.notna() & q[f"l{lead}"].notna()).values
        o = (q.obs > thr[g]).values
        s = (q[f"l{lead}"] > thr[g]).values
        sel = ok & was_low.values & o
        idx = q.index[sel]
        wy = np.array([d.year + (1 if d.month >= 10 else 0) for d in idx])
        h = s[sel]
        for y in range(2006, 2015):
            m = wy == y
            hits[(y, g)] = int(h[m].sum())
            ons[(y, g)] = int(m.sum())
    return gids, hits, ons


def boot(name, lead, n=4000, seed=1):
    gids, hits, ons = cells(name, lead)
    ys = list(range(2006, 2015))
    H = np.array([[hits[(y, g)] for g in gids] for y in ys])
    O = np.array([[ons[(y, g)] for g in gids] for y in ys])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yi, bi = rng.integers(0, len(ys), len(ys)), rng.integers(0, len(gids), len(gids))
        h, o = H[np.ix_(yi, bi)].sum(), O[np.ix_(yi, bi)].sum()
        out.append(h / o if o else np.nan)
    out = np.array(out)
    return H.sum() / O.sum(), np.nanpercentile(out, 2.5), np.nanpercentile(out, 97.5), int(O.sum())


res = {}
for name, lead in [('lstm_wq', 1), ('lstm_wqn', 1), ('lstm_wqf', 1), ('lstm_wqa', 1), ('lstm_wqaF_sub', 2), ('lstm_wqaF_sub', 3)]:
    v, lo, hi, n = boot(name, lead)
    res[f"{name}.l{lead}"] = dict(pooled=v, lo=lo, hi=hi, n_onsets=n)
    print(f"{name:14s} lead {lead}: pooled {100 * v:.1f}% (n onsets {n})  independent bootstrap 95%: {100 * lo:.1f}% to {100 * hi:.1f}%")
jsave("d2a_first_pass_bootstrap.json", res)
