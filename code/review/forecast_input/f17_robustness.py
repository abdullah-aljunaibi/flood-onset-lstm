"""E2 review, 'how much to trust the gains': three views of the uncertainty of the headline differences that the project's own rules (3 trainings, 12 basins, Wilcoxon) do not give. Only the saved prediction sets are read.
(1) a moving-block bootstrap over TARGET DAYS (blocks of 30 consecutive days, drawn with replacement, 4000 replicates, seed 20261008) of the test period of each set's own split: the difference in the share of the
    onsets warned (points), in the false-warning share, and in the median NSE over the 12 basins, with the 95% interval (2.5th to 97.5th percentile) and the share of replicates above zero. Blocks keep the correlation
    of neighbouring days and of the basins of one storm that the paired sign test over onsets ignores.
(2) by water year (1 October to 30 September) of the test period: the difference in the share of onsets warned in each year, and in how many years it is above zero.
(3) leave one basin out: the pooled difference in onsets warned without each basin in turn (range), and in how many of the 12 basins the per-basin share of onsets warned is higher.
Pairs: E2a wqaF_sub - wqa, wqaF_sub - wqaF_clim, wqaP - wqa, wqaP - wqaF_sub, and the simple models with and without the forecast; E2b wqaF_s - wqa_s, wqaF_s - wqaF_clim_s, wqaP_s - wqaF_s.
Uses the seed-averaged forecasts (lstm_<set>) the headline numbers use. About 1 minute."""
import sys; sys.dont_write_bytecode = True
import time
from f_common import *
from f_scoring import *
import flood as F

t0 = time.time()
B, BLOCK, SEED = 4000, 30, 20261008
PAIRS = [("main", "lstm_wqaF_sub", "lstm_wqa"), ("main", "lstm_wqaF_sub", "lstm_wqaF_clim"), ("main", "lstm_wqaP", "lstm_wqa"), ("main", "lstm_wqaP", "lstm_wqaF_sub"), ("main", "lstm_wqF_sub", "lstm_wq"),
         ("main", "base_linreg_wqaF", "base_linreg_wqa"), ("main", "base_rf_wqaF", "base_rf_wqa"), ("main", "lstm_wqaF_sub", "base_rf_wqaF"), ("main", "lstm_wqaF_sub", "base_linreg_wqaF"),
         ("sat", "lstm_wqaF_s", "lstm_wqa_s"), ("sat", "lstm_wqaF_s", "lstm_wqaF_clim_s"), ("sat", "lstm_wqaP_s", "lstm_wqaF_s")]
rng = np.random.default_rng(SEED)

def pair_arrays(T, C, lead):
    """Per basin the aligned rows of the test period (target day, observed, issue-day flow, forecast T, forecast C); only rows where all four exist."""
    dT, bT, oT, pT = load_npz(T); dC, bC, oC, pC = load_npz(C)
    assert (dT == dC).all() and bT == bC and np.array_equal(oT, oC, equal_nan=True), "the two sets must share dates, basins and observations"
    sp = split_of(T); a, b = sp["test"]; test = part(dT, a, b); thr = thresholds(bT, dT, sp)
    rows = []
    for i, g in enumerate(bT):
        o = oT[i]; issue = np.concatenate([np.full(lead, np.nan), o[:-lead]]); fT, fC = pT[i, :, lead - 1], pC[i, :, lead - 1]
        ok = test & np.isfinite(o) & np.isfinite(fT) & np.isfinite(fC)
        rows.append(dict(g=g, date=dT[ok], o=o[ok], issue=issue[ok], fT=fT[ok], fC=fC[ok], thr=thr[g]))
    return rows, pd.Timestamp(a), pd.Timestamp(b)

def block_sums(rows, a, nblocks, lead):
    """Per block: onset events (n), hits of T and C, warnings of T and C on non-high onset days (false alarms), and per basin the sums for the NSE."""
    ev = np.zeros((nblocks, 5))
    nse = np.zeros((len(rows), nblocks, 5))                                     # n, sum o, sum o^2, sse T, sse C
    for j, r in enumerate(rows):
        blk = ((r["date"] - a).days // BLOCK).astype(int)
        on = r["issue"] <= r["thr"]; high = r["o"] > r["thr"]; wT = r["fT"] > r["thr"]; wC = r["fC"] > r["thr"]
        for bi in range(nblocks):
            m = blk == bi
            if not m.any():
                continue
            mo = m & on
            ev[bi] += [(mo & high).sum(), (mo & high & wT).sum(), (mo & high & wC).sum(), (mo & ~high & wT).sum(), (mo & ~high & wC).sum()]
            o = r["o"][m]
            nse[j, bi] = [m.sum(), o.sum(), (o ** 2).sum(), ((r["fT"][m] - o) ** 2).sum(), ((r["fC"][m] - o) ** 2).sum()]
    return ev, nse

def stat(ev, nse):
    """pod and far of T and C, and the median NSE of T and C, from block sums (ev: [..., 5], nse: [basins, ..., 5] already summed over the drawn blocks)."""
    n, hT, hC, fT, fC = ev[..., 0], ev[..., 1], ev[..., 2], ev[..., 3], ev[..., 4]
    podT, podC = hT / n, hC / n
    farT = np.where(hT + fT > 0, fT / np.maximum(hT + fT, 1), np.nan); farC = np.where(hC + fC > 0, fC / np.maximum(hC + fC, 1), np.nan)
    N, S, SS, ET, EC = [nse[..., k] for k in range(5)]
    sst = SS - S ** 2 / N
    nseT, nseC = 1 - ET / sst, 1 - EC / sst
    return podT - podC, farT - farC, np.median(nseT, axis=0) - np.median(nseC, axis=0), podT, podC

out = {}
print(f"block bootstrap ({B} replicates, blocks of {BLOCK} days), by-water-year and leave-one-basin-out views; differences first minus second; onsets in points, NSE in units")
for arm, T, C in PAIRS:
    key = f"{T[5:] if T.startswith('lstm_') else T} - {C[5:] if C.startswith('lstm_') else C}"
    out[key] = {}
    for lead in KS:
        rows, a, b = pair_arrays(T, C, lead)
        nblocks = int(((b - a).days // BLOCK) + 1)
        ev, nse = block_sums(rows, a, nblocks, lead)
        d_pod, d_far, d_nse, podT, podC = stat(ev.sum(0), nse.sum(1))
        idx = rng.integers(0, nblocks, size=(B, nblocks)); cnt = np.stack([np.bincount(r, minlength=nblocks) for r in idx])        # [B, nblocks]
        evb = cnt @ ev                                                                  # [B, 5]
        nsb = np.einsum("rb,jbk->jrk", cnt.astype(float), nse)                          # [basins, B, 5]
        bp, bf, bn, _, _ = stat(evb, nsb)
        ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        # by water year
        wy = {}
        for r in rows:
            yr = np.where(r["date"].month >= 10, r["date"].year + 1, r["date"].year)
            on = r["issue"] <= r["thr"]; high = r["o"] > r["thr"]
            for y in np.unique(yr):
                m = (yr == y) & on & high
                h = wy.setdefault(int(y), [0, 0, 0]); h[0] += int(m.sum()); h[1] += int((m & (r["fT"] > r["thr"])).sum()); h[2] += int((m & (r["fC"] > r["thr"])).sum())
        yrs = {y: 100 * (v[1] - v[2]) / v[0] for y, v in sorted(wy.items()) if v[0] > 0}
        # leave one basin out and per basin
        per_basin = []
        for r in rows:
            on = (r["issue"] <= r["thr"]) & (r["o"] > r["thr"]); per_basin.append((int(on.sum()), int((on & (r["fT"] > r["thr"])).sum()), int((on & (r["fC"] > r["thr"])).sum())))
        pb = np.array(per_basin, float); tot = pb.sum(0)
        loo = [100 * ((tot[1] - pb[j, 1]) - (tot[2] - pb[j, 2])) / (tot[0] - pb[j, 0]) for j in range(len(rows))]
        higher = int(sum(1 for v in per_basin if v[1] > v[2])); lower = int(sum(1 for v in per_basin if v[1] < v[2]))
        out[key][f"l{lead}"] = dict(n_onsets=int(tot[0]), pod_points=100 * d_pod, pod_points_ci95=[100 * x for x in ci(bp)], share_replicates_pod_above_zero=float((bp > 0).mean()),
                                    far_points=100 * float(d_far), far_points_ci95=[100 * x for x in ci(bf)], nse=float(d_nse), nse_ci95=ci(bn), share_replicates_nse_above_zero=float((bn > 0).mean()),
                                    water_years=yrs, years_above_zero=int(sum(v > 0 for v in yrs.values())), n_years=len(yrs), leave_one_basin_out_points=[min(loo), max(loo)], basins_higher=higher, basins_lower=lower)
    print(f"\n{key}")
    for lead in KS:
        o = out[key][f"l{lead}"]
        print(f"   lead {lead}: onsets {o['pod_points']:+6.1f} points (95% {o['pod_points_ci95'][0]:+6.1f} .. {o['pod_points_ci95'][1]:+6.1f}; replicates above 0: {100 * o['share_replicates_pod_above_zero']:5.1f}%) | false-warning share {o['far_points']:+5.1f} ({o['far_points_ci95'][0]:+5.1f} .. {o['far_points_ci95'][1]:+5.1f}) | "
              f"NSE {o['nse']:+.3f} ({o['nse_ci95'][0]:+.3f} .. {o['nse_ci95'][1]:+.3f}; above 0: {100 * o['share_replicates_nse_above_zero']:5.1f}%) | water years above 0: {o['years_above_zero']}/{o['n_years']} | without one basin: {o['leave_one_basin_out_points'][0]:+.1f} .. {o['leave_one_basin_out_points'][1]:+.1f} | basins higher/lower {o['basins_higher']}/{o['basins_lower']}")
out["settings"] = dict(replicates=B, block_days=BLOCK, seed=SEED, seconds=round(time.time() - t0, 1))
jsave("f17_robustness.json", out)
print(f"\n{time.time() - t0:.0f} s")
