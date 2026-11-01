"""AORC review, check (d) part 2: what do the saved wqa networks do when the AORC input of the day is changed?  No training; the saved networks (flood.Run.predict), one thread.
 T1  Stage IV in place of AORC on the days I hold Stage IV for (builder's 50 days + my 28 days, all 12 basins): the value of aorc_to_midnight of day t (the rain of 00:00-24:00 local) is replaced by the Stage IV
     sum over the same hours (a product that is available about an hour after each hour: real time), raw, and multiplied by k = sum(A)/sum(S) of the sample (the level of AORC); the rest of the series is AORC. The
     warnings of the onsets whose issue day is in the sample (lead 1) with AORC, with Stage IV, and with the Stage IV level-matched; the same for wqa_rep.
 T2  a dose of hindsight on the whole record: A(lambda) = A_pre + lambda * A_post, A_post = AORC rain from local midnight to 12 UTC of day t+1 (what a retrospective scaling by a gauge day that ends at 12 UTC could
     carry into the hours before midnight), for lambda from -0.5 to +1: negative doses REMOVE a leak of that size, positive ones ADD one. The onsets warned 1 day ahead (own counts) as a function of lambda
     show how large a leak would have to be to explain the gain. The doses are set against the calibration of a7_hindsight.py.
 T3  the multiplicative form: the hours between 12 UTC of day t and local midnight divided by exp(c * share_post), share_post = A_post / (A_exp + A_post): a factor that grows with the rain after midnight.
About two minutes."""
import sys; sys.dont_write_bytecode = True
import time
from a_stage4_data import *
from a_scoring import TEST, TRAIN
import camels

t0 = time.time()
D, B = F.load_data(), F.basins()
Z = np.load(f"{OUT}/my_series_aorc.npz"); assert list(Z["basins"]) == ids
mine = {g: Z["midnight"][i] for i, g in enumerate(ids)}
D2 = {g: D[g].assign(aorc_to_midnight=mine[g], aorc_to_09=Z["to09"][i], aorc_utc_day=Z["utc"][i]) for i, g in enumerate(ids)}
DATES = D[ids[0]].index
THR = {g: float(np.quantile(D[g].q_mm.values[(DATES >= TRAIN[0]) & (DATES <= TRAIN[1])], 0.95)) for g in ids}
test_mask = np.asarray((DATES >= TEST[0]) & (DATES <= TEST[1]))
OBS = {g: D[g].q_mm.values for g in ids}
CAL = json.load(open(f"{OUT}/a7_hindsight.json"))["calibration"]
out = {"calibration": CAL}


def load_run(name, seed):
    z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"]); n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])


RUNS = {n: [load_run(n, s) for s in (0, 1, 2)] for n in ("wqa", "wqa_rep")}


def forecasts(name, tables, leads=(1,)):
    """seed-averaged forecasts (dict gauge -> {lead: array over DATES}) of a set on the given tables."""
    ps = [r.predict(tables, ids, B) for r in RUNS[name]]
    return {g: {k: np.mean([p[g][f"l{k}"].values for p in ps], axis=0) for k in leads} for g in ids}


def onset_mask(g):
    obs = OBS[g]; issue = np.concatenate([[np.nan], obs[:-1]])
    return (obs > THR[g]) & (issue <= THR[g]) & test_mask


def lead1(fc):
    """pooled onsets warned (own counts), hits, and per-basin hit masks."""
    H = M = 0; hits = {}
    for g in ids:
        on = onset_mask(g); w = fc[g][1] > THR[g]
        hits[g] = on & w; H += int((on & w).sum()); M += int((on & ~w).sum())
    return dict(hits=H, onsets=H + M, share=H / (H + M)), hits


def nse_med(fc):
    v = []
    for g in ids:
        o, s = OBS[g][test_mask], fc[g][1][test_mask]; ok = np.isfinite(s) & np.isfinite(o)
        v.append(1 - ((s[ok] - o[ok]) ** 2).sum() / ((o[ok] - o[ok].mean()) ** 2).sum())
    return float(np.median(v))


BASE = {n: forecasts(n, D2) for n in RUNS}
for n in RUNS:
    r, _ = lead1(BASE[n]); print(f"{n}: seed-averaged forecasts from my tables give {r['hits']} of {r['onsets']} onsets warned ({100 * r['share']:.1f}%), median NSE {nse_med(BASE[n]):.4f} (the saved set: {313 if n == 'wqa' else 315})")
out["base"] = {n: dict(lead1(BASE[n])[0], nse=nse_med(BASE[n])) for n in RUNS}

# ---- T1: Stage IV on the sampled days -----------------------------------------------------------------------------------------------------------------
df = pd.read_csv(f"{OUT}/a7_basin_days.csv", dtype={"gauge": str})
df["day"] = pd.to_datetime(df.day)
for g in ids:
    sub = df[df.gauge == g]
    assert np.allclose(sub.A_pre.values, mine[g][(sub.day - DATES[0]).dt.days.values], atol=1e-6), "A_pre of the Stage IV table is not my aorc_to_midnight"
k_level = float(df.A_pre.sum() / df.S_pre.sum())
print(f"\nT1: Stage IV in place of AORC on {df.day.nunique()} days x 12 basins = {len(df)} basin-days; the AORC level over Stage IV on them: k = {k_level:.3f}")
sample_days = sorted(df.day.unique())
cover = {g: set(pd.to_datetime(df[df.gauge == g].day)) for g in ids}
variants = {"AORC (as saved)": None, "Stage IV raw": 1.0, f"Stage IV x {k_level:.2f} (the level of AORC)": k_level}
T1 = {}
for name in RUNS:
    T1[name] = {}
    for vname, k in variants.items():
        if k is None: fc = BASE[name]
        else:
            tabs = {g: D2[g].copy() for g in ids}
            for g in ids:
                sub = df[df.gauge == g]; idxs = (sub.day - DATES[0]).dt.days.values
                col = tabs[g].columns.get_loc("aorc_to_midnight")
                tabs[g].iloc[idxs, col] = k * sub.S_pre.values
            fc = forecasts(name, tabs)
        # the onsets whose issue day t (target t+1) is a sampled day of that basin
        n_on = n_hit = n_hit_wq = 0; chg = []
        for g in ids:
            on = onset_mask(g); w = fc[g][1] > THR[g]
            tgt = np.flatnonzero(on)
            sel = [i for i in tgt if (DATES[i - 1] in cover[g])]
            n_on += len(sel); n_hit += int(sum(w[i] for i in sel))
        T1[name][vname] = dict(onsets_in_sample=n_on, warned=n_hit)
        # all sampled basin-days: mean lead-1 forecast on the day after (target t+1), relative to AORC
        rel = []
        for g in ids:
            for dday in cover[g]:
                i = (dday - DATES[0]).days + 1
                if i < len(DATES) and np.isfinite(fc[g][1][i]) and np.isfinite(BASE[name][g][1][i]): rel.append((fc[g][1][i], BASE[name][g][1][i], OBS[g][i]))
        rel = np.array(rel)
        T1[name][vname].update(mean_forecast=float(rel[:, 0].mean()), mean_forecast_aorc=float(rel[:, 1].mean()), mean_abs_change=float(np.abs(rel[:, 0] - rel[:, 1]).mean()), mean_observed=float(rel[:, 2].mean()),
                               n_basin_days=int(len(rel)), warnings_all_basin_days=int((rel[:, 0] > np.array([THR[g] for g in ids for _ in cover[g]])).sum()) if False else None)
    print(f"  {name}: onsets with the issue day in the sample: {T1[name]['AORC (as saved)']['onsets_in_sample']}")
    for vname in variants:
        v = T1[name][vname]
        print(f"     {vname:42s}: onsets warned {v['warned']:3d} of {v['onsets_in_sample']}  | mean 1-day forecast {v['mean_forecast']:.3f} mm/day (AORC {v['mean_forecast_aorc']:.3f}; observed {v['mean_observed']:.3f}), mean |change| {v['mean_abs_change']:.3f}")
out["T1"] = dict(k_level=k_level, n_days=int(df.day.nunique()), n_basin_days=int(len(df)), results=T1)
# the same onsets with the model WITHOUT the new input (lstm_wq), as the reference of what AORC adds on these days
dates_i = {g: DATES for g in ids}
z = np.load(f"{PROJ}/results/preds/lstm_wq.npz", allow_pickle=False); wq_pred = z["pred"]; wq_basins = [str(b) for b in z["basins"]]
n_wq = 0
for gi, g in enumerate(wq_basins):
    on = onset_mask(g); w = wq_pred[gi, :, 0] > THR[g]
    n_wq += int(sum(w[i] for i in np.flatnonzero(on) if DATES[i - 1] in cover[g]))
out["T1"]["onsets_warned_by_wq_in_sample"] = n_wq
print(f"     for reference, lstm_wq (no AORC) warns of {n_wq} of those onsets")

# ---- T2/T3: doses on the whole record -----------------------------------------------------------------------------------------------------------------
def pieces(g):
    """A_early (00:00 local .. 12 UTC of day t), A_exp (12 UTC .. 24:00 local), A_post (24:00 local .. 12 UTC of day t+1) for every day of the record."""
    t, x, fl, nv, nc = C.read_aorc(g)
    cs = np.concatenate([[0.0], np.cumsum(x)]); t0_ = t[0]
    pos = lambda inst: int(((np.datetime64(inst, "m") - t0_) / np.timedelta64(60, "m")))
    early, exp, post = np.zeros(len(DATES)), np.zeros(len(DATES)), np.zeros(len(DATES))
    m0 = [C.local_instant_utc(C.FIRST_DAY + i, 0, ZONE[g]) for i in range(len(DATES) + 1)]
    for i in range(len(DATES)):
        d64 = C.FIRST_DAY + i
        a0, a24 = pos(m0[i]), pos(m0[i + 1])
        z12, z12n = pos(np.datetime64(str(d64), "m") + np.timedelta64(12 * 60, "m")), pos(np.datetime64(str(d64 + 1), "m") + np.timedelta64(12 * 60, "m"))
        early[i] = cs[z12 + 1] - cs[a0 + 1]; exp[i] = cs[a24 + 1] - cs[z12 + 1]; post[i] = cs[min(z12n, len(x) - 1) + 1] - cs[a24 + 1]
    return early, exp, post
P = {g: pieces(g) for g in ids}
for g in ids:
    assert np.allclose(P[g][0] + P[g][1], mine[g], atol=1e-6), "early + exposed must equal the local-day sum"
print("\npieces of the local day built: early + exposed = my aorc_to_midnight on every day of the record (12 basins)")
LAMBDAS = [-0.5, -0.25, -0.12, -0.05, -0.028, 0.0, 0.05, 0.12, 0.25, 0.5, 1.0]
T2 = {n: {} for n in RUNS}
for name in RUNS:
    for lam in LAMBDAS:
        tabs = {g: D2[g].copy() for g in ids}
        for g in ids:
            tabs[g]["aorc_to_midnight"] = np.maximum(mine[g] + lam * P[g][2], 0.0)
        fc = forecasts(name, tabs); r, _ = lead1(fc)
        T2[name][str(lam)] = dict(hits=r["hits"], share=r["share"], nse=nse_med(fc))
    print(f"T2 {name}: onsets warned (of 611) by the dose lambda of AORC's rain from midnight to 12 UTC added to the day's input: " + "  ".join(f"{lam:+.3f}: {T2[name][str(lam)]['hits']}" for lam in LAMBDAS))
out["T2"] = T2
CS = [0.0, 0.035, 0.1, 0.25, 0.5, 1.0, 2.0]
T3 = {n: {} for n in RUNS}
for name in RUNS:
    for c in CS:
        tabs = {g: D2[g].copy() for g in ids}
        for g in ids:
            early, exp, post = P[g]; share = np.where(exp + post > 0, post / np.maximum(exp + post, 1e-12), 0.0)
            tabs[g]["aorc_to_midnight"] = early + exp * np.exp(-c * share)
        fc = forecasts(name, tabs); r, _ = lead1(fc)
        T3[name][str(c)] = dict(hits=r["hits"], share=r["share"], nse=nse_med(fc))
    print(f"T3 {name}: hours from 12 UTC to midnight divided by exp(c * share_post): " + "  ".join(f"c={c}: {T3[name][str(c)]['hits']}" for c in CS))
out["T3"] = T3
jsave("a8_hindsight_networks.json", out)
print("seconds:", round(time.time() - t0))
