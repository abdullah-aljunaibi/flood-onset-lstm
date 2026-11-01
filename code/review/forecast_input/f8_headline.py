"""E2 review, check (e): the headline numbers of every set from results/preds, with the engine (flood.evaluate, each set on ITS OWN split: the thresholds of the satellite-era sets come from THEIR training years 2000-10-01 .. 2008-09-30,
the test years are 2010-10-01 .. 2014-09-30) and with the reviewer's own pooled onset counts (own thresholds, own loops, no flood.evaluate), against summary_test and yardstick.table_test of results/forecast_input.json;
seed averages equal the day-by-day mean of the three seed files; the observed flow of every file equals the project's flow table; per-seed values against seed_ranges; every number of the brief's claim list;
how the script passes the split (read in the code: ARMS[arm]['thr'] = F.thresholds(D, split=SAT_SPLIT), evaluate_set(P, arm, part) = F.evaluate(P, thr, part, split=split)). About 40 seconds."""
import sys; sys.dont_write_bytecode = True
from f_common import *
import flood as F
from f_scoring import *

R = jload("results/forecast_input.json")
D = F.load_data(); ids = list(D)
THR = {"main": F.thresholds(D, split=SPLIT_MAIN), "sat": F.thresholds(D, split=SPLIT_SAT)}
LSTM = SETS_E2 + ["wq", "wqa", "wqf", "wqaF_sub_rawpred", "wqaF_sub_perfectpred", "wqaP_forecastpred"]
SIMPLE = ["base_linreg_wqaF", "base_rf_wqaF", "base_linreg_wq", "base_rf_wq", "base_linreg_wqa", "base_rf_wqa", "base_persistence"]
out = {"engine": {}, "own": {}, "per_seed": {}, "seed_average": {}}
# thresholds: own against the engine's
own_thr = {a: {g: float(np.quantile(D[g].q_mm.loc[SPLITS[a]["train"][0]:SPLITS[a]["train"][1]].values, 0.95)) for g in ids} for a in SPLITS}
out["thresholds_equal"] = {a: bool(all(own_thr[a][g] == THR[a][g] for g in ids)) for a in SPLITS}
print("thresholds (95th percentile of the training years of the arm), own against flood.thresholds: equal", out["thresholds_equal"], "; example Catawba main/sat:", round(own_thr["main"]["02137727"], 3), round(own_thr["sat"]["02137727"], 3))
dev = 0.0; EV = {}
for n in LSTM + SIMPLE:
    arm = ARM.get(n, "main")
    P = F.load_preds(f"lstm_{n}" if n in LSTM else n)
    E = F.evaluate(P, THR[arm], "test", split=SPLITS[arm]); EV[n] = E
    out["engine"][n] = {}
    for k in F.LEADS:
        s = E["summary"][f"l{k}"]
        mine = dict(nse_median=s["nse_median"], hits=s["onset"]["hits"], misses=s["onset"]["misses"], false_alarms=s["onset"]["false_alarms"], pod=s["onset"]["pod"], far=s["onset"]["far"], csi=s["onset"]["csi"])
        out["engine"][n][f"l{k}"] = mine
        j = R["summary_test"]["sets"][n][f"l{k}"] if n in LSTM else R["yardstick"]["table_test"][f"l{k}"][n]
        st = dict(nse_median=j["nse_median"], hits=j["onset"]["hits"], misses=j["onset"]["misses"], false_alarms=j["onset"]["false_alarms"], pod=j["onset"]["pod"], far=j["onset"]["far"], csi=j["onset"]["csi"])
        for c in mine:
            a, b = mine[c], st[c]
            if a != a and b is None: continue
            dev = max(dev, abs(a - b))
out["max_abs_dev_engine_vs_json"] = dev
print(f"engine recomputation (each set on its own split) against the JSON: {len(LSTM) + len(SIMPLE)} sets x 3 leads x 7 numbers, max abs deviation {dev}")
# own counts
dev_own = 0.0
for n in LSTM + SIMPLE:
    name = f"lstm_{n}" if n in LSTM else n
    out["own"][n] = {}
    for k in F.LEADS:
        # the set's arm decides thresholds and test years
        a, b = SPLITS[ARM.get(n, "main")]["test"]
        dates, basins, obs, pred = load_npz(name); thr = own_thr[ARM.get(n, "main")]
        test = part(dates, a, b); H = M = Fa = 0
        for i, g in enumerate(basins):
            o, f = obs[i], pred[i, :, k - 1]; issue = np.concatenate([np.full(k, np.nan), o[:-k]])
            ok = test & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g])
            H += int(((o > thr[g]) & (f > thr[g]) & ok).sum()); M += int(((o > thr[g]) & ~(f > thr[g]) & ok).sum()); Fa += int((~(o > thr[g]) & (f > thr[g]) & ok).sum())
        e = out["engine"][n][f"l{k}"]
        eq = (H, M, Fa) == (e["hits"], e["misses"], e["false_alarms"])
        out["own"][n][f"l{k}"] = dict(hits=H, misses=M, false_alarms=Fa, n_onsets=H + M, pod=H / (H + M), far=Fa / (H + Fa) if H + Fa else float("nan"), equal_engine=eq)
        dev_own = max(dev_own, abs(H / (H + M) - e["pod"]))
out["max_abs_dev_own_vs_engine"] = dev_own
out["all_counts_equal_engine"] = bool(all(v["equal_engine"] for n in out["own"] for v in out["own"][n].values()))
print(f"own pooled onset counts (own thresholds, own loops) equal the engine's in all {3 * len(out['own'])} set-lead cases: {out['all_counts_equal_engine']}; number of onsets by lead: main {[out['own']['wqa'][f'l{k}']['n_onsets'] for k in F.LEADS]}, satellite-era split {[out['own']['wqa_s'][f'l{k}']['n_onsets'] for k in F.LEADS]}")
# per seed, seed averages, observed flow
def own_pod(name, k, arm):
    dates, basins, obs, pred = load_npz(name); thr = own_thr[arm]; a, b = SPLITS[arm]["test"]; test = part(dates, a, b); H = M = 0
    for i, g in enumerate(basins):
        o, f = obs[i], pred[i, :, k - 1]; issue = np.concatenate([np.full(k, np.nan), o[:-k]]); ok = test & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g])
        H += int(((o > thr[g]) & (f > thr[g]) & ok).sum()); M += int(((o > thr[g]) & ~(f > thr[g]) & ok).sum())
    return H / (H + M)


def own_nse_median(name, k, arm):
    dates, basins, obs, pred = load_npz(name); a, b = SPLITS[arm]["test"]; test = part(dates, a, b); v = []
    for i in range(len(basins)):
        ok = test & np.isfinite(obs[i]) & np.isfinite(pred[i, :, k - 1]); o, f = obs[i][ok], pred[i, :, k - 1][ok]
        v.append(1 - np.sum((f - o) ** 2) / np.sum((o - o.mean()) ** 2))
    return float(np.median(v))


dev_json_seed = 0.0
for n in SETS_E2:
    name = f"lstm_{n}"; arm = ARM[n]
    pods = [[own_pod(f"{name}_s{s}", k, arm) for k in F.LEADS] for s in SEEDS]
    nses = [[own_nse_median(f"{name}_s{s}", k, arm) for k in F.LEADS] for s in SEEDS]
    j = R["seed_ranges"]["sets"][n]
    d = max(max(abs(pods[s][k - 1] - j[f"l{k}"]["by_seed"]["onset_pod"][s]) for s in range(3) for k in F.LEADS), max(abs(nses[s][k - 1] - j[f"l{k}"]["by_seed"]["nse_median"][s]) for s in range(3) for k in F.LEADS))
    out["per_seed"][n] = dict(onset_pod_by_seed=pods, nse_median_by_seed=nses, max_dev_vs_json=d); dev_json_seed = max(dev_json_seed, d)
    A = load_npz(name); S = [load_npz(f"{name}_s{s}") for s in SEEDS]
    mean = np.mean([x[3] for x in S], axis=0)
    out["seed_average"][n] = dict(max_abs_diff=float(np.nanmax(np.abs(mean - A[3]))), nan_pattern_equal=bool(np.array_equal(np.isnan(mean), np.isnan(A[3]))), obs_equal_across_seeds=bool(all(np.array_equal(S[0][2], x[2], equal_nan=True) for x in S)), obs_equal_to_average=bool(np.array_equal(S[0][2], A[2], equal_nan=True)))
    dates, basins, obs, pred = A
    out["seed_average"][n]["obs_equal_project_flow"] = bool(all(np.allclose(obs[i], D[g].q_mm.reindex(dates).values, equal_nan=True, rtol=0, atol=1e-6) for i, g in enumerate(basins)))
out["max_dev_per_seed_vs_json"] = dev_json_seed
print(f"per-seed onset shares and median NSE (3 seeds x 3 leads x 10 sets) against seed_ranges of the JSON: max deviation {dev_json_seed:.1e}")
print("seed averages equal the mean of the three seed files (max |diff|):", {n: f"{v['max_abs_diff']:.1e}" for n, v in out["seed_average"].items()})
print("obs equal across seeds, in the average and equal to the project's flow table, every set:", all(v["obs_equal_across_seeds"] and v["obs_equal_to_average"] and v["obs_equal_project_flow"] for v in out["seed_average"].values()))
for n in ("wq", "wqa", "wqf", "wqaF_sub_rawpred", "wqaF_sub_perfectpred", "wqaP_forecastpred"):
    A = load_npz(f"lstm_{n}"); out["seed_average"][n] = dict(obs_equal_project_flow=bool(all(np.allclose(A[2][i], D[g].q_mm.reindex(A[0]).values, equal_nan=True, rtol=0, atol=1e-6) for i, g in enumerate(A[1]))))
for n in SIMPLE:
    A = load_npz(n); out["seed_average"][n] = dict(obs_equal_project_flow=bool(all(np.allclose(A[2][i], D[g].q_mm.reindex(A[0]).values, equal_nan=True, rtol=0, atol=1e-6) for i, g in enumerate(A[1]))))
# the claims of the brief
e = out["engine"]; o = out["own"]
pc = lambda n, k: round(100 * o[n][f"l{k}"]["pod"], 1)
ns = lambda n, k: round(e[n][f"l{k}"]["nse_median"], 2)
claims = {
    "wqa onsets 51.2/20.6/11.6": [pc("wqa", k) for k in F.LEADS], "wqa NSE 0.77/0.38/0.25": [ns("wqa", k) for k in F.LEADS],
    "wqaF_sub onsets 44.5/38.8/36.9": [pc("wqaF_sub", k) for k in F.LEADS], "wqaF_sub NSE 0.69/0.53/0.45": [ns("wqaF_sub", k) for k in F.LEADS],
    "control wqaF_clim onsets 14.4/3.9/4.2": [pc("wqaF_clim", k) for k in F.LEADS], "control NSE 0.60/0.19/0.03": [ns("wqaF_clim", k) for k in F.LEADS],
    "wqF_sub onsets 31.4/36.3/36.4": [pc("wqF_sub", k) for k in F.LEADS], "wqaP onsets 66.1/69.2/71.4": [pc("wqaP", k) for k in F.LEADS], "wqaP NSE 0.80/0.78/0.77": [ns("wqaP", k) for k in F.LEADS],
    "E2b wqa_s onsets 40.9/13.9/7.8": [pc("wqa_s", k) for k in F.LEADS], "E2b wqaF_s onsets 61.2/49.1/44.2": [pc("wqaF_s", k) for k in F.LEADS], "E2b NSE wqa_s 0.71/0.39/0.28": [ns("wqa_s", k) for k in F.LEADS], "E2b NSE wqaF_s 0.77/0.62/0.52": [ns("wqaF_s", k) for k in F.LEADS],
    "E2b control wqaF_clim_s 40.6/14.8/8.2": [pc("wqaF_clim_s", k) for k in F.LEADS], "E2b wqaP_s": [pc("wqaP_s", k) for k in F.LEADS], "E2b wq_s": [pc("wq_s", k) for k in F.LEADS],
    "lead-1 change wqaF_sub minus wqa (points)": round(100 * (o["wqaF_sub"]["l1"]["pod"] - o["wqa"]["l1"]["pod"]), 1)}
out["claims"] = claims
print("claims of the brief against my own numbers:")
for k, v in claims.items(): print(f"   {k}: mine {v}")
verd = {}
for n in ("wqa", "wqaF_sub", "wqaF_clim", "wqaP", "wqaF_s", "wqaP_s", "wqa_s", "wq_s", "wqF_sub", "wqaF_clim_s"):
    verd[n] = [("GO" if (o[n][f"l{k}"]["pod"] >= 0.70 and o[n][f"l{k}"]["far"] <= 0.30) else "CAUTION" if o[n][f"l{k}"]["hits"] / (o[n][f"l{k}"]["hits"] + o[n][f"l{k}"]["misses"] + o[n][f"l{k}"]["false_alarms"]) >= 0.40 else "NOT YET") for k in F.LEADS]
out["verdict_onsets_plain_level_own"] = verd
print("onset verdicts at the plain level by the stored criteria (GO: pod >= 0.70 and far <= 0.30; CAUTION: csi >= 0.40), own application:", verd)
print("   (the stored verdicts: wqaP CAUTION at all leads, wqaP_s CAUTION; everything else NOT YET)")
jsave("f8_headline.json", out)
