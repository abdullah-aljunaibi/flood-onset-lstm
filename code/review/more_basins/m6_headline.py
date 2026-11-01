"""Check (b): the headline numbers of every new prediction set, per lead, by two routes, against results/more_basins.json.

Route 1: the engine (flood.load_preds, flood.thresholds on the stored 12-basin tables, flood.evaluate).
Route 2: the reviewer's own code (m_common.load_npz, own 95th percentile of the training-year flow, own NSE and own pooled onset counting with the issue-day flow taken by
         array position).
Also: the stored sets hold exactly the 12 study basins (assert); obs of every set equals the project's observed flow bit for bit; every seed average equals the mean of its
seed files; the metadata of each file (script, quick mode, training set size and digest, config); the controls re-scored the same way (lstm_wq, lstm_w, lstm_wqf, lobo_*);
the warning rule 'own level' of 05_lobo.py for the sets without a gauge, recomputed; and the numbers quoted in the task description laid beside the recomputed ones.
Run: python m6_headline.py
"""
import hashlib

from m_common import *

F = engine()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
ids12 = F.study_ids()
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
flow = {g: D12[g].q_mm for g in ids12}
thr_own = own_thresholds(ids12, flow)
thr_eng = F.thresholds(D12)
res = dict(thresholds=dict(own_equals_engine_max_abs_diff=float(max(abs(thr_own[g] - thr_eng[g]) for g in ids12)), values=thr_eng))
print("thresholds: own 95th percentile of the training-year flow vs flood.thresholds, max abs diff", res["thresholds"]["own_equals_engine_max_abs_diff"])

NEW = ["lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic"]
CTRL = ["lstm_wq", "lstm_w", "lstm_wqf", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w"]
SEEDS = [0, 1, 2]
dates_expected = pd.date_range(*PERIOD)
res["integrity"] = {}
# ---- the sets: basins, dates, obs, metadata, seed averages ---------------------------------------------------------------------------------------
cfg04 = json.loads((PROJ / "results" / "lstm_all.json").read_text())["config"]
for n in NEW:
    names = [n] + [f"{n}_s{s}" for s in SEEDS]
    rec = {}
    for m in names:
        z = load_npz(m)
        meta = z["meta"]
        exp_conf = dict(cfg04, updates=5 * cfg04["updates"], eval_every=5 * cfg04["eval_every"]) if n == "lstm63eq_wq" else cfg04
        rec[m] = dict(basins_equal_the_12=sorted(z["basins"]) == sorted(ids12), n_basins=len(z["basins"]), dates_equal_period=bool(z["dates"].equals(dates_expected)), shape_obs=list(z["obs"].shape), shape_pred=list(z["pred"].shape),
                      obs_bitwise_equal_project_flow=all(np.array_equal(z["obs"][i], flow[g].values) for i, g in enumerate(z["basins"])),
                      script=meta.get("script"), quick_mode=meta.get("quick_mode"), trained_on=meta.get("trained_on"), seeds=meta.get("seeds"), static=len(meta.get("static", [])), scored_on=meta.get("scored_on"),
                      config_equals_expected=meta.get("config") == exp_conf, n_nan_pred=int(np.isnan(z["pred"]).sum()), pred_finite_nonneg=bool(np.all(z["pred"][np.isfinite(z["pred"])] >= 0)))
    # the seed average is the day-by-day mean of its three seed files
    zs = [load_npz(f"{n}_s{s}") for s in SEEDS]
    za = load_npz(n)
    mean3 = (zs[0]["pred"] + zs[1]["pred"] + zs[2]["pred"]) / 3
    rec["seed_average"] = dict(bitwise_equal_to_mean_of_seed_files=bool(np.array_equal(za["pred"], mean3, equal_nan=True)), max_abs_diff=float(np.nanmax(np.abs(za["pred"] - mean3))),
                               same_nan_pattern=bool(np.array_equal(np.isnan(za["pred"]), np.isnan(mean3))), obs_equal_seed0=bool(np.array_equal(za["obs"], zs[0]["obs"])),
                               seeds_differ_from_each_other=bool(not np.array_equal(zs[0]["pred"], zs[1]["pred"], equal_nan=True)))
    res["integrity"][n] = rec
    bad = [k for k, v in rec.items() if isinstance(v, dict) and any(v.get(c) is False for c in ("basins_equal_the_12", "dates_equal_period", "obs_bitwise_equal_project_flow", "config_equals_expected"))]
    print(f"{n:20s} files: {names[0]} + {len(SEEDS)} seeds | 12 basins only: {all(rec[m]['basins_equal_the_12'] and rec[m]['n_basins'] == 12 for m in names)} | obs = project flow, bit for bit: "
          f"{all(rec[m]['obs_bitwise_equal_project_flow'] for m in names)} | config as expected: {all(rec[m]['config_equals_expected'] for m in names)} | script {rec[n]['script']} quick {rec[n]['quick_mode']} "
          f"trained_on {rec[n]['trained_on']} | seed average = mean of seeds, bitwise: {rec['seed_average']['bitwise_equal_to_mean_of_seed_files']} (max abs diff {rec['seed_average']['max_abs_diff']:.1e}) | problems: {bad}")
for n in CTRL:
    z = load_npz(n)
    res["integrity"][n] = dict(basins_equal_the_12=sorted(z["basins"]) == sorted(ids12), obs_bitwise_equal_project_flow=all(np.array_equal(z["obs"][i], flow[g].values) for i, g in enumerate(z["basins"])),
                               script=z["meta"].get("script"))
print("controls: 12 basins and obs = project flow:", {n: (v["basins_equal_the_12"], v["obs_bitwise_equal_project_flow"]) for n, v in res["integrity"].items() if n in CTRL})

# ---- scores: both routes against the results file --------------------------------------------------------------------------------------------------
S = J["summary_test"]["sets"]
rows, worst = {}, {"own_vs_engine": 0.0, "own_vs_json": 0.0, "engine_vs_json": 0.0}
for n in NEW + CTRL:
    z = load_npz(n)
    E = F.evaluate(F.load_preds(n), thr_eng)["summary"]
    rows[n] = {}
    for k in LEADS:
        o = own_pooled(z, thr_own, k)
        e, js = E[f"l{k}"], S[n][f"l{k}"]
        rec = dict(nse_median=o["nse_median"], onset=o["onset"], all=o["all"])
        d_oe = max(abs(o["nse_median"] - e["nse_median"]), *[abs(o[kind][c] - e[kind][c]) for kind in ("onset", "all") for c in ("hits", "misses", "false_alarms", "pod", "far", "csi") if o[kind][c] == o[kind][c] or e[kind][c] == e[kind][c]])
        d_oj = max(abs(o["nse_median"] - js["nse_median"]), *[abs(o["onset"][c] - js["onset"][c]) for c in ("hits", "misses", "false_alarms", "pod", "far", "csi") if js["onset"][c] is not None],
                   *[abs(o["all"][c] - js["all"][c]) for c in ("hits", "misses", "false_alarms", "pod", "far", "csi") if js["all"][c] is not None])
        d_ej = max(abs(e["nse_median"] - js["nse_median"]), *[abs(e["onset"][c] - js["onset"][c]) for c in ("hits", "misses", "false_alarms", "pod") if js["onset"][c] is not None])
        rec.update(dev_own_vs_engine=d_oe, dev_own_vs_json=d_oj, dev_engine_vs_json=d_ej)
        rows[n][f"l{k}"] = rec
        worst["own_vs_engine"], worst["own_vs_json"], worst["engine_vs_json"] = max(worst["own_vs_engine"], d_oe), max(worst["own_vs_json"], d_oj), max(worst["engine_vs_json"], d_ej)
res["scores"] = {n: {l: dict(nse_median=v["nse_median"], onset_hits=v["onset"]["hits"], onset_misses=v["onset"]["misses"], onset_false_alarms=v["onset"]["false_alarms"], onset_pod=v["onset"]["pod"], onset_far=v["onset"]["far"],
                             onset_csi=v["onset"]["csi"], all_pod=v["all"]["pod"], all_far=v["all"]["far"], dev_own_vs_engine=v["dev_own_vs_engine"], dev_own_vs_json=v["dev_own_vs_json"], dev_engine_vs_json=v["dev_engine_vs_json"]) for l, v in r.items()} for n, r in rows.items()}
res["max_abs_deviation"] = worst
print("\nsets x leads compared:", len(NEW + CTRL) * 3, "| max abs deviation: own code vs engine", worst["own_vs_engine"], "| own code vs results/more_basins.json", worst["own_vs_json"], "| engine vs json", worst["engine_vs_json"])
print("\nHEADLINE NUMBERS (test years, the 12 rivers; observed high-flow level): median NSE | onsets warned (hits of 611, share) | false share of onset warnings")
for n in NEW + CTRL:
    print(f"{n:22s}", *[f"| l{k}: {rows[n][f'l{k}']['nse_median']:.4f} {rows[n][f'l{k}']['onset']['hits']:3d}/{rows[n][f'l{k}']['onset']['hits'] + rows[n][f'l{k}']['onset']['misses']} {100 * rows[n][f'l{k}']['onset']['pod']:5.2f}% far {100 * rows[n][f'l{k}']['onset']['far']:4.1f}%" for k in LEADS])
# seed files too
res["seed_files"] = {}
for n in NEW:
    for s in SEEDS:
        z = load_npz(f"{n}_s{s}")
        E = F.evaluate(F.load_preds(f"{n}_s{s}"), thr_eng)["summary"]
        res["seed_files"][f"{n}_s{s}"] = {f"l{k}": dict(nse_median_own=own_pooled(z, thr_own, k)["nse_median"], nse_median_engine=E[f"l{k}"]["nse_median"], onset_pod_own=own_pooled(z, thr_own, k)["onset"]["pod"], onset_pod_engine=E[f"l{k}"]["onset"]["pod"],
                                                   onset_hits_own=own_pooled(z, thr_own, k)["onset"]["hits"]) for k in LEADS}
dev = max(abs(v[l]["nse_median_own"] - v[l]["nse_median_engine"]) + abs(v[l]["onset_pod_own"] - v[l]["onset_pod_engine"]) for v in res["seed_files"].values() for l in v)
res["seed_files_max_dev_own_vs_engine"] = dev
print("\nsingle seed files: own vs engine max deviation", dev)

# ---- the claims of the task description laid beside the recomputed numbers -----------------------------------------------------------------------------
def g(n, k, key):
    r = rows[n][f"l{k}"]
    return r["nse_median"] if key == "nse" else 100 * r["onset"]["pod"]
claims = [("63 rivers with gauge, median NSE, lead 1", 0.604, g("lstm63_wq", 1, "nse")), ("63 rivers with gauge, onsets warned %, lead 1", 17.3, g("lstm63_wq", 1, "on")),
          ("stored lstm_wq NSE", 0.590, g("lstm_wq", 1, "nse")), ("stored lstm_wq onsets %", 18.7, g("lstm_wq", 1, "on")),
          ("63 rivers x5, NSE", 0.610, g("lstm63eq_wq", 1, "nse")), ("63 rivers x5, onsets %", 17.7, g("lstm63eq_wq", 1, "on")),
          ("no gauge 63, NSE", 0.453, g("lstm63_w", 1, "nse")), ("no gauge 63, onsets %", 18.0, g("lstm63_w", 1, "on")), ("lstm_w NSE", 0.443, g("lstm_w", 1, "nse")), ("lstm_w onsets %", 17.2, g("lstm_w", 1, "on")),
          ("never seen 50, with properties NSE", 0.376, g("unseen50_w_static", 1, "nse")), ("never seen 50, without properties NSE", 0.434, g("unseen50_w_nostatic", 1, "nse")),
          ("lobo 11 with properties NSE", 0.216, g("lobo_w_static", 1, "nse")), ("lobo 11 without properties NSE", 0.313, g("lobo_w_nostatic", 1, "nse")),
          ("lead 2 gauge NSE 12 -> 63: from", 0.323, g("lstm_wq", 2, "nse")), ("lead 2 gauge NSE 12 -> 63: to", 0.345, g("lstm63_wq", 2, "nse")),
          ("lead 3 gauge NSE 12 -> 63: from", 0.252, g("lstm_wq", 3, "nse")), ("lead 3 gauge NSE 12 -> 63: to", 0.279, g("lstm63_wq", 3, "nse"))]
res["claims_vs_recomputed"] = [dict(claim=c, claimed=a, recomputed=b, rounds_to_claim=bool(abs(round(b, 3 if a < 1 and "%" not in c else 1) - a) < 1e-9)) for c, a, b in claims]
print("\nthe numbers of the task description against the recomputed ones:")
for c in res["claims_vs_recomputed"]:
    print(f"  {c['claim']:48s} claimed {c['claimed']:<7} recomputed {c['recomputed']:.4f}  rounds to the claim: {c['rounds_to_claim']}")

# ---- the 'own level' warning rule of 05_lobo.py (for the sets without a gauge), recomputed ------------------------------------------------------------
def own_level_pod(npz, lead, part="test"):
    """Warn when the forecast is above the 95th percentile of the model's OWN forecasts of that basin and lead in the training years (05_lobo.own_levels), onsets pooled."""
    dates, obs, pred = npz["dates"], npz["obs"], npz["pred"][:, :, lead - 1]
    tr, sel = window(dates, "train"), window(dates, part)
    h = m = f = 0
    for i, gid in enumerate(npz["basins"]):
        x = pred[i][tr]
        lev = float(np.nanquantile(x[np.isfinite(x)], 0.95))
        trig = lev / thr_own[gid]
        o, s = obs[i], pred[i]
        ok = sel & np.isfinite(o) & np.isfinite(s)
        issue = np.full(len(o), np.nan)
        issue[lead:] = o[:-lead]
        low = issue <= thr_own[gid]
        hi, warn = o > thr_own[gid], s > trig * thr_own[gid]
        mm = ok & low
        h += int((hi & warn & mm).sum()); m += int((hi & ~warn & mm).sum()); f += int((~hi & warn & mm).sum())
    return dict(hits=h, misses=m, false_alarms=f, pod=h / (h + m), far=f / (h + f) if h + f else float("nan"))


UN = ["lstm_w", "lstm63_w", "lobo_ref_w_nostatic", "lobo_w_static", "lobo_w_nostatic", "lobo_rf_w", "unseen50_w_static", "unseen50_w_nostatic"]
res["own_level"] = {}
mx = 0.0
for n in UN:
    z = load_npz(n)
    res["own_level"][n] = {}
    for k in LEADS:
        mine = own_level_pod(z, k)
        js = J["unseen"]["sets"][n]["own_level"][f"l{k}"]["onset"]
        d = max(abs(mine["pod"] - js["pod"]), abs(mine["hits"] - js["hits"]), abs(mine["false_alarms"] - js["false_alarms"]))
        mx = max(mx, d)
        res["own_level"][n][f"l{k}"] = dict(mine, json_pod=js["pod"], json_hits=js["hits"], dev=d)
res["own_level_max_dev_vs_json"] = mx
print("\nown-level onset warnings (rule of 05_lobo.py) recomputed for", len(UN), "sets x 3 leads: max deviation from the results file", mx)
for n in UN:
    print(f"  {n:22s}", *[f"| l{k}: {res['own_level'][n][f'l{k}']['hits']:3d} hits {100 * res['own_level'][n][f'l{k}']['pod']:5.2f}% far {100 * res['own_level'][n][f'l{k}']['far']:4.1f}%" for k in LEADS])
ok = (worst["own_vs_engine"] < 1e-9 and worst["own_vs_json"] < 1e-9 and mx < 1e-9 and all(v[c] for n in NEW for v in [res["integrity"][n]["seed_average"]] for c in ("bitwise_equal_to_mean_of_seed_files",)))
res["all_pass"] = bool(ok)
print("\nALL PASS:", ok)
save("m6_headline", res)
sys.exit(0 if ok else 1)
