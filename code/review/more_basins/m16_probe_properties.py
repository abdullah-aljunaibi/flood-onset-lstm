"""Information: the builder's probe 'the 12 rivers' basin properties replaced by the median of the 50' (leak_probes.properties_replaced_by_the_median_of_the_50 in the results file) recomputed from the
saved networks with the reviewer's own code: for the three seeds of unseen50_w_static and lstm63_w, every one of the 12 is given the median property values of the 50 training rivers (as the model's
property scaler sees them), the 12 are forecast again, the seeds averaged, and the median NSE at 1 day ahead compared with the stored forecasts and with the results file.
What it shows: whether a model that reads properties uses the 12's own values (a river in training can be recognised by its properties; a river never seen should not need them).
Run: python m16_probe_properties.py"""
import torch

from m_common import *

F = engine()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
ids12 = F.study_ids()
ids50 = J["design"]["rivers"]["other_50"]
A = attribute_tables()
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
S12, S50 = static_table(ids12, A), static_table(ids50, A)
med50 = S50.median()
test = window(D12[ids12[0]].index, "test")
res = {}
for name in ("unseen50_w_static", "lstm63_w"):
    P0, P1 = [], []
    for s in (0, 1, 2):
        z = torch.load(PROJ / "results" / "models" / f"{name}_s{s}.pt", map_location="cpu", weights_only=True)
        cfg = F.Cfg(**z["cfg"])
        net = F.LSTMNet(len(cfg.dyn) + int(cfg.use_flow) + len(cfg.static), cfg)
        net.load_state_dict(z["state_dict"])
        net.eval()
        sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
        run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
        sdf_true = pd.DataFrame(z["static_values_scored"], index=list(cfg.static)).T
        sdf_med = pd.DataFrame([med50.values] * len(ids12), index=ids12, columns=STATIC)
        P0.append(run.predict(D12, ids12, sdf_true))
        P1.append(run.predict(D12, ids12, sdf_med))
    nse = lambda P: np.array([own_nse(D12[g].q_mm.values[test & np.isfinite(sum(p[g].l1.values for p in P))], (sum(p[g].l1.values for p in P) / 3)[test & np.isfinite(sum(p[g].l1.values for p in P))]) for g in ids12])
    a, b = nse(P0), nse(P1)
    mean_abs = {g: float(np.mean([np.nanmean(np.abs(P1[i][g].l1.values[test] - P0[i][g].l1.values[test])) for i in range(3)])) for g in ids12}
    maxchg = float(max(np.nanmax(np.abs(P1[i][g][["l1", "l2", "l3"]].values - P0[i][g][["l1", "l2", "l3"]].values)) for i in range(3) for g in ids12))      # the largest change of any single forecast of any seed
    per_river_mean_of_seed_max = {g: float(np.mean([np.nanmax(np.abs(P1[i][g][["l1", "l2", "l3"]].values - P0[i][g][["l1", "l2", "l3"]].values)) for i in range(3)])) for g in ids12}
    maxchg_json_definition = max(per_river_mean_of_seed_max.values())                                                                                       # what the results file reports: the mean over seeds of each river's largest change, largest river
    js = J["leak_probes"]["properties_replaced_by_the_median_of_the_50"][name]
    res[name] = dict(median_nse_true_properties=float(np.median(a)), median_nse_median_properties=float(np.median(b)), n_rivers_lower=int((b < a).sum()), largest_forecast_change_mm_day=maxchg, largest_change_by_the_results_files_definition=maxchg_json_definition,
                     json=dict(true=js["median_nse_with_true_properties"]["l1"], median=js["median_nse_with_the_median_of_the_50"]["l1"], n_lower=js["n_rivers_with_lower_nse_when_replaced"]["l1"], largest=js["largest_forecast_change_mm_day"]),
                     max_abs_diff_to_json=max(abs(float(np.median(a)) - js["median_nse_with_true_properties"]["l1"]), abs(float(np.median(b)) - js["median_nse_with_the_median_of_the_50"]["l1"]), abs(maxchg_json_definition - js["largest_forecast_change_mm_day"])),
                     median_change_in_nse=float(np.median(b - a)))
    print(f"{name}: median NSE {np.median(a):.4f} with the true properties of the 12, {np.median(b):.4f} with the median of the 50; {int((b < a).sum())} of 12 rivers lower; largest change of one forecast of one seed {maxchg:.2f} mm/day, by the results file's definition (mean over seeds of each river's largest change, largest river) {maxchg_json_definition:.2f} | results file: "
          f"{js['median_nse_with_true_properties']['l1']:.4f} -> {js['median_nse_with_the_median_of_the_50']['l1']:.4f}, {js['n_rivers_with_lower_nse_when_replaced']['l1']} lower, {js['largest_forecast_change_mm_day']:.2f} | max abs diff {res[name]['max_abs_diff_to_json']:.1e}")
save("m16_probe_properties", res)
