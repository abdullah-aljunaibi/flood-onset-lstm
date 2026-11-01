"""Information, not a verdict: does the skill of the saved 50-river networks on rivers they never saw fall with the distance to the nearest training gauge, over MORE than the 12 rivers?

The saved unseen50 networks (results/models, three seeds each, with and without basin properties) are run on the 15 south-eastern candidate rivers that the data audit left out because their flow record has
gaps (same selection rule as the 50 training rivers: HUC 03/06, snow fraction below 0.1, aridity below 1, 100-2500 km2; complete weather; flow missing on 0.4-26% of the days of the period). They
were in neither training set of any experiment. Each network forecasts them from weather (and properties) alone; the test-year NSE (1 day ahead) is scored on the days with an observed flow. Nothing is trained;
the leave-one-out networks of 05_lobo.py were not saved, so the gain over them cannot be computed for these rivers - only the dependence of the 50-river skill on distance.
Spearman rank correlation of NSE with the distance (km) to the nearest of the 50 training gauges, over the 12 study rivers, the 15 extra rivers and the 27 together.
Run: python m10_extra_rivers.py      (about 1 minute)
"""
import time

import torch
from scipy.stats import spearmanr

from m_common import *

F = engine()
t0 = time.time()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
ids12 = F.study_ids()
ids50 = J["design"]["rivers"]["other_50"]
aud = pd.read_csv(PROJ / "results" / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
extra = list(aud.index[~aud.complete.astype(bool)])
A = attribute_tables()
LAT, LON = A.gauge_lat.astype(float), A.gauge_lon.astype(float)
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
D = {g: raw_table(g) for g in extra}
D.update(D12)
rivers = ids12 + extra
Sprops = static_table(rivers, A)
test = (D[extra[0]].index >= pd.Timestamp(SPLIT["test"][0])) & (D[extra[0]].index <= pd.Timestamp(SPLIT["test"][1]))
km = {g: float(min(haversine_km(LAT[g], LON[g], LAT[h], LON[h]) for h in ids50)) for g in rivers}
km12 = {g: float(min(haversine_km(LAT[g], LON[g], LAT[h], LON[h]) for h in ids12 if h != g)) for g in rivers}


def forecast(name, seed):
    z = torch.load(PROJ / "results" / "models" / f"{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    assert not set(z["ids"]) & set(rivers), "a river scored here was in the training set"
    return run.predict(D, rivers, Sprops if cfg.static else None)


res = dict(extra_rivers=extra, n_extra=len(extra), per_river={})
nse = {}
for name in ("unseen50_w_static", "unseen50_w_nostatic"):
    P = [forecast(name, s) for s in (0, 1, 2)]
    for g in rivers:
        obs = P[0][g].obs.values
        avg = sum(p[g].l1.values for p in P) / 3
        ok = test & np.isfinite(obs) & np.isfinite(avg)
        cov = float(np.isfinite(obs)[test].mean())
        nse.setdefault(g, {})[name] = dict(nse=own_nse(obs[ok], avg[ok]), seeds=[own_nse(obs[ok & np.isfinite(p[g].l1.values)], p[g].l1.values[ok & np.isfinite(p[g].l1.values)]) for p in P], n_days=int(ok.sum()), coverage=cov)
# check against the stored forecasts of the 12 (the networks on the 12 are the stored forecasts: same numbers as the results file)
chk = max(abs(nse[g][n]["nse"] - J["per_river"]["nse"][n]["l1"][g]) for g in ids12 for n in ("unseen50_w_static", "unseen50_w_nostatic"))
res["check_the_12_equal_results_file_max_abs_diff"] = chk
print(f"the 12 study rivers scored here equal the results file (max abs diff {chk:.1e})")
print(f"\n{'river':10s} {'name':32s} {'km to nearest of the 50':>24s} {'km to nearest of the 12':>24s} {'obs days':>9s} {'NSE props':>10s} {'NSE no props':>13s}")
for g in rivers:
    nm = str(A.loc[g, "gauge_name"])[:32]
    print(f"{g:10s} {nm:32s} {km[g]:24.1f} {km12[g]:24.1f} {nse[g]['unseen50_w_static']['n_days']:9d} {nse[g]['unseen50_w_static']['nse']:10.3f} {nse[g]['unseen50_w_nostatic']['nse']:13.3f}{'   (study river)' if g in ids12 else ''}")
sp = lambda x, y: dict(zip(("rho", "p"), map(float, spearmanr(x, y))))
for label, sel in (("the 12 study rivers", ids12), ("the 15 extra rivers", extra), ("all 27", rivers)):
    for name in ("unseen50_w_static", "unseen50_w_nostatic"):
        v = [nse[g][name]["nse"] for g in sel]
        res.setdefault("spearman_nse_vs_km_to_nearest_training_gauge", {}).setdefault(label, {})[name] = dict(sp(v, [km[g] for g in sel]), n=len(sel), median_nse=float(np.median(v)))
bins = {"within 50 km": [g for g in rivers if km[g] <= 50], "50 to 100 km": [g for g in rivers if 50 < km[g] <= 100], "beyond 100 km": [g for g in rivers if km[g] > 100]}
res["by_distance_bin"] = {b: dict(n=len(gs), rivers=gs, median_nse_with_properties=float(np.median([nse[g]["unseen50_w_static"]["nse"] for g in gs])) if gs else None,
                                  median_nse_without_properties=float(np.median([nse[g]["unseen50_w_nostatic"]["nse"] for g in gs])) if gs else None) for b, gs in bins.items()}
res["per_river"] = {g: dict(name=str(A.loc[g, "gauge_name"]), study=g in ids12, km_nearest_of_the_50=km[g], km_nearest_of_the_12=km12[g], **{n: nse[g][n] for n in nse[g]}) for g in rivers}
res["seconds"] = round(time.time() - t0, 1)
print()
for label, d in res["spearman_nse_vs_km_to_nearest_training_gauge"].items():
    print(f"Spearman NSE vs km, {label:22s}: with properties rho {d['unseen50_w_static']['rho']:+.2f} (p {d['unseen50_w_static']['p']:.3f}), without {d['unseen50_w_nostatic']['rho']:+.2f} (p {d['unseen50_w_nostatic']['p']:.3f}), n = {d['unseen50_w_static']['n']}")
for b, d in res["by_distance_bin"].items():
    print(f"  {b:15s} n = {d['n']:2d}: median NSE with properties {d['median_nse_with_properties']}, without {d['median_nse_without_properties']}")
save("m10_extra_rivers", res)
print(f"({res['seconds']} s)")
