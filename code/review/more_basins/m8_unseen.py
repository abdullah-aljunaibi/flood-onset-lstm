"""Check (d): rivers never seen, at scale (unseen50_w_static, unseen50_w_nostatic against results/lstm_lobo.json).

 1. NESTING: the CAMELS outlines of all 63 basins intersected in an equal-area projection (EPSG:5070): which pairs share area; 03471500 against 03473000 by outline and by the attribute tables
    (same river, smaller area, upstream coordinates, daily flows) - the reason the 51 become 50;
 2. the 12 never read in training, shown three ways: the training ids of every saved never-seen network exclude the 12 (from the files); flood.fit run with a dictionary in which the 12 are ABSENT
    gives the same weights as with all 63 tables present (one update, same seed); flood.fit run with a dictionary that RAISES on any access to the 12's tables or properties completes (the
    reviewer's own probe: the builder spoiled the records, this makes any read an error);
 3. PROXIMITY recomputed from camels_topo.txt (gauge_lat, gauge_lon) and the ten properties: median distance of each of the 12 gauges to the nearest training gauge for the 11 of 05_lobo.py, the 50
    and the 51; nearest training basin in standardised property space, by the script's definition (mean and spread of the training pool) and by the task's (mean and spread over the 63);
 4. the COMPARISON with 05_lobo.py: one training per held-out river (capped at 1000 updates, patience 4) against three seeds of one model (2000 updates, patience 6) that scores all 12:
    the budget caps in the 05 trainings and what the cap rule would keep on the 50-river curves; the three seeds one by one against the 05 numbers; the effect of seed averaging in the
    12-river models; per-river numbers;
 5. what the gain goes with: per river, the gain of the 50-river model over the leave-one-out model (NSE, 1 day ahead) against the distance to the nearest training gauge in the two designs,
    the nearest gauge's identity, and the change of that distance; Spearman and Kendall rank correlations; the rivers whose nearest training gauge did not come closer.
Run: python m8_unseen.py     (about 1-2 minutes; the two one-update fits take about a minute)
"""
import time

import torch
from scipy.stats import kendalltau, spearmanr, wilcoxon

from m_common import *

F = engine()
t0 = time.time()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
LOBO = json.loads((PROJ / "results" / "lstm_lobo.json").read_text())
ids12 = F.study_ids()
ids63 = J["design"]["rivers"]["sixty_three"]
ids51 = ids63[12:]
ids50 = J["design"]["rivers"]["other_50"]
NESTED, NESTED_IN = "03471500", "03473000"
A = attribute_tables()
res = {}

# ---- 1. nesting ------------------------------------------------------------------------------------------------------------------------------------
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

r = shapefile.Reader(str(CAM / "basin_set_full_res" / "HCDN_nhru_final_671.shp"))
fields = [f[0] for f in r.fields[1:]]
to_area = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True).transform
polys_geo, polys = {}, {}
for sr in r.iterShapeRecords():
    gid = str(int(sr.record[fields.index("hru_id")])).zfill(8)
    if gid in set(ids63):
        g = shape(sr.shape.__geo_interface__).buffer(0)
        polys_geo[gid], polys[gid] = g, transform(to_area, g)
assert set(polys) == set(ids63)
pairs = []
for i, a in enumerate(ids63):
    for b in ids63[i + 1:]:
        if polys[a].intersects(polys[b]):
            x = polys[a].intersection(polys[b]).area / 1e6
            if x > 1e-6:
                pairs.append(dict(a=a, b=b, shared_km2=float(x), share_of_a=float(x * 1e6 / polys[a].area), share_of_b=float(x * 1e6 / polys[b].area)))
touching = sum(1 for i, a in enumerate(ids63) for b in ids63[i + 1:] if polys[a].intersects(polys[b])) - len(pairs)
touching_pairs = [(a, b) for i, a in enumerate(ids63) for b in ids63[i + 1:] if polys[a].intersects(polys[b]) and polys[a].intersection(polys[b]).area / 1e6 <= 1e-6]
n_pairs = len(ids63) * (len(ids63) - 1) // 2
dist_ll = lambda a, b: float(haversine_km(A.loc[a, "gauge_lat"], A.loc[a, "gauge_lon"], A.loc[b, "gauge_lat"], A.loc[b, "gauge_lon"]))
fl = {g: pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")[g].q_mm for g in ids12 if g == NESTED_IN}
D63 = F.load_data(ids=ids63)
corr = float(np.corrcoef(D63[NESTED].q_mm.values, D63[NESTED_IN].q_mm.values)[0, 1])
res["nesting"] = dict(pairs_in_the_63_checked=n_pairs, pairs_sharing_area=pairs, pairs_touching_only=touching, touching_pairs=touching_pairs,
                      touching_pairs_between_a_study_basin_and_one_of_the_50=[(a, b) for a, b in touching_pairs if (a in ids12 and b in ids50) or (b in ids12 and a in ids50)],
                      touching_pairs_among_the_12=[(a, b) for a, b in touching_pairs if a in ids12 and b in ids12],
                      nested_pair=[p for p in pairs if {p["a"], p["b"]} == {NESTED, NESTED_IN}],
                      pairs_involving_a_study_basin_and_another_basin_of_the_63=[p for p in pairs if (p["a"] in ids12 or p["b"] in ids12)],
                      attribute_check=dict(names={g: str(A.loc[g, "gauge_name"]) for g in (NESTED, NESTED_IN)}, area_gages2_km2={g: float(A.loc[g, "area_gages2"]) for g in (NESTED, NESTED_IN)},
                                           gauge_distance_km=dist_ll(NESTED, NESTED_IN), upstream_gauge_point_inside_downstream_outline=bool(polys_geo[NESTED_IN].buffer(1e-4).contains(Point(A.loc[NESTED, "gauge_lon"], A.loc[NESTED, "gauge_lat"]))),
                                           daily_flow_correlation=corr, same_river_name=str(A.loc[NESTED, "gauge_name"]).split(" RIVER")[0] == str(A.loc[NESTED_IN, "gauge_name"]).split(" RIVER")[0]))
print("   outlines that only touch (shared border, no shared area): between a study river and one of the 50:", res["nesting"]["touching_pairs_between_a_study_basin_and_one_of_the_50"], "| among the 12:", res["nesting"]["touching_pairs_among_the_12"])
print("1. outlines of the 63: pairs", n_pairs, "| sharing area:", [(p["a"], p["b"], round(p["shared_km2"], 1), round(p["share_of_a"], 3), round(p["share_of_b"], 3)) for p in pairs], "| touching only:", touching)
print("   attribute check:", res["nesting"]["attribute_check"])

# ---- 2. the 12 are not read in training ---------------------------------------------------------------------------------------------------------------
tr_ids = {}
for n in ("unseen50_w_static", "unseen50_w_nostatic"):
    for s in (0, 1, 2):
        z = torch.load(PROJ / "results" / "models" / f"{n}_s{s}.pt", map_location="cpu", weights_only=True)
        tr_ids[f"{n}_s{s}"] = dict(n=len(z["ids"]), none_of_the_12=not (set(z["ids"]) & set(ids12)), equals_the_50=list(z["ids"]) == ids50, contains_nested_gauge=NESTED in z["ids"],
                                   static_values_train_has_a_study_basin=bool(set(z["static_values_train"]) & set(ids12)))
res["training_ids_of_the_saved_never_seen_networks"] = tr_ids
print("2. training ids of the 6 never-seen networks exclude the 12:", all(v["none_of_the_12"] and v["equals_the_50"] and not v["contains_nested_gauge"] for v in tr_ids.values()))

cfg04 = json.loads((PROJ / "results" / "lstm_all.json").read_text())["config"]
B63 = F.basins(ids63)


class Guard(dict):
    """A table dictionary that raises as soon as the table of one of the 12 is asked for (by key, get, iteration or values)."""
    def __getitem__(self, k):
        if k in ids12:
            raise AssertionError(f"the trainer asked for the table of {k}, one of the 12")
        return super().__getitem__(k)
    def get(self, k, default=None):
        if k in ids12:
            raise AssertionError(f"the trainer asked (get) for the table of {k}")
        return super().get(k, default)
    def __iter__(self):
        raise AssertionError("the trainer iterated over the table dictionary")
    def values(self):
        raise AssertionError("the trainer asked for all tables")
    def items(self):
        raise AssertionError("the trainer asked for all tables")


probe = {}
for label, static in (("static", tuple(F.STATIC)), ("nostatic", ())):
    cfg = F.Cfg(use_flow=False, static=static, seed=0, **dict(cfg04, updates=1, eval_every=1, patience=1))
    D50 = {g: D63[g] for g in ids50}                                          # the 12 are ABSENT from the dictionary
    B50 = B63.loc[ids50]
    run_a = F.fit(cfg, D50, ids50, static_df=B50 if static else None)
    Dg = Guard(D63)                                                            # the 12 are present but any access to them raises
    Bg = B63.drop(index=ids12)                                                 # and their properties are absent from the property table
    run_b = F.fit(cfg, Dg, ids50, static_df=Bg if static else None)
    run_c = F.fit(cfg, D63, ids50, static_df=B63 if static else None)        # all 63 tables present, as in the script
    wgap = lambda x, y: max(float((x.net.state_dict()[k] - y.net.state_dict()[k]).abs().max()) for k in x.net.state_dict())
    probe[label] = dict(fit_with_the_12_absent_vs_all_63_present_max_weight_gap=wgap(run_a, run_c), fit_with_a_raising_guard_completed=True, guard_vs_all_present_max_weight_gap=wgap(run_b, run_c),
                        scalers_equal_absent_vs_present=all(np.array_equal(np.asarray(run_a.sc[k]), np.asarray(run_c.sc[k])) for k in run_a.sc), ids_equal_the_50=run_a.ids == ids50 and run_c.ids == ids50)
    print(f"   probe ({label}): weights with the 12 absent vs all 63 present, max gap {probe[label]['fit_with_the_12_absent_vs_all_63_present_max_weight_gap']:.1e}; raising guard completed; scalers equal {probe[label]['scalers_equal_absent_vs_present']}")
res["the_12_are_not_read_probe"] = probe
res["builder_probe_in_json"] = {k: J["leak_probes"]["training_ignores_the_12"][k] for k in ("updates", "trained_on", "same_training_twice_curve_gap", "spoiled_12_curve_gap", "spoiled_12_weight_gap", "the_12_are_not_read")}
print("   builder's probe in the results file:", res["builder_probe_in_json"])

# ---- 3. proximity ------------------------------------------------------------------------------------------------------------------------------------
LAT, LON = A.gauge_lat.astype(float), A.gauge_lon.astype(float)
Sall = static_table(ids63, A)
S63_mu, S63_sd = Sall.mean(), Sall.std(ddof=0)


def nearest_km(g, pool):
    d = {h: float(haversine_km(LAT[g], LON[g], LAT[h], LON[h])) for h in pool if h != g}
    h = min(d, key=d.get)
    return d[h], h


def nearest_prop(g, pool, over63=False):
    o = Sall.loc[[h for h in pool if h != g]]
    mu, sd = (S63_mu, S63_sd) if over63 else (o.mean(), o.std(ddof=0))
    z = (Sall.loc[g] - mu) / sd
    d = np.sqrt((((o - mu) / sd - z) ** 2).sum(axis=1))
    return float(d.min()), str(d.idxmin())


DESIGN = {"11_leave_one_out": lambda g: [h for h in ids12 if h != g], "50": lambda g: ids50, "51": lambda g: ids51}
prox = {name: {g: dict(km=nearest_km(g, pool(g))[0], nearest=nearest_km(g, pool(g))[1], prop_pool=nearest_prop(g, pool(g))[0], prop_nearest=nearest_prop(g, pool(g))[1], prop_over63=nearest_prop(g, pool(g), True)[0])
               for g in ids12} for name, pool in DESIGN.items()}
summ = {name: dict(median_km=float(np.median([v["km"] for v in t.values()])), n_within_50_km=int(sum(v["km"] <= 50 for v in t.values())), n_within_100_km=int(sum(v["km"] <= 100 for v in t.values())),
                   min_km=float(min(v["km"] for v in t.values())), max_km=float(max(v["km"] for v in t.values())),
                   median_prop_distance_pool_standardised=float(np.median([v["prop_pool"] for v in t.values()])), median_prop_distance_standardised_over_the_63=float(np.median([v["prop_over63"] for v in t.values()])))
        for name, t in prox.items()}
res["proximity"] = dict(per_basin=prox, summary=summ, json_summary={k: {c: J["proximity"]["summary"][k][c] for c in ("median_km", "median_property_distance", "n_within_50_km", "n_within_100_km")} for k in J["proximity"]["summary"]})
print("\n3. proximity (own haversine on camels_topo gauge coordinates; ten properties standardised):")
for name, v in summ.items():
    j = J["proximity"]["summary"][name]
    print(f"   {name:18s} median km {v['median_km']:.2f} (results file {j['median_km']:.2f}) | within 50 km: {v['n_within_50_km']} | within 100 km: {v['n_within_100_km']} | median property distance, pool-standardised {v['median_prop_distance_pool_standardised']:.3f} "
          f"(file {j['median_property_distance']:.3f}), standardised over the 63 {v['median_prop_distance_standardised_over_the_63']:.3f}")
def outside_range(g, pool):
    o = Sall.loc[[h for h in pool if h != g]]
    return [c for c in STATIC if Sall.loc[g, c] > o[c].max() or Sall.loc[g, c] < o[c].min()]


for name, pool in DESIGN.items():
    n_out = sum(bool(outside_range(g, pool(g))) for g in ids12)
    summ[name]["n_with_a_property_outside_the_training_range"] = int(n_out)
    summ[name]["largest_abs_z_pool_standardised"] = float(max(((Sall.loc[g] - Sall.loc[[h for h in pool(g) if h != g]].mean()) / Sall.loc[[h for h in pool(g) if h != g]].std(ddof=0)).abs().max() for g in ids12))
    j = J["proximity"]["summary"][name]
    print(f"   {name:18s} rivers with a property outside the training range: {n_out} (results file {j['n_with_a_property_outside_the_training_range']}); largest |z| {summ[name]['largest_abs_z_pool_standardised']:.2f} (file {j['largest_abs_z']:.2f})")
zero = {g: float((D63[g].q_mm == 0).mean()) for g in ids63}
res["zero_flow"] = dict(twelve={g: zero[g] for g in ids12 if zero[g] > 0}, largest_in_the_50=max(zero[g] for g in ids50), where=max(ids50, key=zero.get), n_of_50_with_zero_days=int(sum(zero[g] > 0 for g in ids50)),
                        twelve_above_all_50=[g for g in ids12 if zero[g] > max(zero[h] for h in ids50)])
res["landscape_counts_of_the_50"] = pd.read_csv(PROJ / "results" / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id").loc[ids50, "landscape"].value_counts().to_dict()
print("   zero-flow days:", res["zero_flow"], "| landscapes of the 50:", res["landscape_counts_of_the_50"])

# ---- 4. the comparison with 05_lobo.py ----------------------------------------------------------------------------------------------------------------
cap = LOBO["settings"]["config"]
costs = {n: load_npz(n)["meta"]["cost"] for n in ("lobo_w_static", "lobo_w_nostatic")}
lobo_cost = {}
for n, c in costs.items():
    rows = list(c.values())
    lobo_cost[n] = dict(n_trainings=len(rows), updates_run=[r_["updates_run"] for r_ in rows], best_update=[r_["best_update"] for r_ in rows], n_at_update_limit=int(sum(r_["updates_run"] >= cap["updates"] for r_ in rows)),
                        n_best_update_within_one_patience_window_of_the_limit=int(sum(cap["updates"] - r_["best_update"] <= cap["patience"] * cap["eval_every"] for r_ in rows)),
                        mean_best_update=float(np.mean([r_["best_update"] for r_ in rows])), median_best_update=float(np.median([r_["best_update"] for r_ in rows])))


def replay(curve, config):
    best, best_u, bad = -np.inf, 0, 0
    for h in curve:
        if h["update"] > config["updates"]:
            break
        if h["val_nse"] > best:
            best, best_u, bad = h["val_nse"], h["update"], 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break
    return best_u


replays = {}
for n in ("unseen50_w_static", "unseen50_w_nostatic"):
    replays[n] = []
    for s in (0, 1, 2):
        z = torch.load(PROJ / "results" / "models" / f"{n}_s{s}.pt", map_location="cpu", weights_only=True)
        ku = replay(z["history"], cap)
        v = {h["update"]: h["val_nse"] for h in z["history"]}
        replays[n].append(dict(seed=s, actual_best_update=z["best_update"], updates_run=z["updates_run"], kept_under_the_05_cap_1000_patience_4=ku, validation_nse_at_the_actual_best=v[z["best_update"]],
                               validation_nse_at_the_update_the_05_rule_keeps=v[ku], validation_nse_given_up=v[z["best_update"]] - v[ku]))
lobo_cost["replay_of_the_05_stopping_rule_on_the_50_river_curves"] = replays
res["lobo_budget"] = dict(cap_in_05=cap, config_of_the_50_river_models=cfg04, per_variant=lobo_cost, in_training_reference_budget=LOBO["settings"]["config_of_04"])
print("\n4. 05_lobo trainings: cap", {k: cap[k] for k in ("updates", "patience")}, "| 50-river models:", {k: cfg04[k] for k in ("updates", "patience")})
for n in ("lobo_w_static", "lobo_w_nostatic"):
    c = lobo_cost[n]
    print(f"   {n}: {c['n_trainings']} trainings, best updates {sorted(c['best_update'])}, at the update limit: {c['n_at_update_limit']}, best within one patience window of the limit: {c['n_best_update_within_one_patience_window_of_the_limit']}")
print("   the 05 stopping rule replayed on the curves of the six 50-river trainings:", replays)

# per-seed and per-river comparison against the leave-one-out models
S = J["per_river"]["nse"]
NSEs = lambda n, k=1: np.array([S[n][f"l{k}"][g] for g in ids12])
BYSEED = J["per_river"]["nse_by_seed"]
cmp = {}
for T, C in (("unseen50_w_static", "lobo_w_static"), ("unseen50_w_nostatic", "lobo_w_nostatic")):
    cmp[T] = {}
    for k in LEADS:
        c = NSEs(C, k)
        row = dict(median_lobo=float(np.median(c)), median_seed_average=float(np.median(NSEs(T, k))), seeds=[])
        for s in (0, 1, 2):
            t = np.array([BYSEED[T][f"l{k}"][str(s)][g] for g in ids12])
            row["seeds"].append(dict(seed=s, median=float(np.median(t)), median_minus_lobo=float(np.median(t) - np.median(c)), n_rivers_higher=int((t > c).sum()), wilcoxon_p=float(wilcoxon(t, c).pvalue)))
        a = NSEs(T, k)
        row.update(seed_average_n_rivers_higher=int((a > c).sum()), seed_average_wilcoxon_p=float(wilcoxon(a, c).pvalue), median_of_river_differences=float(np.median(a - c)), mean_of_river_differences=float((a - c).mean()),
                   lowest_seed_minus_lobo=min(x["median_minus_lobo"] for x in row["seeds"]))
        cmp[T][f"l{k}"] = row
res["per_seed_against_lobo"] = cmp
print("\n   three seeds one by one against the 11-river models, lead 1:")
for T in cmp:
    r1 = cmp[T]["l1"]
    print(f"   {T:20s} lobo median {r1['median_lobo']:.4f} | seed medians {[round(x['median'], 4) for x in r1['seeds']]} | each minus lobo {[round(x['median_minus_lobo'], 4) for x in r1['seeds']]} | rivers higher {[x['n_rivers_higher'] for x in r1['seeds']]} | "
          f"p {[round(x['wilcoxon_p'], 4) for x in r1['seeds']]} | seed average: higher in {r1['seed_average_n_rivers_higher']}/12, p {r1['seed_average_wilcoxon_p']:.4f}")
# the effect of seed averaging, from the 12-river models (three seeds of the same model, so a pure seed effect): average of three vs the typical single seed
seed_eff = {}
for n, nm in (("lstm_w", "lstm_w"), ("lstm_wq", "lstm_wq")):
    singles = []
    for s in (0, 1, 2):
        z = load_npz(f"{nm}_s{s}")
        D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
        thr = own_thresholds(ids12, {g: D12[g].q_mm for g in ids12})
        singles.append(own_pooled(z, thr, 1)["nse_median"])
    seed_eff[n] = dict(single_seed_medians=singles, seed_average_median=float(np.median(NSEs(nm, 1))), mean_of_single_seed_medians=float(np.mean(singles)), average_minus_mean_single=float(np.median(NSEs(nm, 1)) - np.mean(singles)))
for n in ("lstm63_w", "lstm63_wq", "unseen50_w_static", "unseen50_w_nostatic"):
    singles = [float(np.median([BYSEED[n]["l1"][str(s)][g] for g in ids12])) for s in (0, 1, 2)]
    seed_eff[n] = dict(single_seed_medians=singles, seed_average_median=float(np.median(NSEs(n, 1))), mean_of_single_seed_medians=float(np.mean(singles)), average_minus_mean_single=float(np.median(NSEs(n, 1)) - np.mean(singles)))
res["seed_averaging_effect_lead1"] = seed_eff
print("\n   effect of averaging three seeds (median NSE, lead 1):")
for n, v in seed_eff.items():
    print(f"   {n:20s} single seeds {[round(x, 4) for x in v['single_seed_medians']]} mean {v['mean_of_single_seed_medians']:.4f} | seed average {v['seed_average_median']:.4f} | average minus mean single {v['average_minus_mean_single']:+.4f}")

# ---- 5. what the gain goes with -------------------------------------------------------------------------------------------------------------------------
tab = {}
for k in LEADS:
    for tag, T, C, Ref in (("with_properties", "unseen50_w_static", "lobo_w_static", "lstm_w"), ("without_properties", "unseen50_w_nostatic", "lobo_w_nostatic", "lobo_ref_w_nostatic")):
        gain = NSEs(T, k) - NSEs(C, k)
        km50 = np.array([prox["50"][g]["km"] for g in ids12])
        km11 = np.array([prox["11_leave_one_out"][g]["km"] for g in ids12])
        p50 = np.array([prox["50"][g]["prop_pool"] for g in ids12])
        p11 = np.array([prox["11_leave_one_out"][g]["prop_pool"] for g in ids12])
        sp = lambda x, y: dict(zip(("rho", "p"), map(float, spearmanr(x, y))))
        kd = lambda x, y: dict(zip(("tau", "p"), map(float, kendalltau(x, y))))
        tab[f"{tag}_l{k}"] = dict(
            gain_by_river={g: float(x) for g, x in zip(ids12, gain)}, median_gain=float(np.median(gain)), n_rivers_with_gain=int((gain > 0).sum()),
            gain_vs_km_to_nearest_training_gauge_50=sp(gain, km50), gain_vs_km_11=sp(gain, km11), gain_vs_reduction_in_km=sp(gain, km11 - km50), gain_vs_log_ratio_km=sp(gain, np.log(km11 / km50)),
            gain_vs_prop_distance_50=sp(gain, p50), gain_vs_prop_distance_11=sp(gain, p11), gain_vs_reduction_in_prop_distance=sp(gain, p11 - p50),
            kendall_gain_vs_km50=kd(gain, km50), kendall_gain_vs_reduction_in_km=kd(gain, km11 - km50),
            nse50_vs_km50=sp(NSEs(T, k), km50), nse11_vs_km11=sp(NSEs(C, k), km11),
            rivers_whose_nearest_training_gauge_is_no_closer=[g for g, a, b in zip(ids12, km50, km11) if a >= b - 1e-9],
            median_gain_where_the_nearest_gauge_is_no_closer=float(np.median([x for x, a, b in zip(gain, km50, km11) if a >= b - 1e-9])) if any(a >= b - 1e-9 for a, b in zip(km50, km11)) else None,
            rivers_whose_nearest_training_gauge_is_the_same_river=[g for g in ids12 if prox["50"][g]["nearest"] == prox["11_leave_one_out"][g]["nearest"]],
            median_gain_where_the_nearest_gauge_is_closer_by_over_20_km=float(np.median([x for x, a, b in zip(gain, km50, km11) if b - a > 20])) if any(b - a > 20 for a, b in zip(km50, km11)) else None,
            median_gain_where_closer_by_at_most_20_km=float(np.median([x for x, a, b in zip(gain, km50, km11) if b - a <= 20])) if any(b - a <= 20 for a, b in zip(km50, km11)) else None)
res["gain_against_proximity"] = tab
print("\n5. per-river gain of the 50-river never-seen model over the leave-one-out model, 1 day ahead (NSE):")
hdr = f"{'river':10s} {'km11':>7s} {'km50':>7s} {'nearest11':>10s} {'nearest50':>10s} {'prop11':>7s} {'prop50':>7s} {'NSE lobo':>9s} {'NSE 50':>8s} {'gain':>7s} | {'NSE lobo-np':>11s} {'NSE 50-np':>9s} {'gain-np':>8s}"
print(hdr)
for g in ids12:
    print(f"{g:10s} {prox['11_leave_one_out'][g]['km']:7.1f} {prox['50'][g]['km']:7.1f} {prox['11_leave_one_out'][g]['nearest']:>10s} {prox['50'][g]['nearest']:>10s} {prox['11_leave_one_out'][g]['prop_pool']:7.2f} {prox['50'][g]['prop_pool']:7.2f} "
          f"{S['lobo_w_static']['l1'][g]:9.3f} {S['unseen50_w_static']['l1'][g]:8.3f} {tab['with_properties_l1']['gain_by_river'][g]:+7.3f} | {S['lobo_w_nostatic']['l1'][g]:11.3f} {S['unseen50_w_nostatic']['l1'][g]:9.3f} {tab['without_properties_l1']['gain_by_river'][g]:+8.3f}")
for key in ("with_properties_l1", "without_properties_l1", "with_properties_l2", "with_properties_l3"):
    t = tab[key]
    print(f"  {key:22s} median gain {t['median_gain']:+.3f}, {t['n_rivers_with_gain']}/12 rivers gain | rho(gain, km to nearest gauge, 50) {t['gain_vs_km_to_nearest_training_gauge_50']['rho']:+.2f} (p {t['gain_vs_km_to_nearest_training_gauge_50']['p']:.2f}) | "
          f"rho(gain, km11) {t['gain_vs_km_11']['rho']:+.2f} (p {t['gain_vs_km_11']['p']:.2f}) | rho(gain, reduction in km) {t['gain_vs_reduction_in_km']['rho']:+.2f} (p {t['gain_vs_reduction_in_km']['p']:.2f}) | "
          f"rho(gain, property distance 50) {t['gain_vs_prop_distance_50']['rho']:+.2f} (p {t['gain_vs_prop_distance_50']['p']:.2f}) | rho(gain, reduction in property distance) {t['gain_vs_reduction_in_prop_distance']['rho']:+.2f} (p {t['gain_vs_reduction_in_prop_distance']['p']:.2f})")
    print(f"      rivers whose nearest training gauge is no closer in the 50-river design: {t['rivers_whose_nearest_training_gauge_is_no_closer']}, their median gain {t['median_gain_where_the_nearest_gauge_is_no_closer']}; "
          f"closer by over 20 km: median gain {t['median_gain_where_the_nearest_gauge_is_closer_by_over_20_km']}; closer by at most 20 km: {t['median_gain_where_closer_by_at_most_20_km']}")
res["seconds"] = round(time.time() - t0, 1)
save("m8_unseen", res)
print(f"\n({res['seconds']} s)")
