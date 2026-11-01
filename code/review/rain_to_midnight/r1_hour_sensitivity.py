"""ROUND 1, check 3g (re-written in round 2): how much would a wrong hour convention matter? If the Open-Meteo stamps were the START of the hour, the daily value of the script would hold the hour
after local midnight (one hour after the issue time). The saved wqh and wh networks (three seeds averaged, flood.Run.predict, inference only) are fed the leak-free daily series of that case and scored on the
test years. Compared with the numbers recorded in results/review_rain_to_midnight.json (round 1, check 3g). About ten seconds."""
import sys; sys.dont_write_bytecode = True
import time
from r2_common import *
import flood as F
import r2_clock as C

t0 = time.time()
D, B = F.load_data(), F.basins(); ids = list(D); thr = F.thresholds(D)
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}
S, S_alt = {}, {}
for g in ids:
    t, x, _ = C.read_era5(g)
    S[g], _ = C.window_sum(t, x, C.offset_rules(t, ZONE[g]), "wall", 24)                          # the script's convention: stamp = end of the hour
    t2 = t + np.timedelta64(60, "m")                                                              # the opposite convention: the stamp is the START of the hour, so the hour ends one hour later
    S_alt[g], _ = C.window_sum(t2, x, C.offset_rules(t2, ZONE[g]), "wall", 24)
diff = np.stack([S_alt[g] - S[g] for g in ids])
print("mean |S_true - S| per day over the 12 basins: %.4f mm (mean daily rain %.3f mm); share of days that differ: %.3f" % (np.abs(diff).mean(), np.mean([S[g].mean() for g in ids]), (np.abs(diff) > 1e-9).mean()))

def predict_all(name, series):
    P = []
    for seed in (0, 1, 2):
        z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
        cfg = F.Cfg(**z["cfg"]); n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
        net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
        sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
        run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
        P.append(run.predict({g: D[g].assign(rain_to_midnight=series[g]) for g in ids}, z["ids"], B))
    return F.ensemble(P)

out = {}
for fam, name in (("with_river_gauge", "wqh"), ("without_river_gauge", "wh")):
    for lab, series in (("script", S), ("flipped", S_alt)):
        E = F.evaluate(predict_all(name, series), thr)["summary"]["l1"]
        out[f"{name}|{lab}"] = dict(onset_pod=E["onset"]["pod"], nse_median=E["nse_median"], hits=E["onset"]["hits"])
        print(f"{fam:20s} {lab:8s} lead 1: onsets warned {100*E['onset']['pod']:.2f}% (hits {E['onset']['hits']}/611), false {100*E['onset']['far']:.1f}%, median NSE {E['nse_median']:.4f}")
rec = next(c for c in jload("results/review_rain_to_midnight.json")["checks"] if c["id"] == "3g")["numbers"]
dev = 0.0
for fam, name in (("with_river_gauge", "wqh"), ("without_river_gauge", "wh")):
    a, b = out[f"{name}|script"], out[f"{name}|flipped"]
    out[f"{name}|change"] = dict(onset_points=100 * (b["onset_pod"] - a["onset_pod"]), nse=b["nse_median"] - a["nse_median"])
    dev = max(dev, abs(a["onset_pod"] - rec[fam]["onsets_script_series"]), abs(b["onset_pod"] - rec[fam]["onsets_flipped"]), abs(out[f"{name}|change"]["onset_points"] - rec[fam]["change_points"]), abs(out[f"{name}|change"]["nse"] - rec[fam]["nse_change"]))
    print(f"  {name}: flipping the convention changes the onsets by {out[f'{name}|change']['onset_points']:+.2f} points and the median NSE by {out[f'{name}|change']['nse']:+.4f}")
print(f"against the round-1 record (check 3g): max deviation {dev:.1e}  ({time.time()-t0:.0f} s)")
out["max_dev_vs_round1_record"] = dev
jsave("r1_hour_sensitivity.json", out)
