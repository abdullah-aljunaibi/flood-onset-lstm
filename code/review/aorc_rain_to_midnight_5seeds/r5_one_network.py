"""Round 2, check n1: ONE saved network, lstm_wqa seed 3 (trained on 30 October 2026 on the Windows PC; script 16's own reload check covers only seed 0), rebuilt from its .pt file
alone and run with one torch thread, nothing trained:
 (a) on a daily table whose aorc_to_midnight column I build myself from data/aorc/aorc_<gauge>.npz (every hour to the wall-clock day in which it ends, zoneinfo): its forecasts against the
     stored results/preds/lstm_wqa_s3.npz (bit for bit expected), and its onsets warned 1 day ahead (321 by my scoring of the stored file);
 (b) with my own Stage IV sums of the issue day of each of the 611 onsets (r4_stage4.py, from data/aorc/stage4_issue_days.csv) in the place of AORC: the onsets warned, against
     single_seed_onsets_warned[3] of results/stage4_realtime.json (stage4: 296), and the same with the last hour and the last two hours taken out (274 and the stored value).
Needs r4_stage4.json in OUT."""
from r_common import *
import torch

torch.set_num_threads(1)
import flood as F

SEED = int(os.environ.get("SPOT_SEED", 3))
C = Compare()
R4 = jread("r4_stage4.json")["mine"]
S4 = jload(STAGE4)
D = F.load_data()
B = F.basins()
ids = list(D)
DAYS = D[ids[0]].index
DW = jload("results/day_windows.json")
ZONE = {g: ("America/Chicago" if int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) == -6 else "America/New_York") for g in ids}


def my_rain(g):
    rel = f"data/aorc/aorc_{g}.npz"
    sha256(rel)
    z = np.load(PROJ / rel, allow_pickle=False)
    t = pd.to_datetime(z["stamp_utc_s"], unit="s")
    x = z["rain_mm"].astype(float)
    z.close()
    local = t.tz_localize("UTC").tz_convert(ZONE[g]).tz_localize(None)
    day = (local - pd.Timedelta(seconds=1)).normalize()                   # an hour ending at 24:00 belongs to the day that ends then
    s = pd.Series(x, index=day).groupby(level=0).sum().reindex(DAYS)
    assert s.notna().all()
    return s.values


TAB = {g: D[g].assign(aorc_to_midnight=my_rain(g)) for g in ids}
rel = f"results/models/lstm_wqa_s{SEED}.pt"
sha256(rel)
z = torch.load(PROJ / rel, map_location="cpu", weights_only=True)
cfg = F.Cfg(**z["cfg"])
n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
net = F.LSTMNet(n_in, cfg)
net.load_state_dict(z["state_dict"])
net.eval()
sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
C.add("network inputs", ["prcp", "srad", "tmax", "tmin", "vp", "dayl", "aorc_to_midnight"], list(cfg.dyn))

d, b, obs, stored, meta = load_npz(f"lstm_wqa_s{SEED}")
assert b == ids and DAYS.equals(d)


def as_array(P):
    return np.stack([P[g][["l1", "l2", "l3"]].values for g in ids])


p0 = as_array(run.predict(TAB, ids, B))
same_nan = bool(np.array_equal(np.isnan(p0), np.isnan(stored)))
mx = float(np.nanmax(np.abs(p0 - stored)))
C.add("(a) same missing days as the stored forecasts", True, same_nan)
C.add("(a) max abs difference to the stored forecasts", 0.0, mx, tol=1e-6)
r0 = score_arrays(d, b, obs, p0)
C.add("(a) onsets warned with my AORC table = stored single seed", score(f"lstm_wqa_s{SEED}")[1]["onset"]["hits"], r0[1]["onset"]["hits"])
del stored

rows = [tuple(x) for x in R4["issue_day_values"]["rows"]]
thr = thresholds(d, b, obs)
targets = [(g, pd.Timestamp(day) + pd.Timedelta(days=1)) for g, day in rows]
res = {}
for kind in ("s4", "s4_no_last_hour", "s4_no_last_two_hours"):
    tabs = {g: TAB[g].copy() for g in ids}
    col = tabs[ids[0]].columns.get_loc("aorc_to_midnight")
    for (g, day), v in zip(rows, R4["issue_day_values"][kind]):
        tabs[g].iloc[DAYS.get_loc(pd.Timestamp(day)), col] = v
    p = as_array(run.predict(tabs, ids, B))
    warned = [bool(p[ids.index(g), DAYS.get_loc(t), 0] > thr[ids.index(g)]) for g, t in targets]
    res[kind] = int(sum(warned))
    del p, tabs
name = {"s4": "stage4", "s4_no_last_hour": "stage4_no_last_hour", "s4_no_last_two_hours": "stage4_no_last_two_hours"}
for kind, n in res.items():
    C.add(f"(b) seed {SEED} {name[kind]} onsets warned", S4["models"]["wqa"]["variants"][name[kind]]["single_seed_onsets_warned"][SEED], n)
jsave(f"r5_one_network_s{SEED}.json", dict(compare=C.rows, summary=C.summary(), seed=SEED, mine=dict(max_abs_diff=mx, onsets_aorc=r0[1]["onset"]["hits"], **res), files_read=READ))
print(json.dumps(C.summary(), indent=1)); print(res, "aorc", r0[1]["onset"]["hits"], "max abs diff", mx)
