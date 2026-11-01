"""Round 2, check 3: spot-check of the new (seed 3-4, Windows) networks and of the derived sets. One thread, nothing trained, one network at a time.
The input tables are built by the FIRST review's own table builder (code/review/forecast_input/f_tables.py, variant 'csv': the reviewer's own AORC local days, the forecast
columns read from data/gefs/basins, the reviewer's copy of the shuffle), not by the script. Each network is rebuilt from its .pt file alone (flood.Cfg / LSTMNet / Run.predict,
as the brief allows) and its forecasts are compared with the stored file:
  lstm_wqaP_s3 (table sanity: the network's own set), lstm_wqaF_s_s4 (satellite-era arm),
  lstm_wqaP_forecastpred_s3 (wqaP seed 3 network given the forecast in the p slots), lstm_wqaF_sub_rawpred_s4 (wqaF_sub seed 4 given the RAW forecast; written by the
  first, stopped attempt), lstm_wqaF_sub_perfectpred_s4 (wqaF_sub seed 4 given perfect rain; written by the second attempt).
Also: Q95 of both splits from flood.thresholds(flood.load_data()) against the reviewer's Q95 from the stored observed flow (v1). Output OUT/v3_networks.json."""
from v_common import *
import gc
sys.path.insert(0, str(PROJ / "code" / "review" / "forecast_input"))
import torch
torch.set_num_threads(1)
import flood as F
from f_tables import build_tables, KS

t0 = time.time()
TAB, ids = build_tables("csv")
print(f"tables built in {time.time() - t0:.0f} s", flush=True)
D = F.load_data(); B = F.basins()
assert list(D) == ids
v1 = jread("v1_avg_check.json")["THR"]
thr_check = {}
for arm in SPLITS:
    th = F.thresholds(D, split=SPLITS[arm])
    thr_check[arm] = max(abs(th[g] - v1[arm][g]) for g in ids)
print("Q95 engine vs mine, max abs diff:", thr_check, flush=True)


def run_from_file(f):
    z = torch.load(f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z


def frames(run, tab, arm):
    p = run.predict(tab, ids, B)
    if arm == "sat":
        p = {g: p[g].reindex(D[g].index).assign(obs=D[g].q_mm) for g in ids}
    return p


def variant(kind):
    t = {}
    for g in ids:
        m = TAB["main"][g].copy()
        for k in KS:
            if kind == "rawpred":
                m[f"fs{k}"] = m[f"fr{k}"].values
            elif kind == "perfectpred":
                m[f"fs{k}"] = m[f"p{k}"].values
            elif kind == "forecastpred":
                m[f"p{k}"] = m[f"fs{k}"].values
        t[g] = m
    return t


CASES = [("lstm_wqaP_s3", "wqaP", 3, "main", None), ("lstm_wqaF_s_s4", "wqaF_s", 4, "sat", None),
         ("lstm_wqaP_forecastpred_s3", "wqaP", 3, "main", "forecastpred"), ("lstm_wqaF_sub_rawpred_s4", "wqaF_sub", 4, "main", "rawpred"),
         ("lstm_wqaF_sub_perfectpred_s4", "wqaF_sub", 4, "main", "perfectpred")]
out = {}
for stored, src, seed, arm, kind in CASES:
    run, z = run_from_file(PROJ / "results" / "models" / f"lstm_{src}_s{seed}.pt")
    tab = TAB[arm] if kind is None else variant(kind)
    new = frames(run, tab, arm)
    d, b, o, p, meta = load_npz(stored)
    assert b == ids
    q = np.stack([new[g][[f"l{k}" for k in LEADS]].reindex(d).values for g in ids])
    fin = np.isfinite(p)
    out[stored] = dict(network=f"results/models/lstm_{src}_s{seed}.pt", network_sha256=sha256(f"results/models/lstm_{src}_s{seed}.pt"), stored_sha256=sha256(f"results/preds/{stored}.npz"),
                       network_script=z.get("script"), network_seed=z.get("seed"), network_input_set=z.get("input_set"), torch_version_of_training=z.get("torch_version"),
                       bit_for_bit=bool(np.array_equal(p, q, equal_nan=True)), same_nan=bool(np.array_equal(np.isnan(p), np.isnan(q))),
                       max_abs_diff=float(np.abs(p[fin] - q[fin]).max()), n_values=int(fin.sum()))
    print(stored, {k: out[stored][k] for k in ("bit_for_bit", "same_nan", "max_abs_diff", "torch_version_of_training")}, flush=True)
    del run, z, new, q, p, tab
    gc.collect()
jsave("v3_networks.json", dict(cases=out, thresholds_engine_vs_mine_max_abs_diff=thr_check, torch=torch.__version__, threads=torch.get_num_threads(), seconds=round(time.time() - t0, 1)))
print(f"done in {time.time() - t0:.0f} s")
