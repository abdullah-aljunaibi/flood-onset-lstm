"""Check n1: three networks of seeds 3-4 reloaded from their .pt files alone, with ONE torch thread, forecasting the 12 again; compared with the stored seed files.

lstm63_wq_s3 (made by the run the memory guard stopped at 12:47), lstm63eq_wq_s4 and unseen50_w_static_s4 (the rerun). The engine (flood.LSTMNet, flood.Run.predict,
flood.load_data on the 63-basin cache) is used for the forward pass; the network is rebuilt by the reviewer's code from the dict in the file (not by the script's load_run);
the comparison is the reviewer's. One network at a time; nothing is trained.
Run: python code/review/more_basins_5seeds/r5_networks.py
"""
import os

os.environ["FLOOD_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
from r_common import *

sys.path.insert(0, str(PROJ / "code"))
import pandas as pd
import torch

torch.set_num_threads(1)
import flood as F

torch.set_num_threads(1)
ids12 = F.study_ids()
ids63 = ids12 + [g for g in F.candidate_ids() if g not in set(ids12)]
D = F.load_data(ids=ids63)
out = {}
for name, seed in (("lstm63_wq", 3), ("lstm63eq_wq", 4), ("unseen50_w_static", 4)):
    f = MODELS / f"{name}_s{seed}.pt"
    z = torch.load(f, map_location="cpu", weights_only=True)
    assert z["set"] == name and z["seed"] == seed
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    sdf = pd.DataFrame(z["static_values_scored"], index=list(cfg.static)).T if cfg.static else None
    with torch.no_grad():
        new = run.predict(D, z["scored_ids"], sdf)
    st = load_npz(f"{name}_s{seed}")
    a = st["pred"]
    b = np.stack([new[g][[f"l{k}" for k in LEADS]].values for g in st["basins"]])
    obs_eq = bool(np.array_equal(st["obs"], np.stack([new[g]["obs"].values for g in st["basins"]]), equal_nan=True))
    out[f"{name}_s{seed}"] = dict(model_sha256=sha256(f), n_values=int(a.size), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), max_abs_diff=float(np.nanmax(np.abs(a - b))),
                                  same_missing=bool(np.array_equal(np.isnan(a), np.isnan(b))), obs_equal=obs_eq, trained_with_threads=z["torch_threads"], reloaded_with_threads=torch.get_num_threads(),
                                  training_ids=len(z["ids"]), the_12_not_in_training=(not set(ids12) & set(z["ids"])) if name.startswith("unseen") else None)
    print(f"{name}_s{seed}: {a.size} values, bit for bit {out[f'{name}_s{seed}']['bit_for_bit']}, max abs diff {out[f'{name}_s{seed}']['max_abs_diff']:.2e}, "
          f"obs equal {obs_eq}, threads {z['torch_threads']} -> {torch.get_num_threads()}", flush=True)
    del z, net, run, new, st, a, b
save("r5_networks", out)
