"""Reload check (added 2026-10-28 for the move to another PC): rebuild lstm_wqa seed 0 from its .pt file alone and compare with results/preds/lstm_wqa_s0.npz.

No training, no network. The AORC column is rebuilt from the cached data/aorc/*.npz with the same function as
code/16_aorc_rain_to_midnight.py (rain_by_wall_clock_day), copied here so the 2,000-line script need not run.
Run from the project root: python code/check_reload.py   (reads only; writes nothing; exit status 1 on FAIL). On the Mac where the networks were trained the
result is bit for bit (max abs difference 0.0); on another CPU or torch build expect tiny differences, not identity.
"""
import json
import sys
sys.dont_write_bytecode = True                         # read-only check: leave no __pycache__ behind
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import pandas as pd
import torch

import fetch_aorc as AO
import flood as F

D, B = F.load_data(), F.basins()
ids = list(D)
DW = json.loads((F.RES / "day_windows.json").read_text())
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
ZONE = {-5: "America/New_York", -6: "America/Chicago"}
DAYS = pd.date_range(F.PERIOD[0], F.PERIOD[1])


def wall_clock(t_utc, zone):
    return t_utc.tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)


def rain_by_wall_clock_day(stamp, x, zone):          # copied from code/16_aorc_rain_to_midnight.py
    by_day = pd.Series(x, index=(wall_clock(stamp, zone) - pd.Timedelta(minutes=1)).normalize()).groupby(level=0)
    return by_day.sum(), by_day.size()


RAIN = {}
for g in ids:
    stamp, x = AO.load_series(g)
    RAIN[g] = rain_by_wall_clock_day(stamp, x, ZONE[UTC_OFF[g]])[0].reindex(DAYS)
    assert RAIN[g].notna().all()
D2 = {g: D[g].assign(aorc_to_midnight=RAIN[g].values) for g in ids}

f = ROOT / "results" / "models" / "lstm_wqa_s0.pt"
z = torch.load(f, map_location="cpu", weights_only=True)
print("how_to_reload:", z["how_to_reload"][:120], "...")
cfg = F.Cfg(**z["cfg"])
n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
print("n_in", n_in, "weights expect", z["state_dict"]["lstm.weight_ih_l0"].shape[1])
net = F.LSTMNet(n_in, cfg)
net.load_state_dict(z["state_dict"])
net.eval()
sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"],
            z["best_val_nse"], z["n_params"], z["split"])
new = run.predict(D2, z["ids"], B)
old = F.load_preds("lstm_wqa_s0")
cols = ["obs", "l1", "l2", "l3"]
assert list(old) == sorted(z["ids"]) or set(old) == set(z["ids"])
a = np.stack([old[g][cols].values for g in z["ids"]])
b = np.stack([new[g][cols].values for g in z["ids"]])
same_index = all(old[g].index.equals(new[g].index) for g in z["ids"])
print("torch", torch.__version__, "numpy", np.__version__, "threads", torch.get_num_threads())
print("same date index:", same_index)
print("same missing days:", bool(np.array_equal(np.isnan(a), np.isnan(b))))
print("values compared:", a.size, "missing in both:", int(np.isnan(a).sum()))
print("max abs difference (all cols):", float(np.nanmax(np.abs(a - b))))
print("max abs difference (forecasts l1-l3 only):", float(np.nanmax(np.abs(a[..., 1:] - b[..., 1:]))))
print("bit for bit:", bool(np.array_equal(a, b, equal_nan=True)))
print("allclose rtol 1e-6 atol 1e-8:", bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)))
# the headline of lstm_wqa from the stored seed-average set, for context
thr = F.thresholds(D)
s_old = F.evaluate(old, thr)["summary"]["l1"]
s_new = F.evaluate(new, thr)["summary"]["l1"]
print("seed 0, 1 day ahead, test: stored NSE median %.6f POD onset %.6f | reloaded NSE median %.6f POD onset %.6f"
      % (s_old["nse_median"], s_old["onset"]["pod"], s_new["nse_median"], s_new["onset"]["pod"]))
ok = same_index and bool(np.array_equal(np.isnan(a), np.isnan(b))) and bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
print("RELOAD CHECK:", "PASS" if ok else "FAIL", "(the tolerance of the scripts' own reload asserts: rtol 1e-6, atol 1e-8)")
sys.exit(0 if ok else 1)
