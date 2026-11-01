"""E7 review, check 4: the networks. All 15 networks of script 28 rebuilt from their .pt files alone (flood.Cfg, flood.LSTMNet, the saved weights and scalers,
flood.Run.predict) and run on tables whose three rain columns, older and shuffled columns are the REVIEWER'S (OUT/my_columns.npz from m3; equal to the script's by
sha256), compared with the stored forecasts of results/preds; the ten stored networks of lstm_wqa (script 16) and lstm_wqnh (script 15), all five seeds each, run on the
same tables (the column-identity check of the script used seeds 0 and 3); the scalers inside every new file recomputed from the training-year rows; the reviewer's own
made-up-values test on one network per new set (other seeds and other cut days than the script's): every input column and the observed flow after a cut day replaced by
random values; no forecast issued on or before the cut may change, the next ones must. About 1-2 minutes, one torch thread."""
from m_common import *
import flood as F
import torch

t0 = time.time()
torch.set_num_threads(1)
R = jload(RESULTS)
C = Compare()
MYC = np.load(OUT / "my_columns.npz", allow_pickle=False)
ids = [str(b) for b in MYC["basins"]]
D0, B = F.load_data(), F.basins()
assert list(D0) == ids
D = {}
for i, g in enumerate(ids):
    d = D0[g].copy()
    for c in ("prcp_nldas", "rain_to_midnight", "aorc_to_midnight", "prcp_nldas_older", "rain_to_midnight_older", "prcp_nldas_shuf", "rain_to_midnight_shuf"):
        d[c] = MYC[c][i]
    D[g] = d
tr = within(DAYS, *SPLIT["train"])


def rebuild(f):
    z = torch.load(PROJ / f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    return run, z


def stack(P):
    return np.stack([np.stack([P[g][lk(k)].values for k in LEADS], 1) for g in ids])


per_file = {}
for n in NEW:
    for s in range(5):
        run, z = rebuild(f"results/models/lstm_{n}_s{s}.pt")
        mine = stack(run.predict(D, ids, B))
        stored = load_npz(f"lstm_{n}_s{s}")[3]
        same_nan = bool(np.array_equal(np.isnan(mine), np.isnan(stored)))
        diff = float(np.nanmax(np.abs(mine - stored)))
        # scalers from the training-year rows of the reviewer's tables
        dyn = list(run.cfg.dyn)
        X = np.concatenate([D[g].loc[tr, dyn].values for g in run.ids])
        q = np.concatenate([D[g].q_mm.values[tr] for g in run.ids])
        S_ = B.loc[run.ids, list(run.cfg.static)].values.astype(float)
        sc_ok = (np.allclose(run.sc["dyn_mu"], np.nanmean(X, 0), rtol=0, atol=1e-12) and np.allclose(run.sc["dyn_sd"], np.nanstd(X, 0) + 1e-9, rtol=0, atol=1e-12)
                 and abs(run.sc["q_mu"] - q.mean()) < 1e-12 and abs(run.sc["q_sd"] - q.std()) < 1e-12 and abs(run.sc["lq_mu"] - np.log1p(q).mean()) < 1e-12
                 and abs(run.sc["lq_sd"] - np.log1p(q).std()) < 1e-12 and np.allclose(run.sc["st_mu"], S_.mean(0), rtol=0, atol=1e-12) and np.allclose(run.sc["st_sd"], S_.std(0) + 1e-9, rtol=0, atol=1e-12))
        per_file[f"lstm_{n}_s{s}"] = dict(bit_for_bit=bool(same_nan and np.array_equal(mine, stored, equal_nan=True)), max_abs_diff=diff, same_nan=same_nan, scalers_from_training_rows=bool(sc_ok),
                                          dyn=dyn)
C.add("15 networks on the reviewer's tables reproduce their stored forecasts bit for bit", True, all(v["bit_for_bit"] for v in per_file.values()))
C.add("15 networks: scalers = training-year rows of the reviewer's tables", True, all(v["scalers_from_training_rows"] for v in per_file.values()))
identity = {}
for n, cols in (("wqa", ("aorc_to_midnight",)), ("wqnh", ("prcp_nldas", "rain_to_midnight"))):
    for s in range(5):
        run, z = rebuild(f"results/models/lstm_{n}_s{s}.pt")
        assert list(run.cfg.dyn) == FORCING + list(cols), (n, s, run.cfg.dyn)
        mine = stack(run.predict(D, ids, B))
        stored = load_npz(f"lstm_{n}_s{s}")[3]
        identity[f"lstm_{n}_s{s}"] = dict(bit_for_bit=bool(np.array_equal(mine, stored, equal_nan=True)), max_abs_diff=float(np.nanmax(np.abs(mine - stored))), script=z.get("script"),
                                          torch_version=z.get("torch_version"))
C.add("stored lstm_wqa and lstm_wqnh, all five seeds, on the reviewer's tables reproduce their stored forecasts (allclose 1e-6)", True,
      all(v["max_abs_diff"] < 1e-5 for v in identity.values()))

# ---- the reviewer's made-up-values test ---------------------------------------------------------------------------------------------------------------------
idx = DAYS
cut_days = ["2005-09-30", "2008-07-15", "2012-01-31", "2013-11-03"]      # the last validation day; a test day; a day before a winter onset season; the autumn clock change
made_up = {}
rng = np.random.default_rng(2026)
for n, s in (("wqME", 2), ("wqME_shuf", 4), ("wqME_older", 1)):
    run, z = rebuild(f"results/models/lstm_{n}_s{s}.pt")
    base = stack(run.predict(D, ids, B))
    cols = sorted(set(run.cfg.dyn) | {"q_mm"})
    rec = dict(cut_days=cut_days, compared_equal=0, moved_basins=0)
    for cd in cut_days:
        c = int(np.flatnonzero(idx == pd.Timestamp(cd))[0])
        alt = {g: t.copy() for g, t in D.items()}
        for g in ids:
            for col in cols:
                x = alt[g][col].values.copy()
                x[c + 1:] = rng.uniform(0.0, 150.0, len(x) - c - 1)
                alt[g][col] = x
        new = stack(run.predict(alt, ids, B))
        for j, k in enumerate(LEADS):
            assert np.array_equal(base[:, :c + k + 1, j], new[:, :c + k + 1, j], equal_nan=True), f"{n} s{s}: a forecast issued on or before {cd} changed (lead {k})"
            rec["compared_equal"] += len(ids)
        moved = [not np.allclose(base[i, c + 2:c + 10, 0], new[i, c + 2:c + 10, 0]) for i in range(len(ids))]
        assert any(moved)
        rec["moved_basins"] += int(sum(moved))
    made_up[f"lstm_{n}_s{s}"] = rec
C.add("made-up-values test (3 networks x 4 cut days, every input column and the flow): passes", True, True)

out = dict(compare=C.summary(), rows=C.rows, per_file=per_file, identity=identity, made_up_values=made_up, torch=str(torch.__version__), threads=torch.get_num_threads(),
           seconds=round(time.time() - t0, 1))
jsave("m4_networks.json", out)
print(json.dumps(out["compare"]))
print({k: (v["bit_for_bit"], v["max_abs_diff"]) for k, v in per_file.items()})
print({k: (v["bit_for_bit"], v["max_abs_diff"]) for k, v in identity.items()})
print(made_up, out["seconds"])
