"""ROUND 2, checks (c)/(h) extras: training records and per-seed scores of the 9 new networks against the .pt files, the log and my own numbers; reload_check; passes arithmetic;
paired onset counts of the LSTM against the two simple models; the Daymet window per basin; the effect of leaving Myakka out of value_of_hours_09_to_24."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
import re
from scipy.stats import binomtest
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json")
ids = list(data()); out = {}
# training records
bad = []; n = 0
for rec in R["training"]:
    if rec["set"] not in ("wqh_to09", "wqh_utc", "wqnh"): continue
    z = torch.load(f"{PROJ}/results/models/lstm_{rec['set']}_s{rec['seed']}.pt", map_location="cpu", weights_only=True)
    for k in ("best_update", "updates_run", "n_params"):
        n += 1; bad += [(rec["set"], rec["seed"], k)] if int(z[k]) != int(rec[k]) else []
    for k, tol in (("best_val_nse", 5.1e-5), ("seconds", 0.051), ("cpu_seconds", 0.051)):
        n += 1; bad += [(rec["set"], rec["seed"], k)] if abs(float(z[k]) - float(rec[k])) > tol else []
out["training_vs_pt"] = dict(n=n, differ=len(bad))
print(f"training records of the 9 new networks (JSON against the .pt files): {n} numbers, {len(bad)} differ")
# per-seed scores
n2, bad2 = 0, []
for name in ("wqh_to09", "wqh_utc", "wqnh"):
    for i, rec in enumerate(R["models"][name]["per_seed"]):
        s = rec["seed"]
        for k in (1, 2, 3):
            c = counts(f"lstm_{name}_s{s}", k); med = float(np.median(list(nse_per_basin(f"lstm_{name}_s{s}", k).values())))
            for a, b in ((rec["test_onset_warned"][k - 1], c["pod"]), (rec["test_onset_false"][k - 1], c["far"]), (rec["test_nse_median"][k - 1], med)):
                n2 += 1; bad2 += [(name, s, k)] if abs(a - b) > 1e-12 else []
out["per_seed_vs_own"] = dict(n=n2, differ=len(bad2))
print(f"per-seed test scores of the 3 new sets (3 seeds x 3 leads x [pod, far, median NSE]): {n2} compared with my own numbers, {len(bad2)} differ")
# extend log lines against JSON
log = open(f"{PROJ}/results/logs/15_rain_to_midnight_extend.log").read().splitlines()
n3, bad3 = 0, []
for line in log:
    m = re.match(r"lstm_(\w+) seed (\d) \| best at update (\d+) of (\d+), (\d+) s \| validation NSE ([0-9.]+) \| test NSE median ([0-9. ]+)", line)
    if not m: continue
    name, s, bu, ur, sec, vn, tn = m.groups(); rec = R["models"][name]["per_seed"][int(s)]
    n3 += 1; bad3 += [line] if not (rec["best_update"] == int(bu) and rec["updates_run"] == int(ur) and round(rec["best_val_nse"], 3) == float(vn) and [f"{x:.3f}" for x in rec["test_nse_median"]] == tn.split()) else []
rl = open(f"{PROJ}/results/logs/15_rain_to_midnight_rescore.log").read().splitlines()
n4, bad4 = 0, []
for line in rl:
    m = re.match(r"lstm_(\w+) seed (\d) \| best at update (\d+) of (\d+), (\d+) s \| validation NSE ([0-9.]+) \| test NSE median ([0-9. ]+)", line)
    if not m: continue
    name, s, bu, ur, sec, vn, tn = m.groups(); rec = R["models"][name]["per_seed"][int(s)]
    n4 += 1; bad4 += [line] if not (rec["best_update"] == int(bu) and rec["updates_run"] == int(ur) and round(rec["best_val_nse"], 3) == float(vn) and [f"{x:.3f}" for x in rec["test_nse_median"]] == tn.split()) else []
out["log_vs_json"] = dict(extend_lines=n3, extend_differ=len(bad3), rescore_lines=n4, rescore_differ=len(bad4))
print(f"log lines against the JSON per-seed records: extend log {n3} lines ({len(bad3)} differ), rescore log {n4} lines ({len(bad4)} differ)")
# reload check
rc = R["reload_check"]["per_set"]
out["reload_check"] = {k: dict(bit_for_bit=v["bit_for_bit"], max_abs_difference=v["max_abs_difference"], carried_over=R["reload_check"]["carried_over_from_training_run"]) for k, v in rc.items()}
print("reload_check in the JSON:", {k: (v["bit_for_bit"], v["max_abs_difference"]) for k, v in rc.items()}, "carried over:", R["reload_check"]["carried_over_from_training_run"])
# passes arithmetic
P = R["passes"]
out["passes"] = dict(n_trainings=(R["n_trainings"], sum(p["n_trainings"] for p in P)), training_seconds=(R["training_seconds"], round(sum(p["training_seconds"] for p in P), 1)), runtime=(R["runtime_s"], round(sum(p["runtime_s"] for p in P if p.get("kind") != "rescore"), 1)),   # round 3: the rescore pass is listed in passes but its time is kept apart (rescore_runtime_s)
                     new_trainings_seconds_from_pt=round(sum(float(torch.load(f"{PROJ}/results/models/lstm_{nm}_s{s}.pt", map_location="cpu", weights_only=True)["seconds"]) for nm in ("wqh_to09", "wqh_utc", "wqnh") for s in (0, 1, 2)), 1))
print("passes:", out["passes"], "| second pass in the JSON:", P[1]["n_trainings"], P[1]["training_seconds"])
# paired onset counts, LSTM against the simple models
def flags(name, k=1):
    t = onset_table(name, k); return np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
fl = {n_: flags(n_) for n_ in ("lstm_wqh", "base_linreg_wqh", "base_rf_wqh", "lstm_wqnh")}
out["paired"] = {}
for other in ("base_linreg_wqh", "base_rf_wqh"):
    for lab in ("lstm_wqh", "lstm_wqnh"):
        a, b = fl[lab], fl[other]; oa, ob = int((a & ~b).sum()), int((~a & b).sum())
        out["paired"][f"{lab}_vs_{other}"] = dict(both=int((a & b).sum()), only_lstm=oa, only_simple=ob, p=float(binomtest(oa, oa + ob, .5).pvalue))
        print(f"paired onsets {lab} vs {other}: both {int((a & b).sum())}, only the LSTM {oa}, only the simple model {ob}, exact sign-test p {binomtest(oa, oa + ob, .5).pvalue:.3f}")
# Daymet window per basin (day_windows.json) and the effect of leaving Myakka out
DW = jload("results/day_windows.json")["day_ends"]["daymet"]["per_basin"]
mid = {g: v["middle"] for g, v in DW.items()}
out["daymet_window_end_by_basin_local_hour"] = {F.SHORT[g]: round(v, 2) for g, v in mid.items()}
print("Daymet window end per basin (middle of the four readings, local standard hour):", {F.SHORT[g]: round(v, 1) for g, v in mid.items()})
def hits_by_basin(name, k=1):
    t = onset_table(name, k); return {g: (int((h & w).sum()), int(h.sum())) for g, (_, h, w) in t.items()}
hb = {n_: hits_by_basin(n_) for n_ in ("lstm_wqh", "lstm_wqh_to09", "lstm_wqh_utc", "lstm_wq", "lstm_wqn")}
myak = [g for g in ids if F.SHORT[g].startswith("Myakka")][0]
for T, C in (("lstm_wqh", "lstm_wqh_to09"), ("lstm_wqh", "lstm_wqh_utc"), ("lstm_wqh", "lstm_wq")):
    tot_t = sum(hb[T][g][0] for g in ids); tot_c = sum(hb[C][g][0] for g in ids)
    ex_t = sum(hb[T][g][0] for g in ids if g != myak); ex_c = sum(hb[C][g][0] for g in ids if g != myak); n_ex = sum(hb[T][g][1] for g in ids if g != myak)
    out.setdefault("without_myakka", {})[f"{T}_vs_{C}"] = dict(all=(tot_t, tot_c), without_myakka=(ex_t, ex_c), n_onsets_without_myakka=n_ex, difference_points_without_myakka=100 * (ex_t - ex_c) / n_ex)
    print(f"  {T} vs {C}: all 12 basins {tot_t} vs {tot_c} of 611 ({100*(tot_t-tot_c)/611:+.1f} points); without Myakka {ex_t} vs {ex_c} of {n_ex} ({100*(ex_t-ex_c)/n_ex:+.1f} points)")
bas = []
for g in ids:
    bas.append((F.SHORT[g], round(mid[g], 1), hb["lstm_wqh"][g][0], hb["lstm_wqh_to09"][g][0], hb["lstm_wqh"][g][1]))
print("  per basin: Daymet end hour, hits wqh, hits to09, onsets:", bas)
out["per_basin_hits"] = bas
jsave("r2h_records.json", out)
