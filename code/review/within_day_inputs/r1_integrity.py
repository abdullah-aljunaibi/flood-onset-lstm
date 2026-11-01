"""E4b review, check 1: integrity and provenance. Hashes of every saved prediction set and network against the results file's own integrity record; the script's hash
now against the one recorded at the start of the run and the file's modification time against the first network written; the stored sets' hashes; the meta of every
prediction set and the settings, seed, input columns, split, stopping point and training record inside every .pt file against results/lstm_all.json and the results file;
the log of the run. No table is built, nothing is predicted. Seconds."""
from r_common import *
import torch

t0 = time.time()
R = jload(RESULTS)
conf = jload("results/lstm_all.json")["config"]
out = {}

# 1. the saved files of this experiment and the stored sets, as they are now
I = R["integrity"]
now = {f: sha256(f) for f in I["sha256_of_saved_files"]}
bad = [f for f in now if now[f] != I["sha256_of_saved_files"][f]]
now_st = {f: sha256(f) for f in I["stored_sets_read"]}
bad_st = [f for f in now_st if now_st[f] != I["stored_sets_read"][f]]
out["saved_files"] = dict(n_files=len(now), all_hashes_equal_to_the_record=not bad, mismatches=bad, made_in_this_pass=len(I["made_in_this_pass"]), only_read=I["only_read_in_this_pass"])
out["stored_set_files"] = dict(n_files=len(now_st), all_hashes_equal_to_the_record=not bad_st, mismatches=bad_st)
print(f"saved files: {len(now)} hashed, {len(bad)} differ from the record; stored-set files: {len(now_st)} hashed, {len(bad_st)} differ")

# 2. the script: hash now against the hash recorded when the run started; modified before the first network was written?
pt_files = sorted((PROJ / "results" / "models").glob("lstm_wqa[QW]*_s?.pt"))
first_pt = min(f.stat().st_mtime for f in pt_files)
mt = (PROJ / SCRIPT).stat().st_mtime
out["script"] = dict(sha256_now=sha256(SCRIPT), sha256_recorded=R["script_sha256"], equal=sha256(SCRIPT) == R["script_sha256"] == I["script_sha256"] == R["passes"][0]["script_sha256"],
                     modified=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)), first_network_written=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first_pt)),
                     results_written=R["written"], script_unchanged_since_before_training=bool(mt < first_pt and sha256(SCRIPT) == R["script_sha256"]),
                     success_rule_in_the_hashed_file="SUCCESS_RULE = (" in (PROJ / SCRIPT).read_text(encoding="utf-8"))
out["code_files_sha256"] = {f: sha256(f) for f in (SCRIPT, "code/subdaily_inputs.py", "code/flood.py", "code/warning_policy.py", "code/fetch_aorc.py", "code/gefs_to_basins.py", "code/check_reload.py",
                                                    "results/within_day_inputs.json", "figures/within_day_inputs.png", "results/lstm_all.json", "results/flood_analysis.json", "results/day_windows.json",
                                                    "results/gefs_basins.json", "results/aorc_rain_to_midnight.json", "results/forecast_input.json", "results/review_forecast_input.json",
                                                    "results/diagnostics_2026-10-28/subdaily_inputs_check.json", "results/logs/21_within_day_inputs_2026-10-28.log", "NEXT_EXPERIMENTS_2026-10-28.md", "HANDOFF.md")}
print("script hash equal to the record:", out["script"]["equal"], "| modified", out["script"]["modified"], "| first network", out["script"]["first_network_written"])

# 3. the settings: results file against results/lstm_all.json; the meta of every prediction set
out["config"] = dict(results_config=R["config"], lstm_all_config=conf, equal=R["config"] == conf, config_source=R["config_source"])
design = R["design"]["sets"]
meta_ok, meta_bad = 0, []
for n in NEW:
    for s in SEEDS[n]:
        m = load_npz(f"lstm_{n}_s{s}")[4]
        checks = dict(script=m.get("script") == "21_within_day_inputs.py", inputs=m.get("inputs") == n, seed=m.get("seed") == s, config=m.get("config") == conf, dyn=m.get("dyn") == design[n]["dyn"],
                      extra=m.get("extra_inputs") == design[n]["extra_columns"], control=m.get("control") == (n == "wqaW_older"), model_file=m.get("model_file") == f"results/models/lstm_{n}_s{s}.pt",
                      maxcell_fill_days=m.get("maxcell_fill_days") == R["maxcell_fill_days"]["n_basin_days_filled"])
        (meta_bad.append((f"lstm_{n}_s{s}", [k for k, v in checks.items() if not v])) if not all(checks.values()) else None)
        meta_ok += all(checks.values())
    m = load_npz(f"lstm_{n}")[4]
    checks = dict(script=m.get("script") == "21_within_day_inputs.py", inputs=m.get("inputs") == n, seeds=m.get("seeds") == SEEDS[n], config=m.get("config") == conf, dyn=m.get("dyn") == design[n]["dyn"])
    (meta_bad.append((f"lstm_{n}", [k for k, v in checks.items() if not v])) if not all(checks.values()) else None)
    meta_ok += all(checks.values())
for n in STORED:
    m = load_npz(f"lstm_{n}")[4]
    checks = dict(script=m.get("script") == R["design"]["stored_sets"][n]["script"], config=m.get("config") == conf, dyn=m.get("dyn") == R["design"]["stored_sets"][n]["dyn"], seeds=m.get("seeds") == SEEDS[n])
    (meta_bad.append((f"lstm_{n}", [k for k, v in checks.items() if not v])) if not all(checks.values()) else None)
    meta_ok += all(checks.values())
    for s in SEEDS[n]:
        m = load_npz(f"lstm_{n}_s{s}")[4]
        checks = dict(config=m.get("config") == conf, seed=m.get("seed") == s, dyn=m.get("dyn") == R["design"]["stored_sets"][n]["dyn"])
        (meta_bad.append((f"lstm_{n}_s{s}", [k for k, v in checks.items() if not v])) if not all(checks.values()) else None)
        meta_ok += all(checks.values())
out["prediction_set_meta"] = dict(sets_checked=meta_ok + len(meta_bad), all_ok=not meta_bad, problems=meta_bad)
print(f"prediction-set meta: {meta_ok} ok, {len(meta_bad)} with a problem {meta_bad}")

# 4. inside every .pt file: settings, seed, columns, split, the stopping point, the training record
n_static = len(STATIC)
pt = {}
training = {(t["set"], t["seed"]): t for t in R["training"]}
for n in NEW:
    for s in SEEDS[n]:
        z = torch.load(PROJ / "results" / "models" / f"lstm_{n}_s{s}.pt", map_location="cpu", weights_only=True)
        cfg = z["cfg"]
        n_in = len(cfg["dyn"]) + 1 + n_static
        hist = z["history"]
        best_i = int(np.argmax([h["val_nse"] for h in hist]))
        params = sum(int(np.prod(t.shape)) for t in z["state_dict"].values())
        tr = training[(n, s)]
        rec = dict(settings_equal_lstm_all=all(cfg[k] == conf[k] for k in conf), seed=cfg["seed"] == s, use_flow=cfg["use_flow"] is True, future_false=cfg["future"] is False, static_is_STATIC=list(cfg["static"]) == STATIC,
                   dyn_as_design=list(cfg["dyn"]) == design[n]["dyn"] == z["dyn"], n_in=n_in, weights_fit_n_in=tuple(z["state_dict"]["lstm.weight_ih_l0"].shape) == (4 * conf["hidden"], n_in),
                   hidden_from_weights=int(z["state_dict"]["lstm.weight_hh_l0"].shape[1]) == conf["hidden"], split_ok=all(tuple(z["split"][k]) == SPLIT[k] for k in SPLIT),
                   best_update_is_argmax_val_nse=hist[best_i]["update"] == z["best_update"], best_val_nse_equals_history=abs(hist[best_i]["val_nse"] - z["best_val_nse"]) < 1e-12,
                   stopped_by_patience_or_cap=(z["updates_run"] - z["best_update"] == conf["patience"] * conf["eval_every"]) or z["updates_run"] == conf["updates"],
                   n_params_equal=params == z["n_params"] == 4 * conf["hidden"] * (n_in + conf["hidden"] + 2) + (conf["hidden"] + 1) * 3,
                   training_record_equal=all(abs(float(tr[k]) - float(z[k])) < 0.051 for k in ("seconds", "cpu_seconds")) and all(tr[k] == z[k] for k in ("updates_run", "best_update", "n_params")) and abs(tr["best_val_nse"] - z["best_val_nse"]) < 5e-5,
                   quick_mode_false=not z["quick_mode"], script=z["script"] == "21_within_day_inputs.py", input_set=z["input_set"] == n, extra_columns=list(z["extra_columns"]) == design[n]["extra_columns"],
                   torch_version=z["torch_version"], history_checks=len(hist), updates_run=z["updates_run"], best_update=z["best_update"], best_val_nse=round(float(z["best_val_nse"]), 4),
                   how_to_reload_present="how_to_reload" in z, maxcell_fill_days_stored="maxcell_fill_days" in z)
        pt[f"lstm_{n}_s{s}"] = rec
flags = [k for k, v in next(iter(pt.values())).items() if isinstance(v, bool)]
out["networks"] = dict(n_files=len(pt), flags_true_in={k: f"{sum(bool(v[k]) for v in pt.values())} of {len(pt)}" for k in flags}, torch_versions=sorted({v["torch_version"] for v in pt.values()}), per_file=pt)
print("network files:", out["networks"]["flags_true_in"])

# 5. the stored sets' networks: not reloaded here (trained on the Mac); their settings only
for n in STORED:
    z = torch.load(PROJ / "results" / "models" / f"lstm_{n}_s0.pt", map_location="cpu", weights_only=True)
    out.setdefault("stored_networks_settings", {})[f"lstm_{n}_s0"] = dict(settings_equal_lstm_all=all(z["cfg"][k] == conf[k] for k in conf), dyn=list(z["cfg"]["dyn"]), script=z.get("script"), torch_version=z.get("torch_version"))

# 6. the log of the run and the training block
log = (PROJ / "results" / "logs" / "21_within_day_inputs_2026-10-28.log").read_text(encoding="utf-8")
lines = [l for l in log.splitlines() if l.startswith("training lstm_")]
import re
seed_lines = {}
for l in log.splitlines():
    m = re.match(r"lstm_(\w+) seed (\d) \| best at update (\d+) of (\d+), (\d+) s \| validation NSE ([0-9.]+) \| test NSE median ([0-9.]+) ([0-9.]+) ([0-9.]+) \| onsets warned 1 day ([0-9.]+)%", l)
    if m:
        seed_lines[(m.group(1), int(m.group(2)))] = dict(best=int(m.group(3)), run=int(m.group(4)), val=float(m.group(6)), nse1=float(m.group(7)), pod1=float(m.group(10)))
log_vs_file = []
for n in NEW:
    rg = R["seed_ranges"]["sets"][n]["l1"]["by_seed"]
    for j, s in enumerate(SEEDS[n]):
        l = seed_lines.get((n, s))
        ok = l is not None and abs(l["pod1"] - 100 * rg["onset_pod"][j]) < 0.051 and abs(l["nse1"] - rg["nse_median"][j]) < 5e-4 and l["best"] == training[(n, s)]["best_update"] and l["run"] == training[(n, s)]["updates_run"]
        log_vs_file.append(ok)
out["log"] = dict(training_lines=len(lines), expected=20, seed_summary_lines=len(seed_lines), seed_lines_agree_with_results_file=all(log_vs_file), n_agree=int(sum(log_vs_file)),
                  reload_check_line="reload check passed" in log, made_up_values_line="made-up-values test passed" in log, subdaily_checks_passed="all checks passed; wrote" in log,
                  first_line=log.splitlines()[0][:160])
out["training_block"] = dict(n_entries=len(R["training"]), n_trainings=R["n_trainings"], passes=len(R["passes"]), pass_kind=R["passes"][0]["kind"], pass_trained=len(R["passes"][0]["trained"]),
                             seeds=R["seeds"], with_forecast=R["passes"][0]["with_forecast_set"], torch_threads=R["passes"][0].get("torch_threads"), training_seconds=R["training_seconds"],
                             finished=R["passes"][0]["finished_local_time"], all_20_pairs_present=sorted(training) == sorted((n, s) for n in NEW for s in SEEDS[n]))
out["status_line"] = R["status"]
out["choices_made_after_looking_at_test_scores"] = R["choices_made_after_looking_at_test_scores"]
out["seconds"] = round(time.time() - t0, 1)
jsave("r1_integrity.json", out)
print("log:", {k: v for k, v in out["log"].items() if k != "first_line"})
print("seconds:", out["seconds"])
