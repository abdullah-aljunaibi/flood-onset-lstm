"""E4e review, check 1: integrity and provenance. Hashes of every saved prediction set and network against the results file's own integrity record; the hourly
files and the stored sets' files; the script's hash now against the one recorded at the start of the run, and the file's modification time against the first network
written (the success rule was in the hashed file before training); the meta of every prediction set; inside every .pt: settings = results/lstm_all.json, seed,
columns, split, stopping point = argmax of the validation NSE, patience stop, parameter count; the log of the run against the file. Nothing is predicted. Seconds."""
from h_common import *
import torch

t0 = time.time()
R = jload(RESULTS)
conf = jload("results/lstm_all.json")["config"]
out = {}

# 1. the saved files of this experiment, the stored sets' files and the hourly files, as they are now
I = R["integrity"]
now = {f: sha256(f) for f in I["sha256_of_saved_files"]}
bad = [f for f in now if now[f] != I["sha256_of_saved_files"][f]]
now_st = {f: sha256(f) for f in I["stored_sets_read"]}
bad_st = [f for f in now_st if now_st[f] != I["stored_sets_read"][f]]
now_h = {g: sha256(f"data/hourly_flow/usgs_iv_{g}.npz") for g in I["hourly_files_read"]}
bad_h = [g for g in now_h if now_h[g] != I["hourly_files_read"][g]]
expected_files = {f"results/preds/lstm_{n}.npz" for n in NEW} | {f"results/preds/lstm_{n}_s{s}.npz" for n in NEW for s in SEEDS[n]} | {f"results/models/lstm_{n}_s{s}.pt" for n in NEW for s in SEEDS[n]}
out["saved_files"] = dict(n_files=len(now), expected_66_files_all_listed=set(now) == expected_files, all_hashes_equal_to_the_record=not bad, mismatches=bad,
                          made_in_this_pass=len(I["made_in_this_pass"]), all_66_made_in_the_one_pass=set(I["made_in_this_pass"]) == expected_files)
out["stored_set_files"] = dict(n_files=len(now_st), all_hashes_equal_to_the_record=not bad_st, mismatches=bad_st)
out["hourly_files"] = dict(n_files=len(now_h), all_hashes_equal_to_the_record=not bad_h, mismatches=bad_h)
print(f"saved files: {len(now)} hashed, {len(bad)} differ; stored-set files: {len(now_st)} hashed, {len(bad_st)} differ; hourly files: {len(now_h)} hashed, {len(bad_h)} differ")

# 2. the script: hash now against the hash recorded when the run started; modified before the first network was written? success rule in the hashed text?
pt_files = sorted((PROJ / "results" / "models").glob("lstm_wqa*H*_s_s?.pt")) + sorted((PROJ / "results" / "models").glob("lstm_wqa*H22_s_s?.pt"))
first_pt = min(f.stat().st_mtime for f in pt_files)
last_pt = max(f.stat().st_mtime for f in pt_files)
mt = (PROJ / SCRIPT).stat().st_mtime
txt = (PROJ / SCRIPT).read_text(encoding="utf-8")
out["script"] = dict(sha256_now=sha256(SCRIPT), sha256_recorded=R["script_sha256"], equal=sha256(SCRIPT) == R["script_sha256"] == I["script_sha256"] == R["passes"][0]["script_sha256"],
                     modified=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)), first_network_written=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first_pt)),
                     last_network_written=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_pt)), results_written=R["written"], n_network_files=len(pt_files),
                     script_unchanged_since_before_training=bool(mt < first_pt and sha256(SCRIPT) == R["script_sha256"]),
                     success_rule_in_the_hashed_file='SUCCESS_RULE = "wqaH_s above wqa_s in every seed at lead 1' in txt,
                     seeds_option_default_3_and_run_with_5='ap.add_argument("--seeds", type=int, default=3, choices=(3, 5)' in txt and R["seeds"] == [0, 1, 2, 3, 4],
                     passes=R["passes"])
out["code_files_sha256"] = {f: sha256(f) for f in (SCRIPT, "code/subdaily_inputs.py", "code/flood.py", "code/warning_policy.py", "code/fetch_usgs_iv.py", "code/gefs_to_basins.py", "code/camels.py",
                                                    RESULTS, FIGURE, "results/lstm_all.json", "results/flood_analysis.json", "results/day_windows.json", "results/gefs_basins.json",
                                                    "results/forecast_input.json", "results/logs/22_flow_state_2026-10-29.log", "data/hourly_flow/usgs_iv_meta.json", "NEXT_EXPERIMENTS_2026-10-28.md", "HANDOFF.md")}
print("script hash equal to the record:", out["script"]["equal"], "| modified", out["script"]["modified"], "| first network", out["script"]["first_network_written"], "| last", out["script"]["last_network_written"])

# 3. the settings: results file against results/lstm_all.json; the meta of every prediction set
out["config"] = dict(results_config=R["config"], lstm_all_config=conf, equal=R["config"] == conf, config_source=R["config_source"])
design = R["design"]["sets"]
meta_rows, meta_bad = {}, []
for n in NEW:
    for name in [f"lstm_{n}"] + [f"lstm_{n}_s{s}" for s in SEEDS[n]]:
        dates, basins, obs, pred, meta = load_npz(name)
        checks = dict(script=meta.get("script") == "22_flow_state.py", inputs=meta.get("inputs") == n, config=meta.get("config") == conf, dyn=meta.get("dyn") == design[n]["dyn"],
                      extra=meta.get("extra_inputs") == design[n]["extra_columns"], arm=meta.get("arm") == "sat", split={k: tuple(v) for k, v in meta.get("split", {}).items()} == SAT_SPLIT,
                      last_hour=meta.get("last_available_hour") == LAST_HOUR[n], basins_12=len(basins) == 12 and basins == sorted(basins), dates_whole_record=(dates[0] == pd.Timestamp("1980-10-01") and dates[-1] == pd.Timestamp("2014-09-30")),
                      seed=(meta.get("seed") == int(name.split("_s")[-1])) if "_s" in name[5:] and name.split("_s")[-1].isdigit() else (meta.get("seeds") == [0, 1, 2, 3, 4]),
                      shape=pred.shape == (12, len(dates), 3) and obs.shape == (12, len(dates)))
        meta_rows[name] = checks
        if not all(checks.values()):
            meta_bad.append((name, [k for k, v in checks.items() if not v]))
out["prediction_meta"] = dict(n_sets=len(meta_rows), all_checks_pass=not meta_bad, failures=meta_bad)
print(f"prediction meta: {len(meta_rows)} sets, {len(meta_bad)} with a failed check", meta_bad[:5])

# 4. inside every .pt file
pt = {}
for n in NEW:
    for s in SEEDS[n]:
        z = torch.load(PROJ / "results" / "models" / f"lstm_{n}_s{s}.pt", map_location="cpu", weights_only=True)
        cfg = z["cfg"]
        hist = z["history"]
        vals = [h["val_nse"] for h in hist]
        best_i = int(np.argmax(vals))
        n_in = len(cfg["dyn"]) + 1 + len(cfg["static"])
        n_params_formula = 4 * conf["hidden"] * (n_in + conf["hidden"] + 2) + (conf["hidden"] + 1) * 3
        rec = dict(settings_equal_lstm_all=all(cfg[k] == conf[k] for k in conf), seed=cfg["seed"] == s and z["seed"] == s, input_set=z["input_set"] == n, script=z["script"] == "22_flow_state.py",
                   dyn_equals_design=list(cfg["dyn"]) == design[n]["dyn"] == z["dyn"], extra_columns=list(z["extra_columns"]) == design[n]["extra_columns"], last_available_hour=z["last_available_hour"] == LAST_HOUR[n],
                   carry_h=z["carry_h"] == CARRY_H, split=all(tuple(z["split"][k]) == SAT_SPLIT[k] for k in SAT_SPLIT), arm=z["arm"] == "sat", use_flow=cfg["use_flow"] is True, future=cfg["future"] is False,
                   static=list(cfg["static"]) == STATIC, n_in=n_in, n_params=z["n_params"] == n_params_formula == sum(int(np.prod(t.shape)) for t in z["state_dict"].values()),
                   best_update_is_argmax_of_val_nse=hist[best_i]["update"] == z["best_update"], best_val_nse_is_max=abs(max(vals) - z["best_val_nse"]) < 1e-12,
                   patience_stop=(len(hist) - 1 - best_i == conf["patience"]) or (hist[-1]["update"] == conf["updates"]), updates_run=z["updates_run"] == hist[-1]["update"],
                   eval_every=all(hist[i]["update"] == (i + 1) * conf["eval_every"] for i in range(len(hist))), quick_mode=z["quick_mode"] is False, torch_version=z["torch_version"],
                   record_in_results=next((r for r in R["models"][n]["per_seed"] if r["seed"] == s), None) is not None)
        r = next(r for r in R["models"][n]["per_seed"] if r["seed"] == s)
        # the results file holds run.summary(): seconds rounded to 0.1, best_val_nse to 4 decimals; its history went through clean() (NaN load average -> null)
        same_hist = len(r["history"]) == len(hist) and all(all((hv is None and (isinstance(hh[k], float) and np.isnan(hh[k]))) or hv == hh[k] for k, hv in hr.items()) for hr, hh in zip(r["history"], hist))
        rec["results_record_matches_file"] = (abs(r["seconds"] - z["seconds"]) <= 0.051 and abs(r["cpu_seconds"] - z["cpu_seconds"]) <= 0.051 and r["updates_run"] == z["updates_run"] and r["best_update"] == z["best_update"]
                                             and abs(r["best_val_nse"] - z["best_val_nse"]) <= 5.1e-5 and r["n_params"] == z["n_params"] and same_hist)
        pt[f"lstm_{n}_s{s}"] = rec
flags = [k for k in next(iter(pt.values())) if isinstance(next(iter(pt.values()))[k], bool)]
out["networks"] = dict(n_files=len(pt), true_in={k: f"{sum(bool(v[k]) for v in pt.values())} of {len(pt)}" for k in flags}, all_true=all(all(v[k] for k in flags) for v in pt.values()),
                       torch_versions=sorted({v["torch_version"] for v in pt.values()}), n_in={n: pt[f"lstm_{n}_s0"]["n_in"] for n in NEW}, per_file=pt)
print("networks:", out["networks"]["true_in"])

# 5. the log of the run against the file: the per-seed lines, the reload and causality lines, the outcome sentence
log = (PROJ / "results/logs/22_flow_state_2026-10-29.log").read_text(encoding="utf-8", errors="replace")
import re
rows = re.findall(r"^lstm_(\w+) seed (\d) \| best at update (\d+) of (\d+), (\d+) s \| validation NSE ([\d.]+) \| test NSE median ([\d.]+) ([\d.]+) ([\d.]+) \| onsets warned 1 day ([\d.]+)%", log, re.M)
log_ok, log_bad = 0, []
for n, s, bu, ur, sec, vn, n1, n2_, n3, pod in rows:
    r = next(x for x in R["models"][n]["per_seed"] if x["seed"] == int(s))
    ok = (r["best_update"] == int(bu) and r["updates_run"] == int(ur) and abs(r["best_val_nse"] - float(vn)) < 6e-4 and all(abs(r["test_nse_median"][j] - float(v)) < 6e-4 for j, v in enumerate((n1, n2_, n3)))
          and abs(100 * r["test_onset_warned"][0] - float(pod)) < 0.06)
    log_ok += ok
    if not ok:
        log_bad.append(f"lstm_{n}_s{s}")
out["log"] = dict(file="results/logs/22_flow_state_2026-10-29.log", training_lines=len(rows), lines_matching_the_results_file=log_ok, mismatching=log_bad, reload_line="reload check passed" in log and "bit for bit" in log,
                  causality_line="causality: made-up hours passed on 12 basins x 2 variants" in log, outcome_in_log=R["outcome_full"] in log, status_in_log=R["status"] in log,
                  n_trainings_file=R["n_trainings"], training_seconds_file=R["training_seconds"], torch_threads_of_the_run=R["passes"][0]["torch_threads"], machine=R["passes"][0]["machine"])
print(f"log: {len(rows)} training lines, {log_ok} match the file; reload line {out['log']['reload_line']}; causality line {out['log']['causality_line']}; outcome in log {out['log']['outcome_in_log']}")
out["seconds"] = round(time.time() - t0, 1)
jsave("h1_integrity.json", out)
print("h1 done in", out["seconds"], "s")
