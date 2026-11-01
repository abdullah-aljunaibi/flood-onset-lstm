"""E7 review, check 1: integrity. The sha256 of every saved file of script 28 and of the stored comparison sets against the results file's own record; the script's
hash now against the hash recorded at the start of the training pass, and its modification time against the first network written (the rule and the sets were in
the hashed file before training); the ROADMAP.md E7 section's hash now against the one recorded; the meta of every new prediction set; inside every .pt: settings =
results/lstm_all.json, seed, columns, split, stopping point = first argmax of the validation NSE, the patience stop, the parameter count, the input-column hashes and
the training record against the results file; the run log against the results file. About 5 s."""
import re

import torch

from m_common import *

t0 = time.time()
R = jload(RESULTS)
C = Compare()
CONF = jload("results/lstm_all.json")["config"]
out = {}

# ---- 1. file hashes against the results file's integrity record ---------------------------------------------------------------------------------------------
I = R["integrity"]
mism = [f for f, h in I["sha256_of_saved_files"].items() if sha256(f) != h]
C.add("saved files: all sha256 equal to integrity.sha256_of_saved_files", True, not mism)
out["saved_files"] = dict(n_files=len(I["sha256_of_saved_files"]), mismatches=mism, made_in_this_pass=len(I["made_in_this_pass"]))
mism_s = [f for f, h in I["stored_sets_read"].items() if sha256(f) != h]
C.add("stored sets: all sha256 equal to integrity.stored_sets_read", True, not mism_s)
mism_n = [f for f, h in I["stored_networks_read_by_the_column_identity_check"].items() if sha256(f) != h]
C.add("identity-check networks: sha256 equal", True, not mism_n)
out["stored_set_files"] = dict(n_files=len(I["stored_sets_read"]), mismatches=mism_s, identity_networks=len(I["stored_networks_read_by_the_column_identity_check"]), identity_mismatches=mism_n)
expected_saved = {f"results/preds/lstm_{n}.npz" for n in NEW} | {f"results/preds/lstm_{n}_s{s}.npz" for n in NEW for s in range(5)} | \
                 {f"results/models/lstm_{n}_s{s}.pt" for n in NEW for s in range(5)} | {"results/preds/base_linreg_wqME.npz", "results/preds/base_rf_wqME.npz"}
C.add("integrity record lists exactly the 35 expected files (3 x (1 + 5 + 5) + 2)", True, set(I["sha256_of_saved_files"]) == expected_saved)

# ---- 2. the script and the rule were fixed before training ---------------------------------------------------------------------------------------------------
sha_now = sha256(SCRIPT)
P = R["passes"]
C.add("script sha256 now = recorded (results script_sha256)", R["script_sha256"], sha_now)
C.add("script sha256 now = training pass", P[0]["script_sha256"], sha_now)
mt_script = (PROJ / SCRIPT).stat().st_mtime
first_pt = min((PROJ / f).stat().st_mtime for f in expected_saved if f.endswith(".pt"))
mt_road = (PROJ / "ROADMAP.md").stat().st_mtime
text = (PROJ / "ROADMAP.md").read_text(encoding="utf-8")
i = text.find("### E7 - several rain products side by side")
j = text.find("\n## ", i)
sec = text[i:j if j > 0 else len(text)].strip()
road_sha = hashlib.sha256(sec.encode("utf-8")).hexdigest()
C.add("ROADMAP E7 section sha256 now = training pass", P[0]["roadmap_e7_section_sha256"], road_sha)
C.add("ROADMAP E7 section sha256 now = fixed_before_training.now", R["fixed_before_training"]["roadmap_e7_section_sha256_now"], road_sha)
C.add("one training pass, 15 trainings, seeds 0-4", True, len(P) == 1 and P[0]["kind"] == "training run" and P[0]["n_trainings"] == 15 and P[0]["seeds"] == [0, 1, 2, 3, 4])
C.add("choices_made_after_looking_at_test_scores is empty", [], R["choices_made_after_looking_at_test_scores"])
fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
out["script"] = dict(sha256_now=sha_now, sha256_recorded=R["script_sha256"], equal=sha_now == R["script_sha256"], modified=fmt(mt_script), roadmap_modified=fmt(mt_road),
                     first_network_written=fmt(first_pt), results_written=R["written"], script_and_roadmap_before_first_network=bool(mt_script < first_pt and mt_road < first_pt),
                     roadmap_e7_sha256_now=road_sha, roadmap_e7_sha256_recorded=P[0]["roadmap_e7_section_sha256"],
                     roadmap_says_split_part_added_before_full_training="the same split part" in sec and "added on 2026-10-30 before the full training" in sec)
C.add("script and ROADMAP modified before the first network was written", True, out["script"]["script_and_roadmap_before_first_network"])
# the smoke test of the same script (results/quick): its record shows what had been seen before the full training
Q = json.loads((PROJ / "results/quick/multi_product.json").read_text(encoding="utf-8")) if (PROJ / "results/quick/multi_product.json").exists() else None
if Q:
    out["smoke_test"] = dict(written=Q["written"], script_sha256=Q["script_sha256"], same_script=Q["script_sha256"] == sha_now, quick_mode=Q["quick_mode"],
                             roadmap_sha256=Q["passes"][0]["roadmap_e7_section_sha256"], seeds=Q["passes"][0]["seeds"], n_trainings=Q["passes"][0]["n_trainings"],
                             shuffle_share_from_another_split_part=Q["inputs"]["shuffle"]["share_from_another_split_part"],
                             note="the smoke test that ran with the final script (22:28, 40 updates, one seed) scored the test years; an earlier smoke test with the whole-record shuffle is described in the script but not stored")

# ---- 3. prediction-set meta ---------------------------------------------------------------------------------------------------------------------------------------
meta_bad = []
for n in NEW:
    dyn = FORCING + list(EXTRA[n])
    for name, seed in [(f"lstm_{n}", None)] + [(f"lstm_{n}_s{s}", s) for s in range(5)]:
        m = load_npz(name)[4]
        ok = m["script"] == "28_multi_product.py" and m["config"] == CONF and m["dyn"] == dyn and m["inputs"] == n and m["shuffle_seed"] == SHUF_SEED and \
            (m.get("seed") == seed if seed is not None else m.get("seeds") == [0, 1, 2, 3, 4])
        if not ok:
            meta_bad.append(name)
for n in ("base_linreg_wqME", "base_rf_wqME"):
    m = load_npz(n)[4]
    if m["script"] != "28_multi_product.py" or m["extra_inputs"] != list(EXTRA["wqME"]) or m["quick_mode"]:
        meta_bad.append(n)
C.add("prediction-set meta (script, config = lstm_all, dyn, seed) of the 20 new sets", [], meta_bad)

# ---- 4. inside every .pt --------------------------------------------------------------------------------------------------------------------------------------------
per_file, pt_bad = {}, []
SHA_IN = R["inputs"]["sha256_of_input_columns"]
for n in NEW:
    for s in range(5):
        z = torch.load(PROJ / f"results/models/lstm_{n}_s{s}.pt", map_location="cpu", weights_only=True)
        cfg = z["cfg"]
        hist = z["history"]
        vals = [h["val_nse"] for h in hist]
        first_best = hist[int(np.argmax(vals))]["update"]
        # patience stop: training ends `patience` checks after the best one, or at the update budget
        stop_ok = z["updates_run"] == min(CONF["updates"], first_best + CONF["patience"] * CONF["eval_every"])
        n_in = len(cfg["dyn"]) + 1 + len(STATIC)
        rec = next(r for r in R["models"][n]["per_seed"] if r["seed"] == s)
        checks = dict(settings_equal_lstm_all=all(cfg[k] == v for k, v in CONF.items()), seed=cfg["seed"] == s == z["seed"], dyn=list(cfg["dyn"]) == FORCING + list(EXTRA[n]) == z["dyn"],
                      use_flow=cfg["use_flow"] is True, future=cfg["future"] is False, static=list(cfg["static"]) == STATIC, split=dict(z["split"]) == {k: list(v) for k, v in SPLIT.items()} or
                      {k: tuple(v) for k, v in z["split"].items()} == SPLIT, ids_12=len(z["ids"]) == 12, input_set=z["input_set"] == n, script=z["script"] == "28_multi_product.py",
                      best_update_is_first_argmax=z["best_update"] == first_best, best_val_nse_is_max=abs(z["best_val_nse"] - max(vals)) < 1e-12, patience_stop=stop_ok,
                      n_params=z["n_params"] == 4 * 64 * (n_in + 64 + 2) + 65 * 3 == sum(t.numel() for t in z["state_dict"].values()),
                      input_sha=all(z["input_sha256"][c] == SHA_IN[c] for c in EXTRA[n]), shuffle_seed=z["shuffle_seed"] == SHUF_SEED, quick=z["quick_mode"] is False,
                      record_equal=(rec["updates_run"] == z["updates_run"] and rec["best_update"] == z["best_update"] and abs(rec["best_val_nse"] - round(z["best_val_nse"], 4)) < 1e-12))
        per_file[f"lstm_{n}_s{s}"] = dict(n_in=n_in, best_update=z["best_update"], updates_run=z["updates_run"], best_val_nse=z["best_val_nse"], **checks)
        if not all(checks.values()):
            pt_bad.append(f"lstm_{n}_s{s}: " + ", ".join(k for k, v in checks.items() if not v))
C.add("all 15 .pt files: settings, seed, columns, split, stopping point, patience, n_params, input hashes, record", [], pt_bad)
out["networks"] = dict(per_file=per_file, problems=pt_bad)

# ---- 5. the log ---------------------------------------------------------------------------------------------------------------------------------------------------
L = (PROJ / LOG).read_text(encoding="utf-8").splitlines()
train_lines = [x for x in L if x.startswith("training lstm_")]
upd_lines = [x for x in L if x.startswith("  [lstm_")]
seed_lines = [x for x in L if re.match(r"lstm_wqME\S* seed \d \|", x)]
agree = 0
for x in seed_lines:
    m = re.match(r"lstm_(\S+) seed (\d) \| best at update (\d+) of (\d+), \d+ s \| validation NSE ([0-9.]+) \| test NSE median ([0-9.]+) ([0-9.]+) ([0-9.]+) \| onsets warned 1 day ([0-9.]+)%", x)
    n, s = m.group(1), int(m.group(2))
    rec = next(r for r in R["models"][n]["per_seed"] if r["seed"] == s)
    ok = (int(m.group(3)) == rec["best_update"] and int(m.group(4)) == rec["updates_run"] and abs(float(m.group(5)) - rec["best_val_nse"]) < 0.0006
          and all(abs(float(m.group(6 + j)) - rec["test_nse_median"][j]) < 0.0006 for j in range(3)) and abs(float(m.group(9)) - 100 * rec["test_onset_warned"][0]) < 0.06)
    agree += ok
n_upd = sum(r["updates_run"] for n in NEW for r in R["models"][n]["per_seed"]) // CONF["eval_every"]
out["log"] = dict(training_lines=len(train_lines), seed_summary_lines=len(seed_lines), seed_lines_agree_with_results=agree, validation_check_lines=len(upd_lines), expected_check_lines=n_upd,
                  identity_line=any(x.startswith("COLUMN-IDENTITY CHECK: PASS") for x in L), reload_line=any(x.startswith("reload check passed") for x in L),
                  made_up_line=any(x.startswith("made-up-values test passed") for x in L), exit_0=L[-1].strip() == "EXIT 0", first_line=L[0][:200])
C.add("log: 15 training lines, 15 seed lines agreeing with the results file", [15, 15, 15], [len(train_lines), len(seed_lines), int(agree)])
C.add("log: one validation line per check (sum of updates_run / eval_every)", n_upd, len(upd_lines))
C.add("log: identity, reload and made-up-values lines present, exit 0", True, out["log"]["identity_line"] and out["log"]["reload_line"] and out["log"]["made_up_line"] and out["log"]["exit_0"])

REVIEWED = ["code/28_multi_product.py", "code/flood.py", "code/warning_policy.py", "code/features.py", "code/subdaily_inputs.py", "code/fetch_aorc.py", "code/camels.py", "code/13_uncertainty.py",
            "results/multi_product.json", "figures/multi_product.png", LOG, "results/lstm_all.json", "results/flood_analysis.json", "results/day_windows.json", "results/baselines.json",
            "results/aorc_rain_to_midnight.json", "results/rain_to_midnight.json", "results/review_aorc_rain_to_midnight.json", "results/rain_timing.json", "ROADMAP.md",
            "NEXT_EXPERIMENT_E7_multi_product.md", "HANDOFF.md", "CLAUDE.md"]
out["files_reviewed_sha256"] = {f: sha256(f) for f in REVIEWED}
out["compare"] = C.summary()
out["rows"] = C.rows
out["seconds"] = round(time.time() - t0, 1)
jsave("m1_integrity.json", out)
print(json.dumps(out["compare"]), json.dumps(out["script"]), json.dumps(out["log"]))
assert not out["compare"]["not_matching"], out["compare"]["not_matching"]
