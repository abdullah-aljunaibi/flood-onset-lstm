"""w0: what exactly is under review. sha256 of the reviewed files against the record's own hashes (script_sha256, inputs_sha256), the file times,
the record's design constants against the brief (grid, cap, halves, splits, bootstrap settings), the GO criteria it read, and the environment it
reports. Output: OUT/w0_integrity.json."""
from w_common import *

R = record()
files = {}
for rel in ["code/20_warning_policy.py", "code/warning_policy.py", "code/flood.py", "code/common.py", "results/warning_policy.json", "figures/warning_policy.png",
            "results/flood_analysis.json", "NEXT_EXPERIMENTS_2026-10-28.md", "results/diagnostics_2026-10-28/warning_policy_rescore.json"]:
    p = PROJ / rel
    files[rel] = dict(sha256=sha256(p), bytes=p.stat().st_size, modified=pd.Timestamp(p.stat().st_mtime, unit="s").tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")) if p.exists() else "MISSING"

out = dict(files_reviewed=files,
           script_hash_matches_record=files["code/20_warning_policy.py"]["sha256"] == R["script_sha256"],
           record_script_sha256=R["script_sha256"], record_finished=R["finished_local_time"], record_runtime_s=R["runtime_s"], record_quick_mode=R["quick_mode"],
           figure_written_after_record=files["figures/warning_policy.png"]["modified"] >= files["results/warning_policy.json"]["modified"],
           environment=R["environment"], status=R["status"], sets_read=R["sets_read"], sets_skipped=R["sets_skipped_no_stored_forecasts"])

# the prediction files the record says it read, hashed again
inp = {}
for rel, h in R["inputs_sha256"].items():
    p = PROJ / rel
    inp[rel] = dict(exists=p.exists(), hash_matches=(sha256(p) == h) if p.exists() else False)
out["inputs"] = dict(n=len(inp), all_exist=all(v["exists"] for v in inp.values()), all_hashes_match=all(v["hash_matches"] for v in inp.values()),
                     mismatched=[k for k, v in inp.items() if not v["hash_matches"]])
# which sets have seed files, and how many
out["seed_files"] = {n: seeds_of(n) for n in SETS}
out["seed_files_as_recorded"] = {n: R["sets"][n]["seed_sets"] == seeds_of(n) for n in SETS if n in R["sets"]}

# the design constants against the brief
d = R["design"]
out["design_vs_brief"] = dict(
    grid_equal=[float(x) for x in d["trigger_grid"]] == GRID, far_cap_equal=float(d["far_cap"]) == FAR_CAP,
    halves_equal={arm: [list(map(list, HALVES[arm]))] == [[list(h) for h in d["halves_of_the_validation_years"][arm]]] for arm in HALVES},
    splits_equal={arm: {p: list(SPLITS[arm][p]) == list(d["splits"][arm][p]) for p in ("train", "val", "test")} for arm in SPLITS},
    bootstrap=d["bootstrap"], bootstrap_block_days_30=d["bootstrap"]["block_days"] == BLOCK_DAYS, bootstrap_draws_2000=d["bootstrap"]["draws"] == 2000,
    go_rule=d["go_rule"])
crit = json.loads((PROJ / "results/flood_analysis.json").read_text(encoding="utf-8"))["verdict"]["criteria"]
out["go_criteria_in_flood_analysis"] = {k: crit[k] for k in ("GO", "CAUTION", "NOT YET")}
out["go_criteria_equal_my_constants"] = crit["GO"] == f"pod >= {POD_GO:.2f} and far <= {FAR_GO:.2f}" and crit["CAUTION"].startswith(f"csi >= {CSI_CAUTION:.2f}")

# the shapes of the stored sets (dates, NaN forecasts) so the reader knows what "NaN forecasts dropped" amounts to inside val/test
shapes = {}
for n, arm in SETS.items():
    S = Preds(n)
    sp = SPLITS[arm]
    nan_in = {part: int(np.isnan(S.pred[:, mask(S.dates, *sp[part]), :]).sum()) for part in ("val", "test")}
    shapes[n] = dict(first_day=str(S.dates[0].date()), last_day=str(S.dates[-1].date()), n_days=len(S.dates), n_basins=len(S.basins), meta_inputs=S.meta.get("inputs"),
                     meta_script=S.meta.get("script"), nan_forecasts_in_val_and_test=nan_in, nan_obs=int(np.isnan(S.obs).sum()),
                     first_finite_forecast_day=str(S.dates[int(np.argmax(np.isfinite(S.pred).all(axis=(0, 2))))].date()))
out["stored_sets"] = shapes
jsave("w0_integrity.json", out)
print(json.dumps({k: out[k] for k in ("script_hash_matches_record", "figure_written_after_record", "inputs", "design_vs_brief", "go_criteria_equal_my_constants")}, indent=1, default=str))
