"""E2 review, the inventory and checks (h): the files under review with sha256, size and modification time (the baseline for files_reviewed of the record), the results file's own integrity record against the disk (72 saved
files, the script hash, the 12 forecast tables, the hash of gefs_to_basins.py), the passes, the order of the file times, the engine files against the hand-in zip (E1_ZIP_DIR, optional) and against what the
E1b review saw (E1B_REVIEW_RECORD, optional), the base set of E1b (lstm_wq, lstm_wqa, lstm_wqf and the networks of script 16) against the hashes the E1b review recorded, the lines of run_all.sh. A few seconds."""
import sys; sys.dont_write_bytecode = True
import hashlib, time, glob, re
from f_common import *

sha = lambda rel: hashlib.sha256(open(f"{PROJ}/{rel}", "rb").read()).hexdigest()
mt = lambda rel: os.path.getmtime(f"{PROJ}/{rel}")
tf = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
fi = lambda rel: dict(sha256=sha(rel), bytes=os.path.getsize(f"{PROJ}/{rel}"), modified=tf(mt(rel)))
R = jload("results/forecast_input.json")
G = jload("results/gefs_basins.json")
out = {}
code = ["code/18_forecast_input.py", "code/gefs_to_basins.py", "code/fetch_gefs.py", "code/flood.py", "code/test_flood.py", "code/features.py", "code/camels.py", "code/common.py", "code/fetch_aorc.py", "code/02_baselines.py", "run_all.sh"]
other = ["data/gefs/README.md", "results/forecast_input.json", "results/gefs_basins.json", "results/baselines.json", "results/lstm_all.json", "results/day_windows.json", "results/flood_analysis.json", "results/data_access.json",
         "results/aorc_rain_to_midnight.json", "results/review_aorc_rain_to_midnight.json", "figures/forecast_input.png", "results/logs/18_forecast_input.log", "results/logs/18_forecast_input_rescore.log", "results/logs/fetch_gefs.log",
         "data/gefs/c00/done_apcp.txt"]
tables = [f"data/gefs/basins/{os.path.basename(f)}" for f in sorted(glob.glob(f"{PROJ}/data/gefs/basins/gefs_c00_*.csv"))]
months = [os.path.relpath(f, PROJ) for f in sorted(glob.glob(f"{PROJ}/data/gefs/c00/apcp_*.npz")) + sorted(glob.glob(f"{PROJ}/data/gefs/c00/apcp_*.json"))]
saved = list(R["integrity"]["sha256_of_saved_files"])
derived = sorted(os.path.relpath(f, PROJ) for n in ("wqaF_sub_rawpred", "wqaF_sub_perfectpred", "wqaP_forecastpred") for f in glob.glob(f"{PROJ}/results/preds/lstm_{n}*.npz"))
base_e1b = ["results/preds/lstm_wq.npz", "results/preds/lstm_wqa.npz", "results/preds/lstm_wqf.npz"] + [f"results/preds/lstm_{n}_s{s}.npz" for n in ("wq", "wqa", "wqf") for s in SEEDS if os.path.exists(f"{PROJ}/results/preds/lstm_{n}_s{s}.npz")] + [f"results/models/lstm_wqa_s{s}.pt" for s in SEEDS]
base_e1b = [f for f in base_e1b if os.path.exists(f"{PROJ}/{f}")]
aorc = [f"data/aorc/aorc_{g}.npz" for g in G["basins"]]
files = code + other + tables + months + saved + derived + base_e1b + aorc
inv = {f: fi(f) for f in files}
out["inventory"] = inv
print(f"{len(inv)} files hashed: {len(code)} code, {len(other)} other records, {len(tables)} forecast tables, {len(months)} monthly store files, {len(saved)} saved sets and networks of E2, {len(derived)} derived sets, {len(base_e1b)} base files of E1b, {len(aorc)} AORC series")
for f in code + other[:9]: print(f"  {f:50s} {inv[f]['modified']}  {inv[f]['bytes']:>9d} bytes  sha256 {inv[f]['sha256'][:12]}")

# the results file's own integrity record
I = R["integrity"]; rec = I["sha256_of_saved_files"]
dis = sorted(p for p, h in rec.items() if not os.path.exists(f"{PROJ}/{p}") or inv[p]["sha256"] != h)
tab_rec = I["sha256_of_the_forecast_tables_read"]
tab_dis = sorted(p for p, h in tab_rec.items() if inv[p]["sha256"] != h)
g_rec = G["files"]; g_dis = sorted(p for p, h in g_rec.items() if inv[p]["sha256"] != h)
print(f"\nintegrity record of the results file: {len(rec)} saved files; disagree with the disk: {dis}; the 12 forecast tables it read: disagree {tab_dis}; the tables recorded in gefs_basins.json: disagree {g_dis}")
out["json_record"] = dict(n_saved=len(rec), disagree=dis, tables_disagree=tab_dis, gefs_basins_tables_disagree=g_dis, made_in_this_pass=len(I["made_in_this_pass"]), only_read_in_this_pass=I["only_read_in_this_pass"], compared_with_the_earlier_results_file=I["compared_with_the_earlier_results_file"])
h_script = inv["code/18_forecast_input.py"]["sha256"]; h_gb = inv["code/gefs_to_basins.py"]["sha256"]
passes = [dict(kind=p["kind"], n_trainings=p["n_trainings"], finished=p["finished_local_time"], script_sha12=p["script_sha256"][:12], sets_trained=p.get("sets_trained")) for p in R["passes"]]
print(f"18_forecast_input.py on disk {h_script[:12]} (modified {inv['code/18_forecast_input.py']['modified']}): equal to integrity.script_sha256 {I['script_sha256'][:12]}: {h_script == I['script_sha256']}; passes:", passes)
print(f"gefs_to_basins.py on disk {h_gb[:12]} equals the hash recorded in gefs_basins.json ({G['code']['sha256'][:12]}) and in forecast_input.json ({I['gefs_basins_json_code_sha256'][:12]}): {h_gb == G['code']['sha256'] == I['gefs_basins_json_code_sha256']}")
t_models = sorted(mt(f) for f in saved if f.endswith(".pt")); t_json = mt("results/forecast_input.json"); t_script = mt("code/18_forecast_input.py")
order = dict(script_modified=tf(t_script), first_network_written=tf(t_models[0]), last_network_written=tf(t_models[-1]), results_file_written=tf(t_json), gefs_basins_written=tf(mt("results/gefs_basins.json")), gefs_to_basins_modified=tf(mt("code/gefs_to_basins.py")),
             tables_written=sorted({tf(mt(f)) for f in tables})[:3], figure_written=tf(mt("figures/forecast_input.png")), script_newer_than_the_last_network=t_script > t_models[-1], results_newer_than_the_script=t_json > t_script, gefs_to_basins_older_than_its_record=mt("code/gefs_to_basins.py") < mt("results/gefs_basins.json"))
print("times:", order)
out["script"] = dict(sha256=h_script, integrity_equal=(h_script == I["script_sha256"]), gefs_to_basins_hash_equal=(h_gb == G["code"]["sha256"] == I["gefs_basins_json_code_sha256"]), passes=passes, order=order,
                     training_pass_script_sha12=R["passes"][0]["script_sha256"][:12], note="the script on disk is the rescore version (the training pass ran an earlier version; its hash is in passes)")

# the base set of E1b against the hashes of the E1b review
try:
    rv = jload("results/review_aorc_rain_to_midnight.json")["files_reviewed"]
    rows = {f: (inv[f]["sha256"] == rv[f]["sha256"]) for f in base_e1b if f in rv}
    nf = [f for f in base_e1b if f not in rv]
    print(f"\nbase files of E1b recorded in the E1b review: {len(rows)} compared, unchanged: {sum(rows.values())}; not in that record: {nf}")
    out["e1b_base_files_vs_the_e1b_review"] = dict(compared=len(rows), unchanged=int(sum(rows.values())), changed=[f for f, v in rows.items() if not v], not_in_record=nf)
    eng = {f: (inv[f]["sha256"] == rv[f]["sha256"]) for f in ("code/flood.py", "code/test_flood.py", "code/features.py", "code/camels.py", "code/02_baselines.py", "results/baselines.json") if f in rv}
    print("engine and baseline files against the E1b review record (unchanged = True):", eng)
    out["engine_vs_the_e1b_review"] = eng
    changed_e1b = {f: dict(then=rv[f]["modified"], now=inv[f]["modified"]) for f in ("code/16_aorc_rain_to_midnight.py", "code/fetch_aorc.py", "results/aorc_rain_to_midnight.json", "run_all.sh") if f in rv and f in inv and rv[f]["sha256"] != inv[f]["sha256"]}
    out["e1b_files_changed_since_the_e1b_review"] = changed_e1b
    print("files of E1b that changed since the E1b review:", changed_e1b)
except Exception as e:
    print("E1b review record not read:", e); out["e1b_review_record"] = str(e)

zd = os.environ.get("E1_ZIP_DIR", "")
if zd and os.path.exists(zd):
    out["engine_vs_zip"] = {f: (hashlib.sha256(open(f"{zd}/{f}", "rb").read()).hexdigest() == sha(f"code/{f}")) for f in ("flood.py", "test_flood.py", "features.py", "camels.py", "02_baselines.py", "common.py") if os.path.exists(f"{zd}/{f}")}
    print("code files equal to the hand-in zip of 4 October:", out["engine_vs_zip"])
ra = open(f"{PROJ}/run_all.sh").read().splitlines()
out["run_all"] = {i + 1: l.strip()[:200] for i, l in enumerate(ra) if re.search(r"16_aorc|17_more|18_forecast|fetch_gefs|gefs_to_basins|check_aorc|15_rain", l)}
print("run_all.sh lines:", out["run_all"])
out["modified_after_the_results_file"] = {f: v["modified"] for f, v in inv.items() if mt(f) > t_json + 1}
print("files of the list modified after the results file was written:", out["modified_after_the_results_file"])
jsave("f1_integrity.json", out)
