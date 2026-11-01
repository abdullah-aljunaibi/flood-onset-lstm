"""w8: does the stored results file come out again from the stored script? A SANDBOX copy of the project is built in OUT/sandbox (code/20_warning_policy.py,
warning_policy.py, flood.py, common.py, camels.py; data/derived/camels12_daymet.pkl; results/flood_analysis.json; the 36 prediction files the record
lists), python code/20_warning_policy.py is run there (2000 draws, about 90 s; nothing is written into the project), and the sandbox's
results/warning_policy.json is compared with the project's leaf by leaf (runtime_s and finished_local_time left out); the figure's bytes are
compared too. The sandbox is deleted at the end. Output: OUT/w8_sandbox_rerun.json."""
from w_common import *
import shutil
import subprocess
import time

R = record()
SB = OUT / "sandbox"
if SB.exists():
    shutil.rmtree(SB)
for rel in ["code/20_warning_policy.py", "code/warning_policy.py", "code/flood.py", "code/common.py", "code/camels.py", "data/derived/camels12_daymet.pkl", "results/flood_analysis.json", "results/data_audit.json"]:
    (SB / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJ / rel, SB / rel)
for rel in R["inputs_sha256"]:
    (SB / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJ / rel, SB / rel)
env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", FLOOD_THREADS="1", MPLBACKEND="Agg")
t0 = time.time()
proc = subprocess.run([sys.executable, str(SB / "code/20_warning_policy.py")], cwd=str(SB), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
secs = round(time.time() - t0, 1)
out = dict(returncode=proc.returncode, seconds=secs, stdout_tail=proc.stdout[-3000:], stderr_tail=proc.stderr[-3000:])
sb_json, sb_fig = SB / "results/warning_policy.json", SB / "figures/warning_policy.png"
if sb_json.exists():
    A = json.loads(sb_json.read_text(encoding="utf-8"))
    skip = {"runtime_s", "finished_local_time"}
    diffs = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k in skip and path == "":
                    continue
                if k not in a or k not in b:
                    diffs.append(dict(where=f"{path}.{k}", sandbox=a.get(k, "MISSING"), project=b.get(k, "MISSING")))
                else:
                    walk(a[k], b[k], f"{path}.{k}" if path else k)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(dict(where=path, sandbox=f"list of {len(a)}", project=f"list of {len(b)}"))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{path}[{i}]")
        else:
            if isinstance(a, float) and isinstance(b, float):
                if not close(a, b, 1e-12):
                    diffs.append(dict(where=path, sandbox=a, project=b))
            elif a != b:
                diffs.append(dict(where=path, sandbox=a, project=b))

    walk(A, R, "")
    out.update(json_regenerated=True, leaf_differences=diffs[:50], n_leaf_differences=len(diffs), sandbox_runtime_s=A.get("runtime_s"), sandbox_finished=A.get("finished_local_time"),
               figure_bytes_identical=(sha256(sb_fig) == sha256(PROJ / "figures/warning_policy.png")) if sb_fig.exists() else None,
               outcome_full_identical=(A["outcome_full"] == R["outcome_full"]))
else:
    out.update(json_regenerated=False)
shutil.rmtree(SB, ignore_errors=True)
jsave("w8_sandbox_rerun.json", out)
print(json.dumps({k: out[k] for k in out if k not in ("stdout_tail", "stderr_tail")}, indent=1, default=str))
if out["returncode"]:
    print(out["stderr_tail"])
