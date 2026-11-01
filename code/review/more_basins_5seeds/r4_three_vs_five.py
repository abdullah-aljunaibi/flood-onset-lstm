"""Check t1-t4: what changed from three to five seeds, and the texts.

t1 every leaf of the backup results file (three seeds, Project-Flood_backup_2026-10-30/results/more_basins.json) against the file now: the reading words, flags and verdicts
   (keys reading, clear_gain, clear_loss, single_seeds_all_above/below, p_below_cut, verdict, trigger) that changed; the headline numbers then and now
t2 the rank correlations whose p crossed 0.05; the words the script derives from them
t3 seed-count words in every text field (three / 3 / five / 5 near 'seed' or 'training')
t4 the SEED_NOTE disclosure: present, its factual claims against the network files, code/schedule.py, the venv; the --note text of PATCHES (note17.txt) equal to it
Run: python code/review/more_basins_5seeds/r4_three_vs_five.py
"""
import re
import subprocess

from r_common import *

R, B = results(), results(BACKUP)
FLAGS = {"reading", "clear_gain", "clear_loss", "single_seeds_all_above", "single_seeds_all_below", "p_below_cut", "verdict", "trigger", "higher_in_most_basins",
         "lstm_w_seed_0_is_the_best_of_its_seeds", "rivers_whose_nearest_training_gauge_is_no_closer_with_the_50"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and not (a and isinstance(a[0], (int, float)) and not isinstance(a[0], bool)):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    else:
        key = re.sub(r"\[\d+\]$", "", path.rsplit("/", 1)[-1])
        if key in FLAGS and a != b:
            out.append(dict(where=path.lstrip("/"), three_seeds=a, five_seeds=b))


changed = []
for blk in ("gain_rule", "verdicts", "unseen", "seed_ranges"):
    walk(B[blk], R[blk], blk, changed)
# the verdict words of the single seeds exist only for the seeds both files have; seeds 3-4 are new
changed = [c for c in changed if "single_seeds_plain" not in c["where"]]
print(f"t1 {len(changed)} flags / words / levels changed:")
for c in changed:
    print("   ", c["where"], ":", c["three_seeds"], "->", c["five_seeds"])

# headline numbers then and now
H = lambda J, n, k: J["summary_test"]["sets"][n][f"l{k}"]
head = {}
for n in NEW + ["lstm_wq", "lstm_w", "lobo_w_static", "lobo_w_nostatic"]:
    for k in LEADS:
        a, b = H(B, n, k), H(R, n, k)
        head[f"{n}/l{k}"] = dict(nse_three=a["nse_median"], nse_five=b["nse_median"], onset_three=a["onset"]["pod"], onset_five=b["onset"]["pod"], hits_three=a["onset"]["hits"],
                                 hits_five=b["onset"]["hits"], fa_three=a["onset"]["false_alarms"], fa_five=b["onset"]["false_alarms"])
for key, v in head.items():
    if v["nse_three"] != v["nse_five"] or v["hits_three"] != v["hits_five"]:
        print(f"    {key:24s} NSE {v['nse_three']:.3f} -> {v['nse_five']:.3f}   onsets {100 * v['onset_three']:.1f}% ({v['hits_three']}, fa {v['fa_three']}) -> {100 * v['onset_five']:.1f}% ({v['hits_five']}, fa {v['fa_five']})")
print("    outcome three seeds:", B["outcome"])
print("    outcome five seeds :", R["outcome"])

# ---- t2 ------------------------------------------------------------------------------------------------------------------------------------------------
cross = []


def rho_walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        if "rho" in a and "p" in a and "p" in b:
            if (a["p"] < P_CUT) != (b["p"] < P_CUT) or abs(a["rho"] - b["rho"]) > 0.15:
                cross.append(dict(where=path, three=dict(rho=a["rho"], p=a["p"]), five=dict(rho=b["rho"], p=b["p"])))
            return
        for k in a:
            if k in b:
                rho_walk(a[k], b[k], f"{path}/{k}")


rho_walk(B["unseen"], R["unseen"], "unseen")
print("t2 rank correlations whose p crossed 0.05 or whose rho moved by more than 0.15:")
for c in cross:
    print("   ", c)

# ---- t3 ---------------------------------------------------------------------------------------------------------------------------------------------
texts = dict(outcome=R["outcome"], outcome_caveat=R["outcome_caveat"], **{f"findings[{i}]": t for i, t in enumerate(R["findings"])}, **{f"notes[{i}]": t for i, t in enumerate(R["notes"])},
             **{f"what_was_not_done[{i}]": t for i, t in enumerate(R["what_was_not_done"])}, **{f"choices[{i}]": t for i, t in enumerate(R["choices_made_after_looking_at_test_scores"])},
             seed_averaging_what=R["unseen"]["seed_averaging"]["what"])
pat = re.compile(r"[^.;(]{0,80}\b(three|five|3|5)(-seed| seeds| trainings| of its| per variant|\b)[^.;)]{0,60}", re.I)
hits = []
for where, t in texts.items():
    for m in re.finditer(r"\b(three|five|[35])[- ](seed|seeds|trainings|training)\b|\bits three seeds\b|\b(three|five) seeds\b", t, re.I):
        a, b = max(0, m.start() - 110), min(len(t), m.end() + 60)
        hits.append(dict(where=where, match=m.group(0), context=t[a:b]))
print(f"t3 {len(hits)} seed-count phrases:")
for h in hits:
    print("   ", h["where"], "|", h["match"], "|", h["context"].replace("\n", " "))
stale = [dict(where=w, text=t[:300]) for w, t in texts.items() if re.search(r"no training was made|by --rescore", t)]
print("    texts that describe this file as written by --rescore / without training:", [s["where"] for s in stale])

# ---- t4 ---------------------------------------------------------------------------------------------------------------------------------------------
C = R["choices_made_after_looking_at_test_scores"]
note = [c for c in C if "Seeds 3 and 4" in c]
I1 = load("r1_integrity")
pt = I1["i3"]["networks"]
vers = sorted({(r["torch_version"], r["numpy_version"]) for n in NEW for s, r in pt[n].items() if s in ("3", "4")})
vers_old = sorted({(r["torch_version"], r["numpy_version"]) for n in NEW for s, r in pt[n].items() if s in ("0", "1", "2")})
py = subprocess.run([sys.executable, "--version"], capture_output=True, text=True).stdout.strip()
sched = (PROJ / "code" / "schedule.py")
sched_txt = sched.read_text(encoding="utf-8") if sched.exists() else ""
sched_lines = [ln.strip() for ln in sched_txt.splitlines() if re.search(r"five|5 seeds|seed", ln, re.I)][:8]
note17 = pathlib.Path(os.environ.get("NOTE17", PROJ / "note17.txt"))
t4 = dict(seed_note_present=len(note) == 1, seed_note=note[0] if note else None, index_in_choices=C.index(note[0]) if note else None, n_choices=len(C),
          three_seed_choices_carried_over=C[:len(B["choices_made_after_looking_at_test_scores"])] == B["choices_made_after_looking_at_test_scores"],
          versions_seeds_3_4=vers, versions_seeds_0_2=vers_old, python_of_the_venv=py, schedule_py_exists=sched.exists(), schedule_seed_lines=sched_lines,
          note17_file=str(note17), note17_equal_to_choice=(note17.read_text(encoding="utf-8").strip() == note[0].strip()) if note17.exists() and note else None,
          claims=dict(python_3_11_15=("3.11.15" in py), torch_2_2_2_cpu=all(v[0] == "2.2.2+cpu" for v in vers), numpy_1_26_4=all(v[1] == "1.26.4" for v in vers),
                      says_three_seed_numbers_in_passes="earlier entries of passes" in (note[0] if note else ""),
                      passes_hold_scores=any(any(k for k in p if "nse" in k or "onset" in k) for p in R["passes"])))
print("t4 SEED_NOTE present:", t4["seed_note_present"], "| carried-over choices equal:", t4["three_seed_choices_carried_over"], "| versions 3-4:", vers, "0-2:", vers_old, "|", py,
      "| schedule.py:", sched.exists(), sched_lines[:3], "| note17.txt equal:", t4["note17_equal_to_choice"], "| passes hold any score:", t4["claims"]["passes_hold_scores"])
save("r4_three_vs_five", dict(changed=changed, headline=head, outcome_three=B["outcome"], outcome_five=R["outcome"], rank_correlations=cross, seed_words=hits, stale_texts=stale, t4=t4))
