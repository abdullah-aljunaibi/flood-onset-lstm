"""Compare the results file and the figure written by `17_more_basins.py --rescore` in a sandbox with the project's own (leaf by leaf). Expected differences only where a rescore records itself:
`passes`, run times, the sentence about passes in `notes`, `integrity.earlier_pass_compared`. Usage: python m14_compare.py SANDBOX_DIR"""
import sys
from m_common import *

SB = pathlib.Path(sys.argv[1])
A = json.loads((PROJ / "results" / "more_basins.json").read_text())
B = json.loads((SB / "results" / "more_basins.json").read_text())
diffs = []


def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append((path + "/" + k, "missing in " + ("project" if k not in a else "sandbox")))
            else:
                walk(a[k], b[k], path + "/" + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"list length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    else:
        if a != b and not (isinstance(a, float) and isinstance(b, float) and abs(a - b) <= 1e-12 * max(1, abs(a))):
            diffs.append((path, f"{str(a)[:90]} -> {str(b)[:90]}"))


walk(A, B)
top = sorted({d[0].split("/")[1].split("[")[0] for d in diffs})
png_equal = sha256(PROJ / "figures" / "more_basins.png") == sha256(SB / "figures" / "more_basins.png")
res = dict(n_differences=len(diffs), top_level_keys_with_differences=top, differences=diffs[:60], figure_byte_identical=png_equal,
           size_project=(PROJ / "results" / "more_basins.json").stat().st_size, size_sandbox=(SB / "results" / "more_basins.json").stat().st_size,
           numbers_equal_outside_the_expected_keys=all(d[0].split("/")[1].split("[")[0] in ("passes", "runtime_s", "rescore_runtime_s", "notes", "integrity", "waited_for_cpu_s", "choices_made_after_looking_at_test_scores") for d in diffs),
           expected_keys_only_are=["passes", "rescore_runtime_s"],
           choices_record_in_project=A["choices_made_after_looking_at_test_scores"], choices_record_after_a_rescore_without_note=B["choices_made_after_looking_at_test_scores"],
           rescore_without_note_drops_the_choices_record=bool(A["choices_made_after_looking_at_test_scores"]) and not B["choices_made_after_looking_at_test_scores"],
           only_passes_run_time_and_choices_differ=sorted({d[0].split("/")[1].split("[")[0] for d in diffs}) == ["choices_made_after_looking_at_test_scores", "passes", "rescore_runtime_s"])
print(f"sandbox --rescore: {len(diffs)} leaf differences from the project's results file, in the top-level keys {top}; figure byte-identical: {png_equal}")
for d in diffs[:25]:
    print("  ", d)
print("every number, note, table and the figure are identical; the differences are the new pass, its run time, and the choices record (dropped when --note is not passed again):", res["only_passes_run_time_and_choices_differ"], "| rescore without --note drops the choices record:", res["rescore_without_note_drops_the_choices_record"])
save("m14_sandbox_rescore", res)
