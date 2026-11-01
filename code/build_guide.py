"""Assemble Project_Guide.html and Project_Guide.pdf: the project explained step by step, for readers who are new to machine learning.

The guide follows the order in which the project was done (guide_common.STEPS). Each file in code/guide_sections/ writes some of
the steps; every machine-learning idea that the project needed is explained in a box where it is first used, with ONE definition for
all parts (guide_common.WORDS). Nothing is recomputed: every number is read from results/*.json and every picture comes from figures/.
Rebuild with: python code/build_guide.py   (python code/build_guide.py g2_model  previews one part in results/quick/).
"""
import datetime
import importlib
import re
import sys

import guide_common as G
import team
from common import RES, ROOT
from report_common import load_results, page, pdf_pages, print_pdf

SECTIONS = ["g1_start", "g2_model", "g3_findings", "g4_concepts"]
names = sys.argv[1:] or SECTIONS
full = not sys.argv[1:]

R, T = load_results()
parts = []
for name in names:
    try:
        parts.append(importlib.import_module(f"guide_sections.{name}").html(R, T))
    except ModuleNotFoundError as e:
        if e.name != f"guide_sections.{name}":
            raise
        print("part not written yet:", name, file=sys.stderr)

today = datetime.date.today()
shown = [int(n) for n in re.findall(r'id="step(\d+)"', "\n".join(parts))]                  # the steps that the parts wrote (an optional step may be missing)
contents = "".join(f'<li><span class="n">{n}</span>{title}</li>' for n, title, _ in G.STEPS if not full or n in shown or n not in G.OPTIONAL_STEPS)
found = "what was found, and what was done next" if 16 in shown else "and what was found"       # the last step is optional (it needs the results of 7 and 8 October 2026)
try:                                        # the group and the members with their roles, as the final report prints them
    import json
    _cfg = json.loads((ROOT / "code" / "final_report" / "config.json").read_text(encoding="utf-8"))
    team_line = (f"Group #{_cfg['group_number']} &middot; team: " if _cfg.get("group_number") else "team: ") + ", ".join(
        f"{m['name']} ({m['role']})" for m in _cfg["members"])
except (OSError, ValueError, KeyError):
    team_line = f"team: {team.names_line()}"
cover = f"""
<h1>How the project was done, step by step</h1>
<p class="sub">Predicting river floods with AI &middot; a guide for readers who are new to machine learning &middot; built on {today.day} {today:%B %Y}
&middot; {team_line} &middot; coding assistant: Claude Code</p>
<div class="verdict"><p><b>What this guide is.</b> It walks through the project in the order in which it was done: the question, the data, the test,
the simple models, the neural network, {found}. It assumes no knowledge of machine learning. The machine-learning ideas that the project
needed are each explained in a blue box at the place where they are first used, and the last pages list all {len(G.WORDS)} of them together.</p>
<p style="margin:0"><b>How to read it.</b> Each step has a short text, its pictures and a line &lsquo;What to remember&rsquo;. The numbers are those of
the final report (<code>3_Final_Project_Report_Flood.pdf</code>, hand-in item 3): they are read from the same results files whenever the guide is built.
Each step names the section of the final report that holds the full detail; the guide covers the work up to {today.day} {today:%B %Y}.</p></div>
<ol class="toc">{contents}</ol>
""" if full else ""

out = ROOT / "Project_Guide.html" if full else RES / "quick" / f"preview_guide_{'-'.join(names)}.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page("How the project was done, step by step", cover + "\n".join(parts), css_extra=G.GUIDE_CSS), encoding="utf-8")
pdf = out.with_suffix(".pdf")
n_pages = pdf_pages(pdf) if print_pdf(out, pdf) else None
print("wrote", out, f"({len(parts)} of {len(names)} parts)")
if n_pages is not None:
    print("wrote", pdf, f"({n_pages} pages)")
if full:                                    # every step once, in order; and say which concepts no part has shown in a box
    text = "\n".join(parts)
    found = [int(n) for n in re.findall(r'id="step(\d+)"', text)]
    expected = [n for n, _, _ in G.STEPS if n not in G.OPTIONAL_STEPS or n in found]
    if found != expected:
        print("build_guide: the steps in the text are", found, "- expected", expected, file=sys.stderr)
    for n in G.OPTIONAL_STEPS:
        if n not in found:
            print(f"build_guide: step {n} ({G.TITLE[n]}) is left out: the results it stands on are missing", file=sys.stderr)
    unused = [G.WORDS[k][0] for k in G.WORDS if k not in G.USED]
    if unused:
        print("build_guide: no part shows a box for:", ", ".join(unused), "(they appear only in the list at the end)", file=sys.stderr)
