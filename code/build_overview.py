"""One-page project overview (PDF): what the project is, the data, the method, the learning outcomes.

Reads results/data_audit.json (numbers) and the CAMELS files (two figures). The block 'What we found' is built from the same
results and the same sentences as page 1 of the report (report_sections/s0_summary.py); while those results do not exist yet,
the page shows the progress strip instead. Since 8 October 2026 the block also gives the rungs of the experiments of 7 and 8 October
(the rain up to midnight, the real-time check, a real weather forecast) as a row of numbers, read through the shared readers of s0_summary
(aorc_facts, stage4_facts, era5_facts, forecast_facts) and worded with its fixed wordings: the hindsight bound stands beside the 51%, the Stage IV figure is
'the real-time radar product in its place' and, in the caveats, REALTIME_WORDING (a check, not a service); the caveats stand in one paragraph (AORC is an analysis made
after the fact; the verdict for onsets is still NOT YET) and beside it page 1's 'who did what' (S0.who, short form: Claude Code wrote and ran everything under
The team lead's direction, the independent AI reviews and the final critic agent, the team's own checks still open). The page must stay one page: the build counts the
pages of its PDF.
"""
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from camels import load_basin
from common import *

use_style()
UPPER = "an upper bound that cannot be had in practice"                            # the fixed wording for the perfect weather forecast (as in build_slides.py and build_qa.py)
R = json.loads((RES / "data_audit.json").read_text())
T = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str})
sel = pd.DataFrame(R["selected"])
COL = {"Mountains (above 600 m)": BLUE, "Northern lowlands and Piedmont": AQUA, "Southern lowlands (south of 32.5N)": ORANGE}
SHORT = {"Mountains (above 600 m)": "Mountains", "Northern lowlands and Piedmont": "Northern lowlands", "Southern lowlands (south of 32.5N)": "Southern lowlands"}

# ---- figure 1: where the 12 basins are ---------------------------------------------------------------------------
states = json.loads((DATA / "basemap" / "us-states.json").read_text())["features"]
fig, ax = plt.subplots(figsize=(3.7, 2.95))
for f in states:
    g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for p in polys:
        xy = np.array(p[0]); ax.fill(xy[:, 0], xy[:, 1], facecolor="#f3f2ee", edgecolor=AXIS, linewidth=.7, zorder=1)
for land, grp in sel.groupby("landscape"):
    ax.scatter(grp.gauge_lon, grp.gauge_lat, s=46, color=COL[land], edgecolors=SURFACE, linewidths=1.1, zorder=3, label=f"{SHORT[land]} ({len(grp)})")
for name, (x, y) in {"North Carolina": (-77.6, 35.75), "Georgia": (-83.4, 32.9), "Florida": (-81.6, 28.6), "Alabama": (-86.8, 32.9), "Mississippi": (-89.7, 32.9),
                     "Tennessee": (-86.3, 35.85), "South Carolina": (-80.9, 33.9), "Virginia": (-78.6, 37.5)}.items():
    ax.text(x, y, name, fontsize=6.3, color=MUTED, ha="center", va="center", zorder=2)
ax.set_xlim(-91.8, -75.2); ax.set_ylim(25.6, 38.6); ax.set_aspect(1.2); ax.axis("off")
ax.legend(frameon=False, fontsize=7.4, loc="lower left", handletextpad=.3, borderaxespad=0)
ax.set_title("The 12 study basins", fontsize=9.5)
fig.tight_layout(pad=.3); fig.savefig(FIG / "overview_map.png", dpi=220); plt.close(fig)

# ---- figure 2: the problem in one picture (rain in, river flow out) ---------------------------------------------------
G = "03439000"
full = load_basin(G).loc[R["period"][0]:R["period"][1]]
p95 = float(full.q_mm.loc[:"2000-09-30"].quantile(.95)); peak_day = full.q_mm.idxmax()      # the high-flow level of the training years, as in flood.thresholds
d = full.loc[peak_day - pd.Timedelta(days=24):peak_day + pd.Timedelta(days=30)]
fig, (a, b) = plt.subplots(2, 1, figsize=(3.9, 2.95), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.1], hspace=.12))
a.bar(d.index, d.prcp, width=.85, color=MUTED, zorder=2); a.invert_yaxis(); a.set_ylabel("rain (mm/day)", fontsize=7); a.tick_params(labelsize=7)
a.set_title("Rain in, river flow out", fontsize=9.5)
b.plot(d.index, d.q_mm, color=BLUE, zorder=3); b.fill_between(d.index, d.q_mm, color=BLUE, alpha=.12, zorder=2)
b.axhline(p95, color=ORANGE, lw=1.2, zorder=2); b.text(d.index[0], p95 * 1.35, "high-flow level", fontsize=6.8, color=INK2, va="bottom")
b.set_ylabel("river flow (mm/day)", fontsize=7); b.tick_params(labelsize=7)
b.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2)); b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
b.set_xlabel(f"{peak_day.year}", fontsize=7.5)
fig.align_ylabels(); fig.subplots_adjust(left=.15, right=.98, top=.92, bottom=.14); fig.savefig(FIG / "overview_hydrograph.png", dpi=220); plt.close(fig)
second = d.q_mm[d.index > peak_day + pd.Timedelta(days=4)]
n_high_days = int((full.q_mm > p95).sum())

# ---- page ----------------------------------------------------------------------------------------------------------------
img = lambda n: f'<img src="data:image/png;base64,{base64.b64encode((FIG / n).read_bytes()).decode()}">'
FONT = "font-family:'Segoe UI','Helvetica Neue',system-ui,sans-serif"


def box(x, w, top, sub, core=False):
    s = f'<rect x="{x}" y="4" width="{w}" height="62" rx="8" fill="{SURFACE}" stroke="{BLUE if core else AXIS}" stroke-width="{2.2 if core else 1.3}"/>'
    s += f'<text x="{x + w / 2}" y="31" text-anchor="middle" style="{FONT};font-size:13px;font-weight:600" fill="{INK}">{top}</text>'
    return s + f'<text x="{x + w / 2}" y="49" text-anchor="middle" style="{FONT};font-size:10.5px" fill="{MUTED}">{sub}</text>'


def arrow(x1, x2):
    return f'<line x1="{x1}" y1="35" x2="{x2 - 5}" y2="35" stroke="{INK2}" stroke-width="1.5"/><path d="M{x2 - 9},30 L{x2},35 L{x2 - 9},40" fill="none" stroke="{INK2}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>'


steps = [("Past weather", "365 days of rain, temperature", False), ("Basin properties", "area, slope, soil, land cover", False),
         ("LSTM model", "a network with memory", True), ("River flow", "next 1 to 3 days", False), ("Flood warning", "flow above the high level", False)]
w, gap = 156, 20
svg = f'<svg viewBox="0 0 {5 * w + 4 * gap} 70" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="method diagram">'
for i, (t, s, c) in enumerate(steps):
    x = i * (w + gap); svg += box(x, w, t, s, c)
    if i < 4: svg += arrow(x + w + 3, x + w + gap - 3)
svg += "</svg>"

done = lambda name: (RES / name).exists()
stages = [("Data downloaded", True), ("Data quality check", True), ("Simple baselines", done("baselines.json")), ("LSTM model", done("lstm_all.json")),
          ("Flood analysis", done("flood_analysis.json")), ("Report", (ROOT / "Flood_Report.pdf").exists())]
if done("lstm_all.json"):
    L = json.loads((RES / "lstm_all.json").read_text())["sets"]["wq"]["test"]["summary"]
    status = (f"LSTM with river gauge, test years: median NSE {L['l1']['nse_median']:.2f} / {L['l2']['nse_median']:.2f} / {L['l3']['nse_median']:.2f} "
              f"at 1 / 2 / 3 days ahead. Results: Flood_Report.pdf.")
else:
    status = "No model has been trained yet, so this page reports no accuracy figures."
strip = "".join(f'<span class="st {"done" if d_ else ""}">{"&#10003; " if d_ else ""}{n}</span>' for n, d_ in stages)
found = f'<h2>Progress</h2>\n<div>{strip}</div>\n<p class="cap">{status}</p>'      # until the experiments have run
try:                                                                               # what was found: the finding, the key numbers, milestone 2 (as on page 1 of the report)
    import sys
    from report_common import load_results, pct
    from report_sections import s0_summary as S0
    R_, T_ = load_results()
    f0 = S0.facts(R_, T_)
    if f0.get("story"):
        m_, what_, status_ = f0["miles"][1]
        first_, rest_ = S0.LEADS[0], S0.LEADS[1:]
        E1, AO, S4, FC, HS = S0.era5_facts(R_), S0.aorc_facts(R_), S0.stage4_facts(R_), S0.forecast_facts(R_), S0.hindsight_facts(R_)
        hind = S0.hindsight_clause(HS)                                              # the bound of the independent review, beside the 51% (HINDSIGHT_WORDING)
        who_ = re.sub(r"\s*\(results/[^)]*\)", "", S0.who(R_, short=True)).replace("(section 9)", "(report, section 9)").replace("for sections 4.6 to 4.8", "for sections 4.6 to 4.8 of the report")     # page 1's text of who did what, without the file names
        ladder = ""
        if E1 and AO and S4 and FC:                                                # the rungs of the experiments of 7 and 8 October 2026, every number read by key
            chip = lambda v, what: f'<div><b>{pct(v)}</b>{what}</div>'
            verdicts = (set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values()))
            verdict = next(iter(verdicts)) if len(verdicts) == 1 else " / ".join(sorted(verdicts))
            far_ = " / ".join(f"{pct(FC['fc'][k])}" for k in rest_)
            ladder = (f'<p class="cap lead">Onsets warned, of the {AO["n"]} onsets of the test years ({S0.LEAD_WORD[first_]} unless said):</p>'
                      f'<div class="lad">{chip(AO["base"][first_], "river gauge")}{chip(E1["nldas"], "+ rain up to the evening")}'
                      f'{chip(AO["with_"][first_], "+ rain up to midnight, AORC*" + (f" ({hind})" if hind else ""))}'
                      f'{chip(S4["stage4"] / S4["n"], "with the real-time radar product in its place")}'
                      f'{chip(AO["upper"][first_], "perfect forecast, an upper bound")}'
                      f'<div class="far"><b>{far_}</b>{S0._leads(rest_)}, with a forecast (rain table alone: {" / ".join(pct(FC["base"][k]) for k in rest_)})</div></div>'
                      f'<div class="fine"><p class="cap">*AORC is {S0.AORC_WORDING}; the {pct(S4["stage4"] / S4["n"])} is {S0.REALTIME_WORDING}; the perfect forecast is {UPPER}; '
                      f'for onsets the verdict is {"still " if verdicts == {"NOT YET"} else ""}{verdict}. '
                      f'Milestone 2 of the plan: {status_.split(";")[0]}. Numbers and key figure: Flood_Report.pdf, page 1.</p>'
                      f'<p class="cap"><b>Who did what:</b> {who_}</p></div>')
        found = (f'<h2>What we found</h2>\n<p><b>{S0.FINDING}.</b> {S0.SUPPORT}.</p>\n{ladder}' if ladder else
                 f'<h2>What we found</h2>\n<p><b>{S0.FINDING}.</b> {S0.SUPPORT}.</p>\n'      # one page: the numbers and the key figure are on page 1 of the report
                 + f'<p class="cap">Milestone 2 of the plan: {status_.split(";")[0]}. Numbers and key figure: Flood_Report.pdf, page 1.</p>')
except Exception as e:                                                             # an overview must build even when a later result is missing
    print(f"build_overview: the block 'What we found' was left out ({type(e).__name__}: {e}); the progress strip is shown", file=sys.stderr)
years = int(R["period"][1][:4]) - int(R["period"][0][:4])
lo, hi = sel.area_gages2.min(), sel.area_gages2.max()

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Flood Prediction Project Overview</title><style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--blue:#2a78d6}}
@page{{size:A4;margin:7.5mm 13mm}} body{{background:var(--surface);color:var(--ink);font:8.9pt/1.36 'Segoe UI','Helvetica Neue',system-ui,sans-serif;margin:0}}
h1{{font-size:17pt;margin:0 0 1pt}} h2{{font-size:10.6pt;margin:5pt 0 2pt}} p{{margin:0 0 4pt}}
.sub{{color:var(--ink2);margin:0 0 4pt}} .q{{border:2px solid var(--blue);border-radius:9px;padding:6pt 10pt;font-size:10.4pt;margin:0 0 4pt}}
.two{{display:grid;grid-template-columns:1fr 1.05fr;gap:10pt;align-items:start}} figure{{margin:0}} figure img{{width:53%;display:block;margin:0 auto}}
figcaption,.cap{{font-size:7.8pt;color:var(--ink2);margin-top:1pt}}
table{{border-collapse:collapse;width:100%;font-size:8.5pt}} th,td{{text-align:left;vertical-align:top;padding:1.6pt 6pt;border-top:1px solid var(--grid)}} th{{color:var(--ink2);font-weight:600;width:17%}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12pt}} ul{{margin:0;padding-left:13pt}} li{{margin-bottom:2.2pt;font-size:8.6pt}}
.st{{display:inline-block;border:1px solid var(--grid);border-radius:20px;padding:2pt 8pt;margin:0 4pt 3pt 0;font-size:8.3pt;color:var(--ink2)}}
.st.done{{border-color:var(--blue);color:var(--ink);font-weight:600}}
.found{{border:2px solid var(--blue);border-radius:9px;padding:1pt 10pt 2pt;margin:5pt 0 0}} .found h2{{margin:5pt 0 2pt}} .found p{{margin:0 0 3pt}}
.lad{{display:grid;grid-template-columns:.78fr .88fr 1.5fr 1.3fr 1fr 1.75fr;gap:4pt;margin:2pt 0 3pt}} .lad div{{border:1px solid var(--grid);border-radius:6px;padding:2pt 5pt 3pt;font-size:7.6pt;line-height:1.2;color:var(--ink2)}}
.lad b{{display:block;font-size:12.5pt;line-height:1.15;color:var(--ink)}} .lad .far{{border-color:var(--blue)}} .cap.lead{{margin:2pt 0 0}}
.fine{{display:grid;grid-template-columns:.85fr 1.15fr;gap:10pt;margin-top:1pt}} .fine .cap{{margin:0}}
</style></head><body>
<h1>Predicting river floods with AI</h1>
<p class="sub">AI course group project &middot; overview &middot; {pd.Timestamp.today().day} {pd.Timestamp.today():%B %Y}</p>
<div class="q"><b>The question:</b> given the weather of the past year, can an AI model predict a river&rsquo;s flow one to three days ahead, and warn of high-flow days before they happen?</div>

<div class="two">
<figure>{img('overview_map.png')}<figcaption>Twelve river basins in the south-eastern United States, four in each of three landscapes. They were chosen by rule from {R['n_candidates']} humid, rain-fed candidates; {R['n_with_gaps']} candidates were rejected because their flow records have gaps.</figcaption></figure>
<figure>{img('overview_hydrograph.png')}<figcaption>French Broad River at Rosman, North Carolina, {peak_day:%B %Y}. Two storms sent the river far above its high-flow level ({p95:.1f} mm/day, exceeded on only 5% of days in the training years), peaking at {full.q_mm.max():.0f} and {second.max():.0f} mm/day. The river reacts within a day and takes weeks to settle, so the model needs a memory of past rain.</figcaption></figure>
</div>

<h2>How it works</h2>
{svg}

<div class="cols"><div>
<h2>Data (all open)</h2>
<table>
<tr><th>Source</th><td>CAMELS-US, a public dataset of {R['n_all_camels']} US river basins</td></tr>
<tr><th>Period</th><td>{years} years of daily records, {R['period'][0][:4]}&ndash;{R['period'][1][:4]}</td></tr>
<tr><th>Target</th><td>Daily river flow measured by the US Geological Survey</td></tr>
<tr><th>Inputs</th><td>Daily rain, temperature, sunlight and humidity, plus basin properties</td></tr>
<tr><th>Basins</th><td>{len(sel)} basins, {lo:.0f} to {hi:,.0f} km&sup2;, with no missing days</td></tr>
</table>
<h2>How it was judged</h2>
<table>
<tr><th>Fair test</th><td>Train on early years, test on later years the model has never seen</td></tr>
<tr><th>Baselines</th><td>&ldquo;Tomorrow equals today&rdquo;, a ridge regression and a random forest: the numbers the LSTM had to beat</td></tr>
<tr><th>Scores</th><td>Overall fit (Nash&ndash;Sutcliffe efficiency), plus hits, misses and false alarms on high-flow days</td></tr>
</table>
</div><div>
<h2>Learning outcomes</h2>
<ul>
<li><b>Check the data before modelling.</b> Audit records for gaps and estimated values, and choose study basins by rule (Session 4).</li>
<li><b>Match the model to the data.</b> Explain why a model with memory suits a time series, and when it does not help (Session 3).</li>
<li><b>Evaluate honestly.</b> Use simple baselines, test on unseen years, and report misses and false alarms as well as hits.</li>
<li><b>Apply remote sensing and GIS.</b> Map the basins, and test whether satellite rainfall and soil moisture improve the prediction.</li>
<li><b>Judge when a model is good enough to use.</b> Decide whether a flood warning from it could be trusted, and where it fails.</li>
<li><b>Work reproducibly.</b> Code produces the results, and the report is built from those results.</li>
</ul>
</div></div>
<div class="found">{found}</div>
</body></html>"""

out = ROOT / "Project_Overview.html"
out.write_text(html, encoding="utf-8")
from report_common import find_browser, pdf_pages
browser = find_browser()
if browser is None:                                                                # no browser on this machine: the HTML stands alone
    print("build_overview: no Chromium browser found - HTML only:", out)
    sys.exit(0)
subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'Project_Overview.pdf'}", out.as_uri()], check=True, timeout=120, capture_output=True)
n_pages = pdf_pages(ROOT / "Project_Overview.pdf")
if n_pages != 1:                                                                   # the overview is one page: say so on stderr when it is not
    print(f"build_overview: Project_Overview.pdf has {n_pages} pages, not one - shorten the page", file=sys.stderr)
print("pages:", n_pages)
print("wrote", ROOT / "Project_Overview.pdf", "| peak", peak_day.date(), round(float(full.q_mm.max()), 1), "| second", round(float(second.max()), 1), second.idxmax().date(), "| p95", round(p95, 2), "| high days", n_high_days)
