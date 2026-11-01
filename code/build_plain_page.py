"""The project on one page, in plain words: Project_in_Plain_Words.html and Project_in_Plain_Words.pdf, with figures/plain_ladder.png.

For a reader with no background in machine learning or river science (a family member, a classmate from another department).
Plain words, short sentences, one idea to a sentence; a term that cannot be left out (LSTM, NSE) is explained in the sentence that uses it.

Nothing is computed here. Every number is read by key from the results files, through the shared readers of report_sections/s0_summary.py
(aorc_facts, stage4_facts, forecast_facts, timing_facts, facts) or straight from results/flood_analysis.json, lstm_all.json,
rain_timing.json, aorc_rain_to_midnight.json, stage4_realtime.json, forecast_input.json and data_audit.json; the fixed wordings (the finding and
its supporting line, 'an analysis made after the fact', 'an upper bound that cannot be had in practice', 'no clear difference') are imported
from s0_summary, not typed again; the date under the title is the day of the build; the names of the team and the state of its own checks come from
team.json through code/team.py. The numbers that two files both hold (what today's inputs warn of, the perfect forecast) are compared, and a
difference larger than rounding is reported on stderr. Wording that depends on an outcome is chosen from the stored numbers: if they no longer
support the finding as worded, the page leaves the finding out and says so on stderr.

The ladder (figures/plain_ladder.png) is drawn here, from the same numbers: the share of the flood onsets of the test years that the LSTM warned of
before they came, 1 day ahead with each richer set of inputs (hatched: the perfect weather forecast, an upper bound), and 2 and 3 days ahead
with a public weather forecast against today's inputs. The page must stay ONE page: the build counts the pages of its PDF.
Rebuild after any rerun: python code/build_plain_page.py
"""
import datetime
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import team
from common import BLUE, FIG, GRID, INK, INK2, MUTED, ROOT, use_style
from flood import HIGH_Q, LEADS
from report_common import fig, load_results, page, pct, pdf_pages, print_pdf
from report_sections import s0_summary as S0

UPPER = "an upper bound that cannot be had in practice"                 # the fixed wording for the perfect weather forecast (as in build_slides.py and build_qa.py)
NAME = "Project_in_Plain_Words"
LADDER = "plain_ladder.png"
SAME = 0.006                                                            # two files that hold the same share may differ by this much (rounding of a stored share)
GO_TINT = "#d8f1e7"                                                     # the light fill of the GO colour (report_common.TAG_TINT): the share a warning service would need


def numbers(R, T):
    """Everything the page quotes, each value read by key from its results file (the shares as fractions); None, with a line on stderr, while a file is missing."""
    first, k2, k3 = LEADS[0], LEADS[1], LEADS[2]
    f = S0.facts(R, T)                                                  # page 1 of the report: n of onsets, whether the stored numbers support the finding
    AO, S4, FC, TF = S0.aorc_facts(R), S0.stage4_facts(R), S0.forecast_facts(R), S0.timing_facts(R)
    if not (AO and S4 and FC and TF):
        print("build_plain_page: a results file of the ladder is missing (aorc_rain_to_midnight, stage4_realtime, forecast_input or rain_timing); the page cannot be built", file=sys.stderr)
        return None
    F, A, D = R["flood_analysis"], R["lstm_all"], R["data_audit"]
    lead1 = f"l{first}"
    today = F["verdict"]["services"]["lstm_wq"][lead1]["onset"]["pod"]                  # flood_analysis.json: the onsets the LSTM with the river gauge warns of, at the warning level chosen on the validation years
    perfect = A["decomposition"]["levels"]["wqf"][lead1]["onset_pod"]                   # lstm_all.json: the same LSTM given the perfect weather forecast
    evening = TF["gauge"][first]["onset_warned_with_nldas"]                             # rain_timing.json: rain stamped up to the evening of the issue day
    midnight, false_share = AO["with_"][first], AO["false"][first]                      # aorc_rain_to_midnight.json
    radar = S4["stage4"] / S4["n"]                                                      # stage4_realtime.json: the real-time radar product in place of AORC
    for what, a, b in (("what today's inputs warn of", today, AO["base"][first]), ("what today's inputs warn of", today, FC["base"][first]),
                       ("what today's inputs warn of", today, TF["gauge"][first]["onset_warned_base"]),
                       ("the perfect forecast", perfect, AO["upper"][first]), ("the number of onsets", f["n_on"], AO["n"])):
        if abs(a - b) > (SAME if what != "the number of onsets" else 0):
            print(f"build_plain_page: two results files disagree on {what} ({a:.4f} against {b:.4f}); check results/ before quoting the page", file=sys.stderr)
    cmp_ = A["comparison"]["wq"]["baselines"][A["comparison"]["wq"]["best_baseline_by_validation"][lead1]][lead1]       # the LSTM against the best simple model, with the river gauge
    start, stop = (datetime.date.fromisoformat(d) for d in D["period"])
    test = A["split"]["test"]
    H = S0.hindsight_facts(R)                                                           # review_aorc_rain_to_midnight.json (check d7): what hindsight in AORC could add to the 51%
    if not H:
        print("build_plain_page: the independent review of the AORC rain (hindsight bound) is missing; the page cannot say how much hindsight could add", file=sys.stderr)
    hind = (round(H["bound"]), H["best"]) if H else None
    # the page paraphrases the two fixed wordings of s0_summary in plain words; if a wording changes, say so, so that the paraphrase is read again
    if hind and f"about {hind[0]} points" not in S0.HINDSIGHT_WORDING:
        print(f"build_plain_page: HINDSIGHT_WORDING of s0_summary ('{S0.HINDSIGHT_WORDING}') no longer matches the stored bound ({hind[0]} points); reread the plain sentence on hindsight", file=sys.stderr)
    if not all(w in S0.REALTIME_WORDING for w in ("test years", "real-time radar product", "in place of AORC", "trained on AORC", "not a service")):
        print(f"build_plain_page: REALTIME_WORDING of s0_summary has changed ('{S0.REALTIME_WORDING}'); reread the plain sentence on the real-time check", file=sys.stderr)
    return dict(
        n_on=AO["n"], today=today, evening=evening, midnight=midnight, radar=radar, perfect=perfect, false=false_share,
        fc={k2: FC["fc"][k2], k3: FC["fc"][k3]}, base={k2: FC["base"][k2], k3: FC["base"][k3]},
        lstm_nse=cmp_["lstm_nse_median"], simple_nse=cmp_["baseline_nse_median"], story=bool(f["story"]), tie=bool(f["gauge_tie"]),
        n_years=round(((stop - start).days + 1) / 365.25), n_rivers=len(D["selected"]), test=(test[0][:4], test[1][:4]),
        one_in=round(1 / (1 - HIGH_Q)), go=S0.criteria(R), hind=hind, k2=k2, k3=k3)


def draw_ladder(N, path):
    """figures/plain_ladder.png: one bar per set of inputs, the share of the flood onsets warned of (hatched: the upper bound); a light band marks what a service would need."""
    use_style()
    plt.rcParams.update({"axes.grid": False, "font.size": 9.5, "hatch.linewidth": 0.9})
    k2, k3 = N["k2"], N["k3"]
    word = S0.NUMBER
    rows = [dict(kind="head", text=f"{word[LEADS[0]].capitalize()} day ahead"),
            dict(kind="bar", label="Today\u2019s inputs", sub="past weather and the river\u2019s own level", v=N["today"]),
            dict(kind="bar", label="+ rain known up to the evening", sub="of the day the forecast is made", v=N["evening"]),
            dict(kind="bar", label="+ rain known up to midnight", sub=f"from {S0.AORC_WORDING}", v=N["midnight"], note=f"({pct(N['radar'])} in a real-time radar check)"),
            dict(kind="bar", label="A perfect weather forecast", sub=UPPER, v=N["perfect"], hatch=True),
            dict(kind="gap"),
            dict(kind="head", text=f"{word[k2].capitalize()} and {word[k3]} days ahead, with a public weather forecast"),
            dict(kind="bar", label=f"{k2} days ahead", sub="", v=N["fc"][k2], note=f"({pct(N['base'][k2])} with today\u2019s inputs)"),
            dict(kind="bar", label=f"{k3} days ahead", sub="", v=N["fc"][k3], note=f"({pct(N['base'][k3])} with today\u2019s inputs)")]
    H = dict(head=0.235, bar=0.325, gap=0.06)                                           # inches
    AXIS_H, TOP, W = 0.27, 0.02, 7.24
    total = TOP + sum(H[r["kind"]] for r in rows) + AXIS_H
    left, right = 0.375, 0.962
    f = plt.figure(figsize=(W, total))
    y_end = total - AXIS_H - TOP
    ax = f.add_axes([left, AXIS_H / total, right - left, y_end / total])
    ax.set_xlim(0, 100)
    ax.set_ylim(y_end, 0)                                                             # inches from the top; the axis runs downwards
    go = N["go"]["go_pod"] * 100
    ax.add_patch(Rectangle((go, 0), 100 - go, y_end, facecolor=GO_TINT, edgecolor="none", zorder=0.5))
    ax.text(go + 1.2, 0.04, f"a warning service\nwould need {go:.0f}% or more", fontsize=7.8, color=INK2, va="top", ha="left", linespacing=1.15, zorder=3,
            bbox=dict(facecolor=GO_TINT, edgecolor="none", pad=1.6))
    for x in range(0, 101, 20):
        ax.axvline(x, color=GRID, lw=0.8, zorder=0.6)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)], fontsize=8.4, color=MUTED)
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    label_at = ax.get_yaxis_transform()                                               # x in axes fractions, y in data (inches from the top)
    y = TOP
    for r in rows:
        h = H[r["kind"]]
        mid = y + h / 2
        if r["kind"] == "head":
            f.text(0.006, 1 - (y + h * 0.62) / total, r["text"], fontsize=9.8, fontweight="bold", color=INK, va="center", ha="left")
            if r is rows[0]:                                                          # what the bars measure, over the plot
                ax.text(0, y + h * 0.62, f"share of the {N['n_on']:,} flood onsets warned of", fontsize=8.4, color=INK2, va="center", ha="left")
        elif r["kind"] == "bar":
            two = bool(r["sub"])
            ax.text(-0.012, mid - (0.075 if two else 0), r["label"], transform=label_at, fontsize=9.6, color=INK, ha="right", va="center", clip_on=False)
            if two:
                ax.text(-0.012, mid + 0.095, r["sub"], transform=label_at, fontsize=7.9, color=INK2, ha="right", va="center", clip_on=False)
            v = r["v"] * 100
            if r.get("hatch"):
                ax.barh(mid, v, height=0.20, facecolor="#cde2fb", edgecolor=BLUE, hatch="////", lw=0, zorder=2)
            else:
                ax.barh(mid, v, height=0.20, color=BLUE, zorder=2)
            value = ax.text(v + 1.2, mid, f"{v:.0f}%", fontsize=10.6, fontweight="bold", color=INK, va="center", ha="left", zorder=4)
            if r.get("note"):                                                         # the note starts where the value ends
                ax.annotate(r["note"], xy=(1, 0.5), xycoords=value, xytext=(4, 0), textcoords="offset points", fontsize=8.2, color=INK2, va="center", ha="left", zorder=4)
        y += h
    f.savefig(path, dpi=220)
    plt.close(f)


def facts_html(N, R):
    """The text of the page, every number from N."""
    team_names = team.names_line()
    n_members = len(team.load()["members"])
    best = "none" if not N["hind"] or not N["hind"][1] else f"{N['hind'][1]:g} points"
    hindsight = (f" The {pct(N['midnight'])} could be flattered by hindsight, since it comes from {S0.AORC_WORDING}; an independent check found this could add at most "
                 f"about {N['hind'][0]} points to it (best estimate: {best}).") if N["hind"] else ""                                  # beside the 51%: S0.HINDSIGHT_WORDING in plain words
    realtime = (f" The {pct(N['radar'])} is a check on {N['test'][0]} to {N['test'][1]}, not a working service: the model, which had learned from that analysis, "
                f"was given the real-time radar product in its place.")                                                          # beside the 48%: S0.REALTIME_WORDING in plain words
    go = N["go"]
    finding = ""
    if N["story"]:
        tie = (f" For example, with today’s inputs the memory model scored {N['lstm_nse']:.2f} and the best simple model {N['simple_nse']:.2f} "
               f"(NSE, a score where 1 is perfect): {S0.reading_word('no clear difference')}.") if N["tie"] else ""
        finding = (f"<p class=\"find\"><b>{S0.FINDING}.</b> {S0.SUPPORT}. The inputs are what we give the model to read; the network is the model itself.{tie}</p>")
    else:
        print("build_plain_page: the stored numbers do not support the finding as worded; the page leaves it out (see page 1 of Flood_Report.pdf)", file=sys.stderr)
    body = f"""
<h1>Predicting river floods with AI, in plain words</h1>
<p class="sub">One page for readers who know neither machine learning nor river science &middot; AI course group project &middot; {datetime.date.today().day} {datetime.date.today():%B %Y}</p>

<div class="verdict"><p><b>The question.</b> Given the weather of the past year and the river&rsquo;s own level, can a computer model say how high the river will be tomorrow, and on the two days after? And can it warn us before the river floods?</p></div>

<div class="ub"><div>
<h2>What we used</h2>
<p>We used {N['n_years']} years of daily records. They hold the rain, the weather and the flow of each river. They cover {N['n_rivers']} rivers in the south-eastern United States. All of the data are public.</p>
</div><div>
<h2>What we built</h2>
<ul>
<li><b>A model with a memory.</b> It reads the days one after another and learns the link between the rain and the river. It is called an LSTM, short for long short-term memory.</li>
<li><b>Simple models beside it, for comparison.</b> The simplest one says: tomorrow will be like today.</li>
<li><b>A fair test.</b> The models learned from the early years. We then scored them on {N['test'][0]} to {N['test'][1]}, years they had never seen. The models' settings were chosen without looking at those years.</li>
</ul>
</div></div>

<h2>What we found</h2>
<p>Each forecast is made at midnight. We counted the flood onsets that the model warned of before they came. A flood onset is a day when the river is above its high level but was still below it when the forecast was made. In the early years a river passes that level on only 1 day in {N['one_in']}. Whether such a day floods anything, the data do not say.</p>
{fig(LADDER, "Top: each row gives the model more information and warns of more onsets. A perfect weather forecast is the real weather of those days, given in advance. The bottom two bars also include the rain known up to midnight." + hindsight + realtime)}
{finding}

<h2>What it means</h2>
<p>The warnings are not yet good enough for a real service. We fixed the bar in advance: at least {pct(go['go_pod'])} of the onsets warned, and at most {pct(go['go_far'])} of the warnings false. With the rain known up to midnight, {pct(N['midnight'])} of the onsets are warned ({pct(N['radar'])} in a real-time radar check, not a working service), but {S0.share_word(N['false'])} of the warnings are false alarms. We know what a service needs: radar rain up to the moment of the forecast, and a weather forecast for the rain still to come.</p>

<h2>Who did it, and where to read more</h2>
<p>The team has {S0.NUMBER.get(n_members, n_members)} members: {team_names} (RS/GIS: satellite images and maps; GNSS: satellite navigation). Claude Code, an AI coding assistant, wrote all the code and ran all the experiments, under the team lead&rsquo;s direction. Independent AI reviewers checked the code and the numbers. {S0._esc(team.checks_sentence(short=True))}. To read more, open page 1 of <code>Flood_Report.pdf</code>, then <code>Project_Guide.pdf</code>.</p>
<p class="foot">Every number on this page is read from the project&rsquo;s results files when the page is built (<code>code/build_plain_page.py</code>).</p>
"""
    return body


CSS = """
@page{margin:9mm 13mm}
body{font-size:9.9pt;line-height:1.38}
h1{font-size:18.5pt;margin:0 0 1pt} .sub{font-size:8.8pt;margin:0 0 6pt}
h2{font-size:11.4pt;margin:6pt 0 1.5pt;color:var(--blue)}
.verdict{font-size:10.2pt;padding:4pt 11pt;margin:3pt 0 2pt} .verdict p{margin:0}
p{margin:0 0 3pt} ul{margin:0 0 2pt;padding-left:14pt} li{margin-bottom:1.5pt}
figure{margin:3pt 0 3pt} figcaption{font-size:8.2pt;line-height:1.3}
.find{font-size:10pt;margin:2pt 0 0}
.ub{display:grid;grid-template-columns:0.7fr 1.5fr;column-gap:16pt;align-items:start} .ub h2{margin-top:7pt}
.foot{font-size:7.8pt;color:var(--muted);margin:3pt 0 0}
"""


def main():
    R, T = load_results()
    N = numbers(R, T)
    if N is None:
        sys.exit(1)
    draw_ladder(N, FIG / LADDER)
    html = page("Predicting river floods with AI, in plain words", facts_html(N, R), CSS)
    out = ROOT / f"{NAME}.html"
    out.write_text(html, encoding="utf-8")
    pdf = out.with_suffix(".pdf")
    wrote = print_pdf(out, pdf)
    n_pages = pdf_pages(pdf) if wrote else None
    if wrote and n_pages != 1:
        print(f"build_plain_page: {pdf.name} has {n_pages} pages, not one - shorten the page", file=sys.stderr)
    print("pages:", n_pages)
    print("wrote", out, "and", pdf if wrote else "(no PDF: no browser)", "and", FIG / LADDER)


if __name__ == "__main__":
    main()
