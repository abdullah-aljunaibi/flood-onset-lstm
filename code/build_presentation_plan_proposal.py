"""The presentation plan for the six-slide PROPOSAL talk (mid-term, 10 minutes): proposal/Presentation_Plan_Proposal.pdf and .md (at most 3 pages).

Sibling of build_presentation_plan_short.py: the style sheet, helpers and the HTML and Markdown renderers of build_presentation_plan.py are imported as they are.
Seconds, presenters, slide titles and the hand-over sentences are read from the deck (build_proposal_slides.deck() and NOTES); nothing of them is retyped.
Written here: the short description of what is on each slide, what each member explains, the likely questions with short answers (Meeting_Discussion_Points section 6
and the proposal), the files to open, the cut for a late talk (by sentence of NOTES; the seconds of the cut are computed) and the rehearsal.
No shares and no percentages per member: only the seconds of each part. Nothing existing is changed.
Run from the project root:  ~/AI_Class/.venv/bin/python code/build_presentation_plan_proposal.py   (after build_proposal_slides.py)
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_presentation_plan as BP
import build_proposal_slides as DK
import team
from common import ROOT
from report_common import page, pdf_pages, print_pdf

OUT = ROOT / "proposal"
NAME = "Presentation_Plan_Proposal"
MAX_PAGES = 3
KEYS = ("A", "R2", "G1", "G2")
ON_SLIDE = {  # what is on each slide, in words (the title comes from the deck)
    1: "course cover (title, four participants, blanks for instructor, date, session) and the overall aim in a box",
    2: "three bullets of background, the gap in an orange box, the three research questions RQ1 to RQ3",
    3: "map of the 12 basins (<code>figures/basin_maps.png</code>) with its caption; table of nine data products with what each gives, resolution and period; the line that no optical or SAR scene is processed",
    4: "the data-flow figure (eleven products into one pipeline, five boxes: downloads, audit, caches, daily inputs, split) and five lines: acquire, features, models, split, scoring",
    5: "five deliverables D1 to D5; the table indicator, baseline, target; the orange box with the preliminary numbers and the caveat",
    6: "three-week timeline table (15 Oct to 5 Nov 2026); the four roles; the AI-assistance disclosure; the three questions to the professor",
}
# the sentences of NOTES (1-based) that each part drops when the talk is cut to 7 minutes; the hand-over is always kept
DROP = {1: [1, 2], 2: [2, 3], 3: [3, 4, 9, 10], 4: [2, 4, 6], 5: [1, 2, 3, 4, 5, 6, 10], 6: [1, 2]}
LATE_TOTAL = 420

# per member: what to explain (3-4 items), questions with short answers, files to open. Facts from the proposal and Meeting_Discussion_Points section 6.
MEMBER = {
    "A": dict(
        explain=["<b>The gap</b> (proposal section 3): no public study on small south-eastern basins separates the clock of the rain record from the forecast; the Daymet day ends about 09:00, not at the midnight when a forecast is issued.",
                 "<b>The three research questions</b> (section 4): RQ1 LSTM against the best simple model, a gain clear only if every training is above every other and most basins agree; RQ2 which information closes the gap; RQ3 can the warnings be trusted (GO: at least 70% warned, at most 30% false).",
                 "<b>The disclosure</b> (section 6.5) in the exact words of slide 6, and the three-week plan to 5 November with one owner per part.",
                 "<b>The three questions to the professor</b> (Meeting_Discussion_Points section 4): US public dataset; negative result acceptable; how to state the use of AI tools."],
        qa=[("Why this gap, and why does it matter?", "A flood builds within a day or two; the rain table files the afternoon rain of the issue day under the next day, so the model cannot see rain that has already fallen. Around the 611 test onsets that hidden rain is 35% of the rain of the 63 hours around the forecast."),
            ("Why a US dataset for a course in China?", "We checked two Chinese options and found them too sparse; no comparable open daily flow archive exists. This is our first question to you."),
            ("What if the LSTM does not win?", "Then that is the finding: the network is not the bottleneck, the information is. We ask whether a rigorously shown negative result is acceptable."),
            ("Who wrote the code?", "Claude Code wrote and ran all code under my direction; independent AI reviewers checked the experiments; the team's own checks are still open. We say so in every document."),
            ("Why an LSTM?", "Rivers remember past rain for weeks; a model with memory fits, and it is the published state of the art on CAMELS (Kratzert 2019, Nearing 2024).")],
        files=["proposal/AI_and_ALLMs_Proposal_Flood.md (sections 1 to 4, 6.5, 8)", "proposal/Meeting_Discussion_Points.md (sections 4 and 6)", "PLAN.md (Roles, Phases)", "proposal/Flood_Proposal_Slides_notes.md"]),
    "R2": dict(
        explain=["<b>What the 12 basins are and how they were chosen</b> (section 5): from 671 CAMELS-US basins; 78 candidates, 63 with a complete record, four per landscape evenly spaced in area, no flow statistic or model score used; 113 to 1,928 km2; mountains, northern lowlands and Piedmont, southern lowlands.",
                 "<b>The data products</b>: what each gives and where it is obtained, all public and free (CAMELS-US, AORC, GEFSv12, Stage IV, ERA5, IMERG and MERRA-2, elevation tiles, NLCD, HURDAT2).",
                 "<b>The map</b>: elevation inside each outline, gauge as a white dot, geographic latitude and longitude, scale bar per panel.",
                 "<b>Why no capture date or sun angle</b>: no optical or SAR scene is processed; the satellite-derived products are used ready-made."],
        qa=[("What is the data, and where do you obtain it?", "CAMELS-US (flow, Daymet weather, 10 basin properties, outlines) plus hourly rain (AORC, ERA5), an archived forecast (GEFSv12), radar rain (Stage IV), satellite rain and soil wetness, elevation and land cover. All public, read without payment or account; about 73 GB read, 5.9 GB kept."),
            ("What is the data flow?", "Eleven products are averaged into basin means (overlap-weighted for IMERG and GEFS), summed over each gauge's local day, joined into one basin-by-day table of 12,418 days per basin, then split by years. The diagram is slide 4."),
            ("How were flow and basin means computed?", "Flow in mm per day: q = cfs x 0.0283168 x 86,400 / area x 1000, recomputed for all 149,016 basin-days (largest difference 0). Basin mean = sum of w x value over sum of w, with w the overlap area of each cell."),
            ("Capture date, sun angle, cloud screening?", "None apply: no optical or SAR scene is processed, so no atmospheric or radiometric correction or co-registration. Elevation is SRTM of February 2000 (static); land cover is the 2011 map."),
            ("Why these twelve basins, and are they representative?", "A rule fixed before any model ran. They are humid south-eastern basins and say nothing about snow-fed, arid or very large rivers.")],
        files=["proposal/AI_and_ALLMs_Proposal_Flood.md (section 5)", "results/data_flow.json", "figures/basin_maps.png", "results/basin_gis.json", "ROADMAP.md (section 3: data that can be had)"]),
    "G1": dict(
        explain=["<b>Why an LSTM</b> against three simple baselines (tomorrow = today, ridge, random forest): memory of past rain, one network for all 12 basins as the literature advises; three seeds each, early stopping on validation.",
                 "<b>Why the test is fair</b> (section 6.3 and 6.4): split by target day, training 1980 to 2000, validation 2000 to 2005, test 2005 to 2014 scored once per finished experiment; leave-one-basin-out; a leak test on made-up rivers; features checked not to read past the issue day.",
                 "<b>The clocks</b>: hourly rain is summed over the local day, the Daymet day ends about 09:00, NLDAS-2 about 17:30, the issue time is local midnight.",
                 "<b>Scoring and verdicts</b>: NSE, KGE, onsets warned, false warnings; GO, CAUTION, NOT YET fixed in advance; headline numbers recomputed by a second AI reviewer."],
        qa=[("Why an LSTM?", "Rivers remember past rain for weeks; a model with memory is the natural fit and the published state of the art on this dataset. We do not assume it wins: RQ1 tests it against simple models."),
            ("How is the test fair?", "Train 1980-2000, choose settings on 2000-2005, score 2005-2014 only; a leak test on made-up rivers; leave-one-basin-out for rivers never seen; reviewers recomputed the numbers."),
            ("What is the split, and why by day?", "Temporal by target day, so no test day is seen in training: 7,305 days train, 1,826 validation, 3,287 test, from the split box of the slide."),
            ("What does 'the clock' mean?", "Each rain table cuts its day at a different hour. We measured them against hourly rain; a forecast issued at local midnight must not read anything later."),
            ("What is an onset, and what are NSE and KGE?", "An onset is a day above the flow exceeded on 5% of training days, forecast while the river is still below it. NSE and KGE score the flow forecast against the gauge; 1 is perfect.")],
        files=["proposal/AI_and_ALLMs_Proposal_Flood.md (section 6)", "figures/data_flow_talk.png (slide 4) and results/data_flow.json", "code/flood.py (split, scoring) and code/test_flood.py", "ROADMAP.md (sections 1 and 2)"]),
    "G2": dict(
        explain=["<b>The five deliverables D1 to D5</b> and what accepts each (section 7).",
                 "<b>Why those targets</b>: baseline 19% of onsets warned and 59% of warnings false; targets above 45% warned (network trained on real-time radar) and below 50% false; five trainings per model where the NSE decides.",
                 "<b>What preliminary means</b>: laptop runs of October 2026, three trainings, not yet repeated five times; median NSE 0.59 against 0.56 (no clear difference); 51% warned with AORC rain to midnight; 48% with Stage IV.",
                 "<b>Why the verdict is NOT YET and the caveat on the radar number</b>: GO needs 70% warned and 30% false; the 48% comes from networks trained on AORC, an analysis made after the fact, so it is a check, not a service."],
        qa=[("Why those targets?", "Each is set against the baseline: onsets warned from 19% to above 45%, false warnings from 59% of warnings to below 50%, with the radar-trained network because radar is what is at hand in real time. They are targets, to be met and not yet achieved."),
            ("What does 'preliminary' mean?", "Laptop runs of October 2026 with three trainings per model, before the team has repeated and checked them. A gain is called clear only by the rule of RQ1."),
            ("Why is the verdict NOT YET?", "GO needs at least 70% of onsets warned and at most 30% of warnings false; the best preliminary figure (AORC rain to midnight, results/aorc_rain_to_midnight.json) is 51% of onsets warned with 50% of the warnings false, short of both; CAUTION needs a CSI of at least 0.40 and that run reaches 0.34."),
            ("What is the caveat on the radar number?", "48% with Stage IV is a check on 368 test-year issue days using networks trained on AORC (hindsight bounded at about 3 points), not a network trained on radar, and not a service. Training on radar is the target."),
            ("Where does it fail?", "Onsets whose rain falls after the forecast is issued; most of the issue day's rain is invisible to the model.")],
        files=["proposal/AI_and_ALLMs_Proposal_Flood.md (sections 7 and 8)", "ROADMAP.md (sections 1 and 2; Step 1b and Step 2 results)", "results/aorc_rain_to_midnight.json, results/stage4_realtime.json, results/forecast_input.json", "results/flood_analysis.json"]),
}


def build():
    S = DK.deck()
    tot = DK.checks(S)                                                       # asserts 600 s and 150 s each
    titles = []
    for s in S:
        first = s["els"][0]
        titles.append(first["paras"][0]["runs"][0][0] if first["k"] == "text" and not s.get("cover") else "Cover: mid-term evaluation")
    return S, tot, titles


def cut_info(i, say, hand, sec):
    sents = BP.sentences(say)
    keep = [x for j, x in enumerate(sents, 1) if j not in DROP[i]]
    full_w, kept_w = DK.words(say) + DK.words(hand), sum(DK.words(x) for x in keep) + DK.words(hand)
    return keep, sents, round(sec * kept_w / full_w)


def content(S, tot, titles):
    esc, clock = BP.esc, BP.clock
    L = lambda k: esc(team.label(k))
    named = any(m["name"] for m in team.load()["members"])
    cs = team.checks_state() if hasattr(team, "checks_state") else dict(n_done=0, n=4)
    start, rows, t = [], [], 0
    for i, (s, (say, hand)) in enumerate(zip(S, DK.NOTES), 1):
        start.append(t)
        rows.append([str(i), f"<b>{L(s['who'])}</b>", f"{s['sec']} s", f"{clock(t)}&ndash;{clock(t + s['sec'])}",
                     f"<b>{esc(titles[i - 1])}</b>: {ON_SLIDE[i]}", f"&ldquo;{esc(hand)}&rdquo;"])
        t += s["sec"]
    a = [s["sec"] for s in S if s["who"] == "A"]
    B = []
    B += [("h1", "Presentation plan: the six-slide proposal talk (10 minutes)"),
          ("sub", f"AI course group project &middot; mid-term proposal &middot; for {esc(team.names_line())} &middot; built {datetime.date.today().day} {datetime.date.today():%B %Y}"
                  + ("" if named else " &middot; the names are the roles of the plan until the team fills in <code>team.json</code>")),
          ("h2", "1. What the talk is"),
          ("p", "Tomorrow&rsquo;s class talk is the <b>mid-term proposal</b>, not the results talk: 10 minutes, six slides, in the structure of the course template (cover and aim; background, gap and research questions; study area and data; "
                "methodology and workflow; expected outputs and evaluation plan; timeline, roles, disclosure and questions). The deck is <code>proposal/Flood_Proposal_Slides.pdf</code> (editable: <code>.pptx</code>), the sentences to say are in "
                "<code>proposal/Flood_Proposal_Slides_notes.md</code>, the proposal it summarises is <code>proposal/AI_and_ALLMs_Proposal_Flood.md</code> (<code>.pdf</code>, <code>.docx</code>), and the professor&rsquo;s likely questions are in "
                "<code>proposal/Meeting_Discussion_Points.md</code> (section 6; section 4 lists what we want to agree). Present full-screen from the PDF. Seconds, presenters and hand-over sentences below are read from the deck, not retyped.")]
    B += [("h2", "2. The talk, part by part"),
          ("table", ["#", "Presenter", "Time", "Clock", "Slide: what is on it", "Hand-over sentence (ends the part)"], rows, ["3%", "11%", "5%", "8%", "47%", "26%"]),
          ("p", f"<span class=\"mu\">Total {sum(s['sec'] for s in S)} s = 10 minutes; each member {tot['A']} s. {L('A')} speaks three parts: {' + '.join(str(x) for x in a)} s. "
                f"The last part ends with the thanks and the invitation to questions. Each part&rsquo;s words fit its time at most {DK.WPS:g} words a second, hand-over counted.</span>")]
    B += [("h2", "3. Each member")]
    for k in KEYS:
        m = MEMBER[k]
        mine = [(i, s) for i, s in enumerate(S, 1) if s["who"] == k]
        part = "; ".join(f"slide {i} ({titles[i - 1].lower() if i > 1 else 'cover and aim'}) &middot; {s['sec']} s from {clock(start[i - 1])}" for i, s in mine)
        B.append(("group", [("h3", f"{L(k)}"), ("p", f"<b>Your part:</b> {part}."),
                            ("p", "<b>Explain in your own words, without notes:</b>"), ("ul", m["explain"]),
                            ("p", "<b>Likely questions, short answers</b>"), ("ul", [f"<b>{esc(q)}</b> {esc(a)}" for q, a in m["qa"]]),
                            ("p", "<b>Open before the talk:</b> " + ", ".join(f"<code>{esc(f)}</code>" if "(" not in f else esc(f) for f in m["files"]))]))
    B.append(("p", "<span class=\"mu\">Questions about the whole project go to " + L("A") + ", who repeats the question and passes it to the member whose slide it is about. If a question goes beyond what we checked, say so; do not guess.</span>"))
    # the cut
    cut_rows, new_total = [], 0
    for i, (s, (say, hand)) in enumerate(zip(S, DK.NOTES), 1):
        keep, sents, new = cut_info(i, say, hand, s["sec"])
        new_total += new
        dropped = " ".join(f"<i>&ldquo;{esc(' '.join(sents[j - 1].split()[:6]))} &hellip;&rdquo;</i>" for j in DROP[i])
        cut_rows.append([str(i), f"{L(s['who'])}", f"{s['sec']} &rarr; {new} s", dropped, "kept"])
    B += [("h2", "4. The cut for a late talk (7 minutes)"),
          ("p", f"If the slot is cut to 7 minutes ({LATE_TOTAL} s), each part drops the sentences below and says the rest at the same pace; <b>every hand-over sentence stays</b>, and the caveats stay "
                f"(the gap, the preliminary label, the radar caveat, NOT YET, the disclosure). The shortened parts add up to about {new_total} s, inside the {LATE_TOTAL}. Decide at the first slide, not in the middle."),
          ("table", ["#", "Presenter", "Seconds", "Sentences dropped (their first words)", "Hand-over"], cut_rows, ["3%", "16%", "10%", "63%", "8%"])]
    B += [("h2", "5. Rehearsal"),
          ("ul", ["<b>Run 1, one part at a time, with a timer.</b> Each member reads their own part aloud from the notes, with the slide on screen, and stops when the time of the table is up; say the hand-over sentence.",
                  f"<b>Run 2, the whole talk, once, without the notes</b>, with a timer everyone can see; write down the clock at the start of each part and compare it with section 2 (checkpoints {', '.join(clock(x) for x in start[1:])}). Then {L('A')} asks the questions of section 3 one by one.",
                  "<b>Nobody reads the slide.</b> The slide is for the audience; say the point in your own words, using the numbers as the notes say them, and look at the class. Never read a table row by row."]),
          ("h2", "6. The honesty line"),
          ("note", "<b>Everyone says the same:</b> Claude Code wrote and ran the code under the team lead&rsquo;s direction; independent AI reviewers checked the experiments; the team&rsquo;s own checks are still open "
                   f"(<code>team.json</code> records {cs['n_done']} of {cs['n']} as done). <b>Preliminary numbers are labelled preliminary</b> (laptop runs, October 2026, three trainings), "
                   "<b>and targets are targets</b>: to be met, not yet achieved. Nobody says that they checked something they have not checked.")]
    return B


def main():
    S, tot, titles = build()
    blocks = content(S, tot, titles)
    css = BP.CSS + "\nbody{font-size:8.1pt} .mem li{font-size:7.7pt} ul{padding-left:10pt}\n"
    OUT.mkdir(exist_ok=True)
    h = OUT / f"{NAME}.html"
    h.write_text(page("Presentation plan, proposal talk", BP.render_html(blocks), css), encoding="utf-8")
    pdf = OUT / f"{NAME}.pdf"
    wrote = print_pdf(str(h.resolve()), str(pdf.resolve()))
    h.unlink()
    n = pdf_pages(str(pdf.resolve())) if wrote else None
    head = ["<!-- Built by code/build_presentation_plan_proposal.py from the deck (build_proposal_slides.deck() and NOTES): a rebuild overwrites this file. -->", ""]
    (OUT / f"{NAME}.md").write_text("\n".join(head + BP.render_md(blocks)) + "\n", encoding="utf-8")
    print("pages:", n)
    if n and n > MAX_PAGES:
        print(f"the plan has {n} pages, more than {MAX_PAGES}", file=sys.stderr)
    print("wrote", pdf, OUT / f"{NAME}.md")


if __name__ == "__main__":
    main()
