"""The presentation plan for the FIVE-slide talk (the short form): Presentation_Plan.html, Presentation_Plan.pdf and Presentation_Plan.md (the plan handed in; the 13-slide deck's plan is Presentation_Plan_full_deck.*).

A small sibling of code/build_presentation_plan.py, which it imports: the helpers, the HTML and Markdown renderers and the style sheet of that builder are reused as they are,
and so are the member checklists and the four things every member must be able to explain of Professor_QA (code/build_qa.py), the files each member opens (the report's
section 9 list), the roles of PLAN.md, the names and the state of the members' own checks (team.json through code/team.py: no builder sets a check to done) and the honesty
wording. What differs from the plan of the 13-slide talk is what the short form needs: the table of the five slides (one part and one presenter each, with the seconds of each part
and the hand-over sentences), for each member the questions of Professor_QA that belong to their part, where the detail is in the full deck (Flood_Slides.pdf stays the detailed
record), the rehearsal and the honesty line. At most three pages. It prints no shares and no totals per member: only the seconds of each part.

Nothing is typed that a file already holds. Seconds, clock times, slide numbers, presenters, the sentences to say, the hand-over sentences, the cut for a late talk and
the slides of the full deck behind each slide are read from results/slide_plan_5.json, which code/build_slides_short.py writes with the slides (run that first). Only the list
of which questions of Professor_QA belong to which part of the talk (QA, by the questions' own keys) is written here.
Rebuild after build_slides_short.py:  python code/build_presentation_plan_short.py
"""
import datetime
import json
import sys

import build_presentation_plan as BP
import team
from common import RES, ROOT
from report_common import load_results, page, pdf_pages, print_pdf
from report_sections import s0_summary as S0
from report_sections.s8_limits import FILES, _plan

NAME = "Presentation_Plan"
MAX_PAGES = 3
# which questions of Professor_QA belong to which part of the talk (slide key, part index -> the keys of the questions): a member answers the questions of their own part
QA = {("cover", 0): ["why_lstm", "fair", "who"],
      ("data", 0): ["basins", "period", "rainday", "allrivers", "props", "rasters", "reanalysis", "imerg"],
      ("key", 0): ["lesson", "holds", "sure", "upper", "samemiss", "published", "aorc", "hindsight"],
      ("more", 0): ["baselines", "nse", "clear", "beat", "verdict", "transformer", "hours", "onsets", "moved", "fcskill", "rivers", "unseen", "spent"],
      ("end", 0): ["failed", "untested", "change", "service", "newcost", "reviewed"]}
DATA_FILES = ("results/data_flow.json", "figures/data_flow.png")                 # what the member of the dataset slide opens besides the files of the role: the record of the data and its whole flow diagram


def load_plan():
    path = RES / "slide_plan_5.json"
    if not path.exists():
        print(f"build_presentation_plan_short: {path} is missing; run python code/build_slides_short.py first (it writes the plan of the talk)", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def part_name(p):
    return f"{p['slide']}{' ' + p['part'] if p['part'] else ''}"


def content(R, T, P):
    """The document as a list of blocks (the kinds of build_presentation_plan.render_html and render_md)."""
    import build_qa
    esc, clock, and_ = BP.esc, BP.clock, BP.and_
    story, items, checklist, pending, page1 = build_qa.build(R, T)
    slides = P["slides"]
    pres = {p["key"]: p for p in P["presenters"]}
    label = lambda k: esc(pres[k]["label"]) if k in pres else esc(team.label(k))
    total, minutes = P["total_seconds"], clock(P["total_seconds"])
    n_min = f"{total // 60} minutes" if total % 60 == 0 else f"{minutes} minutes"
    members = team.load()["members"]
    role_name = {m["key"]: m["role"] for m in members}
    named = any(m["name"] for m in members)
    plan_roles = dict(_plan()[0])                                                # PLAN.md, Roles: member (as the plan writes it) -> what the member owns
    owns = {k: plan_roles.get(role_name[k], "") for k in BP.KEYS}
    parts = [dict(p, slide=s["slide"], key=s["key"], slide_title=s["title"], shared=s["shared"], idx=j) for s in slides for j, p in enumerate(s["parts"])]
    by_item = {i["key"]: i for i in items}
    blocks = []

    # ---- title and purpose
    blocks += [("h1", f"Presentation plan: the five-slide, {n_min.replace(' minutes', '-minute')} talk and the questions after it"),
               ("sub", f"AI course group project &middot; for {esc(team.names_line())} &middot; built {datetime.date.today().day} {datetime.date.today():%B %Y}"
                       + ("" if named else " &middot; the names are the roles of the plan until the team fills in <code>team.json</code>")),
               ("h2", "1. Purpose"),
               ("p", f"The short form of the talk, for the hand-in: {P['talk_slides']} slides in {n_min}, only what is critical to show the work; questions follow. "
                     f"Each slide is one part with one presenter (slide 4 has two halves to look at, but one speaker); the table of section 2 gives the seconds of each part, the clock at "
                     f"which it starts and the hand-over sentence that ends it. "
                     f"The full deck, <code>Flood_Slides.pdf</code> ({P['full_deck_slides']} slides and {P['full_deck_backup_slides']} backup slides, notes in <code>slides_notes.md</code>), "
                     f"stays the detailed record for the questions that go beyond the five slides (section 3). Present from <code>Flood_Slides_5.pdf</code> in full-screen mode; "
                     f"<code>slides_notes_5.md</code> has the sentences to say, <code>Professor_QA.pdf</code> the questions the professor may ask. Seconds, clock times and slide numbers are "
                     f"read from <code>results/slide_plan_5.json</code>, which <code>code/build_slides_short.py</code> writes with the slides.")]

    # ---- 2. the talk, part by part
    rows = []
    for i, p in enumerate(parts):
        nxt = parts[i + 1] if i + 1 < len(parts) else None
        if nxt is None:                                                          # the last part hands over to the question time
            hand = f"<b>&rarr; question time, led by {label('A')}:</b> &ldquo;{esc(p['say'][-1])}&rdquo;" if p["say"] else ""
        elif nxt["presenter"] == p["presenter"]:
            hand = "<span class=\"mu\">same speaker continues</span>"
        else:
            sentence = p["handover"] or (BP.sentences(p["say"][-1])[-1] if p["say"] else "")
            inside = " (inside the slide)" if nxt["slide"] == p["slide"] else ""
            hand = f"<b>&rarr; {label(nxt['presenter'])}{inside}:</b> &ldquo;{esc(sentence)}&rdquo;"
        which = f"{p['slide']}" + (f"<br><span class=\"mu\">{p['part']}</span>" if p["part"] else "")
        rows.append([which, f"<b>{label(p['presenter'])}</b>", f"{p['seconds']} s<br><span class=\"mu nw\">{p['start']}&ndash;{p['end']}</span>", esc(p["says"]), hand])
    ck = P["clock_checkpoints"]
    blocks += [("h2", "2. The talk, part by part"),
               ("table", ["#", "Presenter", "Time", "It says", "Hand-over to the next speaker"], rows, ["6%", "11%", "9%", "40%", "34%"]),
               ("p", "<span class=\"mu\">Hand-over: the sentence that ends the part (<code>Hand-over</code> in <code>slides_notes_5.md</code>; where a part has none, its last sentence). "
                     + f"The sentences of each part fit its time at a calm pace of {P['words_per_second']:g} words a second, the hand-over counted.</span>")]

    # (the section 'Who speaks how much' - shares per member - was removed on 2026-10-08 at the team lead's request; the per-part seconds stay in the table above)

    # ---- 3. where the detail is (on the first page, with the table of the talk)
    drows = []
    behind = lambda ds: "; ".join(f"{('backup slide B' + str(d['backup'])) if d['backup'] else 'slide ' + str(d['slide'])} <span class=\"mu\">({esc(d['label'])})</span>" for d in ds)
    for s in slides:
        drows.append([str(s["slide"]), " &rarr; ".join(f"<b>{label(p)}</b>" for p in s["presenters"]), behind(s["full_deck"])])
    data_slide = next((s for s in slides if s["key"] == "data"), None)
    blocks += [("h2", "3. Where the detail is (the full deck)"),
               ("table", ["#", "Presenter", "Slides of the full deck behind it (the presenter opens them; the facts for questions are in slides_notes.md)"], drows, ["8%", "22%", "70%"])]
    if data_slide:
        blocks.append(("p", f"<span class=\"mu\">Slide {data_slide['slide']} (the dataset): every product with its grid, step, period and imagery facts is in <code>results/data_flow.json</code> and under "
                            f"<i>If asked</i> in <code>slides_notes_5.md</code>; the whole flow diagram is <code>figures/data_flow.png</code>.</span>"))

    # ---- 4. the questions after the talk (on the first page: who answers what)
    n_top = len(page1["top"]) if page1 and page1.get("top") else 0
    blocks += [("h2", "4. The questions after the talk"),
               ("p", f"{label('A')} leads the question time. He repeats each question and answers it when it is about the project as a whole (the data, why an LSTM, who did what); otherwise "
                     f"he passes it on to the member who presented the slide it is about, or to the owner of that part of the plan (the roles in section 5); a question that needs a slide of the full deck goes to the member who presented the five-slide "
                       "slide it belongs to. If a question goes beyond what we checked, say so; do not guess."
                     + (f" The {S0.NUMBER.get(n_top, n_top)} most likely questions are on page 1 of <code>Professor_QA.pdf</code>." if n_top else ""))]

    # ---- 5. each member's part: the slides they stand beside, the questions of their part, the checks of Professor_QA, the files to open
    A1_4 = [build_qa.md(i["q"]) for i in items if i["group"] == "A"]
    a_numbers = [i["number"] for i in items if i["group"] == "A"]
    a_first, a_last = (a_numbers[0], a_numbers[-1]) if a_numbers else ("A1", "A4")
    everyone = checklist[0][1] if checklist and build_qa.md(checklist[0][0]).startswith("Everyone") else []
    order = [BP.html_inline(build_qa.md(t)) for t in everyone if build_qa.md(t).startswith("Say ")]
    mem_checks = dict(zip(BP.KEYS, checklist[1:5])) if len(checklist) >= 5 else {}
    if not mem_checks:
        print("build_presentation_plan_short: the checklist of Professor_QA does not have the four member lists; the plan leaves them out", file=sys.stderr)
    blocks += [("pb",), ("h2", "5. Each member's part"),
               ("p", "<b>Every member must be able to explain, aloud and without notes:</b> " + " ".join(f"<i>{BP.html_inline(q)}</i>" for q in A1_4)
                     + f" The short answers are questions {a_first} to {a_last} of <code>Professor_QA.pdf</code>. Under each name: the parts you present; the questions of "
                       "<code>Professor_QA.pdf</code> that belong to them (their number, then the question); the first sentence of each item of your list at the end of that sheet; "
                       "the files to open.")]
    for k in BP.KEYS:
        if k not in pres:
            continue
        p = pres[k]
        mine = [x for x in parts if x["presenter"] == k]
        shows = []
        for x in mine:
            beside = (f"<code>figures/{x['figure']}</code>" if x.get("figure") else "a table on the slide" if x["has_table"] else "text only")
            shows.append(f"<b>{x['slide']}{' ' + x['part'] if x['part'] else ''}</b> &middot; {x['seconds']} s &middot; <span class=\"mu\">{beside}</span>")
        qa_keys = []
        for x in mine:
            qa_keys += [q for q in QA.get((x["key"], x["idx"]), []) if q not in qa_keys]
        qas = [f"<b>{by_item[q]['number']}</b> {BP.html_inline(build_qa.md(by_item[q]['q']))}" for q in qa_keys if q in by_item]
        missing = [q for q in qa_keys if q not in by_item]
        if missing:
            print(f"build_presentation_plan_short: Professor_QA has no question with the key {', '.join(missing)}; left out of the list of {team.label(k)}", file=sys.stderr)
        files = list(next((FILES[f] for f in FILES if role_name[k].startswith(f)), []))
        if any(x["key"] == "data" for x in mine):                                  # the member who presents the dataset opens its record and its whole diagram too
            files += [f for f in DATA_FILES if f not in files]
        checks = [BP.html_inline(BP.first_sentence(build_qa.md(t), 270)) for t in mem_checks[k][1]] if k in mem_checks else []
        blocks.append(("group", [("h3", f"{label(k)} <span class=\"mu\">&middot; plan role: {esc(owns[k] or p['job'])} &middot; {esc(BP.slides_word(p['slides']))}</span>"),
                                 ("p", "<b>Your part:</b> " + "; ".join(shows)),
                                 ("p", "<b>Questions of your part</b> (<code>Professor_QA.pdf</code>)"), ("ul", qas, 2),
                                 ("p", "<b>Check yourself</b>"), ("ul", checks, 2),
                                 ("p", "<b>Open before the talk:</b> " + ", ".join(f"<code>{esc(f)}</code>" for f in files))]))

    # ---- 6. rehearsal
    cut = P.get("late_cut")
    example = " or ".join(next((i["number"] for i in items if i["group"] == g), g) for g in ("D", "E"))
    run2 = (f"<b>Run-through 2, timed, without the notes.</b> The whole {minutes}, with a timer that everyone can see. Write down the clock at the start of each of your parts"
            + (" and compare it with the table; at the checkpoints it must show " + ", ".join(f"{c['clock']} (slide {c['slide']}{' ' + c['part'] if c['part'] else ''})" for c in ck) + "." if ck else ".")
            + f" Then {label('A')} reads the questions of page 1 of <code>Professor_QA.pdf</code> one by one, and the member whose part it is answers; end with {a_first} to {a_last}, once, without the sheet.")
    blocks += [("h2", "6. Rehearsal"),
               ("ul", [f"<b>Run-through 1, together, with the notes.</b> Each member reads their own parts aloud from <code>slides_notes_5.md</code>, in order, with <code>Flood_Slides_5.pdf</code> open "
                       f"full-screen. Stop at every hand-over and say the hand-over sentence of the table. Say the numbers as the notes say them (<i>Saying it aloud</i> "
                       f"in <code>slides_notes_5.md</code>) and check each number against the slide. Letters and numbers such as {example} are questions of <code>Professor_QA.pdf</code>.",
                       run2,
                       *[f"<b>Everyone:</b> {x}" for x in order],
                       ("<b>If the talk runs long, there is one agreed cut.</b> " + f"If the clock shows more than {cut['clock_over']} when slide {cut['slide']}{' (' + cut['part'] + ')' if cut['part'] else ''} "
                        f"starts, {esc(cut['presenter_label'])} says only: &ldquo;{esc(cut['say'])}&rdquo;" + (" and then the hand-over sentence." if cut["then_handover"] else "")
                        + f" That saves about {cut['saves_seconds']} s. No other part is shortened: the other sentences carry the caveats.") if cut else ""])]

    # ---- 7. the honesty line
    g1 = next((i for i in items if i["key"] == "who"), None)
    trap = BP.html_inline(" ".join(BP.sentences(build_qa.md(g1["trap"]))[:3]).replace("(list at the end)", "(list at the end of Professor_QA.pdf)")) if g1 else ""
    cs = team.checks_state()
    blocks += [("h2", "7. The honesty line"),
               ("note", "<b>If anyone asks who wrote the code, every member says the same:</b> Claude Code, our coding assistant, wrote all the code and ran all the experiments under "
                        "The team lead&rsquo;s direction; " + esc(S0.REVIEW_WORDING) + "; " + esc(team.checks_sentence(short=False)) + ". "
                        "<b>Nobody says</b> that they checked something they have not checked."
                        + (f" The trap, as <code>Professor_QA.pdf</code> words it: {trap}" if trap else "")),
               ("p", "<b>Where a finished check is recorded:</b> in <code>team.json</code>, by the member, in their own entry: <code>check_done</code> true, the date and one sentence on what the check "
                     "showed. After a rebuild the report, the slides, the question sheet and this plan say so in the same words. No script sets it, and no member sets another member&rsquo;s check. "
                     f"Today <code>team.json</code> records {cs['n_done']} of the {cs['n']} own checks as done.")]
    return blocks


def main():
    R, T = load_results()
    P = load_plan()
    secs = {p["key"]: p["seconds"] for p in P["presenters"]}
    if len(secs) != len(BP.KEYS) or len(set(secs.values())) != 1:
        print("build_presentation_plan_short: the four members do not speak the same time (" + ", ".join(f"{p['label']} {p['seconds']} s" for p in P["presenters"]) + "); change the seconds in "
              "code/build_slides_short.py", file=sys.stderr)
    blocks = content(R, T, P)
    html = page("Presentation plan, five slides: Predicting river floods with AI", BP.render_html(blocks), BP.CSS)
    out = ROOT / f"{NAME}.html"
    out.write_text(html, encoding="utf-8")
    head = ["<!-- Built by code/build_presentation_plan_short.py from results/slide_plan_5.json: edit this copy by hand if you like, but a rebuild overwrites it, so keep a copy of your edits. -->", ""]
    (ROOT / f"{NAME}.md").write_text("\n".join(head + BP.render_md(blocks)) + "\n", encoding="utf-8")
    pdf = out.with_suffix(".pdf")
    wrote = print_pdf(out, pdf)
    n_pages = pdf_pages(pdf) if wrote else None
    print("pages:", n_pages)
    if n_pages and n_pages > MAX_PAGES:
        print(f"build_presentation_plan_short: the plan has {n_pages} pages, more than the {MAX_PAGES} it may have; shorten it", file=sys.stderr)
    print("wrote", out, ("and " + str(pdf)) if wrote else "(no PDF: no browser)", "and", ROOT / f"{NAME}.md")


if __name__ == "__main__":
    main()
