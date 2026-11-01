"""The presentation plan for the four members of the team: Presentation_Plan_full_deck.html, .pdf and .md (the plan of the 13-slide deck; the plan handed in, Presentation_Plan.*, is the five-slide one from build_presentation_plan_short.py).

The PDF is the sheet to print; the .md is the same content for the team to edit (a rebuild overwrites it, so keep a copy of edits). It says who presents which
slide of the 10-minute talk and for how long, what is said to hand over to the next speaker, how much of the talk each member speaks, what each member shows, stands
beside, must be able to answer and should open before the talk, which member takes each backup slide, how to answer the questions after the talk, how to rehearse
(two run-throughs, the clock checkpoints, the one agreed cut for a late talk) and what everyone says about who wrote the code.

Nothing is typed that a file already holds. Seconds, clock times, shares, slide numbers, presenters, figures, the sentences to say, the hand-over sentences and the
cut for a late talk are read from results/slide_plan.json, which code/build_slides.py writes together with the slides (run that first); the member checklists and
the four things every member must be able to explain are those of Professor_QA (code/build_qa.py, imported: the same text, the first sentence of each item); the
files each member opens are s8_limits.FILES, the report's section 9 list; the roles are the Roles table of PLAN.md; the names and the state of the members' own checks
come from team.json through code/team.py (no builder sets a check to done); the honesty wording is that of the report's 'Who did what' (s0_summary.who) and of the
question 'What did the AI assistant do, and what did the team do?' of Professor_QA. Only the short labels of the slides (SHORT) and the talk slide that each backup
slide extends (EXTENDS) are written here, as words without numbers; both are keyed by the slides' own keys, so a slide that moves keeps its label.
Rebuild after build_slides.py:  python code/build_presentation_plan.py
"""
import datetime
import html as html_lib
import json
import re
import sys

import team
from common import RES, ROOT
from report_common import load_results, page, pct, pdf_pages, print_pdf, table
from report_sections import s0_summary as S0
from report_sections.s8_limits import FILES, _plan

NAME = "Presentation_Plan_full_deck"
KEYS = ("A", "R2", "G1", "G2")                  # the four presenters of build_slides.py, in the order of the Roles table of PLAN.md
BAND = (0.46, 0.54)                             # the project lead's band for the share of the talk that the team lead speaks: outside it the build says so on stderr
# what each slide is about, in a few words (no numbers): the slide's own key -> label. A key that is not here is labelled with the slide's title.
SHORT = {"cover": "the question and the short answer", "data": "the data and the rule that chose the 12 rivers", "fair": "why the test is fair",
         "simple": "the number to beat", "all": "the LSTM against the simple models", "event": "one onset, day by day", "verdict": "what the warnings are worth",
         "fail": "where the onset warnings fail", "timing": "the rain-timing test", "rain": "the rain known up to the issue time", "forecast": "a real weather forecast",
         "need": "what it took, and what is missing", "key": "the key figure and the conclusion", "curves": "the learning curves", "memory": "how far back the memory reaches",
         "lobo": "a river never seen in training", "rivers": "more training rivers", "scores": "the warning scores at every lead time",
         "cost": "the Transformer and what training costs", "space": "satellite rain as an input"}
# the talk slide that each backup slide extends: where a question that leads to it is most likely to come from
EXTENDS = {"curves": "all", "memory": "data", "lobo": "need", "rivers": "need", "scores": "verdict", "cost": "all", "space": "timing"}


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def esc(text):
    """Plain text (a name, a sentence of a results file) -> HTML: escaped, apostrophes typeset."""
    return S0._esc(text)


def clock(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def sentences(text):
    return [x for x in re.split(r"(?<=[.!?])\s+", str(text).strip()) if x]


def and_(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def slides_word(nums):
    return ("slide " if len(nums) == 1 else "slides ") + and_(nums)


def md_inline(h):
    """The small set of HTML used here -> Markdown."""
    s = h.replace("<br>", " ")
    s = re.sub(r"</?b>", "**", s)
    s = re.sub(r"</?i>", "*", s)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(html_lib.unescape(s).split())


def html_inline(md):
    """Markdown with **bold**, *italic* and `code` (the Professor_QA text) -> HTML."""
    s = html_lib.escape(md, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*(?!\*)(.+?)(?<![*\s])\*(?!\*)", r"<i>\1</i>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def first_sentence(md, limit=300):
    """The first sentence of a checklist item of Professor_QA (as Markdown); a longer one is cut at its last ';', ':' or ',' before `limit` and ends in an ellipsis."""
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z`‘“(])", md.strip())[0]
    if len(first) <= limit:
        return first
    cut = max(first.rfind(c, 0, limit) for c in (";", ":", ","))
    return first[:cut if cut > limit // 2 else limit].rstrip(" ,;:") + " …"


def load_plan():
    path = RES / "slide_plan.json"
    if not path.exists():
        print(f"build_presentation_plan: {path} is missing; run python code/build_slides.py first (it writes the plan of the talk)", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


# ---- the content, as blocks that both the HTML and the Markdown are made from ----------------------------------------------
def content(R, T, P):
    """The document as a list of blocks: ('h1', text), ('sub', html), ('h2', text), ('h3', html), ('p', html), ('note', html), ('ul', [html]), ('table', header, rows, widths),
    ('bar', [(share, is_first)]) and ('two', [blocks], [blocks], left_share) - a left and a right column."""
    import build_qa
    story, items, checklist, pending, page1 = build_qa.build(R, T)               # the sheet's own text: the four things everyone explains, the member checks, the honest sentence
    slides = P["slides"]
    talk = [s for s in slides if s["kind"] == "talk"]
    back = [s for s in slides if s["kind"] == "backup"]
    pres = {p["key"]: p for p in P["presenters"]}
    label = lambda k: esc(pres[k]["label"]) if k in pres else esc(team.label(k))
    total, minutes = P["total_seconds"], clock(P["total_seconds"])
    n_min = f"{total // 60} minutes" if total % 60 == 0 else f"{minutes} minutes"
    by_key = {s["key"]: s for s in slides}
    short = lambda s: SHORT.get(s["key"], s["title"])
    members = team.load()["members"]
    role_name = {m["key"]: m["role"] for m in members}
    named = any(m["name"] for m in members)
    plan_roles = dict(_plan()[0])                                                # PLAN.md, Roles: member (as the plan writes it) -> what the member owns
    owns = {k: plan_roles.get(role_name[k], "") for k in KEYS}
    A = pres["A"]
    blocks = []

    # ---- title and purpose
    blocks += [("h1", f"Presentation plan: the {n_min.replace(' minutes', '-minute')} talk and the questions after it"),
               ("sub", f"AI course group project &middot; for {esc(team.names_line())} &middot; built {datetime.date.today().day} {datetime.date.today():%B %Y}"
                       + ("" if named else " &middot; the names are the roles of the plan until the team fills in <code>team.json</code>")),
               ("h2", "1. Purpose"),
               ("p", f"The talk lasts {n_min} and has {P['talk_slides']} slides; questions follow it. This sheet is for the four members: who presents which slide and for how long, what is "
                     f"said to hand over, what each member must be able to answer, who takes the backup slides, how to rehearse, and what we all say about who wrote the code. Present "
                     f"from <code>Flood_Slides.pdf</code> in full-screen mode; <code>slides_notes.md</code> has the sentences to say, <code>Professor_QA.pdf</code> the questions the "
                     f"professor may ask. Seconds and slide numbers are read from <code>results/slide_plan.json</code>, which <code>code/build_slides.py</code> writes with the slides.")]

    # ---- 2. the talk, slide by slide
    rows = []
    for i, s in enumerate(talk):
        nxt = talk[i + 1] if i + 1 < len(talk) else None
        if nxt is None:                                                          # the last slide hands over to the question time
            hand = f"<b>&rarr; question time, led by {label('A')}:</b> &ldquo;{esc(s['say'][-1])}&rdquo;" if s["say"] else ""
        elif nxt["presenter"] == s["presenter"]:
            hand = "<span class=\"mu\">same speaker continues</span>"
        else:
            sentence = s["handover"] or (sentences(s["say"][-1])[-1] if s["say"] else "")
            hand = f"<b>&rarr; {label(nxt['presenter'])}:</b> &ldquo;{esc(sentence)}&rdquo;"
        rows.append([str(s["slide"]), f"<b>{label(s['presenter'])}</b>", f"{s['seconds']} s<br><span class=\"mu nw\">{s['start']}&ndash;{s['end']}</span>", esc(s["title"]), hand])
    ck = P["clock_checkpoints"]
    blocks += [("h2", "2. The talk, slide by slide"),
               ("table", ["#", "Presenter", "Time", "The slide says", "Hand-over to the next speaker"], rows, ["3.5%", "11.5%", "9.5%", "38.5%", "37%"]),
               ("p", "<span class=\"mu\">Hand-over: the sentence that ends the slide (<code>Hand-over</code> in <code>slides_notes.md</code>; where a slide has none, its last sentence). "
                     + ("Clock checkpoints: " + "; ".join(f"when slide {c['slide']} starts the clock shows {c['clock']}" for c in ck) + ". " if ck else "")
                     + f"The sentences of each slide fit its time at a calm pace of {P['words_per_second']:g} words a second, the hand-over counted.</span>")]

    # ---- 3. the share of the talk
    share_rows = [[f"<b>{label(p['key'])}</b>", esc(slides_word(p["slides"])), clock(p["seconds"]), f"<b>{pct(p['share'])}</b>"] for p in P["presenters"]]
    share_rows.append(["", "", f"<b>{minutes}</b>", "<b>100%</b>"])
    opens = talk[0]["presenter"] == "A" and talk[-1]["presenter"] == "A"
    others = sorted(len(p["slides"]) for p in P["presenters"] if p["key"] != "A")
    _few = lambda ns: S0.NUMBER.get(ns[0], ns[0]) if ns[0] == ns[-1] else f"{S0.NUMBER.get(ns[0], ns[0])} or {S0.NUMBER.get(ns[-1], ns[-1])}"
    mine_words = (f"{label('A')} speaks {A['seconds']} s of the {total} s, {pct(A['share'])} of the talk, {S0.share_word(A['share'])}. "
                  + (f"He opens the talk (slide {talk[0]['slide']}) and closes it (slide {talk[-1]['slide']}), and he " if opens else "He ")
                  + "leads the question time, which brings his part to about half of the whole session. The other three members share the rest, each with "
                  + (f"{_few(others)} slides." if others else "no slides."))
    # (the section 'Who speaks how much' - shares per member - was removed on 2026-10-08 at the team lead's request; the per-part seconds stay in the table above)

    # ---- 4. each member's part
    A1_4 = [build_qa.md(i["q"]) for i in items if i["group"] == "A"]
    a_numbers = [i["number"] for i in items if i["group"] == "A"]
    a_first, a_last = (a_numbers[0], a_numbers[-1]) if a_numbers else ("A1", "A4")
    example = " or ".join(next((i["number"] for i in items if i["group"] == g), g) for g in ("D", "E"))      # two question labels of the sheet, as examples
    everyone = checklist[0][1] if checklist and build_qa.md(checklist[0][0]).startswith("Everyone") else []
    order = [html_inline(build_qa.md(t)) for t in everyone if build_qa.md(t).startswith("Say ")]
    if not order:
        print("build_presentation_plan: Professor_QA has no 'Say ... in this order' item under Everyone; the plan leaves them out", file=sys.stderr)
    blocks += [("pb",), ("h2", "4. Each member's part"),
               ("p", "<b>Every member must be able to explain, aloud and without notes:</b> " + " ".join(f"<i>{html_inline(q)}</i>" for q in A1_4)
                     + f" The short answers are questions {a_first} to {a_last} of <code>Professor_QA.pdf</code>. Under each name: the slides and the figure you stand beside; the first sentence of each item of that "
                     "member&rsquo;s list at the end of <code>Professor_QA.pdf</code>; the files to open.")]
    mem_checks = {}
    if len(checklist) >= 5:
        mem_checks = dict(zip(KEYS, checklist[1:5]))                             # the order of build_qa.build: Everyone, the team lead, RS/GIS member 2, GNSS member 1, GNSS member 2
    else:
        print("build_presentation_plan: the checklist of Professor_QA does not have the four member lists; the plan leaves them out", file=sys.stderr)
    for k in KEYS:
        if k not in pres:
            continue
        p = pres[k]
        mine = [s for s in talk if s["presenter"] == k]
        shows = []
        for s in mine:
            if s["figure"]:
                part = s.get("figure_part") or ""
                beside = f"<code>figures/{s['figure']}</code>" + (f": {esc(part)}" if "panel" in part and len(part) <= 50 else "")      # a named panel is worth saying; a long crop note is on the slide
            else:
                beside = "a table on the slide" if s["has_table"] else "text only"
            shows.append(f"<b>{s['slide']}</b> &middot; {s['seconds']} s &middot; {esc(short(s))} &middot; <span class=\"mu\">{beside}</span>")
        files = next((FILES[f] for f in FILES if role_name[k].startswith(f)), [])
        checks = [html_inline(first_sentence(build_qa.md(t), 400)) for t in mem_checks[k][1]] if k in mem_checks else []
        blocks.append(("group", [("h3", f"{label(k)} <span class=\"mu\">&middot; plan role: {esc(owns[k] or p['job'])} &middot; {esc(slides_word(p['slides']))} &middot; {clock(p['seconds'])}, {pct(p['share'])}</span>"),
                                 ("ul", shows, 2),
                                 ("p", "<b>Be ready to answer, and check yourself</b>"), ("ul", checks, 2),
                                 ("p", "<b>Open before the talk:</b> " + ", ".join(f"<code>{esc(f)}</code>" for f in files))]))

    # ---- 5. backup slides
    brows, odd = [], []
    for b in back:
        ext = by_key.get(EXTENDS.get(b["key"], ""))
        differs = ext is not None and ext["presenter"] != b["presenter"]
        if differs:
            odd.append(b)
        brows.append([f"B{b['backup']} <span class=\"mu\">(slide {b['slide']})</span>", esc(short(b)), f"<b>{label(b['presenter'])}</b>",
                      (f"slide {ext['slide']}, presented by {label(ext['presenter'])}" if ext else "&ndash;") + (" &dagger;" if differs else "")])
    blocks += [("h2", "5. Backup slides"),
               ("p", f"The {P['backup_slides']} backup slides come after slide {talk[-1]['slide']} in the PDF and have no time in the talk. Show one only if a question asks for it; its owner shows it "
                     f"(<code>slides_notes.md</code>: <i>If shown, say</i>). A backup slide keeps the owner it had: the member whose part of <code>PLAN.md</code> it belongs to."),
               ("table", ["Backup", "About", "Owner", "Extends the talk slide"], brows, ["13%", "33%", "20%", "34%"]),
               ("p", "<span class=\"mu\">&dagger; The talk slide it extends is presented by another member: the owner of the backup slide answers, and the presenter of the talk slide may add.</span>" if odd else "")]

    # ---- 6. the questions after the talk
    n_top = len(page1["top"]) if page1 and page1.get("top") else 0
    blocks += [("h2", "6. The questions after the talk"),
               ("p", f"{label('A')} leads the question time. He repeats each question and answers it when it is about the project as a whole (the data, why an LSTM, who did what); otherwise "
                     f"he passes it on: a question about a slide goes to the member who presented it, a question about a part of the plan to the owner of that part (the roles in section 4), "
                     f"and a backup slide is shown by its owner. If a question goes beyond what we checked, say so; do not guess."
                     + (f" The {S0.NUMBER.get(n_top, n_top)} most likely questions are on page 1 of <code>Professor_QA.pdf</code>." if n_top else ""))]

    # ---- 7. rehearsal
    cut = P.get("late_cut")
    run2 = (f"<b>Run-through 2, timed, without the notes.</b> The whole {minutes}, with a timer that everyone can see. Write down the clock at the start of each of your slides"
            + (" and compare it with the table; at the checkpoints it must show " + ", ".join(f"{c['clock']} (slide {c['slide']})" for c in ck) + "." if ck else ".")
            + f" Then {label('A')} reads the questions of page 1 of <code>Professor_QA.pdf</code> one by one, and the member whose part it is answers; end with {a_first} to {a_last}, once, without the sheet.")
    blocks += [("h2", "7. Rehearsal"),
               ("ul", [f"<b>Run-through 1, together, with the notes.</b> Each member reads their own slides aloud from <code>slides_notes.md</code>, in order, with <code>Flood_Slides.pdf</code> open "
                       f"full-screen. Stop at every hand-over and say the hand-over sentence of the table. Say the numbers as the notes say them (<i>Saying it aloud</i> in <code>slides_notes.md</code>) "
                       f"and check each number against the slide. Letters and numbers such as {example} are questions of <code>Professor_QA.pdf</code>.",
                       run2,
                       *[f"<b>Everyone:</b> {x}" for x in order],
                       ("<b>If the talk runs long, there is one agreed cut.</b> " + f"If the clock shows more than {cut['clock_over']} when slide {cut['slide']} starts, {esc(cut['presenter_label'])} says only: "
                        f"&ldquo;{esc(cut['say'])}&rdquo;" + (" and then the hand-over sentence." if cut["then_handover"] else "") + f" That saves about {cut['saves_seconds']} s. No other sentence "
                        f"is skipped: the other sentences carry the caveats.") if cut else ""])]

    # ---- 8. the honesty line
    g1 = next((i for i in items if i["key"] == "who"), None)
    trap = html_inline(" ".join(sentences(build_qa.md(g1["trap"]))[:3]).replace("(list at the end)", "(list at the end of Professor_QA.pdf)")) if g1 else ""
    cs = team.checks_state()
    blocks += [("h2", "8. The honesty line"),
               ("note", "<b>If anyone asks who wrote the code, every member says the same:</b> Claude Code, our coding assistant, wrote all the code and ran all the experiments under "
                        "The team lead&rsquo;s direction; " + esc(S0.REVIEW_WORDING) + "; " + esc(team.checks_sentence(short=False)) + ". "
                        "<b>Nobody says</b> that they checked something they have not checked."
                        + (f" The trap, as <code>Professor_QA.pdf</code> words it: {trap}" if trap else "")),
               ("p", "<b>Where a finished check is recorded:</b> in <code>team.json</code>, by the member, in their own entry: <code>check_done</code> true, the date and one sentence on what the check "
                     "showed. After a rebuild the report, the slides, the question sheet and this plan say so in the same words. No script sets it, and no member sets another member&rsquo;s check. "
                     f"Today <code>team.json</code> records {cs['n_done']} of the {cs['n']} own checks as done.")]
    return blocks


# ---- rendering -----------------------------------------------------------------------------------------------------------
CSS = """
@page{margin:9mm 13mm}
body{font-size:8.3pt;line-height:1.33}
h1{font-size:16pt;margin:0 0 1pt} .sub{font-size:8.2pt;color:var(--ink2);margin:0 0 3pt}
h2{font-size:10.4pt;margin:6pt 0 2.5pt;border-left:4px solid var(--blue);padding-left:6pt} h3{font-size:9.2pt;margin:4pt 0 1.5pt;padding-top:2.5pt;border-top:1px solid var(--grid)}
p{margin:0 0 2.5pt} ul{margin:0 0 2pt;padding-left:11pt} li{margin-bottom:1.2pt} code{font-size:7.4pt}
table{font-size:7.7pt;margin:1pt 0 2pt} th,td{padding:1.5pt 4pt;line-height:1.25} th{font-size:7.4pt}
.mu{color:var(--ink2);font-size:7.5pt} .nw{white-space:nowrap}
.two{display:grid;column-gap:12pt;align-items:start} .two > div > :first-child{margin-top:0} .two ul{margin-bottom:0}
.mem{break-inside:avoid} .mem li{font-size:7.55pt;line-height:1.25;margin-bottom:1.3pt;break-inside:avoid} .mem p{margin-bottom:1pt;margin-top:1.5pt} ul.c2{column-count:2;column-gap:14pt}
.note{background:#fdf6f2;border-left:3px solid var(--orange);padding:3.5pt 8pt;font-size:8.5pt;margin:2pt 0 3pt}
.bar{display:flex;gap:2px;margin:3pt 0 2pt} .bar div{padding:2pt 0;font-size:7.8pt;color:var(--ink);background:#e1e0d9;text-align:center;border-radius:2px}
.bar div.me{background:var(--blue);color:#fff;font-weight:600}
"""


def render_html(blocks):
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            out.append(f"<h1>{b[1]}</h1>")
        elif kind == "sub":
            out.append(f'<p class="sub">{b[1]}</p>')
        elif kind in ("h2", "h3"):
            out.append(f"<{kind}>{b[1]}</{kind}>")
        elif kind == "p":
            out.append(f"<p>{b[1]}</p>" if b[1] else "")
        elif kind == "note":
            out.append(f'<div class="note">{b[1]}</div>')
        elif kind == "ul":
            tag = '<ul class="c2">' if len(b) > 2 and b[2] == 2 else "<ul>"
            out.append(tag + "".join(f"<li>{x}</li>" for x in b[1] if x) + "</ul>" if any(b[1]) else "")
        elif kind == "group":
            out.append('<div class="mem">' + render_html(b[1]) + "</div>")
        elif kind == "table":
            out.append(table(b[1], b[2], cls="", widths=b[3]))
        elif kind == "bar":
            out.append('<div class="bar">' + "".join(f'<div class="{"me" if me else ""}" style="width:{100 * v:.1f}%">{pct(v)}</div>' for v, me in b[1]) + "</div>")
        elif kind == "pb":
            out.append('<div class="pb"></div>')
        elif kind == "two":
            left = b[3]
            out.append(f'<div class="two mem" style="grid-template-columns:{100 * left:.0f}% 1fr"><div>' + render_html(b[1]) + "</div><div>" + render_html(b[2]) + "</div></div>")
    return "\n".join(out)


def render_md(blocks):
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            out += [f"# {md_inline(b[1])}", ""]
        elif kind == "sub":
            out += [md_inline(b[1]), ""]
        elif kind == "h2":
            out += [f"## {md_inline(b[1])}", ""]
        elif kind == "h3":
            out += [f"### {md_inline(b[1])}", ""]
        elif kind in ("p", "note"):
            out += [md_inline(b[1]), ""] if b[1] else []
        elif kind == "ul":
            out += [f"- {md_inline(x)}" for x in b[1] if x] + [""] if any(b[1]) else []
        elif kind == "table":
            out += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])] + ["| " + " | ".join(md_inline(c) for c in r) + " |" for r in b[2]] + [""]
        elif kind in ("two", "group"):
            out += render_md(b[1]) + (render_md(b[2]) if kind == "two" else [])
    return out


def main():
    R, T = load_results()
    P = load_plan()
    A = next(p for p in P["presenters"] if p["key"] == "A")
    if not BAND[0] <= A["share"] <= BAND[1]:
        print(f"build_presentation_plan: the team lead speaks {pct(A['share'])} of the talk, outside the intended {pct(BAND[0])} to {pct(BAND[1])}; change the owners in code/build_slides.py", file=sys.stderr)
    blocks = content(R, T, P)
    html = page("Presentation plan: Predicting river floods with AI", render_html(blocks), CSS)
    out = ROOT / f"{NAME}.html"
    out.write_text(html, encoding="utf-8")
    head = ["<!-- Built by code/build_presentation_plan.py from results/slide_plan.json: edit this copy by hand if you like, but a rebuild overwrites it, so keep a copy of your edits. -->", ""]
    (ROOT / f"{NAME}.md").write_text("\n".join(head + render_md(blocks)) + "\n", encoding="utf-8")
    pdf = out.with_suffix(".pdf")
    wrote = print_pdf(out, pdf)
    n_pages = pdf_pages(pdf) if wrote else None
    print("pages:", n_pages)
    print("wrote", out, ("and " + str(pdf)) if wrote else "(no PDF: no browser)", "and", ROOT / f"{NAME}.md")


if __name__ == "__main__":
    main()
