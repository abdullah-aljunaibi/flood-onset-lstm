"""One printable page per presenter from proposal/Flood_Proposal_Presentation_notes.md (the builder writes that file): proposal/scripts/<name>.pdf and all_in_one.pdf."""
import html, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report_common import page, print_pdf, pdf_pages
ROOT = Path(__file__).resolve().parents[2]
md = (ROOT / "proposal/Flood_Proposal_Presentation_notes.md").read_text(encoding="utf-8")
keys = {n: "Cues in [brackets] are actions on the screen; everything else is what to say. Keys: 1-4 move the issue time (screen 2); O next onset, S step a day, P play (screen 4); F full screen, T timer, N notes." for n in "123456"}
parts = re.findall(r"## Screen (\d) · (.*?) · (\d+) s \((\d+) words\)\n\n(.*?)\n(?:\*\*Hand-over:\*\* (.*?)\n)?", md, re.S)
out = ROOT / "proposal/scripts"; out.mkdir(exist_ok=True)
by = {}
for n, who, sec, words, say, hand in parts:
    by.setdefault(who, []).append((n, sec, words, say.strip(), hand.strip()))
css = "<style>body{font-size:12.5pt;line-height:1.5} h1{font-size:17pt;margin:0 0 4pt} h2{font-size:13pt;margin:14pt 0 4pt} .meta{color:#555;font-size:10.5pt;margin:0 0 10pt} .do{background:#eef3fb;border-left:4px solid #2a78d6;padding:6pt 10pt;font-size:11pt;margin:6pt 0 10pt} .hand{color:#b54a1e;font-weight:700} .say{font-size:13pt}</style>"
allpages = []
for who, items in by.items():
    total = sum(int(s) for _, s, _, _, _ in items)
    body = [f"<h1>{html.escape(who)}</h1><p class='meta'>Flood proposal talk · 15 October 2026 · your part: {total} s · at most 2.3 words a second; the notes are shorter, so breathe.</p>"]
    for n, sec, words, say, hand in items:
        body.append(f"<h2>Screen {n} · {sec} s · {words} words</h2><div class='do'>{html.escape(keys[n])}</div><p class='say'>{html.escape(say)}</p>" + (f"<p class='hand'>Hand-over: {html.escape(hand)}</p>" if hand else ""))
    h = out / (re.sub(r"[^A-Za-z]+", "_", who.split(" (")[0]).strip("_") + ".html"); h.write_text(page(who, css + "".join(body)), encoding="utf-8")
    pdf = h.with_suffix(".pdf"); ok = print_pdf(h.resolve(), pdf.resolve()); print(pdf.name, "pages", pdf_pages(pdf) if ok else "FAILED"); allpages += body
h = out / "all_in_one.html"; h.write_text(page("All scripts", css + "".join(allpages)), encoding="utf-8"); print("all_in_one.pdf pages", pdf_pages(h.with_suffix(".pdf")) if print_pdf(h.resolve(), h.with_suffix(".pdf").resolve()) else "FAILED")
