"""r3_new_numbers: which numbers of the hand-in documents of 8 October 2026 (17:31-17:39) are new, compared with the texts judged in rounds 1 and 2 (token level), and which of the new ones the text of results/data_flow.json contains.

    R3_TEXT_DIR=<folder with Flood_Report_3pages.txt, Flood_Slides_5.txt, Presentation_Plan_5slides.txt, pptx_text.txt>  R2_TEXT_DIR=<folder with the PDF texts of the long documents judged in round 2>  python r3_new_numbers.py

For each of the new documents (report PDF, slides PDF, slide notes, plan MD) it lists the numeric tokens (integers, decimals, percentages, signed values; single digits and 2026 left out) that occur in none of the baseline texts (the long report, slides, plan, question sheet, guide,
overview and plain-words page of round 2, and the Markdown of the long deck as it is now), and says which of them occur as written in the text of results/data_flow.json (strings, and numbers formatted with thousands separators). The rest is traced by hand: the critic's list is in the record
(round_3.checks r3c). Output: stdout and OUT/r3_new_numbers.json
"""
import re
import sys

from dcommon import *


def toks(t):
    t = re.sub(r"\n=====PAGE \d+=====\n", " ", t).replace("−", "-").replace("–", "-").replace("’", "'")
    t = re.sub(r"\s+", " ", t)
    return set(re.findall(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*%?", t))


R3, R2 = Path(os.environ["R3_TEXT_DIR"]), Path(os.environ["R2_TEXT_DIR"])
base = set()
for p in sorted(R2.glob("*.txt")):
    base |= toks(p.read_text())
for name in ("slides_notes.md", "short_report.md", "README.md"):
    if (PROJ / name).exists():
        base |= toks((PROJ / name).read_text())
DF = json.loads((PROJ / "results" / "data_flow.json").read_text())
dftext = json.dumps(DF, ensure_ascii=False).replace("−", "-").replace("–", "-")
def in_df(tok):
    t = tok.lstrip("+")
    cands = {t, t.rstrip("%")}
    for c in list(cands):
        try:
            v = float(c.replace(",", ""))
            cands |= {f"{v:,.0f}" if v == int(v) else f"{v:,}", str(int(v)) if v == int(v) else str(v)}
        except ValueError:
            pass
    return any(re.search(r"(?<![\w.,])" + re.escape(c) + r"(?![\w]|[.,]\d)", dftext) for c in cands)

docs = {"report (PDF, 3 pages)": R3 / "Flood_Report_3pages.txt", "slides (PDF, 5 slides)": R3 / "Flood_Slides_5.txt", "slide notes (PPTX notes + slides_notes_5.md)": PROJ / "slides_notes_5.md", "plan (PDF)": R3 / "Presentation_Plan_5slides.txt"}
res = {}
for name, p in docs.items():
    allt = toks(p.read_text())
    new = sorted(t for t in allt if t not in base and not re.fullmatch(r"\d", t) and t != "2026")
    found = [t for t in new if in_df(t)]
    rest = [t for t in new if not in_df(t)]
    res[name] = dict(distinct_numbers=len(allt), not_in_the_texts_judged_before=len(new), of_these_in_data_flow_json=len(found), not_in_data_flow_json=rest)
    print(f"{name}: {len(allt)} distinct numbers; {len(new)} not in the texts judged before; {len(found)} of these are in data_flow.json; the other {len(rest)}: {rest}")
jsave("r3_new_numbers.json", res)
