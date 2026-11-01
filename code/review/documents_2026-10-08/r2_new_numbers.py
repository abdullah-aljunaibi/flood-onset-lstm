"""r2_new_numbers: which numbers are new in a rebuilt document, compared with the version that was judged (token level).

    python code/review/documents_2026-10-08/r2_new_numbers.py NAME=OLD.txt:NEW.txt [NAME=OLD.txt:NEW.txt ...]

OLD and NEW are text files made by r2_check / pdf2txt (pages split by '=====PAGE n====='; page footers such as 'n / 31' are removed). For each document it prints the numeric tokens (integers, decimals,
percentages, signed values) that occur more often in NEW than in OLD (single digits and the year 2026 left out). Each must be traced to a results file by a person; the review record lists the ones of 8 October 2026.
Output: stdout and OUT/r2_new_numbers.json
"""
import collections
import re
import sys

from dcommon import jsave


def toks(path):
    t = open(path).read()
    t = re.sub(r"\n=====PAGE \d+=====\n", " ", t)
    t = re.sub(r"Predicting river floods with AI · [^·]*?(?:project|step by step|probing questions|sheet)?\s*\d+ / \d+", "", t)
    t = t.replace("−", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t)
    return collections.Counter(re.findall(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*%?", t))


res = {}
for arg in sys.argv[1:]:
    name, files = arg.split("=", 1)
    old, new = files.split(":", 1)
    a, b = toks(old), toks(new)
    more = [(t, n - a.get(t, 0)) for t, n in b.items() if n > a.get(t, 0) and not re.fullmatch(r"\d", t) and t not in ("2026",)]
    res[name] = sorted(more, key=lambda x: -x[1])
    print(f"== {name}: numbers that are more frequent in the new text: {res[name]}")
jsave("r2_new_numbers.json", res)
