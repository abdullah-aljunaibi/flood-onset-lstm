"""d5: the key figure (results/key_figure.json, written by code/11_key_figure.py; figures/key_figure.png, key_figure_plain.png, key_figure_talk.png) checked with the critic's own code.

What the file states about itself: 607 numbers in 'sources' (491 read from a results file by a key, 116 computed from others), 159 of them drawn in the full rendering; its 'checks' block says that 1,039 comparisons were made at
a tolerance of 0.0005 with a largest difference of 5.6e-17, no disagreement, and that every number the drawing plots or prints is found in 'sources' (and flagged in_figure) and is read again from its file or recomputed.

What this script does, with its own path reader and its own scoring (nothing of 11_key_figure.py is imported or run; the picture is not redrawn):
  A  re-reads each of the 491 values of 'sources' from the results file named in it, by the key named in it, and compares (floats at 1e-12);
  B  re-reads each of the cross-check entries ('checked_against': 'results/x.json: key') and compares them with the source value at the builder's tolerance 0.0005 (and reports the largest difference);
  C  recomputes each of the 116 computed sources (differences, ratios of steps, highest minus lowest of the kinds of model, shares) from the other sources;
  D  recomputes, from the stored forecasts (results/preds/*.npz) with its own scoring (high-flow level = 95th percentile of the training years; onset = high-flow day forecast while the river was at or below the level
     lead days before; warned = forecast above the level; NSE median of the 12 basins over the test years), every source of the form l<lead>.<set>.nse / onset_warned / onset_false and its per-seed versions where
     single-seed forecasts are stored, and compares it with the stored value (1e-9);
  E  the drawn values: the numbers printed in the picture - read from the three PNG renderings by eye and cross-checked with an OCR of the picture (Apple Vision, ocr.swift; optional input FIGURE_OCR_DIR) - are each
     matched with the source they must come from and formatted the way the figure prints them; and the lettering numbers of the figure (611, 951, 1,116 onsets, '8 points apart', '0.03 NSE apart', 'higher in only 5 of 12 basins',
     'its LSTM twin: 18%, 0.60', 'GO: at least 70%') and the caption's numbers are matched with their sources; the 'mixed' labels with the readings stored in the file;
  F  recounts what the checks block counts (sources, computed, flagged as drawn, sources with a file) and the hashes of the file and the three pictures that were checked.
Output: OUT/d5_key_figure.json
"""
import hashlib
import re
import time

from dcommon import *
import flood as F

t0 = time.time()
K = jload("key_figure")
S = K["sources"]
JF = {}


def jfile(rel):
    if rel not in JF:
        with open(PROJ / rel) as f:
            JF[rel] = json.load(f)
    return JF[rel]


# ---- the reader of a key such as 'sets.w.per_seed.0.test.l1.nse_median', 'closure.wqn.by_lead[0].gain' or 'comparison.test.l1.lstm.nse_median_by_seed[1]' -----------------------------------------------------------
def walk(obj, key):
    toks = re.findall(r"\[\d+\]|[^.\[\]]+", key)

    def rec(o, i):
        if i == len(toks):
            return o
        t = toks[i]
        if t.startswith("["):
            if not isinstance(o, list):
                raise KeyError(key)
            return rec(o[int(t[1:-1])], i + 1)
        if isinstance(o, list):                                    # a list reached by a bare number
            return rec(o[int(t)], i + 1)
        if not isinstance(o, dict):
            raise KeyError(key)
        j_max = next((j for j in range(i, len(toks)) if toks[j].startswith("[")), len(toks))
        for j in range(i + 1, j_max + 1):                          # keys may themselves contain dots
            k = ".".join(toks[i:j])
            if k in o:
                try:
                    return rec(o[k], j)
                except (KeyError, IndexError, ValueError):
                    pass
        raise KeyError(key)
    return rec(obj, 0)


def read(rel, key):
    """The value a source names. Three forms of key are not plain paths: 'x (number of entries)' = the length of the list x, 'x (hits + misses)' = the sum of the two fields of the table x, and a value that the file holds
    as a sentence ('pod >= 0.70 and far <= 0.30', the GO criterion) is taken as the first number after '>=' when the source holds a number."""
    m = re.match(r"^(.*) \(number of entries\)$", key)
    if m:
        return len(walk(jfile(rel), m[1]))
    m = re.match(r"^(.*) \(hits \+ misses\)$", key)
    if m:
        t = walk(jfile(rel), m[1])
        return t["hits"] + t["misses"]
    return walk(jfile(rel), key)


def number_in_sentence(v, other):
    """If the file holds a sentence and the other value is a number, the first number after '>=' in the sentence; otherwise v."""
    if isinstance(v, str) and isinstance(other, (int, float)) and not isinstance(other, bool):
        m = re.search(r">=\s*([0-9.]+)", v)
        if m:
            return float(m[1])
    return v


def undefined(x):
    return x is None or (isinstance(x, float) and x != x)


def same(a, b, tol):
    """(equal, difference): numbers within tol, other values by equality; None and NaN both mean 'undefined' and are equal to each other."""
    a = number_in_sentence(a, b)
    if undefined(a) and undefined(b):
        return True, 0.0
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, (str, list, dict)) or isinstance(b, (str, list, dict)):
        return a == b, 0.0 if a == b else float("nan")
    d = abs(float(a) - float(b))
    return d <= tol, d


out = {}
# ---- A ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
A = []
for name, s in S.items():
    if "file" in s and "key" in s:
        try:
            v = read(s["file"], s["key"])
        except Exception as e:                                                   # noqa
            A.append(dict(source=name, ok=False, why=f"not readable: {e!r}"))
            continue
        ok, d = same(v, s["value"], 1e-12)
        A.append(dict(source=name, ok=bool(ok), diff=d))
out["A_values_read_again"] = dict(n=len(A), n_equal=sum(a["ok"] for a in A), largest_difference=max([a.get("diff", 0.0) for a in A if a["ok"]] or [0.0]), not_equal=[a for a in A if not a["ok"]])

# ---- B ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
B, generic = [], []
for name, s in S.items():
    for c in s["checked_against"]:
        m = re.match(r"^(results/\S+\.json): (.+)$", c)
        if not m:
            generic.append(dict(source=name, text=c))
            continue
        try:
            v = read(m[1], m[2])
        except Exception as e:                                                   # noqa
            B.append(dict(source=name, against=c, ok=False, why=f"not readable: {e!r}"))
            continue
        ok, d = same(v, s["value"], 5e-4)
        B.append(dict(source=name, against=c, ok=bool(ok), diff=d))
out["B_cross_checks_read_again"] = dict(n=len(B), n_equal=sum(b["ok"] for b in B), largest_difference=max([b.get("diff", 0.0) for b in B if b["ok"]] or [0.0]), not_equal=[b for b in B if not b["ok"]],
                                        not_machine_readable=len(generic), not_machine_readable_text=sorted({g["text"] for g in generic}))

# ---- C ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
C = []


def recompute(name, s):
    if "difference_of" in s:
        a, b = s["difference_of"]
        return S[a]["value"] - S[b]["value"]
    c, lead = s["computed"], name.split(".")[0]
    m = re.match(r"^(l\d\.real_time\.\S+) / (l\d\.real_time\.onsets_tested)$", c)
    if m:
        return S[m[1]]["value"] / S[m[2]]["value"]
    m = re.match(r"^step\.(\w+) / step\.perfect_forecast, (onsets warned|NSE)$", c)
    if m:
        sx = "onset_warned" if m[2].startswith("onsets") else "nse"
        return S[f"{lead}.step.{m[1]}.{sx}"]["value"] / S[f"{lead}.step.perfect_forecast.{sx}"]["value"]
    m = re.match(r"^step\.far\.forecast_over_rain_to_midnight / \(perfect rain minus rain to midnight\), (onsets warned|NSE)$", c)
    if m:
        sx = "onset_warned" if m[1].startswith("onsets") else "nse"
        return S[f"{lead}.step.far.forecast_over_rain_to_midnight.{sx}"]["value"] / S[f"{lead}.step.far.perfect_rain_over_rain_to_midnight.{sx}"]["value"]
    m = re.match(r"^highest minus lowest of (.+) \((l\d), (nse|onset_warned)\)$", c)
    if m:
        vals = [S[f"{m[2]}.{x.strip()}.{m[3]}"]["value"] for x in m[1].split(",")]
        return max(vals) - min(vals)
    raise ValueError(f"unknown form of computation: {c!r}")


for name, s in S.items():
    if "computed" in s:
        try:
            v = recompute(name, s)
        except Exception as e:                                                   # noqa
            C.append(dict(source=name, ok=False, why=repr(e)))
            continue
        ok, d = same(v, s["value"], 1e-12)
        C.append(dict(source=name, ok=bool(ok), diff=d))
out["C_computed_recomputed"] = dict(n=len(C), n_equal=sum(c["ok"] for c in C), largest_difference=max([c.get("diff", 0.0) for c in C if c["ok"]] or [0.0]), not_equal=[c for c in C if not c["ok"]])

# ---- D: the scores again from the stored forecasts -----------------------------------------------------------------------------------------------------------------------------------
D = F.load_data()
ids = list(D)
THR = high_flow_levels(D)
T0, T1 = pd.Timestamp("2005-10-01"), pd.Timestamp("2014-09-30")
CACHE = {}


def score(fname, lead):
    """My own scoring of a stored set: median NSE of the 12 basins and the onset table at trigger 1.0 (test years)."""
    key = (fname, lead)
    if key in CACHE:
        return CACHE[key]
    z = np.load(RES / "preds" / f"{fname}.npz", allow_pickle=False)
    assert list(z["basins"]) == ids
    dates = pd.DatetimeIndex(z["dates"])
    test = (dates >= T0) & (dates <= T1)
    nses, H, M, FA = [], 0, 0, 0
    for i, g in enumerate(ids):
        o, p = z["obs"][i], z["pred"][i, :, lead - 1]
        low = np.concatenate([np.full(lead, False), o[:-lead] <= THR[g]])
        ok = test & np.isfinite(o) & np.isfinite(p)
        oo, pp = o[ok], p[ok]
        nses.append(float(1 - np.sum((pp - oo) ** 2) / np.sum((oo - oo.mean()) ** 2)))
        hi, w, m = oo > THR[g], pp > THR[g], low[ok]
        H += int((hi & w & m).sum()); M += int((hi & ~w & m).sum()); FA += int((~hi & w & m).sum())
    CACHE[key] = dict(nse=float(np.median(nses)), onset_warned=H / (H + M), onset_false=FA / (H + FA) if H + FA else float("nan"), n_onsets=H + M)
    return CACHE[key]


Dres, no_file = [], []
for name, s in S.items():
    m = re.match(r"^l([123])\.((?:lstm|base|cmp)_\w+?)\.(nse|onset_warned|onset_false)(?:\.seed([012]))?$", name)
    if not m or "computed" in s:
        continue
    lead, setname, metric, seed = int(m[1]), m[2], m[3], m[4]
    fname = setname if seed is None else f"{setname}_s{seed}"
    if not (RES / "preds" / f"{fname}.npz").exists():
        no_file.append(name)
        continue
    mine = score(fname, lead)[metric]
    ok, d = same(mine, s["value"], 1e-9)
    Dres.append(dict(source=name, ok=bool(ok), mine=mine, stored=s["value"], diff=d, in_figure=bool(s["in_figure"])))
out["D_rescored_from_stored_forecasts"] = dict(n=len(Dres), n_equal=sum(d["ok"] for d in Dres), n_of_them_drawn_in_the_figure=sum(d["in_figure"] for d in Dres), largest_difference=max([d["diff"] for d in Dres if d["ok"]] or [0.0]),
                                               not_equal=[d for d in Dres if not d["ok"]], sets_scored=len({k[0] for k in CACHE}), no_stored_forecast_file_for=sorted(no_file)[:60], n_no_stored_forecast_file=len(no_file))

# ---- E: the printed numbers -----------------------------------------------------------------------------------------------------------------------------------------------------------
# What the three pictures print beside their bars (read by eye from figures/key_figure.png, key_figure_plain.png, key_figure_talk.png on 8 October 2026; R = full rendering, P = plain, T = talk).
LABELS = [
    ("0%", "l1.base_persistence.onset_warned", "R"), ("17%", "l1.lstm_w.onset_warned", "R"), ("19%", "l1.lstm_wq.onset_warned", "RPT"), ("30%", "l1.lstm_wqn.onset_warned", "RPT"),
    ("51%", "l1.lstm_wqa.onset_warned", "RPT"), ("45%", "l1.real_time.stage4_no_last_hour.onset_warned", "RPT"), ("48%", "l1.real_time.stage4.onset_warned", "RPT"), ("64%", "l1.lstm_wqf.onset_warned", "RPT"),
    ("0.43", "l1.base_persistence.nse", "R"), ("0.44", "l1.lstm_w.nse", "R"), ("0.59", "l1.lstm_wq.nse", "RPT"), ("0.67", "l1.lstm_wqn.nse", "RPT"), ("0.77", "l1.lstm_wqa.nse", "RPT"), ("0.86", "l1.lstm_wqf.nse", "RPT"),
    ("8%", "l2.lstm_wq.onset_warned", "RPT"), ("21%", "l2.lstm_wqa.onset_warned", "RPT"), ("39%", "l2.lstm_wqaF_sub.onset_warned", "RPT"), ("69%", "l2.lstm_wqaP.onset_warned", "RP"), ("67%", "l2.lstm_wqf.onset_warned", "RPT"),
    ("6%", "l3.lstm_wq.onset_warned", "RPT"), ("12%", "l3.lstm_wqa.onset_warned", "RPT"), ("37%", "l3.lstm_wqaF_sub.onset_warned", "RPT"), ("71%", "l3.lstm_wqaP.onset_warned", "RP"), ("68%", "l3.lstm_wqf.onset_warned", "RPT"),
    ("0.32", "l2.lstm_wq.nse", "RPT"), ("0.38", "l2.lstm_wqa.nse", "RPT"), ("0.53", "l2.lstm_wqaF_sub.nse", "RPT"), ("0.78", "l2.lstm_wqaP.nse", "RP"), ("0.82", "l2.lstm_wqf.nse", "RPT"),
    ("0.25", "l3.lstm_wq.nse", "RPT"), ("0.25", "l3.lstm_wqa.nse", "RPT"), ("0.45", "l3.lstm_wqaF_sub.nse", "RPT"), ("0.77", "l3.lstm_wqaP.nse", "RP"), ("0.78", "l3.lstm_wqf.nse", "RPT"),
]


def fmt(text, v):
    return f"{v * 100:.0f}%" if text.endswith("%") else f"{v:.2f}"


E = []
for text, src, where in LABELS:
    v = S[src]["value"]
    E.append(dict(printed=text, source=src, value=v, formatted=fmt(text, v), ok=fmt(text, v) == text, in_figure_flag=bool(S[src]["in_figure"]), renderings=where,
                  score_from_forecasts=score(src.split(".", 1)[1].rsplit(".", 1)[0], int(src[1]))[("onset_warned" if src.endswith("onset_warned") else "nse")] if src.split(".")[1] in ("lstm_wq", "lstm_w", "lstm_wqn", "lstm_wqa", "lstm_wqf", "lstm_wqaF_sub", "lstm_wqaP", "base_persistence") else None))
for e in E:
    if e["score_from_forecasts"] is not None:
        e["forecast_score_formats_to_the_printed_number"] = fmt(e["printed"], e["score_from_forecasts"]) == e["printed"]
# the stage IV diamonds come from stage4_realtime.json (294 and 275 of 611): checked in d3
n_by_rendering = {r: sum(r in e["renderings"] for e in E) for r in "RPT"}

# lettering and caption numbers
SS, FD, ST = K["steps"], K["finding"], K["steps"]
lettering = [
    ("611 onsets", K["n_onsets"]["l1"], 611), ("951 onsets", K["n_onsets"]["l2"], 951), ("1,116 onsets", K["n_onsets"]["l3"], 1116),
    ("8 points apart", round(FD["spread_between_model_kinds"]["onset_warned_points"]), 8), ("0.03 NSE apart", round(FD["spread_between_model_kinds"]["nse"], 2), 0.03),
    ("higher in only 5 of 12 basins", SS["rain_up_to_evening"]["n_basins_better"], 5), ("its LSTM twin: 18%", round(S["l1.cmp_lstm.onset_warned"]["value"] * 100), 18), ("its LSTM twin: 0.60", round(S["l1.cmp_lstm.nse"]["value"], 2), 0.60),
    ("GO: at least 70%", round(K["criteria"]["go_min_warned"] * 100), 70), ("12 basins", K["n_basins"], 12),
    ("caption: +11 points (rain up to the evening)", round(SS["rain_up_to_evening"]["onset_warned_points"]), 11), ("caption: +33 points (AORC over the rain table)", round(S["l1.step.aorc_over_rain_table.onset_warned"]["value"] * 100), 33),
    ("caption: +46 points (perfect forecast)", round(SS["perfect_forecast"]["onset_warned_points"]), 46), ("caption: +0.08 NSE (rain up to the evening)", round(SS["rain_up_to_evening"]["nse"], 2), 0.08),
]
out["E_drawn_values"] = dict(
    labels_read=len(E), labels_per_rendering=n_by_rendering, labels_equal_to_the_formatted_source=sum(e["ok"] for e in E), labels_whose_source_is_flagged_in_figure=sum(e["in_figure_flag"] for e in E),
    labels_with_a_score_recomputed_from_forecasts=sum(e["score_from_forecasts"] is not None for e in E), of_them_formatting_to_the_printed_number=sum(bool(e.get("forecast_score_formats_to_the_printed_number")) for e in E),
    labels=E, lettering=[dict(what=w, mine=m, printed=p, ok=bool(abs(m - p) < 1e-9)) for w, m, p in lettering],
    mixed_labels=dict(printed_beside=["0.67 (1 day, rain up to the evening)", "0.53 (2 days, GEFS)", "0.45 (3 days, GEFS)"],
                      stored_readings=dict(rain_up_to_evening=SS["rain_up_to_evening"]["nse_in_words"], gefs_2_days=K["far"]["l2"]["steps"]["forecast_over_rain_table"]["nse_in_words"],
                                           gefs_3_days=K["far"]["l3"]["steps"]["forecast_over_rain_table"]["nse_in_words"]),
                      not_labelled_because=dict(aorc_2_days=K["far"]["l2"]["steps"]["rain_to_midnight"]["nse_in_words"], aorc_3_days=K["far"]["l3"]["steps"]["rain_to_midnight"]["nse_in_words"],
                                                aorc_1_day=SS["aorc_over_rain_table"]["nse_in_words"], perfect_1_day=SS["perfect_forecast"]["nse_reading"])))
# the OCR of the three pictures, when supplied
ocr_dir = os.environ.get("FIGURE_OCR_DIR")
if ocr_dir and Path(ocr_dir).exists():
    ocr = {}
    for r, f in (("R", "key_figure.json"), ("P", "key_figure_plain.json"), ("T", "key_figure_talk.json")):
        lines = json.load(open(Path(ocr_dir) / f))
        ocr[r] = " | ".join(l["text"] for l in lines)
    found, missing = {r: 0 for r in "RPT"}, {r: [] for r in "RPT"}
    for e in E:
        pat = re.compile(r"(?<![\d.])" + re.escape(e["printed"]) + r"(?![\d])")
        for r in e["renderings"]:
            if pat.search(ocr[r]):
                found[r] += 1
            else:
                missing[r].append(f"{e['printed']} ({e['source']})")
    out["E_drawn_values"]["ocr_cross_check"] = dict(tool="Apple Vision VNRecognizeTextRequest (ocr.swift)", found_in_the_ocr_text=found, labels_expected=n_by_rendering, not_found_in_the_ocr_text=missing,
                                                    note="a label that the OCR did not read was read by eye on the picture; OCR misreads sometimes (for example 0.53 read as 0.58)")

# ---- F: counts and hashes ----------------------------------------------------------------------------------------------------------------------------------------------------------------
chk = K["checks"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out["F_counts_and_hashes"] = dict(
    sources=len(S), with_file_and_key=sum("file" in s for s in S.values()), computed=sum("computed" in s for s in S.values()), flagged_in_figure=sum(bool(s["in_figure"]) for s in S.values()),
    builder_says=dict(compared=chk["compared"], tolerance=chk["tolerance"], largest_difference=chk["largest_difference"], disagreements=len(chk["disagreements"]), flagged_as_in_the_figure=chk["drawn_numbers"]["flagged_as_in_the_figure"],
                      numbers_drawn_or_printed=chk["drawn_numbers"]["numbers_drawn_or_printed"], problems=chk["drawn_numbers"]["problems"]),
    flagged_matches_the_builders_count=sum(bool(s["in_figure"]) for s in S.values()) == chk["drawn_numbers"]["flagged_as_in_the_figure"],
    my_comparisons=out["A_values_read_again"]["n"] + out["B_cross_checks_read_again"]["n"] + out["C_computed_recomputed"]["n"] + out["D_rescored_from_stored_forecasts"]["n"] + len(E) + len(lettering),
    sha256={p: sha(PROJ / p) for p in ("results/key_figure.json", "figures/key_figure.png", "figures/key_figure_plain.png", "figures/key_figure_talk.png")},
    files_modified=dict(zip(("results/key_figure.json", "figures/key_figure.png", "figures/key_figure_plain.png", "figures/key_figure_talk.png"),
                            [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((PROJ / p).stat().st_mtime)) for p in ("results/key_figure.json", "figures/key_figure.png", "figures/key_figure_plain.png", "figures/key_figure_talk.png")])))
out["seconds"] = round(time.time() - t0, 1)
jsave("d5_key_figure.json", out)
for k in ("A_values_read_again", "B_cross_checks_read_again", "C_computed_recomputed", "D_rescored_from_stored_forecasts"):
    v = out[k]
    print(f"{k}: {v['n_equal']} of {v['n']} equal, largest difference {v['largest_difference']:.3g}; not equal: {len(v['not_equal'])}")
e = out["E_drawn_values"]
print(f"E: {e['labels_equal_to_the_formatted_source']} of {e['labels_read']} printed labels equal the formatted source; recomputed from forecasts: {e['of_them_formatting_to_the_printed_number']} of {e['labels_with_a_score_recomputed_from_forecasts']}; lettering ok: {sum(x['ok'] for x in e['lettering'])} of {len(e['lettering'])}")
print("F:", {k: v for k, v in out["F_counts_and_hashes"].items() if k not in ("sha256", "files_modified", "builder_says")})
if "ocr_cross_check" in e:
    print("OCR:", e["ocr_cross_check"]["found_in_the_ocr_text"], e["ocr_cross_check"]["not_found_in_the_ocr_text"])
