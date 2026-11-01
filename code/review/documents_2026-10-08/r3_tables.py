"""r3_tables: Tables 1 and 2 of the three-page report (Flood_Report_3pages.pdf, page 2) recomputed with the critic's own scoring from the stored forecasts and results/uncertainty.json.

Each row of the tables prints, for one prediction set and lead time: the onsets warned of by the average of three trainings (large), the lowest-highest single training and the 95% range when the 9 test years and the 12 basins are
drawn again; the same for the median NSE of the 12 basins; and for 1 day ahead the share of the onset warnings that were false. This script computes every one of these numbers (the average forecast = results/preds/<set>.npz,
the single trainings = <set>_s0..2.npz, scored by my own functions as in d2 and d5; the 95% ranges are read from results/uncertainty.json, which d2 has redrawn) in the way the report prints them and looks for the row as printed in the text of page 2.
Input: R3_TEXT_DIR with Flood_Report_3pages.txt (pdf2txt: pages split by '=====PAGE n====='). Output: OUT/r3_tables.json
"""
from dcommon import *
import flood as F
import re

D = F.load_data()
ids = list(D)
THR = high_flow_levels(D)
T0, T1 = pd.Timestamp("2005-10-01"), pd.Timestamp("2014-09-30")
U = jload("uncertainty")["stats"]
CACHE = {}


def score(fname, lead, t0=T0, t1=T1):
    """Median NSE of the 12 basins and the onset table (trigger 1.0) of a stored set."""
    key = (fname, lead, t0, t1)
    if key in CACHE:
        return CACHE[key]
    z = np.load(RES / "preds" / f"{fname}.npz", allow_pickle=False)
    dates = pd.DatetimeIndex(z["dates"])
    test = (dates >= t0) & (dates <= t1)
    nses, H, M, FA = [], 0, 0, 0
    for i, g in enumerate(ids):
        o, p = z["obs"][i], z["pred"][i, :, lead - 1]
        low = np.concatenate([np.full(lead, False), o[:-lead] <= THR[g]])
        ok = test & np.isfinite(o) & np.isfinite(p)
        oo, pp = o[ok], p[ok]
        nses.append(float(1 - np.sum((pp - oo) ** 2) / np.sum((oo - oo.mean()) ** 2)))
        hi, w, m = oo > THR[g], pp > THR[g], low[ok]
        H += int((hi & w & m).sum()); M += int((hi & ~w & m).sum()); FA += int((~hi & w & m).sum())
    CACHE[key] = dict(nse=float(np.median(nses)), pod=H / (H + M), far=FA / (H + FA) if H + FA else float("nan"), n=H + M)
    return CACHE[key]


def seeds(name, lead, t0=T0, t1=T1):
    s = [score(f"{name}_s{k}", lead, t0, t1) for k in (0, 1, 2)]
    return s


pc = lambda x: f"{100 * x:.0f}"
n2 = lambda x: f"{x:.2f}"


def rng(vals, f):
    a, b = f(min(vals)), f(max(vals))
    return a if a == b else f"{a}-{b}"


def unc(stat, f):
    u = U[stat]["both"]
    return f"{f(u['lo'])}-{f(u['hi'])}"


def cells(name, lead, with_unc=True):
    """The printed cells of one set and lead: (onsets warned, its seed range, its 95% range, median NSE, NSE seed range, NSE 95% range, false share)."""
    m, ss = score(name, lead), seeds(name, lead)
    out = dict(pod=f"{pc(m['pod'])}%", pod_seeds=rng([x["pod"] for x in ss], pc), nse=n2(m["nse"]), nse_seeds=rng([x["nse"] for x in ss], n2), far=f"{pc(m['far'])}%")
    if with_unc:
        out.update(pod_unc=unc(f"onsets_warned.{name}.l{lead}", pc) + "%", nse_unc=unc(f"nse_median.{name}.l{lead}", n2))
    return out


ROWS_1 = [  # (label in the report, set, extra text after the NSE cells)
    ("Rain table alone (past weather + river gauge)", "lstm_wq", ""), ("+ NLDAS rain, day ends about 17:30", "lstm_wqn", " mixed"), ("+ ERA5 rain to midnight (model rain)", "lstm_wqh", ""),
    ("+ AORC rain to midnight, an analysis made after the fact", "lstm_wqa", ""), ("control: AORC over the table's window, 09:00 to 09:00", "lstm_wqa_to09", ""), ("control: AORC over the UTC day, ends about 19:00", "lstm_wqa_utc", ""),
    ("Perfect weather forecast", "lstm_wqf", "")]
ROWS_2 = [("Rain table alone", "lstm_wq"), ("+ AORC to midnight, an analysis made after the fact", "lstm_wqa"), ("+ GEFS forecast, next three days, substituted (E2a)", "lstm_wqaF_sub"), ("perfect rain of the next three days", "lstm_wqaP"), ("perfect weather of the next three days", "lstm_wqf")]

import os
txt = open(Path(os.environ["R3_TEXT_DIR"]) / "Flood_Report_3pages.txt").read()
parts = re.split(r"\n=====PAGE (\d+)=====\n", "\n" + txt)
P2 = re.sub(r"\s+", " ", parts[4]).replace("–", "-").replace("−", "-")
res, bad = [], []
for label, name, extra in ROWS_1:
    c = cells(name, 1)
    exp = f"{c['pod']} {c['pod_seeds']} · {c['pod_unc']} {c['nse']} {c['nse_seeds']} · {c['nse_unc']}{extra} {c['far']}"
    found = exp in P2
    res.append(dict(row=label, set=name, lead=1, expected=exp, found=found))
    if not found:
        bad.append(res[-1])
# the two simple models, given the AORC rain (printed with their own ranges)
lin, rf = score("base_linreg_wqa", 1), score("base_rf_wqa", 1)
exp = f"{pc(lin['pod'])}% / {pc(rf['pod'])}% {unc('onsets_warned.base_linreg_wqa.l1', pc)}% / {unc('onsets_warned.base_rf_wqa.l1', pc)}% {n2(lin['nse'])} / {n2(rf['nse'])} {pc(lin['far'])}% / {pc(rf['far'])}%"
res.append(dict(row="Ridge regression / random forest, given the AORC rain", expected=exp, found=exp in P2))
# Stage IV
S4 = jload("stage4_realtime")["models"]["wqa"]["variants"]
exp = f"{pc(S4['stage4']['share_of_the_onsets'])}% {pc(S4['stage4_no_last_hour']['share_of_the_onsets'])}% with the last hour out"
res.append(dict(row="Stage IV in place of AORC", expected=exp, found=exp in P2))
for label, name in ROWS_2:
    c1, c2 = cells(name, 2), cells(name, 3)
    exp = f"{c1['pod']} {c1['pod_seeds']} · {c1['pod_unc']} {c2['pod']} {c2['pod_seeds']} · {c2['pod_unc']} {c1['nse']} {c1['nse_seeds']} · {c1['nse_unc']} {c2['nse']} {c2['nse_seeds']} · {c2['nse_unc']}"
    found = exp in P2
    res.append(dict(row=label, set=name, lead="2,3", expected=exp, found=found))
# E2b (another split): stored in forecast_input.json: the retrained sets of the shorter split
FI = jload("forecast_input")
res_e2b = {}
for k in (1, 2, 3):
    r = FI["summary_test"]["sets"]["wqaF_s"][f"l{k}"]
    res_e2b[k] = dict(pod=r["onset"]["pod"], nse=r["nse_median"])
exp_e2b = [f"{pc(res_e2b[2]['pod'])}%", f"{pc(res_e2b[3]['pod'])}%", n2(res_e2b[2]["nse"]), n2(res_e2b[3]["nse"])]
res.append(dict(row="E2b (another split)", expected=exp_e2b, found=all(e in P2 for e in exp_e2b), note="the 1-day value 61% is given in the footnote: " + pc(res_e2b[1]["pod"]) + "%"))
jsave("r3_tables.json", dict(rows=res, n_rows=len(res), n_found=sum(r["found"] for r in res), not_found=[r for r in res if not r["found"]]))
print(f"{sum(r['found'] for r in res)} of {len(res)} rows of Tables 1 and 2 found as printed")
for r in res:
    if not r["found"]:
        print("NOT FOUND:", r["row"], "| expected:", r["expected"])
