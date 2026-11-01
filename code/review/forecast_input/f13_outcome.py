"""E2 review, check (h): outcome, outcome_caveat (seven clauses) and outcome_full of results/forecast_input.json rebuilt, clause by clause, as sentences from the reviewer's own numbers (own onset counts on each set's own split, own NSE,
own closures, own Wilcoxon and sign tests, own verdict) and compared with the text of the file. The numbers behind the clause about the simple models (what the forecast adds to ridge and forest, the reading of the LSTM against
them at 1 day by the NSE rule of report section 4.2 and by a paired sign test of the onsets), the clause about the clock (the margin of the last needed file at the issue time, recomputed from the posting times of
results/data_access.json) and the numbers quoted from the E1b review (checked against results/review_aorc_rain_to_midnight.json) are recomputed too. Then the arithmetic of the clauses is judged: which sentence says more than the numbers
support. About 30 seconds."""
import sys; sys.dont_write_bytecode = True
import re, math
from scipy.stats import wilcoxon, binomtest
from f_common import *
import flood as F
from f_scoring import *

R = jload("results/forecast_input.json"); G = jload("results/gefs_basins.json")
D = F.load_data(); ids = list(D); n_b = len(ids)
SH = SETS_E2 + ["wq", "wqa", "wqf"]
CNT, NSE_B, NSE_S, POD, FL = {}, {}, {}, {}, {}
for n in SH:
    for k in KS:
        CNT[n, k] = counts(f"lstm_{n}", k); nb = nse_per_basin(f"lstm_{n}", k); NSE_B[n, k] = np.array([nb[g] for g in ids])
        NSE_S[n, k] = [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", k).values()))) for s in SEEDS]; POD[n, k] = [counts(f"lstm_{n}_s{s}", k)["pod"] for s in SEEDS]
        t = onset_table(f"lstm_{n}", k); FL[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
SIMPLE = ["base_linreg_wqa", "base_rf_wqa", "base_linreg_wqaF", "base_rf_wqaF", "base_linreg_wq", "base_rf_wq"]
for n in SIMPLE:
    for k in KS:
        CNT[n, k] = counts(n, k); nb = nse_per_basin(n, k); NSE_B[n, k] = np.array([nb[g] for g in ids]); t = onset_table(n, k); FL[n, k] = np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
med = lambda n, k: float(np.median(NSE_B[n, k]))
above = lambda a, b: min(a) > max(b)
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"
p1 = lambda x: f"{100 * x:.1f}%"
on = lambda n: "/".join(pc(CNT[n, k]["pod"]) for k in KS)
ns = lambda n: "/".join(f"{med(n, k):.2f}" for k in KS)
yes = lambda b: "yes" if b else "no"
words = lambda v: f"{v[0]}, {v[1]} and {v[2]}"
pts = lambda T, C, k: 100 * (CNT[T, k]["pod"] - CNT[C, k]["pod"])
def onr(T, C, k): return "clear gain" if above(POD[T, k], POD[C, k]) else "clear loss" if above(POD[C, k], POD[T, k]) else "no clear difference"
def nser(T, C, k):
    n_hi = int((NSE_B[T, k] > NSE_B[C, k]).sum()); n_lo = int((NSE_B[T, k] < NSE_B[C, k]).sum())
    try: p = float(wilcoxon(NSE_B[T, k], NSE_B[C, k]).pvalue)
    except ValueError: p = float("nan")
    up, down = above(NSE_S[T, k], NSE_S[C, k]), above(NSE_S[C, k], NSE_S[T, k]); small = p == p and p < 0.05
    gain, loss = up and n_hi > n_b / 2 and small, down and n_lo > n_b / 2 and small
    return "clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"
# verdict, own application of the criteria at the plain level
crit = R["verdicts"]["criteria"]; pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
c1_ = CNT["wqaF_sub", 1]; V = "GO" if (c1_["pod"] >= pod_go and c1_["far"] <= far_go) else "CAUTION" if c1_["csi"] >= csi_c else "NOT YET"
X = "the GEFS control-member rain forecast for the next three days (bias-corrected) as an input beside the rain to midnight"
outcome = f"Given {X}, onsets warned 1/2/3 days ahead go from {on('wqa')} to {on('wqaF_sub')} (control {on('wqaF_clim')}); median NSE from {ns('wqa')} to {ns('wqaF_sub')} (control {ns('wqaF_clim')}); the verdict is {V}."
cl = lambda S, b, c, k: ((CNT[S, k]["pod"] - CNT[b, k]["pod"]) / (CNT[c, k]["pod"] - CNT[b, k]["pod"]), (med(S, k) - med(b, k)) / (med(c, k) - med(b, k)))
c1 = (f"Against lstm_wqa, the same network without the forecast, the forecast adds {words([format(pts('wqaF_sub', 'wqa', k), '+.1f') for k in KS])} points of onsets warned at 1, 2 and 3 days ahead (reading of the onsets by the seed rule, every single "
      f"training of one above every single training of the other: {words([onr('wqaF_sub', 'wqa', k) for k in KS])}) and {words([format(med('wqaF_sub', k) - med('wqa', k), '+.3f') for k in KS])} of median NSE (reading of the NSE by the rule of report section 4.2: {words([nser('wqaF_sub', 'wqa', k) for k in KS])}).")
c2 = (f"The control, the same forecasts shuffled within basin and calendar month, warns of {on('wqaF_clim')} and has a median NSE of {ns('wqaF_clim')}: the forecast is {words([format(pts('wqaF_sub', 'wqaF_clim', k), '+.1f') for k in KS])} points above it (every training above: "
      f"{words([yes(above(POD['wqaF_sub', k], POD['wqaF_clim', k])) for k in KS])}), and the control itself lies {words([format(pts('wqaF_clim', 'wqa', k), '+.1f') for k in KS])} points from lstm_wqa, so a network that was trained on perfect rain and trusts its slots "
      f"{'loses' if pts('wqaF_clim', 'wqa', 1) < 0 else 'does not lose'} skill when the slots hold a forecast with no information about the day.")
c3 = (f"Of the gap in onsets warned between lstm_wqa and a perfect rain forecast of the three days (lstm_wqaP warns of {on('wqaP')}) the forecast closes {words([pc(cl('wqaF_sub', 'wqa', 'wqaP', k)[0]) for k in KS])} at 1, 2 and 3 days ahead, and "
      f"{words([pc(cl('wqaF_sub', 'wqa', 'wqaP', k)[1]) for k in KS])} of the NSE gap (a negative share: the forecast model is below lstm_wqa); against the old perfect-weather upper bound (lstm_wqf: {on('wqf')}) it closes {words([pc(cl('wqaF_sub', 'wqa', 'wqf', k)[0]) for k in KS])} of the onset gap beyond lstm_wqa.")
c4 = (f"Retrained with the forecast as an input on the shorter satellite-era split (train 2000-2008, test 2010-2014; its own scores, NOT comparable with the main split), onsets warned go from {on('wqa_s')} (no forecast) to {on('wqaF_s')} "
      f"(control {on('wqaF_clim_s')}; perfect rain {on('wqaP_s')}), {words([format(pts('wqaF_s', 'wqa_s', k), '+.1f') for k in KS])} points, against {words([format(pts('wqaF_sub', 'wqa', k), '+.1f') for k in KS])} for the substitution on the main split.")
c5 = f"On the plain gauge model, without the rain to midnight, the same forecast takes onsets warned from {on('wq')} to {on('wqF_sub')} ({words([format(pts('wqF_sub', 'wq', k), '+.1f') for k in KS])} points)."
# c6: the yardstick
gain_pts = lambda new, old, k: 100 * (CNT[new, k]["pod"] - CNT[old, k]["pod"])
slash = lambda f: "/".join(f(k) for k in KS)
g_ridge = slash(lambda k: f"{gain_pts('base_linreg_wqaF', 'base_linreg_wqa', k):+.1f}"); g_rf = slash(lambda k: f"{gain_pts('base_rf_wqaF', 'base_rf_wqa', k):+.1f}"); g_lstm = slash(lambda k: f"{pts('wqaF_sub', 'wqa', k):+.1f}")
simple_gain_everywhere = all(gain_pts(f"base_{m}_wqaF", f"base_{m}_wqa", k) > 0 and (med(f"base_{m}_wqaF", k) - med(f"base_{m}_wqa", k)) > 0 for m in ("linreg", "rf") for k in KS)
def vs_simple(simple):
    T, k = "wqaF_sub", 1
    a, b = NSE_B[T, k], NSE_B[simple, k]; bm = float(np.median(b)); sm = NSE_S[T, k]
    n_hi = int((a > b).sum()); small = float(wilcoxon(a, b).pvalue) < 0.05; below = bm < min(sm)
    most, few = n_hi > n_b / 2, n_hi < n_b / 2
    r = ("lstm" if below else "lstm_avg") if (most and small) else ("simple" if (few and small) else ("lstm_med" if (most and below) else "tie"))
    words_ = {"lstm": "the LSTM is ahead", "lstm_avg": "the LSTM is ahead only as the average of its seeds", "lstm_med": "the LSTM scores higher, but p is above 0.05", "tie": "no clear difference", "simple": "the simple model is ahead"}[r]
    fa, fc = FL[T, k], FL[simple, k]; oa, oc = int((fa & ~fc).sum()), int((~fa & fc).sum()); psign = float(binomtest(oa, oa + oc, 0.5).pvalue)
    every_above, every_below = min(POD[T, k]) > CNT[simple, k]["pod"], max(POD[T, k]) < CNT[simple, k]["pod"]
    on_read = "lstm" if every_above and psign < 0.05 else "simple" if every_below and psign < 0.05 else "tie"
    return dict(nse_reading=r, nse_reading_words=words_, n_basins_higher=n_hi, onset_reading={"lstm": "the LSTM warns of more onsets", "simple": "the simple model warns of more onsets", "tie": "no clear difference in the onsets"}[on_read], only_lstm=oa, only_simple=oc, sign_p=psign)
vr, vf = vs_simple("base_linreg_wqaF"), vs_simple("base_rf_wqaF")
s_ridge, s_rf = on("base_linreg_wqaF"), on("base_rf_wqaF")
far = lambda n: slash(lambda k: pc(CNT[n, k]["far"]))
c6 = (f"Ridge regression and the random forest given the same inputs (the forecast substituted in the same way) warn of {s_ridge} and {s_rf} of the onsets at 1/2/3 days ahead, with a median NSE of {ns('base_linreg_wqaF')} and {ns('base_rf_wqaF')} "
      f"(the LSTM: {on('wqaF_sub')}, {ns('wqaF_sub')}; the forecast adds {g_ridge} and {g_rf} points of onsets to them, {g_lstm} to the LSTM; false warnings among the onset warnings: ridge {far('base_linreg_wqaF')}, forest {far('base_rf_wqaF')}, LSTM {far('wqaF_sub')}, so the simple models warn of more onsets with more false warnings); "
      f"at 1 day ahead, by the paired reading of the onsets: ridge, {vr['onset_reading'].replace('the LSTM warns of more onsets', 'the LSTM warns of more onsets')}; forest, {vf['onset_reading']}; by the NSE (report section 4.2): "
      f"against ridge, {vr['nse_reading_words']}; against the forest, {vf['nse_reading_words']} - " + ("the gain of the forecast, in onsets and in NSE at every lead, does not need the network." if simple_gain_everywhere else "the simple models do not gain at every lead."))
# c7: the caveats; the margin from the posting times
A = jload("results/data_access.json")["gefs_reforecast_record"]["task5_issue_time"]["operational_gefs_posting_latency_gec00_0p25"]
hours = sorted({int(h[1:]) for d in A.values() for h in d}); late = {h: max(d[f"f{h:03d}"] for d in A.values()) for h in hours}
margin = {}
for zone, runs in (("America/New_York", (4, 5)), ("America/Chicago", (5, 6))):
    for age in runs:
        hi3 = age + 72                                                     # the end of local day t+3 in forecast hours on an ordinary day (23/25-hour days move it by one)
        last_file = int(math.ceil(hi3 / 3) * 3); late_files = [h for h in hours if h <= last_file and late[h] > age]
        margin[f"{zone}, run {age} h old at the issue time"] = dict(last_file=f"f{last_file:03d}", margin_minutes=round(60 * (age - late[last_file]), 1), n_files_posted_after_the_issue_time=len(late_files), first=f"f{min(late_files):03d}" if late_files else None, worst_lateness_minutes=round(60 * max([late[h] - age for h in late_files] + [0]), 1))
mg_j = G["realtime_margin"]["by_zone_and_clock"]
mg_cmp = {k: {f: (v[f], mg_j[k][{"last_file": "last_file_needed", "margin_minutes": "margin_minutes_of_the_last_file", "n_files_posted_after_the_issue_time": "files_needed_that_are_posted_after_the_issue_time", "first": "first_such_file", "worst_lateness_minutes": "worst_lateness_minutes"}[f]]) for f in v} for k, v in margin.items()}
print("the real-time margin of the run, from the posting times of the OPERATIONAL control member (4 days of 2026), mine against gefs_basins.json:")
for k, v in mg_cmp.items(): print(f"   {k}: {v}")
areas = [G["basins"][g]["outline_area_km2_equal_area"] for g in ids]
bias_med = float(np.median([G["bias"]["factors"][g]["l1"] for g in ids]))
mg_e4 = margin["America/New_York, run 4 h old at the issue time"]
rv = jload("results/review_aorc_rain_to_midnight.json")
t1 = rv["checks"]; d3 = next(c for c in t1 if c["id"] == "d3")["numbers"]; d5 = next(c for c in t1 if c["id"] == "d5")["numbers"]
quote_ok = dict(coefficient=d3["coefficient_on_stage_iv_after_midnight_mm_per_mm"]["whole day t, both"], onsets_in_sample=d5["onsets_in_the_sample"], warned=d5["wqa_onsets_warned"])
print("numbers quoted from the E1b review in the caveat: coefficient (95%) -0.03 (-0.06, -0.001); 86 to 95 of 171 against 92; 440 of 611 -> the review record says:", quote_ok)
c7 = (f"Caveats that are part of the result: control member only (no ensemble); 0.25 degree boxes over basins of {min(areas):.0f} to {max(areas):,.0f} km2; the bias factor (one per basin and lead, median {bias_med:.2f}) was fitted on the validation years 2000-2005; "
      "the substitution networks never saw a forecast in training; the rain to midnight of the base rung is the AORC analysis, made after the fact (corrections about 10 days after the hour; hours before 2002 are disaggregated daily totals), not rain that could be read at midnight - "
      "the independent review of E1b tested the hindsight risk on 78 test-year days (coefficient of the pre-midnight AORC rain on the later Stage IV rain about -0.03, 95% interval -0.06 to -0.001: nothing positive) and found that with Stage IV, a real-time product, in its place for the "
      "issue day the same networks warned of 86 to 95 of the 171 onsets of those days against 92 with AORC; the other 440 of the 611 onsets were not tested (numbers as the review's verdict states them: results/review_aorc_rain_to_midnight.json); "
      f"in Eastern summer the last files of the run (from forecast hour {int(mg_e4['first'][1:])} on) are posted up to {mg_e4['worst_lateness_minutes']:.0f} minutes after the issue time (results/gefs_basins.json, realtime_margin); "
      "three trainings per set, so the single-training ranges are the only repetition.")
cav = " ".join([c1, c2, c3, c4, c5, c6, c7]); full = outcome + " " + cav
res = dict(outcome_equal=(R["outcome"] == outcome), caveat_equal=(R["outcome_caveat"] == cav), full_equal=(R["outcome_full"] == full), clauses_equal={f"c{i}": (c in R["outcome_caveat"]) for i, c in enumerate([c1, c2, c3, c4, c5, c6, c7], 1)})
print("\noutcome rebuilt from my own numbers equals the file:", res["outcome_equal"], "| outcome_caveat:", res["caveat_equal"], "| outcome_full:", res["full_equal"], "| each clause found verbatim in the file:", res["clauses_equal"])
if not res["caveat_equal"]:
    for i, c in enumerate([c1, c2, c3, c4, c5, c6, c7], 1):
        if c not in R["outcome_caveat"]: print(f"  CLAUSE c{i} differs:\n   MINE: {c[:600]}\n   FILE: {R['outcome_caveat'][:300]}")
out = dict(res, verdict_lead1_plain=V, yardstick=dict(ridge=vr, forest=vf, gains_points=dict(ridge=g_ridge, forest=g_rf, lstm=g_lstm), simple_gain_everywhere=bool(simple_gain_everywhere)), margin=dict(mine=margin, equal_json=all(v[f][0] == v[f][1] for v in mg_cmp.values() for f in v)), review_numbers=quote_ok)
# the numbers the sentences rest on, for the judgement of the clauses
out["numbers"] = dict(onsets={n: [round(100 * CNT[n, k]["pod"], 1) for k in KS] for n in ("wq", "wqa", "wqaF_sub", "wqaF_clim", "wqaP", "wqf", "wqF_sub", "wqa_s", "wqaF_s", "wqaF_clim_s", "wqaP_s", "base_linreg_wqaF", "base_rf_wqaF", "base_linreg_wqa", "base_rf_wqa")},
                      false_warnings={n: [round(100 * CNT[n, k]["far"], 0) for k in KS] for n in ("wqa", "wqaF_sub", "base_linreg_wqaF", "base_rf_wqaF")}, nse={n: [round(med(n, k), 3) for k in KS] for n in ("wqa", "wqaF_sub", "base_linreg_wqaF", "base_rf_wqaF", "wqaF_s", "wqa_s")},
                      csi={n: [round(CNT[n, k]["csi"], 3) for k in KS] for n in ("wqa", "wqaF_sub", "base_linreg_wqaF", "base_rf_wqaF")}, n_onsets=[CNT["wqa", k]["n_onsets"] for k in KS], n_onsets_sat=[CNT["wqa_s", k]["n_onsets"] for k in KS])
print("csi (lead 1/2/3): wqa", out["numbers"]["csi"]["wqa"], "wqaF_sub", out["numbers"]["csi"]["wqaF_sub"], "ridge+F", out["numbers"]["csi"]["base_linreg_wqaF"], "forest+F", out["numbers"]["csi"]["base_rf_wqaF"])
jsave("f13_outcome.json", out)
