"""E2 review, check (h) continued: the statements about E2 in ROADMAP.md (section 'Step 2 result', written by another agent after the results file) against the reviewer's numbers. ROADMAP.md is not on the brief's list of files under
review; it is the place where the result is told to the team, and the E1b review found wording problems there. Each statement is quoted (shortened), the reviewer's number is computed here (from the outputs of the other scripts of this
folder and the saved prediction sets) and the result is 'agrees', 'agrees to rounding' or 'differs / overstated' with a reason. About 5 seconds."""
import sys; sys.dont_write_bytecode = True
import re
from f_common import *
from f_scoring import *

H8 = jread("f8_headline.json"); F4 = jread("f4_basins.json"); F5 = jread("f5_skill.json"); F12 = jread("f12_interpretation.json"); F9 = jread("f9_readings.json")
RM = open(f"{PROJ}/ROADMAP.md").read()
i0 = RM.index("### Step 2 result"); i1 = RM.index("### Step 3")
SEC = RM[i0:i1]
own = H8["own"]; eng = H8["engine"]
pod = lambda n, k: 100 * own[n][f"l{k}"]["pod"]
nse = lambda n, k: eng[n][f"l{k}"]["nse_median"]
rows = []
def claim(text, mine, status, why=""):
    rows.append(dict(statement=text, reviewer=mine, status=status, why=why))
    print(f"[{status:18s}] {text}\n      reviewer: {mine}" + (f"\n      why: {why}" if why else ""))

# data paragraph
nb = [v["n_boxes_mine"] for v in F4["weights"].values()]
claim("exact cell overlaps (1 to 9 cells per basin)", f"{min(nb)} to {max(nb)} boxes per basin", "agrees" if (min(nb), max(nb)) == (1, 9) else "differs")
claim("4 to 6 hours old at the issue time", f"ages {F4['rule_proven']['ages_h']}", "agrees")
claim("correlation 0.72 / 0.66 / 0.59 at 1 / 2 / 3 days", [round(F5['skill'][f'l{k}']['corr_all_median'], 3) for k in KS], "agrees to rounding")
# seasonal ratios from my own series
z = np.load(f"{OUT}/f4_series.npz", allow_pickle=True); days = pd.to_datetime(z["days"].astype(str)); raw, aorc = z["raw"], z["aorc"]
def ratio(months, k):
    m = np.isin(days.month, months)
    return float(np.median([np.nansum(raw[b, m, k]) / np.nansum(aorc[b, m, k]) for b in range(raw.shape[0])]))
rs = {n: [round(ratio(ms, k), 3) for k in range(3)] for n, ms in (("summer JJA", [6, 7, 8]), ("winter DJF", [12, 1, 2]), ("all", list(range(1, 13))))}
claim("mean ratio about 1.0 (summer 1.12, winter 0.88; Florida basins 1.2-1.4)", dict(median_over_basins_of_forecast_total_over_AORC_total=rs, florida_bias_factors_lead1=[round(F5['bias']['factors_mine'][g]['l1'], 3) for g in ("02296500", "02298608")]),
      "agrees to rounding" if abs(rs["summer JJA"][0] - 1.12) < 0.03 and abs(rs["winter DJF"][0] - 0.88) < 0.03 else "differs", "ratio of totals of the raw forecast to AORC by season (lead 1); Florida factors 1.19 and 1.38 at 1 day (1.19-1.41 over the leads)")
bs = [round(F5["brier"][f"l{k}_5mm"]["bss_calibrated_isotonic_pooled"], 3) for k in KS]
claim("calibrated Brier skill for 5-mm days 0.44 / 0.39 / 0.33 - the same as the published GEFSv12 numbers (0.42 and 0.33, results/published_context.json)", dict(mine_pooled=bs, builder_text_in_gefs_basins_json="Guan et al. quote Brier skill 0.42 and 0.33 for the probabilities of the WHOLE ENSEMBLE (11 members, 1 degree grid, CCPA analysis, 24-h totals); the numbers here are of another kind (one member, a map fitted on the validation years, the sample's own frequency as climatology, these basins and years) and are not a check of that paper"),
      "overstated", "the numbers I reproduce are 0.439 / 0.388 / 0.327, but the builder's own record (results/gefs_basins.json, skill.how_to_read) says they are NOT a check of the paper; 'the same as the published numbers' contradicts it. The uncalibrated yes/no Brier skill of the same member is 0.156 / 0.069 / -0.047")
claim("30 trainings, 80 minutes", "4576.5 s of training (76 min) plus 77.8 s for the simple models, runtime 4827 s", "agrees to rounding")
# E2a table
exp = {"wq": ([19, 8, 6], [0.59, 0.32, 0.25]), "wqa": ([51, 21, 12], [0.77, 0.38, 0.25]), "wqaF_sub": ([45, 39, 37], [0.69, 0.53, 0.45]), "wqaF_clim": ([14, 4, 4], [0.60, 0.19, 0.03]), "wqF_sub": ([31, 36, 36], [0.59, 0.51, 0.45]), "wqaP": ([66, 69, 71], [0.80, 0.78, 0.77]), "wqf": ([64, 67, 69], [0.86, 0.82, 0.78])}
for n, (po, ns) in exp.items():
    m_po = [int(np.floor(pod(n, k) + 0.5)) for k in KS]; m_ns = [round(nse(n, k), 2) for k in KS]
    ok = m_po == po and m_ns == ns
    claim(f"E2a table, {n}: onsets {po}, NSE {ns}", dict(onsets_pct=[round(pod(n, k), 2) for k in KS], nse=[round(nse(n, k), 3) for k in KS]), "agrees" if ok else "differs",
          "" if ok else f"rounded half up I get onsets {m_po}, NSE {m_ns}" + ("; 68.46 rounds to 68 (the results file prints 68%): a rounding slip of one point in the ROADMAP table" if n == "wqf" else ""))
claim("ridge / random forest given the same inputs: 69 / 55% at 1 day, 0.71 / 0.72 at 1 day", dict(onsets=[round(pod('base_linreg_wqaF', 1), 1), round(pod('base_rf_wqaF', 1), 1)], nse=[round(nse('base_linreg_wqaF', 1), 3), round(nse('base_rf_wqaF', 1), 3)]), "agrees")
# E2b
exp_s = {"wq_s": ([15, 6, 4], [0.53, 0.33, 0.26]), "wqa_s": ([41, 14, 8], [0.71, 0.39, 0.28]), "wqaF_s": ([61, 49, 44], [0.77, 0.62, 0.52]), "wqaF_clim_s": ([41, 15, 8], None), "wqaP_s": ([72, 77, 77], None)}
for n, (po, ns) in exp_s.items():
    m_po = [int(np.floor(pod(n, k) + 0.5)) for k in KS]; m_ns = [round(nse(n, k), 2) for k in KS]
    ok = m_po == po and (ns is None or m_ns == ns)
    claim(f"E2b, {n}: onsets {po}" + (f", NSE {ns}" if ns else ""), dict(onsets_pct=[round(pod(n, k), 2) for k in KS], nse=[round(nse(n, k), 3) for k in KS]), "agrees" if ok else "differs", "" if ok else f"rounded half up I get {m_po} {m_ns}")
# readings
d = {k: pod("wqaF_sub", k) - pod("wqa", k) for k in KS}; dn = {k: nse("wqaF_sub", k) - nse("wqa", k) for k in KS}
claim("+18 and +25 points of onsets at 2 and 3 days (clear in every training), +0.16 / +0.20 NSE; costs 7 points at 1 day", dict(points=[round(d[k], 1) for k in KS], nse=[round(dn[k], 3) for k in KS], every_seed_above_wqa=[F9['e2_success']['mine'][f'l{k}']['every_seed_above_wqa'] for k in KS]), "agrees")
claim("its own networks given perfect rain at run time warn of 69%", f"{100 * F12['slot_content_ladder']['lstm_wqaF_sub_perfectpred']['onsets'][0]:.1f}%", "agrees")
dS = {k: pod("wqaF_s", k) - pod("wqa_s", k) for k in KS}
clos = {k: (pod("wqaF_s", k) - pod("wqa_s", k)) / (pod("wqaP_s", k) - pod("wqa_s", k)) for k in KS}; clos_sub = {k: (pod("wqaF_sub", k) - pod("wqa", k)) / (pod("wqaP", k) - pod("wqa", k)) for k in KS}
claim("retraining: +20 / +35 / +36 points; closes 65 / 56 / 52% of the gap to perfect rain against 37-42% for substitution at 2-3 days", dict(points=[round(dS[k], 1) for k in KS], closes_pct=[round(100 * clos[k], 1) for k in KS], substitution_closes_pct=[round(100 * clos_sub[k], 1) for k in KS]), "agrees to rounding")
claim("the rain to midnight still matters with the forecast at 1 day (+13 points) and not at 2-3 days", [round(pod("wqaF_sub", k) - pod("wqF_sub", k), 1) for k in KS], "agrees")
claim("The shuffled control shows the information is real (a network given an uninformative forecast collapses)", dict(E2a_control_vs_wqa_points=[round(pod("wqaF_clim", k) - pod("wqa", k), 1) for k in KS], E2b_control_vs_wqa_s_points=[round(pod("wqaF_clim_s", k) - pod("wqa_s", k), 1) for k in KS]), "overstated",
      "the E2a control collapses because the substituted network trusts its slots: that shows the slots matter to the network, not that the forecast carries information. The evidence that the information is real is the gain over wqa at 2-3 days (+18 and +25 points), the retrained E2b gain (+20 / +35 / +36 points) with a control that is harmless there (-0.3 / +0.9 / +0.4 points), and the skill statistics")
claim("The simple models given the same inputs gain as much: the information, not the network, once more", dict(gain_ridge=[round(pod('base_linreg_wqaF', k) - pod('base_linreg_wqa', k), 1) for k in KS], gain_forest=[round(pod('base_rf_wqaF', k) - pod('base_rf_wqa', k), 1) for k in KS], gain_lstm=[round(d[k], 1) for k in KS]), "agrees", "they gain MORE than the LSTM at every lead (at the plain trigger, with more false warnings; by average precision the forest equals the LSTM)")
# verdicts
R = jload("results/forecast_input.json"); V = R["verdicts"]["sets"]
va = lambda n, key="plain": [V[n][f"l{k}"][key]["all"]["verdict"] if key == "plain" else V[n][f"l{k}"][key]["all"]["verdict"] for k in KS]
claim("Verdicts: onsets NOT YET at every lead (E2a 35-41% of onset warnings false)", dict(onset_verdicts=[V["wqaF_sub"][f"l{k}"]["plain"]["onset"]["verdict"] for k in KS], far_pct=[round(100 * own['wqaF_sub'][f'l{k}']['far'], 1) for k in KS]), "agrees")
claim("all high-flow days CAUTION / CAUTION / NOT YET in E2a and GO / CAUTION / CAUTION in E2b on its split",
      dict(E2a_plain=va("wqaF_sub"), E2b_plain=va("wqaF_s"), E2a_at_the_level_chosen_on_validation_years=va("wqaF_sub", "chosen_on_validation_all_days"), E2b_at_the_chosen_level=va("wqaF_s", "chosen_on_validation_all_days")), "agrees at the plain trigger",
      "the JSON's criteria text says verdicts are applied at the warning level chosen on the validation years; at that level E2a is GO / CAUTION / NOT YET. ROADMAP.md does not say which level it quotes")
claim("The E2 success rule of the work order (a clear gain at 2 and 3 days) is met", F9["e2_success"]["met_at_2_and_3_days"], "agrees")
from scipy.stats import wilcoxon
nb_hi, nb_p = {}, {}
for k in KS:
    a = nse_per_basin("lstm_wqaF_sub", k); b = nse_per_basin("lstm_wqa", k); g = sorted(a)
    nb_hi[k] = int(sum(a[x] > b[x] for x in g)); nb_p[k] = float(wilcoxon([a[x] for x in g], [b[x] for x in g]).pvalue)
claim("three seeds (NSE readings 'mixed' at 1-2 days)", dict(basins_with_higher_NSE_of_12={f"l{k}": nb_hi[k] for k in KS}, wilcoxon_p={f"l{k}": round(nb_p[k], 3) for k in KS}), "differs",
      "by the rule of report section 4.2 (every seed above, more than half of the basins higher, p below 0.05) the NSE of wqaF_sub against wqa is 'mixed' at all three leads (the results file says so: mixed, mixed and mixed); the sentence should say 'at 1, 2 and 3 days'")
# the data paragraph's size statement (design text)
claim("about 50-61 GB streamed and reduced to the south-eastern box (about 2 GB kept)", dict(fetched_gb=H8 and jread("f3_store.json")["bytes_fetched_total_gb"], kept_mb="c00 folder 141 MB + basins 6.8 MB"), "differs", "design text not updated: 48.5 GB were fetched, about 0.15 GB are kept (the box as uint16 per month)")
n = {s: sum(1 for r in rows if r["status"].startswith(s)) for s in ("agrees", "differs", "overstated")}
print(f"\n{len(rows)} statements: {n}")
jsave("f18_roadmap.json", dict(rows=rows, counts=n))
