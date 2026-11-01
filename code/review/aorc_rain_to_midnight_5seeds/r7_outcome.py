"""Round 2, checks o1-o3: the sentences people quote. o1 every number of outcome / outcome_caveat (= outcome_full) of results/aorc_rain_to_midnight.json rebuilt from my own
numbers (r2, r4, r6 outputs and my scoring) and searched for in the text; o2 the outcome of results/stage4_realtime.json; o3 the verdict words (trigger 1.0 and 1.05) and the
mean absolute errors on the onset days. Clauses that quote round 1's three-seed hindsight numbers are listed with what they would read with five seeds."""
from r_common import *

J = jload(RESULTS)
S4 = jload(STAGE4)
H2 = jread("r2_headline.json")["mine"]
R4 = jread("r4_stage4.json")["mine"]
R6 = jread("r6_hindsight.json")
C = Compare()
OC, OUTC, S4O = J["outcome_caveat"], J["outcome"], S4["outcome"]
C.add("o1 outcome_full = outcome + ' ' + outcome_caveat", True, J["outcome_full"] == OUTC + " " + OC)


def pc(x, nd=0):
    return f"{100 * x:.{nd}f}"


def pod(n):
    return score(n)[1]["onset"]["pod"]


wqa, wq, ctl, wqh, wqn, rep = (score(f"lstm_{n}")[1] for n in ("wqa", "wq", "wqa_to09", "wqh", "wqn", "wqa_rep"))
lin, rf = score("base_linreg_wqa")[1], score("base_rf_wqa")[1]
lin0, rf0 = score("base_linreg_wq")[1], score("base_rf_wq")[1]
R = H2["e1b_success"]["seed_ranges"]
nse_s = H2["wqa_l1_single_seed_nse_median"]
OV = H2["overlaps"]["value_of_hours_09_to_24"]
above = H2["lstm_trainings_above_simple_onsets_l1"]
rep_rng = R["wqa_rep"]
wq_rng = R["wq"]
d_rep_wq = 100 * (rep["onset"]["pod"] - wq["onset"]["pod"])
d_wqa_ctl = 100 * (wqa["onset"]["pod"] - ctl["onset"]["pod"])
EXPECT = {
    "outcome: 19% -> 52%": (OUTC, f"go from {pc(wq['onset']['pod'])}% with the rain table alone to {pc(wqa['onset']['pod'])}%"),
    "outcome: single trainings": (OUTC, f"(single trainings {pc(R['wqa'][0])}% to {pc(R['wqa'][1])}%"),
    "outcome: +33.1 / +33.7 / 18%": (OUTC, f"+{100 * (wqa['onset']['pod'] - wq['onset']['pod']):.1f} points over the rain table alone and +{d_wqa_ctl:.1f} points over the control with the same AORC rain over the table's own window ({pc(ctl['onset']['pod'])}%)"),
    "outcome: NSE 0.59 -> 0.76": (OUTC, f"the median NSE goes from {wq['nse_median']:.2f} to {wqa['nse_median']:.2f}"),
    "outcome: NSE single trainings": (OUTC, f"single trainings {min(nse_s):.2f} to {max(nse_s):.2f}"),
    "caveat: ERA5": (OC, f"lower share ({pc(wqh['onset']['pod'])}%, {wqh['onset']['hits']} hits of 611) than the AORC rain to midnight ({pc(wqa['onset']['pod'])}%, {wqa['onset']['hits']} hits of 611)"),
    "caveat: NLDAS": (OC, f"lower share ({pc(wqn['onset']['pod'])}%, {wqn['onset']['hits']} hits of 611)"),
    "caveat: five single trainings": (OC, f"the five single trainings with the AORC rain to midnight span {pc(R['wqa'][0], 1)}% to {pc(R['wqa'][1], 1)}%"),
    "caveat: control flat": (OC, f"is flat ({pc(ctl['onset']['pod'], 1)}% against {pc(wq['onset']['pod'], 1)}% with the rain table alone)"),
    "caveat: 228 against 22": (OC, f"({OV['warned_only_by_first']} onsets warned only by the AORC rain to midnight against {OV['warned_only_by_second']} only by that control)"),
    "caveat: against ERA5": (OC, f"+{100 * (wqa['onset']['pod'] - wqh['onset']['pod']):.1f} points ({wqa['onset']['hits']} against {wqh['onset']['hits']} hits of 611; every training with AORC above every training with ERA5: {'yes' if R['wqa'][0] > R['wqh'][1] else 'no'})"),
    "caveat: ridge and forest": (OC, f"warn of {pc(lin['onset']['pod'], 1)}% and {pc(rf['onset']['pod'], 1)}% (the new input adds {100 * (lin['onset']['pod'] - lin0['onset']['pod']):.1f} and {100 * (rf['onset']['pod'] - rf0['onset']['pod']):.1f} points to them, {100 * (wqa['onset']['pod'] - wq['onset']['pod']):.1f} to the LSTM)"),
    "caveat: NSE against ridge and forest": (OC, f"the LSTM ({wqa['nse_median']:.2f}) against ridge regression ({lin['nse_median']:.2f}) and the random forest ({rf['nse_median']:.2f})"),
    "caveat: ridge on the onsets": (OC, f"({pc(lin['onset']['pod'], 1)}% against {pc(wqa['onset']['pod'], 1)}%, {pc(lin['onset']['far'])}% of its warnings false against {pc(wqa['onset']['far'])}%; {above['base_linreg_wqa']} of the 5 LSTM trainings is above it)"),
    "caveat: forest on the onsets": (OC, f"({pc(rf['onset']['pod'], 1)}% against {pc(wqa['onset']['pod'], 1)}%, {pc(rf['onset']['far'])}% of its warnings false against {pc(wqa['onset']['far'])}%; {above['base_rf_wqa']} of the 5 LSTM trainings are above it)"),
    "caveat: replaced table": (OC, f"warns of {pc(rep['onset']['pod'])}% of the onsets against {pc(wq['onset']['pod'])}% with the Daymet rain table ({pc(rep_rng[0], 1)}% to {pc(rep_rng[1], 1)}% against {pc(wq_rng[0], 1)}% to {pc(wq_rng[1], 1)}% in the single trainings; +{d_rep_wq:.1f} points over the rain table alone"),
    "caveat: product alone": (OC, f"adds {100 * (ctl['onset']['pod'] - wq['onset']['pod']):+.1f} points to the rain table alone, over the whole day it adds {d_wqa_ctl:+.1f} points to that control, the replaced table adds {d_rep_wq:+.1f} points"),
    "caveat: replaced against extra column": (OC, f"against AORC as an extra column it adds {100 * (rep['onset']['pod'] - wqa['onset']['pod']):+.1f} points"),
    "caveat: about 34 points": (OC, f"worth about {round((d_wqa_ctl + d_rep_wq) / 2)} points (the two gains above, averaged)"),
}
V = S4["models"]["wqa"]["variants"]
a, t = S4["models"]["wqa"]["aorc"], S4["models"]["wqa"]["rain_table_alone"]
sv = lambda k: (V[k]["onsets_warned"] - t) / (a - t)
iv = R4["intervals_exact_replication"]["over_issue_days"]["share_surviving"]
iv_on = R4["intervals_exact_replication"]["over_onsets"]["share_surviving"]
CT = S4["models"]["wqa"]["controls_on_the_rain_of_the_issue_day"]
# MAE on the onset days, own computation for AORC and the rain table
d, b, obs, pa, meta = load_npz("lstm_wqa")
rows = [tuple(x) for x in R4["issue_day_values"]["rows"]]
ti = [(b.index(g), d.get_loc(pd.Timestamp(day) + pd.Timedelta(days=1))) for g, day in rows]
mae_a = float(np.mean([abs(pa[i, j, 0] - obs[i, j]) for i, j in ti]))
del pa
_, _, _, pq, _ = load_npz("lstm_wq")
mae_t = float(np.mean([abs(pq[i, j, 0] - obs[i, j]) for i, j in ti]))
del pq
C.add("o3 MAE on the onset days with AORC = my value", S4["models"]["wqa"]["reference_fits_on_the_onset_days"]["aorc"]["mae_mm_day"], mae_a, tol=1e-9)
C.add("o3 MAE on the onset days, rain table = my value", S4["models"]["wqa"]["reference_fits_on_the_onset_days"]["rain_table_alone"]["mae_mm_day"], mae_t, tol=1e-9)
mae_s = float(np.mean([abs(r["forecast_mm_stage4"] - obs[i, j]) for r, (i, j) in zip(S4["per_onset"], ti)]))
C.add("o3 MAE with Stage IV from the 4-decimal per_onset forecasts (within 1e-4)", True, abs(mae_s - V["stage4"]["fit_on_the_onset_days"]["mae_mm_day"]) < 1e-4)
EXPECT.update({
    "caveat: Stage IV 294/316/114": (OC, f"{V['stage4']['onsets_warned']} of 611 warned against {a} with AORC and {t} with the rain table alone, {pc(sv('stage4'))}% of the AORC effect survives (95% interval {pc(iv[0])}% to {pc(iv[1])}%, drawing issue days)"),
    "caveat: last hour out": (OC, f"{V['stage4_no_last_hour']['onsets_warned']} are warned ({pc(sv('stage4_no_last_hour'))}% of the effect)"),
    "caveat: level restored": (OC, f"(Stage IV times {S4['k_level_aorc_over_stage4']:.2f}) {V['stage4_k']['onsets_warned']}"),
    "caveat: controls": (OC, f"in its place {round(CT['another_onset_stage4']['mean'])} onsets are warned, with the AORC rain of an ordinary day {round(CT['ordinary_day_aorc']['mean'])}, with no rain {round(CT['no_rain']['mean'])}"),
    "caveat: MAE": (OC, f"is {V['stage4']['fit_on_the_onset_days']['mae_mm_day']:.2f} mm/day with Stage IV, {mae_a:.2f} with AORC and {CT['another_onset_stage4']['fit_on_the_onset_days']['mae_mm_day']:.2f} with another onset's rain"),
    "stage4 outcome: headline": (S4O, f"warn of {V['stage4']['onsets_warned']} ({pc(V['stage4']['onsets_warned'] / 611)}%) against {a} with AORC ({pc(a / 611)}%) and {t} with the rain table alone ({pc(t / 611)}%): {pc(sv('stage4'))}% of the AORC effect survives (95% interval {pc(iv_on[0])}% to {pc(iv_on[1])}% drawing onsets, {pc(iv[0])}% to {pc(iv[1])}% drawing issue days)"),
    "stage4 outcome: variants": (S4O, f"{V['stage4_k']['onsets_warned']} are warned; with the last hour of the issue day taken out {V['stage4_no_last_hour']['onsets_warned']}, with the last two hours {V['stage4_no_last_two_hours']['onsets_warned']}"),
    "stage4 outcome: MAE": (S4O, f"is {mae_a:.2f} mm/day with AORC, {V['stage4']['fit_on_the_onset_days']['mae_mm_day']:.2f} with Stage IV, {CT['another_onset_stage4']['fit_on_the_onset_days']['mae_mm_day']:.2f} with another onset's Stage IV rain, {CT['ordinary_day_aorc']['fit_on_the_onset_days']['mae_mm_day']:.2f} with an ordinary day's rain and {mae_t:.2f} with the rain table alone"),
    "stage4 outcome: control range": (S4O, f"{round(CT['another_onset_stage4']['mean'])} ({CT['another_onset_stage4']['lowest']} to {CT['another_onset_stage4']['highest']} in 10 draws)"),
})
for k, (txt, phrase) in EXPECT.items():
    C.add(f"o1 {k}: '{phrase}' found", True, phrase in txt)
# o3 verdict words at trigger 1.0 and the level chosen on the validation years (1.05)
def verdict(r):
    return "GO" if r["pod"] >= 0.70 and r["far"] <= 0.30 else "CAUTION" if r["csi"] >= 0.40 else "NOT YET"
VB = J["verdicts"]["sets"]["wqa"]["l1"]
w105 = score("lstm_wqa", 1.05)[1]
for part in ("onset", "all"):
    C.add(f"o3 plain verdict {part}", VB["plain"][part]["verdict"], verdict(wqa[part]))
    C.add(f"o3 verdict at 1.05 {part}", VB["chosen_on_validation_onsets"][part]["verdict"], verdict(w105[part]))
    for x in ("hits", "misses", "false_alarms"):
        C.add(f"o3 at 1.05 {part} {x}", VB["chosen_on_validation_onsets"][part][x], w105[part][x])
C.add("o1 outcome verdict words", True, "NOT YET for the onsets and CAUTION for all high-flow days" in OUTC)
C.add("stage4 outcome trigger at the validation level: onsets at 1.05 with AORC = my count", V["stage4"]["onsets_warned_at_the_level_chosen_on_validation"]["aorc"], w105["onset"]["hits"])
# the clauses that quote round 1 (three seeds) inside the five-seed text
F5, F3 = R6["five_seeds"], R6["three_seeds"]
stale = {
    "86 to 95 of the 171 onsets of those days against 92 with AORC (the rain table alone: 15), that is 92% to 104% of the AORC effect": dict(
        measured_on="the three saved wqa networks of seeds 0-2 (round 1, check d5)", five_seed_reading=f"{F5['stage4_raw_sample']} (raw Stage IV) of the 171 against {F5['aorc_sample']} with AORC (rain table: {F5['rain_table_sample']}); level-matched not rerun"),
    "at most about 3 points of the 52% (16 of 316 hits)": dict(measured_on="16 of 313 hits, three networks (round 1, check d7); the text now divides it by the five-seed 316",
                                                            five_seed_reading=f"round 1's Stage IV route gives {F5['round1_route_extrapolated_hits']:.0f} hits ({F5['round1_route_extrapolated_points']:.1f} points) with five seeds; direct raw loss on all 611 onsets {F5['direct_all_611_raw_loss_hits']} hits ({F5['direct_all_611_raw_loss_points']:.1f} points); level matched {F5['direct_all_611_level_matched_loss_hits']} hits ({F5['direct_all_611_level_matched_loss_points']:.1f} points)"),
}
found = {k: (k in OC) for k in stale}
C.add("o1 the three-seed hindsight clauses are present (to be labelled)", True, all(found.values()))
C.add("o1 the hindsight clause says the networks were three", True, ("three" in OC[OC.find("An independent review"):OC.find("the other 440 onsets")]))
C.add("stage4 outcome seed count in words", True, "the five saved lstm_wqa networks" in S4O)
jsave("r7_outcome.json", dict(compare=C.rows, summary=C.summary(), stale_clauses=stale, files_read=READ, mine=dict(mae_aorc=mae_a, mae_rain_table=mae_t, mae_stage4_from_rounded=mae_s,
                                                                                              wqa_at_1_05=dict(onset=w105["onset"], all=w105["all"]))))
print(json.dumps(C.summary(), indent=1))
for k, v in C.rows.items():
    if not v["ok"]:
        print("NOT OK:", k)
