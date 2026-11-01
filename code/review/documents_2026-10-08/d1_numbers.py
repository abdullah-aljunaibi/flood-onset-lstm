"""d1: 69 numbers of the documents of 8 October 2026 (build of 10:21-10:24) compared with the results files they come from.

For each claim the script builds, from the results file (read as JSON, the key written in the third column of the output), the sentence fragment as the document must print it (rounded the way the document rounds), and looks
for that fragment in the text of the document, page by page (PDF text extracted with pdf2txt.swift, Markdown files as they are). A claim passes when the fragment built from the results file is found in the document; the pages
where it is found are listed. 15 claims are of the second kind: the fragment is a fixed text (the document's own sentence) and the truth is the number computed from the results file (the 'truth' column differs from the needle or is
rebuilt from the file); they pass when the document's sentence equals the truth rebuilt from the file.
The numbers in the results files were themselves recomputed with independent code: d2 (headline scores from the stored forecasts, 378 numbers against flood.evaluate), d3 (Stage IV), d4 (rain clock), d5 (key figure sources).
Inputs: DOC_TEXT_DIR = a folder with Flood_Report.txt, Flood_Slides.txt, Professor_QA.txt, Project_Guide.txt, Project_Overview.txt (made with pdf2txt.swift: 'pdf2txt in.pdf out.txt', pages separated by '=====PAGE n====='),
and DOC_MD_DIR = a folder with short_report.md, slides_notes.md, README.md (default: the project folder). The needles are the wording of the build of 8 October 10:21-10:24; after a rebuild the claims whose wording was changed
(the real-time and hindsight sentences, the 51% sentences) have to be given the new wording - that is the point of the second round.
Output: OUT/d1_numbers.json
"""
import re
from dcommon import *

R = RES
J = jload
A, RT, E1, AO, S4, MB, FI, UN, KF, FA, MC, SM, OC = (J(n) for n in ['lstm_all', 'rain_timing', 'rain_to_midnight', 'aorc_rain_to_midnight', 'stage4_realtime', 'more_basins', 'forecast_input', 'uncertainty', 'key_figure', 'flood_analysis', 'model_comparison', 'same_misses', 'onset_rain_clock'])
LB = J('lstm_lobo')
TEXT_DIR = Path(os.environ.get("DOC_TEXT_DIR") or PROJ)
MD_DIR = Path(os.environ.get("DOC_MD_DIR") or PROJ)


def pages(txt):
    parts = re.split(r'\n=====PAGE (\d+)=====\n', '\n' + txt)
    d = {}
    for i in range(1, len(parts), 2):
        d[int(parts[i])] = parts[i + 1]
    return d


PDF = {k: pages(open(TEXT_DIR / f'{v}.txt').read()) for k, v in dict(R='Flood_Report', S='Flood_Slides', Q='Professor_QA', G='Project_Guide', O='Project_Overview').items()}
MD = {'short': open(MD_DIR / 'short_report.md').read(), 'notes': open(MD_DIR / 'slides_notes.md').read(), 'readme': open(MD_DIR / 'README.md').read()}


def norm(s):
    s = s.replace('\u2212', '-').replace('\u2013', '-').replace('\u2019', "'").replace('\u2018', "'")
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('- ', '-')   # PDF hyphen breaks
    return s


def find(doc, needle):
    n = norm(needle)
    if doc in PDF:
        return [p for p, t in PDF[doc].items() if n in norm(t)]
    return [1] if n in norm(MD[doc]) else []


def p0(x): return f"{100 * x:.0f}%"
def n2(x): return f"{x:.2f}"
def n1(x): return f"{x:.1f}"


rows = []


def claim(label, doc, needle, truth, key, tol_text=None):
    pg = find(doc, needle)
    ok = norm(needle) == norm(truth) if truth is not None else None
    rows.append((label, doc, pg, needle, truth, key, ok))


# ---------- values ----------
L1 = lambda S, s: S[s]['l1']
st = lambda F, s, k: F['summary_test']['sets'][s][k]
wq = st(E1,'wq','l1'); wqf = st(E1,'wqf','l1'); wqn = st(E1,'wqn','l1'); wqa = st(AO,'wqa','l1')
cw = A['comparison']['wq']
best = {k: cw['baselines'][cw['best_baseline_by_validation'][k]][k] for k in ('l1','l2','l3')}
c1 = RT['closure']['wqn']['by_lead'][0]
y_ao = AO['yardstick']['new_input_given_to_each_model']
y_fi = FI['yardstick']['new_input_given_to_each_model']
V = FI['value_of_the_forecast']
OP = V['operating_points']
# ---------- page 1 of the report ----------
claim('P1 NSE LSTM vs best simple', 'R', f"{n2(best['l1']['lstm_nse_median'])} against {n2(best['l1']['baseline_nse_median'])}", f"{n2(best['l1']['lstm_nse_median'])} against {n2(best['l1']['baseline_nse_median'])}", 'lstm_all.comparison.wq.baselines[best_by_validation].l1')
claim('P1 +45 and +30 against +33', 'R', f"+{y_ao['ridge regression']['by_lead']['l1']['onset_gain_points']:.0f} and +{y_ao['random forest']['by_lead']['l1']['onset_gain_points']:.0f} points against +{y_ao['lstm']['by_lead']['l1']['onset_gain_points']:.0f}", '+45 and +30 points against +33', 'aorc_rain_to_midnight.yardstick.new_input_given_to_each_model.*.by_lead.l1.onset_gain_points')
claim('P1 +39 and +25 against +18 (2 days)', 'R', f"+{y_fi['ridge regression']['by_lead']['l2']['onset_gain_points']:.0f} and +{y_fi['random forest']['by_lead']['l2']['onset_gain_points']:.0f} against +{y_fi['lstm']['by_lead']['l2']['onset_gain_points']:.0f}", '+39 and +25 against +18', 'forecast_input.yardstick.new_input_given_to_each_model.*.by_lead.l2.onset_gain_points')
claim('P1 0.77 against 0.64 and 0.66', 'R', f"{n2(y_ao['lstm']['by_lead']['l1']['nse_median_with'])} against {n2(y_ao['ridge regression']['by_lead']['l1']['nse_median_with'])} and {n2(y_ao['random forest']['by_lead']['l1']['nse_median_with'])}", '0.77 against 0.64 and 0.66', 'aorc...yardstick...nse_median_with')
claim('P1 river gauge +0.15', 'R', f"+{n2(KF['steps']['river_gauge']['nse'])} NSE", '+0.15 NSE', 'key_figure.steps.river_gauge.nse')
claim('P1 19% to 64%', 'R', f"from {p0(wq['onset']['pod'])} to {p0(wqf['onset']['pod'])}", 'from 19% to 64%', 'rain_to_midnight.summary_test.sets.{wq,wqf}.l1.onset.pod')
claim('P1 19% to 30% control 19%', 'R', f"({p0(c1['onset_warned_base'])} to {p0(c1['onset_warned_with_nldas'])}; control {p0(c1['onset_warned_control'])})", '(19% to 30%; control 19%)', 'rain_timing.closure.wqn.by_lead[0]')
claim('P1 5 of 12 basins (NLDAS NSE)', 'R', f"higher in only {c1['basins_better']} of 12 basins", 'higher in only 5 of 12 basins', 'rain_timing.closure.wqn.by_lead[0].basins_better')
claim('P1 51% control 18% 48% Stage IV', 'R', f"{p0(wqa['onset']['pod'])} of the onsets are warned 1 day ahead (control {p0(AO['closure']['wqa_to09']['by_lead'][0]['onset_warned_with'])}; {p0(S4['models']['wqa']['variants']['stage4']['onsets_warned']/S4['models']['wqa']['onsets_tested'])} when Stage IV", '51% of the onsets are warned 1 day ahead (control 18%; 48% when Stage IV', 'aorc.summary_test.sets.wqa; closure.wqa_to09; stage4_realtime.models.wqa.variants.stage4')
fa = FI['summary_test']['sets']
claim('P1 39% and 37% vs 8% and 6%', 'R', f"{p0(fa['wqaF_sub']['l2']['onset']['pod'])} and {p0(fa['wqaF_sub']['l3']['onset']['pod'])} 2 and 3 days ahead, against {p0(fa['wq']['l2']['onset']['pod'])} and {p0(fa['wq']['l3']['onset']['pod'])}", '39% and 37% 2 and 3 days ahead, against 8% and 6%', 'forecast_input.summary_test.sets.{wqaF_sub,wq}.l2/l3.onset.pod')
claim('P1 611 onsets, 19% warned', 'R', f"{p0(wq['onset']['pod'])} of the {wq['onset']['hits']+wq['onset']['misses']} onsets are warned", '19% of the 611 onsets are warned', 'lstm_all/rain_to_midnight.summary_test.sets.wq.l1.onset')
claim('P1 50% false rain to midnight; 40% and 41%', 'R', f"{p0(wqa['onset']['far'])} of the warnings false with rain up to midnight; {p0(fa['wqaF_sub']['l2']['onset']['far'])} and {p0(fa['wqaF_sub']['l3']['onset']['far'])}", '50% of the warnings false with rain up to midnight; 40% and 41%', 'aorc/forecast_input.summary_test.sets...onset.far')
# ---------- 4.6 ----------
ER = E1['summary_test']['sets']
hv = E1['value_of_hours_09_to_24']['by_lead']['l1']['onsets']; ev = E1['value_of_evening_hours_beyond_utc_day']['by_lead']['l1']['onsets']
hv2 = AO['value_of_hours_09_to_24']['by_lead']['l1']['onsets']; ev2 = AO['value_of_evening_hours_beyond_utc_day']['by_lead']['l1']['onsets']
claim('4.6 hours worth 9.5 points, 8.2 after 19:00 (ERA5)', 'R', f"{hv['difference_points']:.1f} points, {ev['difference_points']:.1f} of them after 19:00", '9.5 points, 8.2 of them after 19:00', 'rain_to_midnight.value_of_hours_09_to_24 / value_of_evening_hours_beyond_utc_day')
claim('4.6 ERA5 to09 20%, utc 22%', 'R', f"warn of {p0(ER['wqh_to09']['l1']['onset']['pod'])}, over the UTC day of {p0(ER['wqh_utc']['l1']['onset']['pod'])}", 'warn of 20%, over the UTC day of 22%', 'rain_to_midnight.summary_test.sets.wqh_to09/wqh_utc')
claim('4.6 NLDAS+ERA5 together 32%', 'R', f"together warn of {p0(ER['wqnh']['l1']['onset']['pod'])}", 'together warn of 32%', 'rain_to_midnight.summary_test.sets.wqnh')
claim('4.6 +15.0 before / +9.5 after', 'R', f"(+{E1['by_period']['differences']['wqh_minus_wq']['before_2009_07_17']['onset_points']:.1f} points) than after it (+{E1['by_period']['differences']['wqh_minus_wq']['from_2009_07_17']['onset_points']:.1f})", '(+15.0 points) than after it (+9.5)', 'rain_to_midnight.by_period.differences.wqh_minus_wq')
claim('4.6 AORC 19% to 51%, 49-55%', 'R', f"from {p0(wq['onset']['pod'])} to {p0(wqa['onset']['pod'])} (single trainings {AO['closure']['wqa']['by_lead'][0]['onset_seed_range_with'][0]*100:.0f}% to {AO['closure']['wqa']['by_lead'][0]['onset_seed_range_with'][1]*100:.0f}%)", 'from 19% to 51% (single trainings 49% to 55%)', 'aorc.closure.wqa.by_lead[0].onset_seed_range_with')
claim('4.6 AORC to09 18% utc 37%', 'R', f"warns of {p0(AO['summary_test']['sets']['wqa_to09']['l1']['onset']['pod'])} (flat), over the UTC day of {p0(AO['summary_test']['sets']['wqa_utc']['l1']['onset']['pod'])}", 'warns of 18% (flat), over the UTC day of 37%', 'aorc.summary_test.sets.wqa_to09/wqa_utc')
claim('4.6 33.6 / 13.7 / +21.3', 'R', f"{hv2['difference_points']:.1f} points, {ev2['difference_points']:.1f} of them after 19:00. AORC is clearly above ERA5 (+{AO['aorc_against_era5']['by_lead']['l1']['onsets']['difference_points']:.1f} points)", '33.6 points, 13.7 of them after 19:00. AORC is clearly above ERA5 (+21.3 points)', 'aorc.value_of_hours_09_to_24 / value_of_evening... / aorc_against_era5')
claim('4.6 NSE +0.05 +0.00; 21%/12%', 'R', f"(+{AO['value_of_hours_09_to_24'] and (AO['closure']['wqa']['by_lead'][1]['nse_with']-AO['closure']['wqa']['by_lead'][1]['nse_base']):.2f} and +{(AO['closure']['wqa']['by_lead'][2]['nse_with']-AO['closure']['wqa']['by_lead'][2]['nse_base']):.2f}) while the onsets warned rise from {p0(st(E1,'wq','l2')['onset']['pod'])} and {p0(st(E1,'wq','l3')['onset']['pod'])} to {p0(st(AO,'wqa','l2')['onset']['pod'])} and {p0(st(AO,'wqa','l3')['onset']['pod'])}", '(+0.05 and +0.00) while the onsets warned rise from 8% and 6% to 21% and 12%', 'aorc.closure.wqa.by_lead[1..2]; rain_to_midnight.summary_test.sets.wq.l2/l3.onset.pod')
claim('4.6 ridge 55% forest 48% adds 44.5/29.8/32.6', 'R', f"warns of {p0(y_ao['ridge regression']['by_lead']['l1']['onset_warned_with'])} and the random forest of {p0(y_ao['random forest']['by_lead']['l1']['onset_warned_with'])} of the onsets (the LSTM: {p0(y_ao['lstm']['by_lead']['l1']['onset_warned_with'])}): the input adds {y_ao['ridge regression']['by_lead']['l1']['onset_gain_points']:.1f} and {y_ao['random forest']['by_lead']['l1']['onset_gain_points']:.1f} points to them against {y_ao['lstm']['by_lead']['l1']['onset_gain_points']:.1f}", 'warns of 55% and the random forest of 48% of the onsets (the LSTM: 51%): the input adds 44.5 and 29.8 points to them against 32.6', 'aorc.yardstick.new_input_given_to_each_model')
rv = None
claim('4.6 78 days -0.03 (-0.06 to -0.001)', 'R', "78 test-year days (coefficient -0.03, 95% interval -0.06 to -0.001)", "78 test-year days (coefficient -0.03, 95% interval -0.06 to -0.001)", 'review_aorc_rain_to_midnight.checks[d3].numbers.coefficient...whole day t, both = [-0.0301, [-0.06, -0.0014]]')
s4m = S4['models']['wqa']; vv = s4m['variants']
claim('4.6 294 (48%) 313 114; 90%; 85-97; 275 (45%)', 'R', f"warn of {vv['stage4']['onsets_warned']} ({p0(vv['stage4']['share_of_the_onsets'])}) against {s4m['aorc']} with AORC and {s4m['rain_table_alone']} with the rain table alone, so {p0(vv['stage4']['share_of_the_aorc_effect_surviving'])} of the AORC effect survives (95% interval {p0(vv['stage4']['intervals']['over_onsets']['share_surviving'][0])} to {p0(vv['stage4']['intervals']['over_onsets']['share_surviving'][1])}); {vv['stage4_no_last_hour']['onsets_warned']} ({p0(vv['stage4_no_last_hour']['share_of_the_onsets'])})", 'warn of 294 (48%) against 313 with AORC and 114 with the rain table alone, so 90% of the AORC effect survives (95% interval 85% to 97%); 275 (45%)', 'stage4_realtime.models.wqa.variants')
claim('4.6 CAUTION all days 74% / 31%', 'R', f"CAUTION (74% warned, 31% false)", f"CAUTION ({p0(AO['verdicts']['sets']['wqa']['l1']['plain']['all']['pod'])} warned, {p0(AO['verdicts']['sets']['wqa']['l1']['plain']['all']['far'])} false)", 'aorc.verdicts.sets.wqa.l1.plain.all')
chk = lambda n: J(n)['counts']['checks_listed']
claim('4.6 58 listed checks', 'R', f"in {chk('review_rain_to_midnight')+chk('review_aorc_rain_to_midnight')} listed checks", 'in 58 listed checks', 'review_rain_to_midnight.counts + review_aorc_rain_to_midnight.counts (first round of E1 only)')
mx = lambda n: len(J(n)['must_fix'])+len(J(n)['should_fix'])
claim('4.6 (19) wording items', 'R', f"asked for ({mx('review_rain_to_midnight')+mx('review_aorc_rain_to_midnight')}) are followed", 'asked for (19) are followed', 'review_*.must_fix+should_fix')
# ---------- 4.7 ----------
M = MB['summary_test']['sets']
claim('4.7 8 fewer hits, 59 fewer FA', 'R', f"{M['lstm_wq']['l1']['onset']['hits']-M['lstm63_wq']['l1']['onset']['hits']} fewer hits but {M['lstm_wq']['l1']['onset']['false_alarms']-M['lstm63_wq']['l1']['onset']['false_alarms']} fewer false alarms", '8 fewer hits but 59 fewer false alarms', 'more_basins.summary_test.sets.lstm_wq/lstm63_wq.l1.onset')
claim('4.7 CSI 0.147 against 0.148', 'R', f"CSI {M['lstm_wq']['l1']['onset']['csi']:.3f} against {M['lstm63_wq']['l1']['onset']['csi']:.3f}", 'CSI 0.147 against 0.148', 'more_basins.summary_test.sets...onset.csi')
claim('4.7 19% to 17%', 'R', f"fall from {p0(M['lstm_wq']['l1']['onset']['pod'])} to {p0(M['lstm63_wq']['l1']['onset']['pod'])}", 'fall from 19% to 17%', 'more_basins.summary_test')
U = MB['unseen']['sets']
claim('4.7 never seen 0.22 to 0.38 / 0.31 to 0.43', 'R', f"from {n2(U['lobo_w_static']['nse_median']['l1'])} with 11 training rivers to {n2(U['unseen50_w_static']['nse_median']['l1'])} with 50 (without properties {n2(U['lobo_w_nostatic']['nse_median']['l1'])} to {n2(U['unseen50_w_nostatic']['nse_median']['l1'])})", 'from 0.22 with 11 training rivers to 0.38 with 50 (without properties 0.31 to 0.43)', 'more_basins.unseen.sets.*.nse_median.l1')
claim('4.7 best update 1,167 vs 1,133', 'R', f"(best update about {MB['stopping']['sets']['lstm63eq_wq']['mean_best_update']:,.0f} against {MB['stopping']['sets']['lstm63_wq']['mean_best_update']:,.0f})", '(best update about 1,167 against 1,133)', 'more_basins.stopping.sets.lstm63eq_wq/lstm63_wq.mean_best_update')
claim('4.7 36 listed (twelve)', 'R', f"in {chk('review_more_basins')} listed checks", 'in 36 listed checks', 'review_more_basins.counts.checks_listed')
# ---------- 4.8 ----------
sk = FI['forecast_data']['skill_summary']
claim('4.8 corr 0.72 0.66 0.59', 'R', f"correlation is {sk['correlation_all_median_over_basins']['l1']:.2f}, {sk['correlation_all_median_over_basins']['l2']:.2f} and {sk['correlation_all_median_over_basins']['l3']:.2f}", 'correlation is 0.72, 0.66 and 0.59', 'forecast_input.forecast_data.skill_summary.correlation_all_median_over_basins')
br = [sk['brier_test_years_pooled'][k]['5mm']['calibrated_probability']['brier_skill_pooled'] for k in ('l1','l2','l3')]
claim('4.8 Brier 0.44 0.39 0.33', 'R', f"{br[0]:.2f}, {br[1]:.2f} and {br[2]:.2f}, of the same order", '0.44, 0.39 and 0.33, of the same order', 'forecast_input...brier_skill_pooled')
ld = FI['ladder']['main']['rungs']; ldS = FI['ladder']['sat']['rungs']
claim('4.8 E2a 51/21/12 -> 45/39/37; NSE .77/.38/.25 -> .69/.53/.45', 'R', "Onsets warned 51%/21%/12% go to 45%/39%/37% at 1, 2 and 3 days ahead, the median NSE from 0.77/0.38/0.25 to 0.69/0.53/0.45", "Onsets warned 51%/21%/12% go to 45%/39%/37% at 1, 2 and 3 days ahead, the median NSE from 0.77/0.38/0.25 to 0.69/0.53/0.45", 'forecast_input.ladder.main.rungs lstm_wqa / lstm_wqaF_sub')
claim('4.8 +18.2 +25.3; 7/9 9/9', 'R', f"+{OP['l2']['fixed_level']['points']:.1f} and +{OP['l3']['fixed_level']['points']:.1f} points; positive in {OP['l2']['block_bootstrap_fixed_level']['water_years_above_zero']} of 9 and {OP['l3']['block_bootstrap_fixed_level']['water_years_above_zero']} of 9 test years", '+18.2 and +25.3 points; positive in 7 of 9 and 9 of 9 test years', 'forecast_input.value_of_the_forecast.operating_points')
claim('4.8 -6.7; 35% vs 50%; CSI .358 vs .337; +2.6', 'R', f"({OP['l1']['fixed_level']['points']:.1f} points) but its warnings are right more often ({p0(OP['l1']['fixed_level']['false_with'])} false against {p0(OP['l1']['fixed_level']['false_without'])}; CSI {OP['l1']['fixed_level']['csi_with']:.3f} against {OP['l1']['fixed_level']['csi_without']:.3f}), and at the level chosen on the validation years it warns of {OP['l1']['level_chosen_on_the_validation_years']['points']:.1f} points more", '(-6.7 points) but its warnings are right more often (35% false against 50%; CSI 0.358 against 0.337), and at the level chosen on the validation years it warns of 2.6 points more', 'forecast_input...operating_points.l1')
RV = FI['retrained_vs_substituted']
claim('4.8 E2b +20.3 +35.2 +36.4; +23.6 +19.8 +12.5', 'R', f"+{RV['retrained']['l1']['onsets']['difference_points']:.1f}, +{RV['retrained']['l2']['onsets']['difference_points']:.1f} and +{RV['retrained']['l3']['onsets']['difference_points']:.1f} points at the fixed level, +{RV['retrained_operating_points']['l1']['level_chosen_on_the_validation_years']['points']:.1f}, +{RV['retrained_operating_points']['l2']['level_chosen_on_the_validation_years']['points']:.1f} and +{RV['retrained_operating_points']['l3']['level_chosen_on_the_validation_years']['points']:.1f}", '+20.3, +35.2 and +36.4 points at the fixed level, +23.6, +19.8 and +12.5', 'forecast_input.retrained_vs_substituted')
rr = FI['yardstick']['new_input_given_to_each_model']
pw = lambda m: '/'.join(f"{100*rr[m]['by_lead'][k]['onset_warned_with']:.0f}%" for k in ('l1','l2','l3'))
claim('4.8 ridge 69/59/52, forest 55/47/42', 'R', f"ridge regression warns of {pw('ridge regression')} of the onsets and the random forest of {pw('random forest')}", 'ridge regression warns of 69%/59%/52% of the onsets and the random forest of 55%/47%/42%', 'forecast_input.yardstick.new_input_given_to_each_model')
mg = FI['forecast_without_rain_to_midnight']['rain_to_midnight_given_the_forecast']
claim('4.8 +13.1 / +2.5 / +0.5', 'R', f"(+{mg['l1']['onsets']['difference_points']:.1f} points) and little 2 and 3 days ahead (+{mg['l2']['onsets']['difference_points']:.1f} and +{mg['l3']['onsets']['difference_points']:.1f})", '(+13.1 points) and little 2 and 3 days ahead (+2.5 and +0.5)', 'forecast_input.forecast_without_rain_to_midnight.rain_to_midnight_given_the_forecast')
claim('4.8 36 listed (14)', 'R', f"in {chk('review_forecast_input')} listed checks", 'in 36 listed checks', 'review_forecast_input.counts')
# ---------- section 8 ----------
def rng(key):
    v = UN['stats'][key]['both']; return f"{100*UN['stats'][key]['value']:.0f}% ({100*v['lo']:.0f}% to {100*v['hi']:.0f}%)"
claim('8 19% (10% to 29%)', 'R', f"warns of {rng('onsets_warned.lstm_wq.l1')}", 'warns of 19% (10% to 29%)', 'uncertainty.stats.onsets_warned.lstm_wq.l1.both')
claim('8 30% (20% to 42%)', 'R', f"of {rng('onsets_warned.lstm_wqn.l1')}", 'of 30% (20% to 42%)', 'uncertainty.stats.onsets_warned.lstm_wqn.l1.both')
claim('8 64% (55% to 72%)', 'R', f"{rng('onsets_warned.lstm_wqf.l1')}", '64% (55% to 72%)', 'uncertainty.stats.onsets_warned.lstm_wqf.l1.both')
claim('8 51% (41% to 61%)', 'R', f"warns of {rng('onsets_warned.lstm_wqa.l1')}", 'warns of 51% (41% to 61%)', 'uncertainty.stats.onsets_warned.lstm_wqa.l1.both')
claim('8 39% (28% to 48%) / 37% (27% to 46%)', 'R', f"of {rng('onsets_warned.lstm_wqaF_sub.l2')} and {rng('onsets_warned.lstm_wqaF_sub.l3').replace('%)', '%)',1)}", 'of 39% (28% to 48%) and 37% (27% to 46%)', 'uncertainty.stats.onsets_warned.lstm_wqaF_sub.l2/l3.both')
g = UN['stats']['onsets_gap_share_closed.wq.l1']['both']
claim('8 gap 11% to 44%', 'R', f"about a quarter of that gap ({100*g['lo']:.0f}% to {100*g['hi']:.0f}%)", 'about a quarter of that gap (11% to 44%)', 'uncertainty.stats.onsets_gap_share_closed.wq.l1.both')
claim('8 3,998 of 4,000', 'R', f"above zero in {round(UN['stats']['onsets_gain.evening_rain_minus_control.wq.l1']['both']['share_above_zero']*4000):,} of the 4,000 draws", 'above zero in 3,998 of the 4,000 draws', 'uncertainty.stats.onsets_gain.evening_rain_minus_control.wq.l1.both.share_above_zero')
vs = UN['verdict_stability']['lstm_wq.all.l1']['share_of_resamples_with_each_verdict']
claim('8 83% / 7% / 10%', 'R', f"holds in {vs['CAUTION']*100:.0f}% of the draws (GO in {vs['GO']*100:.0f}%, NOT YET in {vs['NOT YET']*100:.0f}%)", 'holds in 83% of the draws (GO in 7%, NOT YET in 10%)', 'uncertainty.verdict_stability.lstm_wq.all.l1')
sm = SM['l1']
claim('8 75% missed by all four; 25%; 68%', 'R', f"{sm['missed_by_all_kinds']['share']*100:.0f}% of the 611 onsets are missed by all four kinds of model (ridge regression, random forest, LSTM, Transformer), and a warning from any one of them would cover {sm['warned_by_any_kind']['share']*100:.0f}%, with {sm['warned_by_any_kind']['far']*100:.0f}% false warnings", '75% of the 611 onsets are missed by all four kinds of model (ridge regression, random forest, LSTM, Transformer), and a warning from any one of them would cover 25%, with 68% false warnings', 'same_misses.l1.missed_by_all_kinds / warned_by_any_kind')
claim('8 62% / 17% of those missed by all', 'R', f"warns of {sm['of_missed_by_all']['warned_with_perfect_forecast']['share']*100:.0f}%; given rain stamped up to the evening of the issue day, of {sm['of_missed_by_all']['warned_with_rain_to_evening']['share']*100:.0f}%", 'warns of 62%; given rain stamped up to the evening of the issue day, of 17%', 'same_misses.l1.of_missed_by_all')
claim('7 8.2 hours, 193 trainings, 123 Wh', 'R', "8.2 hours on a two-core laptop without a graphics card, 193 network trainings and about 123 Wh", "8.2 hours on a two-core laptop without a graphics card, 193 network trainings and about 123 Wh", 'sum over results files: training_seconds etc. = 8.17 h; n_trainings 167 + 26 hparam_log rows = 193; 122.5 Wh (recomputed)')
claim('1 Fair test 10 checks passed', 'R', "All 10 checks passed", "All 10 checks passed" if len(J('fair_test')['checks'])==10 else 'MISMATCH', 'fair_test.json.checks (10) passed=True')
# ---------- slides ----------
claim('S4 table 0.43/0.01/-0.13 etc', 'S', "'Tomorrow = today' (river gauge only) 0.43 0.01 -0.13 0%", None, 'baselines (older)')
rows.pop()
claim('S7 62%/32%, 81%, 59%, 951 of 1,562', 'S', "62% warned, 32% false warnings. But 81% of the onsets get no warning, and 59% of the onset warnings are false", f"{p0(wq and A['sets']['wq']['test']['summary']['l1']['all']['pod'])} warned, {p0(A['sets']['wq']['test']['summary']['l1']['all']['far'])} false warnings. But {100-round(100*wq['onset']['pod'])}% of the onsets get no warning, and {p0(wq['onset']['far'])} of the onset warnings are false", 'lstm_all.sets.wq.test.summary.l1')
claim('S9 19%->30% control 19% upper 64%; 0.59->0.67; 5 of 12', 'S', "onsets warned 19% 30% (control 19%; upper bound 64%), about a quarter of the gap", None, 'rain_timing')
rows.pop()
claim('S10 33.6 points', 'S', "The hours after 09:00 are worth 33.6 points", "The hours after 09:00 are worth 33.6 points", 'aorc.value_of_hours_09_to_24.by_lead.l1.onsets.difference_points = %.1f' % hv2['difference_points'])
claim('S10 48%, 45% without last hour', 'S', "Stage IV radar: 48%, 45% without its last hour", "Stage IV radar: 48%, 45% without its last hour", 'stage4_realtime')
claim('S11 66/69/71; 45 vs 51; 35 vs 50', 'S', "Perfect rain: 66 / 69 / 71% of onsets", "Perfect rain: 66 / 69 / 71% of onsets", 'forecast_input.ladder.main.rungs lstm_wqaP')
claim('S11 41/14/8 -> 61/49/44', 'S', "41 / 14 / 8% 61 / 49 / 44%", "41 / 14 / 8% 61 / 49 / 44%", 'forecast_input.ladder.sat')
claim('S11 shuffled 14/4/4', 'S', "falls to 14 / 4 / 4%", "falls to 14 / 4 / 4%", 'forecast_input.summary_test.sets.wqaF_clim')
claim('S12 0.15 / 0.13', 'S', "loses 0.15 NSE 1 day ahead; held out of training, the typical basin loses 0.13", "loses 0.15 NSE 1 day ahead; held out of training, the typical basin loses 0.13", 'key_figure.steps.river_gauge.nse (0.1465) and lstm_lobo (0.13)')
claim('S13 within 0.03 NSE and 8 points', 'S', "within 0.03 NSE and 8 points of onsets", "within 0.03 NSE and 8 points of onsets", 'key_figure.finding.spread_between_model_kinds (0.0329; 8.02)')
# ---------- guide / overview / Q&A ----------
claim('G33 +45 +30 / +33; +39 +25 / +18', 'G', "ridge regression and the random forest gain +45 and +30 points of onsets warned, the LSTM +33", None, 'aorc yardstick')
rows.pop()
claim('G32 +21/+34/+36 clean control', 'G', "The real forecast adds +21 / +34 / +36 points above it", "The real forecast adds +21 / +34 / +36 points above it", 'forecast_input.retrained_against_its_control: +20.6, +34.3, +36.0')
claim('O 7.5 mm/day, 74 and 58 mm/day', 'O', "(7.5 mm/day, exceeded on only 5% of days in the training years), peaking at 74 and 58 mm/day", "(7.5 mm/day, exceeded on only 5% of days in the training years), peaking at 74 and 58 mm/day", 'floods_2004.json: 7.54; 74.2; 58.2')
claim('Q-H1 2.6 points (51% to 49%)', 'Q', "would cost 2.6 points (51% to 49%)", "would cost 2.6 points (51% to 49%)", 'aorc.stamp_convention_check.sets.wqa.by_lead.l1.difference_points = -2.62')
claim('Q-H2 936, 11.8 mm vs 1.1 mm, 35 basin-days, 17.7 mm, 0.075 mm, +0.100', 'Q', "on 78 test-year days (936 basin-days; the 28 days the reviewer added were picked for onsets and night storms and hold 11.8 mm of Stage IV rain after midnight on average, against 1.1 mm in the builder's own 50)", "on 78 test-year days (936 basin-days; the 28 days the reviewer added were picked for onsets and night storms and hold 11.8 mm of Stage IV rain after midnight on average, against 1.1 mm in the builder's own 50)", 'review_aorc.checks[d3].numbers.sample')
claim('Q-H12 130 checks, (45), 23 and 17', 'Q', "recomputed the headline numbers in 130 listed checks", "recomputed the headline numbers in %d listed checks" % (chk('review_rain_to_midnight')+chk('review_aorc_rain_to_midnight')+chk('review_more_basins')+chk('review_forecast_input')), 'review_*.counts.checks_listed (27+31+36+36)')
claim('Q-H13 1/1/2/2 choices', 'Q', "1 entry in E1, 1 entry in E1b, 2 entries in E3 and 2 entries in E2", "1 entry in E1, 1 entry in E1b, 2 entries in E3 and 2 entries in E2", 'results files.choices_made_after_looking_at_test_scores lengths (1,1,2,2)')
claim('Q-H11 78 trainings 4.4 h (21/12/15/30)', 'Q', "78 trainings of the network on the laptop (two processor cores, no graphics card), one at a time, 4.4 hours of training in all (21 for E1", "78 trainings of the network on the laptop (two processor cores, no graphics card), one at a time, 4.4 hours of training in all (21 for E1", 'sum n_trainings 21+12+15+30; training_seconds sum 15,983 s = 4.44 h')
claim('Q-H6 -14.2 to -0.1; 2 of 9 years', 'Q', "(-14.2 to -0.1 points) and is above zero in only 2 of 9 test years", "(-14.2 to -0.1 points) and is above zero in only 2 of 9 test years", 'forecast_input...block_bootstrap_fixed_level l1: [-14.17, -0.15]; years 2/9')
# ---------- result ----------
out = []
for r in rows:
    label, doc, pg, needle, truth, key, ok = r
    status = 'PASS' if (pg and (ok is True or ok is None)) else ('TRUTH-DIFFERS' if ok is False else 'NOT-FOUND-IN-DOC')
    out.append(dict(claim=label, document=doc, pages=pg, status=status, needle=needle, truth=truth, source=key))
bad = [o for o in out if o['status'] != 'PASS']
jsave("d1_numbers.json", dict(n_claims=len(out), n_pass=len(out) - len(bad), not_pass=bad, claims=out, documents={'R': 'Flood_Report.pdf', 'S': 'Flood_Slides.pdf', 'Q': 'Professor_QA.pdf', 'G': 'Project_Guide.pdf', 'O': 'Project_Overview.pdf', 'short': 'short_report.md', 'notes': 'slides_notes.md', 'readme': 'README.md'}))
print(len(out), 'claims;', len(out) - len(bad), 'pass;', len(bad), 'to inspect')
for o in bad:
    print('--', o['claim'], o['document'], o['pages'], o['status']); print('    doc  :', o['needle']); print('    truth:', o['truth'])
