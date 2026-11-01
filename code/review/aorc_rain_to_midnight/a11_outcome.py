"""AORC review, check (g): outcome / outcome_caveat / outcome_full. Every clause rebuilt as a sentence from the reviewer's own counts of results/preds and compared with the text of the results file; the reading of the LSTM
against the simple models by the rule of report section 4.2 with own NSE and own Wilcoxon (the clause 'although the LSTM is clearly ahead of at least one of them' rests on it); the three comparisons that
bear on the last clause ('the mis-stamped table itself costs skill'): product alone (AORC over the table's own window as an extra column), clock alone (AORC to midnight against the same product over the table's window),
and both together (the replaced table). About 15 seconds."""
import sys; sys.dont_write_bytecode = True
from scipy.stats import wilcoxon
from a_common import *
import flood as F
from a_scoring import *

R = jload("results/aorc_rain_to_midnight.json")
D = F.load_data(); ids = list(D)
SEEDS = (0, 1, 2)
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
out, bad = {}, []
def check(label, cond, detail=""):
    if not cond: bad.append((label, detail))
    return bool(cond)

names = ["lstm_wq", "lstm_wqh", "lstm_wqn", "lstm_wqa", "lstm_wqa_to09", "lstm_wqa_utc", "lstm_wqa_rep", "base_linreg_wqa", "base_rf_wqa", "base_linreg_wq", "base_rf_wq"]
C = {n: counts(n, 1) for n in names}
sd = {x: [counts(f"lstm_{x}_s{s}", 1)["pod"] for s in SEEDS] for x in ("wq", "wqh", "wqn", "wqa", "wqa_to09", "wqa_utc", "wqa_rep")}
above = lambda a, b: min(a) > max(b)
def flags(name, lead=1):
    t = onset_table(name, lead); fl, dd = [], []
    for g in ids:
        d, h, w = t[g]; fl.append(w[h]); dd += [(g, str(x.date())) for x in d[h]]
    return np.concatenate(fl), dd
fa, da = flags("lstm_wqa"); fc, dc = flags("lstm_wqa_to09"); check("paired", da == dc)
only_a, only_c = int((fa & ~fc).sum()), int((~fa & fc).sum())
n_on = C["lstm_wqa"]["n_onsets"]; th = C["lstm_wqa"]
# the NSE reading of the LSTM against each simple model, own implementation of the rule of section 4.2
def nse_arr(name):
    d = nse_per_basin(name, 1); return np.array([d[g] for g in ids])
nl = nse_arr("lstm_wqa"); nse_seed = [float(np.median(list(nse_per_basin(f"lstm_wqa_s{s}", 1).values()))) for s in SEEDS]
readings = {}
for n in ("base_linreg_wqa", "base_rf_wqa"):
    nb = nse_arr(n); bm = float(np.median(nb)); n_hi = int((nl > nb).sum())
    try: p = float(wilcoxon(nl, nb).pvalue)
    except ValueError: p = float("nan")
    below = bm < min(nse_seed); most = n_hi > len(ids) / 2; few = n_hi < len(ids) / 2; small = p < 0.05
    rd = ("lstm" if below else "lstm_avg") if (most and small) else ("simple" if (few and small) else ("lstm_med" if (most and below) else "tie"))
    readings[n] = dict(reading=rd, n_higher=n_hi, p=p, baseline_nse=bm, lstm_nse=float(np.median(nl)), lstm_seed_nse=nse_seed, baseline_below_every_seed=bool(below),
                       json_reading=R["yardstick"]["lstm_against_simple"][f"lstm_wqa__{n}"]["l1"]["reading"])
    check(f"reading vs {n}", rd == readings[n]["json_reading"], (rd, readings[n]["json_reading"]))
    print(f"LSTM (wqa) against {n} by the NSE: median NSE {readings[n]['lstm_nse']:.3f} against {bm:.3f}, higher in {n_hi} of 12 basins, Wilcoxon p {p:.4f}; the simple model is {'below every seed' if below else 'inside or above the seed spread'} (seeds {[round(x, 3) for x in nse_seed]}) -> reading '{rd}' (JSON: '{readings[n]['json_reading']}')")
out["readings"] = readings
# the sentence, rebuilt
share_clause = lambda nm, label: (f"{label} gives the same share ({pc(th['pod'])}, {th['hits']} hits of {n_on} in both)" if (C[nm]["hits"], C[nm]["misses"]) == (th["hits"], th["misses"]) else
                                  f"{label} gives {'a higher' if C[nm]['pod'] > th['pod'] else 'a lower'} share ({pc(C[nm]['pod'])}, {C[nm]['hits']} hits of {n_on}) than the AORC rain to midnight ({pc(th['pod'])}, {th['hits']} hits of {n_on})")
era5_c = share_clause("lstm_wqh", "the ERA5 rain to midnight of script 15"); nldas_c = share_clause("lstm_wqn", "the NLDAS rain up to the evening")
seed_c = f"the three single trainings with the AORC rain to midnight span {p1(min(sd['wqa']))} to {p1(max(sd['wqa']))}"
met = above(sd["wqa"], sd["wqn"]) and above(sd["wqa"], sd["wqa_to09"]) and above(sd["wqa"], sd["wq"]) and not above(sd["wqa_to09"], sd["wq"])
rule_c = f"the success rule of the work order (clearly above the NLDAS result in every training, control flat) is {'met' if met else 'not met'}"
cav = f"{era5_c[0].upper() + era5_c[1:]}; {nldas_c}; {seed_c}; {rule_c}."
flat = not above(sd["wqa_to09"], sd["wq"]); ab = above(sd["wqa"], sd["wqa_to09"])
gain = 100 * (C["lstm_wqa"]["pod"] - C["lstm_wqa_to09"]["pod"])
head = f"The control with the same AORC rain over the table's own window is {'flat' if flat else 'not flat'} ({p1(C['lstm_wqa_to09']['pod'])} against {p1(C['lstm_wq']['pod'])} with the rain table alone)"
only = f"({only_a} onsets warned only by the AORC rain to midnight against {only_c} only by that control)"
cav += " " + head + (f", so the gain of {gain:.1f} points comes from the hours after about 09:00 {only}." if flat and ab else f" [other branch]")
d_era5 = 100 * (C["lstm_wqa"]["pod"] - C["lstm_wqh"]["pod"]); grows = above(sd["wqa"], sd["wqh"]) and above(sd["wqa"], sd["wqa_to09"]) and above(sd["wqa"], sd["wq"])
cav += (f" Against the ERA5 rain to midnight over the same hours the AORC rain warns of {d_era5:+.1f} points ({C['lstm_wqa']['hits']} against {C['lstm_wqh']['hits']} hits of {n_on}; every training with AORC above every training with ERA5: {'yes' if above(sd['wqa'], sd['wqh']) else 'no'}), "
        f"so the E1 gain {'grows' if grows else 'holds'} with it (step 1b of the roadmap).")
g_r, g_f, g_l = (100 * (C[a]["pod"] - C[b]["pod"]) for a, b in (("base_linreg_wqa", "base_linreg_wq"), ("base_rf_wqa", "base_rf_wq"), ("lstm_wqa", "lstm_wq")))
ahead = [readings[n]["reading"] == "lstm" for n in ("base_linreg_wqa", "base_rf_wqa")]
tail2 = "so the gain does not need the network." if min(g_r, g_f) > 0 and not any(ahead) else "so the gain does not need the network, although the LSTM is clearly ahead of at least one of them." if min(g_r, g_f) > 0 else "so the simple models do not gain as the LSTM does."
cav += f" Ridge regression and the random forest given the same input warn of {p1(C['base_linreg_wqa']['pod'])} and {p1(C['base_rf_wqa']['pod'])} (the new input adds {g_r:.1f} and {g_f:.1f} points to them, {g_l:.1f} to the LSTM), {tail2}"
rep_above = above(sd["wqa_rep"], sd["wq"]); rep_below = above(sd["wq"], sd["wqa_rep"])
rd_rep = ("the replaced table warns of more onsets than the rain table in every training: the mis-stamped table itself costs skill" if rep_above else
          "the replaced table warns of fewer onsets than the rain table in every training: the Daymet table is better on its own" if rep_below else "no clear difference in the onsets between the replaced table and the rain table")
span = lambda v: f"{p1(min(v))} to {p1(max(v))}"
cav += f" With the AORC rain in the place of the Daymet rain column (no extra column) the LSTM warns of {pc(C['lstm_wqa_rep']['pod'])} of the onsets against {pc(C['lstm_wq']['pod'])} with the Daymet rain table ({span(sd['wqa_rep'])} against {span(sd['wq'])} in the single trainings): {rd_rep}."
short = (f"Given the rain of the whole issue day up to local midnight (hourly AORC rain, 1 km, from 00:00 to 24:00 by the local wall clock) as one extra input to the LSTM with the river gauge, onsets warned 1 day ahead go from "
         f"{pc(C['lstm_wq']['pod'])} to {pc(C['lstm_wqa']['pod'])} (control {pc(C['lstm_wqa_to09']['pod'])}); the verdict is {R['verdicts']['sets']['wqa']['l1']['plain']['onset']['verdict']}.")
check("outcome", R["outcome"] == short, (short, R["outcome"])); check("outcome_caveat", R["outcome_caveat"] == cav, (cav, R["outcome_caveat"])); check("outcome_full = outcome + ' ' + outcome_caveat", R["outcome_full"] == R["outcome"] + " " + R["outcome_caveat"])
print("\noutcome and outcome_caveat rebuilt from my own counts equal the text of the file:", R["outcome"] == short, R["outcome_caveat"] == cav, "| outcome_full is the two joined:", R["outcome_full"] == R["outcome"] + " " + R["outcome_caveat"])
if R["outcome_caveat"] != cav: print("  MINE:", cav, "\n  JSON:", R["outcome_caveat"])
out["clauses"] = dict(hits={n: C[n]["hits"] for n in names}, pods={n: C[n]["pod"] for n in names}, seeds_percent={k: [round(100 * x, 2) for x in v] for k, v in sd.items()}, only_aorc=only_a, only_control=only_c, gain_over_control_points=gain, gain_over_era5_points=d_era5,
                      gains_points=dict(ridge=g_r, forest=g_f, lstm=g_l), success_rule_met=met, control_flat=flat, every_wqa_seed_above_every_wqh_seed=above(sd["wqa"], sd["wqh"]), rep_above_every_wq_seed=rep_above, sentence_equal=(R["outcome_caveat"] == cav))
print(f"gain of the AORC rain to midnight over: the rain table alone {g_l:.1f} points; the same product over the table's window {gain:.1f}; ERA5 to midnight {d_era5:.1f}; the UTC-day window {100 * (C['lstm_wqa']['pod'] - C['lstm_wqa_utc']['pod']):.1f}")
print(f"onsets: LSTM {p1(C['lstm_wqa']['pod'])} ({C['lstm_wqa']['hits']}), ridge {p1(C['base_linreg_wqa']['pod'])} ({C['base_linreg_wqa']['hits']}), forest {p1(C['base_rf_wqa']['pod'])} ({C['base_rf_wqa']['hits']}): in onsets the LSTM is behind ridge by {100 * (C['base_linreg_wqa']['pod'] - C['lstm_wqa']['pod']):.1f} points and ahead of the forest by {100 * (C['lstm_wqa']['pod'] - C['base_rf_wqa']['pod']):.1f}; the clause 'the LSTM is clearly ahead of at least one' is the NSE reading")

# ---- the last clause ----------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nthe last clause ('the mis-stamped table itself costs skill'): what each comparison changes")
q = lambda a, b: (100 * (C[a]["pod"] - C[b]["pod"]), C[a]["hits"] - C[b]["hits"])
prod = q("lstm_wqa_to09", "lstm_wq"); clock = q("lstm_wqa", "lstm_wqa_to09"); both = q("lstm_wqa_rep", "lstm_wq"); extra = q("lstm_wqa", "lstm_wq"); repx = q("lstm_wqa_rep", "lstm_wqa")
print(f"  product alone, AORC over the table's own window as an EXTRA column (wqa_to09 - wq): {prod[0]:+.1f} points ({prod[1]:+d} hits); seeds {span(sd['wqa_to09'])} against {span(sd['wq'])}: every seed below every seed of wq: {above(sd['wq'], sd['wqa_to09'])}")
print(f"  clock alone (the same product over 00:00-24:00 against its own 09:00-09:00 window, both as extra columns; wqa - wqa_to09): {clock[0]:+.1f} points ({clock[1]:+d} hits)")
print(f"  product and clock together, replacing the Daymet column (wqa_rep - wq): {both[0]:+.1f} points ({both[1]:+d} hits); the same AORC rain as an extra column beside Daymet (wqa - wq): {extra[0]:+.1f}; replaced against extra column (wqa_rep - wqa): {repx[0]:+.1f} points ({repx[1]:+d} hits)")
print("  not run: AORC over the table's window REPLACING the Daymet column (the product with the table's clock and nothing else changed); without it the replaced-table result cannot be split into 'the product' and 'the clock' by a single comparison")
out["last_clause"] = dict(product_alone_points=prod[0], product_alone_hits=prod[1], clock_alone_points=clock[0], clock_alone_hits=clock[1], both_points=both[0], both_hits=both[1], extra_column_points=extra[0], replaced_vs_extra_points=repx[0], replaced_vs_extra_hits=repx[1],
                          wqa_to09_all_seeds_below_wq=above(sd["wq"], sd["wqa_to09"]), not_run="wqa_to09_rep: AORC over 09:00-09:00 in the place of the Daymet column")
out["differences"] = [(a, str(b)[:300]) for a, b in bad]
print("\nTOTAL differences between my sentences and the file:", len(bad), [(a, str(b)[:60]) for a, b in bad[:5]])
jsave("a11_outcome.json", out)
