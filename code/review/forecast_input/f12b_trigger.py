"""E2 review, check (i) continued: how much of the clear loss at 1 day ahead (-6.7 points of onsets warned at the plain trigger 1.0) and of the gains at 2 and 3 days depends on the warning trigger. The reviewer's own sweep over
the warning level (a warning when the forecast exceeds m x the high-flow level, m = 0.30 .. 1.50 in steps of 0.05; onsets pooled over the basins; each set on its own split) for the main-split sets (wq, wqa, wqaF_sub,
wqaF_clim, wqaP, wqF_sub), the E2b sets (wq_s, wqa_s, wqaF_s, wqaF_clim_s, wqaP_s) and the simple models: (1) onsets warned and false warnings at the plain trigger and at the trigger the project chooses on the VALIDATION years
(best pooled CSI of the onsets: the second reading of the verdicts); (2) a threshold-free look on the TEST years: the highest onset share warned at a false-warning share of at most 40% and 50%, and the best CSI over the levels
(chosen on the test years: an optimistic bound that treats both sets alike). Reads only the saved prediction sets. About 40 seconds."""
import sys; sys.dont_write_bytecode = True
from f_common import *
import flood as F
from f_scoring import *

R = jload("results/forecast_input.json")
D = F.load_data(); ids = list(D)
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
cache = {}
def sweep(name, lead, which):
    key = (name, which)
    if key not in cache: cache[key] = load_npz(name)
    dates, basins, obs, pred = cache[key]; sp = split_of(name); thr = thresholds(basins, dates, sp); a, b = sp[which]; sel = part(dates, a, b)
    rows = []
    for m in LEVELS:
        H = M = Fa = 0
        for i, g in enumerate(basins):
            o, f = obs[i], pred[i, :, lead - 1]; issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
            ok = sel & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g]); high, warn = (o > thr[g]), (f > m * thr[g])
            H += int((high & warn & ok).sum()); M += int((high & ~warn & ok).sum()); Fa += int((~high & warn & ok).sum())
        rows.append((m, H / (H + M), Fa / (H + Fa) if H + Fa else 0.0, H / (H + M + Fa)))
    return rows
SETS = {"wq": "lstm_wq", "wqa": "lstm_wqa", "wqaF_sub": "lstm_wqaF_sub", "wqaF_clim": "lstm_wqaF_clim", "wqaP": "lstm_wqaP", "wqF_sub": "lstm_wqF_sub", "wq_s": "lstm_wq_s", "wqa_s": "lstm_wqa_s", "wqaF_s": "lstm_wqaF_s", "wqaF_clim_s": "lstm_wqaF_clim_s", "wqaP_s": "lstm_wqaP_s",
        "base_linreg_wqa": "base_linreg_wqa", "base_linreg_wqaF": "base_linreg_wqaF", "base_rf_wqa": "base_rf_wqa", "base_rf_wqaF": "base_rf_wqaF"}
out = {}
for key, name in SETS.items():
    out[key] = {}
    for k in KS:
        te = sweep(name, k, "test"); va = sweep(name, k, "val")
        csi_v = np.array([r[3] for r in va]); best = np.flatnonzero(csi_v >= csi_v.max() - 1e-12); lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
        at = lambda m: next(r for r in te if abs(r[0] - m) < 1e-9)
        pod_far = lambda cap: max([r[1] for r in te if r[2] <= cap] or [float("nan")])
        out[key][f"l{k}"] = dict(plain=dict(pod=at(1.0)[1], far=at(1.0)[2], csi=at(1.0)[3]), validation_chosen_level=lvl, at_chosen=dict(pod=at(lvl)[1], far=at(lvl)[2], csi=at(lvl)[3]), best_pod_with_far_at_most_40=pod_far(0.40), best_pod_with_far_at_most_50=pod_far(0.50), best_csi_over_levels_test=max(r[3] for r in te))
def show(keys, label):
    print(label)
    for key in keys:
        print(f"   {key:18s} " + " | ".join(f"lead {k}: plain {100 * out[key][f'l{k}']['plain']['pod']:4.1f}/{100 * out[key][f'l{k}']['plain']['far']:2.0f}% csi {out[key][f'l{k}']['plain']['csi']:.3f}; level {out[key][f'l{k}']['validation_chosen_level']:.2f}: {100 * out[key][f'l{k}']['at_chosen']['pod']:4.1f}/{100 * out[key][f'l{k}']['at_chosen']['far']:2.0f}% csi {out[key][f'l{k}']['at_chosen']['csi']:.3f}" for k in KS))
show(["wqa", "wqaF_sub", "wqaP", "wqaF_clim", "wqF_sub"], "main split: onsets warned / false warnings, csi; plain trigger, then the level chosen on the validation years")
show(["wqa_s", "wqaF_s", "wqaP_s", "wqaF_clim_s", "wq_s"], "satellite-era split:")
show(["base_linreg_wqa", "base_linreg_wqaF", "base_rf_wqa", "base_rf_wqaF"], "simple models:")
print("highest onset share warned with at most 40% / 50% false warnings (test years, any level):")
for key in ("wqa", "wqaF_sub", "wqaP", "wqa_s", "wqaF_s", "base_linreg_wqaF", "base_rf_wqaF"):
    print(f"   {key:18s} " + " | ".join(f"lead {k}: {100 * out[key][f'l{k}']['best_pod_with_far_at_most_40']:.1f}% / {100 * out[key][f'l{k}']['best_pod_with_far_at_most_50']:.1f}%  (best csi {out[key][f'l{k}']['best_csi_over_levels_test']:.3f})" for k in KS))
d = {k: dict(plain_points=100 * (out["wqaF_sub"][f"l{k}"]["plain"]["pod"] - out["wqa"][f"l{k}"]["plain"]["pod"]), chosen_points=100 * (out["wqaF_sub"][f"l{k}"]["at_chosen"]["pod"] - out["wqa"][f"l{k}"]["at_chosen"]["pod"]), csi_plain=out["wqaF_sub"][f"l{k}"]["plain"]["csi"] - out["wqa"][f"l{k}"]["plain"]["csi"],
             csi_chosen=out["wqaF_sub"][f"l{k}"]["at_chosen"]["csi"] - out["wqa"][f"l{k}"]["at_chosen"]["csi"], pod_at_far50=100 * (out["wqaF_sub"][f"l{k}"]["best_pod_with_far_at_most_50"] - out["wqa"][f"l{k}"]["best_pod_with_far_at_most_50"])) for k in KS}
d_s = {k: dict(plain_points=100 * (out["wqaF_s"][f"l{k}"]["plain"]["pod"] - out["wqa_s"][f"l{k}"]["plain"]["pod"]), chosen_points=100 * (out["wqaF_s"][f"l{k}"]["at_chosen"]["pod"] - out["wqa_s"][f"l{k}"]["at_chosen"]["pod"]), csi_plain=out["wqaF_s"][f"l{k}"]["plain"]["csi"] - out["wqa_s"][f"l{k}"]["plain"]["csi"],
               csi_chosen=out["wqaF_s"][f"l{k}"]["at_chosen"]["csi"] - out["wqa_s"][f"l{k}"]["at_chosen"]["csi"], pod_at_far50=100 * (out["wqaF_s"][f"l{k}"]["best_pod_with_far_at_most_50"] - out["wqa_s"][f"l{k}"]["best_pod_with_far_at_most_50"])) for k in KS}
out["gain_of_the_forecast_E2a_wqaF_sub_minus_wqa"] = d; out["gain_of_the_forecast_E2b_wqaF_s_minus_wqa_s"] = d_s
print("wqaF_sub minus wqa (E2a), points of onsets warned at the plain trigger / at the validation-chosen levels / at a false-warning share of at most 50% (test years), and the CSI at the two levels:")
for k in KS: print(f"   lead {k}: {d[k]['plain_points']:+.1f} / {d[k]['chosen_points']:+.1f} / {d[k]['pod_at_far50']:+.1f} points; csi {d[k]['csi_plain']:+.3f} / {d[k]['csi_chosen']:+.3f}")
print("wqaF_s minus wqa_s (E2b):")
for k in KS: print(f"   lead {k}: {d_s[k]['plain_points']:+.1f} / {d_s[k]['chosen_points']:+.1f} / {d_s[k]['pod_at_far50']:+.1f} points; csi {d_s[k]['csi_plain']:+.3f} / {d_s[k]['csi_chosen']:+.3f}")
# the json's own records of the validation-chosen level for comparison
V = R["verdicts"]["sets"]
dev = max(abs(out[k]["l" + str(l)]["at_chosen"]["pod"] - V[k]["l" + str(l)]["chosen_on_validation_onsets"]["onset"]["pod"]) for k in ("wqa", "wqaF_sub", "wqaP", "wqa_s", "wqaF_s", "base_linreg_wqaF", "base_rf_wqaF") for l in KS)
out["max_dev_vs_verdict_records"] = dev
print(f"my validation-chosen onset records against verdicts.chosen_on_validation_onsets of the JSON: largest difference {dev:.1e}")
jsave("f12b_trigger.json", out)
