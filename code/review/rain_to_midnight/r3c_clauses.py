"""ROUND 3, check (c): the arithmetic of what was added after round 2, recomputed with the reviewer's own code from results/preds (never from the script):
 c1  the two new clauses of outcome_caveat / outcome_full (the matched-window control is flat; the simple models given the same input), rebuilt as sentences from my own numbers and compared with the JSON's text
 c2  the new onset-level fields of the yardstick (yardstick.lstm_against_simple.<pair>.l<k>.onsets and onsets_in_words): 5 pairs x 3 leads, every number and every sentence
 c3  the standard errors in by_period (8 sets x 2 parts) and the sentence about them in the note on ERA5's change of 17 July 2009
 c4  the verdict records of the three baselines added to verdicts.sets (base_persistence, base_linreg_wq, base_rf_wq): own choice of the warning level on the validation years, own test counting
 c5  what_was_not_done[3] (the spread of single-training NSE against the validation NSE; value_of_hours_09_to_24 in NSE) and the two sentences appended to the readings value_of_hours_09_to_24 and
     value_of_evening_hours_beyond_utc_day, and figure_labels.nldas_if_utc_stamped
 c6  'about 4 hours' for IMERG-Early everywhere, and no '4 to 5 hours' left
About one minute."""
import sys; sys.dont_write_bytecode = True
import re, math, time
from scipy.stats import binomtest, wilcoxon
from r2_common import *
import flood as F
from r2_scoring import *

t0 = time.time()
R = jload("results/rain_to_midnight.json")
ids = list(data())
SEEDS = (0, 1, 2)
nanok = lambda a, b: (a is None and b != b) or (b is None and a != a) or (a != a and b != b) or abs(a - b) < 1e-9
p1 = lambda x: f"{100 * x:.1f}%"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
out, bad = {}, []
def check(label, cond, detail=""):
    if not cond: bad.append((label, detail))
    return bool(cond)

# ---- own tables (cached): one npz per set; onset tables at the plain level; per-basin onset shares
_c = {}
def cached(name):
    if name not in _c: _c[name] = load_npz(name)
    return _c[name]
def table(name, lead, a, b, level, kind):
    dates, basins, obs, pred = cached(name); thr = thresholds(basins, dates); sel = part(dates, a, b)
    H = M = Fa = 0
    for i, g in enumerate(basins):
        o, f = obs[i], pred[i, :, lead - 1]
        issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = sel & np.isfinite(o) & np.isfinite(f)
        if kind == "onset": ok = ok & (issue <= thr[g])
        high, warn = (o > thr[g]), (f > level * thr[g])
        H += int((high & warn & ok).sum()); M += int((high & ~warn & ok).sum()); Fa += int((~high & warn & ok).sum())
    return dict(hits=H, misses=M, false_alarms=Fa, pod=H / (H + M) if H + M else float("nan"), far=Fa / (H + Fa) if H + Fa else float("nan"), csi=H / (H + M + Fa) if H + M + Fa else float("nan"))
def plain(name, lead): return table(name, lead, *TEST, 1.0, "onset")
def onset_flags(name, lead):
    """warned flags of the onsets, basin by basin in the order of the study, with the dates (to prove that two sets are paired on the same onsets)"""
    t = onset_table(name, lead); fl, dd = [], []
    for g in ids:
        d, h, w = t[g]; fl.append(w[h]); dd += [(g, str(x.date())) for x in d[h]]
    return np.concatenate(fl), dd
def basin_pods(name, lead):
    t = onset_table(name, lead); return np.array([((h & w).sum() / h.sum()) if h.sum() else float("nan") for g in ids for (_, h, w) in [t[g]]])
def seed_pods(T, lead): return [plain(f"lstm_{T}_s{s}", lead)["pod"] for s in SEEDS]

# ================================================================ c1: the two new clauses
c = {n: plain(n, 1) for n in ("lstm_wq", "lstm_wqh", "lstm_wqh_older", "lstm_wqh_to09", "lstm_wqh_utc", "lstm_wqn", "base_linreg_wq", "base_rf_wq", "base_linreg_wqh", "base_rf_wqh")}
sd = {T: seed_pods(T, 1) for T in ("wq", "wqh", "wqh_to09")}
flat = not (min(sd["wqh_to09"]) > max(sd["wq"])); above = min(sd["wqh"]) > max(sd["wqh_to09"])
fa, da = onset_flags("lstm_wqh", 1); fc, dc = onset_flags("lstm_wqh_to09", 1)
check("c1 paired on the same onsets", da == dc)
only_a, only_c = int((fa & ~fc).sum()), int((~fa & fc).sum())
gain = 100 * (c["lstm_wqh"]["pod"] - c["lstm_wqh_to09"]["pod"])
s1 = (f"The control with the same rain over the table's own window is {'flat' if flat else 'not flat'} ({p1(c['lstm_wqh_to09']['pod'])} against {p1(c['lstm_wq']['pod'])} with the rain table alone)"
      + (f", so the gain of {gain:.1f} points comes from the hours after about 09:00 ({only_a} onsets warned only by the rain to midnight against {only_c} only by that control)." if flat and above else " [other branch]"))
g_lin, g_rf, g_lstm = (100 * (c[a]["pod"] - c[b]["pod"]) for a, b in (("base_linreg_wqh", "base_linreg_wq"), ("base_rf_wqh", "base_rf_wq"), ("lstm_wqh", "lstm_wq")))
reading_tie = all(R["yardstick"]["lstm_against_simple"][f"lstm_wqh__{n}"]["l1"]["reading"] != "lstm" for n in ("base_linreg_wqh", "base_rf_wqh"))   # the NSE reading of both pairs was recomputed by r2e2_yardstick.py
s2 = (f"Ridge regression and the random forest given the same input warn of {p1(c['base_linreg_wqh']['pod'])} and {p1(c['base_rf_wqh']['pod'])} (the new input adds {g_lin:.1f} and {g_rf:.1f} points to them, {g_lstm:.1f} to the LSTM), "
      + ("so the gain does not need the network." if min(g_lin, g_rf) > 0 and reading_tie else "[other branch]"))
caveat, full, short = R["outcome_caveat"], R["outcome_full"], R["outcome"]
out["c1"] = dict(hits={n: v["hits"] for n, v in c.items()}, pods={n: v["pod"] for n, v in c.items()}, seed_pods={k: v for k, v in sd.items()}, control_flat=flat, every_wqh_seed_above_every_to09_seed=above,
                 only_rain_to_midnight=only_a, only_matched_control=only_c, gains_points=dict(ridge=g_lin, forest=g_rf, lstm=g_lstm, over_matched_control=gain), sentence_1=s1, sentence_2=s2)
ok1 = check("c1 clause 1 is the end of outcome_caveat", (" " + s1 + " " + s2) in caveat and caveat.endswith(s2), (s1, s2))
ok2 = check("c1 outcome_full = outcome + ' ' + outcome_caveat", full == short + " " + caveat)
ok3 = check("c1 the five older clauses are still in front (r2e_readings.py checks them)", caveat.startswith("The NLDAS rain up to the evening gives the same share (30%, 183 hits of 611 in both)") and " the control is not flat (21% against 19% with the rain table alone); " in caveat)
print("c1  the two new clauses of outcome_caveat, rebuilt from my own numbers:")
print(f"    wq {p1(c['lstm_wq']['pod'])} ({c['lstm_wq']['hits']}), matched control wqh_to09 {p1(c['lstm_wqh_to09']['pod'])} ({c['lstm_wqh_to09']['hits']}), wqh {p1(c['lstm_wqh']['pod'])} ({c['lstm_wqh']['hits']}); matched control seeds {[round(100*x,1) for x in sd['wqh_to09']]} against wq seeds {[round(100*x,1) for x in sd['wq']]}: flat {flat}; every wqh seed {[round(100*x,1) for x in sd['wqh']]} above every control seed: {above}")
print(f"    onsets warned only by the rain to midnight / only by the matched control: {only_a} / {only_c}; gain over the control {gain:.2f} points")
print(f"    ridge {p1(c['base_linreg_wqh']['pod'])} ({c['base_linreg_wqh']['hits']} hits; wq {c['base_linreg_wq']['hits']}) +{g_lin:.2f} points; forest {p1(c['base_rf_wqh']['pod'])} ({c['base_rf_wqh']['hits']}; wq {c['base_rf_wq']['hits']}) +{g_rf:.2f}; LSTM +{g_lstm:.2f}")
print(f"    JSON text equals my sentences: clause 1 and 2 at the end of outcome_caveat {ok1}; outcome_full = outcome + caveat {ok2}; older clauses unchanged in front {ok3}")

# ================================================================ c2: the onset-level yardstick
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
VAL = ("2000-10-01", "2005-09-30")
def chosen(name, lead):
    csi = np.array([(lambda v: v if v == v else -1.0)(table(name, lead, *VAL, m, "onset")["csi"]) for m in LEVELS])
    best = np.flatnonzero(csi >= csi.max() - 1e-12); lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
    return lvl, table(name, lead, *TEST, lvl, "onset")
PAIRS = [("wqh", "base_linreg_wqh"), ("wqh", "base_rf_wqh"), ("wqh", "base_persistence"), ("wq", "base_linreg_wq"), ("wq", "base_rf_wq")]
n_cmp, c2rows = 0, {}
for T, n in PAIRS:
    for k in (1, 2, 3):
        j = R["yardstick"]["lstm_against_simple"][f"lstm_{T}__{n}"][f"l{k}"]
        cl, cs = plain(f"lstm_{T}", k), plain(n, k)
        sp = seed_pods(T, k)
        fl, dl = onset_flags(f"lstm_{T}", k); fs, ds = onset_flags(n, k)
        check(f"c2 {T} {n} l{k} paired", dl == ds)
        only_l, only_s, both = int((fl & ~fs).sum()), int((~fl & fs).sum()), int((fl & fs).sum())
        sp_p = float(binomtest(only_l, only_l + only_s, 0.5).pvalue) if only_l + only_s else float("nan")
        pl, pb = basin_pods(f"lstm_{T}", k), basin_pods(n, k); okb = np.isfinite(pl) & np.isfinite(pb)
        try: wp = float(wilcoxon(pl[okb], pb[okb]).pvalue)
        except ValueError: wp = float("nan")
        nh, nc, md = int((pl[okb] - pb[okb] > 0).sum()), int(okb.sum()), 100 * float(np.median(pl[okb] - pb[okb]))
        above_n = int(sum(x > cs["pod"] for x in sp))
        every_above, every_below = bool(min(sp) > cs["pod"]), bool(max(sp) < cs["pod"])
        lv_l, tl = chosen(f"lstm_{T}", k); lv_s, ts = chosen(n, k)
        read = "lstm" if every_above and sp_p < 0.05 else "simple" if every_below and sp_p < 0.05 else "tie"
        words = {"lstm": "the LSTM warns of more onsets", "simple": "the simple model warns of more onsets", "tie": "no clear difference in the onsets"}[read]
        sent = (f"On the onsets at the plain level the LSTM warns of {p1(cl['pod'])} ({cl['hits']} of {cl['hits'] + cl['misses']}) against {p1(cs['pod'])} ({cs['hits']}); {above_n} of its {len(sp)} trainings are above the simple model; "
                f"of the onsets warned by only one of the two, {only_l} are the LSTM's and {only_s} the simple model's (sign test, {pv(sp_p)}; the onsets of one storm are not independent); the LSTM warns of more of the onsets in {nh} of "
                f"{nc} basins ({pv(wp)}). At the warning level chosen on the validation years for onsets the CSI is {tl['csi']:.3f} against {ts['csi']:.3f} (levels {lv_l:.2f} and {lv_s:.2f}).")
        po, ch = j["onsets"]["plain_level"], j["onsets"]["at_the_level_chosen_on_validation_for_onsets"]
        pairs_ = [("lstm_warned", cl["pod"], po["lstm_warned"]), ("simple_warned", cs["pod"], po["simple_warned"]), ("lstm_hits", cl["hits"], po["lstm_hits"]), ("simple_hits", cs["hits"], po["simple_hits"]), ("onsets", cl["hits"] + cl["misses"], po["onsets"]),
                  ("seed_lo", min(sp), po["lstm_seed_range"][0]), ("seed_hi", max(sp), po["lstm_seed_range"][1]), ("n_trainings", len(sp), po["n_lstm_trainings"]), ("above", above_n, po["lstm_trainings_above_the_simple_model"]),
                  ("both", both, po["paired_onsets"]["warned_by_both"]), ("only_lstm", only_l, po["paired_onsets"]["only_the_lstm"]), ("only_simple", only_s, po["paired_onsets"]["only_the_simple_model"]), ("sign_p", sp_p, po["paired_onsets"]["sign_test_p"]),
                  ("n_higher", nh, po["basins"]["n_basins_lstm_higher"]), ("n_compared", nc, po["basins"]["n_basins_compared"]), ("median_diff_points", md, po["basins"]["median_difference_points"]), ("wilcoxon_p", wp, po["basins"]["wilcoxon_p"]),
                  ("lstm_level", lv_l, ch["lstm_level"]), ("simple_level", lv_s, ch["simple_level"]), ("lstm_warned_v", tl["pod"], ch["lstm_warned"]), ("simple_warned_v", ts["pod"], ch["simple_warned"]), ("lstm_false", tl["far"], ch["lstm_false"]),
                  ("simple_false", ts["far"], ch["simple_false"]), ("lstm_csi", tl["csi"], ch["lstm_csi"]), ("simple_csi", ts["csi"], ch["simple_csi"])]
        for nm, a, b in pairs_:
            n_cmp += 1
            if not nanok(a, b): bad.append((f"c2 {T}/{n} l{k} {nm}", (a, b)))
        n_cmp += 4
        for nm, a, b in (("every_above", every_above, po["every_lstm_training_above"]), ("every_below", every_below, po["every_lstm_training_below"]), ("reading", words, j["onsets"]["reading"]), ("sentence", sent, j["onsets_in_words"])):
            if a != b: bad.append((f"c2 {T}/{n} l{k} {nm}", (a, b)))
        c2rows[f"{T}__{n}__l{k}"] = dict(lstm_hits=cl["hits"], simple_hits=cs["hits"], seeds_above=f"{above_n}/{len(sp)}", only_lstm=only_l, only_simple=only_s, sign_p=sp_p, basins=f"{nh}/{nc}", wilcoxon_p=wp, csi=(tl["csi"], ts["csi"]), levels=(lv_l, lv_s), reading=words)
print(f"\nc2  yardstick onsets: 5 pairs x 3 leads: {n_cmp} values, flags and sentences recomputed with own code, {sum(1 for b in bad if b[0].startswith('c2'))} differ")
for key in ("wqh__base_linreg_wqh__l1", "wqh__base_rf_wqh__l1"):
    r = c2rows[key]; print(f"    {key}: hits {r['lstm_hits']} against {r['simple_hits']}; every LSTM training above: {r['seeds_above']}; only the LSTM {r['only_lstm']} / only the simple model {r['only_simple']}, exact sign test p {r['sign_p']:.4f}; basins {r['basins']} (Wilcoxon p {r['wilcoxon_p']:.4f}); CSI at the validated level {r['csi'][0]:.3f} against {r['csi'][1]:.3f} (levels {r['levels'][0]:.2f}, {r['levels'][1]:.2f}); reading: {r['reading']}")
out["c2"] = dict(n_values=n_cmp, rows=c2rows)

# ================================================================ c3: standard errors of by_period
PER = {"before_2009_07_17": ("2005-10-01", "2009-07-16"), "from_2009_07_17": ("2009-07-17", "2014-09-30")}
n3 = 0; se_rows = {}
for sname, per in R["by_period"]["sets"].items():
    for label, (a, b) in PER.items():
        t = table(f"lstm_{sname}", 1, a, b, 1.0, "onset"); p, nn = t["pod"], t["hits"] + t["misses"]
        se = math.sqrt(p * (1 - p) / nn); j = per[label]; n3 += 3
        for nm, x, y in (("se", se, j["onsets_warned_standard_error"]), ("pod", p, j["onsets_warned"]), ("n", nn, j["onsets"])):
            if not nanok(x, y): bad.append((f"c3 {sname} {label} {nm}", (x, y)))
        se_rows[f"{sname}|{label}"] = (round(100 * p, 1), nn, round(100 * se, 2))
w, wq = R["by_period"]["sets"]["wqh"], R["by_period"]["sets"]["wq"]
gb, ga = 100 * (w["before_2009_07_17"]["onsets_warned"] - wq["before_2009_07_17"]["onsets_warned"]), 100 * (w["from_2009_07_17"]["onsets_warned"] - wq["from_2009_07_17"]["onsets_warned"])
sb, sa = se_rows["wqh|before_2009_07_17"], se_rows["wqh|from_2009_07_17"]
sent3 = (f"the rain to midnight warns of {sb[0]:.1f}% of the {sb[1]} onsets before the change (standard error {sb[2]:.1f} points) and of {sa[0]:.1f}% of the {sa[1]} onsets after it ({sa[2]:.1f} points), and its gain over the rain table alone is {gb:+.1f} points before and {ga:+.1f} after, "
         + ("so the gain is not smaller before the change than after it: the worry is not borne out." if gb >= ga else "so the gain is larger after the change: the worry cannot be excluded."))
ok = check("c3 the sentence in the note on ERA5's change", any(sent3 in n for n in R["notes"]), sent3)
check("c3 how_to_read names the formula", "onsets_warned_standard_error is sqrt(p(1-p)/n) of the share" in R["by_period"]["how_to_read"])
print(f"\nc3  by_period: standard error sqrt(p(1-p)/n) for 8 sets x 2 parts: {n3} values recomputed, {sum(1 for b in bad if b[0].startswith('c3'))} differ; wqh {sb[0]:.1f}% of {sb[1]} (SE {sb[2]:.2f} points) before, {sa[0]:.1f}% of {sa[1]} (SE {sa[2]:.2f}) after; gain over wq {gb:+.1f} / {ga:+.1f} points; the sentence in the note equals mine: {ok}")
out["c3"] = dict(rows=se_rows, gain_before=gb, gain_after=ga, sentence=sent3, sentence_found=ok)
# the difference of the two gains is within chance? (two independent binomial parts for each set; a description, not a test of the script)
se_gain_b = math.sqrt(sum(0 for _ in ()) + (w["before_2009_07_17"]["onsets_warned"] * (1 - w["before_2009_07_17"]["onsets_warned"]) + wq["before_2009_07_17"]["onsets_warned"] * (1 - wq["before_2009_07_17"]["onsets_warned"])) / 200)
print(f"    (for scale: the gain over wq before the change, {gb:+.1f} points, has a standard error of at most {100 * se_gain_b:.1f} points if the two sets were independent; they are paired, so less)")

# ================================================================ c4: verdict records of the three new baselines
crit = R["verdicts"]["criteria"]
pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
def verdict(t):
    if t["hits"] + t["misses"] == 0: return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go: return "GO"
    return "CAUTION" if t["csi"] >= csi_c else "NOT YET"
n4 = 0; v_rows = {}
for name in ("base_persistence", "base_linreg_wq", "base_rf_wq"):
    jv = R["verdicts"]["sets"][name]
    for k in (1, 2, 3):
        pl_ = jv[f"l{k}"]["plain"]
        for kind, label in (("all", "chosen_on_validation_all_days"), ("onset", "chosen_on_validation_onsets")):
            csi = np.array([(lambda v: v if v == v else -1.0)(table(name, k, *VAL, m, kind)["csi"]) for m in LEVELS])
            best = np.flatnonzero(csi >= csi.max() - 1e-12); lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
            j = jv[f"l{k}"][label]; n4 += 1
            if abs(lvl - j["trigger"]) > 1e-9: bad.append((f"c4 {name} l{k} {label} trigger", (lvl, j["trigger"])))
            for kk in ("all", "onset"):
                t = table(name, k, *TEST, lvl, kk); jj = j[kk]
                for fld in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    n4 += 1
                    if not nanok(t[fld], jj[fld]): bad.append((f"c4 {name} l{k} {label} {kk} {fld}", (t[fld], jj[fld])))
                n4 += 1
                if verdict(t) != jj["verdict"]: bad.append((f"c4 {name} l{k} {label} {kk} verdict", (verdict(t), jj["verdict"])))
            if kind == "onset": v_rows[f"{name}:l{k}"] = (lvl, table(name, k, *TEST, lvl, "onset")["pod"], verdict(table(name, k, *TEST, lvl, "onset")))
        for kk in ("all", "onset"):                                                 # the 'plain' level (trigger 1.0)
            t = table(name, k, *TEST, 1.0, kk); jj = pl_[kk]
            for fld in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                n4 += 1
                if not nanok(t[fld], jj[fld]): bad.append((f"c4 {name} l{k} plain {kk} {fld}", (t[fld], jj[fld])))
            n4 += 1
            if verdict(t) != jj["verdict"]: bad.append((f"c4 {name} l{k} plain {kk} verdict", (verdict(t), jj["verdict"])))
print(f"\nc4  verdict records of base_persistence, base_linreg_wq, base_rf_wq (3 leads x [plain; 2 validated choices] x 2 tables): {n4} values recomputed with own validation choice and test counting, {sum(1 for b in bad if b[0].startswith('c4'))} differ")
for key, (lvl, pod, vd) in v_rows.items(): print(f"    {key}: level {lvl:.2f}, onsets warned on the test years {100 * pod:.1f}%, onset verdict {vd}")
out["c4"] = dict(n_values=n4, rows=v_rows)

# ================================================================ c5: what_was_not_done[3], the two appended sentences, the figure label
VARSETS = ("wqh", "wqh_older", "wh", "wh_older", "wqh_to09", "wqh_utc", "wqnh")
nse_seed = {n: [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", 1).values()))) for s in SEEDS] for n in VARSETS}
span = {n: max(v) - min(v) for n, v in nse_seed.items()}; wide = max(span, key=span.get)
val_nse = [float(torch.load(f"{PROJ}/results/models/lstm_{wide}_s{s}.pt", map_location="cpu", weights_only=True)["best_val_nse"]) for s in SEEDS]
nb = {n: nse_per_basin(f"lstm_{n}", 1) for n in ("wqh", "wqh_to09")}
gains = np.array([nb["wqh"][g] - nb["wqh_to09"][g] for g in ids]); n_hi = int((gains > 0).sum())
nse_gain = float(np.median([nb["wqh"][g] for g in ids]) - np.median([nb["wqh_to09"][g] for g in ids]))
try: wp5 = float(wilcoxon([nb["wqh"][g] for g in ids], [nb["wqh_to09"][g] for g in ids]).pvalue)
except ValueError: wp5 = float("nan")
seeds_above_nse = min(nse_seed["wqh"]) > max(nse_seed["wqh_to09"])
reading5 = "clear gain" if (seeds_above_nse and n_hi > len(ids) / 2 and wp5 < 0.05) else "no clear difference"
s5 = (f"More seeds where the NSE decides: the median NSE of the single trainings of a set spans up to {span[wide]:.3f} (lead 1, lstm_{wide}: {min(nse_seed[wide]):.3f} to {max(nse_seed[wide]):.3f}), while the validation NSE of the same trainings spans only "
      f"{min(val_nse):.3f} to {max(val_nse):.3f}, so the validation years cannot tell a good training from a weak one. With three seeds the NSE rule of report section 4.2 (every single seed above every single seed, higher in most basins, p below 0.05) can hardly ever call a gain of this size clear: "
      f"value_of_hours_09_to_24 is {nse_gain:+.3f} in NSE, higher in {n_hi} of {len(ids)} basins ({pv(wp5)}), and the rule reads it as '{reading5}'. The onsets warned, pooled over the basins, are decided by the seeds more easily.")
ok5 = check("c5 what_was_not_done[3]", R["what_was_not_done"][3] == s5, (s5, R["what_was_not_done"][3]))
print(f"\nc5  single-training NSE spans, lead 1: " + ", ".join(f"{n} {min(v):.3f} to {max(v):.3f} ({span[n]:.3f})" for n, v in nse_seed.items()))
print(f"    widest: lstm_{wide}, validation NSE of its three trainings {[round(x, 4) for x in val_nse]} (spans {min(val_nse):.3f} to {max(val_nse):.3f}); value_of_hours_09_to_24 in NSE {nse_gain:+.4f}, higher in {n_hi}/12 basins, p {wp5:.5f}, reading '{reading5}'; what_was_not_done[3] equals my sentence: {ok5}")
out["c5"] = dict(nse_seed=nse_seed, span=span, widest=wide, validation_nse=val_nse, nse_gain=nse_gain, n_higher=n_hi, wilcoxon_p=wp5, reading=reading5, sentence_equal=ok5)
# the sentence appended to value_of_hours_09_to_24
b0, b1 = c["lstm_wqh_to09"], c["lstm_wq"]
every_up = bool(min(sd["wqh_to09"]) > max(sd["wq"]))
a1 = (f" ITS COMPANION is second_rain_product_alone (lstm_wqh_to09 minus lstm_wq: {p1(b0['pod'])} against {p1(b1['pod'])}, {100 * (b0['pod'] - b1['pod']):+.1f} points, every seed above every seed: {'yes' if every_up else 'no'}): "
      f"the control is {'not flat' if every_up else 'flat'}, so this difference is {'the worth of the hours plus part of the effect of having a second product' if every_up else 'the worth of the hours after about 09:00 and not that of having a second product'}. Quote the two together.")
ok = check("c5 sentence appended to value_of_hours_09_to_24", R["value_of_hours_09_to_24"]["what"].endswith(a1), a1)
nl, e5, u5 = c["lstm_wqn"], c["lstm_wqh"], c["lstm_wqh_utc"]
m = re.search(r"\(([0-9:]+) as measured, ([0-9:]+) if it is stamped by the UTC day\)", R["value_of_evening_hours_beyond_utc_day"]["what"])
end_utc = 24 + max(R["daily_rain"]["winter_utc_offset_h"].values())                       # the UTC day ends at 19:00 standard time in the Eastern basins (offset -5)
a2 = (f" THIS IS A READING OF ERA5, whose rain is poorer than NLDAS's, and it is to be quoted together with nldas_rain_against_era5_over_the_utc_day: the NLDAS rain, whose day ends at about the hour at which the UTC day ends "
      f"({m.group(1) if m else '?'} as measured, {end_utc}:00 if it is stamped by the UTC day), reaches {p1(nl['pod'])} ({nl['hits']} hits)"
      + (f", the same share as the ERA5 rain to midnight ({p1(e5['pod'])}, {e5['hits']} hits)," if nl["hits"] == e5["hits"] else f" against {p1(e5['pod'])} ({e5['hits']} hits) for the ERA5 rain to midnight,")
      + f" while the ERA5 rain over the UTC day reaches {p1(u5['pod'])} ({u5['hits']} hits). So the {100 * (e5['pod'] - u5['pod']):+.1f} points are what the evening hours beyond the UTC day add to ERA5, not what they add to any rain product.")
ok_b = check("c5 sentence appended to value_of_evening_hours_beyond_utc_day", R["value_of_evening_hours_beyond_utc_day"]["what"].endswith(a2), a2)
ok_c = check("c5 figure label", R["figure_labels"]["nldas_if_utc_stamped"] == f"{end_utc}:00 if stamped by the UTC day", R["figure_labels"]["nldas_if_utc_stamped"])
inp = [e for e in R["inputs"]["clock"] if e["input"].startswith("prcp_nldas")][0]["covers"]
ok_d = check("c5 the 17:30 of the sentence is the measured end of the NLDAS day in inputs.clock", m is not None and m.group(1) in inp, (m.group(1) if m else None, inp))
print(f"    sentence appended to value_of_hours_09_to_24 equals mine: {ok}; to value_of_evening_hours_beyond_utc_day: {ok_b} (NLDAS {p1(nl['pod'])} {nl['hits']} hits, ERA5 to midnight {e5['hits']}, ERA5 UTC day {p1(u5['pod'])} {u5['hits']}, difference {100 * (e5['pod'] - u5['pod']):+.1f} points); figure label '{R['figure_labels']['nldas_if_utc_stamped']}' equals mine (UTC day ends {end_utc}:00 standard time in the Eastern basins): {ok_c}; '17:30 as measured' is the NLDAS end in inputs.clock: {ok_d}")
out["c5"].update(appended_companion_ok=ok, appended_evening_ok=ok_b, figure_label_ok=ok_c)

# ================================================================ c6: IMERG-Early delay
txt = json.dumps(R, ensure_ascii=False)
script = open(f"{PROJ}/code/15_rain_to_midnight.py").read()
n_new_json, n_old_json = txt.count("delay of about 4 hours"), txt.count("4 to 5 hours") + txt.count("4-5 hours") + txt.count("4 to 5 h")
n_new_py, n_old_py = script.count("delay of about 4 hours"), script.count("4 to 5 hours") + script.count("4-5 hours")
check("c6 no '4 to 5 hours' left", n_old_json == 0 and n_old_py == 0, (n_old_json, n_old_py))
print(f"\nc6  IMERG-Early's delay: 'about 4 hours' appears {n_new_json} times in the results file and {n_new_py} in the script; '4 to 5 hours' remains {n_old_json} times in the file and {n_old_py} in the script")
out["c6"] = dict(about_4_hours_json=n_new_json, about_4_hours_script=n_new_py, old_wording_json=n_old_json, old_wording_script=n_old_py)

out["differences"] = [(a, str(b)[:200]) for a, b in bad]
print(f"\nTOTAL: {len(bad)} differences in c1-c6 ({time.time() - t0:.0f} s)", [(a, str(b)[:80]) for a, b in bad[:6]])
jsave("r3c_clauses.json", out)
