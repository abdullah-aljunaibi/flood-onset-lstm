"""ROUND 2, check (e) extra: the verdict records of the new sets. The warning level is chosen on the VALIDATION years (best pooled CSI over the levels 0.30 to 1.50 in steps of 0.05, ties to the level closest to 1.0), then the TEST years are scored
at that level; own counting, own choice, compared with the JSON (verdicts.sets / yardstick.sets) for the three new LSTM sets and the two simple models, all leads, 'all' and 'onset' warning tables."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
import re
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json")
ids = list(data())
crit = R["verdicts"]["criteria"]
pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
VAL, TST = ("2000-10-01", "2005-09-30"), TEST
def table(name, lead, a, b, level, kind):
    dates, basins, obs, pred = load_npz(name); thr = thresholds(basins, dates); sel = part(dates, a, b)
    H = M = Fa = C = 0
    for i, g in enumerate(basins):
        o, f = obs[i], pred[i, :, lead - 1]
        issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = sel & np.isfinite(o) & np.isfinite(f)
        if kind == "onset": ok = ok & (issue <= thr[g])
        high, warn = (o > thr[g]), (f > level * thr[g])
        H += int((high & warn & ok).sum()); M += int((high & ~warn & ok).sum()); Fa += int((~high & warn & ok).sum()); C += int((~high & ~warn & ok).sum())
    pod = H / (H + M) if H + M else float("nan"); far = Fa / (H + Fa) if H + Fa else float("nan"); csi = H / (H + M + Fa) if H + M + Fa else float("nan")
    return dict(hits=H, misses=M, false_alarms=Fa, pod=pod, far=far, csi=csi)
def verdict(t):
    if t["hits"] + t["misses"] == 0: return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go: return "GO"
    return "CAUTION" if t["csi"] >= csi_c else "NOT YET"
bad = []; n = 0; rows = {}
for name in ("lstm_wqh_to09", "lstm_wqh_utc", "lstm_wqnh", "base_linreg_wqh", "base_rf_wqh"):
    jv = R["verdicts"]["sets"][name[5:] if name.startswith("lstm_") else name]
    for k in (1, 2, 3):
        for label, kind in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
            csi = [table(name, k, *VAL, m, kind)["csi"] for m in LEVELS]
            csi = np.array([c if c == c else -1.0 for c in csi]); best = np.flatnonzero(csi >= csi.max() - 1e-12)
            lvl = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
            j = jv[f"l{k}"][label]
            n += 1; bad += [(name, k, label, "trigger", lvl, j["trigger"])] if abs(lvl - j["trigger"]) > 1e-9 else []
            for kk in ("all", "onset"):
                t = table(name, k, *TST, lvl, kk); jj = j[kk]
                for fld in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    a, b = t[fld], jj[fld]
                    n += 1
                    if not ((b is None and a != a) or abs(a - b) < 1e-9): bad.append((name, k, label, kk, fld, a, b))
                n += 1; bad += [(name, k, label, kk, "verdict", verdict(t), jj["verdict"])] if verdict(t) != jj["verdict"] else []
            rows[f"{name}:l{k}:{label}"] = dict(trigger=lvl, onset_pod=table(name, k, *TST, lvl, "onset")["pod"], onset_verdict=verdict(table(name, k, *TST, lvl, "onset")))
print(f"verdict records of 3 new LSTM sets and 2 simple models (3 leads x 2 choices x [trigger, 2 tables x 6 numbers + verdict]): {n} values recomputed with own validation choice and own test counting, {len(bad)} differ", bad[:3])
for key in ("lstm_wqh_to09:l1:chosen_on_validation_onsets", "lstm_wqh_utc:l1:chosen_on_validation_onsets", "lstm_wqnh:l1:chosen_on_validation_onsets", "base_linreg_wqh:l1:chosen_on_validation_onsets", "base_rf_wqh:l1:chosen_on_validation_onsets"):
    r = rows[key]; print(f"  {key}: warning level {r['trigger']:.2f} of the high-flow level, onsets warned on the test years {100*r['onset_pod']:.1f}%, verdict {r['onset_verdict']}")
all_verdicts = sorted({v["onset_verdict"] for v in rows.values()})
print("onset verdicts over all these records:", all_verdicts)
jsave("r2e3_verdicts.json", dict(n=n, differ=len(bad), details=bad[:5], onset_verdicts=all_verdicts, rows=rows))
