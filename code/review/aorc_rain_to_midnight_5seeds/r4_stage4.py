"""Round 2, checks s1-s6: results/stage4_realtime.json (code/check_stage4_realtime.py rerun on 30 October 2026 with STAGE4_NO_FETCH=1 on the five lstm_wqa networks) checked
without the networks:
s1 the counts it rests on (AORC 316, rain table 114, wqa_rep 321) against my own scoring of results/preds and against results/aorc_rain_to_midnight.json;
s2 the 611 rows of per_onset: the onsets, the AORC and rain-table flags and forecasts against results/preds (my scoring), the Stage IV flags recounted (294), lost and gained, per basin;
s3 the rain of each issue day rebuilt by my own route: AORC from data/aorc/aorc_<gauge>.npz and Stage IV from data/aorc/stage4_issue_days.csv over the hour-ending UTC labels of the
   wall-clock day (zoneinfo), with AORC in the place of a missing hour; against aorc_mm and stage4_mm of per_onset; k (the AORC level) and the agreement block;
s4 every variant's derived numbers (share of the AORC effect surviving, difference, share of the onsets) from its own counts, the single seeds against the log;
s5 the 95% intervals of the raw Stage IV variant re-drawn: (a) with my own generator (an independent bootstrap), (b) with the script's generator and order (exact replication);
s6 three seeds against five: the stored three-seed file of the 10:41 backup against today's file."""
from r_common import *
from zoneinfo import ZoneInfo

C = Compare()
S4 = jload(STAGE4)
J = jload(RESULTS)
mine = {}
sha256("data/aorc/stage4_issue_days.csv")

# ---- s1 ----------------------------------------------------------------------------------------------------------------------------------------------
W, WQ, REP = score("lstm_wqa"), score("lstm_wq"), score("lstm_wqa_rep")
C.add("s1 wqa aorc = my lstm_wqa l1 hits", S4["models"]["wqa"]["aorc"], W[1]["onset"]["hits"])
C.add("s1 wqa aorc = results file", S4["models"]["wqa"]["aorc"], J["summary_test"]["sets"]["wqa"]["l1"]["onset"]["hits"])
C.add("s1 rain_table_alone = my lstm_wq l1 hits", S4["models"]["wqa"]["rain_table_alone"], WQ[1]["onset"]["hits"])
C.add("s1 wqa_rep aorc = my lstm_wqa_rep l1 hits", S4["models"]["wqa_rep"]["aorc"], REP[1]["onset"]["hits"])
C.add("s1 reproduction wqa rerun = stored", S4["reproduction"]["wqa"]["onsets_warned_rerun"], W[1]["onset"]["hits"])
C.add("s1 reproduction wqa max abs diff", S4["reproduction"]["wqa"]["max_abs_difference_to_the_stored_single_seed_forecasts"], 0.0)
C.add("s1 reproduction wqa_rep rerun", S4["reproduction"]["wqa_rep"]["onsets_warned_rerun"], REP[1]["onset"]["hits"])
C.add("s1 onsets tested", S4["onsets"]["tested"], len(W[1]["onset_keys"]))
C.add("s1 issue days", S4["onsets"]["issue_days_tested"], 368)

# ---- s2 per_onset --------------------------------------------------------------------------------------------------------------------------------------
PO = S4["per_onset"]
d, b, obs, pa, meta = load_npz("lstm_wqa")
thr = thresholds(d, b, obs)
bi = {g: i for i, g in enumerate(b)}
keys = W[1]["onset_keys"]
mykeys = sorted((g, str(d[j].date())) for g, j in keys)
fkeys = [(r["gauge"], r["target_day"]) for r in PO]
C.add("s2 per_onset rows = my onsets (gauge, target day), in the order gauge, day", mykeys, fkeys)
fa = np.array([pa[bi[r["gauge"]], d.get_loc(pd.Timestamp(r["target_day"])), 0] for r in PO])
del pa
_, _, _, pq, _ = load_npz("lstm_wq")
fq = np.array([pq[bi[r["gauge"]], d.get_loc(pd.Timestamp(r["target_day"])), 0] for r in PO])
del pq
th = np.array([thr[bi[r["gauge"]]] for r in PO])
wa, wq = fa > th, fq > th
C.add("s2 warned_aorc flags = my flags", True, all(bool(r["warned_aorc"]) == bool(x) for r, x in zip(PO, wa)))
C.add("s2 warned_rain_table flags = my flags", True, all(bool(r["warned_rain_table"]) == bool(x) for r, x in zip(PO, wq)))
C.add("s2 forecast_mm_aorc = lstm_wqa.npz (4 decimals)", True, float(np.max(np.abs(np.array([r["forecast_mm_aorc"] for r in PO]) - fa))) <= 5e-5 + 1e-12)
C.add("s2 forecast_mm_rain_table = lstm_wq.npz (4 decimals)", True, float(np.max(np.abs(np.array([r["forecast_mm_rain_table"] for r in PO]) - fq))) <= 5e-5 + 1e-12)
C.add("s2 high_flow_level = my Q95 (4 decimals)", True, float(np.max(np.abs(np.array([r["high_flow_level_mm_day"] for r in PO]) - th))) <= 5e-5 + 1e-12)
ws = np.array([bool(r["warned_stage4"]) for r in PO])
fs = np.array([r["forecast_mm_stage4"] for r in PO])
borderline = np.abs(fs - th) <= 1e-4
C.add("s2 warned_stage4 = forecast_mm_stage4 > level (away from the 4-decimal rounding)", True, bool(np.all((fs > th)[~borderline] == ws[~borderline])))
mine["stage4_borderline_rows"] = int(borderline.sum())
V = S4["models"]["wqa"]["variants"]
C.add("s2 stage4 onsets warned recount", V["stage4"]["onsets_warned"], int(ws.sum()))
C.add("s2 aorc recount", S4["models"]["wqa"]["aorc"], int(wa.sum()))
C.add("s2 rain table recount", S4["models"]["wqa"]["rain_table_alone"], int(wq.sum()))
LG = V["stage4"]["onsets_lost_and_gained"]
for k_, v_ in dict(warned_with_aorc_not_with_stage4=int((wa & ~ws).sum()), warned_with_stage4_not_with_aorc=int((~wa & ws).sum()), warned_by_both=int((wa & ws).sum()),
                   warned_by_neither=int((~wa & ~ws).sum())).items():
    C.add(f"s2 stage4 {k_}", LG[k_], v_)
gauges = np.array([r["gauge"] for r in PO])
for g in b:
    m = gauges == g
    pb = V["stage4"]["per_basin"][g]
    C.add(f"s2 per_basin {g} onsets", pb["onsets"], int(m.sum()))
    C.add(f"s2 per_basin {g} aorc", pb["aorc"], int(wa[m].sum()))
    C.add(f"s2 per_basin {g} rain_table_alone", pb["rain_table_alone"], int(wq[m].sum()))
    C.add(f"s2 per_basin {g} stage4", pb["stage4"], int(ws[m].sum()))

# ---- s3 the rain of the issue days, my own route -------------------------------------------------------------------------------------------------------
DW = jload("results/day_windows.json")
ZONE = {g: ("America/Chicago" if int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) == -6 else "America/New_York") for g in b}
S4H = pd.read_csv(PROJ / "data/aorc/stage4_issue_days.csv", index_col=0, parse_dates=True)
S4H.index = pd.DatetimeIndex(S4H.index)


def labels(day, zone):
    z = ZoneInfo(zone)
    a = pd.Timestamp(day).tz_localize(z).tz_convert("UTC").tz_localize(None)
    e = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(z).tz_convert("UTC").tz_localize(None)
    return pd.date_range(a + pd.Timedelta(hours=1), e, freq="h")          # hour-ending stamps inside (00:00, 24:00] local


myA, myS, myS_nolast, myS_nolast2, nfb, nh = [], [], [], [], 0, 0
AOR = {}
for g in b:
    rel = f"data/aorc/aorc_{g}.npz"
    sha256(rel)
    z = np.load(PROJ / rel, allow_pickle=False)
    AOR[g] = pd.Series(z["rain_mm"].astype(float), index=pd.to_datetime(z["stamp_utc_s"], unit="s"))
    z.close()
for r in PO:
    g = r["gauge"]
    lab = labels(r["issue_day"], ZONE[g])
    a = AOR[g].reindex(lab).values
    s = S4H[g].reindex(lab).values.astype(float)
    fb = np.isnan(s)
    s = np.where(fb, a, s)
    nfb += int(fb.sum()); nh += len(lab)
    myA.append(a.sum()); myS.append(s.sum()); myS_nolast.append(s.sum() - s[-1]); myS_nolast2.append(s.sum() - s[-2:].sum())
myA, myS = np.array(myA), np.array(myS)
fA, fS = np.array([r["aorc_mm"] for r in PO]), np.array([r["stage4_mm"] for r in PO])
mine["max_abs_diff_aorc_mm"] = float(np.max(np.abs(fA - myA)))
mine["max_abs_diff_stage4_mm"] = float(np.max(np.abs(fS - myS)))
C.add("s3 aorc_mm of the 611 issue days = my sums (3 decimals)", True, mine["max_abs_diff_aorc_mm"] <= 5e-4 + 1e-9)
C.add("s3 stage4_mm of the 611 issue days = my sums (3 decimals)", True, mine["max_abs_diff_stage4_mm"] <= 5e-4 + 1e-9)
C.add("s3 hours replaced by AORC", S4["fallback"]["hours_replaced_by_aorc"], nfb)
C.add("s3 hours total", S4["fallback"]["hours_total"], nh)
C.add("s3 per_onset hours replaced sum", S4["fallback"]["hours_replaced_by_aorc"], int(sum(r["stage4_hours_replaced_by_aorc"] for r in PO)))
K = myA.sum() / myS.sum()
mine["k"] = K
C.add("s3 k_level_aorc_over_stage4", S4["k_level_aorc_over_stage4"], K, tol=1e-9)
AG = S4["agreement_of_the_two_products_on_the_issue_days"]
C.add("s3 correlation", AG["correlation"], float(np.corrcoef(myA, myS)[0, 1]), tol=1e-9)
C.add("s3 mean_ratio_stage4_over_aorc", AG["mean_ratio_stage4_over_aorc"], myS.sum() / myA.sum(), tol=1e-9)
C.add("s3 mean_mm_aorc", AG["mean_mm_aorc"], myA.mean(), tol=1e-9)
C.add("s3 mean_mm_stage4", AG["mean_mm_stage4"], myS.mean(), tol=1e-9)
mine["issue_day_values"] = dict(s4=myS.tolist(), s4_no_last_hour=np.array(myS_nolast).tolist(), s4_no_last_two_hours=np.array(myS_nolast2).tolist(), aorc=myA.tolist(),
                                rows=[(r["gauge"], r["issue_day"]) for r in PO])

# ---- s4 variants -------------------------------------------------------------------------------------------------------------------------------------
LOG = {"stage4": [319, 270, 277, 296, 284]}
for mdl in ("wqa", "wqa_rep"):
    M = S4["models"][mdl]
    a, r_ = M["aorc"], M["rain_table_alone"]
    for kind, v in M["variants"].items():
        s = v["onsets_warned"]
        C.add(f"s4 {mdl} {kind} share surviving", v["share_of_the_aorc_effect_surviving"], (s - r_) / (a - r_))
        C.add(f"s4 {mdl} {kind} differs", v["differs_from_aorc_by_onsets"], s - a)
        C.add(f"s4 {mdl} {kind} share of the onsets", v["share_of_the_onsets"], s / 611)
        C.add(f"s4 {mdl} {kind} single seeds: five", 5, len(v["single_seed_onsets_warned"]))
        C.add(f"s4 {mdl} {kind} per_basin sums", s, sum(pb["stage4"] for pb in v["per_basin"].values()))
        lg = v["onsets_lost_and_gained"]
        C.add(f"s4 {mdl} {kind} lost/gained consistent", True, lg["warned_by_both"] + lg["warned_with_stage4_not_with_aorc"] == s and lg["warned_by_both"] + lg["warned_with_aorc_not_with_stage4"] == a
              and sum(lg[x] for x in ("warned_by_both", "warned_with_stage4_not_with_aorc", "warned_with_aorc_not_with_stage4", "warned_by_neither")) == 611)
C.add("s4 wqa stage4 single seeds = the log line", LOG["stage4"], V["stage4"]["single_seed_onsets_warned"])
CTRL = S4["models"]["wqa"]["controls_on_the_rain_of_the_issue_day"]
C.add("s4 another_onset_stage4 draws = log", [279, 290, 282, 286, 282, 288, 289, 295, 285, 285], CTRL["another_onset_stage4"]["onsets_warned"])
C.add("s4 ordinary_day_aorc draws = log", [38, 40, 30, 41, 38, 31, 34, 30, 40, 35], CTRL["ordinary_day_aorc"]["onsets_warned"])
C.add("s4 no_rain = log", [13], CTRL["no_rain"]["onsets_warned"])
C.add("s4 another_onset mean", CTRL["another_onset_stage4"]["mean"], float(np.mean([279, 290, 282, 286, 282, 288, 289, 295, 285, 285])))
mine["variants_wqa"] = {k: v["onsets_warned"] for k, v in V.items()}
mine["single_seeds_wqa"] = {k: v["single_seed_onsets_warned"] for k, v in V.items()}

# ---- s5 intervals ----------------------------------------------------------------------------------------------------------------------------------------
a_, r_, s_ = wa.astype(int), wq.astype(int), ws.astype(int)
codes = pd.factorize(pd.Series([r["issue_day"] for r in PO]))[0]


def draw(gen, a, r, s, codes, nb=2000):
    n = len(a)
    idx = gen.integers(0, n, (nb, n))
    A_, R_, S_ = a[idx].sum(1), r[idx].sum(1), s[idx].sum(1)
    sv = np.where(A_ > R_, (S_ - R_) / np.maximum(A_ - R_, 1), np.nan)
    on = dict(share_surviving=[float(x) for x in np.percentile(sv[np.isfinite(sv)], [2.5, 97.5])],
              stage4_minus_aorc_share_of_onsets=[float(x) for x in np.percentile((S_ - A_) / n, [2.5, 97.5])])
    days = np.unique(codes)
    per = {c: np.flatnonzero(codes == c) for c in days}
    sa, sr, ss, cn = (np.array([x[per[c]].sum() for c in days]) for x in (a, r, s, np.ones(n)))
    di = gen.integers(0, len(days), (nb, len(days)))
    A_, R_, S_, N_ = sa[di].sum(1), sr[di].sum(1), ss[di].sum(1), cn[di].sum(1)
    sv = np.where(A_ > R_, (S_ - R_) / np.maximum(A_ - R_, 1), np.nan)
    dy = dict(share_surviving=[float(x) for x in np.percentile(sv[np.isfinite(sv)], [2.5, 97.5])],
              stage4_minus_aorc_share_of_onsets=[float(x) for x in np.percentile((S_ - A_) / N_, [2.5, 97.5])])
    return dict(over_onsets=on, over_issue_days=dy)


exact = draw(np.random.default_rng(1), a_, r_, s_, codes)
mine_iv = draw(np.random.default_rng(20261030), a_, r_, s_, codes, nb=4000)
FI = V["stage4"]["intervals"]
for over in ("over_onsets", "over_issue_days"):
    for x in ("share_surviving", "stage4_minus_aorc_share_of_onsets"):
        for i in (0, 1):
            C.add(f"s5 exact replication {over} {x}[{i}]", FI[over][x][i], exact[over][x][i], tol=1e-12)
            C.add(f"s5 own bootstrap {over} {x}[{i}] (within 0.015)", True, abs(FI[over][x][i] - mine_iv[over][x][i]) <= 0.015)
mine["intervals_own_bootstrap_4000"] = mine_iv
mine["intervals_exact_replication"] = exact

# ---- s6 three seeds against five --------------------------------------------------------------------------------------------------------------------
B3 = jload("results/stage4_realtime.json", BACKUP)
three = {k: v["onsets_warned"] for k, v in B3["models"]["wqa"]["variants"].items()}
mine["three_seed_file"] = dict(aorc=B3["models"]["wqa"]["aorc"], variants=three, share_surviving_stage4=B3["models"]["wqa"]["variants"]["stage4"]["share_of_the_aorc_effect_surviving"],
                               intervals=B3["models"]["wqa"]["variants"]["stage4"]["intervals"], single_seeds=B3["models"]["wqa"]["variants"]["stage4"]["single_seed_onsets_warned"])
b3 = B3["models"]["wqa"]["variants"]["stage4"]["single_seed_onsets_warned"]
C.add("s6 seeds 0-2 of the five-seed run = the three-seed run (stage4 raw)", b3, V["stage4"]["single_seed_onsets_warned"][:3])
for kind in V:
    if kind in B3["models"]["wqa"]["variants"]:
        C.add(f"s6 seeds 0-2 unchanged {kind}", B3["models"]["wqa"]["variants"][kind]["single_seed_onsets_warned"], V[kind]["single_seed_onsets_warned"][:3])
# the per-onset rows of the three-seed file: the issue-day rain is the same (only the networks changed)
C.add("s6 stage4_mm of the 611 rows unchanged", [r["stage4_mm"] for r in B3["per_onset"]], [r["stage4_mm"] for r in PO])
mine["three_seed_per_onset_warned_stage4"] = [bool(r["warned_stage4"]) for r in B3["per_onset"]]
mine["three_seed_per_onset_warned_aorc"] = [bool(r["warned_aorc"]) for r in B3["per_onset"]]
mine["five_seed_per_onset_warned_stage4"] = ws.tolist()
mine["five_seed_per_onset_warned_aorc"] = wa.tolist()
mine["five_seed_per_onset_warned_rain_table"] = wq.tolist()

jsave("r4_stage4.json", dict(compare=C.rows, summary=C.summary(), mine=mine, files_read=READ))
print(json.dumps(C.summary(), indent=1))
print("k", K, "max diffs", mine["max_abs_diff_aorc_mm"], mine["max_abs_diff_stage4_mm"], "borderline", mine["stage4_borderline_rows"])
print("own bootstrap:", json.dumps(mine_iv))
print("three-seed:", json.dumps({k: v for k, v in mine["three_seed_file"].items() if k != "intervals"}))
