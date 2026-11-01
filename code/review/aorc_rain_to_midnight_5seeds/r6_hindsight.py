"""Round 2, check k1: does the hindsight bound of round 1 ('at most about 3 points', 16 of 313 hits; check d7, from the three-seed wqa networks) still hold for the five-seed set?
Without networks: the 171 onsets of round 1's Stage IV sample (issue day among the builder's 50 days of results/aorc_hindsight.json and the 28 days of round 1's
a7a_select_days.py, rerun into OUT/round1_days) read from the per-onset flags of results/stage4_realtime.json (five networks, today) and of the three-seed file of the
10:41 backup; plus the direct measurement on all 611 onsets (the raw and the level-matched Stage IV variants) for three and five seeds. The dose-response route of round 1
(a8 T2, the second route of the bound) needs every network and was not rerun. Needs r4_stage4.json."""
from r_common import *

R4 = jread("r4_stage4.json")["mine"]
S4 = jload(STAGE4)
B3 = jload("results/stage4_realtime.json", BACKUP)
H = jload("results/aorc_hindsight.json")
days28 = json.load(open(OUT / "round1_days" / "a7a_days.json", encoding="utf-8"))["days"]
days78 = set(H["numbers"]["days"]) | set(days28)
rows = [tuple(x) for x in R4["issue_day_values"]["rows"]]
sel = np.array([day in days78 for g, day in rows])
out = dict(n_days=len(days78), onsets_in_sample=int(sel.sum()), issue_days_in_sample=len({day for (g, day), s in zip(rows, sel) if s}))
f5a, f5s, f5t = (np.array(R4[k]) for k in ("five_seed_per_onset_warned_aorc", "five_seed_per_onset_warned_stage4", "five_seed_per_onset_warned_rain_table"))
f3a, f3s = np.array(R4["three_seed_per_onset_warned_aorc"]), np.array(R4["three_seed_per_onset_warned_stage4"])
# three-seed AORC on the sample from my own average of seeds 0-2 (checks the sample against round 1's 92)
acc = None
for s in (0, 1, 2):
    d, b, obs, p, meta = load_npz(f"lstm_wqa_s{s}")
    acc = p[:, :, 0].copy() if acc is None else acc + p[:, :, 0]
    del p
acc /= 3
thr = thresholds(d, b, obs)
my3 = np.array([acc[b.index(g), d.get_loc(pd.Timestamp(day) + pd.Timedelta(days=1))] > thr[b.index(g)] for g, day in rows])
out["three_seeds"] = dict(aorc_sample_my_average=int(my3[sel].sum()), aorc_sample_file=int(f3a[sel].sum()), stage4_raw_sample=int(f3s[sel].sum()), rain_table_sample=int(f5t[sel].sum()),
                          aorc_all=int(f3a.sum()), stage4_raw_all=int(f3s.sum()), stage4_k_all=B3["models"]["wqa"]["variants"]["stage4_k"]["onsets_warned"],
                          k=B3["k_level_aorc_over_stage4"])
out["five_seeds"] = dict(aorc_sample=int(f5a[sel].sum()), stage4_raw_sample=int(f5s[sel].sum()), rain_table_sample=int(f5t[sel].sum()), aorc_all=int(f5a.sum()), stage4_raw_all=int(f5s.sum()),
                         stage4_k_all=S4["models"]["wqa"]["variants"]["stage4_k"]["onsets_warned"], k=S4["k_level_aorc_over_stage4"])
for n, v in (("three_seeds", out["three_seeds"]), ("five_seeds", out["five_seeds"])):
    a_s = v.get("aorc_sample", v.get("aorc_sample_file"))
    loss = (a_s - v["stage4_raw_sample"]) / a_s
    v["relative_loss_on_the_sample_raw"] = loss
    v["round1_route_extrapolated_hits"] = loss * v["aorc_all"]
    v["round1_route_extrapolated_points"] = 100 * loss * v["aorc_all"] / 611
    v["direct_all_611_raw_loss_hits"] = v["aorc_all"] - v["stage4_raw_all"]
    v["direct_all_611_raw_loss_points"] = 100 * (v["aorc_all"] - v["stage4_raw_all"]) / 611
    v["direct_all_611_level_matched_loss_hits"] = v["aorc_all"] - v["stage4_k_all"]
    v["direct_all_611_level_matched_loss_points"] = 100 * (v["aorc_all"] - v["stage4_k_all"]) / 611
    v["share_kept_on_the_sample_raw"] = (v["stage4_raw_sample"] - v["rain_table_sample"]) / (a_s - v["rain_table_sample"])
out["round1_bound"] = dict(points=2.6, hits=16, of=313, stage4_route_points=3.3, stage4_route_hits=20, sample=dict(aorc=92, stage4_raw=86, stage4_level_matched=95, rain_table=15))
out["files_read"] = READ
jsave("r6_hindsight.json", out)
print(json.dumps(out, indent=1))
