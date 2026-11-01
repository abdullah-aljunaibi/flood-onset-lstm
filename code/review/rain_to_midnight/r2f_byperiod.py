"""ROUND 2, check (f): the by_period table of the JSON recomputed with the reviewer's own loops (onset counts and NSE on the target days before and from 2009-07-17)."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json")
BP = R["by_period"]
ids = list(data())
PER = {"before_2009_07_17": ("2005-10-01", "2009-07-16"), "from_2009_07_17": ("2009-07-17", "2014-09-30")}
out = {"periods": {}, "sets": {}, "differences": {}}
dev = 0
for label, (a, b) in PER.items():
    n_days = len(pd.date_range(a, b)); share = n_days / len(pd.date_range(*TEST))
    j = BP["periods"][label]
    out["periods"][label] = dict(first=a, last=b, n_days=n_days, share=share, json_equal=(j["first_target_day"] == a and j["last_target_day"] == b and j["n_days"] == n_days and abs(j["share_of_the_test_days"] - share) < 1e-12))
    for n in ("wq", "wqn", "wqh", "wqh_older", "wqh_to09", "wqh_utc", "wqnh", "wqf"):
        c = counts(f"lstm_{n}", 1, a=a, b=b)
        nse = float(np.median(list(nse_per_basin(f"lstm_{n}", 1, a, b).values())))
        mine = dict(onsets_warned=c["pod"], hits=c["hits"], onsets=c["n_onsets"], false_warnings=c["far"], nse_median=nse)
        jj = BP["sets"][n][label]
        d = max(abs(mine[k] - jj[k]) for k in mine)
        dev = max(dev, d)
        out["sets"].setdefault(n, {})[label] = mine
print("by_period: 8 sets x 2 parts x 5 numbers recomputed with own loops: max abs deviation from the JSON:", dev)
print("periods equal the JSON:", {k: v["json_equal"] for k, v in out["periods"].items()}, "| days", {k: v["n_days"] for k, v in out["periods"].items()}, "| shares", {k: round(v["share"], 3) for k, v in out["periods"].items()})
out["max_dev"] = dev
print("\nonsets in each part (the same for every set):", {l: out["sets"]["wq"][l]["onsets"] for l in PER})
print("%-10s %-30s %-30s" % ("set", "before 2009-07-17: pod (hits/onsets) NSE", "from 2009-07-17: pod (hits/onsets) NSE"))
for n, v in out["sets"].items():
    print("%-10s %6.1f%% (%3d/%3d)  %.3f              %6.1f%% (%3d/%3d)  %.3f" % (n, 100 * v["before_2009_07_17"]["onsets_warned"], v["before_2009_07_17"]["hits"], v["before_2009_07_17"]["onsets"], v["before_2009_07_17"]["nse_median"],
                                                                                     100 * v["from_2009_07_17"]["onsets_warned"], v["from_2009_07_17"]["hits"], v["from_2009_07_17"]["onsets"], v["from_2009_07_17"]["nse_median"]))
print("\ndifferences (first minus second), points of onsets / hits / NSE, before and from:")
dd = 0
for key in BP["differences"]:
    T, C = key.split("_minus_")
    rec = {}
    for p in PER:
        rec[p] = dict(onset_points=100 * (out["sets"][T][p]["onsets_warned"] - out["sets"][C][p]["onsets_warned"]), hits=out["sets"][T][p]["hits"] - out["sets"][C][p]["hits"], nse=out["sets"][T][p]["nse_median"] - out["sets"][C][p]["nse_median"])
        j = BP["differences"][key][p]
        dd = max(dd, abs(rec[p]["onset_points"] - j["onset_points"]), abs(rec[p]["hits"] - j["hits"]), abs(rec[p]["nse"] - j["nse"]))
    out["differences"][key] = rec
    print(f"  {key:22s} before {rec['before_2009_07_17']['onset_points']:+6.1f} pts ({rec['before_2009_07_17']['hits']:+4d} hits, NSE {rec['before_2009_07_17']['nse']:+.3f})   from {rec['from_2009_07_17']['onset_points']:+6.1f} pts ({rec['from_2009_07_17']['hits']:+4d} hits, NSE {rec['from_2009_07_17']['nse']:+.3f})")
out["max_dev_differences"] = dd
print("differences against the JSON: max abs deviation", dd)
# how large a difference between the two parts could be chance? binomial standard errors of one pooled share in each part
for label in PER:
    n = out["sets"]["wq"][label]["onsets"]; p = out["sets"]["wqh"][label]["onsets_warned"]
    print(f"  one pooled share near {100*p:.0f}% of {n} onsets has a binomial standard error of {100*(p*(1-p)/n)**.5:.1f} points ({label})")
    out.setdefault("binomial_se_points", {})[label] = 100 * (p * (1 - p) / n) ** .5
jsave("r2f_byperiod.json", out)
