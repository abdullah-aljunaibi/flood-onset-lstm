"""The key figure: what limits the forecast, the kind of model or the information in its inputs? 1, 2 and 3 days ahead.

One figure for page 1 of the report and for the closing slide of the talk. The ladder runs down the page: one row for each set of information, from the
least (top) to the most (bottom), with the share of onsets warned on the left and the median NSE of the 12 basins on the right.
 Upper block, 1 day ahead, six sets:
  1. the river gauge only ('tomorrow = today'), 2. past weather only, 3. past weather + river gauge,
  4. + rain up to the evening of the issue day (NLDAS; its control, the same rain one day older, is a hollow circle),
  5. + rain up to midnight from AORC (hourly 1 km rain anchored on radar and gauges: an analysis made after the fact; the real-time check, Stage IV radar in
     its place on the issue day, is two hollow diamonds: with and without the last hour of the day),
  6. + a perfect weather forecast (an upper bound that cannot be had in practice; hatched).
 Lower block, 2 and 3 days ahead, five sets (each lead time in its own pair of narrow panels): the rain table + river gauge; + the AORC rain up to midnight;
  + a real weather forecast of the next three days (GEFS reforecast, put in place of the AORC rain of those days that the network was trained on); the perfect rain
  of the three days; the perfect weather of the three days (the last two are upper bounds, hatched).
The LSTM (the forecast averaged over its seeds) is a blue dot, its single seeds a thin line behind it. A thick bar runs from the level of past weather + river gauge
(a dashed vertical line) to the dot, so its length is what the extra information gives (in the lower block the light part is what the rain up to midnight gives, the
dark part what the forecast or the perfect rain adds to it). The other kinds of model are small marks, only where they read the same inputs as the LSTM: ridge
regression and random forest with past weather only, with past weather + river gauge, with the AORC rain (1 day) and with the GEFS forecast (2 and 3 days), and the
Transformer (its own experiment) with past weather + river gauge; a bracket there shows how far apart the four kinds lie. 'mixed' beside an NSE value says that the
gain over past weather + river gauge is a mixed result by the rule of section 4.2 of the report. A green band marks what GO asks for (its words run down the band; in the talk rendering, where the band has no room for them, the one word GO stands at its top).
The title, the subtitle and the headline state the finding only when the numbers show it (key 'finding': on the same inputs, past weather + river gauge, the kinds of
model lie closer together than every gain of the LSTM from more information, in onsets warned and in NSE); otherwise they only say what is drawn.

No model is trained and nothing is scored here. Every number is read from the results files of 02_baselines.py, 04_lstm_all.py, 10_rain_timing.py, 06_flood_analysis.py,
09_model_comparison.py, 16_aorc_rain_to_midnight.py and 18_forecast_input.py and, where they exist, of check_stage4_realtime.py (the real-time marks) and of
15_rain_to_midnight.py and 17_more_basins.py (notes only: the figure is drawn without them). Where two files (or two keys) hold the same quantity the script takes ONE of
them, checks the others against it and records both (key 'sources'); if they disagree by more than rounding it stops, names the two places and writes nothing. The reading
of a gain by the rule of section 4.2 (clear / mixed / none) is the one the results file stores where it stores one, and otherwise computed here by the same rule from the
per-basin NSE and the single seeds the files hold (rule_reading); it reproduces every stored reading, which is compared at every build. After drawing, every number that
the drawing plots or prints is looked up in 'sources', read again from its file or computed again as a difference, and the build stops if one does not match (key 'checks').
Writes results/key_figure.json (the drawn numbers, the sets, the notes, the headline, the verdicts, the same quantities for the sets that are not drawn) and three renderings
of the ONE drawing, made by one function from that file on every run:
  figures/key_figure.png        with its own title, subtitle and footnote;
  figures/key_figure_plain.png  the same drawing and lettering without them and without the rows 'river gauge only' and 'past weather only' (page 1 of the report: the caption
                                carries the words, and the page has room for one figure of this height; key figure.plain in the JSON says which rows and why);
  figures/key_figure_talk.png   for the closing slide: 8.0 inches wide, lettering 1.3 times larger, without the rows and small marks that are not in the headline.
Their names, sizes and what each leaves out are in key_figure.json under 'files' and 'figure.renderings'. Run time: some seconds.
Smoke test: FLOOD_QUICK=1 python code/11_key_figure.py  reads and writes only results/quick and figures/quick; if a smoke-test file of an earlier script is missing (or lacks
a table) it says so in one line and ends without an error.
"""
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import flood as F
from common import *

use_style()
t_start = time.time()
LEAD = 1                                                   # the lead time of the upper block of the drawing
FAR = (2, 3)                                               # the lead times of the lower block
UPSTREAM = {"baselines": "02_baselines.py", "lstm_all": "04_lstm_all.py", "rain_timing": "10_rain_timing.py", "flood_analysis": "06_flood_analysis.py",
            "model_comparison": "09_model_comparison.py", "aorc_rain_to_midnight": "16_aorc_rain_to_midnight.py", "forecast_input": "18_forecast_input.py"}
OPTIONAL = {"stage4_realtime": "check_stage4_realtime.py", "more_basins": "17_more_basins.py", "rain_to_midnight": "15_rain_to_midnight.py"}   # read when they exist: the real-time check of the AORC rung, notes on the experiments that are not drawn
TOL = 5e-4                                                 # 'agree to rounding': the documents print NSE to 0.01 and shares to 1%
P_CUT = 0.05                                               # the line for 'p is small' in section 4.2 of the report
# the sets of information of the upper block (1 day ahead), from top to bottom: key, name as in the report, the LSTM that reads it
SETS = [dict(key="q", label="river gauge only ('tomorrow = today')", lstm=None),
        dict(key="w", label="past weather only", lstm="lstm_w"),
        dict(key="wq", label="past weather + river gauge", lstm="lstm_wq"),
        dict(key="wqn", label="+ rain up to the evening of the issue day", lstm="lstm_wqn"),
        dict(key="wqa", label="+ rain up to midnight, radar- and gauge-anchored (AORC)", lstm="lstm_wqa"),
        dict(key="wqf", label="+ perfect weather forecast (upper bound)", lstm="lstm_wqf")]
# the sets of the lower block (2 and 3 days ahead); the first three are the same networks as above at the other lead times
FAR_SETS = [dict(key="wq", label="rain table + river gauge", lstm="lstm_wq"),
            dict(key="wqa", label="+ rain up to midnight (AORC)", lstm="lstm_wqa"),
            dict(key="wqaF_sub", label="+ a real weather forecast of the next three days (GEFS)", lstm="lstm_wqaF_sub"),
            dict(key="wqaP", label="perfect rain of the three days (upper bound)", lstm="lstm_wqaP"),
            dict(key="wqf", label="perfect weather of the three days (upper bound)", lstm="lstm_wqf")]
UPPER = ("wqf", "wqaP")                                    # the sets that are upper bounds
DRAWN = {1: {"q", "w", "wq", "wqn", "wqa", "wqf"}, 2: {"wq", "wqa", "wqaF_sub", "wqaP", "wqf"}, 3: {"wq", "wqa", "wqaF_sub", "wqaP", "wqf"}}   # LSTM sets in the figure, by lead
KIND = {"persistence": "'tomorrow = today'", "linreg": "ridge regression", "rf": "random forest", "transformer": "Transformer", "lstm": "LSTM"}
MARK = {"persistence": "D", "linreg": "^", "rf": "s", "transformer": "*", "lstm": "o"}       # as in the other figures; the Transformer gets a shape of its own
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PHRASE = {"lstm": "the LSTM is ahead", "lstm_avg": "the LSTM is ahead only as the average of its seeds", "lstm_med": f"the LSTM scores higher, but p is above {P_CUT:g}",
          "tie": "no clear difference", "simple": "the simple model is ahead", "nocall": "not like-for-like: no winner is called"}
READ = {"clear gain": "clear", "mixed": "mixed", "no clear difference": "none", "clear loss": "loss"}          # the readings the newer results files store (report section 4.2)
IN_WORDS = {"clear": "a clear gain", "mixed": "mixed", "none": "no clear difference", "loss": "a clear loss"}
BOUND = "an upper bound that cannot be had in practice"    # the fixed wording for the perfect forecasts
AFTER = "an analysis made after the fact"                  # the fixed wording for AORC
RES_DIR = "results/quick" if F.QUICK else "results"
OUT_RES, OUT_FIG = F.RES_OUT, F.FIG_OUT                    # where the file and the pictures are written (a driver may point them elsewhere)
R, SOURCES, CHECKS, CRIT = {}, {}, dict(compared=0, tolerance=TOL, largest_difference=0.0, disagreements=[]), {}

lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"
gone = lambda x: x is None or x != x                        # no value (a rate whose denominator is zero)
pc = lambda x: "no value" if gone(x) else f"{round(100 * x) + 0}%"                              # a share in per cent (never '-0%')
n2 = lambda x: "no value" if gone(x) else f"{round(x, 2) + 0:.2f}"
sg = lambda x: "no value" if gone(x) else f"{round(x, 2) + 0:+.2f}"                             # a step in NSE
pts = lambda x: "no value" if gone(x) else f"{round(100 * x) + 0:+d}"                           # a step of a share, in percentage points
word = lambda n: NUMBER.get(n, str(n))
above = lambda a, b: bool(a) and bool(b) and min(a) > max(b)                                    # every single seed of a above every single seed of b
where = lambda file, path=None: f"{RES_DIR}/{file}.json" + (f": {path}" if path else "")
nm = lambda tag, s, q: f"{tag}.lstm_{s}.{q}"                # the name under which a number of an LSTM is kept in 'sources'
hm = lambda t: t["hits"] + t["misses"]


# ---- reading the results files: one source per number, every other copy checked ----------------------------------------
def at(file, path):
    """One value of a results file by its key path, e.g. at('rain_timing', 'closure.wqn.by_lead[0].gain')."""
    v = R[file]
    for p in re.findall(r"\[\d+\]|[^.\[\]]+", path):
        v = v[int(p[1:-1])] if p.startswith("[") else v[p]
    return v


def has(file, path):
    """Does the results file hold this key path (and is the file there)?"""
    try:
        return R.get(file) is not None and at(file, path) is not None
    except (KeyError, IndexError, TypeError):
        return False


def agree(name, v, *also, origin="computed in 11_key_figure.py"):
    """Each (file, key path) of `also` holds the same quantity as v, which came from `origin`: count the comparison, keep a disagreement."""
    for file, path in also:
        w = at(file, path)
        if isinstance(v, str) or isinstance(w, str):
            d = 0.0 if v == w else float("inf")
        else:
            d = 0.0 if gone(v) and gone(w) else float("inf") if gone(v) or gone(w) else abs(v - w)
        CHECKS["compared"] += 1
        if d > TOL:
            CHECKS["disagreements"].append(f"{name}: {v!r} ({origin}), but {w!r} in {where(file, path)}")
        else:
            CHECKS["largest_difference"] = max(CHECKS["largest_difference"], d)


def take(name, file, path, *also, k=LEAD, drawn=True, rec=True):
    """One number from its ONE source (file, key path); `also` lists the other places that hold it. The source is kept under `name` (drawn: the
    number is in the figure, as a mark or in its lettering)."""
    v = at(file, path)
    agree(name, v, *also, origin=where(file, path))
    if rec:
        SOURCES[name] = dict(file=where(file), file_id=file, key=path, value=v, in_figure=drawn, checked_against=[where(f, p) for f, p in also])
    return v


def derive(name, v, how, *also, k=LEAD, drawn=True, parts=None, rec=True):
    """A number computed here from numbers taken above (a step, a spread); `also`: where a results file holds the same thing. parts: the names
    of the two numbers it is the difference of (plus, minus), so that the check at the end of the build can compute it again."""
    agree(name, v, *also, origin=f"computed here: {how}")
    if rec:
        SOURCES[name] = dict(computed=how, value=v, in_figure=drawn, checked_against=[where(f, p) for f, p in also], **(dict(difference_of=list(parts)) if parts else {}))
    return v


# ---- the report's rules for a difference, restated here so that the notes use the report's words ---------------------------
def reading(c):
    """LSTM against a simple model on the same inputs (section 4.2 of the report): the LSTM is ahead only when the simple model is below every single seed, the LSTM is higher in most basins and p is small."""
    if c.get("like_for_like") is False:
        return "nocall"
    most, few = c["n_basins_lstm_higher"] > c["n_basins"] / 2, c["n_basins_lstm_higher"] < c["n_basins"] / 2
    small_p, below = c["wilcoxon_p"] < P_CUT, c["baseline_against_seed_spread"] == "below every seed"
    if most and small_p:
        return "lstm" if below else "lstm_avg"
    if few and small_p:
        return "simple"
    return "lstm_med" if most and below else "tie"


def share_word(v):
    """A share in plain words where it lies near a simple fraction ('about a quarter'), otherwise as a percentage."""
    for lo, hi, w in ((0.22, 0.28, "about a quarter"), (0.30, 0.36, "about a third"), (0.45, 0.55, "about half"), (0.64, 0.69, "about two thirds"), (0.72, 0.78, "about three quarters")):
        if not gone(v) and lo <= v <= hi:
            return w
    return pc(v)


def verdict(t, crit):
    """GO / CAUTION / NOT YET for one warning table (pod, far, csi) by the criteria stored in flood_analysis.json."""
    if gone(t["pod"]) or t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= crit["go_min_warned"] and not gone(t["far"]) and t["far"] <= crit["go_max_false"]:
        return "GO"
    return "CAUTION" if t["csi"] >= crit["caution_min_csi"] else "NOT YET"


def against_of(c, best):
    """How the LSTM of an input set compares with one simple model given the same inputs (a comparison as lstm_all.json stores it)."""
    r = reading(c)
    return dict(like_for_like=c.get("like_for_like", True), best_simple_model_by_validation=best, nse_lstm_minus_model=c["difference_of_medians"], n_basins_lstm_higher=c["n_basins_lstm_higher"],
                n_basins=c["n_basins"], n_basins_lstm_higher_by_seed=c.get("n_basins_lstm_higher_by_seed"), wilcoxon_p=c["wilcoxon_p"], model_against_lstm_seeds=c["baseline_against_seed_spread"],
                reading=r, in_words=PHRASE[r])


# ---- the sets of the newer scripts (16_aorc_rain_to_midnight.py, 18_forecast_input.py): the same numbers read from two or more places ----
def copies(file, s, k):
    """Where else the results files hold the numbers of set s of a newer script at lead k, by quantity: lists of (file, key path) for `agree`.
    Both files hold copies of the sets they share (wq, wqa, wqf); each also holds an assembled ladder, a table beside the simple models and a closure table."""
    L, i = lk(k), k - 1
    out = dict(nse=[], onset_warned=[], onset_false=[], seeds=[])
    for f2, ladder in (("aorc_rain_to_midnight", "ladder.wq.rungs"), ("forecast_input", "ladder.main.rungs")):
        if f2 != file and has(f2, f"summary_test.sets.{s}.{L}"):
            b = f"summary_test.sets.{s}.{L}"
            out["nse"].append((f2, f"{b}.nse_median")); out["onset_warned"].append((f2, f"{b}.onset.pod")); out["onset_false"].append((f2, f"{b}.onset.far"))
        j = next((j for j, r in enumerate(at(f2, ladder)) if r["set"] == f"lstm_{s}"), None)
        if j is not None:
            r = f"{ladder}[{j}].by_lead.{L}"
            out["nse"].append((f2, f"{r}.nse_median")); out["onset_warned"].append((f2, f"{r}.onset_warned")); out["onset_false"].append((f2, f"{r}.onset_false"))
        if has(f2, f"yardstick.lstm_beside.{L}.{s}"):
            b = f"yardstick.lstm_beside.{L}.{s}"
            out["nse"].append((f2, f"{b}.nse_median")); out["onset_warned"].append((f2, f"{b}.onset.pod")); out["onset_false"].append((f2, f"{b}.onset.far"))
        cl = None                                           # the closure table of the set: keyed by the set in 16_, by 'set' in 18_
        if f2 == "forecast_input":
            cl = next((c for c, v in at(f2, "closure").items() if v.get("set") == f"lstm_{s}"), None)
        elif has(f2, f"closure.{s}.by_lead"):
            cl = s
        if cl is not None:
            b = f"closure.{cl}.by_lead[{i}]"
            out["nse"].append((f2, f"{b}.nse_with")); out["onset_warned"].append((f2, f"{b}.onset_warned_with")); out["onset_false"].append((f2, f"{b}.onset_false_with"))
        if f2 != file and has(f2, "seeds"):
            out["seeds"].append((f2, "seeds"))
    return out


def newer(file, s, k, crit, drawn):
    """The all-river LSTM of set s of 16_aorc_rain_to_midnight.py or 18_forecast_input.py at lead k: the forecast averaged over its seeds and the
    single seeds, read from the file that ran it (summary_test, seed_ranges) and compared with every other place that holds the same numbers."""
    L = lk(k)
    base, sr = f"summary_test.sets.{s}.{L}", f"seed_ranges.sets.{s}.{L}"
    dup = copies(file, s, k)
    T = lambda q, path, *also, shown=drawn: take(nm(L, s, q), file, path, *also, k=k, drawn=shown)
    out = dict(nse=T("nse", f"{base}.nse_median", *dup["nse"]), onset_warned=T("onset_warned", f"{base}.onset.pod", *dup["onset_warned"]),
               onset_false=T("onset_false", f"{base}.onset.far", *dup["onset_false"], shown=False), onset_csi=T("onset_csi", f"{base}.onset.csi", shown=False))
    out["seeds"] = at(file, "seeds")
    for f2, p in dup["seeds"]:
        CHECKS["compared"] += 1
        if at(f2, p) != out["seeds"]:
            CHECKS["disagreements"].append(f"seeds of {s}: {out['seeds']} in {where(file)}, but {at(f2, p)} in {where(f2, p)}")
    out["nse_seeds"] = [T(f"nse.seed{x}", f"{sr}.by_seed.nse_median[{j}]") for j, x in enumerate(out["seeds"])]
    out["onset_warned_seeds"] = [T(f"onset_warned.seed{x}", f"{sr}.by_seed.onset_pod[{j}]") for j, x in enumerate(out["seeds"])]
    origin = f"the single seeds of {where(file, f'{sr}.by_seed')}"
    for q, key in (("nse", "nse_median"), ("onset_warned", "onset_pod")):
        for f2 in [file] + [f for f in ("aorc_rain_to_midnight", "forecast_input") if f != file and has(f, f"seed_ranges.sets.{s}.{L}")]:
            agree(f"{L}.lstm_{s}.{q}, lowest single seed", min(out[f"{q}_seeds"]), (f2, f"seed_ranges.sets.{s}.{L}.{key}[0]"), origin=origin)
            agree(f"{L}.lstm_{s}.{q}, highest single seed", max(out[f"{q}_seeds"]), (f2, f"seed_ranges.sets.{s}.{L}.{key}[1]"), origin=origin)
    out["n_onsets"] = hm(at(file, f"{base}.onset"))
    out["onset_verdict"] = verdict(at(file, f"{base}.onset"), crit)                  # at the fixed warning level, the level of the figure
    for f2 in ("aorc_rain_to_midnight", "forecast_input"):
        if has(f2, f"verdicts.sets.{s}.{L}.plain.onset.verdict"):
            agree(f"{L}.lstm_{s}.onset_verdict", out["onset_verdict"], (f2, f"verdicts.sets.{s}.{L}.plain.onset.verdict"), origin="the stored criteria applied to the onset table")
    c = f"verdicts.sets.{s}.{L}.chosen_on_validation_onsets"                         # at the warning level chosen on the validation years: the verdict as stored
    other = [(f2, f"{c}.onset.verdict") for f2 in ("aorc_rain_to_midnight", "forecast_input") if f2 != file and has(f2, f"{c}.onset.verdict")]
    out["onset_verdict_at_chosen_level"] = take(f"{L}.lstm_{s}.onset_verdict_at_chosen_level", file, f"{c}.onset.verdict", *other, k=k, drawn=False)
    out["chosen_level"] = take(f"{L}.lstm_{s}.chosen_level", file, f"{c}.trigger", k=k, drawn=False)
    return out


def newer_simple(file, s, name, k, drawn):
    """A simple model given the same inputs as the LSTM of set s of a newer script (the yardstick of that script): median NSE, the share of onsets
    warned and the share of those warnings that were false, at the fixed warning level."""
    L = lk(k)
    t, sm = f"yardstick.table_test.{L}.{name}", f"yardstick.sets.{name}.test.summary.{L}"
    out = {}
    for q, key in (("nse", "nse_median"), ("onset_warned", "onset.pod"), ("onset_false", "onset.far")):
        also = [(file, f"{sm}.{key}")] + [(f2, f"yardstick.table_test.{L}.{name}.{key}") for f2 in ("aorc_rain_to_midnight", "forecast_input") if f2 != file and has(f2, f"yardstick.table_test.{L}.{name}")]
        out[q] = take(f"{L}.{name}.{q}", file, f"{t}.{key}", *also, k=k, drawn=drawn and q != "onset_false")
    on = at(file, f"{t}.onset")
    out["onset_verdict"] = verdict(on, CRIT)
    agree(f"{L}.{name}.onset_verdict", out["onset_verdict"], (file, f"{t}.onset_verdict_plain"), origin="the stored criteria applied to the onset table")
    return out


def newer_against(file, s, name, k):
    """The comparison of the LSTM of set s with a simple model given the same inputs, as the newer script stores it, read by the rule of section 4.2."""
    L = lk(k)
    c = at(file, f"yardstick.lstm_against_simple.lstm_{s}__{name}.{L}")
    t = f"yardstick.table_test.{L}"
    inputs = name.split("_", 2)[2]
    best = (at(file, f"yardstick.best_simple_on_the_new_input_by_validation.{L}") if has(file, f"yardstick.best_simple_on_the_new_input_by_validation.{L}")
            else max((f"base_linreg_{inputs}", f"base_rf_{inputs}"), key=lambda n: at(file, f"{t}.{n}.val_nse_median")))        # by the validation years, as lstm_all.json does
    a = against_of(c, best == name)
    agree(f"{L}.{name}.reading", a["reading"], (file, f"yardstick.lstm_against_simple.lstm_{s}__{name}.{L}.reading"), origin="the rule of report section 4.2 applied to the stored comparison")
    return a


def real_time(k, n_onsets):
    """The real-time check of the AORC rung (check_stage4_realtime.py): the three saved networks of the AORC set given Stage IV radar and gauge rain,
    the product a service at midnight would read, in place of AORC on the issue day of every onset; the number of onsets warned."""
    st = R.get("stage4_realtime")
    if st is None or k != LEAD:
        return None
    L, m = lk(k), "models.wqa"
    T = lambda name, path, *also, drawn=True: take(f"{L}.real_time.{name}", "stage4_realtime", path, *also, k=k, drawn=drawn)
    tested = T("onsets_tested", f"{m}.onsets_tested", ("stage4_realtime", "onsets.tested"), ("stage4_realtime", "onsets.total"), drawn=False)
    CHECKS["compared"] += 1
    if tested != n_onsets:
        CHECKS["disagreements"].append(f"{L}.real_time: {tested} onsets tested in {where('stage4_realtime')}, but {n_onsets} in {where('lstm_all')}")
    aorc = T("aorc.onsets_warned", f"{m}.aorc", ("aorc_rain_to_midnight", f"summary_test.sets.wqa.{L}.onset.hits"), drawn=False)
    table = T("rain_table_alone.onsets_warned", f"{m}.rain_table_alone", ("lstm_all", f"sets.wq.test.summary.{L}.onset.hits"), drawn=False)
    out = dict(product="Stage IV", onsets_tested=tested, aorc_onsets_warned=aorc, rain_table_alone_onsets_warned=table, variants=[])
    for key, label in (("stage4", "Stage IV radar and gauge rain in place of AORC on the issue day"),
                       ("stage4_no_last_hour", "the same with the last hour of the issue day (23:00 to 24:00) left out, as the delay of a real-time product would")):
        v = f"{m}.variants.{key}"
        hits = T(f"{key}.onsets_warned", f"{v}.onsets_warned", drawn=False)
        share = derive(f"{L}.real_time.{key}.onset_warned", hits / tested, f"{L}.real_time.{key}.onsets_warned / {L}.real_time.onsets_tested", ("stage4_realtime", f"{v}.share_of_the_onsets"), k=k)
        out["variants"].append(dict(key=key, label=label, onsets_warned=hits, onset_warned=share, share_of_the_aorc_effect_surviving=T(f"{key}.share_of_the_aorc_effect_surviving", f"{v}.share_of_the_aorc_effect_surviving", drawn=False),
                                    single_seed_onsets_warned=[T(f"{key}.single_seed_onsets_warned[{j}]", f"{v}.single_seed_onsets_warned[{j}]", drawn=False) for j in range(len(at("stage4_realtime", f"{v}.single_seed_onsets_warned")))]))
    s4 = out["variants"][0]
    out["interval_share_of_the_aorc_effect_surviving"] = [T(f"stage4.interval_share_surviving[{j}]", f"{m}.variants.stage4.intervals.over_onsets.share_surviving[{j}]", drawn=False) for j in range(2)]
    agree(f"{L}.real_time.stage4.share_surviving", (s4["onsets_warned"] - table) / (aorc - table) if aorc != table else None, ("stage4_realtime", f"{m}.variants.stage4.share_of_the_aorc_effect_surviving"),
          origin="computed here: (Stage IV - rain table) / (AORC - rain table), onsets warned")
    out["note"] = ("the saved networks of the AORC set (trained on AORC, not on Stage IV) given Stage IV in place of AORC on the issue day of the onsets; the issue day only, so the false warnings are "
                   "not tested here (results/stage4_realtime.json: honesty_points)")
    return out


def rule_reading(file, T, C, k, stored=None):
    """Set T against set C at lead k by the rule of report section 4.2, as 16_aorc_rain_to_midnight.py and 18_forecast_input.py apply it, from the per-basin NSE and the
    single seeds those files store: a clear gain needs every single seed of T above every seed of C, T higher in most basins AND a Wilcoxon p below 0.05; 'mixed' when
    only the seeds say so. `stored`: (file, key path) of the reading the file already holds for this pair; the two must agree (that is the test of this function)."""
    from scipy.stats import wilcoxon
    L = lk(k)
    ids = list(at(file, f"per_basin.nse.{C}.{L}").keys())
    t, c = [at(file, f"per_basin.nse.{T}.{L}.{g}") for g in ids], [at(file, f"per_basin.nse.{C}.{L}.{g}") for g in ids]
    sT, sC = at(file, f"seed_ranges.sets.{T}.{L}.by_seed.nse_median"), at(file, f"seed_ranges.sets.{C}.{L}.by_seed.nse_median")
    up, down = above(sT, sC), above(sC, sT)
    hi, lo = sum(a > b for a, b in zip(t, c)), sum(a < b for a, b in zip(t, c))
    try:
        p = float(wilcoxon(t, c).pvalue)
    except ValueError:
        p = float("nan")
    small = bool(p < P_CUT)
    r = "clear gain" if up and hi > len(ids) / 2 and small else "clear loss" if down and lo > len(ids) / 2 and small else "mixed" if (up or down) else "no clear difference"
    if stored:
        agree(f"{lk(k)}.reading {T} against {C}", r, stored, origin=f"the rule of report section 4.2 applied to the per-basin NSE of {where(file)}")
        agree(f"{lk(k)}.basins {T} against {C}", hi, (stored[0], stored[1].rsplit(".", 1)[0] + ".n_basins_higher"), origin=f"computed here from {where(file)}")
    return dict(reading=READ[r], n_basins_better=hi, n_basins=len(ids), wilcoxon_p=p, seeds_all_above=up, source=where(file, "per_basin.nse"))


# ---- everything of one lead time ---------------------------------------------------------------------------------------------
def collect(k, crit, n_b):
    """Levels, control, steps and onset count of one lead time, read from the results files. Every number is kept in 'sources' with the file and the
    key it was read from (in_figure says whether the figure draws it at this lead time). 'ladder2' and 'steps2' are the lower block of the drawing."""
    L, i = lk(k), k - 1
    shown = lambda s: s in DRAWN[k]                         # is the LSTM of set s in the figure at this lead time?
    T = lambda name, file, path, *also, drawn=True: take(f"{L}.{name}", file, path, *also, k=k, drawn=drawn)
    D = lambda name, v, how, *also, drawn=True, parts=None: derive(f"{L}.{name}", v, how, *also, k=k, drawn=drawn, parts=parts)
    skill, comp = R["flood_analysis"]["skill"], R["lstm_all"]["comparison"]
    services = R["flood_analysis"]["verdict"]["services"]
    n_onsets = hm(at("lstm_all", f"sets.wq.test.summary.{L}.onset"))

    def simple(name, inputs, shown_):
        """A simple model of 02_baselines.py; `inputs`: the LSTM input set whose comparison table in lstm_all.json repeats its scores."""
        out = {}
        for q, key, ckey in (("nse", "nse_median", "baseline_nse_median"), ("onset_warned", "onset.pod", "baseline_onset_pod"), ("onset_false", "onset.far", "baseline_onset_far")):
            also = [("lstm_all", f"baselines.{name}.test.summary.{L}.{key}"), ("lstm_all", f"comparison.{inputs}.baselines.{name}.{L}.{ckey}")]
            also += [("flood_analysis", f"skill.{name}.plain.test.summary.{L}.{key}")] if name in skill else []
            out[q] = T(f"{name}.{q}", "baselines", f"models.{name}.test.summary.{L}.{key}", *also, drawn=shown_ and q != "onset_false")
        out["onset_verdict"] = verdict(at("baselines", f"models.{name}.test.summary.{L}.onset"), crit)
        return out

    def against(inputs, name):
        """How the LSTM of an input set compares with one simple model given the same inputs (lstm_all.json, comparison)."""
        return against_of(comp[inputs]["baselines"][name][L], comp[inputs]["best_baseline_by_validation"][L] == name)

    def lstm(s):
        """The all-basin LSTM of 04_lstm_all.py on one input set: the forecast averaged over its seeds, and the single seeds."""
        row, col = f"closure.{'wqn' if 'q' in s else 'wn'}.by_lead[{i}]", "perfect_forecast" if "f" in s else "base"
        out = {}
        for q, key, dkey, ckey in (("nse", "nse_median", "nse_median", "lstm_nse_median"), ("onset_warned", "onset.pod", "onset_pod", "lstm_onset_pod"),
                                   ("onset_false", "onset.far", "onset_far", "lstm_onset_far"), ("onset_csi", "onset.csi", "onset_csi", None)):
            also = [("lstm_all", f"decomposition.levels.{s}.{L}.{dkey}"), ("rain_timing", f"models.{s}.test.summary.{L}.{key}")]
            also += [("rain_timing", f"{row}.{q}_{col}")] if q != "onset_csi" else []
            also += [("flood_analysis", f"skill.lstm_{s}.plain.test.summary.{L}.{key}")] if f"lstm_{s}" in skill else []
            also += [("lstm_all", f"comparison.{s}.baselines.{n}.{L}.{ckey}") for n in comp[s]["baselines"]] if ckey else []
            if s in ("wq", "wqf"):                          # the newer files hold copies of these two sets
                also += [(f2, f"summary_test.sets.{s}.{L}.{key}") for f2 in ("aorc_rain_to_midnight", "forecast_input") if has(f2, f"summary_test.sets.{s}.{L}.{key}")]
            out[q] = T(f"lstm_{s}.{q}", "lstm_all", f"sets.{s}.test.summary.{L}.{key}", *also, drawn=q in ("nse", "onset_warned") and shown(s))
        out["seeds"] = at("lstm_all", f"sets.{s}.seeds")
        by_seed = skill.get(f"lstm_{s}", {}).get("seeds", {}).get(L, {}).get("plain", {}).get("onset", {}).get("by_seed", {})
        out["nse_seeds"] = [T(f"lstm_{s}.nse.seed{x}", "lstm_all", f"sets.{s}.per_seed.{x}.test.{L}.nse_median", drawn=shown(s)) for x in out["seeds"]]
        out["onset_warned_seeds"] = [T(f"lstm_{s}.onset_warned.seed{x}", "lstm_all", f"sets.{s}.per_seed.{x}.test.{L}.onset.pod",
                                       *([("flood_analysis", f"skill.lstm_{s}.seeds.{L}.plain.onset.by_seed.{x}.pod")] if str(x) in by_seed else []), drawn=shown(s)) for x in out["seeds"]]
        origin = f"the single seeds of {where('lstm_all', f'sets.{s}.per_seed')}"
        for q, lo, hi, rng in (("nse", "nse_seed_median_min", "nse_seed_median_max", "seed_range_base"), ("onset_warned", "onset_pod_seed_min", "onset_pod_seed_max", "onset_seed_range_base")):
            agree(f"{L}.lstm_{s}.{q}, lowest single seed", min(out[f"{q}_seeds"]), ("lstm_all", f"decomposition.levels.{s}.{L}.{lo}"), *([("rain_timing", f"{row}.{rng}[0]")] if col == "base" else []), origin=origin)
            agree(f"{L}.lstm_{s}.{q}, highest single seed", max(out[f"{q}_seeds"]), ("lstm_all", f"decomposition.levels.{s}.{L}.{hi}"), *([("rain_timing", f"{row}.{rng}[1]")] if col == "base" else []), origin=origin)
        out["onset_verdict"] = verdict(at("lstm_all", f"sets.{s}.test.summary.{L}.onset"), crit)       # the verdict for onsets at the high-flow level itself
        if f"lstm_{s}" in services:
            agree(f"{L}.lstm_{s}.onset_verdict", out["onset_verdict"], ("flood_analysis", f"verdict.services.lstm_{s}.{L}.onset.at_plain_level"), origin="the stored criteria applied to the onset table")
            also = [(f2, f"verdicts.sets.{s}.{L}.chosen_on_validation_onsets.onset.verdict") for f2 in ("aorc_rain_to_midnight", "forecast_input") if has(f2, f"verdicts.sets.{s}.{L}.chosen_on_validation_onsets.onset.verdict")]
            out["onset_verdict_at_chosen_level"] = T(f"lstm_{s}.onset_verdict_at_chosen_level", "flood_analysis", f"verdict.services.lstm_{s}.{L}.onset.at_level_chosen_for_onsets", *also, drawn=False)
        return out

    def timing(s, col):
        """A model of the rain-timing experiment (10_rain_timing.py): lstm_wqn (col 'with_nldas') or its control lstm_wqn_older (col 'control')."""
        row, m = f"closure.wqn.by_lead[{i}]", f"models.{s}.test.summary.{L}"
        nse_key, rng = ("nse_with_nldas", "seed_range_with_nldas") if col == "with_nldas" else ("nse_control_one_day_older", "seed_range_control")
        is_main = col == "with_nldas"
        on = shown("wqn") if is_main else k == LEAD         # the control is drawn at the upper block only
        out = dict(nse=T(f"lstm_{s}.nse", "rain_timing", f"{row}.{nse_key}", ("rain_timing", f"{m}.nse_median"), drawn=on),
                   onset_warned=T(f"lstm_{s}.onset_warned", "rain_timing", f"{row}.onset_warned_{col}", ("rain_timing", f"{m}.onset.pod"), drawn=on),
                   onset_false=T(f"lstm_{s}.onset_false", "rain_timing", f"{row}.onset_false_{col}", ("rain_timing", f"{m}.onset.far"), drawn=False),
                   onset_csi=T(f"lstm_{s}.onset_csi", "rain_timing", f"{m}.onset.csi", drawn=False))
        per = at("rain_timing", f"models.{s}.per_seed")
        out["seeds"] = [p["seed"] for p in per]
        out["nse_seeds"] = [T(f"lstm_{s}.nse.seed{p['seed']}", "rain_timing", f"models.{s}.per_seed[{j}].test_nse_median[{i}]", drawn=is_main and shown("wqn")) for j, p in enumerate(per)]
        out["onset_warned_seeds"] = [T(f"lstm_{s}.onset_warned.seed{p['seed']}", "rain_timing", f"models.{s}.per_seed[{j}].test_onset_warned[{i}]", drawn=is_main and shown("wqn")) for j, p in enumerate(per)]
        origin = f"the single seeds of {where('rain_timing', f'models.{s}.per_seed')}"
        for q, r_ in (("nse", rng), ("onset_warned", f"onset_{rng}")):
            agree(f"{L}.lstm_{s}.{q}, lowest single seed", min(out[f"{q}_seeds"]), ("rain_timing", f"{row}.{r_}[0]"), origin=origin)
            agree(f"{L}.lstm_{s}.{q}, highest single seed", max(out[f"{q}_seeds"]), ("rain_timing", f"{row}.{r_}[1]"), origin=origin)
        out["onset_verdict"] = verdict(at("rain_timing", f"models.{s}.test.summary.{L}.onset"), crit)
        if has("aorc_rain_to_midnight", f"verdicts.sets.{s}.{L}.plain.onset.verdict"):
            agree(f"{L}.lstm_{s}.onset_verdict", out["onset_verdict"], ("aorc_rain_to_midnight", f"verdicts.sets.{s}.{L}.plain.onset.verdict"), origin="the stored criteria applied to the onset table")
            out["onset_verdict_at_chosen_level"] = T(f"lstm_{s}.onset_verdict_at_chosen_level", "aorc_rain_to_midnight", f"verdicts.sets.{s}.{L}.chosen_on_validation_onsets.onset.verdict", drawn=False)
        return out

    def compared(m):
        """Transformer or its LSTM twin (09_model_comparison.py): the forecast averaged over its seeds, and the single seeds."""
        c, s, seeds = f"comparison.test.{L}.{m}", f"models.{m}.seed_average.test.summary.{L}", at("model_comparison", "seeds")
        on = k == LEAD
        return dict(nse=T(f"cmp_{m}.nse", "model_comparison", f"{c}.nse_median", ("model_comparison", f"{s}.nse_median"), drawn=on),
                    onset_warned=T(f"cmp_{m}.onset_warned", "model_comparison", f"{c}.onset_pod", ("model_comparison", f"{s}.onset.pod"), drawn=on),
                    onset_false=T(f"cmp_{m}.onset_false", "model_comparison", f"{c}.onset_far", ("model_comparison", f"{s}.onset.far"), drawn=False), seeds=seeds,
                    nse_seeds=[T(f"cmp_{m}.nse.seed{x}", "model_comparison", f"{c}.nse_median_by_seed[{j}]", ("model_comparison", f"models.{m}.per_seed.{x}.test.{L}.nse_median"), drawn=False)
                               for j, x in enumerate(seeds)],
                    onset_warned_seeds=[T(f"cmp_{m}.onset_warned.seed{x}", "model_comparison", f"{c}.onset_pod_by_seed[{j}]", ("model_comparison", f"models.{m}.per_seed.{x}.test.{L}.onset.pod"), drawn=False)
                                        for j, x in enumerate(seeds)])

    def with_new_input(lv, file, s, models):
        """Set s of a newer script: its LSTM, and the simple models given the same inputs (drawn at the lead times where the set is in the figure)."""
        lv["lstm"] = newer(file, s, k, crit, shown(s))
        for n in models:
            kind = "linreg" if "linreg" in n else "rf"
            a = newer_against(file, s, n, k)
            lv["others"][n] = dict(label=KIND[kind], kind=kind, same_inputs=True, drawn=shown(s) and (k == LEAD or s == "wqaF_sub"), **newer_simple(file, s, n, k, shown(s) and (k == LEAD or s == "wqaF_sub")), against_lstm=a)

    # ---- the six levels of the upper block ----
    levels = []
    for j, S in enumerate(SETS):
        key = S["key"]
        lv = dict(key=key, number=j + 1, label=S["label"], prediction_set=S["lstm"] or "base_persistence", lstm=None, others={})
        if key == "q":                                      # no LSTM reads the gauge alone: the model of this set is 'tomorrow = today'
            lv["others"]["base_persistence"] = dict(label=KIND["persistence"], kind="persistence", same_inputs=True, drawn=k == LEAD, **simple("base_persistence", "wq", k == LEAD),
                                                    against_lstm_of_set_wq=against("wq", "base_persistence"))
        elif key == "wqn":
            lv["lstm"] = timing("wqn", "with_nldas")
        elif key == "wqa":                                  # the AORC rain to midnight (16_aorc_rain_to_midnight.py), beside the simple models given the same input
            with_new_input(lv, "aorc_rain_to_midnight", "wqa", ("base_linreg_wqa", "base_rf_wqa"))
            lv["real_time"] = real_time(k, n_onsets)
            lv["control_not_drawn"] = dict(label="control: the same AORC rain over the rain table's own window (09:00 to 09:00), no new hour", prediction_set="lstm_wqa_to09",
                                           **newer("aorc_rain_to_midnight", "wqa_to09", k, crit, False))
        else:
            lv["lstm"] = lstm(key)
            for n in [n for n in comp[key]["baselines"] if n != "base_persistence"]:
                a, kind = against(key, n), at("baselines", f"models.{n}.model")
                lv["others"][n] = dict(label=KIND[kind], kind=kind, same_inputs=a["like_for_like"], drawn=k == LEAD and a["like_for_like"], **simple(n, key, k == LEAD and a["like_for_like"]), against_lstm=a)
        if key == "wq":                                     # the Transformer: its own experiment, beside an LSTM twin
            tr, twin, mc = compared("transformer"), compared("lstm"), R["model_comparison"]
            lv["others"]["cmp_transformer"] = dict(
                label=KIND["transformer"], kind="transformer", same_inputs="the same inputs, but from its own experiment: like-for-like only against its LSTM twin cmp_lstm", drawn=k == LEAD, **tr,
                twin=dict(prediction_set="cmp_lstm", **twin), nse_minus_twin=tr["nse"] - twin["nse"], onset_warned_points_minus_twin=100 * (tr["onset_warned"] - twin["onset_warned"]),
                nse_seeds_overlap_twin=not (above(tr["nse_seeds"], twin["nse_seeds"]) or above(twin["nse_seeds"], tr["nse_seeds"])),
                onset_warned_seeds_overlap_twin=not (above(tr["onset_warned_seeds"], twin["onset_warned_seeds"]) or above(twin["onset_warned_seeds"], tr["onset_warned_seeds"])),
                experiment=dict(file=where("model_comparison"), sequence_days=mc["settings"]["sequence_days"], max_updates=mc["settings"]["shared"]["updates"], seeds=mc["seeds"]),
                placed="at its own scores, beside the marks of set 3 (past weather + river gauge); not shifted or rescaled. It was trained beside an LSTM twin with the same settings "
                       "(prediction set cmp_lstm), not beside the LSTM drawn at set 3; the twin's scores are under 'twin' and in the footnote of the figure.")
        levels.append(lv)
    lstm_of = {lv["key"]: lv["lstm"] for lv in levels if lv["lstm"]}
    for s, x in lstm_of.items():                            # the verdict for onsets at the high-flow level itself, by the stored criteria
        if "onset_verdict" not in x:
            x["onset_verdict"] = verdict(at("rain_timing", f"models.{s}.test.summary.{L}.onset") if s == "wqn" else at("lstm_all", f"sets.{s}.test.summary.{L}.onset"), crit)
    control = dict(prediction_set="lstm_wqn_older", of_set="wqn", label="control: the same rain one day older", **timing("wqn_older", "control"))

    # ---- the real forecast and the perfect rain (18_forecast_input.py): the sets of the lower block that the upper block does not have ----
    forecast = {}
    for s in ("wqaF_sub", "wqaP"):
        lv = dict(key=s, label=next(x["label"] for x in FAR_SETS if x["key"] == s), prediction_set=f"lstm_{s}", lstm=None, others={})
        with_new_input(lv, "forecast_input", s, ("base_linreg_wqaF", "base_rf_wqaF") if s == "wqaF_sub" else ())
        forecast[s] = lv
    ladder2 = []                                            # the lower block: the same networks as above where they are, and the sets of the forecast experiment
    for S in FAR_SETS:
        src = next((x for x in levels if x["key"] == S["key"]), None) or forecast[S["key"]]
        ladder2.append(dict(key=S["key"], number=len(ladder2) + 1, label=S["label"], prediction_set=S["lstm"], lstm=src["lstm"],
                            others=src["others"] if S["key"] == "wqaF_sub" else {}, upper_bound=S["key"] in UPPER))
    lstm_of2 = {x["key"]: x["lstm"] for x in ladder2}

    # ---- the steps of the LSTM from one set to the next, and how far apart the kinds of model lie on the same inputs ----
    row, dec, src = R["rain_timing"]["closure"]["wqn"]["by_lead"][i], R["lstm_all"]["decomposition"]["steps"], f"closure.wqn.by_lead[{i}]"
    ca = f"closure.wqa.by_lead[{i}]"

    def step(name, a, b, what, also_nse=(), also_onset=(), drawn=False, of=None):
        of = of or lstm_of
        nse = D(f"step.{name}.nse", of[b]["nse"] - of[a]["nse"], f"{L}.lstm_{b}.nse minus {L}.lstm_{a}.nse", *also_nse, drawn=drawn, parts=(nm(L, b, "nse"), nm(L, a, "nse")))
        on = D(f"step.{name}.onset_warned", of[b]["onset_warned"] - of[a]["onset_warned"], f"{L}.lstm_{b}.onset_warned minus {L}.lstm_{a}.onset_warned", *also_onset, drawn=drawn,
               parts=(nm(L, b, "onset_warned"), nm(L, a, "onset_warned")))
        return dict(from_set=a, to_set=b, what=what, nse=nse, onset_warned_points=100 * on, nse_from=of[a]["nse"], nse_to=of[b]["nse"],
                    onset_warned_from=of[a]["onset_warned"], onset_warned_to=of[b]["onset_warned"])

    def stored(file, path, **fix):
        """The readings of a gain rule that a newer results file stores for one comparison (report section 4.2): the NSE and the onsets."""
        g = at(file, path)
        return dict(nse_reading=READ[g["nse"]["reading"]], nse_in_words=IN_WORDS[READ[g["nse"]["reading"]]], n_basins_better=g["nse"]["n_basins_higher"], n_basins=g["nse"]["n_basins"],
                    nse_wilcoxon_p=g["nse"]["wilcoxon_p"], onset_reading=READ[g["onset"]["reading"]], reading_source=where(file, path), **fix)

    g, f = dec["river_gauge_without_forecast"][L], dec["perfect_forecast_with_gauge"][L]
    ga = lambda ref: f"gain_rule.wqa.{ref}.{L}"
    steps = dict(
        river_gauge=dict(step("river_gauge", "w", "wq", "what the river gauge adds to past weather",
                              [("lstm_all", f"decomposition.steps.river_gauge_without_forecast.{L}.nse_median_gain")], [("lstm_all", f"decomposition.steps.river_gauge_without_forecast.{L}.onset_pod_gain")]),
                         nse_seed_by_seed=[g["nse_median_gain_seed_min"], g["nse_median_gain_seed_max"]], nse_clear_of_seed_noise=g["clear_of_seed_noise"], n_basins_better=g["n_basins_better"],
                         onset_warned_points_seed_by_seed=[100 * g["onset_pod_gain_seed_min"], 100 * g["onset_pod_gain_seed_max"]],
                         onset_warned_clear=g["onset_pod_gain_seed_min"] > 0 or g["onset_pod_gain_seed_max"] < 0),
        rain_up_to_evening=dict(step("rain_up_to_evening", "wq", "wqn", "what rain stamped up to the evening of the issue day adds (the time-stamp part of the gap)", [("rain_timing", f"{src}.gain")]),
                                n_basins_better=row["basins_better"], n_basins_better_than_control=row["basins_better_than_control"], median_basin_gain_nse=row["median_basin_gain"]),
        aorc_beyond_evening=dict(step("aorc_beyond_evening", "wqn", "wqa", "what the AORC rain up to midnight adds beyond the NLDAS rain up to the evening: the better product and the hours after 17:30",
                                      [("aorc_rain_to_midnight", f"{ga('against_nldas_rain')}.nse.gain")], [("aorc_rain_to_midnight", f"{ga('against_nldas_rain')}.onset.gain")]),
                                 **stored("aorc_rain_to_midnight", ga("against_nldas_rain"))),
        aorc_over_rain_table=dict(step("aorc_over_rain_table", "wq", "wqa", "what the AORC rain up to midnight adds to past weather + river gauge (the gain counted from the rain table alone)",
                                       [("aorc_rain_to_midnight", f"{ca}.gain"), ("aorc_rain_to_midnight", f"{ga('against_rain_table_alone')}.nse.gain")],
                                       [("aorc_rain_to_midnight", f"{ga('against_rain_table_alone')}.onset.gain")]),
                                  **stored("aorc_rain_to_midnight", ga("against_rain_table_alone"))),
        perfect_forecast_beyond_aorc=step("perfect_forecast_beyond_aorc", "wqa", "wqf", "what is left of the gap to the perfect weather forecast, an upper bound, beyond the AORC rain up to midnight"),
        perfect_forecast_beyond_that_rain=step("perfect_forecast_beyond_that_rain", "wqn", "wqf", "what is left of the gap to the perfect weather forecast, an upper bound, beyond the NLDAS rain up to the evening", drawn=False),
        perfect_forecast=dict(step("perfect_forecast", "wq", "wqf", "the whole gap: what a perfect weather forecast, an upper bound, would add to past weather + river gauge",
                                   [("rain_timing", f"{src}.gap"), ("lstm_all", f"decomposition.steps.perfect_forecast_with_gauge.{L}.nse_median_gain")],
                                   [("lstm_all", f"decomposition.steps.perfect_forecast_with_gauge.{L}.onset_pod_gain")], drawn=False),
                              nse_clear_of_seed_noise=f["clear_of_seed_noise"], n_basins_better=f["n_basins_better"], **dict(zip(("nse_reading", "nse_wilcoxon_p"), (lambda r_: (r_["reading"], r_["wilcoxon_p"]))(rule_reading("aorc_rain_to_midnight", "wqf", "wq", k))))))
    rain, gap, base = steps["rain_up_to_evening"], steps["perfect_forecast"], lstm_of["wq"]
    div = lambda a, b: a / b if b else None
    share_on = D("step.rain_up_to_evening.onset_share_of_gap", div(rain["onset_warned_points"], gap["onset_warned_points"]), "step.rain_up_to_evening / step.perfect_forecast, onsets warned",
                 *([("rain_timing", f"{src}.onset_share_of_gap_closed")] if gap["onset_warned_points"] else []), drawn=False)
    share_nse = D("step.rain_up_to_evening.nse_share_of_gap", div(rain["nse"], gap["nse"]), "step.rain_up_to_evening / step.perfect_forecast, NSE",
                  *([("rain_timing", f"{src}.share_of_gap_closed")] if gap["nse"] else []), drawn=False)
    seeds_on = above(lstm_of["wqn"]["onset_warned_seeds"], base["onset_warned_seeds"]) and above(lstm_of["wqn"]["onset_warned_seeds"], control["onset_warned_seeds"])
    seeds_nse = above(lstm_of["wqn"]["nse_seeds"], base["nse_seeds"])
    nse_reading = "clear" if seeds_nse and rain["n_basins_better"] > n_b / 2 else "mixed" if seeds_nse else "none"
    rain.update(n_basins=n_b, onset_share_of_gap_closed=share_on, onset_share_in_words=share_word(share_on), onset_share_of_gap_closed_beyond_control=row["onset_share_of_gap_closed_beyond_control"],
                onset_gain_clear=seeds_on, onset_rule="clear when every single seed with the extra rain is above every seed without it and every seed of the control",
                nse_share_of_gap_closed=share_nse, nse_share_of_gap_closed_beyond_control=row["share_of_gap_closed_beyond_control"], nse_reading=nse_reading,
                nse_in_words={"clear": "a clear gain", "mixed": "mixed", "none": "no clear difference"}[nse_reading],
                nse_rule="clear when every single seed with the extra rain is above every seed without it AND the NSE is higher in most basins; mixed when only the seeds say so")
    rr = rule_reading("aorc_rain_to_midnight", "wqn", "wq", k)         # the same step by the full rule of 4.2 (with the Wilcoxon p), from the per-basin NSE that 16_ stores
    rain.update(nse_reading_by_the_full_rule=rr["reading"], nse_wilcoxon_p=rr["wilcoxon_p"])
    CHECKS["compared"] += 1
    if rr["reading"] != nse_reading:
        CHECKS["disagreements"].append(f"{L}.step.rain_up_to_evening: the NSE reading is {nse_reading!r} by the seeds and the basins won, but {rr['reading']!r} by the full rule of section 4.2 with the Wilcoxon p")
    aorc = steps["aorc_over_rain_table"]
    aorc.update(onset_share_of_gap_closed=D("step.aorc_over_rain_table.onset_share_of_gap", div(aorc["onset_warned_points"], gap["onset_warned_points"]), "step.aorc_over_rain_table / step.perfect_forecast, onsets warned",
                                            ("aorc_rain_to_midnight", f"{ca}.onset_share_of_gap_closed"), drawn=False),
                nse_share_of_gap_closed=D("step.aorc_over_rain_table.nse_share_of_gap", div(aorc["nse"], gap["nse"]), "step.aorc_over_rain_table / step.perfect_forecast, NSE",
                                          ("aorc_rain_to_midnight", f"{ca}.share_of_gap_closed"), drawn=False))

    def spread(key):
        """How far apart the kinds of model lie on one set of inputs (the LSTM's seed average and the other models given the same inputs)."""
        lv = next(x for x in levels if x["key"] == key)
        v_all = {lv["prediction_set"]: lv["lstm"], **{n: o for n, o in lv["others"].items() if o["same_inputs"] is not False}}
        out = dict(set=key, models=list(v_all))
        for q in ("nse", "onset_warned"):
            v = {n: o[q] for n, o in v_all.items()}
            lo, hi = min(v, key=v.get), max(v, key=v.get)
            out[q] = dict(lowest=v[lo], lowest_model=lo, highest=v[hi], highest_model=hi,
                          spread=D(f"spread.{key}.{q}", v[hi] - v[lo], f"highest minus lowest of {', '.join(v)} ({L}, {q})", drawn=key == "wq" and k == LEAD, parts=(f"{L}.{hi}.{q}", f"{L}.{lo}.{q}")))
            if "cmp_transformer" in v:                      # the Transformer comes from another experiment: the spread of the other kinds alone
                own = [x for n, x in v.items() if n != "cmp_transformer"]
                out[q]["spread_without_transformer"] = max(own) - min(own)
        out["onset_warned"]["spread_points"] = 100 * out["onset_warned"]["spread"]
        return out

    steps["spread_between_model_kinds"] = {key: spread(key) for key in ("wq", "w", "wqa")}

    # ---- the steps of the lower block: from the rain table to the rain to midnight, to the real forecast, to the two upper bounds ----
    gf = lambda a, b: f"gain_rule.{a}_against_{b}.{L}"
    d2 = k in FAR

    def far_step(name, a, b, what, file, stored=None):
        """The step from set a to set b of the lower block: sizes, and the reading of report section 4.2 (computed by rule_reading, compared with the stored one where there is one)."""
        st = step(f"far.{name}", a, b, what, drawn=False, of=lstm_of2)
        r = rule_reading(file, b, a, k, stored=stored)
        st.update(nse_reading=r["reading"], nse_in_words=IN_WORDS[r["reading"]], n_basins_better=r["n_basins_better"], n_basins=r["n_basins"], nse_wilcoxon_p=r["wilcoxon_p"],
                  reading_source=r["source"], **(dict(stored_reading_source=where(*stored)) if stored else {}))
        return st

    steps2 = dict(
        rain_to_midnight=far_step("rain_to_midnight", "wq", "wqa", "what the AORC rain up to midnight adds to the rain table + river gauge", "forecast_input", ("aorc_rain_to_midnight", f"{ga('against_rain_table_alone')}.nse.reading")),
        forecast_over_rain_table=far_step("forecast_over_rain_table", "wq", "wqaF_sub", f"what the AORC rain up to midnight and the real weather forecast of the next three days (put in place of the AORC rain of those days, {AFTER}, that the network was trained on) add to the rain table + river gauge",
                                          "forecast_input"),
        forecast_over_rain_to_midnight=far_step("forecast_over_rain_to_midnight", "wqa", "wqaF_sub", "what the real weather forecast adds to the rain up to midnight", "forecast_input", ("forecast_input", f"{gf('wqaF_sub', 'wqa')}.nse.reading")),
        perfect_rain_over_rain_table=far_step("perfect_rain_over_rain_table", "wq", "wqaP", "what the rain up to midnight and the perfect rain of the three days, an upper bound, add to the rain table + river gauge", "forecast_input"),
        perfect_rain_over_rain_to_midnight=far_step("perfect_rain_over_rain_to_midnight", "wqa", "wqaP", "what the perfect rain of the three days adds to the rain up to midnight", "forecast_input", ("forecast_input", f"{gf('wqaP', 'wqa')}.nse.reading")),
        perfect_weather_over_rain_table=far_step("perfect_weather_over_rain_table", "wq", "wqf", "what the perfect weather of the three days, an upper bound, adds to the rain table + river gauge", "forecast_input"))
    cf = f"closure.wqaF_sub__base_wqa__ceiling_wqaP.by_lead[{i}]"
    fo = steps2["forecast_over_rain_to_midnight"]
    fo.update(onset_share_of_gap_closed=D("step.far.forecast_over_rain_to_midnight.onset_share_of_gap", div(fo["onset_warned_points"], 100 * (lstm_of2["wqaP"]["onset_warned"] - lstm_of2["wqa"]["onset_warned"])),
                                          "step.far.forecast_over_rain_to_midnight / (perfect rain minus rain to midnight), onsets warned", ("forecast_input", f"{cf}.onset_share_of_gap_closed"), drawn=False),
              nse_share_of_gap_closed=D("step.far.forecast_over_rain_to_midnight.nse_share_of_gap", div(fo["nse"], lstm_of2["wqaP"]["nse"] - lstm_of2["wqa"]["nse"]),
                                        "step.far.forecast_over_rain_to_midnight / (perfect rain minus rain to midnight), NSE", ("forecast_input", f"{cf}.share_of_gap_closed"), drawn=False),
              gap_is="the gap between the rain to midnight and the perfect rain of the three days (lstm_wqaP), the rain-only upper bound")

    # ---- the count of onsets at this lead time: the same in every file ----
    agree(f"{L}.n_onsets", n_onsets, *([("flood_analysis", "events.n_onsets")] if k == 1 else []), origin=where("lstm_all", f"sets.wq.test.summary.{L}.onset, hits + misses"))
    for file, path in (("baselines", f"models.base_persistence.test.summary.{L}.onset"), ("rain_timing", f"models.wqn.test.summary.{L}.onset"),
                       ("model_comparison", f"models.transformer.seed_average.test.summary.{L}.onset"), ("aorc_rain_to_midnight", f"summary_test.sets.wqa.{L}.onset"),
                       ("forecast_input", f"summary_test.sets.wqaF_sub.{L}.onset"), ("forecast_input", f"summary_test.sets.wqaP.{L}.onset")):
        CHECKS["compared"] += 1
        if hm(at(file, path)) != n_onsets:
            CHECKS["disagreements"].append(f"{L}.n_onsets: {n_onsets} in {where('lstm_all')}, but {hm(at(file, path))} in {where(file, path)}")
    SOURCES[f"{L}.n_onsets"] = dict(file=where("lstm_all"), key=f"sets.wq.test.summary.{L}.onset (hits + misses)", value=n_onsets, in_figure=k == LEAD or d2,
                                    checked_against=([where("flood_analysis", "events.n_onsets")] if k == 1 else []) + ["the onset tables of baselines.json, rain_timing.json, model_comparison.json, aorc_rain_to_midnight.json and forecast_input.json"])
    if k == LEAD:
        SOURCES[f"{L}.n_basins_better_with_rain_up_to_evening"] = dict(file=where("rain_timing"), key=f"{src}.basins_better", value=row["basins_better"], in_figure=True, checked_against=[])
    return dict(lead=k, levels=levels, steps=steps, control=control, n_onsets=n_onsets, ladder2=ladder2, steps2=steps2, forecast=forecast)


# ---- does the picture show what its title says? ------------------------------------------------------------------------------
def finding(K):
    """The rule behind the title, the same as the closing slide's (build_slides._key, 'ordered'): on the same inputs (past weather + river gauge)
    the kinds of model lie closer together than EVERY step of the LSTM to the right of them in the upper block (rain up to the evening; the AORC rain
    up to midnight beyond it; what is left of the gap to the perfect weather forecast), in onsets warned and in NSE. The title, the subtitle and the
    headline state the finding only when it holds; otherwise they only say what is drawn. Also kept, for the notes: the same comparison with past
    weather only, and the spread of the three kinds that read the AORC rain (not the rule; the bracket of the figure is at past weather + river gauge)."""
    S = K["steps"]
    right = (S["rain_up_to_evening"], S["aorc_beyond_evening"], S["perfect_forecast_beyond_aorc"])
    least = {q: None if any(gone(st[q]) for st in right) else min(st[q] for st in right) for q in ("onset_warned_points", "nse")}

    def closer(key):
        sp = S["spread_between_model_kinds"][key]
        spread = dict(onset_warned_points=sp["onset_warned"]["spread_points"], nse=sp["nse"]["spread"])
        return spread, {q: not gone(v) and not gone(least[q]) and v < least[q] for q, v in spread.items()}
    spread, ok = closer("wq")
    spread_w, ok_w = closer("w")
    spread_a, ok_a = closer("wqa")
    return dict(holds=all(ok.values()),
                rule="holds when, on the same inputs (past weather + river gauge), highest minus lowest of the kinds of model is smaller than every step of the LSTM to the "
                     "right of them (rain up to the evening; the AORC rain up to midnight beyond it; what is left of the gap to the perfect weather forecast, an upper bound), in onsets warned AND in NSE",
                spread_between_model_kinds=spread, smallest_step_to_the_right=least, smaller_than_every_step=ok,
                with_past_weather_only=dict(spread_between_model_kinds=spread_w, smaller_than_every_step=ok_w, note="not part of the rule: the bracket of the figure is at past weather + river gauge"),
                with_aorc_rain_to_midnight=dict(spread_between_model_kinds=spread_a, smaller_than_every_step=ok_a,
                                                note="not part of the rule: the LSTM and the two simple models given the AORC rain up to midnight; the simple models warn of more onsets with more false warnings (a different operating point)"))


def verdicts_everywhere(K):
    """The verdict for onsets of every set the figure draws, by the stored criteria, at the fixed warning level (computed) and at the level chosen on the validation years (as the
    results files store it), at every lead time drawn. 'practice' are the sets that could be had in practice (the upper bounds are left out); the headline says NOT YET for
    them only if it is so everywhere."""
    out = dict(criteria=K["criteria"]["GO"], lead_times=[], practice=dict(sets=[], all_not_yet=True, exceptions=[]), upper_bounds=[])

    NAMES = {"w": "past weather only", "wq": "past weather + river gauge", "wqn": "rain up to the evening", "wqa": "AORC rain up to midnight", "wqaF_sub": "real forecast (GEFS)",
             "wqaP": "perfect rain of the three days", "wqf": "perfect weather forecast"}

    def keep(where_, label, lead, plain, chosen, upper, name=None):
        e = dict(set=label, name=name or label, lead=lead, at_the_fixed_level=plain, at_the_level_chosen_on_validation=chosen)
        if upper:
            out["upper_bounds"].append(e)
            return
        out["practice"]["sets"].append(e)
        if plain != "NOT YET" or (chosen is not None and chosen != "NOT YET"):
            out["practice"]["all_not_yet"] = False
            out["practice"]["exceptions"].append(e)

    for k in (LEAD,) + FAR:
        L = lk(k)
        src = dict(levels=K["levels"]) if k == LEAD else dict(levels=K["far"][L]["levels"])
        out["lead_times"].append(k)
        for lv in src["levels"]:
            if lv["lstm"]:
                keep(L, f"lstm_{lv['key']}", k, lv["lstm"]["onset_verdict"], lv["lstm"].get("onset_verdict_at_chosen_level"), lv["key"] in UPPER, name=f"LSTM, {NAMES.get(lv['key'], lv['key'])}")
            for n, o in lv["others"].items():
                if o.get("drawn") and "onset_verdict" in o:
                    keep(L, n, k, o["onset_verdict"], None, False, name=f"{o['label']}, {NAMES.get(lv['key'], 'river gauge only')}")
    return out


# ---- main (data part) --------------------------------------------------------------------------------------------------------
def clean(x):
    """NaN -> null, so that the file is plain JSON."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return None if isinstance(x, float) and x != x else x


def read_results():
    """Read the results files into R: the ones that must be there (a smoke test that lacks one says so in one line and ends without an error) and the optional ones."""
    for name, script in UPSTREAM.items():
        f = F.RES_OUT / f"{name}.json"
        if not f.exists():
            if F.QUICK:                                     # a smoke test of this script alone: nothing to draw from, and that is not an error
                print(f"11_key_figure.py: {where(name)} is missing (run FLOOD_QUICK=1 python code/{script} first) - no key figure drawn")
                sys.exit(0)
            sys.exit(f"11_key_figure.py: {where(name)} is missing - run code/{script} first")
        R[name] = json.loads(f.read_text())
        assert F.QUICK or not R[name].get("quick_mode"), f"{where(name)} is a smoke-test file"
    for name, script in OPTIONAL.items():
        f = F.RES_OUT / f"{name}.json"
        R[name] = json.loads(f.read_text()) if f.exists() else None
        if R[name] is None:
            print(f"11_key_figure.py: {where(name)} is missing (code/{script} writes it): the figure is drawn without what it holds", file=sys.stderr)
        else:
            assert F.QUICK or not R[name].get("quick_mode"), f"{where(name)} is a smoke-test file"


def build():
    """Everything the figure and the words need, read from the results files: K (not yet worded)."""
    c = R["flood_analysis"]["verdict"]["criteria"]
    CRIT.update(GO=c["GO"], CAUTION=c["CAUTION"], NOT_YET=c["NOT YET"], definitions=c["definitions"],
                go_min_warned=float(re.search(r"pod\s*>=\s*([\d.]+)", c["GO"]).group(1)), go_max_false=float(re.search(r"far\s*<=\s*([\d.]+)", c["GO"]).group(1)),
                caution_min_csi=float(re.search(r"csi\s*>=\s*([\d.]+)", c["CAUTION"]).group(1)),
                drawn="onset panels: the band from the GO share of onsets warned upwards. The false-warning half of GO is not drawn; onset_false and onset_verdict of every LSTM are in 'levels'.")
    crit = CRIT
    SOURCES["criteria.go_min_warned"] = dict(file=where("flood_analysis"), key="verdict.criteria.GO", value=crit["go_min_warned"], in_figure=True, checked_against=[where(f, "verdicts.criteria.GO") for f in ("aorc_rain_to_midnight", "forecast_input") if has(f, "verdicts.criteria.GO")])
    for f in ("aorc_rain_to_midnight", "forecast_input"):   # the newer files state the criteria they applied: the same text
        agree("criteria.GO", c["GO"], (f, "verdicts.criteria.GO"), origin=where("flood_analysis", "verdict.criteria.GO"))
        agree("criteria.CAUTION", c["CAUTION"], (f, "verdicts.criteria.CAUTION"), origin=where("flood_analysis", "verdict.criteria.CAUTION"))
    # not drawn, for one sentence of the notes: the warning level at which the verdict table of the report counts the warnings of the three LSTMs of 06_flood_analysis.py
    # (chosen on the validation years), and the share of onsets warned there. The figure counts at the high-flow level itself.
    crit["level_of_the_verdict_table"] = {}
    for s in ("w", "wq", "wqf"):
        try:
            crit["level_of_the_verdict_table"][s] = dict(
                times_the_high_flow_level=float(at("flood_analysis", f"verdict.services.lstm_{s}.{lk(LEAD)}.trigger")),
                onset_warned=take(f"{lk(LEAD)}.lstm_{s}.onset_warned_at_the_level_of_the_verdict_table", "flood_analysis", f"verdict.services.lstm_{s}.{lk(LEAD)}.onset.pod", drawn=False))
        except (KeyError, IndexError, TypeError, ValueError):
            pass                                            # an older results file without that table: the sentence is left out
    # the test years and the basins: the same in every file
    split, n_b = take("test_period", "lstm_all", "split.test"), len(R["lstm_all"]["basins"])
    for file in ("baselines", "flood_analysis", "model_comparison"):
        agree("test_period", split[0], (file, "split.test[0]"), origin=where("lstm_all", "split.test"))
        agree("test_period", split[1], (file, "split.test[1]"), origin=where("lstm_all", "split.test"))
        SOURCES["test_period"]["checked_against"].append(where(file, "split.test"))
    for file in ("aorc_rain_to_midnight", "forecast_input"):
        agree("test_period", split[0], (file, "split.test[0]") if file == "aorc_rain_to_midnight" else (file, "split_main.test[0]"), origin=where("lstm_all", "split.test"))
        agree("test_period", split[1], (file, "split.test[1]") if file == "aorc_rain_to_midnight" else (file, "split_main.test[1]"), origin=where("lstm_all", "split.test"))
        SOURCES["test_period"]["checked_against"].append(where(file, "split.test" if file == "aorc_rain_to_midnight" else "split_main.test"))
    counts = {"baselines": len(R["baselines"]["thresholds"]), "rain_timing": len(R["rain_timing"]["stamps"]["per_basin"]), "flood_analysis": len(R["flood_analysis"]["basins"]),
              "model_comparison": len(R["model_comparison"]["thresholds"])}
    CHECKS["compared"] += len(counts)
    CHECKS["disagreements"] += [f"n_basins: {n_b} in {where('lstm_all')}, but {n} in {where(file)}" for file, n in counts.items() if n != n_b]
    SOURCES["n_basins"] = dict(file=where("lstm_all"), key="basins (number of entries)", value=n_b, in_figure=True, checked_against=[where(file) for file in counts])
    return split, n_b, {lk(k): collect(k, crit, n_b) for k in F.LEADS}


# ---- the words inside the drawing (labels of the rows, titles, legend, the notes beside the marks), every number read from K ---------------------
def drawn_words(K):
    """The lettering of the drawing itself: the names of the sets, the titles of the panels, the legend and the short notes beside marks. Numbers are read from K."""
    S, crit, n_b = K["steps"], K["criteria"], K["n_basins"]
    n_on = K["n_onsets"]
    seed_counts = sorted({len(x["lstm"]["seeds"]) for x in K["levels"] + [y for k in FAR for y in K["far"][lk(k)]["levels"]] if x.get("lstm") and x["lstm"].get("seeds")}) or [3]      # 3 for the sets of 04_lstm_all.py, 5 where seeds 3-4 were added (30 October 2026)
    sp = S["spread_between_model_kinds"]["wq"]
    kinds = len(sp["models"])
    notes = {"upper.wqa": AFTER}                            # a note under the name of a set
    mixed = {}                                              # 'mixed' beside an NSE value whose gain over past weather + river gauge is a mixed result (report section 4.2)
    if S["rain_up_to_evening"]["nse_reading"] == "mixed":
        mixed[f"upper.wqn.{lk(LEAD)}"] = True
    for k in FAR:
        if K["far"][lk(k)]["steps"]["forecast_over_rain_table"]["nse_reading"] == "mixed":
            mixed[f"lower.wqaF_sub.{lk(k)}"] = True
    return dict(
        panel_titles=[("Onsets warned", f"share of {n_on[lk(LEAD)]:,} onsets"), ("Median NSE", f"of {n_b} basins")],
        groups=dict(upper=days(LEAD), lower="2 and 3 days ahead", **{f"l{k}": f"{k} days ahead" for k in FAR}, **{f"n{k}": f"of {n_on[lk(k)]:,} onsets" for k in FAR}),
        rows=dict(upper={"q": "river gauge only", "w": "past weather only", "wq": "past weather + river gauge", "wqn": "+ rain up to the evening",
                         "wqa": "+ rain up to midnight, radar- and gauge-anchored (AORC)", "wqf": "+ perfect weather forecast"},
                  lower={"wq": "rain table + river gauge", "wqa": "+ rain up to midnight (AORC)", "wqaF_sub": "+ real forecast (GEFS)",
                         "wqaP": "perfect rain (upper bound)", "wqf": "perfect weather (upper bound)"}),
        rows_talk=dict(upper={**{"q": "river gauge only", "w": "past weather only", "wq": "past weather + river gauge", "wqn": "+ rain up to the evening", "wqa": "+ rain up to midnight (AORC)", "wqf": "+ perfect weather forecast"}},
                       lower={"wq": "rain table + river gauge", "wqa": "+ rain up to midnight (AORC)", "wqaF_sub": "+ real forecast (GEFS)", "wqaP": "perfect rain (upper bound)", "wqf": "perfect weather (upper bound)"}),
        row_notes=notes, mixed_rows=mixed, mixed="mixed",
        spread=[f"same inputs, {word(kinds)} kinds\nof model: {sp['onset_warned']['spread_points']:.0f} points apart", f"same inputs,\n{word(kinds)} kinds of model:\n{n2(sp['nse']['spread'])} NSE apart"],
        control="control", persistence="'tomorrow = today'",
        realtime="real time (Stage IV radar)",
        go=f"GO asks for at least {pc(crit['go_min_warned'])} warned", go_short=f"GO: at least {pc(crit['go_min_warned'])}", go_tiny=f"GO \u2265 {pc(crit['go_min_warned'])}", go_word="GO",
        legend=dict(lstm=f"LSTM, average of {' or '.join(word(n) for n in seed_counts)} seeds", seed="single seeds (their range)", linreg=KIND["linreg"], rf=KIND["rf"], transformer=KIND["transformer"] + " *",
                    bound=f"hatched: {BOUND}", ref="the level of past weather + river gauge"))


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# ---- figure (drawn from the saved JSON) ------------------------------------------------------------------------------------
# The ladder runs down the page: one row per set of information, its name on the left, then the share of onsets warned and the median NSE. Each LSTM is a dot at its
# score; a thick bar runs from the level of past weather + river gauge (a dashed line) to the dot, so that its length is what the extra information gives.
# Upper block: 1 day ahead, six sets. Lower block: 2 and 3 days ahead, five sets, each lead time in its own pair of narrow panels.
FS = 8.0                                                   # lettering of the report renderings, in points (drawn 6.7 inches wide, printed 1.08 times larger: 8.7 pt)
ON_RANGE, NSE_RANGE = (0.0, 0.80), (0.0, 1.0)              # what the horizontal axes show
SIZE = dict(lstm=7.6, linreg=5.2, rf=4.6, transformer=7.2, persistence=5.2, control=4.8, realtime=5.0)   # marker sizes in points (report renderings)
LANE = dict(linreg=-1.25, rf=1.25, transformer=2.25, control=1.75, realtime=2.75)                           # where the small marks sit in a row, in lanes below (+) or above (-) the LSTM
RENDER = {"report": dict(file="key_figure.png", use="the figure with its own title, subtitle and footnote", width=6.7, lettering=1.0, title=True, footnote=True, drop=()),
          "plain": dict(file="key_figure_plain.png", use="page 1 of the report: the same drawing without title, subtitle and footnote (the caption carries the words) and without the rows "
                                                         "'river gauge only' and 'past weather only' (neither is in the headline; page 1 holds one page)", width=6.7,
                        lettering=1.0, title=False, footnote=False, drop=(("upper", "q"), ("upper", "w")), go_label_down_to_the_real_time_texts=True),
          "talk": dict(file="key_figure_talk.png", use="the closing slide: larger lettering, without the rows that are not in the headline, no title, subtitle or footnote", width=8.0,
                       lettering=1.3, title=False, footnote=False, drop=(("upper", "q"), ("upper", "w"), ("lower", "wqaP")), marks=("wq",), control=False, seeds=False, labw=0.29, short_labels=True,
                       go_word_on_top=True)}
DRAWN_VALUES = {}                                          # by rendering: (name in 'sources', value) of every number the drawing prints or plots; checked against the files at the end


def extent(fig, text, size, **kw):
    """(width, height) in inches of a text drawn with the figure's own fonts."""
    t = fig.text(0, 0, text, fontsize=size, **kw)
    b = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def wrap(fig, text, width, size, **kw):
    """Break a text into lines no wider than `width` inches at font size `size`; explicit line breaks stay."""
    out = []
    for para in text.split("\n"):
        cur = ""
        for w_ in para.split():
            if cur and extent(fig, f"{cur} {w_}", size, **kw)[0] > width:
                out.append(cur)
                cur = w_
            else:
                cur = f"{cur} {w_}".strip()
        out.append(cur)
    return out


class Board:
    """The drawing surface: one axes over the whole figure in points (x to the right, y DOWN), and the boxes already taken, so that a text finds a free place."""

    def __init__(self, fig, ax, fs):
        self.fig, self.ax, self.fs, self.taken, self.misplaced, self.left_out = fig, ax, fs, [], [], []

    def take(self, x0, y0, x1, y1):
        self.taken.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    def free(self, box, pad=0.3):
        x0, y0, x1, y1 = box
        return not any(x0 < b[2] + pad and x1 > b[0] - pad and y0 < b[3] + pad and y1 > b[1] - pad for b in self.taken)

    def put(self, text, spots, size=None, bold=False, color=INK2, zorder=8, linespacing=1.12, clear=False, optional=False, inside=None, tail=None):
        """Draw `text` (lines joined by \\n) at the first of `spots` = [(x, y, ha)] where its box is free and inside the x-range `inside`, vertically centred on y.
        Where none is free the first spot is used (and the text is noted in .misplaced), or the text is left out if `optional`. clear: on a background of the surface colour."""
        size = size or self.fs
        lines = text.split("\n")
        ext = [extent(self.fig, l, size, fontweight="bold" if bold else "normal") for l in lines]
        w = max(e[0] for e in ext) * 72
        w_tail = (extent(self.fig, tail, size)[0] * 72 + 3) if tail else 0.0                # a lighter word after the value, on the same line
        w += w_tail
        h = size * 0.78 + (len(lines) - 1) * size * linespacing * 1.0                     # the height of the ink: a line is about 0.78 of its font size
        for n, (x, y, ha) in enumerate(list(spots) + [spots[0]]):
            x0 = x if ha == "left" else x - w if ha == "right" else x - w / 2
            box = (x0, y - h / 2, x0 + w, y + h / 2)
            fits = inside is None or (box[0] >= inside[0] and box[2] <= inside[1])
            last = n == len(spots)
            if last and optional:
                self.left_out.append(text.replace("\n", " "))
                return None
            if last or (fits and self.free(box)):
                if last:
                    self.misplaced.append(text.replace("\n", " "))
                self.ax.text(x, y, text, fontsize=size, fontweight="bold" if bold else "normal", color=color, ha=ha, va="center", linespacing=linespacing, zorder=zorder,
                             bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.3) if clear else None)
                if tail:
                    self.ax.text(x + (w - w_tail) + 3 if ha == "left" else x + 3, y, tail, fontsize=size, color=INK2, ha="left", va="center", zorder=zorder)
                self.take(*box)
                return box


def draw(K, name):
    """One rendering (a key of RENDER) of the drawing: the ladder of the sets of information, onsets warned beside median NSE; 1 day ahead above, 2 and 3 days ahead below."""
    spec, txt, crit = RENDER[name], K["figure"], K["criteria"]
    words = txt["drawn_words"]
    up_levels = [x for x in K["levels"] if ("upper", x["key"]) not in spec["drop"]]
    lo_levels = {k: [x for x in K["far"][lk(k)]["levels"] if ("lower", x["key"]) not in spec["drop"]] for k in FAR}
    s, W = spec["lettering"], spec["width"]
    fs, Wp = FS * s, spec["width"] * 72
    drawn = DRAWN_VALUES[name] = []
    L1 = lk(LEAD)
    reg = lambda nme, v: (drawn.append((nme, v)), v)[1]       # a number the drawing uses: kept by its name in 'sources' for the check at the end of the build
    matplotlib.rcParams["hatch.linewidth"] = 0.6 * s
    fig = plt.figure(figsize=(W, 4.0))
    wid = lambda text, size=fs, **kw: extent(fig, text, size, **kw)[0] * 72           # width of a text in points

    # ---- columns (points from the left edge) ----
    ML, MR, GAPL, BGAP, MG = 4 * s, 12 * s, 5 * s, 19 * s, 22 * s
    LABW = spec.get("labw", 0.255) * Wp
    for _ in range(6):                                      # the margins hold the last value labels with their 'mixed' (they run past their panel); the panel widths depend on the margins
        rem = Wp - ML - LABW - GAPL - BGAP - MR
        BW = (0.545 * rem, 0.455 * rem)
        pw = (BW[1] - MG) / 2                               # width of one narrow NSE panel
        w_lab = wid("0.45", fontweight="bold") + 3 * s + wid("mixed") + 3.8 * s + 3 * s       # from the dot's centre: radius, gap, the value, a gap, 'mixed'
        MR = max(12 * s, w_lab - (1 - 0.46) * pw + 3 * s)
        MG = max(22 * s, w_lab - (1 - 0.53) * pw + 1.5 * s + 2 * s)
    x_on = ML + LABW + GAPL
    x_ns = x_on + BW[0] + BGAP
    half = lambda b: (b - MG) / 2
    PAN = {("up", "on"): (x_on, x_on + BW[0], *ON_RANGE), ("up", "ns"): (x_ns, x_ns + BW[1], *NSE_RANGE)}
    for k, off in zip(FAR, (0, 1)):
        PAN[(k, "on")] = (x_on + off * (half(BW[0]) + MG), x_on + off * (half(BW[0]) + MG) + half(BW[0]), *ON_RANGE)
        PAN[(k, "ns")] = (x_ns + off * (half(BW[1]) + MG), x_ns + off * (half(BW[1]) + MG) + half(BW[1]), *NSE_RANGE)
    X = lambda p, v: PAN[p][0] + (v - PAN[p][2]) / (PAN[p][3] - PAN[p][2]) * (PAN[p][1] - PAN[p][0])

    # ---- the rows: label, lanes used, height ----
    LH, PADY = 5.4 * s, 4.0 * s
    note_of = lambda block, key: words["row_notes"].get(f"{block}.{key}", "")

    small_here = lambda x: [o for o in x["others"].values() if o["drawn"] and (spec.get("marks") is None or x["key"] in spec["marks"])]     # the small marks that this rendering draws at a set

    def lanes_needed(x, extra=()):
        used = [0.0] + [LANE[o["kind"]] for o in small_here(x) if o["kind"] in LANE] + list(extra)
        return min(used), max(used)

    def make_row(block, x, extra=()):
        lab = wrap(fig, words["rows_talk" if spec.get("short_labels") else "rows"][block][x["key"]], (LABW - 2) / 72, fs)
        nt = note_of(block, x["key"])
        a, b = lanes_needed(x, extra)
        n_lines = len(lab) + len(wrap(fig, nt, (LABW - 2) / 72, fs)) * bool(nt)
        return dict(block=block, x=x, lanes=(a, b), label=lab, note=wrap(fig, nt, (LABW - 2) / 72, fs) if nt else [], h=max((b - a) * LH + 2 * PADY, n_lines * fs * 1.1 + 3 * s, 2 * PADY + 2 * s))
    EXTRA = {"wq": (3.85,), "wqn": (LANE["control"],) if spec.get("control", True) else (), "wqa": (LANE["realtime"],)}
    up = [make_row("upper", x, EXTRA.get(x["key"], ())) for x in up_levels]
    dn = [make_row("lower", x) for x in lo_levels[FAR[0]]]
    dn_by_lead = {k: {x["key"]: x for x in lo_levels[k]} for k in FAR}

    # ---- the vertical stack (points from the top) ----
    y = 3 * s + (23 * s if spec["title"] else 0.0)
    y_titles = y + 6 * s
    y += 12 * s
    y_g1 = y
    for r in up:
        r["y0"] = y
        y += r["h"]
    y_g1e = y
    y_tick1 = y + 5.5 * s
    y += 10.5 * s + 2 * s
    y_head2 = y + 9 * s
    y += 18 * s
    y_g2 = y
    for r in dn:
        r["y0"] = y
        y += r["h"]
    y_g2e = y
    y_tick2 = y + 5.5 * s
    y += 10.5 * s

    # ---- the legend: entries in as many rows as the width needs ----
    entries = [(k_, words["legend"][k_]) for k_ in ("lstm", "seed", "linreg", "rf", "transformer", "bound", "ref") if spec.get("seeds", True) or k_ != "seed"]
    hw, avail = 12 * s, Wp - ML - MR
    legend_rows, cur, curw = [], [], 0.0
    for e in entries:
        w_ = hw + 3 * s + wid(e[1]) + 9 * s
        if cur and curw + w_ > avail:
            legend_rows.append(cur)
            cur, curw = [], 0.0
        cur.append((e, curw))
        curw += w_
    legend_rows.append(cur)
    y_leg = y + 2 * s
    y += 2 * s + len(legend_rows) * 10.8 * s
    note_lines, nsize = [], fs
    if spec["footnote"]:
        note_lines = wrap(fig, txt["footnote"], (Wp - ML - MR) / 72, nsize)
        while len(note_lines) > 5 and nsize > 7.9:
            nsize = round(nsize - 0.1, 1)
            note_lines = wrap(fig, txt["footnote"], (Wp - ML - MR) / 72, nsize)
        y_note = y + 3 * s
        y += 3 * s + len(note_lines) * nsize * 1.3
    H = (y + 3 * s) / 72
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, Wp)
    ax.set_ylim(H * 72, 0)
    ax.axis("off")
    B = Board(fig, ax, fs)
    cy = lambda r, lane=0.0: r["y0"] + r["h"] / 2 + (lane - (r["lanes"][0] + r["lanes"][1]) / 2) * LH         # y of a lane in a row (the LSTM's lane is 0)
    ftxt = lambda x, yy, t, **kw: ax.text(x, yy, t, **{**dict(fontsize=fs, color=INK2, ha="left", va="center", zorder=8), **kw})

    # ---- backgrounds: the GO band, the gridlines, the hatching of the upper bounds, the hairlines between the rows ----
    blocks = [(["up"], y_g1, y_g1e, up, y_tick1), (list(FAR), y_g2, y_g2e, dn, y_tick2)]
    for panels, ya, yb, rs, ytk in blocks:
        for p in panels:
            x0, x1 = X((p, "on"), crit["go_min_warned"]), X((p, "on"), ON_RANGE[1])
            ax.add_patch(Rectangle((x0, ya), x1 - x0, yb - ya, facecolor=AQUA, alpha=.14, lw=0, zorder=0.3))
            for q in ("on", "ns"):
                ticks = ([0.0, 0.2, 0.4, 0.6, 0.8] if q == "on" else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) if p == "up" else ([0.0, 0.4] if q == "on" else [0.0, 0.5, 1.0])
                for t in ticks:
                    ax.plot([X((p, q), t)] * 2, [ya, yb], color=GRID, lw=0.8, zorder=0.5, solid_capstyle="butt")
                    if p == "up" and q == "on" and t == ticks[-1]:
                        continue                            # its label would touch the 0.0 of the next panel
                    ftxt(X((p, q), t), ytk, f"{round(100 * t)}%" if q == "on" else (f"{t:.1f}" if p == "up" else f"{t:g}"), color=MUTED, ha="center")
        for r in rs:
            if r["x"]["key"] in UPPER:
                for p in panels:
                    for q in ("on", "ns"):
                        ax.add_patch(Rectangle((PAN[(p, q)][0], r["y0"]), PAN[(p, q)][1] - PAN[(p, q)][0], r["h"], facecolor="none", edgecolor=AXIS, hatch="////", lw=0, zorder=0.4))
    for r in up[1:] + dn[1:]:
        ax.plot([ML, x_ns + BW[1]], [r["y0"]] * 2, color=GRID, lw=0.6, zorder=0.45, solid_capstyle="butt")
    go_label = None                                     # the words drawn on the GO band of the upper block (None: the band carries none), recorded with the rendering
    if up:                                              # the words of the GO band, turned to run down the band, above the first row whose texts reach into it
        xb = (X(("up", "on"), crit["go_min_warned"]) + X(("up", "on"), ON_RANGE[1])) / 2
        stop = next((r["y0"] for r in up if r["x"]["key"] == "wqa"), y_g1e)
        tries = [(t, stop) for t in (words["go"], words["go_short"])]
        row_a = next((r for r in up if r["x"]["key"] == "wqa"), None)
        if spec.get("go_label_down_to_the_real_time_texts") and row_a is not None:
            stop2 = cy(row_a, LANE["realtime"]) - 5 * s
            tries += [(t, stop2) for t in (words["go"], words["go_short"], words["go_tiny"])]
        for t, bottom in tries:
            if wid(t) <= bottom - y_g1 - 4 * s:
                ax.text(xb, (y_g1 + bottom) / 2, t, fontsize=fs, color=INK2, rotation=90, ha="center", va="center", zorder=1)
                go_label = t
                break
        if go_label is None and spec.get("go_word_on_top"):   # no room to run the words down the band: the one word GO, upright, at the top of the band, if the band is wider than the word
            t = words["go_word"]
            go_w, go_h = (v * 72 for v in extent(fig, t, fs, fontweight="bold"))
            if go_w <= X(("up", "on"), ON_RANGE[1]) - X(("up", "on"), crit["go_min_warned"]) - 5 * s:
                yc = y_g1 + 3.5 * s + go_h / 2
                ax.text(xb, yc, t, fontsize=fs, fontweight="bold", color=INK2, ha="center", va="center", zorder=1)
                B.take(xb - go_w / 2, yc - go_h / 2, xb + go_w / 2, yc + go_h / 2)      # no other text takes this place
                go_label = t

    # ---- titles ----
    for k in (LEAD,) + FAR:
        reg(f"{lk(k)}.n_onsets", K["n_onsets"][lk(k)])
    reg("n_basins", K["n_basins"])
    reg("criteria.go_min_warned", crit["go_min_warned"])
    if spec["footnote"]:
        reg("test_period", K["test_period"])
        reg(f"{L1}.n_basins_better_with_rain_up_to_evening", K["steps"]["rain_up_to_evening"]["n_basins_better"])
        twin = next(x for x in K["levels"] if x["key"] == "wq")["others"]["cmp_transformer"]["twin"]
        reg(f"{L1}.cmp_lstm.onset_warned", twin["onset_warned"])
        reg(f"{L1}.cmp_lstm.nse", twin["nse"])
    if spec["title"]:
        fig.text(0.012 * 6.7 / W, 1 - 3 * s / (H * 72), txt["title"], fontsize=10.2, fontweight="bold", color=INK, ha="left", va="top")
        fig.text(0.012 * 6.7 / W, 1 - 15.5 * s / (H * 72), txt["subtitle"], fontsize=FS, color=INK2, ha="left", va="top")

    def two_part(x, yy, bold_text, rest, size=fs, ha="left"):
        """A title in bold followed by a lighter second part on the same line."""
        ax.text(x, yy, bold_text, fontsize=size, fontweight="bold", color=INK, ha="left", va="center", zorder=8)
        ax.text(x + wid(bold_text, size, fontweight="bold") + 3 * s, yy, rest, fontsize=size, color=INK2, ha="left", va="center", zorder=8)
    ftxt(ML, y_titles, words["groups"]["upper"], fontsize=9 * s, fontweight="bold", color=INK)
    two_part(x_on, y_titles, words["panel_titles"][0][0], words["panel_titles"][0][1])
    two_part(x_ns, y_titles, words["panel_titles"][1][0], words["panel_titles"][1][1])
    ftxt(ML, y_head2, words["groups"]["lower"], fontsize=9 * s, fontweight="bold", color=INK)
    for k in FAR:
        for q in ("on", "ns"):
            xc = (PAN[(k, q)][0] + PAN[(k, q)][1]) / 2
            ftxt(xc, y_head2 - 4.5 * s, words["groups"][f"l{k}"], fontweight="bold", color=INK, ha="center")
            if q == "on":
                ftxt(xc, y_head2 + 5 * s, words["groups"][f"n{k}"], color=INK2, ha="center")

    # ---- the names of the sets (left), muted where a note follows ----
    for r in up + dn:
        lines = r["label"] + r["note"]
        y0 = r["y0"] + r["h"] / 2 - (len(lines) - 1) * fs * 1.1 / 2
        for n, l in enumerate(lines):
            ftxt(ML, y0 + n * fs * 1.1, l, color=INK2)
        B.take(ML, r["y0"], ML + LABW, r["y0"] + r["h"])

    # ---- the marks. Texts are asked for here and placed afterwards, in order of importance, where nothing else is ----
    asks = []

    def ask(prio, text, spots, **kw):
        asks.append((prio, text, spots, kw))

    def small_mark(yy, x, kind):
        ax.plot(x, yy, ls="none", marker=MARK[kind], ms=SIZE[kind] * s, color=F.MODEL_COL[kind], markeredgecolor=SURFACE, markeredgewidth=0.6 * s, zorder=5)
        rad = SIZE[kind] * s / 2 + 0.6
        B.take(x - rad, yy - rad, x + rad, yy + rad)

    def lstm_marks(p, q, r, x, k, ref=None, mid=None, light=False):
        """The LSTM of one row in one panel: its single seeds, the bar from the level of past weather + river gauge (`ref`, in the units of the panel) to the dot (in two tones where
        `mid`, the level with the rain up to midnight, lies between), the dot and its value; returns the x of the dot."""
        L, key = lk(k), x["key"]
        qq = "nse" if q == "ns" else "onset_warned"
        v = reg(nm(L, key, qq), x["lstm"][qq])
        seeds = [reg(nm(L, key, f"{qq}.seed{sd}"), sv) for sd, sv in zip(x["lstm"]["seeds"], x["lstm"][f"{qq}_seeds"])]
        yy, xv = cy(r), X(p, v)
        bound = key in UPPER
        if ref is not None:
            xr = X(p, ref)
            cuts = [xr] + ([X(p, mid)] if mid is not None else []) + [xv]
            for j, (a_, b_) in enumerate(zip(cuts, cuts[1:])):
                ax.plot([a_, b_], [yy, yy], color=SEQ[2] if (light or (mid is not None and j == 0)) else BLUE, lw=2.2 * s, ls=(0, (2.2, 1.4)) if bound else "-", solid_capstyle="butt", dash_capstyle="butt", zorder=3)
            B.take(min(xr, xv), yy - 1.3 * s, max(xr, xv), yy + 1.3 * s)
        sx = [X(p, t) for t in seeds] if spec.get("seeds", True) else [xv]
        if spec.get("seeds", True):
            ax.plot([min(sx), max(sx)], [yy, yy], color=INK, lw=0.9 * s, solid_capstyle="butt", zorder=5.4)
            for t in sx:
                ax.plot([t, t], [yy - 2.3 * s, yy + 2.3 * s], color=INK, lw=1.0 * s, solid_capstyle="butt", zorder=5.5)
        ax.plot(xv, yy, ls="none", marker="o", ms=SIZE["lstm"] * s, markerfacecolor=SEQ[0] if bound else BLUE, markeredgecolor=BLUE if bound else SURFACE, markeredgewidth=(1.6 if bound else 1.1) * s, zorder=6)
        rad = SIZE["lstm"] * s / 2 + 1.1 * s
        B.take(xv - rad, yy - rad, xv + rad, yy + rad)
        B.take(min(sx) - 0.8, yy - 2.6 * s, max(sx) + 0.8, yy + 2.6 * s)
        right = max(xv + rad, max(sx) + 0.8)
        tail = words["mixed"] if (q == "ns" and words["mixed_rows"].get(f"{'upper' if k == LEAD else 'lower'}.{key}.{L}")) else None
        last_column = p[0] in ("up", FAR[-1])                # the right edge of a label: into the margin, into the gap before the next panel of the same measure, or before the other block
        right_edge = PAN[p][1] + ((MR - 1 if p[1] == "ns" else BGAP - 2) if last_column else MG - 1.5)
        spots = [(right + 3.0 * s, yy, "left")] + ([] if tail else [(xv - rad - 3.0 * s, yy, "right"), (xv, yy - 7.0 * s, "center"), (xv, yy + 7.0 * s, "center")])
        ask(1, (pc if q == "on" else n2)(v), spots, bold=True, color=INK, inside=(PAN[p][0] - 3, right_edge), tail=tail)
        return xv

    def others_marks(p, q, r, x, k):
        """The other kinds of model at a row (where they read the same inputs as the LSTM), as small marks in their own lanes; returns their x."""
        qq = "nse" if q == "ns" else "onset_warned"
        out = []
        for n, o in x["others"].items():
            if o["drawn"] and (spec.get("marks") is None or x["key"] in spec["marks"]):
                v = reg(f"{lk(k)}.{n}.{qq}", o[qq])
                small_mark(cy(r, LANE[o["kind"]]), X(p, v), o["kind"])
                out.append(X(p, v))
        return out

    def ref_line(p, r_ref, value, y_end):
        xr = X(p, value)
        ax.plot([xr, xr], [cy(r_ref), y_end], color=SEQ[2], lw=1.0 * s, ls=(0, (3, 2)), zorder=2, solid_capstyle="butt")

    # -- the upper block --
    L1 = lk(LEAD)
    ref_val = {}
    for r in up:
        x, key = r["x"], r["x"]["key"]
        for q in ("on", "ns"):
            pq, qq = ("up", q), "nse" if q == "ns" else "onset_warned"
            if key == "q":                                  # 'tomorrow = today': the only mark of its row
                o = x["others"]["base_persistence"]
                v = reg(f"{L1}.base_persistence.{qq}", o[qq])
                xv = X(pq, v)
                ax.plot(xv, cy(r), ls="none", marker="D", ms=SIZE["persistence"] * s, color=F.MODEL_COL["persistence"], markeredgecolor=SURFACE, markeredgewidth=0.6 * s, zorder=5)
                B.take(xv - 3.8 * s, cy(r) - 3.8 * s, xv + 3.8 * s, cy(r) + 3.8 * s)
                ask(1, (pc if q == "on" else n2)(v), [(xv + 5.5 * s, cy(r), "left"), (xv, cy(r) - 7.5 * s, "center")], bold=True, color=INK, inside=(PAN[pq][0] - 3, PAN[pq][1] + 7 * s))
                if q == "on":
                    ask(3, words["persistence"], [(xv + 5.5 * s + wid(pc(v), fontweight="bold") + 4 * s, cy(r), "left")])
                continue
            others_marks(pq, q, r, x, LEAD)
            if key == "wq":
                ref_val[q] = x["lstm"][qq]
            lstm_marks(pq, q, r, x, LEAD, ref=None if key in ("w", "wq") else ref_val[q])
            if key == "wq":                                  # the dashed level that the bars below start from
                ref_line(pq, r, ref_val[q], y_g1e)
            if key == "wqn" and spec.get("control", True):  # the control: the same rain, one day older
                v = reg(nm(L1, "wqn_older", qq), K["control"][qq])
                xc, yc = X(pq, v), cy(r, LANE["control"])
                ax.plot(xc, yc, ls="none", marker="o", ms=SIZE["control"] * s, markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.2 * s, zorder=5)
                B.take(xc - 3.4 * s, yc - 3.4 * s, xc + 3.4 * s, yc + 3.4 * s)
                if q == "on":
                    ask(3, words["control"], [(xc + 5 * s, yc, "left")])
            if key == "wqa" and q == "on" and x.get("real_time"):                        # the real-time check: Stage IV radar in place of AORC on the issue day
                rt = {v_["key"]: v_ for v_ in x["real_time"]["variants"]}
                pts_ = []
                for vk, size_ in (("stage4", 1.0), ("stage4_no_last_hour", 0.85)):
                    v = reg(f"{L1}.real_time.{vk}.onset_warned", rt[vk]["onset_warned"])
                    xd, yd = X(pq, v), cy(r, LANE["realtime"])
                    ax.plot(xd, yd, ls="none", marker="D", ms=SIZE["realtime"] * s * size_, markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.2 * s, zorder=5)
                    B.take(xd - 3.4 * s, yd - 3.4 * s, xd + 3.4 * s, yd + 3.4 * s)
                    pts_.append((xd, yd, v))
                (xa, ya_, va), (xb, yb_, vb) = pts_
                ask(2, pc(va), [(xa + 4.5 * s, ya_, "left")], color=INK)
                ask(2, pc(vb), [(xb - 4.5 * s, yb_, "right")], color=INK)
                ask(3, words["realtime"], [(xb - 4.5 * s - wid(pc(vb)) - 3.5 * s, yb_, "right"), (xa + 4.5 * s + wid(pc(va)) + 3.5 * s, ya_, "left")])
            if key == "wq":                                  # the bracket: how far apart the kinds of model lie on the same inputs
                sp = K["steps"]["spread_between_model_kinds"]["wq"][("onset_warned" if q == "on" else "nse")]
                lo_x, hi_x = X(pq, sp["lowest"]), X(pq, sp["highest"])         # the ends are the values of two of the marks of this row, drawn and registered above
                reg(f"{L1}.spread.wq.{qq}", sp["spread"])
                yb = cy(r, 3.85)
                ax.plot([lo_x, lo_x, hi_x, hi_x], [yb - 2.4 * s, yb, yb, yb - 2.4 * s], color=INK, lw=0.9 * s, solid_capstyle="butt", zorder=7)
                B.take(lo_x, yb - 2.4 * s, hi_x, yb + 0.5)
                spots = ([(hi_x + 8 * s, cy(r, 2.75), "left"), (hi_x + 8 * s, cy(r, 3.0), "left"), (hi_x + 30 * s, cy(r, 1.3), "left")] if q == "on" else
                         [(lo_x - 7 * s, cy(r, 2.1), "right"), (lo_x - 7 * s, cy(r, 2.6), "right")])
                ask(2, words["spread"][0 if q == "on" else 1], spots, color=INK)
    # -- the lower block --
    for k in FAR:
        L = lk(k)
        ref, mid = {}, {}
        for r in dn:
            key = r["x"]["key"]
            x = dn_by_lead[k][key]
            for q in ("on", "ns"):
                pq, qq = (k, q), "nse" if q == "ns" else "onset_warned"
                others_marks(pq, q, r, x, k)
                if key == "wq":
                    ref[q] = x["lstm"][qq]
                if key == "wqa":
                    mid[q] = x["lstm"][qq]
                lstm_marks(pq, q, r, x, k, ref=None if key == "wq" else ref[q], mid=mid.get(q) if key in ("wqaF_sub", "wqaP") else None, light=key == "wqa")
                if key == "wq":
                    ref_line(pq, r, ref[q], y_g2e)

    # ---- the texts ----
    for prio, text, spots, kw in sorted(asks, key=lambda a: a[0]):
        B.put(text, spots, **kw)

    # ---- the legend ----
    for ri, row in enumerate(legend_rows):
        yy = y_leg + ri * 10.8 * s + 5 * s
        for (kind, text), x0 in row:
            xx = ML + x0 + hw / 2
            if kind == "lstm":
                ax.plot(xx, yy, ls="none", marker="o", ms=SIZE["lstm"] * s, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.1 * s, zorder=6)
            elif kind == "seed":
                ax.plot([xx - 4.5 * s, xx + 4.5 * s], [yy, yy], color=INK, lw=0.9 * s, zorder=6)
                for t in (-4.5, 4.5):
                    ax.plot([xx + t * s] * 2, [yy - 2.3 * s, yy + 2.3 * s], color=INK, lw=1.0 * s, zorder=6)
            elif kind == "bound":
                ax.add_patch(Rectangle((xx - hw / 2, yy - 4 * s), hw, 8 * s, facecolor="none", edgecolor=AXIS, hatch="////", lw=0.6 * s, zorder=6))
            elif kind == "ref":
                ax.plot([xx, xx], [yy - 4.5 * s, yy + 4.5 * s], color=SEQ[2], lw=1.0 * s, ls=(0, (3, 2)), zorder=6)
            else:
                ax.plot(xx, yy, ls="none", marker=MARK[kind], ms=SIZE[kind] * s, color=F.MODEL_COL[kind], markeredgecolor=SURFACE, markeredgewidth=0.6 * s, zorder=6)
            ftxt(ML + x0 + hw + 3 * s, yy, text)
    if spec["footnote"]:
        for n, l in enumerate(note_lines):
            ax.text(ML, y_note + n * nsize * 1.3 + nsize * 0.65, l, fontsize=nsize, color=INK2, ha="left", va="center")
    out = OUT_FIG / spec["file"]
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.png")     # written beside it, then moved into place: a document built meanwhile never reads half a picture
    fig.savefig(tmp, dpi=220)
    with Image.open(tmp) as im:
        pixels = list(im.size)                              # of the file as written (the rounding of the figure size to whole pixels is the library's)
    tmp.replace(out)
    plt.close(fig)
    return dict(file=("figures/quick/" if F.QUICK else "figures/") + spec["file"], use=spec["use"], width_in=W, height_in=round(H, 2), dpi=220, pixels=pixels, smallest_lettering_pt=round(min(nsize, fs), 2),
                title_and_subtitle=spec["title"], footnote_lines=len(note_lines), legend_rows=len(legend_rows),
                left_out=dict(rows=[f"{b}.{k}" for b, k in spec["drop"]], small_marks_of_the_other_kinds=("all but at past weather + river gauge" if spec.get("marks") else "none"),
                              control_circle=not spec.get("control", True), single_seed_lines=not spec.get("seeds", True)),
                go_band_label=go_label, texts_without_a_free_place=B.misplaced, texts_left_out=B.left_out, numbers_drawn=len(drawn))


# ---- words -------------------------------------------------------------------------------------------------------------------
TEXT_VALUES = []                                            # names in 'sources' of the numbers that the lettering of the figure (titles, footnote, notes beside marks) prints


def ref(name):
    """The value of a number kept in 'sources', for use in a text of the figure: the name is logged so that the check at the end of the build can follow it."""
    TEXT_VALUES.append(name)
    return SOURCES[name]["value"]


def extras():
    """Facts of the other experiments of 7 and 8 October 2026 that the figure does not draw but its notes name (each read from its file and kept in 'sources', not in the figure):
    the ERA5 rain to midnight (E1), the 63 training rivers (E3), the forecast retrained as an input on its own split (E2b)."""
    out = {}
    if R.get("rain_to_midnight"):
        L = lk(LEAD)
        out["era5"] = dict(onset_warned=take("extra.era5.onset_warned", "rain_to_midnight", f"summary_test.sets.wqh.{L}.onset.pod", ("aorc_rain_to_midnight", f"summary_test.sets.wqh.{L}.onset.pod"), drawn=False),
                           nse=take("extra.era5.nse", "rain_to_midnight", f"summary_test.sets.wqh.{L}.nse_median", ("aorc_rain_to_midnight", f"summary_test.sets.wqh.{L}.nse_median"), drawn=False))
    if R.get("more_basins"):
        L = lk(LEAD)
        out["rivers63"] = dict(nse=take("extra.rivers63.nse", "more_basins", f"summary_test.sets.lstm63_wq.{L}.nse_median", drawn=False),
                               onset_warned=take("extra.rivers63.onset_warned", "more_basins", f"summary_test.sets.lstm63_wq.{L}.onset.pod", drawn=False),
                               onset_false=take("extra.rivers63.onset_false", "more_basins", f"summary_test.sets.lstm63_wq.{L}.onset.far", drawn=False),
                               nse_reading=at("more_basins", f"gain_rule.lstm63_wq_against_lstm_wq.{L}.nse.reading"), n_rivers=len(at("more_basins", "design.rivers.sixty_three")),
                               nse_base=take("extra.rivers63.nse_base", "more_basins", f"summary_test.sets.lstm_wq.{L}.nse_median", ("lstm_all", f"sets.wq.test.summary.{L}.nse_median"), drawn=False),
                               onset_base=take("extra.rivers63.onset_base", "more_basins", f"summary_test.sets.lstm_wq.{L}.onset.pod", ("lstm_all", f"sets.wq.test.summary.{L}.onset.pod"), drawn=False))
    sat = {}
    ladder = at("forecast_input", "ladder.sat.rungs")
    for s in ("wqa_s", "wqaF_s", "wqaP_s"):
        j = next(j for j, r in enumerate(ladder) if r["set"] == f"lstm_{s}")
        sat[s] = {lk(k): dict(onset_warned=take(f"extra.e2b.{s}.{lk(k)}.onset_warned", "forecast_input", f"ladder.sat.rungs[{j}].by_lead.{lk(k)}.onset_warned",
                                                ("forecast_input", f"summary_test.sets.{s}.{lk(k)}.onset.pod"), drawn=False),
                              n_onsets=at("forecast_input", f"ladder.sat.rungs[{j}].by_lead.{lk(k)}.n_onsets")) for k in F.LEADS}
    out["e2b"] = dict(sets=sat, test_years=at("forecast_input", "ladder.sat.test_years"), split=at("forecast_input", "split_satellite_era"))
    return out


def sentences(K):
    """Headline, notes and the lettering of the figure, every number read from K (the drawn lead times)."""
    lv = {x["key"]: x for x in K["levels"]}
    far = {k: {x["key"]: x for x in K["far"][lk(k)]["levels"]} for k in FAR}
    fst = {k: K["far"][lk(k)]["steps"] for k in FAR}
    S, ctrl, crit, n_b = K["steps"], K["control"], K["criteria"], K["n_basins"]
    w, wq, wqn, wqa, wqf = (lv[s]["lstm"] for s in ("w", "wq", "wqn", "wqa", "wqf"))
    per, tr = lv["q"]["others"]["base_persistence"], lv["wq"]["others"]["cmp_transformer"]
    sp, rain, gap, aorc = S["spread_between_model_kinds"], S["rain_up_to_evening"], S["perfect_forecast"], S["aorc_over_rain_table"]
    rt, ctrl_a, X = lv["wqa"].get("real_time"), lv["wqa"]["control_not_drawn"], K["extras"]
    y0, y1 = K["test_years"]
    kinds = len(sp["wq"]["models"])
    n1, n2_, n3 = ref(f"{lk(LEAD)}.n_onsets"), ref("l2.n_onsets"), ref("l3.n_onsets")
    ref("n_basins"); ref("test_period")
    fl = {k: far[k]["wqaF_sub"]["lstm"] for k in FAR}
    pw = {k: far[k]["wqf"]["lstm"] for k in FAR}
    pr = {k: far[k]["wqaP"]["lstm"] for k in FAR}
    base2 = {k: far[k]["wq"]["lstm"] for k in FAR}
    mid2 = {k: far[k]["wqa"]["lstm"] for k in FAR}
    best = lambda key: next((o for o in lv[key]["others"].values() if o.get("against_lstm", {}).get("best_simple_model_by_validation")), None)
    listed = lambda key, q, fmt: ", ".join(f"{fmt(o[q])} ({o['label']})" for o in lv[key]["others"].values() if o["same_inputs"] is not False and o["kind"] != "transformer")
    rng = lambda v, fmt: f"{fmt(min(v))} to {fmt(max(v))}"
    mixed = {"mixed": f"its NSE result is mixed: the median rises from {n2(wq['nse'])} to {n2(wqn['nse'])}, but the NSE is higher in only {rain['n_basins_better']} of {n_b} basins",
             "clear": f"its NSE rises from {n2(wq['nse'])} to {n2(wqn['nse'])}, in every seed and in {rain['n_basins_better']} of {n_b} basins",
             "none": f"in NSE there is no clear difference ({n2(wq['nse'])} against {n2(wqn['nse'])}; the single seeds overlap)"}[rain["nse_reading"]]
    pt = lambda st: pts(st["onset_warned_points"] / 100)
    plural = lambda st: "s" * (abs(round(st["onset_warned_points"])) != 1)
    b_wq, b_w = best("wq"), best("w")
    verdicts = K["verdicts"]
    all_ns = verdicts["practice"]["all_not_yet"]
    # the perfect weather forecast is the upper bound of THIS LSTM: a simple model given it may warn of more onsets (not drawn: it does not read the same days of weather)
    more5 = [o for o in lv["wqf"]["others"].values() if o["kind"] != "transformer" and not gone(o["onset_warned"]) and o["onset_warned"] > wqf["onset_warned"]]
    bound_of = ("" if not more5 else
                " The bound is that of this LSTM, not of every kind of model: given a perfect weather forecast, "
                + " and ".join(f"the {o['label']} warns of {pc(o['onset_warned'])} of the onsets, with {pc(o['onset_false'])} of its onset warnings false" for o in more5)
                + f" (the LSTM: {pc(wqf['onset_false'])} false; GO allows at most {pc(crit['go_max_false'])}). The simple models are not drawn there, because they do not read the same days of "
                f"weather as the LSTM {days(LEAD)}.")
    chosen = {s: (crit.get("level_of_the_verdict_table") or {}).get(s) for s in ("w", "wq", "wqf")}
    same_level = [s for s, c in chosen.items() if c and abs(c["times_the_high_flow_level"] - 1) < 1e-9]
    other_level = [(s, c) for s, c in chosen.items() if c and abs(c["times_the_high_flow_level"] - 1) >= 1e-9]
    name5 = {"w": "past weather only", "wq": "past weather + river gauge", "wqf": "the perfect weather forecast (the upper bound)"}
    at_level = ("" if not all(chosen.values()) else
                f" The verdict table of the report (section 5) counts warnings at the level chosen on the validation years instead: {days(LEAD)} that is the high-flow level itself for "
                + (" and ".join(name5[s] for s in same_level) if same_level else "none of the three LSTMs")
                + "".join(f", and {c['times_the_high_flow_level']:.2f} times that level for {name5[s]}, where the LSTM warns of {pc(c['onset_warned'])} of the onsets (in the figure: "
                          f"{pc(lv[s]['lstm']['onset_warned'])})" for s, c in other_level) + ".")
    bound_hits = pr[3]
    short_of_go = bound_hits["onset_false"] - crit["go_max_false"]

    # the finding is stated (title, subtitle, headline, caption) only when the numbers show it: the rule is in finding()
    fd = K["finding"]
    claim = fd["holds"]
    real_time = (f" ({pc(rt['variants'][0]['onset_warned'])} with the real-time radar product Stage IV in its place)" if rt else "")
    headline = (("SMOKE TEST, not a result: " if F.QUICK else "")
                + ("Among the models tried, the limit is the information in the inputs, not the network: " if claim else "")
                + f"{days(LEAD)}, on the same inputs (past weather + river gauge) {word(kinds)} kinds of model "
                f"lie within {n2(sp['wq']['nse']['spread'])} NSE and {sp['wq']['onset_warned']['spread_points']:.0f} percentage points of onsets warned of each other, while rain up to the evening "
                f"of the issue day lifts the LSTM's onsets warned from {pc(wq['onset_warned'])} to {pc(wqn['onset_warned'])} ({rain['onset_share_in_words']} of the gap; in NSE the result is "
                f"{rain['nse_in_words']}) and rain up to midnight from a radar- and gauge-anchored analysis (AORC, {AFTER}) lifts them from {pc(wq['onset_warned'])} to {pc(wqa['onset_warned'])}{real_time}; "
                f"{FAR[0]} and {FAR[1]} days ahead a real weather forecast (GEFS) lifts them from {pc(base2[2]['onset_warned'])} and {pc(base2[3]['onset_warned'])} to {pc(fl[2]['onset_warned'])} and "
                f"{pc(fl[3]['onset_warned'])}"
                + (" (in NSE the gain is mixed)" if all(fst[k]["forecast_over_rain_table"]["nse_reading"] == "mixed" for k in FAR) else "")
                + f"; a perfect weather forecast, {BOUND}, would give {pc(wqf['onset_warned'])} {days(LEAD)} and {pc(pw[2]['onset_warned'])} and {pc(pw[3]['onset_warned'])} {FAR[0]} and {FAR[1]} days ahead; "
                + ("for onsets the verdict is still NOT YET for every set that could be had in practice."
                   if all_ns else "for onsets the verdict is NOT YET for the sets that could be had in practice except " + _and(f"{e['name']} {days(e['lead'])}" for e in verdicts["practice"]["exceptions"]) + "."))
    # what the title rests on, in numbers - and where the margin is small or the rule does not hold, so that no document says more than the numbers do
    p0 = lambda x: "no value" if gone(x) else f"{round(x) + 0:d}"                                   # percentage points, without and with a sign
    s0 = lambda x: "no value" if gone(x) else f"{round(x) + 0:+d}"
    on_sp, least, wide = sp["wq"]["onset_warned"], fd["smallest_step_to_the_right"], fd["with_past_weather_only"]["smaller_than_every_step"]
    gains = (S["rain_up_to_evening"], S["aorc_over_rain_table"], S["perfect_forecast"])
    both = (f"({n2(sp['wq']['nse']['spread'])} NSE; {p0(on_sp['spread_points'])} percentage points of onsets warned) than every gain of the LSTM over past weather + river gauge "
            f"({', '.join(sg(g_['nse']) for g_ in gains[:2])} and {sg(gains[2]['nse'])} NSE; {', '.join(pt(g_) for g_ in gains[:2])} and {pt(gains[2])} points)")
    tight = claim and on_sp["spread_points"] > 0.5 * rain["onset_warned_points"]                    # claim: neither of the two is missing
    apart = dict(nse=f"in NSE ({n2(sp['w']['nse']['spread'])} apart, against {sg(least['nse'])} for the smallest of those gains)",
                 onset_warned_points=f"in onsets warned ({p0(sp['w']['onset_warned']['spread_points'])} percentage points apart, against {s0(least['onset_warned_points'])} for the smallest of those gains)")
    not_w = [apart[q] for q, ok in wide.items() if not ok]
    rests = ((f"What the title rests on: on the same inputs (past weather + river gauge) the {word(kinds)} kinds of model lie closer together {both}."
              + (f" In onsets warned the margin against the rain up to the evening is small ({p0(on_sp['spread_points'])} against {pt(rain)}): the kinds of model range from "
                 f"{pc(on_sp['lowest'])} to {pc(on_sp['highest'])} (the note 'Kinds of model on the same inputs' lists them). What they share is that none of them warns of more than "
                 f"{pc(on_sp['highest'])} of the onsets, where the rain up to the evening gives {pc(wqn['onset_warned'])}, the AORC rain up to midnight {pc(wqa['onset_warned'])} and the upper bound {pc(wqf['onset_warned'])}." if tight else "")
              + (f" With past weather only the same does not hold {' and '.join(not_w)}: the note 'With past weather only' says how that difference is read." if not_w else ""))
             if claim else
             f"The title states no finding: on the same inputs (past weather + river gauge) the {word(kinds)} kinds of model do NOT lie closer together {both}.")
    a_sp = sp["wqa"]
    a_models = [o for o in lv["wqa"]["others"].values()]
    e2b = X["e2b"]["sets"]
    areas = [v["area_km2"] for v in at("forecast_input", "per_basin.basins").values()]
    n_sat = {k: e2b["wqaF_s"][lk(k)]["n_onsets"] for k in F.LEADS}
    def more_fewer(a, b):
        return "more" if a > b else "fewer" if a < b else "as many"
    add = lambda o, key, base: 100 * (o["onset_warned"] - lv[base]["others"][key]["onset_warned"])
    new_inputs_note = (
        f"Kinds of model given the new inputs (the small marks). With the AORC rain up to midnight, {days(LEAD)}: "
        + "; ".join(f"{o['label']} warns of {pc(o['onset_warned'])} of the onsets ({pc(o['onset_false'])} of its warnings false) and scores a median NSE of {n2(o['nse'])}" for o in a_models)
        + f"; the LSTM {pc(wqa['onset_warned'])} ({pc(wqa['onset_false'])} false) and {n2(wqa['nse'])}. The new input adds {add(a_models[0], 'base_linreg_wq', 'wq'):+.0f} points of onsets warned to ridge regression, "
          f"{add(a_models[1], 'base_rf_wq', 'wq'):+.0f} to the random forest and {pt(aorc)} to the LSTM, so the gain in onsets does not need the network; the simple models sit at another operating point "
          f"(" + ", ".join(f"{o['label']} {more_fewer(o['onset_warned'], wqa['onset_warned'])} onsets and {more_fewer(o['onset_false'], wqa['onset_false'])} false warnings than the LSTM" for o in a_models)
        + f"); by the NSE and the rule of section 4.2 " + ", ".join(f"{o['against_lstm']['in_words']} against {o['label']}" for o in a_models) + f". The three lie {n2(a_sp['nse']['spread'])} NSE and "
          f"{a_sp['onset_warned']['spread_points']:.0f} percentage points of onsets apart here, against {n2(sp['wq']['nse']['spread'])} and {sp['wq']['onset_warned']['spread_points']:.0f} with past weather + river gauge. "
        + f"With the GEFS forecast, {FAR[0]} and {FAR[1]} days ahead: "
        + "; ".join(f"{o['label']} warns of {pc(o['onset_warned'])} and {pc(far[3]['wqaF_sub']['others'][n]['onset_warned'])} of the onsets ({pc(o['onset_false'])} and {pc(far[3]['wqaF_sub']['others'][n]['onset_false'])} of its warnings false) "
                    f"with a median NSE of {n2(o['nse'])} and {n2(far[3]['wqaF_sub']['others'][n]['nse'])}" for n, o in far[2]["wqaF_sub"]["others"].items())
        + f"; the LSTM {pc(fl[2]['onset_warned'])} and {pc(fl[3]['onset_warned'])} ({pc(fl[2]['onset_false'])} and {pc(fl[3]['onset_false'])} false), {n2(fl[2]['nse'])} and {n2(fl[3]['nse'])}. "
          "Against the random forest the LSTM shows "
        + " and ".join(f"{far[k]['wqaF_sub']['others']['base_rf_wqaF']['against_lstm']['in_words']} {days(k)}" for k in FAR)
        + ("; both simple models warn of more onsets than the LSTM, with more of their warnings false (another operating point): the gain from the forecast does not need the network either." if
           all(far[k]["wqaF_sub"]["others"][n]["onset_warned"] > fl[k]["onset_warned"] and far[k]["wqaF_sub"]["others"][n]["onset_false"] > fl[k]["onset_false"] for k in FAR for n in far[2]["wqaF_sub"]["others"])
           else "; the simple models and the LSTM sit at different operating points."))
    left_out_names = {key: next(x["label"] for x in K["levels"] if x["key"] == key) for _, key in RENDER["plain"]["drop"]}
    plain_drop_words = (" and without the rows " + " and ".join(f"'{v.split(' (')[0]}'" for v in left_out_names.values())) if left_out_names else ""
    aorc_gain_words = (f"both {IN_WORDS[aorc['nse_reading']]}" if aorc["onset_reading"] == aorc["nse_reading"] else f"{IN_WORDS[aorc['onset_reading']]} in onsets and {aorc['nse_in_words']} in NSE")
    notes = [
        f"Test years {y0}-{y1}, {n_b} basins. The upper block of the figure is {days(LEAD)}, the lower block {word(FAR[0])} and {word(FAR[1])} days ahead. Onsets warned: the share of the onsets of the test years "
        f"({n1:,}, {n2_:,} and {n3:,} at 1, 2 and 3 days) for which a warning was given, the {n_b} basins counted together, at the high-flow level itself. NSE: the median of the {n_b} basins "
        f"(1 is a perfect forecast).{at_level}",
        f"An onset is a high-flow day forecast while the river was still at or below the high-flow level (the level exceeded on {100 * (1 - F.HIGH_Q):.0f}% of the days of the training years): "
        "the warning that matters, because the river is not yet high when it is issued.",
        f"The perfect weather forecast (hatched) is {BOUND}: the LSTM reads the weather recorded for the days being forecast. A real weather forecast has errors, so a real system lands below it: "
        f"the real forecast of the lower block (GEFS) is one measurement of where, {pc(fl[2]['onset_warned'])} and {pc(fl[3]['onset_warned'])} of the onsets {FAR[0]} and {FAR[1]} days ahead against "
        f"{pc(pw[2]['onset_warned'])} and {pc(pw[3]['onset_warned'])} for the perfect weather. Even with the perfect weather the LSTM warns of {pc(wqf['onset_warned'])} of the onsets {days(LEAD)}"
        + (f", below the {pc(crit['go_min_warned'])} that GO asks for." if wqf["onset_warned"] < crit["go_min_warned"] else ".") + bound_of
        + f" The lower block has a second upper bound, the perfect rain of the three days (the AORC rain of the three days being forecast, {AFTER}, nothing else, beside the rain up to midnight): "
          f"{pc(pr[2]['onset_warned'])} and {pc(pr[3]['onset_warned'])} of the onsets.",
        f"AORC (the set '+ rain up to midnight, radar- and gauge-anchored') is {AFTER}: hourly rain at 1 km anchored on gauges and radar, corrected about 10 days after the hour, and before 2002, which "
        f"are all the training years, disaggregated daily totals (results/aorc_rain_to_midnight.json). It says what well-measured rain up to midnight is worth, not what a service "
        "reads at midnight."
        + (f" The real-time check (hollow diamonds; results/stage4_realtime.json): the three saved networks of that set, given Stage IV radar and gauge rain (the product a service at midnight would read) "
           f"in place of AORC on the issue day of each of the {rt['onsets_tested']} onsets, warn of {rt['variants'][0]['onsets_warned']} ({pc(rt['variants'][0]['onset_warned'])}) against {rt['aorc_onsets_warned']} "
           f"({pc(wqa['onset_warned'])}) with AORC and {rt['rain_table_alone_onsets_warned']} ({pc(wq['onset_warned'])}) with the rain table alone: {pc(rt['variants'][0]['share_of_the_aorc_effect_surviving'])} of the AORC "
           f"effect survives (95% interval {pc(rt['interval_share_of_the_aorc_effect_surviving'][0])} to {pc(rt['interval_share_of_the_aorc_effect_surviving'][1])}); with the last hour of the issue day left "
           f"out, as the delay of a real-time product would, {rt['variants'][1]['onsets_warned']} ({pc(rt['variants'][1]['onset_warned'])}). The networks were trained on AORC, not on Stage IV, and the false "
           f"warnings were not tested on days without an onset." if rt else " The real-time check (results/stage4_realtime.json) was not found, so the figure is drawn without it."),
        f"The control of the rain-timing experiment (the hollow circle) is the same rain one day older: still a second rain product, but with no rain later than the rain table's own. It warns of {pc(ctrl['onset_warned'])} of the onsets "
        f"(median NSE {n2(ctrl['nse'])}), against {pc(wq['onset_warned'])} ({n2(wq['nse'])}) without the extra rain and {pc(wqn['onset_warned'])} ({n2(wqn['nse'])}) with the rain up to the evening: "
        + ("in onsets warned the gain is the rain of the issue day, not the second rain product (every single seed with that rain is above every seed without it and every seed of the control)."
           if rain["onset_gain_clear"] else "the single seeds overlap, so in onsets warned the gain is not clear."),
        f"The control of the AORC set (not drawn) is the same AORC rain over the rain table's own window, 09:00 to 09:00, with no new hour: it warns of {pc(ctrl_a['onset_warned'])} of the onsets (median NSE {n2(ctrl_a['nse'])}), "
        f"against {pc(wq['onset_warned'])} ({n2(wq['nse'])}) without it and {pc(wqa['onset_warned'])} ({n2(wqa['nse'])}) with the AORC rain to midnight: the gain comes from the hours after about 09:00, not from the product.",
        f"Rain up to the evening of the issue day closes {rain['onset_share_in_words']} of the gap in onsets warned: {pc(wq['onset_warned'])} to {pc(wqn['onset_warned'])}, where the upper bound is "
        f"{pc(wqf['onset_warned'])}{', in every seed' if rain['onset_gain_clear'] else ''}; {mixed}.",
        f"Rain up to midnight (AORC) closes {pc(aorc['onset_share_of_gap_closed'])} of that gap in onsets warned ({pc(wq['onset_warned'])} to {pc(wqa['onset_warned'])}) and {pc(aorc['nse_share_of_gap_closed'])} of the gap in NSE "
        f"({n2(wq['nse'])} to {n2(wqa['nse'])}, where the upper bound is {n2(wqf['nse'])}): by the rule of section 4.2 the gains in onsets and in NSE are {aorc_gain_words} "
        f"(the NSE is higher in {aorc['n_basins_better']} of {n_b} basins). Its single seeds span {rng(wqa['onset_warned_seeds'], pc)} of the onsets and {rng(wqa['nse_seeds'], n2)} NSE.",
        f"Gains of the LSTM over past weather + river gauge, {days(LEAD)} (the length of the bars; differences taken before rounding): the river gauge itself adds {sg(S['river_gauge']['nse'])} NSE and "
        f"{pt(S['river_gauge'])} percentage point{plural(S['river_gauge'])} of onsets warned to past weather only"
        + ("" if S["river_gauge"]["onset_warned_clear"] else " (no clear change in the onsets: seed by seed "
           f"{pts(S['river_gauge']['onset_warned_points_seed_by_seed'][0] / 100)} to {pts(S['river_gauge']['onset_warned_points_seed_by_seed'][1] / 100)})")
        + f"; rain up to the evening {sg(rain['nse'])} and {pt(rain)}; rain up to midnight (AORC) {sg(aorc['nse'])} and {pt(aorc)}; the perfect weather forecast, an upper bound, {sg(gap['nse'])} and {pt(gap)}. "
          f"Beyond the NLDAS rain up to the evening the AORC rain adds {sg(S['aorc_beyond_evening']['nse'])} and {pt(S['aorc_beyond_evening'])}; what is left to the upper bound beyond the AORC rain is "
          f"{sg(S['perfect_forecast_beyond_aorc']['nse'])} and {pt(S['perfect_forecast_beyond_aorc'])}.",
        "The sets are alternatives, not a chain: each reads past weather + river gauge plus its own input. The AORC network does not read the NLDAS rain, and the perfect weather forecast reads the weather recorded for "
        "the three coming days (all of it, with the rain that fell after the morning of the issue day, which the rain table stamps on the next day: an inference from the project's own data, section 4.5 of "
        "the report) and not the extra rain of the other sets. All bars start at the level of past weather + river gauge (the dashed line) so that their lengths compare. In the lower block the light part of a bar is the "
        "gain of the rain up to midnight and the dark part what the real forecast (or the perfect rain) adds to it, because those networks read both.",
        f"Kinds of model on the same inputs, past weather + river gauge, {days(LEAD)}: median NSE {listed('wq', 'nse', n2)}, {n2(wq['nse'])} (LSTM) and {n2(tr['nse'])} (Transformer), "
        f"{n2(sp['wq']['nse']['spread'])} apart; onsets warned {listed('wq', 'onset_warned', pc)}, {pc(wq['onset_warned'])} (LSTM) and {pc(tr['onset_warned'])} (Transformer), "
        f"{sp['wq']['onset_warned']['spread_points']:.0f} percentage points apart."
        + (f" Between the LSTM and the best simple model ({b_wq['label']}, chosen on the validation years) the NSE shows {b_wq['against_lstm']['in_words']}: the LSTM is higher in "
           f"{b_wq['against_lstm']['n_basins_lstm_higher']} of {n_b} basins (p = {b_wq['against_lstm']['wilcoxon_p']:.2f})." if b_wq else ""),
        f"With past weather only the kinds of model lie {n2(sp['w']['nse']['spread'])} NSE apart: {listed('w', 'nse', n2)} and {n2(w['nse'])} (LSTM, single seeds {rng(w['nse_seeds'], n2)})."
        + ((f" Against the best simple model ({b_w['label']}) the LSTM is ahead only as the average of its {word(len(w['seeds']))} seeds: the averaged forecast is higher in "
            f"{b_w['against_lstm']['n_basins_lstm_higher']} of {n_b} basins, but the {b_w['label']} lies inside the range of the single LSTM seeds, so one training alone is not clearly ahead."
            if b_w["against_lstm"]["reading"] == "lstm_avg" else f" Against the best simple model ({b_w['label']}): {b_w['against_lstm']['in_words']}.") if b_w else ""),
        new_inputs_note,
        rests,
        f"The Transformer comes from its own experiment ({tr['experiment']['file']}): it was trained beside an LSTM twin with the same settings (sequences of {tr['experiment']['sequence_days']} days, "
        f"at most {tr['experiment']['max_updates']:,} updates, {word(len(tr['seeds']))} seed{'s' * (len(tr['seeds']) != 1)} each), not beside the LSTM drawn with past weather + river gauge. It is drawn at its "
        f"own scores (median NSE {n2(tr['nse'])}, {pc(tr['onset_warned'])} of onsets warned) beside that LSTM, not shifted or rescaled; its LSTM twin scores {n2(tr['twin']['nse'])} and "
        f"{pc(tr['twin']['onset_warned'])}, the LSTM drawn there {n2(wq['nse'])} and {pc(wq['onset_warned'])}.",
        "With the rain up to the evening and with the perfect weather forecast (the upper bound) only the LSTM is drawn. No other kind of model was run with the rain up to the evening; with the perfect weather forecast the LSTM and the simple models do not read the "
        f"same days of weather {days(LEAD)} (the LSTM reads the weather of all three coming days, the simple models only up to the day being forecast), so no comparison is like-for-like there "
        "and no winner is called.",
        f"'Tomorrow = today' reads the river gauge only: median NSE {n2(per['nse'])}, and {pc(per['onset_warned'])} of the onsets warned, because its forecast is above the high-flow level only "
        "when the river already is.",
        f"GO, by criteria fixed before the analysis: at least {pc(crit['go_min_warned'])} warned and at most {pc(crit['go_max_false'])} false warnings. The figure draws the first half (the green band). For onsets at "
        f"the high-flow level itself, {days(LEAD)}: past weather + river gauge {pc(wq['onset_warned'])} warned, {pc(wq['onset_false'])} false ({wq['onset_verdict']}); rain up to the evening "
        f"{pc(wqn['onset_warned'])} and {pc(wqn['onset_false'])} ({wqn['onset_verdict']}); AORC rain up to midnight {pc(wqa['onset_warned'])} and {pc(wqa['onset_false'])} ({wqa['onset_verdict']}); the upper bound "
        f"{pc(wqf['onset_warned'])} and {pc(wqf['onset_false'])} ({wqf['onset_verdict']}).",
        "Verdicts for onsets, by the criteria of results/flood_analysis.json, at the fixed warning level of the figure and at the level chosen on the validation years (as the results files store it): "
        + ("NOT YET for every set that could be had in practice, at both levels and at every lead time drawn (" + f"{len(verdicts['practice']['sets'])} sets and models counted)" if all_ns else
           "NOT YET for the sets that could be had in practice except " + _and(f"{e['name']} {days(e['lead'])} ({e['at_the_fixed_level']} at the fixed level, {e['at_the_level_chosen_on_validation']} at the chosen level)" for e in verdicts["practice"]["exceptions"]))
        + ". The upper bounds: " + "; ".join(f"{e['name'].replace('LSTM, ', '')} {days(e['lead'])} {e['at_the_fixed_level']}" + (f" ({e['at_the_level_chosen_on_validation']} at the chosen level)" if e["at_the_level_chosen_on_validation"] != e["at_the_fixed_level"] else "")
                                                       for e in verdicts["upper_bounds"]) + f". The perfect rain of the three days warns of {pc(bound_hits['onset_warned'])} of the onsets 3 days ahead with {bound_hits['onset_false'] * 100:.2f}% of its warnings false: "
          f"{bound_hits['onset_verdict']}, {100 * short_of_go:.2f} points of false warnings short of GO.",
        f"The lower block, 2 and 3 days ahead (results/forecast_input.json): the real weather forecast is the control member of the GEFSv12 reforecast, rain only, for the three days being forecast, from the 00 UTC run "
        f"of the UTC date t+1 (the latest run that exists at the issue time), averaged over each basin and bias-corrected per basin on the validation years; it is put in place of the AORC rain of those days ({AFTER}) "
        f"that the networks of the AORC set were trained on (a substitution, not a retraining). It adds {pts(fst[2]['forecast_over_rain_to_midnight']['onset_warned_points'] / 100)} and {pts(fst[3]['forecast_over_rain_to_midnight']['onset_warned_points'] / 100)} "
        f"points of onsets warned to the rain up to midnight ({pc(mid2[2]['onset_warned'])} and {pc(mid2[3]['onset_warned'])} to {pc(fl[2]['onset_warned'])} and {pc(fl[3]['onset_warned'])}; clear in every training) and "
        f"{sg(fst[2]['forecast_over_rain_to_midnight']['nse'])} and {sg(fst[3]['forecast_over_rain_to_midnight']['nse'])} NSE ({fst[2]['forecast_over_rain_to_midnight']['nse_in_words']} by the rule of section 4.2: higher in "
        f"{fst[2]['forecast_over_rain_to_midnight']['n_basins_better']} and {fst[3]['forecast_over_rain_to_midnight']['n_basins_better']} of {n_b} basins); counted from the rain table alone, "
        f"{pts(fst[2]['forecast_over_rain_table']['onset_warned_points'] / 100)} and {pts(fst[3]['forecast_over_rain_table']['onset_warned_points'] / 100)} points and {sg(fst[2]['forecast_over_rain_table']['nse'])} and "
        f"{sg(fst[3]['forecast_over_rain_table']['nse'])} NSE (the bars). It closes {pc(fst[2]['forecast_over_rain_to_midnight']['onset_share_of_gap_closed'])} and {pc(fst[3]['forecast_over_rain_to_midnight']['onset_share_of_gap_closed'])} "
        f"of the gap in onsets warned from the rain up to midnight to the perfect rain. One member only, 0.25-degree cells over basins of {min(areas):,.0f} to {max(areas):,.0f} km2, one bias factor per basin: a measurement of what such a forecast is worth, "
        "not of what a service would deliver (in Eastern summer the last files of the run are posted up to 15 minutes after the issue time, a margin measured on four days of the operational archive).",
        f"{days(LEAD).capitalize()} the same forecast, substituted in the same way, warns of {pc(ref_forecast_1(K, 'onset_warned'))} of the onsets against {pc(wqa['onset_warned'])} without it at the fixed warning level, with "
        f"{pc(ref_forecast_1(K, 'onset_false'))} of its warnings false against {pc(wqa['onset_false'])}: fewer onsets and fewer false warnings, a moved operating point, which the independent review of that experiment "
        "(results/review_forecast_input.json) does not read as lost skill; it is not drawn. The experiment's own shuffled-forecast control is not a neutral baseline (a network trained on the AORC rain and given noise "
        "collapses), so it is not drawn either. Retrained with the forecast as an input on a shorter split (train 2000-2008, test " + str(X["e2b"]["test_years"]) + ", with "
        + ", ".join(f"{n_sat[k]:,}" for k in F.LEADS) + " onsets at 1, 2 and 3 days; its own scores, not comparable in level with the figure) the onsets warned go from "
        + ", ".join(pc(X["e2b"]["sets"]["wqa_s"][lk(k)]["onset_warned"]) for k in F.LEADS) + " to " + ", ".join(pc(X["e2b"]["sets"]["wqaF_s"][lk(k)]["onset_warned"]) for k in F.LEADS) + " (perfect rain "
        + ", ".join(pc(X["e2b"]["sets"]["wqaP_s"][lk(k)]["onset_warned"]) for k in F.LEADS) + "); that retrained result is left to the report and is not drawn.",
        f"Not drawn, and where to find it: the ERA5 rain to midnight (E1: {pc(X['era5']['onset_warned'])} of the onsets, median NSE {n2(X['era5']['nse'])}, the same share as the NLDAS rain up to the evening, a stand-in for AORC "
        f"with the right clock; the report section on it) and the 63 training rivers instead of 12 (E3: for the same {n_b} rivers {days(LEAD)} median NSE {n2(X['rivers63']['nse_base'])} to {n2(X['rivers63']['nse'])}, "
        f"onsets warned {pc(X['rivers63']['onset_base'])} to {pc(X['rivers63']['onset_warned'])} with {pc(X['rivers63']['onset_false'])} of the warnings false: {X['rivers63']['nse_reading']} in NSE; more rivers are not more information about the day)."
        if "era5" in X and "rivers63" in X else "Not drawn: the ERA5 rain to midnight and the 63 training rivers (their results files were not found).",
        "The renderings: report = the drawing with its own title, subtitle and footnote; plain = the same drawing without them" + plain_drop_words + " (page 1 of the report); talk = for the closing slide, at larger lettering, "
        "without the rows 'river gauge only', 'past weather only' and 'perfect rain', without the small marks except at past weather + river gauge, without the control and the lines of the single seeds "
        "(none of them is in the headline); the numbers it draws are the same.",
        f"The LSTM mark is the forecast averaged over its seeds ({', '.join(str(len(x['lstm']['seeds'])) for x in (lv['w'], lv['wq'], lv['wqn'], lv['wqa'], lv['wqf']))} seeds, from top to bottom of the upper block); the thin "
        "line with caps is the range of the single seeds. The averaged forecast can score above every single seed.",
        f"Among the models tried: 'tomorrow = today', ridge regression, random forest, an LSTM with {R['lstm_all']['config']['hidden']} memory cells and a small Transformer. Larger networks were not tried.",
        "The same quantities 2 and 3 days ahead for the sets of the upper block (including the ones the lower block does not draw) are under 'not_drawn' (l2, l3), with the same keys as 'levels', 'steps' and 'control'; "
        "'far' holds the five sets of the lower block, with their gains over the rain table + river gauge and the forecast's own part.",
        f"Every number of the figure is read from a results file ('sources' names the file and the key); {CHECKS['compared']} values that a second file or key also holds were compared"
        + (f" and all agree (largest difference {CHECKS['largest_difference']:.1g})." if not CHECKS["disagreements"] else f"; {len(CHECKS['disagreements'])} disagree (see 'checks')."),
    ]
    if F.QUICK:
        notes.append("QUICK smoke test: the results files read here are those of the smoke tests (tiny budgets, and they may mix smoke-test and real prediction sets) - numbers are not results.")
    seeds_all = {len(x["lstm"]["seeds"]) for x in (lv["w"], lv["wq"], lv["wqn"], lv["wqa"], lv["wqf"])}
    marks = dict(seed="single seeds (their range)", control="control", linreg=KIND["linreg"], rf=KIND["rf"], transformer=KIND["transformer"] + " *")
    star = f"* The Transformer comes from its own experiment (its LSTM twin: {pc(ref(f'{lk(LEAD)}.cmp_lstm.onset_warned'))} of onsets warned, median NSE {n2(ref(f'{lk(LEAD)}.cmp_lstm.nse'))})."
    is_mixed = rain["nse_reading"] == "mixed"
    far_mixed = [k for k in FAR if fst[k]["forecast_over_rain_table"]["nse_reading"] == "mixed"]
    foot_mixed = ((f"NSE with rain up to the evening: mixed (higher in only {ref(f'{lk(LEAD)}.n_basins_better_with_rain_up_to_evening')} of {n_b} basins)" if is_mixed else "")
                  + ("; " if is_mixed and far_mixed else "") + (f"with the GEFS forecast: mixed at {_and([str(k) for k in far_mixed])} days" if far_mixed else "") + (". " if (is_mixed or far_mixed) else ""))
    more_false_everywhere = (all(o["onset_false"] > wqa["onset_false"] for o in a_models)
                             and all(o["onset_false"] > fl[k]["onset_false"] for k in FAR for o in far[k]["wqaF_sub"]["others"].values()))     # the simple models given the new inputs: more false warnings than the LSTM
    figure = dict(
        title=(f"Changing the model changes little; changing the information changes a lot, 1 to 3 days ahead" if claim else f"1 to 3 days ahead: {word(kinds)} kinds of model on the same inputs, and the LSTM given more information"),
        subtitle=("SMOKE TEST of the script: tiny training budgets, the numbers are not results." if F.QUICK else
                  "Among the models tried, the limit is the information in the inputs, not the network." if claim else
                  "The kinds of model do not lie closer together than every gain of the LSTM: no finding is stated here."),
        mixed_label="mixed" if is_mixed else "",
        footnote=(("SMOKE TEST, not results. " if F.QUICK else "")
                  + f"Test years {y0}-{y1}, {n_b} basins. Onset: a high-flow day forecast while the river was still below the high-flow level. Bars start at past weather + river gauge. "
                  f"Hatched: {BOUND}. AORC: {AFTER}; diamonds: Stage IV radar, a real-time product, in its place (right), without its last hour (left). "
                  f"GEFS: archived forecast rain of the next three days, one member, put in place of the AORC rain of those days that the model was trained on; lower block, light part of a bar: the rain up to midnight, dark part: what the forecast adds. Control: the same rain one day older. "
                  f"{foot_mixed}* Transformer: its own experiment (its LSTM twin: {pc(ref(f'{lk(LEAD)}.cmp_lstm.onset_warned'))}, {n2(ref(f'{lk(LEAD)}.cmp_lstm.nse'))})."),
        asterisk=star,
        panel_titles=["Onsets warned (share of the onsets)", f"Median NSE of the {n_b} basins (1 = perfect)"],
        legend=dict(lstm=f"LSTM, average of {word(seeds_all.pop()) if len(seeds_all) == 1 else 'its'} seeds", **marks),
        how_to_read=[
            "One row per set of information, from the least (top) to the most (bottom). Left panel: the share of onsets warned (the green band is what GO asks for); right panel: the median NSE of the basins. The upper block is "
            f"{days(LEAD)}; the lower block is {word(FAR[0])} and {word(FAR[1])} days ahead, one pair of narrow panels for each lead time.",
            "Large blue dot: the LSTM, the forecast averaged over its seeds; thin black line with caps behind it: the range of its single seeds (hidden where they lie inside the dot); bold numbers: the value of the dot.",
            "Thick bar: from the level of past weather + river gauge (the dashed vertical line) to the dot, so its length is what the extra information gives. In the lower block the light part of a bar is the gain of "
            "the rain up to midnight and the dark part what the real forecast, or the perfect rain, adds to it.",
            f"Hatched row, dashed bar, light dot: a perfect weather forecast, {BOUND}.",
            "Small hollow circle under the rain up to the evening: the control of the rain-timing experiment, the same rain one day older. Two hollow diamonds under the AORC rain: the real-time check, the same "
            "networks given Stage IV radar rain in place of AORC on the issue day (right) and with its last hour left out (left).",
            "Small coloured marks: the other kinds of model, drawn only where they read the same information as the LSTM - 'tomorrow = today' (grey diamond) with the river gauge only, ridge regression (triangle) "
            "and random forest (square) with past weather only, with past weather + river gauge, with the AORC rain up to midnight (1 day ahead) and with the GEFS forecast (2 and 3 days ahead), the Transformer (star, from its own experiment) "
            "with past weather + river gauge.",
            "Bracket at past weather + river gauge, with the note beside it: from the lowest to the highest of the kinds of model on the same inputs.",
            "'mixed' beside an NSE value: the gain over past weather + river gauge is a mixed result by the rule of section 4.2 (the median rises, but not clearly)."],
        caption=(f"What to see: on the same inputs (past weather + river gauge) the {word(kinds)} kinds of model lie within {n2(sp['wq']['nse']['spread'])} NSE and {sp['wq']['onset_warned']['spread_points']:.0f} percentage points of each other; "
                 f"the gains of the LSTM from more information are larger ({pt(rain)}, {pt(aorc)} and {pt(gap)} points of onsets warned with the rain up to the evening, the AORC rain up to midnight and a perfect weather forecast, "
                 f"{BOUND}). AORC is {AFTER}; with Stage IV radar in its place {pc(rt['variants'][0]['onset_warned']) if rt else 'n/a'} of the onsets are warned. "
                 f"{FAR[0]} and {FAR[1]} days ahead a real weather forecast lifts the onsets warned from {pc(base2[2]['onset_warned'])} and {pc(base2[3]['onset_warned'])} to {pc(fl[2]['onset_warned'])} and {pc(fl[3]['onset_warned'])}."
                 + (f" The NSE gain with rain up to the evening ({sg(rain['nse'])}) is a mixed result: higher in only {rain['n_basins_better']} of {n_b} basins." if is_mixed else "")
                 + (" For onsets the verdict is NOT YET for every set that could be had in practice." if all_ns else "")),
        caption_words_the_plain_figure_does_not_carry=None, beside_the_plain_and_talk_renderings=None)
    figure["caption_words_the_plain_figure_does_not_carry"] = (
        f"Bars start at the dashed line, the level of past weather + river gauge; in the lower block ({word(FAR[0])} and {word(FAR[1])} days ahead, {n2_:,} and {n3:,} onsets) the light part of a bar is the gain of the rain up "
        f"to midnight and the dark part what the forecast, or the perfect rain, adds to it. AORC is {AFTER}; the hollow diamonds are the same networks given Stage IV radar rain, a real-time product, in its place on the issue day"
        + (f" ({pc(rt['variants'][0]['onset_warned'])} of the onsets warned) and with its last hour left out ({pc(rt['variants'][1]['onset_warned'])})" if rt else "")
        + ". GEFS: archived forecast rain of the next three days, one member, put in place of the AORC rain of those days that the model was trained on. 'mixed' beside an NSE value: the gain over past weather + river gauge is a mixed result."
        + " Small marks are drawn also with the AORC rain (1 day) and with the GEFS forecast (2 and 3 days): "
        + ("at the fixed warning level they sit at another operating point, with more of their warnings false than the LSTM's." if more_false_everywhere else "see the notes for how they compare with the LSTM."))
    figure["beside_the_plain_and_talk_renderings"] = ("These two carry no title, subtitle or footnote: the document prints the words. What the footnote of the full rendering says: " + figure["footnote"] + " " + star)
    figure["plain"] = dict(
        file=RENDER["plain"]["file"], rows_left_out=[f"upper.{key}" for _, key in RENDER["plain"]["drop"]], names_of_the_rows_left_out=list(left_out_names.values()),
        why="Page 1 of the report holds one page, and this rendering is the one printed there: without these two rows it is about 0.5 in shorter (about 13 mm of the printed page), so that "
            "the page can keep more of its optional paragraphs. Neither row is in the headline: its comparisons are at past weather + river gauge and above. Their numbers are in 'levels' "
            "(sets q and w) and in 'sources' and are drawn in figures/key_figure.png, the rendering of the report body and of README.md. Apart from these two rows this rendering is the full "
            "drawing (the talk rendering has its own omissions: figure.renderings.talk.left_out).",
        what_the_two_rows_hold={x["key"]: dict(onset_warned=(x["others"]["base_persistence"]["onset_warned"] if x["key"] == "q" else x["lstm"]["onset_warned"]),
                                               nse=(x["others"]["base_persistence"]["nse"] if x["key"] == "q" else x["lstm"]["nse"])) for x in K["levels"] if x["key"] in left_out_names})
    figure["drawn_words"] = drawn_words(K)
    return headline, notes, figure


def ref_forecast_1(K, q):
    """The real forecast substituted into the AORC networks, 1 day ahead (not drawn): from the results file, kept in 'sources'."""
    f = K["forecast_1_day"]["wqaF_sub"]["lstm"]
    return f[q]


# ---- the check at the end of the build: every number the drawing printed or plotted, followed back to the file it was read from ----------------------------
def audit(K):
    """Every number that the drawing plots or prints (DRAWN_VALUES, by rendering) and every number in the lettering of the figure (TEXT_VALUES) must be in 'sources' flagged as in the figure,
    with the same value; a number read from a file is read there AGAIN and must still be the same, and a number computed here (a gain, a spread) is computed again from the two numbers it is
    the difference of. And the other way round: everything flagged as in the figure must be drawn or printed by the full rendering. The result goes to 'checks'."""
    bad, n_checked, n_reread, n_recomputed = [], 0, 0, 0
    names_drawn = {nme for items in DRAWN_VALUES.values() for nme, _ in items} | set(TEXT_VALUES)

    def reread(S_):
        f = S_["file_id"]
        return at(f, S_["key"])
    for rendering, items in DRAWN_VALUES.items():
        for nme, v in items:
            n_checked += 1
            S_ = SOURCES.get(nme)
            if S_ is None:
                bad.append(f"{rendering}: {nme} is drawn but has no entry in 'sources'")
                continue
            if not S_["in_figure"]:
                bad.append(f"{rendering}: {nme} is drawn but 'sources' says it is not in the figure")
            if S_["value"] != v and not (gone(S_["value"]) and gone(v)):
                bad.append(f"{rendering}: {nme} is drawn as {v!r} but 'sources' holds {S_['value']!r}")
            if "file_id" in S_:                                  # read from a file: read it again
                n_reread += 1
                again = reread(S_)
                if again != v and not (gone(again) and gone(v)):
                    bad.append(f"{rendering}: {nme} is drawn as {v!r} but {S_['file']} {S_['key']} now holds {again!r}")
            elif "difference_of" in S_:                          # computed here as a difference: compute it again
                n_recomputed += 1
                a, b = (SOURCES[x]["value"] for x in S_["difference_of"])
                if abs((a - b) - v) > 1e-12:
                    bad.append(f"{rendering}: {nme} is drawn as {v!r} but {S_['difference_of'][0]} minus {S_['difference_of'][1]} is {a - b!r}")
    missing = sorted(nme for nme, S_ in SOURCES.items() if S_["in_figure"] and nme not in names_drawn)
    if "report" in DRAWN_VALUES:                                 # the full rendering draws everything that is flagged as in the figure
        bad += [f"{nme} is flagged as in the figure but no rendering draws or prints it" for nme in missing]
    return dict(numbers_drawn_or_printed={r: len(i) for r, i in DRAWN_VALUES.items()}, numbers_in_the_lettering=len(set(TEXT_VALUES)), found_in_sources=n_checked, read_again_from_the_file=n_reread,
                recomputed_as_a_difference=n_recomputed, flagged_as_in_the_figure=sum(S_["in_figure"] for S_ in SOURCES.values()), flagged_but_not_drawn=missing, problems=bad,
                what="each number the drawing plots or prints, by rendering, is looked up in 'sources' (flagged in_figure, same value), read again from its file or computed again as a difference; and every "
                     "number flagged in_figure is drawn or printed by the full rendering")


# ---- main ------------------------------------------------------------------------------------------------------------------
def run():
    read_results()
    try:
        split, n_b, by_lead = build()
        X = extras()
    except (KeyError, IndexError, TypeError, AttributeError, StopIteration, ValueError, ZeroDivisionError) as e:
        if F.QUICK:                                         # a smoke-test file written before an earlier script had run may lack a table
            print(f"11_key_figure.py: a smoke-test results file does not hold what the figure needs ({type(e).__name__}: {e}) - no key figure drawn")
            sys.exit(0)
        raise
    if CHECKS["disagreements"] and not F.QUICK:             # never pick one of two different values silently
        sys.exit("11_key_figure.py stopped, nothing written: results files disagree by more than rounding\n  " + "\n  ".join(CHECKS["disagreements"]))
    drawn = by_lead[lk(LEAD)]
    crit = CRIT
    K = dict(question="What limits the forecast, the kind of model or the information in its inputs? 1, 2 and 3 days ahead.", quick_mode=F.QUICK, lead_drawn=LEAD, leads_drawn_below=list(FAR),
             test_years=[int(split[0][:4]), int(split[1][:4])], test_period=split, n_basins=n_b, headline=None,
             levels=drawn["levels"], steps=drawn["steps"], control=drawn["control"], criteria=crit, n_onsets={L: v["n_onsets"] for L, v in by_lead.items()},
             far={L: dict(lead=v["lead"], n_onsets=v["n_onsets"], levels=v["ladder2"], steps=v["steps2"]) for L, v in by_lead.items() if v["lead"] in FAR},
             forecast_1_day=drawn["forecast"], extras=X,
             not_drawn={L: {x: v[x] for x in ("levels", "steps", "control")} for L, v in by_lead.items() if L != lk(LEAD)},
             how_to_read=dict(levels="The six sets of information of the upper block, from top to bottom, for the drawn lead time (1 day). lstm: the forecast averaged over the seeds (nse = median NSE of the basins; onset_warned and "
                                     "onset_false = share of the onsets warned and share of the onset warnings that were false, at the high-flow level itself; onset_verdict by 'criteria' at that level, "
                                     "onset_verdict_at_chosen_level as the results files store it for the level chosen on the validation years; *_seeds = the single seeds, in the order of 'seeds'). others: the other kinds of model "
                                     "(drawn = it is in the figure; same_inputs = it reads the same information as the LSTM of that set; against_lstm = the comparison stored in the results file, read by the rule of section 4.2 of "
                                     "the report). real_time (set wqa): the check with Stage IV radar in place of AORC. control_not_drawn (set wqa): the same AORC rain over the rain table's own window.",
                              far="The lower block: the five sets of 2 days ahead (far.l2) and 3 days ahead (far.l3), each as in 'levels' (lstm, others), and 'steps': the gain of each set over the rain table + river gauge, "
                                  "and the forecast's own part (forecast_over_rain_to_midnight), with the reading of section 4.2 (nse_reading: clear / mixed / none / loss).",
                              steps="LSTM of one set minus LSTM of another, differences taken before rounding; onset_warned_points in percentage points. aorc_over_rain_table and rain_up_to_evening are the gains over "
                                    "past weather + river gauge that the bars of the figure show, and perfect_forecast the whole gap. spread_between_model_kinds: highest minus lowest of the models on the same inputs "
                                    "(wq is the bracket of the figure; w and wqa are not drawn).",
                              finding="Whether the numbers show what the title says, and the rule: holds = on the same inputs (past weather + river gauge) the kinds of model lie closer together than every step of the LSTM "
                                      "to the right of them, in onsets warned and in NSE. The title, the subtitle and the headline state the finding only when it holds.",
                              verdicts="The verdict for onsets of every set the figure draws, at the fixed level and at the level chosen on the validation years; practice.all_not_yet is the sentence of the headline.",
                              n_onsets="Onsets of the test years by lead time, the basins counted together; the drawn share is of l1 above and of l2 and l3 below.",
                              not_drawn="The quantities of the upper block's sets at 2 and 3 days (l2, l3); the sets that the lower block draws are also in 'far'.",
                              forecast_1_day="The real forecast and the perfect rain substituted in the same way at 1 day, which the figure does not draw (the forecast warns of fewer onsets at the fixed level there, with fewer false warnings).",
                              extras="Numbers of the experiments of 7 and 8 October that the figure does not draw, named in the notes: ERA5 rain to midnight, 63 training rivers, the forecast retrained on its own split.",
                              files="Three renderings of the one drawing, made by the same function from this file: report = with its own title, subtitle and footnote; plain = the same drawing without them and without the rows 'river gauge only' and 'past weather only' (page 1 of the "
                                    "report, where the caption carries the words and the page holds one figure of this height; see figure.plain); talk = for the closing slide, larger lettering, without the rows and small marks that are not in the headline. figure.renderings holds the "
                                    "size and the lettering of each, and what it leaves out.",
                              sources="For every number: the ONE file and key it was read from (or how it was computed here), and the other places that hold the same quantity and were checked against it. in_figure: the "
                                      "number is plotted or printed in the figure; file_id and key let the check at the end of the build read it again."))
    K = clean(K)
    K["finding"] = finding(K)                               # do the numbers show what the title says? (the title states the finding only then)
    K["verdicts"] = verdicts_everywhere(K)
    K["headline"], K["notes"], K["figure"] = sentences(K)
    K.update(sources=clean(SOURCES), checks=CHECKS)
    K["files"] = {name: ("figures/quick/" if F.QUICK else "figures/") + spec["file"] for name, spec in RENDER.items()}
    K = json.loads(json.dumps(K))                           # the figure reads exactly what the JSON holds
    K["figure"]["renderings"] = {name: draw(K, name) for name in RENDER}      # one drawing function, three renderings, all from what the JSON holds
    K["checks"]["drawn_numbers"] = audit(K)
    if K["checks"]["drawn_numbers"]["problems"] and not F.QUICK:
        sys.exit("11_key_figure.py stopped, nothing written: a number of the drawing does not match its source\n  " + "\n  ".join(K["checks"]["drawn_numbers"]["problems"][:20]))
    K["runtime_s"] = round(time.time() - t_start, 1)
    out = OUT_RES / "key_figure.json"
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp")   # written beside it, then moved into place: a document built meanwhile never reads half a file
    tmp.write_text(json.dumps(K, indent=1))
    tmp.replace(out)
    return K


if __name__ == "__main__":
    K = run()
    print(K["headline"])
    print("notes:", *K["notes"], sep="\n  ")
    if K["checks"]["disagreements"]:
        print("smoke-test files that disagree (not an error in a smoke test):", *K["checks"]["disagreements"][:5], sep="\n  ")
    d = K["checks"]["drawn_numbers"]
    print(f"{sum(v['in_figure'] for v in K['sources'].values())} numbers in the figure, each with its source; {K['checks']['compared']} comparisons between files and keys, "
          f"{len(K['checks']['disagreements'])} disagreements; the drawing: {sum(d['numbers_drawn_or_printed'].values())} numbers plotted or printed in three renderings, {d['read_again_from_the_file']} read again from "
          f"their files, {d['recomputed_as_a_difference']} computed again, {len(d['problems'])} problems")
    print("the title states the finding" if K["finding"]["holds"] else "the title states NO finding: the kinds of model do not lie closer together than every gain of the LSTM (key 'finding')")
    print(f"wrote {OUT_RES / 'key_figure.json'} and, in {OUT_FIG}: "
          + ", ".join(f"{v['file'].split('/')[-1]} ({v['width_in']} x {v['height_in']} in, {v['pixels'][0]} x {v['pixels'][1]} px, lettering from {v['smallest_lettering_pt']} pt)" for v in K["figure"]["renderings"].values())
          + f" in {K['runtime_s']} s")
