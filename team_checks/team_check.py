#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
team_check.py - a short, guided check of ONE member's part of Project-Flood.

WHAT IT DOES
  You say who you are (--member). The script recomputes a few headline numbers of your part
  from the stored forecast files (results/preds/*.npz), with its own short code, and prints
  each one next to the number in results/final_report_numbers.json - the registry that the
  report and the slides print. Every line ends with AGREE or DIFFER. "Agree" means: the same
  to the printed rounding (19%, 0.59) and the same counts of hits, misses and false alarms.

WHAT IT DOES NOT DO
  It writes nothing: no file, no cache, not team.json. It trains and loads no network.
  Recording your check in team.json is YOUR act, by hand (see TEAM_CHECK_SHEET.pdf).
  No script and no AI agent may fill team.json in for you.

NEEDS
  Python 3.10 or newer with numpy and pandas. Nothing else (no torch, no project imports).
  Folder layout: <anywhere>/team_checks/team_check.py next to <anywhere>/results/ (the project's results folder).

RUN (from the project folder)
  python -X utf8 team_checks/team_check.py --member member2
  members: lead, member2, member3, member4
  other options: --files (list what the check reads), -v (show registry keys), --root FOLDER

EXIT CODE
  0 = every comparison AGREE;  1 = at least one DIFFER or missing key;  2 = the check could not run.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True      # this script leaves no trace on the disk, not even a __pycache__

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# The rules of the project, written here again from scratch (they are the rules of code/flood.py)
# ---------------------------------------------------------------------------------------------
MAIN = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
SAT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}  # satellite-era split (E4e)
LEVEL_Q = 0.95      # flood level = the 95th percentile of the daily flow of the TRAINING years, per basin
SIMPLE = ["base_persistence", "base_climatology", "base_linreg_w", "base_linreg_wq", "base_rf_w", "base_rf_wq"]   # simple models without a weather forecast
SIMPLE_LABEL = {"base_persistence": "tomorrow = today", "base_climatology": "calendar-day average",
                "base_linreg_w": "ridge regression, weather only", "base_linreg_wq": "ridge regression + river gauge",
                "base_rf_w": "random forest, weather only", "base_rf_wq": "random forest + river gauge"}
PUNCT = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2192": "->", "\u2212": "-", "\u00a0": " "}


def ascii_safe(text):
    """Printable on any console, whatever its code page (the registry's wordings contain curly quotes)."""
    s = "".join(PUNCT.get(ch, ch) for ch in str(text))
    return s.encode("ascii", "replace").decode("ascii")


class CheckError(Exception):
    """The check cannot run (a file is missing or cannot be read)."""


# ---------------------------------------------------------------------------------------------
# formatting like the registry: percentages 19%, NSE 0.59, counts 1,116
# ---------------------------------------------------------------------------------------------
def ratio(a, b):
    return a / b if b else float("nan")


def pct(x):
    return "%.0f%%" % (100 * x)


def nse2(x):
    return "%.2f" % x


def num(n):
    return "{:,}".format(int(n))


def day(s):
    return np.datetime64(s, "D")


def in_range(dates, first, last):
    return (dates >= day(first)) & (dates <= day(last))


# ---------------------------------------------------------------------------------------------
# the numbers: flood level, onsets, warnings, NSE - all from the stored forecast files
# ---------------------------------------------------------------------------------------------
def levels(S, split):
    """Flood level per basin: the 95th percentile of the daily flow in the training years of the split."""
    tr = in_range(S["dates"], *split["train"])
    return np.array([np.nanquantile(S["obs"][b][tr], LEVEL_Q) for b in range(len(S["basins"]))])


def earlier(o, lead):
    """The flow `lead` days before each target day (NaN where there is none)."""
    prev = np.full(o.shape, np.nan)
    prev[lead:] = o[:-lead]
    return prev


def onsets_per_basin(S, split, lead=1):
    """Onsets of the test years per basin, counted from the observed flow alone (no forecast needed)."""
    te, lvl, out = in_range(S["dates"], *split["test"]), levels(S, split), []
    for b in range(len(S["basins"])):
        o = S["obs"][b]
        with np.errstate(invalid="ignore"):
            out.append(int((te & (o > lvl[b]) & (earlier(o, lead) <= lvl[b])).sum()))
    return out


def warning_table(S, split, lead=1):
    """Onset warnings of the test years, the 12 basins added together: hits, misses, false alarms.
    A day with a missing flow or a missing forecast is dropped."""
    te, lvl = in_range(S["dates"], *split["test"]), levels(S, split)
    hits = misses = false_alarms = 0
    for b in range(len(S["basins"])):
        o, p = S["obs"][b], S["pred"][b][:, lead - 1]
        with np.errstate(invalid="ignore"):
            keep = te & ~np.isnan(o) & ~np.isnan(p) & (earlier(o, lead) <= lvl[b])      # onset days: the flow `lead` days before was low
            high, warn = o > lvl[b], p > lvl[b]
        hits += int((keep & high & warn).sum())
        misses += int((keep & high & ~warn).sum())
        false_alarms += int((keep & ~high & warn).sum())
    return hits, misses, false_alarms


def nse_median(S, first, last, lead=1):
    """Median over the basins of NSE = 1 - sum((forecast - flow)^2) / sum((flow - mean flow)^2) on the days first..last."""
    mask, vals = in_range(S["dates"], first, last), []
    for b in range(len(S["basins"])):
        o, p = S["obs"][b], S["pred"][b][:, lead - 1]
        ok = mask & ~np.isnan(o) & ~np.isnan(p)
        o, p = o[ok], p[ok]
        vals.append(1.0 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))
    return float(np.median(vals))


# ---------------------------------------------------------------------------------------------
# the context: files, registry, printing, bookkeeping
# ---------------------------------------------------------------------------------------------
class Context:
    def __init__(self, root, verbose=False):
        self.root = Path(root)
        self.res = self.root / "results"
        self.verbose = verbose
        self.sets, self.files, self.results = {}, [], []
        self.stale, self.current = {}, None       # which forecast sets are newer than the registry; the set whose lines are printed now
        self.registry_path = self.res / "final_report_numbers.json"
        try:
            with open(self.registry_path, encoding="utf-8") as f:
                self.reg = json.load(f)
        except (OSError, ValueError) as e:
            raise CheckError("cannot read %s (%s). Copy the results/ folder of the project." % (self.registry_path, type(e).__name__))

    # ---- printing
    def say(self, text=""):
        print(ascii_safe(text))

    def head(self, text):
        self.say("")
        self.say(text)

    # ---- the registry
    def r(self, key):
        """The registry entry of a key (value, text, source), or None when the key does not exist."""
        e = self.reg.get(key)
        return e if isinstance(e, dict) and "value" in e else None

    def word(self, key):
        """A fixed wording of the project, read from the registry (never typed here)."""
        e = self.r(key)
        return ascii_safe(e["text"]) if e else "(missing key %s)" % key

    # ---- files (everything is only read)
    def read_json(self, name):
        path = self.res / name
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise CheckError("cannot read %s (%s)" % (path, type(e).__name__))
        self.files.append(path)
        return data

    def read_table(self, name):
        path = self.res / name
        try:
            t = pd.read_csv(path, dtype={"gauge_id": str})
        except (OSError, ValueError) as e:
            raise CheckError("cannot read %s (%s)" % (path, type(e).__name__))
        self.files.append(path)
        return t

    def load(self, name):
        """One stored forecast set: basins, dates, obs [basin, day], pred [basin, day, lead], meta."""
        if name in self.sets:
            return self.sets[name]
        path = self.res / "preds" / (name + ".npz")
        try:
            with np.load(path, allow_pickle=False) as z:
                S = dict(name=name, path=path, basins=[str(b) for b in z["basins"]], dates=z["dates"],
                         obs=np.asarray(z["obs"], dtype=float), pred=np.asarray(z["pred"], dtype=float), meta=str(z["meta"]))
        except Exception as e:      # a missing file, or one that a training run is writing right now
            raise CheckError("cannot read %s (%s: %s). If a training run is writing it, wait a minute and run again."
                             % (path, type(e).__name__, e))
        n_b, n_d = len(S["basins"]), len(S["dates"])
        if S["obs"].shape != (n_b, n_d) or S["pred"].shape != (n_b, n_d, 3):
            raise CheckError("%s does not have the shape [basin, day] / [basin, day, lead]" % path)
        try:
            S["seeds"] = len(json.loads(S["meta"]).get("seeds") or []) or None
        except ValueError:
            S["seeds"] = None
        self.sets[name] = S
        self.files.append(path)
        self.stale[name] = os.path.getmtime(path) > os.path.getmtime(self.registry_path) + 2
        return S

    # ---- one comparison = one printed line
    def row(self, label, mine, theirs, ok, keys=()):
        """ok is True (AGREE), False (DIFFER) or None (the registry has no such key)."""
        status = "AGREE" if ok is True else ("DIFFER" if ok is False else "MISSING")
        self.results.append(status)
        mark = "" if status == "AGREE" else "   <<<<<"
        self.say("    %-24s yours %-24s registry %-24s %s%s" % (label, mine, theirs, status, mark))
        if status != "AGREE" and self.current and self.stale.get(self.current):
            self.say("        ^ %s.npz is newer than the registry: probably a training run; see the NOTE at the end" % self.current)
        if self.verbose:
            for k in keys:
                e = self.r(k)
                self.say("        key %s  <-  %s" % (k, e["source"] if e else "NOT IN THE REGISTRY"))

    def present(self, keys):
        return all(self.r(k) is not None for k in keys)


# ---------------------------------------------------------------------------------------------
# building blocks used by several members
# ---------------------------------------------------------------------------------------------
def cmp_text(c, label, mine, key, shown=None):
    """One line: yours (`mine`, or `shown` if given) against the registry's printed text of `key`."""
    e = c.r(key)
    c.row(label, shown or mine, e["text"] if e else "(key missing)", None if not e else mine == e["text"], [key])


def cmp_int(c, label, mine, key, also=()):
    """One line: an integer against the registry's value; `also` are further numbers that must be the same."""
    e = c.r(key)
    c.row(label, num(mine), e["text"] if e else "(key missing)", None if not e else all(x == e["value"] for x in (mine,) + tuple(also)), [key])


def share_row(c, label, count, base, key_pct, key_count, key_other):
    """A share such as 'onsets warned' = hits / (hits + misses): the percentage to the printed rounding and both counts exactly.
    The registry's base is its own key_count + key_other."""
    keys = [key_pct, key_count, key_other]
    mine = "%s (%s of %s)" % (pct(ratio(count, base)), num(count), num(base))
    if not c.present(keys):
        c.row(label, mine, "(key missing)", None, keys)
        return
    e = [c.r(k) for k in keys]
    reg_base = e[1]["value"] + e[2]["value"]
    theirs = "%s (%s of %s)" % (e[0]["text"], num(e[1]["value"]), num(reg_base))
    c.row(label, mine, theirs, pct(ratio(count, base)) == e[0]["text"] and count == e[1]["value"] and base == reg_base, keys)


def set_block(c, name, label, stem, split, note=None, show_false=True, show_nse=False):
    """One stored forecast set, 1 day ahead: onsets warned (and the share of warnings that were false, and the NSE) next to the
    registry. `stem` is the start of the set's registry keys."""
    S = c.load(name)
    hits, misses, fa = warning_table(S, split, 1)
    c.say("")
    c.say("  %s  %s  (%s)" % (name, label, "%d seeds averaged" % S["seeds"] if S["seeds"] else "one model"))
    if note:
        c.say("      " + note)
    c.current = name
    k = lambda s: "%s_l1_%s" % (stem, s)
    share_row(c, "onsets warned", hits, hits + misses, k("onset_pod"), k("onset_hits"), k("onset_misses"))
    if show_false:
        share_row(c, "warnings that were false", fa, hits + fa, k("onset_far"), k("onset_false_alarms"), k("onset_hits"))
    if show_nse:
        cmp_text(c, "median NSE, 12 basins", nse2(nse_median(S, *split["test"], lead=1)), k("nse"))
    c.current = None
    return hits, misses, fa


def rule_block(c):
    c.head("THE RULE (the same as code/flood.py, written here again from scratch)")
    c.say("  flood level      per basin: the 95th percentile of the daily flow in the training years")
    c.say("  high-flow day    the flow is above the level.   Warning: the forecast is above the level.")
    c.say("  onset, k days    a high-flow day whose flow k days before was at or below the level")
    c.say("  scored on        the test years (target days), the 12 basins added together;")
    c.say("                   a day with a missing flow or a missing forecast is dropped")


# ---------------------------------------------------------------------------------------------
# the four checks
# ---------------------------------------------------------------------------------------------
def check_lead(c):
    c.say("THE IDEA  The finding of the project is a ladder: the same network, given more information, warns of more of")
    c.say("          the flood onsets one day ahead. You recompute three rungs of it from the stored forecasts.")
    rule_block(c)
    c.head("THE RUNGS, 1 DAY AHEAD  (main split: train 1980-10-01..2000-09-30, test 2005-10-01..2014-09-30)")
    h, m, _ = set_block(c, "lstm_wq", "rain table + river gauge", "lstm_wq", MAIN, show_false=False, show_nse=True)
    set_block(c, "lstm_wqa", "+ AORC rain up to midnight", "lstm_wqa", MAIN, show_false=False,
              note="%s; %s." % (c.word("fw_aorc"), c.word("fw_hindsight")))
    set_block(c, "lstm_wqf", "+ a perfect weather forecast", "lstm_wqf", MAIN, show_false=False, note=c.word("fw_upper") + ".")
    c.head("THE BASE OF EVERY PERCENTAGE")
    cmp_int(c, "onsets in the test years", h + m, "onsets_l1")
    c.head("DO ONE BY HAND  Take the hits and the onsets printed for lstm_wq and divide the first by the second on a")
    c.say("                calculator. Do you get the percentage printed? Then look at the ladder on your slides.")


def check_member2(c):
    c.say("THE IDEA  Slide 2 says how the 12 basins were chosen and that our GIS layers agree with the CAMELS table.")
    c.say("          You count the basins, check the GIS table, and recount the flood onsets from the flow records.")
    # ---- step 1: the funnel
    audit, table = c.read_json("data_audit.json"), c.read_table("basin_audit.csv")
    true = lambda col: table[col].astype(str).str.lower().eq("true")
    complete, chosen = true("complete"), true("selected")
    c.head("STEP 1  The basins, counted. Yours = your count of the rows of results/basin_audit.csv; it must also equal")
    c.say("        the number in results/data_audit.json. (The first line comes from data_audit.json alone.)")
    cmp_int(c, "all CAMELS basins", audit["n_all_camels"], "basins_all_camels")
    cmp_int(c, "south-eastern candidates", len(table), "basins_candidates", also=(audit["n_candidates"],))
    cmp_int(c, "complete record", int(complete.sum()), "basins_complete", also=(audit["n_complete"],))
    cmp_int(c, "record has gaps", int((~complete).sum()), "basins_with_gaps", also=(audit["n_with_gaps"],))
    cmp_int(c, "chosen", int(chosen.sum()), "basins_chosen", also=(len(audit["selected"]),))
    funnel = [audit["n_all_camels"], len(table), int(complete.sum()), int(chosen.sum())]
    e = c.r("basins_funnel")
    c.row("the funnel", " -> ".join(str(x) for x in funnel), e["text"] if e else "(key missing)", None if not e else funnel == e["value"], ["basins_funnel"])
    everything = pd.Series(True, index=table.index)
    by = {part: table[mask].groupby("landscape").size().to_dict() for part, mask in (("candidates", everything), ("complete", complete), ("chosen", chosen))}
    lands = sorted(by["candidates"])
    c.say("")
    c.say("    %-36s %10s %10s %10s" % ("per landscape (alphabetical)", "candidates", "complete", "chosen"))
    for land in lands:
        c.say("      %-34s %10d %10d %10d" % (land, by["candidates"].get(land, 0), by["complete"].get(land, 0), by["chosen"].get(land, 0)))
    for part, key in (("candidates", "basins_candidates_by_landscape"), ("complete", "basins_complete_by_landscape"), ("chosen", "basins_chosen_by_landscape")):
        e = c.r(key)
        mine = " + ".join(str(by[part].get(land, 0)) for land in lands)
        theirs = " + ".join(str(e["value"].get(land, 0)) for land in lands) if e else "(key missing)"
        c.row(part + " by landscape", mine, theirs, None if not e else mine == theirs, [key])
    # ---- step 2: the GIS check table
    g = c.read_json("gis_layers.json")
    rows = g["check_table"]
    area = [(r["short"], 100.0 * (r["area_km2"] - r["camels_area_gages2_km2"]) / r["camels_area_gages2_km2"]) for r in rows]
    elev = [(r["short"], r["elev_mean_m"] - r["camels_elev_mean_m"]) for r in rows]
    big_a, big_e = max(area, key=lambda t: abs(t[1])), max(elev, key=lambda t: abs(t[1]))
    c.head("STEP 2  The GIS check table (results/gis_layers.json): our area and elevation against the CAMELS table, %d basins" % len(rows))
    c.say("        area difference = (ours - CAMELS area_gages2) / CAMELS;  elevation difference = ours - CAMELS mean elevation")
    cmp_int(c, "basins in the table", len(rows), "basins_chosen", also=(len(g["basins"]),))
    cmp_text(c, "area, median difference", "%+.1f%%" % float(np.median([a for _, a in area])), "gis_area_vs_gages2_median_pct")
    cmp_text(c, "area, largest difference", "%+.1f%%" % big_a[1], "gis_area_vs_gages2_largest_pct", shown="%+.1f%% (%s)" % (big_a[1], big_a[0]))
    cmp_text(c, "elevation, median", "%+.1f m" % float(np.median([d for _, d in elev])), "gis_elev_median_diff_m")
    e1, e2 = c.r("gis_elev_largest_diff_m"), c.r("gis_elev_largest_diff_basin")
    mine = "%+.0f m" % big_e[1]
    c.row("elevation, largest", "%s (%s)" % (mine, big_e[0]), "%s (%s)" % (e1["text"], e2["text"]) if (e1 and e2) else "(key missing)",
          None if not (e1 and e2) else (mine == e1["text"] and big_e[0] == e2["text"]), ["gis_elev_largest_diff_m", "gis_elev_largest_diff_basin"])
    c.say("")
    c.say("    The mountain basins, mean elevation (a mark where ours differs from CAMELS by more than 25 m):")
    c.say("      %-18s %10s %12s %12s" % ("basin", "ours (m)", "CAMELS (m)", "difference"))
    for r in sorted((r for r in rows if r["landscape"].startswith("Mountains")), key=lambda r: r["camels_elev_mean_m"] - r["elev_mean_m"]):
        d = r["elev_mean_m"] - r["camels_elev_mean_m"]
        c.say("      %-18s %10.1f %12.1f %+12.1f%s" % (r["short"], r["elev_mean_m"], r["camels_elev_mean_m"], d, "   <- more than 25 m" if abs(d) > 25 else ""))
    folder = c.res / "gis_layers"
    n_files = len([p for p in folder.iterdir() if p.is_file()]) if folder.is_dir() else None
    e = c.r("gis_layers_files")
    c.row("files in gis_layers/", "folder missing" if n_files is None else str(n_files), e["text"] if e else "(key missing)",
          None if (not e or n_files is None) else n_files == e["value"], ["gis_layers_files"])
    # ---- step 3: onsets from the flow records
    S = c.load("lstm_wq")
    per = onsets_per_basin(S, MAIN, 1)
    names = {r["gauge_id"]: r["short"] for r in rows}
    c.head("STEP 3  One count from the flow records (the observed flow stored in results/preds/lstm_wq.npz)")
    c.say("        Flood level per basin = 95th percentile of 1980-10-01..2000-09-30. An onset = a day of the test years")
    c.say("        (2005-10-01..2014-09-30) above the level whose day before was at or below it.")
    cells = ["%s: %d" % (names.get(b, b), n) for b, n in zip(S["basins"], per)]
    c.say("")
    for i in range(0, len(cells), 3):
        c.say("      " + "".join("%-27s" % x for x in cells[i:i + 3]))
    c.say("")
    cmp_int(c, "onsets, 12 basins added", sum(per), "onsets_l1")
    c.head("DO ONE BY HAND  Add the twelve onset counts of step 3: do you get the total? Then take one basin of step 2 and")
    c.say("                subtract: ours minus CAMELS. Do you get the difference printed?")


def check_member3(c):
    S = c.load("lstm_wq")
    c.say("THE IDEA  Slide 3 (and slide 2's split bar): every period of the record has one job, and the number to beat is the best SIMPLE model,")
    c.say("          chosen on the validation years. You check the split, the number to beat, and the LSTM next to it.")
    # ---- step 1: the split
    dates = S["dates"]
    c.head("STEP 1  The split. It is the TARGET day of a forecast that decides which period the forecast belongs to.")
    masks = {}
    for label, part in (("train", "train"), ("validation", "val"), ("test", "test")):
        a, b = MAIN[part]
        masks[part] = in_range(dates, a, b)
        e_d = c.r("split_%s_dates" % part)
        c.row(label + ": dates", "%s..%s" % (a, b), "%s..%s" % tuple(e_d["value"]) if e_d else "(key missing)",
              None if not e_d else [a, b] == e_d["value"], ["split_%s_dates" % part])
        cmp_int(c, label + ": days", int(masks[part].sum()), "split_%s_days" % part)
    overlap = int(((masks["train"] & masks["val"]) | (masks["train"] & masks["test"]) | (masks["val"] & masks["test"])).sum())
    total = int(sum(m.sum() for m in masks.values()))
    e = c.r("days_per_basin")
    c.row("the three add up", "%s days, %d overlap" % (num(total), overlap), "%s days in all" % e["text"] if e else "(key missing)",
          None if not e else (total == len(dates) == e["value"] and overlap == 0), ["days_per_basin"])
    # ---- step 2: the flood level
    lv = levels(S, MAIN)
    c.head("STEP 2  The flood level is fitted on the TRAINING years only (95th percentile of the daily flow, per basin)")
    cmp_text(c, "lowest level, mm/day", "%.2f" % lv.min(), "high_flow_level_min_mm_day")
    cmp_text(c, "highest level, mm/day", "%.2f" % lv.max(), "high_flow_level_max_mm_day")
    # ---- step 3: the number to beat is chosen on the validation years
    simple = {n: c.load(n) for n in SIMPLE}
    val = {n: [nse_median(simple[n], *MAIN["val"], lead=k) for k in (1, 2, 3)] for n in SIMPLE}
    best = [max(SIMPLE, key=lambda n: val[n][i]) for i in range(3)]
    c.head("STEP 3  The number to beat = the simple model with the highest NSE on the VALIDATION years (2000-10-01..2005-09-30)")
    c.say("        Median NSE of the 12 basins on the validation years:")
    c.say("      %-16s %-32s %s" % ("", "", "".join("%9s  " % h for h in ("1 day", "2 days", "3 days"))))
    for n in SIMPLE:
        c.say("      %-16s %-32s %s" % (n, SIMPLE_LABEL[n], "".join("%8.2f%s  " % (val[n][i], "*" if best[i] == n else " ") for i in range(3))))
    c.say("        * = the highest of its column: this simple model sets the number to beat at that lead")
    # ---- step 4: the test years
    test = {n: [nse_median(simple[n], *MAIN["test"], lead=k) for k in (1, 2, 3)] for n in SIMPLE}
    c.head("STEP 4  Now the TEST years (2005-10-01..2014-09-30): that simple model, and the LSTM (lstm_wq) next to it")
    for k in (1, 2, 3):
        c.say("")
        c.say("  %d day%s ahead" % (k, "" if k == 1 else "s"))
        key_m, key_v, key_t, key_l = ("best_simple_without_weather_forecast_l%d_model" % k, "best_simple_without_weather_forecast_l%d_val_nse" % k,
                                       "best_simple_without_weather_forecast_l%d_nse" % k, "lstm_wq_l%d_nse" % k)
        em, ev = c.r(key_m), c.r(key_v)
        n = best[k - 1]
        c.row("best simple, validation", "%s (%s)" % (n, nse2(val[n][k - 1])), "%s (%s)" % (em["text"], ev["text"]) if (em and ev) else "(key missing)",
              None if not (em and ev) else (n == em["value"] and nse2(val[n][k - 1]) == ev["text"]), [key_m, key_v])
        t, l = test[n][k - 1], nse_median(S, *MAIN["test"], lead=k)
        cmp_text(c, "its NSE, test years", nse2(t), key_t)
        cmp_text(c, "lstm_wq, test years", nse2(l), key_l)
        c.say("      the LSTM minus the simple model: %+.2f NSE" % (0.0 if abs(l - t) < 0.005 else l - t))
        winner = max(SIMPLE, key=lambda m: test[m][k - 1])
        if nse2(test[winner][k - 1]) != nse2(t) and test[winner][k - 1] > t:
            c.say("      for reading: on the test years %s scores %s, more than the chosen %s (%s). It was NOT chosen:" % (
                winner, nse2(test[winner][k - 1]), n, nse2(t)))
            c.say("      only the validation years may choose; the test years are only scored.")
    c.head("DO ONE BY HAND  Add the three day counts of step 1: do you get all the days of the record? Then read the")
    c.say("                'for reading' lines of step 4, if there are any: why is a model that scores higher on the test")
    c.say("                years not the number to beat?")


def check_member4(c):
    c.say("THE IDEA  Slide 4 shows the ladder (slide 5 the river-state results): one network, more information, more onsets warned one day ahead - and")
    c.say("          the share of the warnings that were false, the other half of the GO rule. You recompute both.")
    rule_block(c)
    c.head("PART 1  The ladder, main split (train 1980-10-01..2000-09-30, test 2005-10-01..2014-09-30), 1 day ahead")
    set_block(c, "lstm_wq", "rain table + river gauge", "lstm_wq", MAIN)
    set_block(c, "lstm_wqn", "+ rain up to the evening (NLDAS)", "lstm_wqn", MAIN)
    set_block(c, "lstm_wqa", "+ AORC rain up to midnight", "lstm_wqa", MAIN, note="%s; %s." % (c.word("fw_aorc"), c.word("fw_hindsight")))
    set_block(c, "lstm_wqf", "+ a perfect weather forecast", "lstm_wqf", MAIN, note=c.word("fw_upper") + ".")
    c.head("PART 2  E4e, the newest results: the satellite-era split (flood level from the training years 2000-10-01..2008-09-30,")
    c.say("        test years 2010-10-01..2014-09-30), 1 day ahead. Comparable only inside this split, not with part 1.")
    for part in ("train", "test"):
        a, b = SAT[part]
        e = c.r("split_sat_%s_dates" % part)
        c.row("E4e %s years" % part, "%s..%s" % (a, b), "%s..%s" % tuple(e["value"]) if e else "(key missing)",
              None if not e else [a, b] == e["value"], ["split_sat_%s_dates" % part])
    aorc = "AORC rain: " + c.word("fw_aorc") + "."
    set_block(c, "lstm_wqa_s", "rain to midnight alone (the E4e base)", "e4e_wqa_s", SAT, note=aorc)
    set_block(c, "lstm_wqaH_s", "+ the river's hourly record to the issue time", "e4e_wqah_s", SAT, note=aorc)
    set_block(c, "lstm_wqaWHF_s", "+ within-day rain + GEFS forecast + river record", "e4e_wqawhf_s", SAT, note=aorc)
    c.head("DO ONE BY HAND  Take lstm_wq: false share = false warnings / (hits + false warnings). The counts are printed in")
    c.say("                brackets. Do you get the percentage printed? Then find the same bars on your slides.")


MEMBERS = {
    "lead": dict(full="The team lead", role="lead, RS/GIS", part="the headline ladder", slides="1, 5 and 6", fn=check_lead,
                     figure="figures/plain_ladder.png",
                     needs=["results/final_report_numbers.json"] + ["results/preds/%s.npz" % n for n in ("lstm_wq", "lstm_wqa", "lstm_wqf")]),
    "member2": dict(full="Member 2", role="RS/GIS", part="data, basins and GIS layers", slides="2", fn=check_member2,
                    figure="figures/basin_gis_check.png",
                    needs=["results/final_report_numbers.json", "results/data_audit.json", "results/basin_audit.csv", "results/gis_layers.json",
                           "results/gis_layers/", "results/preds/lstm_wq.npz"]),
    "member3": dict(full="Member 3", role="GNSS", part="model, fair test, number to beat", slides="3", fn=check_member3,
                     figure="figures/baselines.png",
                     needs=["results/final_report_numbers.json", "results/preds/lstm_wq.npz"] + ["results/preds/%s.npz" % n for n in SIMPLE]),
    "member4": dict(full="Member 4", role="GNSS", part="results", slides="4 (the E4e numbers are on slide 5)", fn=check_member4,
                 figure="figures/flow_state.png",
                 needs=["results/final_report_numbers.json"] + ["results/preds/%s.npz" % n for n in
                        ("lstm_wq", "lstm_wqn", "lstm_wqa", "lstm_wqf", "lstm_wqa_s", "lstm_wqaH_s", "lstm_wqaWHF_s")]),
}


# ---------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------
def pick_member(text):
    t = text.strip().lower()
    if t in MEMBERS:
        return t
    found = [k for k, m in MEMBERS.items() if t and (t == m["full"].lower() or t == m["full"].lower().split()[0])]
    return found[0] if len(found) == 1 else None


def find_root(given):
    if given:
        return Path(given).resolve()
    here = Path(__file__).resolve().parent.parent
    if (here / "results").is_dir():
        return here
    cwd = Path.cwd()
    return cwd if (cwd / "results").is_dir() else here


def stamp(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def notes_on_freshness(c):
    """Say so when a file that was read is newer than the registry, or holds another number of seeds than the registry was built from."""
    reg_time, notes = os.path.getmtime(c.registry_path), []
    for p in c.files:
        t = os.path.getmtime(p)
        if t > reg_time + 2:
            notes.append("%s was written %s, after the registry (%s)" % (p.name, stamp(t), stamp(reg_time)))
    for n, S in c.sets.items():
        e = c.r(n.lower() + "_l1_n_seeds")
        if e and S["seeds"] and S["seeds"] != e["value"]:
            notes.append("%s.npz holds %d seeds, the registry was built from %d" % (n, S["seeds"], e["value"]))
    return notes


def run(member, root, verbose):
    info = MEMBERS[member]
    c = Context(root, verbose)
    bar = "=" * 100
    c.say(bar)
    c.say("PROJECT-FLOOD TEAM CHECK   %s (%s)" % (info["full"], info["role"]))
    c.say("Your part: %s - deck slides %s" % (info["part"], info["slides"]))
    c.say('"yours" = recomputed here from the stored forecasts;  "registry" = the number the report and slides print')
    c.say("(results/final_report_numbers.json, built %s)." % stamp(os.path.getmtime(c.registry_path)))
    c.say("Reads files under results/ only. Writes nothing. Does not touch team.json: you record your own check there, by hand.")
    c.say(bar)
    info["fn"](c)
    c.head("-" * 100)
    n_all, n_agree = len(c.results), c.results.count("AGREE")
    notes = notes_on_freshness(c) if n_agree != n_all else []       # said only when something does not agree: it is the likely reason
    if notes:
        c.say("NOTE  Some files are newer than the registry or differ from it in seeds (a training run may still be going on):")
        for line in notes:
            c.say("      - " + line)
        c.say("      If a line above says DIFFER, the registry is probably out of date: do not record, wait until the team lead has")
        c.say("      rebuilt it, then run this again.")
    fig = Path(root) / info["figure"]
    c.say("FIGURE TO OPEN  %s%s" % (info["figure"], "" if fig.exists() else "   (not in this copy: ask the team lead for it)"))
    if n_all and n_agree == n_all:
        c.say("RESULT  all %d comparisons AGREE." % n_all)
        c.say("  Next: open the figure, read your page of TEAM_CHECK_SHEET.pdf, and look at your slides: they must print the same")
        c.say("  numbers as the registry column. If a slide shows another number, the deck is older: tell the team lead first.")
        c.say("  Only then record the check yourself in team.json. This script never does it, and nobody may do it for you.")
    else:
        c.say("RESULT  %d of %d comparisons do NOT agree (marked DIFFER or MISSING above)." % (n_all - n_agree, n_all))
        c.say("  Do NOT record your check as passed. Tell the team lead which lines differ.")
    if verbose:
        c.say("")
        c.say("Files read: " + ", ".join(sorted(set(p.name for p in c.files))))
    return 0 if (n_all and n_agree == n_all) else 1


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="A short guided check of one member's part of Project-Flood. Writes nothing.")
    ap.add_argument("--member", "-m", required=True, help="lead, member2, member3 or member4")
    ap.add_argument("--root", default=None, help="the folder that contains results/ (default: the folder above team_checks/)")
    ap.add_argument("--files", action="store_true", help="only list the files this check reads (to copy them to another computer)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also show the registry key and its source under every line")
    args = ap.parse_args(argv)
    member = pick_member(args.member)
    if not member:
        print("Unknown member '%s'. Use one of: %s" % (args.member, ", ".join(MEMBERS)), file=sys.stderr)
        return 2
    if args.files:
        print("Files the check of %s reads (relative to the project folder):" % MEMBERS[member]["full"])
        for p in MEMBERS[member]["needs"]:
            print("  " + p)
        print("Also copy, to open it (the check still runs without it): " + MEMBERS[member]["figure"])
        return 0
    root = find_root(args.root)
    missing = [p for p in MEMBERS[member]["needs"] if not ((root / p).is_dir() if p.endswith("/") else (root / p).is_file())]
    if missing:
        print("The check cannot run: these are missing under %s:" % root, file=sys.stderr)
        for p in missing:
            print("  " + p, file=sys.stderr)
        print("Copy them from the project (python team_checks/team_check.py --member %s --files lists everything it reads)." % member, file=sys.stderr)
        return 2
    try:
        return run(member, root, args.verbose)
    except CheckError as e:
        print("The check could not run: %s" % ascii_safe(e), file=sys.stderr)
        return 2
    except KeyError as e:
        print("The check could not run: a file has no entry %s (is it a copy of an older or damaged results folder?)" % ascii_safe(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
