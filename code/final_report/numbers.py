"""The numbers registry of the final report and the final deck (hand-in before 2 November 2026).

Run from the project root with the system Python:   python -X utf8 code/final_report/numbers.py

It reads results/*.json (never the preliminary diagnostics of 2026-10-28), code/final_report/config.json, team.json,
handoff/environment/this_mac.txt, code/schedule.py and the fixed wordings of code/report_sections/s0_summary.py and
code/build_slides.py (by regular expression, never by import: those modules import torch), and writes

    results/final_report_numbers.json     {key: {"value", "text", "source", "note"?}, "_meta": ..., "_optional_sources": [...]}
    code/final_report/NUMBERS_KEYS.md     the same keys, grouped by report section, for the writers ({{num:key}})

Rules it keeps: every number comes from a results file (the source names the file and the dotted path); the text is the
number as the documents print it (report_common: percentages as f"{100 x:.0f}%", NSE as f"{x:.2f}", counts with thousands
separators); no number is typed by hand; nothing from results/diagnostics_2026-10-28/ is read; the optional E4 files are
read only if they exist. Imports: json, re, csv, pathlib only (no statistics: see the note under the imports; no torch, no flood).
"""
import csv
import json
import re
import time
from pathlib import Path
# No `import statistics`: the file is named numbers.py (the brief's name), so when it runs as a script its folder heads sys.path and the
# standard library's own `numbers` module (which statistics -> fractions -> decimal import) would be shadowed by this file. The one median
# needed is computed by hand (median3). Load this module by file path, never by `import numbers` with code/final_report on sys.path.


def median3(xs):
    """The median of a short list (an odd or even count)."""
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

HERE = Path(__file__).resolve().parent            # code/final_report
ROOT = HERE.parents[1]                             # the project root
RES = ROOT / "results"
CODE = ROOT / "code"
OUT_JSON = RES / "final_report_numbers.json"
OUT_MD = HERE / "NUMBERS_KEYS.md"

OPTIONAL_SOURCES = ["results/warning_policy.json", "results/within_day_inputs.json", "results/flow_state.json",
                    "results/review_warning_policy.json", "results/review_within_day_inputs.json", "results/review_flow_state.json"]

MONTHS = ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
LEADS = ("l1", "l2", "l3")

# ---- reading -----------------------------------------------------------------------------------------------------------------------
_CACHE = {}


def load(name):
    """results/<name>.json, read once."""
    if name not in _CACHE:
        _CACHE[name] = json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))
    return _CACHE[name]


def get(d, path):
    """Dotted path into dicts and lists: 'sets.wq.test.summary.l1.nse_median', 'levels.2.lstm.nse'."""
    cur = d
    for p in path.split("."):
        if p == "":
            continue
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur[p]
    return cur


def opt(d, path, default=None):
    try:
        return get(d, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def src(name, path):
    return f"results/{name}.json -> {path}"


# ---- formatting, as report_common.py prints the numbers -------------------------------------------------------------------------------
def _bad(v):
    return v is None or (isinstance(v, float) and v != v)


def pct(v, d=0):
    return "n/a" if _bad(v) else f"{100 * v:.{d}f}%"


def n2(v, d=2):
    return "n/a" if _bad(v) else f"{v:.{d}f}"


def count(v):
    return "n/a" if _bad(v) else f"{int(round(v)):,}"


def signed(v, d=2):
    return "n/a" if _bad(v) else ("+" if v >= 0 else "-") + f"{abs(v):.{d}f}"


def points(v, d=1):
    """A difference of shares as percentage points: 0.326 -> '+32.6 points'."""
    return "n/a" if _bad(v) else signed(100 * v, d) + " points"


def pts(v, d=1):
    """A difference already in points: 32.6 -> '+32.6 points'."""
    return "n/a" if _bad(v) else signed(v, d) + " points"


def pct_range(lo, hi):
    """One decimal when the two ends round to the same whole percent (critic 3, N6: never '18% to 18%')."""
    d = 1 if (not _bad(lo) and not _bad(hi) and lo != hi and pct(lo) == pct(hi)) else 0
    return f"{pct(lo, d)} to {pct(hi, d)}"


def n2_range(lo, hi):
    return f"{n2(lo)} to {n2(hi)}"


def size_text(nbytes):
    if _bad(nbytes):
        return "n/a"
    if nbytes >= 10e9:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e3:.1f} kB"


def long_date(iso):
    """'2026-10-02' -> '2 October 2026'."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso))
    if not m:
        return str(iso)
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{d} {MONTHS[mo]} {y}"


def minutes_text(seconds):
    return "n/a" if _bad(seconds) else f"{seconds / 60:.0f} min"


# ---- the registry ------------------------------------------------------------------------------------------------------------------
REG = {}
SECTIONS = []          # [(title, [keys])]
_CUR = None


def section(title):
    global _CUR
    _CUR = [title, []]
    SECTIONS.append(_CUR)


def add(key, value, text, source, note=None):
    key = key.lower()          # prediction-set names carry capitals (lstm_wqaF_sub, lstm_wqaP): the keys are their lower-case form
    if not re.fullmatch(r"[a-z0-9_]+", key):
        raise ValueError(f"key is not snake_case: {key}")
    if key in REG:
        raise KeyError(f"duplicate key: {key}")
    e = {"value": value, "text": str(text), "source": source}
    if note:
        e["note"] = note
    REG[key] = e
    _CUR[1].append(key)
    return e


def read_text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def rsquo(s):
    """The HTML entities the builders use, as characters."""
    return s.replace("&rsquo;", "’").replace("&lsquo;", "‘").replace("&ndash;", "–").replace("&minus;", "−").replace("&amp;", "&")


# ---- the review records: rounds -------------------------------------------------------------------------------------------------------
def _round_keys(rv):
    """The later rounds of a review record (round_2, round_3, ...) in order; round 1 is the record's top level."""
    return sorted((k for k in rv if re.fullmatch(r"round_\d+", k) and isinstance(rv[k], dict)), key=lambda k: int(k.split("_")[1]))


def _rounds(rv):
    """[(n, path prefix, block)] for every round of a review record: (1, '', top level), (2, 'round_2.', ...), ..."""
    return [(1, "", rv)] + [(int(k.split("_")[1]), k + ".", rv[k]) for k in _round_keys(rv)]


def _round_date(R):
    m = re.search(r"\d{4}-\d{2}-\d{2}", str(opt(R, "dates.review") or R.get("when", "")))
    return m.group(0) if m else None


def _failed_checks(R):
    """The checks a round marks failed ('result': 'fail' or 'passed': false); 'finding' and 'pass with a finding' are not failures."""
    ch = R.get("checks") or []
    ch = list(ch.values()) if isinstance(ch, dict) else ch
    return [c for c in ch if isinstance(c, dict) and (c.get("passed") is False or str(c.get("result", "")).strip().lower() in ("fail", "failed"))]


def _number_failures(R):
    """(failed checks about a NUMBER, failed checks about anything else) of one round. A failed check is about a number when it records
    mismatches; a failure the counts block holds that no listed check explains is counted as a number failure, to be safe. The other
    failed checks are about wording, disclosure or the scripts' bookkeeping."""
    fc = _failed_checks(R)
    n_num = sum(1 for c in fc if (c.get("mismatches") or 0) > 0)
    unexplained = max(0, ((R.get("counts") or {}).get("failed") or 0) - len(fc))
    return n_num + unexplained, len(fc) - n_num


def _hindsight_bounds():
    """The hindsight bound of the independent review of E1b: five seeds (the latest round's hindsight block, else recomputed from the
    results files) and three seeds (round 1, check d7)."""
    RV = load("review_aorc_rain_to_midnight")
    AR, ST = load("aorc_rain_to_midnight"), load("stage4_realtime")
    on = get(AR, "summary_test.sets.wqa.l1.onset")
    onsets = on["hits"] + on["misses"]
    H, hk = None, None
    for k in reversed(_round_keys(RV)):
        if opt(RV, f"{k}.hindsight.five_seed_loosest_bound_points") is not None:
            H, hk = RV[k]["hindsight"], k
            break
    if H is not None:
        points, hits = H["five_seed_loosest_bound_points"], H.get("five_seed_loosest_bound_hits")
        of_hits = opt(H, "five_seeds.aorc_all", on["hits"])
        source = src("review_aorc_rain_to_midnight", f"{hk}.hindsight.five_seed_loosest_bound_points") + " (rounded to whole points)"
        restored, restored_hits = H.get("five_seed_level_matched_points"), opt(H, "five_seeds.direct_all_611_level_matched_loss_hits")
        restored_src = src("review_aorc_rain_to_midnight", f"{hk}.hindsight.five_seed_level_matched_points")
    else:
        s4 = get(ST, "models.wqa.variants.stage4.onsets_warned")
        hits, of_hits = on["hits"] - s4, on["hits"]
        points = hits / onsets * 100
        source = (f"computed: ({src('aorc_rain_to_midnight', 'summary_test.sets.wqa.l1.onset.hits')} - {src('stage4_realtime', 'models.wqa.variants.stage4.onsets_warned')})"
                  f" / {onsets} onsets x 100, rounded to whole points")
        restored = restored_hits = restored_src = None
    d7 = next(c for c in RV["checks"] if isinstance(c, dict) and c.get("id") == "d7")
    p3 = d7["numbers"]["loosest_bound_points"]
    h3 = d7["numbers"]["loosest_bound_hits"]
    of3 = opt(H, "round1_of_hits") if H is not None else None
    if of3 is None:
        a3 = RES / "archive" / "aorc_rain_to_midnight_3seeds_2026-10-30.json"
        of3 = get(json.loads(a3.read_text(encoding="utf-8")), "summary_test.sets.wqa.l1.onset.hits") if a3.exists() else None
    return dict(points=points, hits=hits, of_hits=of_hits, onsets=onsets, source=source, wording=f"hindsight at most about {round(points)} points",
                restored=restored, restored_hits=restored_hits, restored_src=restored_src, round=hk,
                points_three=p3, hits_three=h3, of_hits_three=of3, wording_three=f"hindsight at most about {round(p3)} points",
                source_three=src("review_aorc_rain_to_midnight", "checks[id=d7].numbers.loosest_bound_points"))


# ====================================================================================================================================
def build():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    team = json.loads((ROOT / "team.json").read_text(encoding="utf-8"))

    # ---- 0. project, team, dates, machines ----------------------------------------------------------------------------------------------
    section("0 Project, team, dates and machines (cover, T1, T3, 4.3, 10.1)")
    add("project_title", cfg["title"], cfg["title"], "code/final_report/config.json -> title")
    gn = cfg.get("group_number")
    add("group_number", gn, f"Group #{gn}" if gn not in (None, "") else "Group #__", "code/final_report/config.json -> group_number",
        "null in config.json: the group number is not recorded in any project file; print the placeholder until the user fills it in")
    add("group_leader", cfg["group_leader"], cfg["group_leader"], "code/final_report/config.json -> group_leader")
    add("course", cfg["course"], cfg["course"], "code/final_report/config.json -> course")
    add("instructor", cfg["instructor"], cfg["instructor"], "code/final_report/config.json -> instructor")
    add("final_presentation", cfg["final_presentation"], cfg["final_presentation"], "code/final_report/config.json -> final_presentation")
    add("submission_deadline", cfg["submission_deadline"], cfg["submission_deadline"], "code/final_report/config.json -> submission_deadline")
    add("n_members", len(cfg["members"]), str(len(cfg["members"])), "code/final_report/config.json -> members (length)")
    for i, m in enumerate(cfg["members"], 1):
        add(f"member{i}_name", m["name"], m["name"], f"code/final_report/config.json -> members.{i - 1}.name")
        add(f"member{i}_id", m["student_id"], m["student_id"], f"code/final_report/config.json -> members.{i - 1}.student_id")
        add(f"member{i}_role", m["role"], m["role"], f"code/final_report/config.json -> members.{i - 1}.role")
    add("flood_cloud_model_authors", cfg["flood_cloud_model_authors"], cfg["flood_cloud_model_authors"],
        "code/final_report/config.json -> flood_cloud_model_authors")
    add("windows_pc", cfg["windows_pc"], cfg["windows_pc"], "code/final_report/config.json -> windows_pc",
        "the hand-in machine; seeds 3 and 4 of scripts 15-18 were trained on it on 30 October 2026 (seed_extension_*), nothing before")
    mac = read_text("handoff/environment/this_mac.txt").splitlines()
    add("mac_machine", mac[0].replace("Made on: ", ""), mac[0].replace("Made on: ", ""), "handoff/environment/this_mac.txt line 1")
    mp = re.search(r"Main env .*?: Python ([\d.]+)", "\n".join(mac))
    add("mac_python", mp.group(1) if mp else None, mp.group(1) if mp else "n/a", "handoff/environment/this_mac.txt line 2")
    MC = load("model_comparison")
    add("mac_cpu", get(MC, "settings.machine.cpu"), get(MC, "settings.machine.cpu"), src("model_comparison", "settings.machine.cpu"))
    add("mac_logical_cpus", get(MC, "settings.machine.logical_cpus"), str(get(MC, "settings.machine.logical_cpus")), src("model_comparison", "settings.machine.logical_cpus"))
    add("torch_threads", get(MC, "settings.machine.torch_threads"), str(get(MC, "settings.machine.torch_threads")), src("model_comparison", "settings.machine.torch_threads"))
    add("torch_version", get(MC, "settings.machine.torch"), get(MC, "settings.machine.torch"), src("model_comparison", "settings.machine.torch"))
    TEAM_DONE = [m for m in team["members"] if m.get("check_done") and str(m.get("check_date") or "").strip() and str(m.get("check_result") or "").strip()]
    n_done = len(TEAM_DONE)                                       # team.json _how_to_use: a check without date and result counts as not done
    add("team_checks_done", n_done, f"{n_done} of {len(team['members'])}", "team.json -> members[*].check_done (count of true)",
        "0 of 4 on 2026-10-28: the team's own checks are not done; only the members may change team.json")
    add("team_n_members", len(team["members"]), str(len(team["members"])), "team.json -> members (length)")
    for i, m in enumerate(team["members"]):
        add(f"team_member{i + 1}_check_done", bool(m.get("check_done")), "done" if m.get("check_done") else "not yet done", f"team.json -> members.{i}.check_done")
    # dates
    ER = load("engine_review")
    add("date_data_audit", get(ER, "dates.data_audit_and_basin_selection"), long_date(get(ER, "dates.data_audit_and_basin_selection")), src("engine_review", "dates.data_audit_and_basin_selection"))
    add("date_first_experiments", get(ER, "dates.everything_else"), long_date(get(ER, "dates.everything_else")), src("engine_review", "dates.everything_else"),
        "baselines, one-river and all-river LSTM, unseen-river test, satellite inputs, rain timing, Transformer: all on this day")
    add("date_engine_review", re.match(r"\d{4}-\d{2}-\d{2}", ER["when"]).group(0), long_date(ER["when"]), src("engine_review", "when"))
    add("date_day_window_check", get(ER, "dates.day_window_check_rerun"), long_date(get(ER, "dates.day_window_check_rerun")), src("engine_review", "dates.day_window_check_rerun"))
    add("date_late_additions", get(ER, "dates.additions_from_stored_results"), long_date(get(ER, "dates.additions_from_stored_results")), src("engine_review", "dates.additions_from_stored_results"),
        "key figure, learning curves, same-misses check, resampling ranges: analyses of stored results, no new training")
    for name, label in (("rain_to_midnight", "e1"), ("aorc_rain_to_midnight", "e1b"), ("more_basins", "e3"), ("forecast_input", "e2"), ("documents_2026-10-08", "documents"),
                        ("warning_policy", "e4a"), ("within_day_inputs", "e4b"), ("flow_state", "e4e"), ("documents_2026-10-29", "documents_2026_10_29"),
                        ("documents_2026-10-29b", "documents_2026_10_29b"), ("documents_2026-10-30", "documents_2026_10_30")):
        if not (RES / f"review_{name}.json").exists():        # the records of 28-29 October (E4 work order, second document pass) are optional
            continue
        when = load(f"review_{name}")["when"]
        d = re.search(r"\d{4}-\d{2}-\d{2}", when).group(0)
        add(f"date_review_{label}", d, long_date(d), src(f"review_{name}", "when"), when)
    DF = load("data_flow")
    add("date_data_record_written", DF["written"], long_date(DF["written"]), src("data_flow", "written"), DF["written"])
    days = get(DF, "all_information.download_dates.days")
    add("download_days", days, ", ".join(long_date(x) for x in days), src("data_flow", "all_information.download_dates.days"))
    for k, v in get(DF, "all_information.download_dates.by_product").items():
        add(f"download_date_{k}", v, v, src("data_flow", f"all_information.download_dates.by_product.{k}"))
    sched = read_text("code/schedule.py")
    m = re.search(r"PROPOSAL_DAY = datetime\.date\((\d+), (\d+), (\d+)\)", sched)
    pd_iso = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    add("date_proposal_talk", pd_iso, long_date(pd_iso), "code/schedule.py -> PROPOSAL_DAY")
    m = re.search(r"DEADLINE = datetime\.date\((\d+), (\d+), (\d+)\)", sched)
    dl_iso = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    add("date_old_deadline", dl_iso, long_date(dl_iso), "code/schedule.py -> DEADLINE",
        "SUPERSEDED by the professor's e-mail of 28 Oct 2026 (hand-in before 2 November, presentations 2 November): print only as 'the planned date'")
    for i, (y, mo, d, text) in enumerate(re.findall(r"\(datetime\.date\((\d+), (\d+), (\d+)\), \"([^\"]+)\"\)", sched), 1):
        iso = f"{y}-{int(mo):02d}-{int(d):02d}"
        add(f"milestone{i}_date", iso, long_date(iso), f"code/schedule.py -> MILESTONES[{i - 1}]", "planned milestone of the three-week plan of 14 Oct 2026")
        add(f"milestone{i}_text", text, text, f"code/schedule.py -> MILESTONES[{i - 1}]")
    for i, (y1, m1, d1, y2, m2, d2, title) in enumerate(re.findall(
            r"from_=datetime\.date\((\d+), (\d+), (\d+)\), to=datetime\.date\((\d+), (\d+), (\d+)\), title=\"([^\"]+)\"", sched), 1):
        add(f"week{i}_dates", [f"{y1}-{int(m1):02d}-{int(d1):02d}", f"{y2}-{int(m2):02d}-{int(d2):02d}"],
            f"{int(d1)} {MONTHS[int(m1)][:3]} to {int(d2)} {MONTHS[int(m2)][:3]} {y2}", f"code/schedule.py -> WEEKS[{i - 1}]")
        add(f"week{i}_title", title, title, f"code/schedule.py -> WEEKS[{i - 1}].title")

    # ---- fixed wordings ----------------------------------------------------------------------------------------------------------------
    section("Fixed wordings, word for word (every document)")
    s0 = read_text("code/report_sections/s0_summary.py")
    s0_lines = s0.splitlines()

    def const(name, text=s0, lines=s0_lines, file="code/report_sections/s0_summary.py"):
        m = re.search(r"^" + name + r" = \"((?:[^\"\\]|\\.)*)\"", text, re.M)
        if not m:
            return None, None
        ln = next(i + 1 for i, l in enumerate(lines) if l.startswith(name + " = "))
        return m.group(1), f"{file}:{ln}"

    for key, name, note in (("fw_finding", "FINDING", "the main finding; no full stop in the source"),
                            ("fw_support", "SUPPORT", "the supporting line of the finding"),
                            ("fw_aorc", "AORC_WORDING", "beside every AORC mention; AORC is never 'observed' rain"),
                            ("fw_moved_point", "MOVED_POINT", "E2's 1-day result and E3's 19% -> 17%: said of the warning scores only"),
                            ("fw_info_not_net", "INFO_NOT_NET", "the gains of the simple models given the same inputs"),
                            ("fw_realtime", "REALTIME_WORDING", "beside every 48% and 45%; the slides use the short form 'a check on the test years, not a service'"),
                            ("fw_hindsight", "HINDSIGHT_WORDING", "beside every AORC share of lstm_wqa (52% with five seeds, 51% with three)"),
                            ("fw_nse_forecast", "NSE_FORECAST_WORDING", "never 'mixed at all three leads'")):
        val, where = const(name)
        if key == "fw_hindsight":                                 # computed from the latest review of E1b (30 October 2026: five seeds), not copied from s0_summary
            HB = _hindsight_bounds()
            add(key, HB["wording"], HB["wording"], HB["source"],
                f"{note}; the bound of the independent review for the five-seed lstm_wqa ({HB['points']:.1f} points, {HB['hits']} of {HB['of_hits']} hits on all {HB['onsets']} onsets, "
                f"raw Stage IV); measured on the three networks of seeds 0-2 in round 1 of the E1b review (7 October 2026) it was '{HB['wording_three']}' (hindsight_three_seed_wording). "
                f"{where} HINDSIGHT_WORDING reads '{val}'" + (" (the three-seed wording: the documents built from the registry print this key, not that constant)" if val != HB["wording"] else ""))
            continue
        add(key, val, val, where, note)
    m = re.search(r"^REVIEW_WORDING = \((.*?)\)\s*$", s0, re.M | re.S)
    parts = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(1))
    ln = next(i + 1 for i, l in enumerate(s0_lines) if l.startswith("REVIEW_WORDING = "))
    add("fw_review", "".join(parts), "".join(parts), f"code/report_sections/s0_summary.py:{ln}-{ln + 1}",
        "the reviewers are AI agents, not team members; use it whenever the reviews are mentioned")
    m = re.search(r"^AORC_LONG = \((.*?)\)\s*$", s0, re.M | re.S)
    if m:
        parts = re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(1))
        long = "".join(parts).replace("{AORC_WORDING}", REG["fw_aorc"]["value"])
        ln = next(i + 1 for i, l in enumerate(s0_lines) if l.startswith("AORC_LONG = "))
        add("fw_aorc_long", long, long, f"code/report_sections/s0_summary.py:{ln}-{ln + 1}", "the long form of the AORC wording")
    bs = read_text("code/build_slides.py")
    bs_lines = bs.splitlines()
    val, where = const("UPPER", bs, bs_lines, "code/build_slides.py")
    add("fw_upper", val, val, where, "every perfect weather / perfect rain row (hatched in the figures)")
    val, where = const("FINDING_SHORT", bs, bs_lines, "code/build_slides.py")
    add("fw_finding_short", val, val, where, "the short form of the finding for a slide title")
    m = re.search(r"return True, \"(The team&rsquo;s own checks are not yet done\.)\"", bs)
    ln = next(i + 1 for i, l in enumerate(bs_lines) if "The team&rsquo;s own checks are not yet done." in l)
    team_now = json.loads((ROOT / "team.json").read_text(encoding="utf-8"))
    t_done = [x.get("name") or x.get("role") for x in team_now["members"]
              if x.get("check_done") and str(x.get("check_date") or "").strip() and str(x.get("check_result") or "").strip()]
    t_n = len(team_now["members"])
    if not t_done:
        tc = rsquo(m.group(1))                                    # the fixed wording while no member has recorded a check
    elif len(t_done) < t_n:
        tc = f"{len(t_done)} of the team\u2019s {t_n} own checks are recorded in team.json ({', '.join(t_done)}); the others are not yet done."
    else:
        tc = f"The team\u2019s own checks are recorded in team.json by all {t_n} members."
    add("fw_team_checks", tc, tc, f"code/build_slides.py:{ln} and team.json -> members[*].check_done, check_date, check_result",
        f"the fixed wording '{m.group(1)}' while no member has recorded a check; the sentence follows team.json (a check counts only with its date and result)")
    bss = read_text("code/build_slides_short.py")
    m = re.search(r"Who did what:</b> (Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction\.)", bss)
    ln = next(i + 1 for i, l in enumerate(bss.splitlines()) if "Who did what:</b> Claude Code wrote all code" in l)
    add("fw_who_did_what", rsquo(m.group(1)), rsquo(m.group(1)), f"code/build_slides_short.py:{ln}", f"raw source text: {m.group(1)}")
    add("fw_realtime_short", "a check on the test years, not a service", "a check on the test years, not a service",
        "HANDOFF.md section 7 (the slides' short form of REALTIME_WORDING)", "use the full REALTIME_WORDING in the report")
    add("fw_preliminary_label", "PRELIMINARY", "PRELIMINARY", "HANDOFF.md section 7 ('label preliminary numbers'); NEXT_EXPERIMENTS_2026-10-28.md",
        "any number from an experiment without its own review record carries this word; nothing from results/diagnostics_2026-10-28/ is quoted at all")

    build_data(DF)
    build_model_and_checks(MC, ER)
    build_sets()
    build_key_figure()
    build_experiments()
    build_failures_clock_misses()
    build_uncertainty()
    build_reviews_and_work()
    build_seed_extension(cfg)
    build_targets_and_published()
    build_optional_e4()
    build_hourly_flow()
    build_handin_additions()
    return REG


# ====================================================================================================================================
def build_data(DF):
    """4.1 data inventory, 4.2 records, split, levels, clocks, Appendix B basins."""
    section("4.1 Data inventory and provenance (T5, Appendix B)")
    AI = DF["all_information"]
    c0 = get(DF, "summary.flow.0.counts")
    add("n_products", c0["products"], str(c0["products"]), src("data_flow", "summary.flow.0.counts.products"))
    add("gb_read", c0["gb_read"], f"{c0['gb_read']} GB", src("data_flow", "summary.flow.0.counts.gb_read"), "about 73 GB read from public services")
    add("gb_read_recorded_transfers", c0["gb_read_recorded_transfers"], f"{c0['gb_read_recorded_transfers']} GB", src("data_flow", "summary.flow.0.counts.gb_read_recorded_transfers"),
        "AORC, GEFS and Stage IV: recorded transfers; the other products count the size of the stored files")
    add("gb_kept", c0["gb_kept_under_data"], f"{c0['gb_kept_under_data']} GB", src("data_flow", "summary.flow.0.counts.gb_kept_under_data"))
    add("bytes_read_total", get(AI, "bytes_read.total"), count(get(AI, "bytes_read.total")) + " B", src("data_flow", "all_information.bytes_read.total"))
    add("bytes_kept_under_data", AI["data_folders_bytes_on_disk"], count(AI["data_folders_bytes_on_disk"]) + " B", src("data_flow", "all_information.data_folders_bytes_on_disk"))
    kinds = get(AI, "bytes_read.kinds")
    for k, v in get(AI, "bytes_read.per_product").items():
        add(f"bytes_read_{k}", v, size_text(v), src("data_flow", f"all_information.bytes_read.per_product.{k}"), f"{v:,} B; {kinds.get(k, '')}".strip("; "))
    add("imagery_note", DF["imagery_note"], DF["imagery_note"], src("data_flow", "imagery_note"), "word for word, as the 3-page paper's Table 1 note")
    add("downloads_words", get(DF, "flow.stages.0.words"), get(DF, "flow.stages.0.words"), src("data_flow", "flow.stages.0.words"))
    add("n_unverified_data_facts", len(DF["unverified"]), str(len(DF["unverified"])), src("data_flow", "unverified (length)"), "list them in Appendix B, never hide them")
    for i, u in enumerate(DF["unverified"]):
        add(f"unverified_fact_{i + 1}", u["fact"], f"[{u['dataset']}] {u['fact']}", src("data_flow", f"unverified.{i}.fact"))
    add("n_documentation_read", len(DF["documentation_read"]), str(len(DF["documentation_read"])), src("data_flow", "documentation_read (length)"),
        "documents read on 2026-10-08 by the data-record agent, not by a team member")
    for ds in DF["datasets"]:
        k = ds["key"]
        for field in ("name", "short", "producer", "version", "spatial_grid", "time_step", "period_used", "time_stamp", "latency_real_time", "role"):
            if field in ds:
                add(f"ds_{k}_{field}", ds[field], ds[field], src("data_flow", f"datasets[key={k}].{field}"))
        ob = ds.get("obtained", {})
        for field in ("url_pattern", "login_needed", "date_downloaded"):
            if field in ob:
                add(f"ds_{k}_{field}", ob[field], ob[field], src("data_flow", f"datasets[key={k}].obtained.{field}"))
        if "bytes_read" in ob:
            add(f"ds_{k}_bytes_read", ob["bytes_read"], size_text(ob["bytes_read"]), src("data_flow", f"datasets[key={k}].obtained.bytes_read"), f"{ob['bytes_read']:,} B")
    for ds in get(DF, "summary.datasets"):
        add(f"ds_{ds['key']}_one_line", ds["one_line"], ds["one_line"], src("data_flow", f"summary.datasets[key={ds['key']}].one_line"))
    add("licence_recorded", False, "no licence field exists in any results file (UNVERIFIED: look up at source before printing a licence name)",
        "results/data_flow.json datasets[*].obtained (only login_needed is recorded); agent7.md section 0 item 2")
    for folder, v in AI["caches"].items():
        k = folder.replace("/", "_")
        add(f"cache_{k}_files", v["files"], count(v["files"]), src("data_flow", f"all_information.caches.{folder}.files"))
        add(f"cache_{k}_bytes", v["bytes_on_disk"], size_text(v["bytes_on_disk"]), src("data_flow", f"all_information.caches.{folder}.bytes_on_disk"), f"{v['bytes_on_disk']:,} B on disk; {v['what']}")
    c2 = get(DF, "summary.flow.2.counts")
    for k, v in c2.items():
        add(f"caches_{k}", v, (f"{v} GB" if k.endswith("gb") else f"{v} MB"), src("data_flow", f"summary.flow.2.counts.{k}"))
    HR = AI["hourly_and_other_records"]
    for k in ("hours_per_basin", "era5_hours_all_basins", "aorc_hours_all_basins", "aorc_filled_basin_hours", "aorc_partial_basin_hours", "gefs_runs", "gefs_windows_per_run",
              "gefs_boxes_in_the_cut", "gefs_values_stored", "gefs_issue_days", "stage4_hourly_labels", "stage4_files_fetched", "stage4_issue_days", "imerg_cells_requested", "power_files_cached"):
        add(f"rec_{k}", HR[k], count(HR[k]), src("data_flow", f"all_information.hourly_and_other_records.{k}"))
    CF = AI["camels_files"]
    add("camels_zip_entries", CF["zip_entries"], count(CF["zip_entries"]), src("data_flow", "all_information.camels_files.zip_entries"))
    add("camels_flow_files", CF["flow_files"], count(CF["flow_files"]), src("data_flow", "all_information.camels_files.flow_files"))
    add("camels_forcing_rows", CF["rows_of_a_forcing_file"], count(CF["rows_of_a_forcing_file"]), src("data_flow", "all_information.camels_files.rows_of_a_forcing_file"))
    add("camels_files_first_day", CF["first"], CF["first"], src("data_flow", "all_information.camels_files.first"))
    add("camels_files_last_day", CF["last"], CF["last"], src("data_flow", "all_information.camels_files.last"))
    add("camels_maurer_last_day", CF["maurer_last"], CF["maurer_last"], src("data_flow", "all_information.camels_files.maurer_last"))

    # ---- basins ---------------------------------------------------------------------------------------------------------------------------
    section("4.1 Basins: selection rule and the 12 (T5, Figure 2, Appendix B)")
    DA = load("data_audit")
    add("basins_all_camels", DA["n_all_camels"], count(DA["n_all_camels"]), src("data_audit", "n_all_camels"))
    add("basins_candidates", DA["n_candidates"], str(DA["n_candidates"]), src("data_audit", "n_candidates"))
    add("basins_complete", DA["n_complete"], str(DA["n_complete"]), src("data_audit", "n_complete"))
    add("basins_with_gaps", DA["n_with_gaps"], str(DA["n_with_gaps"]), src("data_audit", "n_with_gaps"))
    add("basins_chosen", get(AI, "basins.chosen"), str(get(AI, "basins.chosen")), src("data_flow", "all_information.basins.chosen"))
    add("basins_funnel", [DA["n_all_camels"], DA["n_candidates"], DA["n_complete"], len(DA["selected"])],
        f"{DA['n_all_camels']} -> {DA['n_candidates']} -> {DA['n_complete']} -> {len(DA['selected'])}", src("data_audit", "n_all_camels, n_candidates, n_complete, selected (length)"))
    rule = get(AI, "basins.rule")
    add("basins_rule_huc", rule["huc_regions"], "HUC regions 03 and 06", src("data_flow", "all_information.basins.rule.huc_regions"))
    add("basins_rule_max_snow_share", rule["max_snow_share"], pct(rule["max_snow_share"]), src("data_flow", "all_information.basins.rule.max_snow_share"), "snow under this share of precipitation")
    add("basins_rule_max_aridity", rule["max_aridity"], str(rule["max_aridity"]), src("data_flow", "all_information.basins.rule.max_aridity"))
    add("basins_rule_area_km2", rule["area_km2"], f"{rule['area_km2'][0]} to {rule['area_km2'][1]:,} km2", src("data_flow", "all_information.basins.rule.area_km2"))
    add("basins_rule_per_landscape", rule["per_landscape"], str(rule["per_landscape"]), src("data_flow", "all_information.basins.rule.per_landscape"))
    add("basins_rule_text", get(AI, "basins.rule_text"), get(AI, "basins.rule_text"), src("data_flow", "all_information.basins.rule_text"))
    for grp, path in (("candidates", "candidates_by_landscape"), ("complete", "complete_by_landscape"), ("chosen", "chosen_by_landscape")):
        d = get(AI, f"basins.{path}")
        add(f"basins_{grp}_by_landscape", d, "; ".join(f"{k}: {v}" for k, v in d.items()), src("data_flow", f"all_information.basins.{path}"))
    add("n_landscapes", len(get(AI, "basins.chosen_by_landscape")), str(len(get(AI, "basins.chosen_by_landscape"))), src("data_flow", "all_information.basins.chosen_by_landscape (length)"))
    add("landscape_names", list(get(AI, "basins.chosen_by_landscape").keys()), "; ".join(get(AI, "basins.chosen_by_landscape").keys()), src("data_flow", "all_information.basins.chosen_by_landscape (keys)"))
    BG = load("basin_gis")
    areas = {k: v["area_km2"] for k, v in BG["basins"].items()}
    kmin, kmax = min(areas, key=areas.get), max(areas, key=areas.get)
    add("basin_area_min_km2", areas[kmin], f"{areas[kmin]:,.0f} km2", src("basin_gis", f"basins.{kmin}.area_km2"), "South Toe, NC; the project's own geodesic area of the outline")
    add("basin_area_max_km2", areas[kmax], f"{areas[kmax]:,.0f} km2", src("basin_gis", f"basins.{kmax}.area_km2"), "Leaf River, MS")
    add("basin_area_range_km2", [areas[kmin], areas[kmax]], f"{areas[kmin]:,.0f} to {areas[kmax]:,.0f} km2", src("basin_gis", "basins.*.area_km2 (min, max)"))
    for row in AI["basin_table"]:
        g = row["gauge_id"]
        p = f"all_information.basin_table[gauge_id={g}]"
        add(f"basin_{g}_name", row["name"], row["name"], src("data_flow", p + ".name"))
        add(f"basin_{g}_camels_name", row["camels_name"], row["camels_name"], src("data_flow", p + ".camels_name"))
        add(f"basin_{g}_state", row["state"], row["state"], src("data_flow", p + ".state"))
        add(f"basin_{g}_landscape", row["landscape"], row["landscape"], src("data_flow", p + ".landscape"))
        add(f"basin_{g}_area_gages2_km2", row["area_km2_gages2"], f"{row['area_km2_gages2']:,.0f}", src("data_flow", p + ".area_km2_gages2"), "CAMELS GAGES-II area, km2")
        add(f"basin_{g}_area_outline_km2", row["area_km2_outline"], f"{row['area_km2_outline']:,.0f}", src("data_flow", p + ".area_km2_outline"), "the project's geodesic area of the CAMELS outline, km2")
        add(f"basin_{g}_elev_mean_m", row["elev_mean_m"], f"{row['elev_mean_m']:,.0f}", src("data_flow", p + ".elev_mean_m"), "m above sea level (CAMELS)")
        add(f"basin_{g}_mean_rain_mm_day", row["mean_rain_mm_day"], n2(row["mean_rain_mm_day"], 1), src("data_flow", p + ".mean_rain_mm_day"))
        add(f"basin_{g}_mean_flow_mm_day", row["mean_flow_mm_day"], n2(row["mean_flow_mm_day"]), src("data_flow", p + ".mean_flow_mm_day"))
        add(f"basin_{g}_runoff_ratio", row["runoff_ratio"], n2(row["runoff_ratio"]), src("data_flow", p + ".runoff_ratio"))
        add(f"basin_{g}_high_flow_level_mm_day", row["high_flow_level_mm_day"], n2(row["high_flow_level_mm_day"]), src("data_flow", p + ".high_flow_level_mm_day"), "95th percentile of the training years")
        add(f"basin_{g}_estimated_share", row["estimated_share"], pct(row["estimated_share"], 1), src("data_flow", p + ".estimated_share"), "share of the days marked estimated by the USGS")
        add(f"basin_{g}_time_zone", row["time_zone"], row["time_zone"], src("data_flow", p + ".time_zone"))
        add(f"basin_{g}_lat_lon", [row["gauge_lat"], row["gauge_lon"]], f"{row['gauge_lat']:.3f}, {row['gauge_lon']:.3f}", src("data_flow", p + ".gauge_lat/gauge_lon"))
    CW = BG["comparison_with_camels"]
    add("gis_area_vs_gages2_median_pct", CW["area_vs_gages2"]["median_difference"], f"{CW['area_vs_gages2']['median_difference']:+.1f}%", src("basin_gis", "comparison_with_camels.area_vs_gages2.median_difference"), "ours minus CAMELS, % of the CAMELS value")
    add("gis_area_vs_gages2_largest_pct", CW["area_vs_gages2"]["largest_difference"], f"{CW['area_vs_gages2']['largest_difference']:+.1f}%", src("basin_gis", "comparison_with_camels.area_vs_gages2.largest_difference"), CW["area_vs_gages2"]["largest_difference_basin"])
    add("gis_area_vs_fabric_median_pct", CW["area_vs_geospatial_fabric"]["median_difference"], f"{CW['area_vs_geospatial_fabric']['median_difference']:+.2f}%", src("basin_gis", "comparison_with_camels.area_vs_geospatial_fabric.median_difference"))
    add("gis_elev_median_diff_m", CW["elev_mean"]["median_difference"], f"{CW['elev_mean']['median_difference']:+.1f} m", src("basin_gis", "comparison_with_camels.elev_mean.median_difference"))
    add("gis_elev_largest_diff_m", CW["elev_mean"]["largest_difference"], f"{CW['elev_mean']['largest_difference']:+.0f} m", src("basin_gis", "comparison_with_camels.elev_mean.largest_difference"), CW["elev_mean"]["largest_difference_basin"] + "; cause not found")
    add("gis_elev_largest_diff_basin", CW["elev_mean"]["largest_difference_basin"], CW["elev_mean"]["largest_difference_basin"], src("basin_gis", "comparison_with_camels.elev_mean.largest_difference_basin"))
    add("gis_elev_rank_correlation", CW["elev_mean"]["rank_correlation"], n2(CW["elev_mean"]["rank_correlation"]), src("basin_gis", "comparison_with_camels.elev_mean.rank_correlation"))
    add("gis_slope_30m_median_pct", CW["slope_mean"]["median_difference"], f"{CW['slope_mean']['median_difference']:+.0f}%", src("basin_gis", "comparison_with_camels.slope_mean.median_difference"), "slope on the 30 m grid against CAMELS slope_mean: several times steeper; grid-dependent")
    add("gis_slope_rank_correlation", CW["slope_mean"]["rank_correlation"], n2(CW["slope_mean"]["rank_correlation"]), src("basin_gis", "comparison_with_camels.slope_mean.rank_correlation"))
    add("gis_forest_share_mean_diff_points", CW["forest_share"]["mean_difference"], f"{CW['forest_share']['mean_difference']:+.1f} points", src("basin_gis", "comparison_with_camels.forest_share.mean_difference"), "NLCD 2011 forest share minus CAMELS frac_forest, percentage points")
    add("gis_outline_crs", get(BG, "sources.outlines.coordinate_system"), get(BG, "sources.outlines.coordinate_system"), src("basin_gis", "sources.outlines.coordinate_system"))
    add("gis_outline_file", get(BG, "sources.outlines.file"), get(BG, "sources.outlines.file"), src("basin_gis", "sources.outlines.file"))
    add("gis_elevation_product", get(BG, "sources.elevation.product"), get(BG, "sources.elevation.product"), src("basin_gis", "sources.elevation.product"))
    add("gis_elevation_resolution", get(BG, "sources.elevation.resolution"), get(BG, "sources.elevation.resolution"), src("basin_gis", "sources.elevation.resolution"))
    add("gis_elevation_tiles", len(get(BG, "sources.elevation.tiles")), str(len(get(BG, "sources.elevation.tiles"))), src("basin_gis", "sources.elevation.tiles (length)"))
    add("gis_land_cover_product", get(BG, "sources.land_cover.product"), get(BG, "sources.land_cover.product"), src("basin_gis", "sources.land_cover.product"))
    add("gis_land_cover_year", get(BG, "sources.land_cover.year"), str(get(BG, "sources.land_cover.year")), src("basin_gis", "sources.land_cover.year"))
    add("gis_land_cover_resolution", get(BG, "sources.land_cover.resolution"), get(BG, "sources.land_cover.resolution"), src("basin_gis", "sources.land_cover.resolution"))
    add("gis_land_cover_classes", len(get(BG, "sources.land_cover.classes")), str(len(get(BG, "sources.land_cover.classes"))), src("basin_gis", "sources.land_cover.classes (length)"))
    for fig, v in BG["figures"].items():
        add(f"caption_{fig}", v["caption"], v["caption"], src("basin_gis", f"figures.{fig}.caption"))
    GIS = ROOT / "results" / "gis_layers.json"
    if GIS.exists():
        G = json.loads(GIS.read_text(encoding="utf-8"))
        ok = str((G.get("verification") or {}).get("status", "")).upper() == "PASS"
        nb = len(G.get("basins") or {}) or (G.get("summary") or {}).get("n_basins") or 12
        nf = len(G.get("files") or {})
        add("gis_layers_exported", bool(ok), (f"exported ({nb} of 12 basins: outlines, gauges, clipped elevation and land cover; {nf} files in results/gis_layers/, checked against the CAMELS tables)" if ok else "export written but its checks did not pass"),
            "results/gis_layers.json -> verification.status, basins, files (code/27_gis_layers.py, 2026-10-29)", "deliverable D3; the team's own check of the layers is not done")
        add("gis_layers_files", nf, str(nf), "results/gis_layers.json -> files")
        big = [r for r in (G.get("check_table") or []) if r.get("elev_differs_from_camels_by_more_than_25_m")]
        if big:
            dd = sorted(r["elev_vs_camels_m"] for r in big)
            add("gis_elev_basins_over_25m", len(big), {1: "one", 2: "two", 3: "three", 4: "four"}.get(len(big), str(len(big))),
                "results/gis_layers.json -> check_table[*].elev_differs_from_camels_by_more_than_25_m (count)")
            add("gis_elev_basins_over_25m_names", [r["short"] for r in big], ", ".join(r["short"] for r in big), "results/gis_layers.json -> check_table[*].short")
            add("gis_elev_over_25m_range", dd, f"{dd[0]:+.0f} to {dd[-1]:+.0f} m", "results/gis_layers.json -> check_table[*].elev_vs_camels_m (those basins)")
        add("gis_layers_written", G.get("written"), str(G.get("written")), "results/gis_layers.json -> written")
    else:
        add("gis_layers_exported", False, "not exported", "HANDOFF.md section 4 step 3; no .gpkg/.shp/.geojson written anywhere (critic agent9.md section 3)",
            "deliverable D3: statistics and maps exist for 12 of 12 basins; the export as GIS layers was NOT done as of 2026-10-28")

    if (RES / "review_gis_layers.json").exists():                       # the independent AI review of that export: a layer review, so not an experiment record (review_records leaves it out)
        RVG_ALL = load("review_gis_layers")
        gis_rounds = sorted((k for k in RVG_ALL if re.fullmatch(r"round_\d+", k)), key=lambda k: int(k.split("_")[1]))
        RVG = RVG_ALL[gis_rounds[-1]] if gis_rounds else RVG_ALL           # the LATEST round (round 1 is the record's top level)
        gis_where = (gis_rounds[-1] + ".") if gis_rounds else ""
        gis_day = re.search(r"\d{4}-\d{2}-\d{2}", str(opt(RVG, "dates.review") or RVG.get("when", "")))
        add("gis_layers_review_rounds", 1 + len(gis_rounds), str(1 + len(gis_rounds)), src("review_gis_layers", "round_N blocks"))
        add("gis_layers_review_verdict", RVG.get("verdict"), str(RVG.get("verdict")), src("review_gis_layers", gis_where + "verdict"),
            f"{RVG.get('who') or RVG_ALL.get('who')}: the latest round's verdict, on the export written {opt(RVG, 'dates.export_run', 'n/a')}; the team's own check of the layers is not done")
        add("gis_layers_review_date", gis_day.group(0) if gis_day else None, long_date(gis_day.group(0)) if gis_day else "n/a", src("review_gis_layers", gis_where + "dates.review"), RVG.get("when") or RVG_ALL.get("when"))

    # ---- records, split, levels, clocks ---------------------------------------------------------------------------------------------------
    section("4.2 Records, split, high-flow level, onsets and day ends (Figure 3)")
    c3 = get(DF, "summary.flow.3.counts")
    add("n_basins", c3["basins"], str(c3["basins"]), src("data_flow", "summary.flow.3.counts.basins"))
    add("days_per_basin", c3["days_per_basin"], count(c3["days_per_basin"]), src("data_flow", "summary.flow.3.counts.days_per_basin"))
    add("basin_days", c3["basin_days"], count(c3["basin_days"]), src("data_flow", "summary.flow.3.counts.basin_days"))
    add("weather_columns", c3["weather_columns"], str(c3["weather_columns"]), src("data_flow", "summary.flow.3.counts.weather_columns"))
    add("flow_columns", c3["flow_columns"], str(c3["flow_columns"]), src("data_flow", "summary.flow.3.counts.flow_columns"))
    add("n_properties", c3["properties"], str(c3["properties"]), src("data_flow", "summary.flow.3.counts.properties"))
    add("missing_values", c3["missing_values"], str(c3["missing_values"]), src("data_flow", "summary.flow.3.counts.missing_values"))
    add("extra_columns_of_the_experiments", c3["extra_columns_of_the_experiments"], str(c3["extra_columns_of_the_experiments"]), src("data_flow", "summary.flow.3.counts.extra_columns_of_the_experiments"))
    R_ = AI["records"]
    add("values_weather_and_flow", R_["values_weather_and_flow"], count(R_["values_weather_and_flow"]), src("data_flow", "all_information.records.values_weather_and_flow"))
    add("estimated_flow_days", R_["estimated_flow_days"], count(R_["estimated_flow_days"]), src("data_flow", "all_information.records.estimated_flow_days"), "days marked estimated by the USGS; kept")
    add("estimated_share_all", R_["estimated_share_of_all_basin_days"], pct(R_["estimated_share_of_all_basin_days"], 1), src("data_flow", "all_information.records.estimated_share_of_all_basin_days"))
    add("estimated_share_min", R_["estimated_share_per_basin_min"], pct(R_["estimated_share_per_basin_min"], 1), src("data_flow", "all_information.records.estimated_share_per_basin_min"))
    add("estimated_share_median", R_["estimated_share_per_basin_median"], pct(R_["estimated_share_per_basin_median"], 1), src("data_flow", "all_information.records.estimated_share_per_basin_median"))
    add("estimated_share_max", R_["estimated_share_per_basin_max"], pct(R_["estimated_share_per_basin_max"], 1), src("data_flow", "all_information.records.estimated_share_per_basin_max"))
    add("zero_flow_days", R_["zero_flow_days"], count(R_["zero_flow_days"]), src("data_flow", "all_information.records.zero_flow_days"))
    add("zero_flow_share", R_["zero_flow_share_of_all_basin_days"], pct(R_["zero_flow_share_of_all_basin_days"], 1), src("data_flow", "all_information.records.zero_flow_share_of_all_basin_days"))
    add("flow_conversion_largest_difference", R_["flow_conversion_largest_difference_from_cache"], str(R_["flow_conversion_largest_difference_from_cache"]), src("data_flow", "all_information.records.flow_conversion_largest_difference_from_cache"), "recomputed from the raw flow files for all basin-days")
    P = AI["period"]
    add("period_first_day", P["first"], P["first"], src("data_flow", "all_information.period.first"))
    add("period_last_day", P["last"], P["last"], src("data_flow", "all_information.period.last"))
    add("period_years_text", [P["first"], P["last"]], f"{P['first'][:4]}-{P['last'][:4]}", src("data_flow", "all_information.period.first/last"), "1980-2014 as the documents write it")
    add("water_years", P["water_years"], str(P["water_years"]), src("data_flow", "all_information.period.water_years"))
    SP = get(AI, "split.main")
    for part in ("train", "val", "test"):
        fl = SP["first_last"][part]
        add(f"split_{part}_dates", fl, f"{fl[0]} to {fl[1]}", src("data_flow", f"all_information.split.main.first_last.{part}"))
        add(f"split_{part}_years_text", fl, f"{fl[0][:4]}-{fl[1][:4]}", src("data_flow", f"all_information.split.main.first_last.{part}"), "water years, as the documents write them")
        add(f"split_{part}_days", SP["days"][part], count(SP["days"][part]), src("data_flow", f"all_information.split.main.days.{part}"))
        add(f"split_{part}_water_years", SP["water_years"][part], str(SP["water_years"][part]), src("data_flow", f"all_information.split.main.water_years.{part}"))
    SS = get(AI, "split.satellite_era")
    for part in ("train", "val", "test"):
        fl = SS["first_last"][part]
        add(f"split_sat_{part}_dates", fl, f"{fl[0]} to {fl[1]}", src("data_flow", f"all_information.split.satellite_era.first_last.{part}"), "the satellite-era split of 08_satellite_inputs.py and E2b: its scores are comparable only inside it")
        add(f"split_sat_{part}_days", SS["days"][part], count(SS["days"][part]), src("data_flow", f"all_information.split.satellite_era.days.{part}"))
    HF = AI["high_flow"]
    add("high_flow_percentile", HF["percentile"], f"{HF['percentile']:.0f}th", src("data_flow", "all_information.high_flow.percentile"), "of the daily flow of the training years, per basin")
    lv = list(HF["level_mm_per_day"].values())
    add("high_flow_level_min_mm_day", min(lv), n2(min(lv)), src("data_flow", "all_information.high_flow.level_mm_per_day (min)"))
    add("high_flow_level_max_mm_day", max(lv), n2(max(lv)), src("data_flow", "all_information.high_flow.level_mm_per_day (max)"))
    add("high_flow_level_range_mm_day", [min(lv), max(lv)], f"{n2(min(lv))} to {n2(max(lv))} mm/day", src("data_flow", "all_information.high_flow.level_mm_per_day (min, max)"))
    add("onsets_l1", HF["onsets_in_the_test_years"], count(HF["onsets_in_the_test_years"]), src("data_flow", "all_information.high_flow.onsets_in_the_test_years"), "flood onsets of the test years, 1 day ahead")
    KF = load("key_figure")
    add("onsets_l2", get(KF, "n_onsets.l2"), count(get(KF, "n_onsets.l2")), src("key_figure", "n_onsets.l2"), "onsets scored 2 days ahead")
    add("onsets_l3", get(KF, "n_onsets.l3"), count(get(KF, "n_onsets.l3")), src("key_figure", "n_onsets.l3"), "onsets scored 3 days ahead")
    add("events_test_years", HF["events_in_the_test_years"], count(HF["events_in_the_test_years"]), src("data_flow", "all_information.high_flow.events_in_the_test_years"))
    add("events_one_day", HF["events_lasting_one_day"], count(HF["events_lasting_one_day"]), src("data_flow", "all_information.high_flow.events_lasting_one_day"))
    add("events_median_days", HF["median_event_days"], f"{HF['median_event_days']:.0f}", src("data_flow", "all_information.high_flow.median_event_days"))
    for k, v in HF["by_storm_type"].items():
        add(f"events_{k.replace('-', '_')}", v, str(v), src("data_flow", f"all_information.high_flow.by_storm_type.{k}"))
    UN = load("uncertainty")
    add("high_flow_days_test", UN["n_high_flow_days_test"], count(UN["n_high_flow_days_test"]), src("uncertainty", "n_high_flow_days_test"), "high-flow days of the test years, 1 day ahead")
    later = UN["n_high_flow_days_test"] - HF["onsets_in_the_test_years"]
    add("high_flow_days_second_or_later", later, count(later), src("uncertainty", "n_high_flow_days_test") + " minus " + src("data_flow", "all_information.high_flow.onsets_in_the_test_years"))
    add("high_flow_days_second_or_later_share", later / UN["n_high_flow_days_test"], pct(later / UN["n_high_flow_days_test"]), "computed: high_flow_days_second_or_later / high_flow_days_test")
    add("n_test_days_per_basin", SP["days"]["test"], count(SP["days"]["test"]), src("data_flow", "all_information.split.main.days.test"))
    FA = load("flood_analysis")
    for k, v in get(FA, "events.by_season").items():
        add(f"events_season_{k.replace('-', '_').lower()}", v, str(v), src("flood_analysis", f"events.by_season.{k}"))
    add("events_tropical_within_300km", get(FA, "events.tropical_within_300km"), str(get(FA, "events.tropical_within_300km")), src("flood_analysis", "events.tropical_within_300km"))
    add("hurdat_storms_in_file", get(FA, "events.hurdat.n_storms_in_file"), count(get(FA, "events.hurdat.n_storms_in_file")), src("flood_analysis", "events.hurdat.n_storms_in_file"))
    add("hurdat_radius_km", get(FA, "events.hurdat.radius_km"), f"{get(FA, 'events.hurdat.radius_km')} km", src("flood_analysis", "events.hurdat.radius_km"))
    add("findings_events_sentence", FA["findings"][0], FA["findings"][0], src("flood_analysis", "findings.0"))
    DW = load("day_windows")
    CL = get(AI, "clocks.day_ends")
    add("day_end_daymet", CL["daymet"], CL["daymet"], src("data_flow", "all_information.clocks.day_ends.daymet"), "measured median, local standard time")
    add("day_end_daymet_rounded", get(DW, "day_ends.daymet.day_ends_local_rounded"), "about " + get(DW, "day_ends.daymet.day_ends_local_rounded"), src("day_windows", "day_ends.daymet.day_ends_local_rounded"), "the documents say 'about 09:00'")
    rng = get(DW, "day_ends.daymet.basin_range_local")
    add("day_end_daymet_basin_range", rng, f"{rng[0]} to {rng[1]}", src("day_windows", "day_ends.daymet.basin_range_local"))
    add("day_end_daymet_misfit_basins", get(DW, "day_ends.daymet.basins_that_do_not_fit"), ", ".join(get(DW, "day_ends.daymet.basins_that_do_not_fit")), src("day_windows", "day_ends.daymet.basins_that_do_not_fit"))
    add("day_end_nldas", CL["nldas"], CL["nldas"], src("data_flow", "all_information.clocks.day_ends.nldas"),
        "measured median 17:40; the documents round it to 'about 17:30' (write 'about 17:30 (measured median 17:40)' if precision is asked: critic C8)")
    add("day_end_nldas_rounded", get(DW, "day_ends.nldas.day_ends_local_rounded"), "about " + get(DW, "day_ends.nldas.day_ends_local_rounded"), src("day_windows", "day_ends.nldas.day_ends_local_rounded"))
    rng = get(DW, "day_ends.nldas.basin_range_local")
    add("day_end_nldas_basin_range", rng, f"{rng[0]} to {rng[1]}", src("day_windows", "day_ends.nldas.basin_range_local"))
    add("day_end_maurer", CL["maurer"], CL["maurer"], src("data_flow", "all_information.clocks.day_ends.maurer"))
    add("day_end_imerg_documented", CL["imerg_documented"], CL["imerg_documented"], src("data_flow", "all_information.clocks.day_ends.imerg_documented"), "the UTC day, local standard time")
    add("day_end_river_flow", CL["river_flow"], CL["river_flow"], src("data_flow", "all_information.clocks.day_ends.river_flow"))
    for k in ("flow_mm_per_day", "basin_mean_with_overlap_weights", "local_day_sum", "scaled_inputs", "nse", "high_flow_level_and_onset", "training_loss_and_stopping", "forecast_rule", "gefs_bias_factor_and_issue_time"):
        hc = get(DF, f"how_computed.{k}")
        if "formula" in hc:
            add(f"formula_{k}", hc["formula"], hc["formula"], src("data_flow", f"how_computed.{k}.formula"))
        add(f"words_{k}", hc["words"], hc["words"], src("data_flow", f"how_computed.{k}.words"))
    gz = get(DF, "how_computed.local_day_sum.gauges_by_zone")
    add("gauges_by_zone", gz, "; ".join(f"{k}: {v}" for k, v in gz.items()), src("data_flow", "how_computed.local_day_sum.gauges_by_zone"))
    # the same mapping as a sentence (critic 1, item 17): the zone most gauges keep, then the exception with its basin named in the method's own words
    WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    zones = sorted(gz.items(), key=lambda kv: -kv[1])
    if len(zones) == 2 and all(n < len(WORDS) for _, n in zones):
        (main_zone, main_n), (other_zone, other_n) = zones
        m = re.search(re.escape(other_zone) + r" for ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)", get(DF, "how_computed.local_day_sum.words"))
        sentence = f"{WORDS[main_n]} gauges keep {main_zone} time and {WORDS[other_n]}{' (' + m.group(1) + ')' if m else ''} {other_zone}"
    else:
        sentence = "; ".join(f"{k}: {v}" for k, v in gz.items())
    add("gauges_by_zone_sentence", gz, sentence, src("data_flow", "how_computed.local_day_sum.gauges_by_zone and .words (the basin of the exception)"),
        "the mapping gauges_by_zone written as a sentence for the prose (critic 1, item 17)")
    SPr = AI["static_properties"]
    add("static_properties_names", [x["name"] for x in SPr["items"]], ", ".join(x["name"] for x in SPr["items"]), src("data_flow", "all_information.static_properties.items[*].name"))
    add("static_climate_properties", SPr["climate_properties"], ", ".join(SPr["climate_properties"]), src("data_flow", "all_information.static_properties.climate_properties"), SPr["note"])
    add("static_climate_period", SPr["climate_averaging_period"], f"{SPr['climate_averaging_period'][0]} to {SPr['climate_averaging_period'][1]}", src("data_flow", "all_information.static_properties.climate_averaging_period"),
        "overlaps the validation and test years: disclose (report section 4.1)")
    add("scalers_fitted_on", AI["scalers"]["fitted_on"], AI["scalers"]["fitted_on"], src("data_flow", "all_information.scalers.fitted_on"))


# ====================================================================================================================================
def build_model_and_checks(MC, ER):
    """5.2 model settings, 5.4 simple models, 4.2/5.3 fair test, 6.1 engine review."""
    section("5.2 Model settings: LSTM, Transformer, simple models, one-river tuning (Appendix C)")
    LA = load("lstm_all")
    for k, v in LA["config"].items():
        add(f"config_{k}", v, str(v), src("lstm_all", f"config.{k}"), "chosen on the validation years by 04_lstm_all.py; used by scripts 10, 15-18")
    DF = load("data_flow")
    c7 = get(DF, "summary.flow.6.counts")
    add("weights_lstm_wq", c7["weights"], count(c7["weights"]), src("data_flow", "summary.flow.6.counts.weights"), "the LSTM with the river gauge: 4 x 64 x (17 + 64 + 2) + 195")
    add("inputs_per_day", c7["inputs"], str(c7["inputs"]), src("data_flow", "summary.flow.6.counts.inputs"), "6 weather + 1 flow + 10 basin properties")
    add("memory_cells", c7["memory_cells"], str(c7["memory_cells"]), src("data_flow", "summary.flow.6.counts.memory_cells"))
    add("sequence_days", c7["sequence_days"], str(c7["sequence_days"]), src("data_flow", "summary.flow.6.counts.sequence_days"), "365 days of history + 365 scored")
    add("sequences_per_update", c7["sequences_per_update"], str(c7["sequences_per_update"]), src("data_flow", "summary.flow.6.counts.sequences_per_update"))
    add("seeds_per_set", c7["seeds"], str(c7["seeds"]), src("data_flow", "summary.flow.6.counts.seeds"),
        "the value of the data-flow record of 8 October (three trainings per set); since 30 October 2026 the sets of scripts 15-18 and 20-22 have five (seeds 3-4 added on the Windows PC), "
        "the stored sets of scripts 02-10 keep three (stored_sets_three_seeds_list); quote the n_seeds key of the set")
    add("seeds_list", get(LA, "sets.wq.seeds"), ", ".join(str(s) for s in get(LA, "sets.wq.seeds")), src("lstm_all", "sets.wq.seeds"))
    for k, v in get(DF, "all_information.model.weights_by_input_set_of_script_15").items():
        add(f"weights_lstm_{k}", v, count(v), src("data_flow", f"all_information.model.weights_by_input_set_of_script_15.{k}"))
    add("settings_chosen_on", get(DF, "all_information.model.chosen_on"), get(DF, "all_information.model.chosen_on"), src("data_flow", "all_information.model.chosen_on"))
    tab = LA["setting_check"]["table"]
    for i, row in enumerate(tab):
        add(f"setting_check_{i + 1}", row, f"hidden {row['hidden']}, lr {row['lr']}: validation NSE {n2(row['val_nse_mean'], 4)}" + (" (chosen)" if row["chosen"] else ""),
            src("lstm_all", f"setting_check.table.{i}"), LA["setting_check"]["criterion"])
    secs = [get(LA, f"sets.wq.cost.{s}.seconds") for s in ("0", "1", "2")]
    add("training_seconds_lstm_wq_min", min(secs), f"{min(secs):.0f} s", src("lstm_all", "sets.wq.cost.*.seconds (min)"))
    add("training_seconds_lstm_wq_max", max(secs), f"{max(secs):.0f} s", src("lstm_all", "sets.wq.cost.*.seconds (max)"))
    add("training_seconds_lstm_wq_range", secs, f"{min(secs):.0f} to {max(secs):.0f} s", src("lstm_all", "sets.wq.cost.*.seconds"), "one training of the LSTM with the river gauge on the Mac (two cores shared with other jobs)")
    add("training_seconds_lstm_wq_median", median3(secs), f"{median3(secs):.0f} s", src("lstm_all", "sets.wq.cost.*.seconds (median)"))
    ups = [get(LA, f"sets.wq.cost.{s}.best_update") for s in ("0", "1", "2")]
    add("best_update_lstm_wq_range", ups, f"{min(ups)} to {max(ups)}", src("lstm_all", "sets.wq.cost.*.best_update"), "the update at which the kept state was saved")
    # Transformer
    for k, v in get(MC, "settings.transformer").items():
        add(f"transformer_{k}", v, str(v), src("model_comparison", f"settings.transformer.{k}"))
    add("transformer_weights", get(MC, "models.transformer.n_params"), count(get(MC, "models.transformer.n_params")), src("model_comparison", "models.transformer.n_params"))
    add("cmp_lstm_weights", get(MC, "models.lstm.n_params"), count(get(MC, "models.lstm.n_params")), src("model_comparison", "models.lstm.n_params"))
    add("cmp_sequence_days", get(MC, "settings.sequence_days"), str(get(MC, "settings.sequence_days")), src("model_comparison", "settings.sequence_days"),
        "the comparison trained its own two-seed LSTM twin with shorter sequences (180 + 185 days, 1,000 updates), not the 730-day LSTM of script 04 (critic C9)")
    add("cmp_seeds", MC["seeds"], str(len(MC["seeds"])), src("model_comparison", "seeds"), "two seeds per model")
    for m_ in ("lstm", "transformer"):
        add(f"cmp_{m_}_seconds_per_model", get(MC, f"cost_of_skill.{m_}.seconds_per_model"), f"{get(MC, f'cost_of_skill.{m_}.seconds_per_model'):.0f} s", src("model_comparison", f"cost_of_skill.{m_}.seconds_per_model"))
        add(f"cmp_{m_}_seconds_per_point", get(MC, f"cost_of_skill.{m_}.seconds_per_point"), f"{get(MC, f'cost_of_skill.{m_}.seconds_per_point'):.1f} s", src("model_comparison", f"cost_of_skill.{m_}.seconds_per_point"), "seconds of training per 0.01 NSE of test skill")
        add(f"cmp_{m_}_energy_wh_per_model", get(MC, f"cost_of_skill.{m_}.energy_wh_per_model_assumed"), f"{get(MC, f'cost_of_skill.{m_}.energy_wh_per_model_assumed'):.2f} Wh", src("model_comparison", f"cost_of_skill.{m_}.energy_wh_per_model_assumed"), "at an ASSUMED 15 W, not measured")
        ops = get(MC, f"models.{m_}.operations.per_update_forward")
        add(f"cmp_{m_}_ops_per_update", ops, f"{ops / 1e6:,.0f} million" if ops < 1e9 else f"{ops / 1e9:.2f} billion", src("model_comparison", f"models.{m_}.operations.per_update_forward"), "multiply-adds of one forward pass of one update")
        per = get(MC, f"models.{m_}.per_seed")
        sec = [v["cost"]["seconds"] for v in per.values()]
        add(f"cmp_{m_}_seconds_per_seed_range", sec, f"{min(sec):.0f} to {max(sec):.0f} s", src("model_comparison", f"models.{m_}.per_seed.*.cost.seconds"))
    for k in ("parameters", "operations_per_update", "seconds_per_update", "seconds_per_model", "seconds_per_point"):
        v = get(MC, f"cost_of_skill.transformer_over_lstm.{k}")
        add(f"cmp_ratio_{k}", v, f"{v:.1f}x", src("model_comparison", f"cost_of_skill.transformer_over_lstm.{k}"), "Transformer over LSTM")
    add("cmp_assumed_power_watt", get(MC, "settings.assumed_power_watt"), f"{get(MC, 'settings.assumed_power_watt'):.0f} W", src("model_comparison", "settings.assumed_power_watt"), "an assumption, not a measurement")
    # simple models
    BL = load("baselines")
    for s in ("w", "wq", "wf", "wqf"):
        add(f"features_{s}", len(BL["features"][s]), str(len(BL["features"][s])), src("baselines", f"features.{s} (length)"), "features of ridge regression and the random forest for this input set")
    rf = get(BL, "models.base_rf_wq.settings")
    add("rf_n_trees", rf["n_trees"], str(rf["n_trees"]), src("baselines", "models.base_rf_wq.settings.n_trees"))
    add("rf_min_samples_leaf", rf["min_samples_leaf"], str(rf["min_samples_leaf"]), src("baselines", "models.base_rf_wq.settings.min_samples_leaf"), "chosen on the validation years")
    add("rf_max_features", rf["max_features"], str(rf["max_features"]), src("baselines", "models.base_rf_wq.settings.max_features"))
    rg = get(BL, "models.base_linreg_wq.settings")
    add("ridge_alpha", rg["ridge_alpha"], str(rg["ridge_alpha"]), src("baselines", "models.base_linreg_wq.settings.ridge_alpha"), "chosen on the validation years; standardised features")
    add("baselines_fit_seconds", BL["fit_seconds"], f"{BL['fit_seconds']:.0f} s", src("baselines", "fit_seconds"), "all simple models together")
    add("baselines_note_fit", BL["notes"][0], BL["notes"][0], src("baselines", "notes.0"))
    # one-river tuning
    LS = load("lstm_single")
    add("tuning_river", get(LS, "basin.short"), get(LS, "basin.short"), src("lstm_single", "basin.short"))
    rows = list(csv.DictReader((RES / "hparam_log.csv").open(encoding="utf-8")))
    add("tuning_runs", len(rows), str(len(rows)), "results/hparam_log.csv (rows)", "one-river trainings logged by 03_lstm.py; 193 trainings in all = 167 recorded in the results files + these 26")
    for k, v in LS["tuning"]["chosen"].items():
        add(f"tuning_chosen_{k}", v, str(v), src("lstm_single", f"tuning.chosen.{k}"))
    add("tuning_reason", LS["tuning"]["reason"], LS["tuning"]["reason"], src("lstm_single", "tuning.reason"))
    add("tuning_seed_spread", LS["tuning"]["noise_seed_spread"], n2(LS["tuning"]["noise_seed_spread"], 3), src("lstm_single", "tuning.noise_seed_spread"), "the seed spread of the validation NSE, the tolerance of the tuning")
    for s in ("w", "wq"):
        for ctx in ("2", "30", "365"):
            v = get(LS, f"memory.test_nse_by_lead.{s}.{ctx}")[0]
            add(f"memory_nse_l1_{s}_{ctx}_days", v, n2(v), src("lstm_single", f"memory.test_nse_by_lead.{s}.{ctx}.0"), f"one-river LSTM ({s}), 1 day ahead, given {ctx} days of history (seed-0 model)")
    add("training_curves_caption", load("training_curves")["caption"], load("training_curves")["caption"], src("training_curves", "caption"))

    # ---- fair test and engine review -------------------------------------------------------------------------------------------------------
    section("5.3 Fair test, leak controls and the engine review (T8 O2, 6.1)")
    FT = load("fair_test")
    add("fair_test_rule", FT["rule"], FT["rule"], src("fair_test", "rule"), "made-up rivers with a known answer: the only simulated data of the project")
    add("fair_test_expected_past_weather", FT["expected"]["past_weather_only"], ", ".join(n2(x) for x in FT["expected"]["past_weather_only"]), src("fair_test", "expected.past_weather_only"), "NSE at 1, 2, 3 days ahead")
    add("fair_test_measured_past_weather", FT["measured"]["past_weather_only"], ", ".join(n2(x, 4) for x in FT["measured"]["past_weather_only"]), src("fair_test", "measured.past_weather_only"))
    add("fair_test_expected_with_forecast", FT["expected"]["with_target_day_weather"], ", ".join(n2(x) for x in FT["expected"]["with_target_day_weather"]), src("fair_test", "expected.with_target_day_weather"))
    add("fair_test_measured_with_forecast", FT["measured"]["with_target_day_weather"], ", ".join(n2(x, 4) for x in FT["measured"]["with_target_day_weather"]), src("fair_test", "measured.with_target_day_weather"))
    add("fair_test_n_checks", len(FT["checks"]), str(len(FT["checks"])), src("fair_test", "checks (length)"))
    add("fair_test_passed", FT["passed"], "passed" if FT["passed"] else "FAILED", src("fair_test", "passed"))
    for i, c in enumerate(FT["checks"]):
        add(f"fair_test_check_{i + 1}", c, c, src("fair_test", f"checks.{i}"))
    for k, v in ER["counts"].items():
        add(f"engine_review_{k}", v, str(v), src("engine_review", f"counts.{k}"))
    add("engine_review_n_reviewers", len(ER["reviews"]), str(len(ER["reviews"])), src("engine_review", "reviews (length)"), ER["who"])
    add("engine_review_who", ER["who"], ER["who"], src("engine_review", "who"))
    add("engine_review_all_fixed", ER["all_fixed"], "all fixed" if ER["all_fixed"] else "not all fixed", src("engine_review", "all_fixed"), ER["all_fixed_note"])
    add("engine_review_effect", ER["effect_on_results"], ER["effect_on_results"], src("engine_review", "effect_on_results"))
    add("engine_review_human_review", ER["human_review"], ER["human_review"], src("engine_review", "human_review"), "the team's own check of the engine is not done")
    add("engine_review_fixed_before", ER["fixed_before"], ER["fixed_before"], src("engine_review", "fixed_before"))
    for i, f_ in enumerate(ER["major_findings"]):
        add(f"engine_review_major_{i + 1}", f_, f_, src("engine_review", f"major_findings.{i}"))
    for i, rv in enumerate(ER["reviews"]):
        add(f"engine_review_lens_{i + 1}", rv["lens"], f"{rv['lens']}: {rv['verdict'].replace('_', ' ')}, {rv['n_checks_passed']} checks passed", src("engine_review", f"reviews.{i}"))
    add("engine_review_note_scripts", ER["notes"][1], ER["notes"][1], src("engine_review", "notes.1"),
        "no per-script review record exists for scripts 01-14 (only this sentence); the four experiments 15-18 have results/review_*.json")
    add("engine_review_late_additions", ER["late_additions"], ER["late_additions"], src("engine_review", "late_additions"))
    add("reload_check_e3_bit_for_bit", get(load("more_basins"), "reload_check.all_bit_for_bit"), "bit for bit" if get(load("more_basins"), "reload_check.all_bit_for_bit") else "not bit for bit",
        src("more_basins", "reload_check.all_bit_for_bit"), "the 15 saved networks of E3 rebuilt from their .pt files on the Mac")


# ====================================================================================================================================
# The prediction sets: where each one's test-year scores, seed ranges and verdicts are stored.
SET_FAMILIES = [
    # (results file, summary path pattern, seed-range path pattern, verdict path pattern, [(set key in the file, prediction-set name)])
    ("lstm_all", "sets.{k}.test.summary", None, None, [("w", "lstm_w"), ("wq", "lstm_wq"), ("wf", "lstm_wf"), ("wqf", "lstm_wqf")]),
    ("rain_timing", "models.{k}.test.summary", None, None, [("wn", "lstm_wn"), ("wqn", "lstm_wqn"), ("wn_older", "lstm_wn_older"), ("wqn_older", "lstm_wqn_older")]),
    ("rain_to_midnight", "summary_test.sets.{k}", "seed_ranges.sets.{k}", "verdicts.sets.{k}",
     [("wqh", "lstm_wqh"), ("wqh_older", "lstm_wqh_older"), ("wh", "lstm_wh"), ("wh_older", "lstm_wh_older"), ("wqh_to09", "lstm_wqh_to09"), ("wqh_utc", "lstm_wqh_utc"), ("wqnh", "lstm_wqnh")]),
    ("aorc_rain_to_midnight", "summary_test.sets.{k}", "seed_ranges.sets.{k}", "verdicts.sets.{k}",
     [("wqa", "lstm_wqa"), ("wqa_to09", "lstm_wqa_to09"), ("wqa_utc", "lstm_wqa_utc"), ("wqa_rep", "lstm_wqa_rep")]),
    ("forecast_input", "summary_test.sets.{k}", "seed_ranges.sets.{k}", "verdicts.sets.{k}",
     [("wqaF_sub", "lstm_wqaF_sub"), ("wqaF_clim", "lstm_wqaF_clim"), ("wqaP", "lstm_wqaP"), ("wqF_sub", "lstm_wqF_sub"), ("wqaF_sub_raw", "lstm_wqaF_sub_raw"),
      ("wqaF_s", "lstm_wqaF_s"), ("wqa_s", "lstm_wqa_s"), ("wqaF_clim_s", "lstm_wqaF_clim_s"), ("wq_s", "lstm_wq_s"), ("wqaP_s", "lstm_wqaP_s")]),
    ("more_basins", "summary_test.sets.{k}", "seed_ranges.sets.{k}", "verdicts.sets.{k}",
     [("lstm63_wq", "lstm63_wq"), ("lstm63_w", "lstm63_w"), ("lstm63eq_wq", "lstm63eq_wq"), ("unseen50_w_static", "unseen50_w_static"), ("unseen50_w_nostatic", "unseen50_w_nostatic"),
      ("lobo_w_static", "lobo_w_static"), ("lobo_w_nostatic", "lobo_w_nostatic"), ("lobo_ref_w_nostatic", "lobo_ref_w_nostatic"), ("lobo_rf_w", "lobo_rf_w")]),
]
SET_LABELS = {
    "lstm_w": "LSTM, past weather only", "lstm_wq": "LSTM, past weather + river gauge (the rain table alone)", "lstm_wf": "LSTM, past weather + perfect weather forecast (upper bound)",
    "lstm_wqf": "LSTM, river gauge + perfect weather forecast (upper bound)", "lstm_wn": "LSTM, past weather + NLDAS rain to the evening", "lstm_wqn": "LSTM, gauge + NLDAS rain to the evening (about 17:30)",
    "lstm_wn_older": "control: NLDAS rain one day older, no gauge", "lstm_wqn_older": "control: NLDAS rain one day older, with gauge",
    "lstm_wqh": "E1: gauge + ERA5 rain to local midnight", "lstm_wqh_older": "E1 control: the same ERA5 rain one day older", "lstm_wh": "E1: ERA5 rain to midnight, no gauge",
    "lstm_wh_older": "E1 control: ERA5 one day older, no gauge", "lstm_wqh_to09": "E1 control: ERA5 over the rain table's window (09:00 to 09:00)", "lstm_wqh_utc": "E1 control: ERA5 over the UTC day",
    "lstm_wqnh": "E1: NLDAS rain of the issue day + ERA5 rain to midnight", "lstm_wqa": "E1b: gauge + AORC rain to local midnight (an analysis made after the fact)",
    "lstm_wqa_to09": "E1b control: the same AORC rain over the table's own window (09:00 to 09:00)", "lstm_wqa_utc": "E1b control: AORC over the UTC day", "lstm_wqa_rep": "E1b: Daymet rain column replaced by the AORC day",
    "lstm_wqaF_sub": "E2a: wqa + GEFS forecast of days t+1..t+3 substituted at forecast time", "lstm_wqaF_clim": "E2a control: shuffled forecast (measures harm, not information)",
    "lstm_wqaP": "E2a: perfect rain of the three days (upper bound)", "lstm_wqF_sub": "E2a: the forecast on the plain gauge model, no rain to midnight", "lstm_wqaF_sub_raw": "E2a: raw (not bias-corrected) forecast substituted",
    "lstm_wqaF_s": "E2b: wqa + forecast, retrained on the satellite-era split (ANOTHER SPLIT)", "lstm_wqa_s": "E2b: wqa retrained on the satellite-era split (ANOTHER SPLIT)",
    "lstm_wqaF_clim_s": "E2b control: shuffled forecast, retrained (ANOTHER SPLIT)", "lstm_wq_s": "E2b: rain table + gauge on the satellite-era split (ANOTHER SPLIT)", "lstm_wqaP_s": "E2b: perfect rain of the three days (upper bound, ANOTHER SPLIT)",
    "lstm63_wq": "E3: one LSTM trained on 63 rivers, with gauge, scored on the 12", "lstm63_w": "E3: trained on 63 rivers, no gauge", "lstm63eq_wq": "E3: 63 rivers with 5x the update limit and patience window",
    "unseen50_w_static": "E3: trained on the 50 other rivers, weather + properties: the 12 never seen", "unseen50_w_nostatic": "E3: trained on the 50 other rivers, weather only: the 12 never seen",
    "lobo_w_static": "05: leave-one-basin-out, weather + properties (one training per river)", "lobo_w_nostatic": "05: leave-one-basin-out, weather only", "lobo_ref_w_nostatic": "05: 12-river reference, weather only, one training",
    "lobo_rf_w": "05: leave-one-basin-out random forest",
}
ANOTHER_SPLIT = {"lstm_wqaF_s", "lstm_wqa_s", "lstm_wqaF_clim_s", "lstm_wq_s", "lstm_wqaP_s"}
SET_SUFFIXES = ["nse", "nse_range", "onset_pod", "onset_pod_range", "onset_far", "onset_far_range", "onset_csi", "onset_hits", "onset_misses", "onset_false_alarms",
                "all_pod", "all_far", "all_csi", "all_hits", "all_misses", "all_false_alarms", "n_seeds", "onset_verdict", "all_verdict", "chosen_trigger", "onset_verdict_chosen", "all_verdict_chosen"]
SET_TABLE = []       # rows for NUMBERS_KEYS.md


def build_sets():
    section("7.1 Test-year scores of every prediction set and lead (T7, the ladder, 7.3)")
    RM = load("rain_to_midnight")
    LA = load("lstm_all")
    for fname, spath, seedpath, vpath, sets in SET_FAMILIES:
        D = load(fname)
        for k, name in sets:
            split_note = "satellite-era split (test years 2010-2014; onsets 335/540/656 at 1/2/3 days): comparable only inside that split" if name in ANOTHER_SPLIT else None
            for L in LEADS:
                S = get(D, spath.format(k=k) + f".{L}")
                base = f"{name}_{L}".lower()
                note = SET_LABELS.get(name, name) + ("; " + split_note if split_note else "")
                p = spath.format(k=k) + f".{L}"
                add(f"{base}_nse", S["nse_median"], n2(S["nse_median"]), src(fname, p + ".nse_median"), note)
                for kind in ("onset", "all"):
                    T = S[kind]
                    add(f"{base}_{kind}_pod", T["pod"], pct(T["pod"]), src(fname, f"{p}.{kind}.pod"), "share warned (POD)")
                    add(f"{base}_{kind}_far", T["far"], pct(T["far"]), src(fname, f"{p}.{kind}.far"), "share of the warnings that were false (FAR)")
                    add(f"{base}_{kind}_csi", T["csi"], n2(T["csi"]), src(fname, f"{p}.{kind}.csi"))
                    add(f"{base}_{kind}_hits", T["hits"], count(T["hits"]), src(fname, f"{p}.{kind}.hits"))
                    add(f"{base}_{kind}_misses", T["misses"], count(T["misses"]), src(fname, f"{p}.{kind}.misses"))
                    add(f"{base}_{kind}_false_alarms", T["false_alarms"], count(T["false_alarms"]), src(fname, f"{p}.{kind}.false_alarms"))
                # seed ranges
                if fname == "lstm_all":
                    dp = f"decomposition.levels.{k}.{L}"
                    dd = get(LA, dp)
                    ranges = {"nse": (dd["nse_seed_median_min"], dd["nse_seed_median_max"]), "onset_pod": (dd["onset_pod_seed_min"], dd["onset_pod_seed_max"]),
                              "onset_far": (dd["onset_far_seed_min"], dd["onset_far_seed_max"])}
                    rsrc, nseeds = src("lstm_all", dp + ".*_seed_min/max"), len(get(LA, f"sets.{k}.seeds"))
                elif fname == "rain_timing":
                    sr = get(RM, f"seed_ranges.sets.{k}.{L}")
                    ranges = {"nse": tuple(sr["nse_median"]), "onset_pod": tuple(sr["onset_pod"]), "onset_far": tuple(sr["onset_far"])}
                    rsrc, nseeds = src("rain_to_midnight", f"seed_ranges.sets.{k}.{L}"), sr.get("n_seeds", 3)
                else:
                    sr = get(D, seedpath.format(k=k) + f".{L}")
                    ranges = {"nse": tuple(sr["nse_median"]), "onset_pod": tuple(sr["onset_pod"]), "onset_far": tuple(sr["onset_far"])}
                    rsrc, nseeds = src(fname, seedpath.format(k=k) + f".{L}"), sr.get("n_seeds", 3)
                add(f"{base}_nse_range", list(ranges["nse"]), n2_range(*ranges["nse"]), rsrc, "lowest and highest single training")
                add(f"{base}_onset_pod_range", list(ranges["onset_pod"]), pct_range(*ranges["onset_pod"]), rsrc, "lowest and highest single training")
                add(f"{base}_onset_far_range", list(ranges["onset_far"]), pct_range(*ranges["onset_far"]), rsrc, "lowest and highest single training")
                add(f"{base}_n_seeds", nseeds, str(nseeds), rsrc, "trainings averaged (1 = a single training)")
                # verdicts
                if fname in ("lstm_all", "rain_timing"):
                    V, vsrc = get(RM, f"verdicts.sets.{k}.{L}"), f"verdicts.sets.{k}.{L}"
                    vfile = "rain_to_midnight"
                else:
                    V, vsrc, vfile = get(D, vpath.format(k=k) + f".{L}"), vpath.format(k=k) + f".{L}", fname
                add(f"{base}_onset_verdict", V["plain"]["onset"]["verdict"], V["plain"]["onset"]["verdict"], src(vfile, vsrc + ".plain.onset.verdict"), "at the fixed level (trigger 1.0); criteria of results/flood_analysis.json verdict.criteria")
                add(f"{base}_all_verdict", V["plain"]["all"]["verdict"], V["plain"]["all"]["verdict"], src(vfile, vsrc + ".plain.all.verdict"), "all high-flow days, fixed level")
                ch = V["chosen_on_validation_all_days"]
                add(f"{base}_chosen_trigger", ch["trigger"], f"x{ch['trigger']:.2f}", src(vfile, vsrc + ".chosen_on_validation_all_days.trigger"), "the warning level chosen on the validation years (best CSI of all high-flow days)")
                add(f"{base}_onset_verdict_chosen", ch["onset"]["verdict"], ch["onset"]["verdict"], src(vfile, vsrc + ".chosen_on_validation_all_days.onset.verdict"))
                add(f"{base}_all_verdict_chosen", ch["all"]["verdict"], ch["all"]["verdict"], src(vfile, vsrc + ".chosen_on_validation_all_days.all.verdict"))
                SET_TABLE.append((name, L, base))
    # the onsets of each arm
    FI = load("forecast_input")
    for L in LEADS:
        n = get(FI, f"ladder.sat.rungs.0.by_lead.{L}.n_onsets")
        add(f"onsets_sat_{L}", n, count(n), src("forecast_input", f"ladder.sat.rungs.0.by_lead.{L}.n_onsets"), "onsets of the satellite-era test years 2010-2014 (E2b)")

    # ---- simple models -------------------------------------------------------------------------------------------------------------------
    section("7.1 Simple models (baselines) and the best simple model by validation (T7, 5.4)")
    BL = load("baselines")
    for row in BL["summary_table"]:
        base = f"{row['model']}_l{row['lead']}"
        p = f"summary_table[model={row['model']},lead={row['lead']}]"
        add(f"{base}_nse", row["nse_median"], n2(row["nse_median"]), src("baselines", p + ".nse_median"), f"inputs {row['inputs']}")
        add(f"{base}_all_pod", row["pod"], pct(row["pod"]), src("baselines", p + ".pod"))
        add(f"{base}_all_far", row["far"], pct(row["far"]), src("baselines", p + ".far"))
        add(f"{base}_all_csi", row["csi"], n2(row["csi"]), src("baselines", p + ".csi"))
        add(f"{base}_onset_pod", row["onset_pod"], pct(row["onset_pod"]), src("baselines", p + ".onset_pod"))
        add(f"{base}_onset_far", row["onset_far"], pct(row["onset_far"]), src("baselines", p + ".onset_far"))
        add(f"{base}_onset_csi", row["onset_csi"], n2(row["onset_csi"]), src("baselines", p + ".onset_csi"))
    for fname, sets in (("aorc_rain_to_midnight", ("base_persistence", "base_linreg_wq", "base_rf_wq", "base_linreg_wqh", "base_rf_wqh", "base_linreg_wqa", "base_rf_wqa")),
                        ("forecast_input", ("base_linreg_wqaF", "base_rf_wqaF"))):
        D = load(fname)
        for s in sets:
            for L in LEADS:
                T = opt(D, f"yardstick.table_test.{L}.{s}")
                if not T:
                    continue
                base = f"{s}_{L}"
                p = f"yardstick.table_test.{L}.{s}"
                if f"{base}_nse" not in REG:
                    add(f"{base}_nse", T["nse_median"], n2(T["nse_median"]), src(fname, p + ".nse_median"), "simple model given the new input (one fit)")
                    add(f"{base}_onset_pod", T["onset"]["pod"], pct(T["onset"]["pod"]), src(fname, p + ".onset.pod"))
                    add(f"{base}_onset_far", T["onset"]["far"], pct(T["onset"]["far"]), src(fname, p + ".onset.far"))
                    add(f"{base}_onset_csi", T["onset"]["csi"], n2(T["onset"]["csi"]), src(fname, p + ".onset.csi"))
                    add(f"{base}_all_pod", T["all"]["pod"], pct(T["all"]["pod"]), src(fname, p + ".all.pod"))
                    add(f"{base}_all_far", T["all"]["far"], pct(T["all"]["far"]), src(fname, p + ".all.far"))
                    add(f"{base}_all_csi", T["all"]["csi"], n2(T["all"]["csi"]), src(fname, p + ".all.csi"))
                for kind in ("onset", "all"):
                    for f_ in ("hits", "misses", "false_alarms"):
                        add(f"{base}_{kind}_{f_}", T[kind][f_], count(T[kind][f_]), src(fname, f"{p}.{kind}.{f_}"))
                add(f"{base}_onset_verdict", T["onset_verdict_plain"], T["onset_verdict_plain"], src(fname, p + ".onset_verdict_plain"))
    for L in LEADS:
        for arm, label in (("without_weather_forecast", "best simple model without a weather forecast, chosen on the validation years"), ("with_perfect_forecast", "best simple model with the perfect weather forecast")):
            B = get(BL, f"best_by_validation.{L}.{arm}")
            p = f"best_by_validation.{L}.{arm}"
            add(f"best_simple_{arm}_{L}_model", B["model"], B["model"], src("baselines", p + ".model"), label)
            add(f"best_simple_{arm}_{L}_val_nse", B["val_nse_median"], n2(B["val_nse_median"]), src("baselines", p + ".val_nse_median"))
            add(f"best_simple_{arm}_{L}_nse", B["test"]["nse_median"], n2(B["test"]["nse_median"]), src("baselines", p + ".test.nse_median"), label)
            add(f"best_simple_{arm}_{L}_onset_pod", B["test"]["onset_pod"], pct(B["test"]["onset_pod"]), src("baselines", p + ".test.onset_pod"))
            add(f"best_simple_{arm}_{L}_onset_far", B["test"]["onset_far"], pct(B["test"]["onset_far"]), src("baselines", p + ".test.onset_far"))
            add(f"best_simple_{arm}_{L}_all_pod", B["test"]["pod"], pct(B["test"]["pod"]), src("baselines", p + ".test.pod"))
            add(f"best_simple_{arm}_{L}_all_far", B["test"]["far"], pct(B["test"]["far"]), src("baselines", p + ".test.far"))
    # LSTM against the best simple model (section 4.2 rule)
    for L in LEADS:
        for b in ("base_linreg_wq", "base_rf_wq", "base_persistence"):
            C = get(LA, f"comparison.wq.baselines.{b}.{L}")
            p = f"comparison.wq.baselines.{b}.{L}"
            add(f"lstm_wq_vs_{b}_{L}_n_basins_lstm_higher", C["n_basins_lstm_higher"], f"{C['n_basins_lstm_higher']} of {C['n_basins']}", src("lstm_all", p + ".n_basins_lstm_higher"))
            add(f"lstm_wq_vs_{b}_{L}_wilcoxon_p", C["wilcoxon_p"], n2(C["wilcoxon_p"]), src("lstm_all", p + ".wilcoxon_p"))
            add(f"lstm_wq_vs_{b}_{L}_difference_of_medians", C["difference_of_medians"], signed(C["difference_of_medians"]), src("lstm_all", p + ".difference_of_medians"), "LSTM median NSE minus the simple model's")
            add(f"lstm_wq_vs_{b}_{L}_seed_spread_reading", C["baseline_against_seed_spread"], C["baseline_against_seed_spread"], src("lstm_all", p + ".baseline_against_seed_spread"))
    f3 = LA["findings"][3]
    add("finding_lstm_vs_simple_wq_l1", f3, f3, src("lstm_all", "findings.3"))
    KF = load("key_figure")
    A = get(KF, "levels.2.others.base_linreg_wq.against_lstm")
    add("lstm_vs_best_simple_l1_reading", A["in_words"], A["in_words"], src("key_figure", "levels.2.others.base_linreg_wq.against_lstm.in_words"), "the headline comparison with the gauge, 1 day ahead")


# ====================================================================================================================================
def build_key_figure():
    section("7.2 The key figure: levels, steps, finding, criteria (Figure 5, slides 8-10)")
    KF = load("key_figure")
    add("kf_headline", KF["headline"], KF["headline"], src("key_figure", "headline"))
    add("kf_figure_title", get(KF, "figure.title"), get(KF, "figure.title"), src("key_figure", "figure.title"))
    add("kf_figure_subtitle", get(KF, "figure.subtitle"), get(KF, "figure.subtitle"), src("key_figure", "figure.subtitle"))
    add("kf_figure_footnote", get(KF, "figure.footnote"), get(KF, "figure.footnote"), src("key_figure", "figure.footnote"))
    add("kf_figure_asterisk", get(KF, "figure.asterisk"), get(KF, "figure.asterisk"), src("key_figure", "figure.asterisk"))
    add("kf_figure_caption", get(KF, "figure.caption"), get(KF, "figure.caption"), src("key_figure", "figure.caption"))
    add("kf_finding_holds", get(KF, "finding.holds"), "holds" if get(KF, "finding.holds") else "does not hold", src("key_figure", "finding.holds"), get(KF, "finding.rule"))
    add("kf_spread_model_kinds_onset_points", get(KF, "finding.spread_between_model_kinds.onset_warned_points"), f"{get(KF, 'finding.spread_between_model_kinds.onset_warned_points'):.0f} points",
        src("key_figure", "finding.spread_between_model_kinds.onset_warned_points"), "highest minus lowest of the four kinds of model on past weather + river gauge, 1 day ahead")
    add("kf_spread_model_kinds_nse", get(KF, "finding.spread_between_model_kinds.nse"), n2(get(KF, "finding.spread_between_model_kinds.nse")), src("key_figure", "finding.spread_between_model_kinds.nse"))
    add("kf_smallest_step_onset_points", get(KF, "finding.smallest_step_to_the_right.onset_warned_points"), f"{get(KF, 'finding.smallest_step_to_the_right.onset_warned_points'):.1f} points", src("key_figure", "finding.smallest_step_to_the_right.onset_warned_points"))
    add("kf_smallest_step_nse", get(KF, "finding.smallest_step_to_the_right.nse"), n2(get(KF, "finding.smallest_step_to_the_right.nse"), 3), src("key_figure", "finding.smallest_step_to_the_right.nse"))
    for i, lv in enumerate(KF["levels"]):
        k = lv["key"]
        add(f"kf_level_{k}_label", lv["label"], lv["label"], src("key_figure", f"levels.{i}.label"))
        if lv.get("lstm"):
            L_ = lv["lstm"]
            add(f"kf_level_{k}_nse", L_["nse"], n2(L_["nse"]), src("key_figure", f"levels.{i}.lstm.nse"), lv["label"])
            add(f"kf_level_{k}_onset_warned", L_["onset_warned"], pct(L_["onset_warned"]), src("key_figure", f"levels.{i}.lstm.onset_warned"))
            add(f"kf_level_{k}_onset_false", L_["onset_false"], pct(L_["onset_false"]), src("key_figure", f"levels.{i}.lstm.onset_false"))
            add(f"kf_level_{k}_onset_csi", L_["onset_csi"], n2(L_["onset_csi"]), src("key_figure", f"levels.{i}.lstm.onset_csi"))
            add(f"kf_level_{k}_nse_seeds", L_["nse_seeds"], n2_range(min(L_["nse_seeds"]), max(L_["nse_seeds"])), src("key_figure", f"levels.{i}.lstm.nse_seeds"))
            add(f"kf_level_{k}_onset_warned_seeds", L_["onset_warned_seeds"], pct_range(min(L_["onset_warned_seeds"]), max(L_["onset_warned_seeds"])), src("key_figure", f"levels.{i}.lstm.onset_warned_seeds"))
            add(f"kf_level_{k}_onset_verdict", L_["onset_verdict"], L_["onset_verdict"], src("key_figure", f"levels.{i}.lstm.onset_verdict"))
        for o, ov in lv.get("others", {}).items():
            add(f"kf_level_{k}_{o}_nse", ov["nse"], n2(ov["nse"]), src("key_figure", f"levels.{i}.others.{o}.nse"), ov.get("label", o))
            add(f"kf_level_{k}_{o}_onset_warned", ov["onset_warned"], pct(ov["onset_warned"]), src("key_figure", f"levels.{i}.others.{o}.onset_warned"))
            add(f"kf_level_{k}_{o}_onset_false", ov["onset_false"], pct(ov["onset_false"]), src("key_figure", f"levels.{i}.others.{o}.onset_false"))
    T = get(KF, "levels.2.others.cmp_transformer")
    add("transformer_twin_nse", get(T, "twin.nse"), n2(get(T, "twin.nse")), src("key_figure", "levels.2.others.cmp_transformer.twin.nse"), "the two-seed LSTM twin of the Transformer comparison (shorter sequences), 1 day ahead")
    add("transformer_twin_onset_warned", get(T, "twin.onset_warned"), pct(get(T, "twin.onset_warned")), src("key_figure", "levels.2.others.cmp_transformer.twin.onset_warned"))
    add("transformer_nse_minus_twin", T["nse_minus_twin"], signed(T["nse_minus_twin"], 3), src("key_figure", "levels.2.others.cmp_transformer.nse_minus_twin"))
    add("transformer_placed", T["placed"], T["placed"], src("key_figure", "levels.2.others.cmp_transformer.placed"))
    for name, st in KF["steps"].items():
        if name == "spread_between_model_kinds":
            continue
        add(f"kf_step_{name}_what", st["what"], st["what"], src("key_figure", f"steps.{name}.what"))
        add(f"kf_step_{name}_nse", st["nse"], signed(st["nse"]), src("key_figure", f"steps.{name}.nse"), f"{st['from_set']} -> {st['to_set']}, 1 day ahead, median NSE")
        add(f"kf_step_{name}_onset_points", st["onset_warned_points"], pts(st["onset_warned_points"]), src("key_figure", f"steps.{name}.onset_warned_points"), f"{st['from_set']} -> {st['to_set']}, 1 day ahead, onsets warned")
        if "nse_in_words" in st:
            add(f"kf_step_{name}_nse_reading", st["nse_in_words"], st["nse_in_words"], src("key_figure", f"steps.{name}.nse_in_words"))
        elif "nse_reading" in st:
            add(f"kf_step_{name}_nse_reading", st["nse_reading"], st["nse_reading"], src("key_figure", f"steps.{name}.nse_reading"))
        if "n_basins_better" in st:
            add(f"kf_step_{name}_n_basins_better", st["n_basins_better"], f"{st['n_basins_better']} of 12", src("key_figure", f"steps.{name}.n_basins_better"))
        if "nse_wilcoxon_p" in st:
            add(f"kf_step_{name}_wilcoxon_p", st["nse_wilcoxon_p"], n2(st["nse_wilcoxon_p"], 3), src("key_figure", f"steps.{name}.nse_wilcoxon_p"))
        if "onset_share_of_gap_closed" in st:
            add(f"kf_step_{name}_onset_gap_closed", st["onset_share_of_gap_closed"], pct(st["onset_share_of_gap_closed"]), src("key_figure", f"steps.{name}.onset_share_of_gap_closed"), "share of the gap to the perfect weather forecast closed")
    for Lk in ("l2", "l3"):
        F = get(KF, f"far.{Lk}")
        for i, lv in enumerate(F["levels"]):
            k = lv["key"]
            L_ = lv["lstm"]
            add(f"kf_{Lk}_{k}_nse", L_["nse"], n2(L_["nse"]), src("key_figure", f"far.{Lk}.levels.{i}.lstm.nse"), lv["label"] + (" (upper bound)" if lv.get("upper_bound") else ""))
            add(f"kf_{Lk}_{k}_onset_warned", L_["onset_warned"], pct(L_["onset_warned"]), src("key_figure", f"far.{Lk}.levels.{i}.lstm.onset_warned"))
            add(f"kf_{Lk}_{k}_onset_false", L_["onset_false"], pct(L_["onset_false"]), src("key_figure", f"far.{Lk}.levels.{i}.lstm.onset_false"))
            add(f"kf_{Lk}_{k}_onset_verdict", L_["onset_verdict"], L_["onset_verdict"], src("key_figure", f"far.{Lk}.levels.{i}.lstm.onset_verdict"))
        for name, st in F["steps"].items():
            add(f"kf_{Lk}_step_{name}_nse", st["nse"], signed(st["nse"]), src("key_figure", f"far.{Lk}.steps.{name}.nse"), st["what"])
            add(f"kf_{Lk}_step_{name}_onset_points", st["onset_warned_points"], pts(st["onset_warned_points"]), src("key_figure", f"far.{Lk}.steps.{name}.onset_warned_points"))
            add(f"kf_{Lk}_step_{name}_nse_reading", st["nse_in_words"], st["nse_in_words"], src("key_figure", f"far.{Lk}.steps.{name}.nse_in_words"))
    for k, lv in get(KF, "forecast_1_day").items():
        L_ = lv["lstm"]
        add(f"kf_l1_{k}_nse", L_["nse"], n2(L_["nse"]), src("key_figure", f"forecast_1_day.{k}.lstm.nse"), lv["label"])
        add(f"kf_l1_{k}_onset_warned", L_["onset_warned"], pct(L_["onset_warned"]), src("key_figure", f"forecast_1_day.{k}.lstm.onset_warned"))
        add(f"kf_l1_{k}_onset_false", L_["onset_false"], pct(L_["onset_false"]), src("key_figure", f"forecast_1_day.{k}.lstm.onset_false"))
        add(f"kf_l1_{k}_onset_verdict", L_["onset_verdict"], L_["onset_verdict"], src("key_figure", f"forecast_1_day.{k}.lstm.onset_verdict"))
    C = KF["criteria"]
    add("go_rule", C["GO"], C["GO"], src("key_figure", "criteria.GO"), "fixed on 3 October 2026 before any added input was tried (results/flood_analysis.json verdict.criteria)")
    add("caution_rule", C["CAUTION"], C["CAUTION"], src("key_figure", "criteria.CAUTION"))
    add("not_yet_rule", C["NOT_YET"], C["NOT_YET"], src("key_figure", "criteria.NOT_YET"))
    add("go_min_warned", C["go_min_warned"], pct(C["go_min_warned"]), src("key_figure", "criteria.go_min_warned"), "GO: at least this share of onsets warned (a target, not met)")
    add("go_max_false", C["go_max_false"], pct(C["go_max_false"]), src("key_figure", "criteria.go_max_false"), "GO: at most this share of false warnings (a target, not met together with the first)")
    add("caution_min_csi", C["caution_min_csi"], n2(C["caution_min_csi"]), src("key_figure", "criteria.caution_min_csi"))
    add("verdict_definitions", C["definitions"], C["definitions"], src("key_figure", "criteria.definitions"))
    add("verdict_applied_to", get(load("flood_analysis"), "verdict.criteria.applied_to"), get(load("flood_analysis"), "verdict.criteria.applied_to"), src("flood_analysis", "verdict.criteria.applied_to"))
    add("kf_practice_all_not_yet", get(KF, "verdicts.practice.all_not_yet"), "NOT YET for onsets in every set that could be had in practice, at every lead" if get(KF, "verdicts.practice.all_not_yet") else "an exception exists",
        src("key_figure", "verdicts.practice.all_not_yet"), f"{len(get(KF, 'verdicts.practice.sets'))} set-lead cells checked; exceptions: {get(KF, 'verdicts.practice.exceptions')}")
    add("kf_practice_n_cells", len(get(KF, "verdicts.practice.sets")), str(len(get(KF, "verdicts.practice.sets"))), src("key_figure", "verdicts.practice.sets (length)"))
    for i, ub in enumerate(get(KF, "verdicts.upper_bounds")):
        add(f"kf_upper_bound_{i + 1}", ub, f"{ub['name']}, {ub['lead']} {'day' if ub['lead'] == 1 else 'days'}: {ub['at_the_fixed_level']} at the fixed level, {ub['at_the_level_chosen_on_validation']} at the chosen level", src("key_figure", f"verdicts.upper_bounds.{i}"), "an upper bound that cannot be had in practice")
    RT = get(KF, "levels.4.real_time")
    add("kf_realtime_interval_share_surviving", RT["interval_share_of_the_aorc_effect_surviving"], pct_range(*RT["interval_share_of_the_aorc_effect_surviving"]), src("key_figure", "levels.4.real_time.interval_share_of_the_aorc_effect_surviving"), RT["note"])
    add("kf_checks_compared", get(KF, "checks.compared"), count(get(KF, "checks.compared")), src("key_figure", "checks.compared"), "numbers of the figure audited against their sources by 11_key_figure.py")
    add("kf_checks_largest_difference", get(KF, "checks.largest_difference"), f"{get(KF, 'checks.largest_difference'):.1e}", src("key_figure", "checks.largest_difference"))
    for r in ("report", "plain", "talk"):
        add(f"kf_numbers_drawn_{r}", get(KF, f"checks.drawn_numbers.numbers_drawn_or_printed.{r}"), count(get(KF, f"checks.drawn_numbers.numbers_drawn_or_printed.{r}")), src("key_figure", f"checks.drawn_numbers.numbers_drawn_or_printed.{r}"))


# ====================================================================================================================================
def build_experiments():
    section("7.3 E1 (ERA5 rain to midnight), E1b (AORC), the Stage IV check and the hindsight bound")
    RM, AR = load("rain_to_midnight"), load("aorc_rain_to_midnight")
    for fname, D, tag in (("rain_to_midnight", RM, "e1"), ("aorc_rain_to_midnight", AR, "e1b")):
        add(f"{tag}_outcome", D["outcome"], D["outcome"], src(fname, "outcome"), "the short sentence: incomplete alone, quote outcome_full in documents (notes[0])")
        add(f"{tag}_outcome_caveat", D["outcome_caveat"], D["outcome_caveat"], src(fname, "outcome_caveat"))
        add(f"{tag}_outcome_full", D["outcome_full"], D["outcome_full"], src(fname, "outcome_full"), "the sentence the documents must quote whole (HANDOFF.md section 2 point 7)")
        add(f"{tag}_n_trainings", D["n_trainings"], str(D["n_trainings"]), src(fname, "n_trainings"))
        add(f"{tag}_training_seconds", D["training_seconds"], minutes_text(D["training_seconds"]), src(fname, "training_seconds"),
            f"{D['training_seconds']} s: seeds 0-2 on the Mac" + (", seeds 3-4 on the Windows PC (30 October 2026)" if len(D["seeds"]) == 5 else ""))
        add(f"{tag}_seeds", D["seeds"], str(len(D["seeds"])), src(fname, "seeds"), f"{len(D['seeds'])} trainings per set of this script; its stored controls from scripts 04 and 10 keep three")
        for i, c in enumerate(D["choices_made_after_looking_at_test_scores"]):
            add(f"{tag}_choice_after_test_{i + 1}", c, c, src(fname, f"choices_made_after_looking_at_test_scores.{i}"), "disclosed as HANDOFF.md section 7 asks")
        for i, w in enumerate(D["what_was_not_done"]):
            add(f"{tag}_not_done_{i + 1}", w, w, src(fname, f"what_was_not_done.{i}"))
    E = get(RM, "e1_success.wqh")
    add("e1_rule", E["rule"], E["rule"], src("rain_to_midnight", "e1_success.wqh.rule"))
    add("e1_met", E["met"], "not met" if not E["met"] else "met", src("rain_to_midnight", "e1_success.wqh.met"), "wqh ties NLDAS (29.95%) and its control is not flat")
    add("e1_control_flat", E["control_flat"], "not flat" if not E["control_flat"] else "flat", src("rain_to_midnight", "e1_success.wqh.control_flat"))
    add("e1_control_to09_flat", E["control_to09_flat"], "flat" if E["control_to09_flat"] else "not flat", src("rain_to_midnight", "e1_success.wqh.control_to09_flat"), "the matched-window control (ERA5 over 09:00 to 09:00) is flat")
    for k in ("with_rain_to_midnight", "nldas", "control", "rain_table_alone"):
        add(f"e1_onset_warned_{k}", E["onset_warned"][k], pct(E["onset_warned"][k]), src("rain_to_midnight", f"e1_success.wqh.onset_warned.{k}"))
    V = get(RM, "value_of_hours_09_to_24.by_lead.l1")
    add("e1_hours_after_09_points", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.onsets.difference_points"), "lstm_wqh minus lstm_wqh_to09: what the ERA5 hours after about 09:00 are worth, onsets warned 1 day ahead")
    add("e1_hours_after_09_reading", get(V, "onsets.reading"), get(V, "onsets.reading"), src("rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.onsets.reading"))
    add("e1_hours_after_09_nse", get(V, "nse.gain"), signed(get(V, "nse.gain")), src("rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.nse.gain"), get(V, "nse.reading"))
    V = get(RM, "value_of_evening_hours_beyond_utc_day.by_lead.l1")
    add("e1_hours_after_utc_day_points", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("rain_to_midnight", "value_of_evening_hours_beyond_utc_day.by_lead.l1.onsets.difference_points"), "lstm_wqh minus lstm_wqh_utc: the evening hours beyond the UTC day (after 19:00 standard time)")
    for k, v in get(RM, "by_period.differences.wqh_minus_wq").items():
        add(f"e1_gain_{k}_points", v["onset_points"], pts(v["onset_points"]), src("rain_to_midnight", f"by_period.differences.wqh_minus_wq.{k}.onset_points"), "ERA5 rain to midnight minus the rain table alone, onsets warned 1 day ahead; ERA5 assimilates radar and gauge rain from 2009-07-17")
        add(f"e1_gain_{k}_nse", v["nse"], signed(v["nse"]), src("rain_to_midnight", f"by_period.differences.wqh_minus_wq.{k}.nse"))
    for model, label in (("ridge regression", "ridge"), ("random forest", "forest")):
        g = opt(RM, f"yardstick.new_input_given_to_each_model.{model}.by_lead.l1.onsets.difference_points")
        if g is not None:
            add(f"e1_yardstick_{label}_gain_points", g, pts(g), src("rain_to_midnight", f"yardstick.new_input_given_to_each_model.{model}.by_lead.l1.onsets.difference_points"), f"what the ERA5 rain to midnight adds to {model} (one fit), onsets warned 1 day ahead")
    g = opt(RM, "yardstick.new_input_given_to_each_model.lstm.by_lead.l1.onsets.difference_points")
    add("e1_yardstick_lstm_gain_points", g, pts(g), src("rain_to_midnight", "yardstick.new_input_given_to_each_model.lstm.by_lead.l1.onsets.difference_points"))
    # E1b
    E = get(AR, "e1b_success.wqa")
    add("e1b_rule", E["rule"], E["rule"], src("aorc_rain_to_midnight", "e1b_success.wqa.rule"))
    add("e1b_met", E["met"], "met" if E["met"] else "not met", src("aorc_rain_to_midnight", "e1b_success.wqa.met"), "clearly above NLDAS in every training, control flat")
    add("e1b_control_flat", E["control_flat"], "flat" if E["control_flat"] else "not flat", src("aorc_rain_to_midnight", "e1b_success.wqa.control_flat"))
    add("e1b_gain_grows", get(E, "roadmap_step_1b.gain_grows"), "the E1 gain grows with AORC" if get(E, "roadmap_step_1b.gain_grows") else "does not grow", src("aorc_rain_to_midnight", "e1b_success.wqa.roadmap_step_1b.gain_grows"))
    for k in ("with_aorc_rain_to_midnight", "nldas", "era5_rain_to_midnight", "control", "rain_table_alone"):
        add(f"e1b_onset_warned_{k}", E["onset_warned"][k], pct(E["onset_warned"][k]), src("aorc_rain_to_midnight", f"e1b_success.wqa.onset_warned.{k}"))
    G = get(AR, "gain_rule.wqa.against_rain_table_alone.l1")
    add("e1b_gain_over_rain_table_points", get(G, "onset.gain") * 100, points(get(G, "onset.gain")), src("aorc_rain_to_midnight", "gain_rule.wqa.against_rain_table_alone.l1.onset.gain"), "AORC rain to midnight minus the rain table alone, onsets warned 1 day ahead (+32.6 points)")
    add("e1b_gain_over_rain_table_reading", get(G, "onset.reading"), get(G, "onset.reading"), src("aorc_rain_to_midnight", "gain_rule.wqa.against_rain_table_alone.l1.onset.reading"))
    add("e1b_gain_over_rain_table_nse", get(G, "nse.gain"), signed(get(G, "nse.gain")), src("aorc_rain_to_midnight", "gain_rule.wqa.against_rain_table_alone.l1.nse.gain"), get(G, "nse.reading"))
    add("e1b_gain_over_rain_table_nse_basins", get(G, "nse.n_basins_higher"), f"{get(G, 'nse.n_basins_higher')} of 12", src("aorc_rain_to_midnight", "gain_rule.wqa.against_rain_table_alone.l1.nse.n_basins_higher"))
    V = get(AR, "value_of_hours_09_to_24.by_lead.l1")
    add("e1b_hours_after_09_points", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("aorc_rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.onsets.difference_points"), "lstm_wqa minus lstm_wqa_to09 (the same AORC rain over the table's own window): the hours after about 09:00 (+33.6 points)")
    add("e1b_hours_after_09_hits", get(V, "onsets.hits_difference"), count(get(V, "onsets.hits_difference")), src("aorc_rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.onsets.hits_difference"))
    add("e1b_hours_after_09_nse", get(V, "nse.gain"), signed(get(V, "nse.gain")), src("aorc_rain_to_midnight", "value_of_hours_09_to_24.by_lead.l1.nse.gain"), get(V, "nse.reading"))
    V = get(AR, "value_of_evening_hours_beyond_utc_day.by_lead.l1")
    add("e1b_hours_after_utc_day_points", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("aorc_rain_to_midnight", "value_of_evening_hours_beyond_utc_day.by_lead.l1.onsets.difference_points"), "lstm_wqa minus lstm_wqa_utc: the evening hours after 19:00 standard time (+13.7 points)")
    V = get(AR, "aorc_against_era5.by_lead.l1")
    add("e1b_aorc_minus_era5_points", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("aorc_rain_to_midnight", "aorc_against_era5.by_lead.l1.onsets.difference_points"), "AORC against ERA5 over the same hours")
    for k, v in get(AR, "by_period.differences.wqa_minus_wq").items():
        add(f"e1b_gain_{k}_points", v["onset_points"], pts(v["onset_points"]), src("aorc_rain_to_midnight", f"by_period.differences.wqa_minus_wq.{k}.onset_points"), "AORC rain to midnight minus the rain table alone, onsets 1 day ahead, before/from 2009-07-17")
        add(f"e1b_gain_{k}_nse", v["nse"], signed(v["nse"]), src("aorc_rain_to_midnight", f"by_period.differences.wqa_minus_wq.{k}.nse"))
    for model, label in (("ridge regression", "ridge"), ("random forest", "forest"), ("lstm", "lstm")):
        g = opt(AR, f"yardstick.new_input_given_to_each_model.{model}.by_lead.l1.onsets.difference_points")
        add(f"e1b_yardstick_{label}_gain_points", g, pts(g), src("aorc_rain_to_midnight", f"yardstick.new_input_given_to_each_model.{model}.by_lead.l1.onsets.difference_points"), f"what the AORC rain to midnight adds to {model}, onsets warned 1 day ahead")
    for pair, label in (("lstm_wqa__base_linreg_wqa", "ridge"), ("lstm_wqa__base_rf_wqa", "forest")):
        for L in LEADS:
            Y = get(AR, f"yardstick.lstm_against_simple.{pair}.{L}")
            add(f"e1b_lstm_vs_{label}_{L}_nse_reading", Y["reading_in_words"], Y["reading_in_words"], src("aorc_rain_to_midnight", f"yardstick.lstm_against_simple.{pair}.{L}.reading_in_words"), "NSE reading by the rule of report section 4.2")
            add(f"e1b_lstm_vs_{label}_{L}_onsets_in_words", Y["onsets_in_words"], Y["onsets_in_words"], src("aorc_rain_to_midnight", f"yardstick.lstm_against_simple.{pair}.{L}.onsets_in_words"))
    add("e1b_stamp_check_correlation", get(AR, "stamp_convention_check.correlation_of_the_two_daily_series"), n2(get(AR, "stamp_convention_check.correlation_of_the_two_daily_series"), 3), src("aorc_rain_to_midnight", "stamp_convention_check.correlation_of_the_two_daily_series"))
    add("aorc_download_gb", get(AR, "aorc_data.download.gb"), f"{get(AR, 'aorc_data.download.gb'):.1f} GB", src("aorc_rain_to_midnight", "aorc_data.download.gb"))
    add("aorc_download_hours", get(AR, "aorc_data.download.hours"), f"{get(AR, 'aorc_data.download.hours'):.1f} h", src("aorc_rain_to_midnight", "aorc_data.download.hours"))
    add("aorc_chunks_fetched", get(AR, "aorc_data.download.chunks_fetched"), count(get(AR, "aorc_data.download.chunks_fetched")), src("aorc_rain_to_midnight", "aorc_data.download.chunks_fetched"))
    add("aorc_hours_filled_from_era5", sum(get(AR, "aorc_data.summary.hours_filled_from_era5_by_basin").values()), count(sum(get(AR, "aorc_data.summary.hours_filled_from_era5_by_basin").values())), src("aorc_rain_to_midnight", "aorc_data.summary.hours_filled_from_era5_by_basin (sum)"))
    add("aorc_ratio_to_daymet_mean_rain", get(AR, "aorc_data.summary.median_ratio_aorc_to_daymet_mean_rain"), n2(get(AR, "aorc_data.summary.median_ratio_aorc_to_daymet_mean_rain")), src("aorc_rain_to_midnight", "aorc_data.summary.median_ratio_aorc_to_daymet_mean_rain"))
    # Stage IV
    ST = load("stage4_realtime")
    n_s4 = len(get(ST, "models.wqa.variants.stage4.single_seed_onsets_warned"))
    add("s4_outcome", ST["outcome"], ST["outcome"], src("stage4_realtime", "outcome"), f"quote with REALTIME_WORDING; the {n_s4} saved networks of lstm_wqa")
    m = re.search(r"(\d+)% interval", ST["outcome"])
    add("s4_interval_level", int(m.group(1)) if m else None, f"{m.group(1)}%" if m else "n/a", src("stage4_realtime", "outcome (the 'N% interval ... drawing onsets' phrase)"),
        "the level of the resampling intervals of the Stage IV check; the bracket beside the share of the AORC effect surviving is such an interval, not a seed range (critic 1, item 12)")
    add("s4_onsets_tested", get(ST, "onsets.tested"), count(get(ST, "onsets.tested")), src("stage4_realtime", "onsets.tested"))
    add("s4_issue_days_tested", get(ST, "onsets.issue_days_tested"), count(get(ST, "onsets.issue_days_tested")), src("stage4_realtime", "onsets.issue_days_tested"))
    W = get(ST, "models.wqa")
    add("s4_aorc_onsets_warned", W["aorc"], count(W["aorc"]), src("stage4_realtime", "models.wqa.aorc"), f"with AORC on the issue day (as saved): {W['aorc']} of 611")
    add("s4_rain_table_onsets_warned", W["rain_table_alone"], count(W["rain_table_alone"]), src("stage4_realtime", "models.wqa.rain_table_alone"))
    for vk, label in (("stage4", "Stage IV on the issue day"), ("stage4_k", "Stage IV times the level of AORC"), ("stage4_no_last_hour", "Stage IV without the last hour"),
                      ("stage4_no_last_two_hours", "Stage IV without the last two hours"), ("aorc_no_last_hour", "AORC without the last hour"), ("aorc_no_last_two_hours", "AORC without the last two hours"),
                      ("stage4_and_day_before", "Stage IV on the issue day and the day before")):
        v = W["variants"][vk]
        p = f"models.wqa.variants.{vk}"
        add(f"s4_{vk}_onsets_warned", v["onsets_warned"], count(v["onsets_warned"]), src("stage4_realtime", p + ".onsets_warned"), label + f" ({v['description']})")
        add(f"s4_{vk}_share", v["share_of_the_onsets"], pct(v["share_of_the_onsets"]), src("stage4_realtime", p + ".share_of_the_onsets"), label)
        add(f"s4_{vk}_effect_surviving", v["share_of_the_aorc_effect_surviving"], pct(v["share_of_the_aorc_effect_surviving"]), src("stage4_realtime", p + ".share_of_the_aorc_effect_surviving"), f"share of the AORC effect ({W['aorc']} - {W['rain_table_alone']} onsets) that survives")
        add(f"s4_{vk}_single_seeds", v["single_seed_onsets_warned"], ", ".join(str(x) for x in v["single_seed_onsets_warned"]), src("stage4_realtime", p + ".single_seed_onsets_warned"), f"onsets warned by each of the {n_s4} saved networks")
        add(f"s4_{vk}_interval_onsets", get(v, "intervals.over_onsets.share_surviving"), pct_range(*get(v, "intervals.over_onsets.share_surviving")), src("stage4_realtime", p + ".intervals.over_onsets.share_surviving"), "95% interval drawing onsets")
        add(f"s4_{vk}_interval_issue_days", get(v, "intervals.over_issue_days.share_surviving"), pct_range(*get(v, "intervals.over_issue_days.share_surviving")), src("stage4_realtime", p + ".intervals.over_issue_days.share_surviving"), "95% interval drawing issue days")
        add(f"s4_{vk}_mae_mm_day", get(v, "fit_on_the_onset_days.mae_mm_day"), n2(get(v, "fit_on_the_onset_days.mae_mm_day")), src("stage4_realtime", p + ".fit_on_the_onset_days.mae_mm_day"), "mean absolute error of the 1-day forecast on the onset days, mm/day")
    add("s4_k_level", ST["k_level_aorc_over_stage4"], n2(ST["k_level_aorc_over_stage4"]), src("stage4_realtime", "k_level_aorc_over_stage4"), "the level of AORC over Stage IV on the issue days")
    for ref in ("aorc", "rain_table_alone"):
        add(f"s4_{ref}_mae_mm_day", get(W, f"reference_fits_on_the_onset_days.{ref}.mae_mm_day"), n2(get(W, f"reference_fits_on_the_onset_days.{ref}.mae_mm_day")), src("stage4_realtime", f"models.wqa.reference_fits_on_the_onset_days.{ref}.mae_mm_day"))
    for ck, label in (("another_onset_stage4", "the Stage IV rain of another onset of the same basin"), ("ordinary_day_aorc", "the AORC rain of an ordinary day"), ("no_rain", "no rain")):
        c = get(W, f"controls_on_the_rain_of_the_issue_day.{ck}")
        add(f"s4_control_{ck}_mean", c["mean"], f"{c['mean']:.0f}", src("stage4_realtime", f"models.wqa.controls_on_the_rain_of_the_issue_day.{ck}.mean"), label + f" ({c['draws']} draws)")
        add(f"s4_control_{ck}_range", [c["lowest"], c["highest"]], f"{c['lowest']} to {c['highest']}", src("stage4_realtime", f"models.wqa.controls_on_the_rain_of_the_issue_day.{ck}.lowest/highest"))
        add(f"s4_control_{ck}_mae_mm_day", get(c, "fit_on_the_onset_days.mae_mm_day"), n2(get(c, "fit_on_the_onset_days.mae_mm_day")), src("stage4_realtime", f"models.wqa.controls_on_the_rain_of_the_issue_day.{ck}.fit_on_the_onset_days.mae_mm_day"))
    for pk in ("aorc", "stage4", "stage4_times_the_level_of_aorc"):
        I = get(W, f"issue_days_of_all_basins.{pk}")
        add(f"s4_issue_days_{pk}_hits", I["hits"], count(I["hits"]), src("stage4_realtime", f"models.wqa.issue_days_of_all_basins.{pk}.hits"), "warnings on the issue days of all 12 basins")
        add(f"s4_issue_days_{pk}_false_warnings", I["false_warnings"], count(I["false_warnings"]), src("stage4_realtime", f"models.wqa.issue_days_of_all_basins.{pk}.false_warnings"))
        add(f"s4_issue_days_{pk}_far", I["far"], pct(I["far"]), src("stage4_realtime", f"models.wqa.issue_days_of_all_basins.{pk}.far"), "share of the warnings on the issue days that were false")
    for i, h in enumerate(ST["honesty_points"]):
        add(f"s4_honesty_{i + 1}", h, h, src("stage4_realtime", f"honesty_points.{i}"))
    add("s4_reproduction_wqa_stored", get(ST, "reproduction.wqa.onsets_warned_stored"), count(get(ST, "reproduction.wqa.onsets_warned_stored")), src("stage4_realtime", "reproduction.wqa.onsets_warned_stored"), f"the saved networks reproduce the stored {get(ST, 'reproduction.wqa.onsets_warned_stored')}")
    add("s4_files_fetched", get(ST, "data.fetch_runs.files"), count(get(ST, "data.fetch_runs.files")), src("stage4_realtime", "data.fetch_runs.files"))
    add("s4_gb_read", get(ST, "data.fetch_runs.gb"), f"{get(ST, 'data.fetch_runs.gb')} GB", src("stage4_realtime", "data.fetch_runs.gb"))
    add("s4_fetch_hours", get(ST, "data.fetch_runs.hours"), f"{get(ST, 'data.fetch_runs.hours'):.1f} h", src("stage4_realtime", "data.fetch_runs.hours"))
    add("s4_stage4_over_aorc_correlation", get(ST, "agreement_of_the_two_products_on_the_issue_days.correlation"), n2(get(ST, "agreement_of_the_two_products_on_the_issue_days.correlation"), 3), src("stage4_realtime", "agreement_of_the_two_products_on_the_issue_days.correlation"))
    add("s4_mean_mm_aorc", get(ST, "agreement_of_the_two_products_on_the_issue_days.mean_mm_aorc"), n2(get(ST, "agreement_of_the_two_products_on_the_issue_days.mean_mm_aorc"), 1), src("stage4_realtime", "agreement_of_the_two_products_on_the_issue_days.mean_mm_aorc"), "mm on the issue days of the onsets")
    add("s4_mean_mm_stage4", get(ST, "agreement_of_the_two_products_on_the_issue_days.mean_mm_stage4"), n2(get(ST, "agreement_of_the_two_products_on_the_issue_days.mean_mm_stage4"), 1), src("stage4_realtime", "agreement_of_the_two_products_on_the_issue_days.mean_mm_stage4"))
    # hindsight
    AH = load("aorc_hindsight")
    RV = load("review_aorc_rain_to_midnight")
    add("hindsight_builder_reading", AH["reading"], AH["reading"], src("aorc_hindsight", "reading"), "the builder's check on the 50 stormiest test-year days")
    add("hindsight_builder_n_days", get(AH, "sample.n_days"), str(get(AH, "sample.n_days")), src("aorc_hindsight", "sample.n_days"))
    cal = get(RV, "numbers.hindsight.calibration")
    add("hindsight_review_coefficient", cal["b2"], f"{cal['b2']:.3f}", src("review_aorc_rain_to_midnight", "numbers.hindsight.calibration.b2"), "mm of AORC rain before midnight per mm of Stage IV rain after it, 78 test-year days (936 basin-days)")
    add("hindsight_review_ci", cal["b2_ci"], f"{cal['b2_ci'][0]:.3f} to {cal['b2_ci'][1]:.3f}", src("review_aorc_rain_to_midnight", "numbers.hindsight.calibration.b2_ci"), "95% interval")
    add("hindsight_review_n_days", 78, "78", src("review_aorc_rain_to_midnight", "verdict ('on 78 test-year days (the builder's 50 plus 28 I chose ...)')"), "50 builder days + 28 review days")
    add("hindsight_review_added_days", len(get(RV, "numbers.hindsight.selected_days.days")), str(len(get(RV, "numbers.hindsight.selected_days.days"))), src("review_aorc_rain_to_midnight", "numbers.hindsight.selected_days.days (length)"))
    T1 = get(RV, "numbers.hindsight.T1.wqa")
    add("hindsight_onsets_in_sample", get(T1, "AORC (as saved).onsets_in_sample"), count(get(T1, "AORC (as saved).onsets_in_sample")), src("review_aorc_rain_to_midnight", "numbers.hindsight.T1.wqa.AORC (as saved).onsets_in_sample"), "onsets on the 78 days: 28% of the 611")
    add("hindsight_onsets_in_sample_share", get(T1, "AORC (as saved).onsets_in_sample") / 611, pct(get(T1, "AORC (as saved).onsets_in_sample") / 611), "computed: hindsight_onsets_in_sample / onsets_l1")
    add("hindsight_warned_aorc", get(T1, "AORC (as saved).warned"), count(get(T1, "AORC (as saved).warned")), src("review_aorc_rain_to_midnight", "numbers.hindsight.T1.wqa.AORC (as saved).warned"))
    s4w = [v["warned"] for k, v in T1.items() if k.startswith("Stage IV")]
    add("hindsight_warned_stage4_range", [min(s4w), max(s4w)], f"{min(s4w)} to {max(s4w)}", src("review_aorc_rain_to_midnight", "numbers.hindsight.T1.wqa.Stage IV *.warned"), "the saved networks given Stage IV in place of AORC on those days (the variants)")
    add("hindsight_warned_rain_table", get(RV, "numbers.hindsight.T1_wq_in_sample"), count(get(RV, "numbers.hindsight.T1_wq_in_sample")), src("review_aorc_rain_to_midnight", "numbers.hindsight.T1_wq_in_sample"))
    add("hindsight_onsets_not_tested", 611 - get(T1, "AORC (as saved).onsets_in_sample"), count(611 - get(T1, "AORC (as saved).onsets_in_sample")), "computed: onsets_l1 minus hindsight_onsets_in_sample (the review verdict: 'the other 440 onsets were not tested')")
    HB = _hindsight_bounds()
    add("hindsight_bound_hits", HB["hits_three"], str(HB["hits_three"]), src("review_aorc_rain_to_midnight", "checks[id=d7].numbers.loosest_bound_hits (verdict: 'at most about 3 points ... 16 of 313 hits')"),
        f"THREE SEEDS (round 1 of the E1b review, 7 October 2026): at most about {round(HB['points_three'])} points of the 51% ({HB['hits_three']} of {HB['of_hits_three']} hits); best estimate none. The five-seed bound: hindsight_bound_five_seeds_points")
    add("hindsight_bound_points", HB["hits_three"] / 611 * 100, f"{HB['hits_three'] / 611 * 100:.1f} points", f"computed: {HB['hits_three']} / 611 onsets",
        f"THREE SEEDS: '{HB['wording_three']}' (hindsight_three_seed_wording); the fixed wording fw_hindsight now carries the five-seed bound")
    add("hindsight_bound_five_seeds_points", HB["points"], f"{HB['points']:.1f}", HB["source"].replace(" (rounded to whole points)", "").replace(", rounded to whole points", ""),
        f"FIVE SEEDS (review of 30 October 2026): raw Stage IV in place of AORC on all {HB['onsets']} onsets loses {HB['hits']} of {HB['of_hits']} hits; "
        f"'{HB['wording']}' (fw_hindsight); an upper bound, best estimate none")
    add("hindsight_bound_five_seeds_hits", HB["hits"], f"{HB['hits']} of {HB['of_hits']}",
        (src("review_aorc_rain_to_midnight", f"{HB['round']}.hindsight.five_seed_loosest_bound_hits and five_seeds.aorc_all") if HB["round"] else HB["source"]),
        "hits lost when raw Stage IV replaces AORC on the issue day, of the hits of the five-seed lstm_wqa")
    add("hindsight_bound_three_seeds_points", HB["points_three"], f"{HB['points_three']:.1f}", HB["source_three"],
        f"THREE SEEDS (round 1 of the E1b review, 7 October 2026, the networks of seeds 0-2): about {round(HB['points_three'])} points, {HB['hits_three']} of {HB['of_hits_three']} hits")
    add("hindsight_bound_three_seeds_hits", HB["hits_three"], f"{HB['hits_three']} of {HB['of_hits_three']}",
        src("review_aorc_rain_to_midnight", "checks[id=d7].numbers.loosest_bound_hits") + "; " + src("review_aorc_rain_to_midnight", "round_2.hindsight.round1_of_hits"),
        "the hits of round 1's loosest bound, of the three-seed lstm_wqa's hits")
    if HB["restored"] is not None:
        add("hindsight_level_restored_points", HB["restored"], f"{HB['restored']:.1f}", HB["restored_src"],
            f"FIVE SEEDS: Stage IV times the level of AORC (the level restored) loses {HB['restored_hits']} hits on all {HB['onsets']} onsets")
        add("hindsight_level_restored_hits", HB["restored_hits"], str(HB["restored_hits"]), src("review_aorc_rain_to_midnight", f"{HB['round']}.hindsight.five_seeds.direct_all_611_level_matched_loss_hits"))
    add("hindsight_three_seed_wording", HB["wording_three"], HB["wording_three"], HB["source_three"] + " (rounded to whole points)",
        "the wording of round 1 of the E1b review (7 October 2026) for the three networks of seeds 0-2; documents explain once (limitations) that with five seeds the bound is fw_hindsight")
    add("hindsight_how_much", get(RV, "answers.d_hindsight.how_much_of_the_51_percent_could_be_hindsight"), get(RV, "answers.d_hindsight.how_much_of_the_51_percent_could_be_hindsight"), src("review_aorc_rain_to_midnight", "answers.d_hindsight.how_much_of_the_51_percent_could_be_hindsight"))
    add("review_e1b_verdict", RV["verdict"], RV["verdict"], src("review_aorc_rain_to_midnight", "verdict"))
    add("review_e1b_trust", get(RV, "answers.how_much_to_trust_the_51_percent"), get(RV, "answers.how_much_to_trust_the_51_percent"), src("review_aorc_rain_to_midnight", "answers.how_much_to_trust_the_51_percent"))

    # ---- E2 -------------------------------------------------------------------------------------------------------------------------------
    section("7.3 E2: the GEFS weather forecast as an input (main split E2a; retrained E2b on another split)")
    FI = load("forecast_input")
    add("e2_outcome", FI["outcome"], FI["outcome"], src("forecast_input", "outcome"))
    add("e2_outcome_caveat", FI["outcome_caveat"], FI["outcome_caveat"], src("forecast_input", "outcome_caveat"))
    add("e2_outcome_full", FI["outcome_full"], FI["outcome_full"], src("forecast_input", "outcome_full"), "quote whole; carries MOVED_POINT and NSE_FORECAST_WORDING")
    add("e2_n_trainings", FI["n_trainings"], str(FI["n_trainings"]), src("forecast_input", "n_trainings"))
    add("e2_training_seconds", FI["training_seconds"], minutes_text(FI["training_seconds"]), src("forecast_input", "training_seconds"), f"{FI['training_seconds']} s")
    add("e2_rule", get(FI, "e2_success.rule"), get(FI, "e2_success.rule"), src("forecast_input", "e2_success.rule"))
    add("e2_met_at_2_and_3_days", get(FI, "e2_success.met_at_2_and_3_days"), "met at 2 and 3 days" if get(FI, "e2_success.met_at_2_and_3_days") else "not met", src("forecast_input", "e2_success.met_at_2_and_3_days"))
    for i, c in enumerate(FI["choices_made_after_looking_at_test_scores"]):
        add(f"e2_choice_after_test_{i + 1}", c, c, src("forecast_input", f"choices_made_after_looking_at_test_scores.{i}"))
    for L in LEADS:
        B = get(FI, f"e2_success.by_lead.{L}")
        add(f"e2_met_{L}", B["met"], "met" if B["met"] else "not met", src("forecast_input", f"e2_success.by_lead.{L}.met"))
        add(f"e2_control_onset_warned_{L}", B["onset_warned_control"], pct(B["onset_warned_control"]), src("forecast_input", f"e2_success.by_lead.{L}.onset_warned_control"), "the shuffled-forecast control: measures harm, not information")
        V = get(FI, f"value_of_the_forecast.by_lead.{L}")
        add(f"e2_forecast_points_{L}", get(V, "onsets.difference_points"), pts(get(V, "onsets.difference_points")), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.onsets.difference_points"), "lstm_wqaF_sub minus lstm_wqa at the fixed level: -6.7 (MOVED_POINT), +18.2, +25.3 points")
        add(f"e2_forecast_reading_{L}", get(V, "onsets.reading"), get(V, "onsets.reading"), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.onsets.reading"), "the seed rule on onsets; at 1 day a moved operating point, not lost skill")
        add(f"e2_forecast_hits_difference_{L}", get(V, "onsets.hits_difference"), signed(get(V, "onsets.hits_difference"), 0), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.onsets.hits_difference"))
        add(f"e2_forecast_nse_{L}", get(V, "nse.gain"), signed(get(V, "nse.gain"), 3), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.nse.gain"), get(V, "nse.reading") + " (rule of report section 4.2)")
        add(f"e2_forecast_nse_reading_{L}", get(V, "nse.reading"), get(V, "nse.reading"), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.nse.reading"))
        add(f"e2_forecast_nse_basins_higher_{L}", get(V, "nse.n_basins_higher"), f"{get(V, 'nse.n_basins_higher')} of 12", src("forecast_input", f"value_of_the_forecast.by_lead.{L}.nse.n_basins_higher"))
        O = get(V, "overlap_of_warned_onsets")
        add(f"e2_only_forecast_model_{L}", O["warned_only_by_first"], count(O["warned_only_by_first"]), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.overlap_of_warned_onsets.warned_only_by_first"), "onsets warned only by lstm_wqaF_sub")
        add(f"e2_only_wqa_{L}", O["warned_only_by_second"], count(O["warned_only_by_second"]), src("forecast_input", f"value_of_the_forecast.by_lead.{L}.overlap_of_warned_onsets.warned_only_by_second"), "onsets warned only by lstm_wqa")
        OP = get(FI, f"value_of_the_forecast.operating_points.{L}")
        add(f"e2_chosen_level_points_{L}", get(OP, "level_chosen_on_the_validation_years.points"), pts(get(OP, "level_chosen_on_the_validation_years.points")), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.level_chosen_on_the_validation_years.points"), "at the warning level chosen on the validation years")
        add(f"e2_chosen_level_trigger_with_{L}", get(OP, "level_chosen_on_the_validation_years.trigger_with"), f"x{get(OP, 'level_chosen_on_the_validation_years.trigger_with'):.2f}", src("forecast_input", f"value_of_the_forecast.operating_points.{L}.level_chosen_on_the_validation_years.trigger_with"))
        add(f"e2_chosen_level_onsets_with_{L}", get(OP, "level_chosen_on_the_validation_years.onsets_warned_with"), pct(get(OP, "level_chosen_on_the_validation_years.onsets_warned_with")), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.level_chosen_on_the_validation_years.onsets_warned_with"))
        add(f"e2_chosen_level_false_with_{L}", get(OP, "level_chosen_on_the_validation_years.false_with"), pct(get(OP, "level_chosen_on_the_validation_years.false_with")), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.level_chosen_on_the_validation_years.false_with"))
        add(f"e2_budget40_points_{L}", get(OP, "equal_false_warning_budgets.points_at_most_40"), pts(get(OP, "equal_false_warning_budgets.points_at_most_40")), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.equal_false_warning_budgets.points_at_most_40"), "at most 40% of the onset warnings false, any level")
        add(f"e2_budget50_points_{L}", get(OP, "equal_false_warning_budgets.points_at_most_50"), pts(get(OP, "equal_false_warning_budgets.points_at_most_50")), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.equal_false_warning_budgets.points_at_most_50"))
        add(f"e2_ap_with_{L}", get(OP, "trigger_free.average_precision_with"), n2(get(OP, "trigger_free.average_precision_with"), 3), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.trigger_free.average_precision_with"), "trigger-free average precision of the onset warnings, with the forecast")
        add(f"e2_ap_without_{L}", get(OP, "trigger_free.average_precision_without"), n2(get(OP, "trigger_free.average_precision_without"), 3), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.trigger_free.average_precision_without"))
        add(f"e2_ap_difference_{L}", get(OP, "trigger_free.average_precision_difference"), signed(get(OP, "trigger_free.average_precision_difference"), 3), src("forecast_input", f"value_of_the_forecast.operating_points.{L}.trigger_free.average_precision_difference"))
        ci = get(OP, "trigger_free.average_precision_difference_ci95")
        add(f"e2_ap_difference_ci_{L}", ci, f"{signed(ci[0], 3)} to {signed(ci[1], 3)}", src("forecast_input", f"value_of_the_forecast.operating_points.{L}.trigger_free.average_precision_difference_ci95"), "95% block-bootstrap interval")
        bb = get(OP, "block_bootstrap_fixed_level.pod_points_ci95")
        add(f"e2_fixed_level_points_ci_{L}", bb, f"{signed(bb[0], 1)} to {signed(bb[1], 1)} points", src("forecast_input", f"value_of_the_forecast.operating_points.{L}.block_bootstrap_fixed_level.pod_points_ci95"), "95% block bootstrap over the test days (30-day blocks)")
        add(f"e2_fixed_level_nse_ci_{L}", get(OP, "block_bootstrap_fixed_level.nse_ci95"), f"{signed(get(OP, 'block_bootstrap_fixed_level.nse_ci95')[0], 3)} to {signed(get(OP, 'block_bootstrap_fixed_level.nse_ci95')[1], 3)}", src("forecast_input", f"value_of_the_forecast.operating_points.{L}.block_bootstrap_fixed_level.nse_ci95"))
        add(f"e2_water_years_above_zero_{L}", get(OP, "block_bootstrap_fixed_level.water_years_above_zero"), f"{get(OP, 'block_bootstrap_fixed_level.water_years_above_zero')} of {get(OP, 'block_bootstrap_fixed_level.n_water_years')}", src("forecast_input", f"value_of_the_forecast.operating_points.{L}.block_bootstrap_fixed_level.water_years_above_zero"), "test water years in which the forecast model warns of more onsets")
        # E2b gains (computed from the stored summaries of the satellite-era split)
        a, b = get(FI, f"summary_test.sets.wqaF_s.{L}.onset.pod"), get(FI, f"summary_test.sets.wqa_s.{L}.onset.pod")
        add(f"e2b_forecast_points_{L}", (a - b) * 100, points(a - b), src("forecast_input", f"summary_test.sets.wqaF_s.{L}.onset.pod minus summary_test.sets.wqa_s.{L}.onset.pod"), "E2b, retrained on the satellite-era split: comparable only inside that split")
        a, b = get(FI, f"summary_test.sets.wqF_sub.{L}.onset.pod"), get(FI, f"summary_test.sets.wq.{L}.onset.pod")
        add(f"e2_plain_gauge_forecast_points_{L}", (a - b) * 100, points(a - b), src("forecast_input", f"summary_test.sets.wqF_sub.{L}.onset.pod minus summary_test.sets.wq.{L}.onset.pod"), "the forecast on the plain gauge model (no rain to midnight)")
        for s, label in (("base_linreg_wqaF", "ridge"), ("base_rf_wqaF", "forest")):
            G = opt(FI, f"yardstick.forecast_added_to_each_model_operating_points.{label}.{L}") or {}
        for model, label in (("ridge regression", "ridge"), ("random forest", "forest"), ("lstm", "lstm")):
            g = opt(FI, f"yardstick.new_input_given_to_each_model.{model}.by_lead.{L}.onsets.difference_points")
            if g is not None:
                add(f"e2_yardstick_{label}_gain_points_{L}", g, pts(g), src("forecast_input", f"yardstick.new_input_given_to_each_model.{model}.by_lead.{L}.onsets.difference_points"), f"what the forecast adds to {model}, onsets warned")
        c = get(FI, f"forecast_data.skill_summary.correlation_all_median_over_basins.{L}")
        add(f"gefs_correlation_{L}", c, n2(c), src("forecast_input", f"forecast_data.skill_summary.correlation_all_median_over_basins.{L}"), "correlation of the control-member rain forecast with the AORC daily rain, median over basins")
        add(f"gefs_bias_factor_median_{L}", get(FI, f"forecast_data.skill_summary.bias_factor_median_over_basins.{L}"), n2(get(FI, f"forecast_data.skill_summary.bias_factor_median_over_basins.{L}")), src("forecast_input", f"forecast_data.skill_summary.bias_factor_median_over_basins.{L}"))
        bs_ = get(FI, f"forecast_data.skill_summary.brier_test_years_pooled.{L}.5mm.calibrated_probability.brier_skill_pooled")
        add(f"gefs_brier_skill_5mm_{L}", bs_, n2(bs_), src("forecast_input", f"forecast_data.skill_summary.brier_test_years_pooled.{L}.5mm.calibrated_probability.brier_skill_pooled"), "calibrated-probability Brier skill for 5 mm days, test years pooled; 'of the same order as' Guan et al. 2022, never 'matches'")
    add("gefs_realtime_margin_worst_minutes", get(FI, "forecast_data.realtime_margin.America/New_York, run 4 h old at the issue time.margin_minutes_of_the_last_file"),
        f"{get(FI, 'forecast_data.realtime_margin.America/New_York, run 4 h old at the issue time.margin_minutes_of_the_last_file'):.0f} min",
        src("forecast_input", "forecast_data.realtime_margin.'America/New_York, run 4 h old at the issue time'.margin_minutes_of_the_last_file"), "negative: the last files are posted about 15 minutes after the issue time in Eastern summer (measured on 4 days of October 2026)")
    add("gefs_realtime_latency_h", get(FI, "forecast_data.realtime_margin.America/New_York, run 4 h old at the issue time.latency_h_of_that_file"), f"{get(FI, 'forecast_data.realtime_margin.America/New_York, run 4 h old at the issue time.latency_h_of_that_file'):.2f} h", src("forecast_input", "forecast_data.realtime_margin.*.latency_h_of_that_file"))
    add("e2_second_half_note", get(FI, "e2_success.second_half_of_the_rule_says_little"), get(FI, "e2_success.second_half_of_the_rule_says_little"), src("forecast_input", "e2_success.second_half_of_the_rule_says_little"))
    for i, rung in enumerate(get(FI, "ladder.main.rungs")):
        add(f"e2_ladder_rung_{i + 1}", rung["rung"], f"{rung['rung']} ({rung['set']})", src("forecast_input", f"ladder.main.rungs.{i}.rung"), rung.get("reads"))

    # ---- E3, unseen rivers, Transformer -------------------------------------------------------------------------------------------------------
    section("7.3 E3 (63 training rivers), rivers never seen (05 and E3) and the Transformer comparison")
    MB = load("more_basins")
    add("e3_outcome", MB["outcome"], MB["outcome"], src("more_basins", "outcome"))
    add("e3_outcome_caveat", MB["outcome_caveat"], MB["outcome_caveat"], src("more_basins", "outcome_caveat"))
    add("e3_outcome_full", MB["outcome_full"], MB["outcome_full"], src("more_basins", "outcome_full"), "quote whole; the 19% -> 17% is MOVED_POINT")
    add("e3_n_trainings", MB["n_trainings"], str(MB["n_trainings"]), src("more_basins", "n_trainings"))
    add("e3_training_seconds", MB["training_seconds"], minutes_text(MB["training_seconds"]), src("more_basins", "training_seconds"), f"{MB['training_seconds']} s")
    add("e3_n_rivers", get(MB, "data_checks.n_rivers"), str(get(MB, "data_checks.n_rivers")), src("more_basins", "data_checks.n_rivers"), "the 12 and 51 others")
    add("e3_other_rivers", len(get(MB, "design.rivers.other_51")), str(len(get(MB, "design.rivers.other_51"))), src("more_basins", "design.rivers.other_51 (length)"))
    add("e3_never_seen_training_rivers", len(get(MB, "design.rivers.other_50")), str(len(get(MB, "design.rivers.other_50"))), src("more_basins", "design.rivers.other_50 (length)"), "one nested river dropped: " + get(MB, "design.rivers.dropped_for_the_never_seen_test.why"))
    for i, c in enumerate(MB["choices_made_after_looking_at_test_scores"]):
        add(f"e3_choice_after_test_{i + 1}", c, c, src("more_basins", f"choices_made_after_looking_at_test_scores.{i}"))
    for i, w in enumerate(MB["what_was_not_done"]):
        add(f"e3_not_done_{i + 1}", w, w, src("more_basins", f"what_was_not_done.{i}"))
    G = get(MB, "gain_rule.lstm63_wq_against_lstm_wq.l1")
    add("e3_nse_12_rivers", get(G, "nse.median_without"), n2(get(G, "nse.median_without")), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.median_without"), "the 12-river LSTM with the gauge, 1 day ahead")
    add("e3_nse_63_rivers", get(G, "nse.median_with"), n2(get(G, "nse.median_with")), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.median_with"), "0.604 -> '0.60'")
    add("e3_nse_gain", get(G, "nse.gain"), signed(get(G, "nse.gain"), 3), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.gain"))
    add("e3_nse_reading", get(G, "nse.reading"), get(G, "nse.reading"), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.reading"), "no clear difference for the same 12 rivers")
    add("e3_nse_basins_higher", get(G, "nse.n_basins_higher"), f"{get(G, 'nse.n_basins_higher')} of 12", src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.n_basins_higher"))
    add("e3_nse_wilcoxon_p", get(G, "nse.wilcoxon_p"), n2(get(G, "nse.wilcoxon_p"), 3), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.nse.wilcoxon_p"))
    add("e3_onsets_12_rivers", get(G, "onset.warned_without"), pct(get(G, "onset.warned_without")), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.warned_without"))
    add("e3_onsets_63_rivers", get(G, "onset.warned_with"), pct(get(G, "onset.warned_with")), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.warned_with"), "MOVED_POINT: fewer onsets warned and fewer false alarms, same CSI")
    add("e3_onsets_12_rivers_pct1", get(G, "onset.warned_without"), pct(get(G, "onset.warned_without"), 1), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.warned_without"))
    add("e3_onsets_63_rivers_pct1", get(G, "onset.warned_with"), pct(get(G, "onset.warned_with"), 1), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.warned_with"))
    add("e3_hits_12_to_63", [get(G, "onset.hits_without"), get(G, "onset.hits_with")], f"{get(G, 'onset.hits_without')} -> {get(G, 'onset.hits_with')}", src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.hits_without/hits_with"))
    add("e3_false_alarms_12_to_63", [get(G, "onset.false_alarms_without"), get(G, "onset.false_alarms_with")], f"{get(G, 'onset.false_alarms_without')} -> {get(G, 'onset.false_alarms_with')}", src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.false_alarms_without/false_alarms_with"))
    add("e3_csi_12_to_63", [get(G, "onset.csi_without"), get(G, "onset.csi_with")], f"{n2(get(G, 'onset.csi_without'), 3)} -> {n2(get(G, 'onset.csi_with'), 3)}", src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.csi_without/csi_with"))
    add("e3_mcnemar_p", get(G, "onset.paired.mcnemar_p"), n2(get(G, "onset.paired.mcnemar_p"), 3), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.paired.mcnemar_p"), "exact McNemar test over the paired onsets")
    add("e3_onset_seed_reading", get(G, "onset.reading"), get(G, "onset.reading"), src("more_basins", "gain_rule.lstm63_wq_against_lstm_wq.l1.onset.reading"), get(G, "onset.reading_is"))
    for L in ("l2", "l3"):
        G2 = get(MB, f"gain_rule.lstm63_wq_against_lstm_wq.{L}")
        add(f"e3_nse_reading_{L}", get(G2, "nse.reading"), get(G2, "nse.reading"), src("more_basins", f"gain_rule.lstm63_wq_against_lstm_wq.{L}.nse.reading"))
        add(f"e3_nse_gain_{L}", get(G2, "nse.gain"), signed(get(G2, "nse.gain"), 3), src("more_basins", f"gain_rule.lstm63_wq_against_lstm_wq.{L}.nse.gain"))
    U = get(MB, "unseen.sets")
    add("unseen_nse_11_rivers_with_properties", get(U, "lobo_w_static.nse_median.l1"), n2(get(U, "lobo_w_static.nse_median.l1")), src("more_basins", "unseen.sets.lobo_w_static.nse_median.l1"), "rivers never seen, 11 training rivers (05_lobo.py, one training per river), 1 day ahead")
    add("unseen_nse_50_rivers_with_properties", get(U, "unseen50_w_static.nse_median.l1"), n2(get(U, "unseen50_w_static.nse_median.l1")), src("more_basins", "unseen.sets.unseen50_w_static.nse_median.l1"), "50 training rivers, three-seed average")
    add("unseen_nse_11_rivers_no_properties", get(U, "lobo_w_nostatic.nse_median.l1"), n2(get(U, "lobo_w_nostatic.nse_median.l1")), src("more_basins", "unseen.sets.lobo_w_nostatic.nse_median.l1"))
    add("unseen_nse_50_rivers_no_properties", get(U, "unseen50_w_nostatic.nse_median.l1"), n2(get(U, "unseen50_w_nostatic.nse_median.l1")), src("more_basins", "unseen.sets.unseen50_w_nostatic.nse_median.l1"))
    A = get(MB, "unseen.against_the_leave_one_out_models.unseen50_w_static_against_lobo_w_static.l1.nse")
    add("unseen_gain_with_properties", A["gain"], signed(A["gain"]), src("more_basins", "unseen.against_the_leave_one_out_models.unseen50_w_static_against_lobo_w_static.l1.nse.gain"), A["reading"] + f"; p = {A['wilcoxon_p']:.3f}; the 05 numbers are single trainings")
    add("unseen_gain_with_properties_reading", A["reading"], A["reading"], src("more_basins", "unseen.against_the_leave_one_out_models.unseen50_w_static_against_lobo_w_static.l1.nse.reading"))
    A = get(MB, "unseen.against_the_leave_one_out_models.unseen50_w_nostatic_against_lobo_w_nostatic.l1.nse")
    add("unseen_gain_no_properties", A["gain"], signed(A["gain"]), src("more_basins", "unseen.against_the_leave_one_out_models.unseen50_w_nostatic_against_lobo_w_nostatic.l1.nse.gain"), A["reading"])
    PX = get(MB, "proximity.summary")
    add("unseen_nearest_gauge_km_11", get(PX, "11_leave_one_out.median_km"), f"{get(PX, '11_leave_one_out.median_km'):.0f} km", src("more_basins", "proximity.summary.11_leave_one_out.median_km"), "median distance to the nearest training gauge in the leave-one-out design")
    add("unseen_nearest_gauge_km_50", get(PX, "50.median_km"), f"{get(PX, '50.median_km'):.0f} km", src("more_basins", "proximity.summary.50.median_km"), "with the 50 other rivers in training: scale and nearness were not separated")
    add("unseen_property_distance_11", get(PX, "11_leave_one_out.median_property_distance"), n2(get(PX, "11_leave_one_out.median_property_distance")), src("more_basins", "proximity.summary.11_leave_one_out.median_property_distance"))
    add("unseen_property_distance_50", get(PX, "50.median_property_distance"), n2(get(PX, "50.median_property_distance")), src("more_basins", "proximity.summary.50.median_property_distance"))
    LB = load("lstm_lobo")
    add("lobo_n_trainings", LB["n_trainings"], str(LB["n_trainings"]), src("lstm_lobo", "n_trainings"))
    add("lobo_training_seconds", LB["training_seconds"], minutes_text(LB["training_seconds"]), src("lstm_lobo", "training_seconds"), f"{LB['training_seconds']} s")
    S1 = get(LB, "summary.l1")
    add("lobo_nse_in_training", get(S1, "nse_median.in_training"), n2(get(S1, "nse_median.in_training")), src("lstm_lobo", "summary.l1.nse_median.in_training"), "lstm_w seed 0, the in-training partner")
    add("lobo_nse_held_out_with_properties", get(S1, "nse_median.held_out_with_properties"), n2(get(S1, "nse_median.held_out_with_properties")), src("lstm_lobo", "summary.l1.nse_median.held_out_with_properties"))
    add("lobo_nse_held_out_without_properties", get(S1, "nse_median.held_out_without_properties"), n2(get(S1, "nse_median.held_out_without_properties")), src("lstm_lobo", "summary.l1.nse_median.held_out_without_properties"))
    add("lobo_nse_forest_held_out", get(S1, "nse_median.forest_held_out"), n2(get(S1, "nse_median.forest_held_out")), src("lstm_lobo", "summary.l1.nse_median.forest_held_out"))
    LO = get(S1, "loss_from_not_seeing_the_basin.with_properties")
    add("lobo_loss_median_nse", LO["median_nse_loss"], n2(LO["median_nse_loss"]), src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.with_properties.median_nse_loss"), "0.126 -> '0.13': the typical basin loses this much NSE when held out of training, 1 day ahead")
    add("lobo_loss_n_basins_lower", LO["n_basins_lower_when_held_out"], f"{LO['n_basins_lower_when_held_out']} of 12", src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.with_properties.n_basins_lower_when_held_out"))
    add("lobo_loss_n_basins_beyond_seed_range", LO["n_basins_loss_larger_than_seed_range"], f"{LO['n_basins_loss_larger_than_seed_range']} of 12", src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.with_properties.n_basins_loss_larger_than_seed_range"))
    add("lobo_loss_wilcoxon_p", LO["wilcoxon_p"], n2(LO["wilcoxon_p"], 3), src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.with_properties.wilcoxon_p"))
    add("lobo_loss_largest_nse", LO["largest_nse_loss"], n2(LO["largest_nse_loss"]), src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.with_properties.largest_nse_loss"), "Charlie Creek, FL")
    add("lobo_loss_median_nse_without_properties", get(S1, "loss_from_not_seeing_the_basin.without_properties.median_nse_loss"), n2(get(S1, "loss_from_not_seeing_the_basin.without_properties.median_nse_loss")), src("lstm_lobo", "summary.l1.loss_from_not_seeing_the_basin.without_properties.median_nse_loss"))
    add("lobo_seed_range_median", get(S1, "in_training_seed_range_median"), n2(get(S1, "in_training_seed_range_median")), src("lstm_lobo", "summary.l1.in_training_seed_range_median"), "the typical seed spread of the in-training NSE (report: seed spread 0.06)")
    WB = LB["worst_basins"][0]
    add("lobo_worst_basin", WB["basin"], WB["basin"], src("lstm_lobo", "worst_basins.0.basin"))
    add("lobo_worst_basin_nse_held_out", WB["nse_held_out_with_properties"][0], n2(WB["nse_held_out_with_properties"][0]), src("lstm_lobo", "worst_basins.0.nse_held_out_with_properties.0"), "1 day ahead, with properties")
    add("lobo_worst_basin_nse_in_training", WB["nse_in_training"][0], n2(WB["nse_in_training"][0]), src("lstm_lobo", "worst_basins.0.nse_in_training.0"))
    HW = LB["higher_when_held_out"][0]
    add("lobo_higher_when_held_out_basin", HW["basin"], HW["basin"], src("lstm_lobo", "higher_when_held_out.0.basin"), "the exception: scores higher when held out")
    add("lobo_higher_when_held_out_nse", get(HW, "nse_1_day_ahead.held_out_with_properties"), n2(get(HW, "nse_1_day_ahead.held_out_with_properties")), src("lstm_lobo", "higher_when_held_out.0.nse_1_day_ahead.held_out_with_properties"), f"against {n2(get(HW, 'nse_1_day_ahead.in_training'))} in training")
    add("lobo_myakka_test_over_training_runoff", HW["test_over_training"], n2(HW["test_over_training"]), src("lstm_lobo", "higher_when_held_out.0.test_over_training"), "share of rain reaching the river, test years over training years")
    tot = [v["test_over_training"] for v in get(LB, "flow_per_rain.basins").values()]
    add("runoff_ratio_shift_min", min(tot), n2(min(tot)), src("lstm_lobo", "flow_per_rain.basins.*.test_over_training (min)"), "test / training share of rain reaching the river: below 1 in every basin (a drier test period)")
    add("runoff_ratio_shift_max", max(tot), n2(max(tot)), src("lstm_lobo", "flow_per_rain.basins.*.test_over_training (max)"))
    add("lobo_finding_0", LB["findings"][0], LB["findings"][0], src("lstm_lobo", "findings.0"))
    add("lobo_finding_1", LB["findings"][1], LB["findings"][1], src("lstm_lobo", "findings.1"))
    # Transformer
    MC = load("model_comparison")
    for i, a in enumerate(MC["answer"]):
        add(f"cmp_answer_{i + 1}", a, a, src("model_comparison", f"answer.{i}"), "the Transformer comparison used its own two-seed LSTM twin with shorter sequences (critic C9)")
    add("cmp_n_trainings", MC["n_trainings"], str(MC["n_trainings"]), src("model_comparison", "n_trainings"))
    add("cmp_training_seconds", MC["training_seconds"], minutes_text(MC["training_seconds"]), src("model_comparison", "training_seconds"), f"{MC['training_seconds']} s")
    for L in LEADS:
        C = get(MC, f"comparison.test.{L}")
        add(f"cmp_lstm_{L}_nse", get(C, "lstm.nse_median"), n2(get(C, "lstm.nse_median"), 3), src("model_comparison", f"comparison.test.{L}.lstm.nse_median"), "the two-seed LSTM twin (cmp_lstm), 365-day sequences, 1,000 updates")
        add(f"cmp_transformer_{L}_nse", get(C, "transformer.nse_median"), n2(get(C, "transformer.nse_median"), 3), src("model_comparison", f"comparison.test.{L}.transformer.nse_median"))
        add(f"cmp_transformer_minus_lstm_{L}", C["nse_median_transformer_minus_lstm"], signed(C["nse_median_transformer_minus_lstm"], 3), src("model_comparison", f"comparison.test.{L}.nse_median_transformer_minus_lstm"), f"seed ranges overlap: {C['seed_ranges_overlap']}; LSTM higher in {C['n_basins_lstm_higher']} of 12 basins")
        add(f"cmp_lstm_{L}_onset_pod", get(C, "lstm.onset_pod"), pct(get(C, "lstm.onset_pod")), src("model_comparison", f"comparison.test.{L}.lstm.onset_pod"))
        add(f"cmp_transformer_{L}_onset_pod", get(C, "transformer.onset_pod"), pct(get(C, "transformer.onset_pod")), src("model_comparison", f"comparison.test.{L}.transformer.onset_pod"))
        add(f"cmp_lstm_{L}_onset_far", get(C, "lstm.onset_far"), pct(get(C, "lstm.onset_far")), src("model_comparison", f"comparison.test.{L}.lstm.onset_far"))
        add(f"cmp_transformer_{L}_onset_far", get(C, "transformer.onset_far"), pct(get(C, "transformer.onset_far")), src("model_comparison", f"comparison.test.{L}.transformer.onset_far"))
    add("cmp_skill_lstm", get(MC, "comparison.test.skill.lstm"), n2(get(MC, "comparison.test.skill.lstm"), 3), src("model_comparison", "comparison.test.skill.lstm"), "mean over the three leads of the median NSE")
    add("cmp_skill_transformer", get(MC, "comparison.test.skill.transformer"), n2(get(MC, "comparison.test.skill.transformer"), 3), src("model_comparison", "comparison.test.skill.transformer"))
    SM = load("same_misses")
    add("same_misses_headline", SM["headline"], SM["headline"], src("same_misses", "headline"))
    add("same_misses_count", get(SM, "l1.missed_by_all_kinds.count"), count(get(SM, "l1.missed_by_all_kinds.count")), src("same_misses", "l1.missed_by_all_kinds.count"), "onsets missed by all four kinds of model (ridge, forest, LSTM, Transformer) with past weather + gauge, 1 day ahead")
    add("same_misses_share", get(SM, "l1.missed_by_all_kinds.share"), pct(get(SM, "l1.missed_by_all_kinds.share"), 1), src("same_misses", "l1.missed_by_all_kinds.share"))
    add("same_misses_share_if_independent", get(SM, "l1.missed_by_all_kinds.share_if_independent"), pct(get(SM, "l1.missed_by_all_kinds.share_if_independent"), 1), src("same_misses", "l1.missed_by_all_kinds.share_if_independent"))
    add("same_misses_rescued_by_evening_rain", get(SM, "l1.of_missed_by_all.warned_with_rain_to_evening.share"), pct(get(SM, "l1.of_missed_by_all.warned_with_rain_to_evening.share"), 1), src("same_misses", "l1.of_missed_by_all.warned_with_rain_to_evening.share"), "of the onsets missed by all kinds: warned by the LSTM given rain up to the evening")
    add("same_misses_rescued_by_perfect_forecast", get(SM, "l1.of_missed_by_all.warned_with_perfect_forecast.share"), pct(get(SM, "l1.of_missed_by_all.warned_with_perfect_forecast.share"), 1), src("same_misses", "l1.of_missed_by_all.warned_with_perfect_forecast.share"), "an upper bound that cannot be had in practice")
    add("same_misses_warned_by_any_kind", get(SM, "l1.warned_by_any_kind.count"), count(get(SM, "l1.warned_by_any_kind.count")), src("same_misses", "l1.warned_by_any_kind.count"))


# ====================================================================================================================================
def build_failures_clock_misses():
    section("7.3 Where the warnings fail (Figure 8), the rain clock (63 hours) and the peaks")
    FA = load("flood_analysis")
    for season, v in get(FA, "failures.by_season.lstm_wq.plain").items():
        k = season.replace("-", "_").lower()
        add(f"fail_season_{k}_onset_pod", v["onset"]["pod"], pct(v["onset"]["pod"]), src("flood_analysis", f"failures.by_season.lstm_wq.plain.{season}.onset.pod"), "LSTM with gauge, onsets warned 1 day ahead")
        add(f"fail_season_{k}_onset_n", v["onset"]["hits"] + v["onset"]["misses"], count(v["onset"]["hits"] + v["onset"]["misses"]), src("flood_analysis", f"failures.by_season.lstm_wq.plain.{season}.onset.hits + misses"))
    for st, v in get(FA, "failures.by_storm_type.lstm_wq.plain").items():
        k = st.replace("-", "_")
        add(f"fail_storm_{k}_onset_pod", v["onset"]["pod"], pct(v["onset"]["pod"]), src("flood_analysis", f"failures.by_storm_type.lstm_wq.plain.{st}.onset.pod"), "LSTM with gauge, onsets warned 1 day ahead")
        add(f"fail_storm_{k}_onset_n", v["onset"]["hits"] + v["onset"]["misses"], count(v["onset"]["hits"] + v["onset"]["misses"]), src("flood_analysis", f"failures.by_storm_type.lstm_wq.plain.{st}.onset.hits + misses"))
    for cls, v in get(FA, "failures.by_rain_timing.lstm_wq.plain").items():
        k = cls.replace("-", "_").replace("%", "").replace(" ", "_")
        add(f"fail_rain_{k}_onset_pod", v["onset"]["pod"], pct(v["onset"]["pod"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wq.plain.{cls}.onset.pod"), f"onsets with {cls} of the two-day rain on the target day; LSTM with gauge, 1 day ahead")
        add(f"fail_rain_{k}_onset_n", v["onset"]["hits"] + v["onset"]["misses"], count(v["onset"]["hits"] + v["onset"]["misses"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wq.plain.{cls}.onset.hits + misses"))
        add(f"fail_rain_{k}_onset_far", v["onset"]["far"], pct(v["onset"]["far"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wq.plain.{cls}.onset.far"))
        add(f"fail_rain_{k}_onset_hits", v["onset"]["hits"], count(v["onset"]["hits"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wq.plain.{cls}.onset.hits"), "the count behind the share (critic 1, item 15)")
    for cls, v in get(FA, "failures.by_rain_timing.lstm_wqf.plain").items():      # the same classes for the LSTM given the perfect weather forecast (critic 1, item 15)
        k = cls.replace("-", "_").replace("%", "").replace(" ", "_")
        add(f"fail_rain_{k}_wqf_onset_pod", v["onset"]["pod"], pct(v["onset"]["pod"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wqf.plain.{cls}.onset.pod"),
            f"onsets with {cls} of the two-day rain on the target day; LSTM with gauge and the perfect weather forecast, an upper bound that cannot be had in practice")
        add(f"fail_rain_{k}_wqf_onset_hits", v["onset"]["hits"], count(v["onset"]["hits"]), src("flood_analysis", f"failures.by_rain_timing.lstm_wqf.plain.{cls}.onset.hits"))
    PR = get(FA, "failures.perfect_forecast_rescue.plain.all_onsets")
    for k in ("caught_with_gauge", "caught_with_perfect_forecast", "missed_with_gauge", "of_those_caught_with_perfect_forecast", "missed_by_both", "caught_with_gauge_but_not_with_forecast"):
        add(f"rescue_{k}", PR[k], count(PR[k]), src("flood_analysis", f"failures.perfect_forecast_rescue.plain.all_onsets.{k}"), "1 day ahead; the perfect weather forecast is an upper bound that cannot be had in practice")
    for cls, v in get(FA, "failures.perfect_forecast_rescue.plain.by_rain_class").items():
        k = cls.replace("-", "_").replace("%", "").replace(" ", "_")
        share = v["caught_with_perfect_forecast"] / v["onsets"] if v["onsets"] else None
        add(f"rescue_rain_{k}_perfect_share", share, pct(share), src("flood_analysis", f"failures.perfect_forecast_rescue.plain.by_rain_class.{cls}.caught_with_perfect_forecast / onsets"), f"{v['caught_with_perfect_forecast']} of {v['onsets']} onsets warned by the LSTM given the perfect weather forecast")
    SPm = get(FA, "failures.by_basin.lstm_wq.spearman.plain")
    add("fail_area_vs_onset_pod_rho", get(SPm, "area_km2.onset_pod.rho"), f"{get(SPm, 'area_km2.onset_pod.rho'):+.2f}", src("flood_analysis", "failures.by_basin.lstm_wq.spearman.plain.area_km2.onset_pod.rho"), f"Spearman over 12 basins, p = {get(SPm, 'area_km2.onset_pod.p'):.3f}")
    add("fail_area_vs_onset_pod_p", get(SPm, "area_km2.onset_pod.p"), n2(get(SPm, "area_km2.onset_pod.p"), 3), src("flood_analysis", "failures.by_basin.lstm_wq.spearman.plain.area_km2.onset_pod.p"))
    add("fail_slope_vs_onset_pod_rho", get(SPm, "slope_m_per_km.onset_pod.rho"), f"{get(SPm, 'slope_m_per_km.onset_pod.rho'):+.2f}", src("flood_analysis", "failures.by_basin.lstm_wq.spearman.plain.slope_m_per_km.onset_pod.rho"), f"p = {get(SPm, 'slope_m_per_km.onset_pod.p'):.1e}")
    add("fail_rho_needed_p05", get(FA, "failures.spearman_rho_needed_for_p05"), n2(get(FA, "failures.spearman_rho_needed_for_p05"), 3), src("flood_analysis", "failures.spearman_rho_needed_for_p05"), f"{get(FA, 'failures.spearman_tests_per_model_and_rule')} tests per model and rule: no multiple-comparison correction")
    PK = get(FA, "failures.peak_size.lstm_wq.all_events")
    add("peak_forecast_share_median", PK["median"], pct(PK["median"]), src("flood_analysis", "failures.peak_size.lstm_wq.all_events.median"), "the 1-day forecast of the peak as a share of the observed peak, median over the 611 events (LSTM with gauge)")
    add("peak_forecast_share_below_half", PK["share_below_half"], pct(PK["share_below_half"]), src("flood_analysis", "failures.peak_size.lstm_wq.all_events.share_below_half"))
    add("peak_forecast_share_median_perfect", get(FA, "failures.peak_size.lstm_wqf.all_events.median"), pct(get(FA, "failures.peak_size.lstm_wqf.all_events.median")), src("flood_analysis", "failures.peak_size.lstm_wqf.all_events.median"), "given the perfect weather forecast (upper bound)")
    add("peak_forecast_below_level_share", get(FA, "failures.peak_size.lstm_wq.forecast_below_high_flow_level_on_peak_day"), pct(get(FA, "failures.peak_size.lstm_wq.forecast_below_high_flow_level_on_peak_day")), src("flood_analysis", "failures.peak_size.lstm_wq.forecast_below_high_flow_level_on_peak_day"), "share of the events whose peak-day forecast stays below the high-flow level")
    add("warning_level_sentence", FA["findings"][11], FA["findings"][11], src("flood_analysis", "findings.11"), "a level of x0.70 would warn of 39% of the onsets with 83% false (description only)")
    WL = FA["warning_levels"]
    add("warning_levels_range", [min(WL), max(WL)], f"x{min(WL):.2f} to x{max(WL):.2f}", src("flood_analysis", "warning_levels (min, max)"), f"{len(WL)} levels tried on the validation years")
    add("warning_levels_n", len(WL), str(len(WL)), src("flood_analysis", "warning_levels (length)"))
    EV = FA["event_shown"]
    for k in ("basin", "onset_date", "peak_date", "duration_days", "storm_type", "cyclone_name", "cyclone_km", "rain_issue_day_mm", "rain_target_day_mm", "onsets_with_this_pattern", "rule"):
        v = EV[k]
        text = v if isinstance(v, str) else (f"{v:.0f}" if isinstance(v, float) else str(v))
        add(f"event_shown_{k}", v, text, src("flood_analysis", f"event_shown.{k}"), "the example event of Figure 5.3: picked by rule, not a measure of typical skill")
    for i, row in enumerate(get(FA, "verdict.table")):
        base = f"verdict_table_{row['model']}_l{row['lead']}_{row['kind']}"
        add(f"{base}_verdict", row["verdict"], row["verdict"], src("flood_analysis", f"verdict.table.{i}.verdict"), f"{row['label']}, {row['lead']} day(s), {row['kind']}, level x{row['trigger']:.2f} chosen on validation")
        add(f"{base}_trigger", row["trigger"], f"x{row['trigger']:.2f}", src("flood_analysis", f"verdict.table.{i}.trigger"))
        add(f"{base}_pod", row["pod"], pct(row["pod"]), src("flood_analysis", f"verdict.table.{i}.pod"))
        add(f"{base}_far", row["far"], pct(row["far"]), src("flood_analysis", f"verdict.table.{i}.far"))
        add(f"{base}_csi", row["csi"], n2(row["csi"]), src("flood_analysis", f"verdict.table.{i}.csi"))
        add(f"{base}_basins", [row["basins_go"], row["basins_caution"], row["basins_not_yet"]], f"GO {row['basins_go']}, CAUTION {row['basins_caution']}, NOT YET {row['basins_not_yet']}", src("flood_analysis", f"verdict.table.{i}.basins_*"), "per-basin verdicts")
    # rain clock (63-hour window)
    RC = load("onset_rain_clock")
    add("clock_what", RC["what"], RC["what"], src("onset_rain_clock", "what"), "defines the 63-hour window (09:00 of t-1 to 24:00 of t+1); never cite ROADMAP.md section 2 for the window (critic C7)")
    add("clock_window_hours", 63, "63 hours", src("onset_rain_clock", "what") + " (24 + 15 + 9 + 15 hours: the table's day, the hidden hours 09:00-24:00, and the two future windows; critic C7 and review_documents_2026-10-08 discrepancies[1])")
    add("clock_n_onsets", RC["n_onsets"], count(RC["n_onsets"]), src("onset_rain_clock", "n_onsets"))
    mm = get(RC, "all.mean_mm")
    for k, v in mm.items():
        add(f"clock_mean_mm_{k}", v, f"{v} mm", src("onset_rain_clock", f"all.mean_mm.{k}"), {"table": "rain in the table at the issue time (09:00 of t-1 to 09:00 of t)", "hidden": "fallen before the issue time but filed under the next day (09:00 to 24:00 of t)", "future1": "the target day t+1", "future2": "the day after, t+2"}[k])
    total = sum(mm.values())
    shares = {"table": mm["table"] / total, "hidden": mm["hidden"] / total, "future": (mm["future1"] + mm["future2"]) / total}
    # largest-remainder rounding so that the three printed shares add up to 100
    floors = {k: int(100 * v) for k, v in shares.items()}
    rem = sorted(shares, key=lambda k: -(100 * shares[k] - floors[k]))
    for k in rem[: 100 - sum(floors.values())]:
        floors[k] += 1
    add("clock_total_mm", total, f"{total:.1f} mm", src("onset_rain_clock", "all.mean_mm (sum)"), "mean rain of the 63-hour window around an onset")
    for k, label in (("table", "in the table at the issue time"), ("hidden", "fallen but filed under the next day"), ("future", "still to come (t+1 and t+2)")):
        add(f"clock_share_{k}", shares[k], f"{floors[k]}%", src("onset_rain_clock", "all.mean_mm (share of the sum)"),
            f"{100 * shares[k]:.1f}% exactly; printed with largest-remainder rounding so that the three shares add up to 100 (18/35/47); {label}")
        add(f"clock_share_{k}_exact", shares[k], pct(shares[k], 1), src("onset_rain_clock", "all.mean_mm (share of the sum)"))
    add("clock_median_share_known_by_table", get(RC, "all.median_share_known_by_table"), pct(get(RC, "all.median_share_known_by_table")), src("onset_rain_clock", "all.median_share_known_by_table"), "median onset: share of its 63-hour rain known from the rain table at the issue time")
    add("clock_median_share_known_by_midnight", get(RC, "all.median_share_known_by_midnight"), pct(get(RC, "all.median_share_known_by_midnight")), src("onset_rain_clock", "all.median_share_known_by_midnight"))
    MISS = RC["missed_by_lstm_wq"]
    add("clock_missed_n", MISS["n"], count(MISS["n"]), src("onset_rain_clock", "missed_by_lstm_wq.n"), "onsets missed by the LSTM with gauge 1 day ahead")
    add("clock_missed_median_share_known_by_table", MISS["median_share_known_by_table"], pct(MISS["median_share_known_by_table"]), src("onset_rain_clock", "missed_by_lstm_wq.median_share_known_by_table"), "of the missed onsets: median share of their rain known at the issue time (4%)")
    add("clock_missed_median_share_known_by_midnight", MISS["median_share_known_by_midnight"], pct(MISS["median_share_known_by_midnight"]), src("onset_rain_clock", "missed_by_lstm_wq.median_share_known_by_midnight"), "with the rain to midnight (50%)")
    for k, v in MISS["mean_mm"].items():
        add(f"clock_missed_mean_mm_{k}", v, f"{v} mm", src("onset_rain_clock", f"missed_by_lstm_wq.mean_mm.{k}"))
    for cls, v in RC["by_rain_class"].items():
        k = cls.replace("-", "_").replace("%", "").replace(" ", "_")
        add(f"clock_class_{k}_n", v["n"], count(v["n"]), src("onset_rain_clock", f"by_rain_class.{cls}.n"), f"onsets with {cls} of the two-day rain on the target day")
        add(f"clock_class_{k}_median_share_known_by_table", v["median_share_known_by_table"], pct(v["median_share_known_by_table"]), src("onset_rain_clock", f"by_rain_class.{cls}.median_share_known_by_table"))
        add(f"clock_class_{k}_median_share_known_by_midnight", v["median_share_known_by_midnight"], pct(v["median_share_known_by_midnight"]), src("onset_rain_clock", f"by_rain_class.{cls}.median_share_known_by_midnight"))
    MW = RC["missed_by_wq_but_warned_by_perfect"]
    add("clock_missed_but_perfect_n", MW["n"], count(MW["n"]), src("onset_rain_clock", "missed_by_wq_but_warned_by_perfect.n"), "onsets missed by the LSTM with gauge and warned by the perfect-forecast model")


# ====================================================================================================================================
def build_uncertainty():
    section("6.3 Resampling intervals and verdict stability (results/uncertainty.json)")
    UN = load("uncertainty")
    add("unc_how_to_read", UN["how_to_read"], UN["how_to_read"], src("uncertainty", "how_to_read"), "the intervals are not a proof and do not contain the seed-to-seed difference")
    add("unc_b", get(UN, "resampling.B"), count(get(UN, "resampling.B")), src("uncertainty", "resampling.B"), "resamples")
    add("unc_seed", get(UN, "resampling.seed"), str(get(UN, "resampling.seed")), src("uncertainty", "resampling.seed"))
    add("unc_units", get(UN, "resampling.units"), get(UN, "resampling.units"), src("uncertainty", "resampling.units"))
    add("unc_interval", get(UN, "resampling.interval"), get(UN, "resampling.interval"), src("uncertainty", "resampling.interval"))
    add("unc_n_years", get(UN, "resampling.n_years"), str(get(UN, "resampling.n_years")), src("uncertainty", "resampling.n_years"))
    add("unc_n_stats", len(UN["stats"]), count(len(UN["stats"])), src("uncertainty", "stats (length)"))
    add("unc_sets_added_2026_10_08", get(UN, "added_2026_10_08.n_sets_added"), str(get(UN, "added_2026_10_08.n_sets_added")), src("uncertainty", "added_2026_10_08.n_sets_added"), "prediction sets of scripts 15-18 added on 8 October 2026: a range stands beside every new headline number")
    wanted = [("onsets_warned", s, L) for s in ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqh", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqaP", "lstm_wqf", "lstm63_wq", "lstm_wqa_to09", "base_linreg_wq", "base_rf_wq") for L in LEADS]
    wanted += [("onset_false_warnings", s, L) for s in ("lstm_wq", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqf", "lstm_wqn") for L in LEADS]
    wanted += [("nse_median", s, L) for s in ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqh", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqaP", "lstm_wqf", "lstm63_wq", "cmp_lstm", "cmp_transformer", "base_linreg_wq", "base_rf_wq") for L in LEADS]
    for stat, s, L in wanted:
        k = f"{stat}.{s}.{L}"
        e = UN["stats"].get(k)
        if not e:
            continue
        fmt = pct if stat != "nse_median" else n2
        add(f"unc_{stat}_{s}_{L}", e["value"], fmt(e["value"]), src("uncertainty", f"stats['{k}'].value"))
        add(f"unc_{stat}_{s}_{L}_interval", [e["both"]["lo"], e["both"]["hi"]], f"{fmt(e['both']['lo'])} to {fmt(e['both']['hi'])}", src("uncertainty", f"stats['{k}'].both"), "95% interval, test years and basins drawn again (design 'both')")
    for pair in ("lstm_wq_minus_best_simple", "lstm_w_minus_best_simple", "wqa_minus_wq", "wqaF_sub_minus_wqa", "transformer_minus_lstm", "wqn_minus_wq", "wqh_minus_wq", "lstm63_wq_minus_wq"):
        for L in LEADS:
            k = f"nse_median_of_diffs.{pair}.{L}"
            e = UN["stats"].get(k)
            if not e:
                continue
            add(f"unc_nse_diff_{pair}_{L}", e["value"], signed(e["value"]), src("uncertainty", f"stats['{k}'].value"), "median over the basins of the NSE difference in the same basin (the typical basin)")
            add(f"unc_nse_diff_{pair}_{L}_interval", [e["both"]["lo"], e["both"]["hi"]], f"{signed(e['both']['lo'])} to {signed(e['both']['hi'])}", src("uncertainty", f"stats['{k}'].both"), f"share of resamples above zero: {e['both']['share_above_zero']:.3f}")
            add(f"unc_nse_diff_{pair}_{L}_share_above_zero", e["both"]["share_above_zero"], pct(e["both"]["share_above_zero"]), src("uncertainty", f"stats['{k}'].both.share_above_zero"))
    for pair in ("wqa_minus_wq", "wqaF_sub_minus_wqa", "wqn_minus_wq", "wqh_minus_wq", "lstm63_wq_minus_wq"):
        for L in LEADS:
            k = f"onsets_diff.{pair}.{L}"
            e = UN["stats"].get(k)
            if not e:
                continue
            add(f"unc_onsets_diff_{pair}_{L}", e["value"], points(e["value"]), src("uncertainty", f"stats['{k}'].value"), "difference in the share of onsets warned, percentage points")
            add(f"unc_onsets_diff_{pair}_{L}_interval", [e["both"]["lo"], e["both"]["hi"]], f"{points(e['both']['lo'])} to {points(e['both']['hi'])}", src("uncertainty", f"stats['{k}'].both"))
    VS = UN["verdict_stability"]
    for s in ("lstm_wq", "lstm_w", "lstm_wqn", "lstm_wqh", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqaP", "lstm_wqf", "lstm63_wq"):
        for kind in ("onset", "all"):
            for L in LEADS:
                k = f"{s}.{kind}.{L}"
                v = VS.get(k)
                if not v:
                    continue
                sh = v["share_of_resamples_with_each_verdict"]
                add(f"vs_{s}_{kind}_{L}", v["verdict"], v["verdict"], src("uncertainty", f"verdict_stability['{k}'].verdict"), f"share of resamples: GO {sh['GO']:.3f}, CAUTION {sh['CAUTION']:.3f}, NOT YET {sh['NOT YET']:.3f}")
                for vd in ("GO", "CAUTION", "NOT YET"):
                    add(f"vs_{s}_{kind}_{L}_{vd.replace(' ', '_').lower()}_share", sh[vd], pct(sh[vd], 1), src("uncertainty", f"verdict_stability['{k}'].share_of_resamples_with_each_verdict.{vd}"))
    MD = get(UN, "spread_and_skill.medians")
    for s in ("lstm_wq", "lstm_wqf", "base_rf_wq", "base_linreg_wq", "base_persistence", "lstm_w"):
        for L in LEADS:
            for part in ("spread_ratio", "correlation", "total_ratio", "skill_vs_persistence", "kge"):
                k = f"{part}_median.{s}.{L}"
                if k in MD:
                    add(f"ss_{part}_{s}_{L}", MD[k], n2(MD[k]), src("uncertainty", f"spread_and_skill.medians['{k}']"), {"spread_ratio": "standard deviation of the forecasts over that of the observed flow (below 1: peaks too low)",
                        "correlation": "correlation of forecast and observed flow", "total_ratio": "mean forecast over mean observed", "skill_vs_persistence": "1 minus squared error over that of 'tomorrow = today'", "kge": "Kling-Gupta efficiency"}[part] + "; median of the 12 basins")


# ====================================================================================================================================
# the runs whose computing time the report adds up (code/report_sections/s7_cost.py RUNS, SECONDS_KEYS, COUNT_KEYS, replicated here)
RUNS = ("baselines", "lstm_single", "lstm_all", "lstm_lobo", "rain_timing", "rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input", "satellite_inputs", "model_comparison")
SECONDS_KEYS = ("training_seconds", "training_seconds_total", "train_seconds", "fit_seconds", "runtime_s", "runtime_seconds", "seconds")
COUNT_KEYS = ("n_trainings", "n_training_runs", "n_runs")


def _first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _counts_blocks(rv):
    """The counts blocks of a review record: its own and those of its later rounds (as code/build_slides.py _counts_blocks)."""
    return [rv.get("counts") or {}] + [v.get("counts") or {} for k, v in rv.items() if k.startswith("round") and isinstance(v, dict)]


def build_reviews_and_work():
    section("6.1/4.3/10.3 The work in numbers: trainings, hours, stored forecasts, reviews, money (slides 6 and 14, T12)")
    DF = load("data_flow")
    ST = get(DF, "all_information.stored")
    add("stored_prediction_sets", ST["prediction_sets"], count(ST["prediction_sets"]), src("data_flow", "all_information.stored.prediction_sets"), "results/preds/*.npz on 2026-10-08")
    add("stored_single_training_sets", ST["single_training_sets"], count(ST["single_training_sets"]), src("data_flow", "all_information.stored.single_training_sets"))
    add("stored_saved_networks", ST["saved_networks"], count(ST["saved_networks"]), src("data_flow", "all_information.stored.saved_networks"), "results/models/*.pt: scripts 15-18 only; the headline sets of 04 (lstm_wq etc.) were never saved (handoff/models_inventory.md)")
    add("trainings_recorded", ST["trainings_recorded"], count(ST["trainings_recorded"]), src("data_flow", "all_information.stored.trainings_recorded"), "network trainings recorded in the results files (critic C4)")
    for k, v in ST["trainings_recorded_by_file"].items():
        add(f"trainings_in_{k.replace('.json', '')}", v, str(v), src("data_flow", f"all_information.stored.trainings_recorded_by_file.{k}"))
    for k, v in ST["saved_networks_by_script"].items():
        add(f"saved_networks_{k.split('_')[0]}", v, str(v), src("data_flow", f"all_information.stored.saved_networks_by_script.{k}"), k)
    for k, v in ST["prediction_sets_by_family"].items():
        add(f"prediction_sets_family_{k}", v, str(v), src("data_flow", f"all_information.stored.prediction_sets_by_family.{k}"))
    add("preds_folder_bytes", get(ST, "folder_sizes.results/preds"), size_text(get(ST, "folder_sizes.results/preds")), src("data_flow", "all_information.stored.folder_sizes.results/preds"))
    add("models_folder_bytes", get(ST, "folder_sizes.results/models"), size_text(get(ST, "folder_sizes.results/models")), src("data_flow", "all_information.stored.folder_sizes.results/models"))
    # computing time and trainings as section 7 of the report adds them up
    hp_rows = list(csv.DictReader((RES / "hparam_log.csv").open(encoding="utf-8")))
    found = []
    for name in RUNS:
        D = load(name)
        if D.get("quick_mode"):
            continue
        secs = _first(D, SECONDS_KEYS)
        n = _first(D, COUNT_KEYS)
        if n is None and name == "lstm_single":
            n = len(hp_rows)
        found.append((name, float(secs), None if n is None else int(n)))
        win = any((p.get("seeds_trained") and min(p["seeds_trained"]) >= 3) or p.get("seed_extension") for p in D.get("passes") or [])
        add(f"run_{name}_seconds", secs, minutes_text(secs), src(name, next(k for k in SECONDS_KEYS if k in D)), f"{secs} s on the Mac" + (" (seeds 0-2) and the Windows PC (seeds 3-4, 30 October 2026)" if win else ""))
        add(f"run_{name}_trainings", n, str(n) if n is not None else "n/a", src(name, next((k for k in COUNT_KEYS if k in D), "hparam_log.csv rows")) if name != "lstm_single" else "results/hparam_log.csv (rows)")
    hours = sum(s for _, s, _ in found) / 3600
    n_all = sum(n for _, _, n in found if n)
    add("work_trainings", n_all, count(n_all), "results/*.json n_trainings of the runs in code/report_sections/s7_cost.py RUNS + results/hparam_log.csv rows (replicated here)", f"{n_all} = {n_all - len(hp_rows)} recorded in the results files of these runs + {len(hp_rows)} one-river tuning runs (critic C4); includes the seed-3/4 trainings of 30 October 2026 (seed_extension_trainings)")
    add("work_hours", hours, f"{hours:.1f} h", "sum of the timing fields of the same runs / 3600 (report section 7 'What the whole project cost to train')", f"{sum(s for _, s, _ in found):.1f} s in all, on a two-core laptop without a graphics card and (seeds 3-4 of scripts 15-18) the Windows PC, CPU only")
    watt = get(load("model_comparison"), "settings.assumed_power_watt")
    add("work_wh_assumed", sum(s for _, s, _ in found) * watt / 3600, f"about {count(sum(s for _, s, _ in found) * watt / 3600)} Wh", "work_hours x " + src("model_comparison", "settings.assumed_power_watt"), "at an ASSUMED 15 W; nothing was measured")
    # reviews: the experiment records (results/review_<experiment>.json) are counted apart from the passes over the documents
    # (results/review_documents_*.json), so review_records / review_rounds / review_checks_listed do not change when a new document
    # pass is recorded (critic 2 of 2026-10-29, item 10); the document passes have keys of their own (document_review_*)
    # review_gis_layers.json (30 October 2026) reviews the GIS-layer export (deliverable D3), not an experiment: it is left out of the experiment counts
    # as well, and has keys of its own (gis_layers_review_verdict and gis_layers_review_date in the GIS section, rev_gis_layers_* below)
    recs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RES.glob("review_*.json"))}
    # review_deck*_<date>.json (30 October 2026) reviews a slide deck, not an experiment: left out of the experiment counts too
    exp = {k: v for k, v in recs.items() if not k.startswith(("review_documents_", "review_deck")) and k != "review_gis_layers"}
    docs = {k: v for k, v in recs.items() if k.startswith("review_documents_")}
    blocks = {k: _counts_blocks(v) for k, v in recs.items()}
    cs_all = [c for b in blocks.values() for c in b]
    cs = [c for k, b in blocks.items() if k in exp for c in b]
    add("review_records", len(exp), str(len(exp)), "results/review_*.json excluding review_documents_*.json and review_gis_layers.json (count)", "the experiment review records: " + ", ".join(sorted(exp)))
    add("review_records_all", len(recs), str(len(recs)), "results/review_*.json (count, the document passes included)", ", ".join(sorted(recs)))
    add("review_rounds", len(cs), str(len(cs)), "counts blocks (own + round_2, round_3, ...) of the experiment review records, as code/build_slides.py _counts_blocks counts them",
        ", ".join(f"{k.replace('review_', '')} {len(blocks[k])}" for k in sorted(exp)))
    n_all_rounds = sum(len(_rounds(v)) for v in recs.values())
    add("review_rounds_total", n_all_rounds, str(n_all_rounds), "results/review_*.json: 1 + the round_N blocks of every record (experiment records, document passes and the GIS-layer review)",
        ", ".join(f"{k.replace('review_', '')} {len(_rounds(v))}" for k, v in sorted(recs.items())))
    add("review_checks_listed", sum(c.get("checks_listed", 0) for c in cs), count(sum(c.get("checks_listed", 0) for c in cs)), "sum of checks_listed over every counts block of the experiment review records",
        "E4a, E4b and E4e list no checks_listed field: they count values recomputed instead (rev_warning_policy_values_compared, rev_within_day_inputs_values_compared, rev_flow_state_values_compared)")
    # review_none_failed rests on the numbers only: numerical mismatches / discrepancies and failed checks of a NUMBER (_number_failures);
    # the failed checks about wording, disclosure or bookkeeping are counted apart (review_wording_findings_open)
    exp_rounds = [(k, n, pre, R) for k in sorted(exp) for n, pre, R in _rounds(exp[k])]
    n_mism = sum(((R.get("counts") or {}).get("numerical_mismatches") or 0) + ((R.get("counts") or {}).get("numerical_discrepancies") or 0) for _, _, _, R in exp_rounds)
    n_num_fail = sum(_number_failures(R)[0] for _, _, _, R in exp_rounds)
    none_failed = n_mism == 0 and n_num_fail == 0
    add("review_none_failed", none_failed, "no number differed and no check of a number failed" if none_failed else "a number differed or a check of a number failed (see the records)",
        "numerical_mismatches and numerical_discrepancies of every counts block of the experiment review records, and their failed checks that record mismatches (or that no listed check explains)",
        f"{n_mism} numerical mismatches, {n_num_fail} failed number checks over {len(exp_rounds)} rounds; failed checks about wording, disclosure or bookkeeping are counted apart (review_wording_findings_open)")
    add("review_number_checks_failed", n_num_fail, str(n_num_fail), "failed checks of the experiment review records that record mismatches, plus counts.failed not explained by a listed check (every round)")
    latest = {k: _rounds(exp[k])[-1] for k in sorted(exp)}
    wf = {k: _number_failures(R)[1] for k, (_, _, R) in latest.items()}
    add("review_wording_findings_open", sum(wf.values()), str(sum(wf.values())), "the failed checks of the LATEST round of each experiment review record that are not about a number (result 'fail' / passed false, no mismatches)",
        "about wording, disclosure or the scripts' bookkeeping, as recorded at review time; a fix made after the review shows only in a later round. By record: "
        + ", ".join(f"{k.replace('review_', '')} {v}" for k, v in wf.items()))
    mf = {k: ((R.get("counts") or {}).get("must_fix") or 0) for k, (_, _, R) in latest.items()}
    add("review_latest_must_fix", sum(mf.values()), str(sum(mf.values())), "counts.must_fix of the LATEST round of each experiment review record (sum)",
        "wording and disclosure items as recorded at review time; by record: " + ", ".join(f"{k.replace('review_', '')} {v}" for k, v in mf.items()))
    for k, (n, pre, R) in latest.items():
        tag = k.replace("review_", "")
        c = R.get("counts") or {}
        d = _round_date(R)

        def rsrc(f, k=k, pre=pre):
            return src(k, pre + f)
        add(f"rev_{tag}_latest_round", n, str(n), src(k, (pre.rstrip(".") if pre else "top level") + " (the highest round_N; round 1 is the top level)"))
        add(f"rev_{tag}_latest_who", R.get("who"), str(R.get("who")), rsrc("who"), "an AI agent, not a team member")
        add(f"rev_{tag}_latest_date", d, long_date(d) if d else "n/a", rsrc("dates.review" if opt(R, "dates.review") else "when"), R.get("when"))
        add(f"rev_{tag}_latest_verdict", R.get("verdict"), str(R.get("verdict")), rsrc("verdict"))
        vc = c.get("values_compared", c.get("numbers_compared_in_all"))
        add(f"rev_{tag}_latest_values_compared", vc, count(vc) if vc is not None else "n/a", rsrc("counts.values_compared" if "values_compared" in c else "counts.numbers_compared_in_all"),
            None if vc is not None else "this round records no count of values compared")
        mm = c.get("numerical_mismatches", c.get("numerical_discrepancies"))
        add(f"rev_{tag}_latest_mismatches", mm, count(mm) if mm is not None else "n/a", rsrc("counts.numerical_mismatches" if "numerical_mismatches" in c else "counts.numerical_discrepancies"))
        add(f"rev_{tag}_latest_must_fix", c.get("must_fix"), str(c.get("must_fix")), rsrc("counts.must_fix"), "wording and disclosure items as recorded at review time")
        add(f"rev_{tag}_latest_covers", R.get("covers"), str(R.get("covers") or "n/a"), rsrc("covers"), None if R.get("covers") else "this round names no covers field")
    fs = [(k, pre, R) for k, n, pre, R in exp_rounds if _round_date(R) == "2026-10-30" and "five-seed" in str(R.get("covers", ""))]
    fs_vc = sum((R.get("counts") or {}).get("values_compared", 0) for _, _, R in fs)
    fs_mm = sum(((R.get("counts") or {}).get("numerical_mismatches") or 0) + ((R.get("counts") or {}).get("numerical_discrepancies") or 0) for _, _, R in fs)
    add("review_five_seed_values_compared", fs_vc, count(fs_vc), " + ".join(f"results/{k}.json -> {pre}counts.values_compared" for k, pre, _ in fs),
        f"values the independent reviewers recomputed for the five-seed numbers on 30 October 2026, {len(fs)} rounds (the rounds dated 2026-10-30 whose covers names the five-seed numbers)")
    add("review_five_seed_mismatches", fs_mm, count(fs_mm), " + ".join(f"results/{k}.json -> {pre}counts.numerical_mismatches" for k, pre, _ in fs),
        f"over the same {len(fs)} rounds")
    doc_dates = []
    for name, rv in docs.items():
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(rv.get("when", "")))
        doc_dates.append((m.group(0) if m else "", name))
    doc_dates.sort()
    uniq = []
    for d, _ in doc_dates:
        if d and d not in uniq:
            uniq.append(d)
    parts = []
    for d in uniq:
        n = sum(1 for x, _ in doc_dates if x == d)
        parts.append((f"{['', 'one', 'two', 'three', 'four'][n] if n < 5 else n} passes on " if n > 1 else "") + long_date(d))   # no parentheses: the text stands inside brackets
    if len(parts) > 2:                                              # "A, B and C", not "A and B and C" (critic 3 of 2026-10-30)
        parts = [", ".join(parts[:-1]), parts[-1]]
    add("document_review_passes", len(docs), str(len(docs)), "results/review_documents_*.json (count)", "the critic passes over the documents, apart from the experiment records: " + ", ".join(sorted(docs)))
    add("document_review_dates", [d for d, _ in doc_dates], " and ".join(parts) if parts else "n/a", "results/review_documents_*.json -> when (the first date each names)")
    add("document_review_latest", doc_dates[-1][0] if doc_dates else None, long_date(doc_dates[-1][0]) if doc_dates else "n/a", "results/review_documents_*.json -> when (the latest pass)", doc_dates[-1][1] if doc_dates else None)
    ncb = []                                                        # every counts block that carries numbers_compared_in_all (critic 1, item 18: the sum names all of them)
    for name, rv in recs.items():
        for path, c in [("counts", rv.get("counts") or {})] + [(f"{k}.counts", v.get("counts") or {}) for k, v in rv.items() if k.startswith("round") and isinstance(v, dict)]:
            if c.get("numbers_compared_in_all"):
                ncb.append(f"results/{name}.json -> {path}.numbers_compared_in_all")
    add("review_numbers_compared", sum(c.get("numbers_compared_in_all", 0) for c in cs_all), count(sum(c.get("numbers_compared_in_all", 0) for c in cs_all)), " + ".join(ncb),
        "numbers the final critic compared against the results files on 8 October 2026, summed over every counts block that has the field")
    add("review_must_fix_total", sum(c.get("must_fix", 0) for c in cs), str(sum(c.get("must_fix", 0) for c in cs)), "sum of must_fix over every counts block of the experiment review records", "wording and record items; no number was wrong")
    for name, rv in recs.items():
        tag = name.replace("review_", "").replace("-", "_")
        c = rv.get("counts", {})
        for k, v in c.items():
            add(f"rev_{tag}_{k}", v, count(v) if isinstance(v, (int, float)) else str(v), src(name, f"counts.{k}"))
        add(f"rev_{tag}_rounds", len(blocks[name]), str(len(blocks[name])), src(name, "counts blocks (own + round_*)"))
        add(f"rev_{tag}_who", rv.get("who"), rv.get("who"), src(name, "who"), "an AI agent, not a team member")
        add(f"rev_{tag}_when", rv.get("when"), rv.get("when"), src(name, "when"))
        add(f"rev_{tag}_verdict", rv.get("verdict"), rv.get("verdict"), src(name, "verdict"))
        if rv.get("human_review"):
            add(f"rev_{tag}_human_review", rv["human_review"], rv["human_review"], src(name, "human_review"), "the team's own check is not done")
    RD = recs["review_documents_2026-10-08"]
    add("rev_documents_round2_verdict", get(RD, "round_2.verdict"), get(RD, "round_2.verdict"), src("review_documents_2026-10-08", "round_2.verdict"))
    add("rev_documents_round3_verdict", get(RD, "round_3.verdict"), get(RD, "round_3.verdict"), src("review_documents_2026-10-08", "round_3.verdict"))
    add("rev_documents_what", RD["what"], RD["what"], src("review_documents_2026-10-08", "what"))
    add("rev_documents_may_say", get(RD, "answers.what_may_the_documents_say_about_this_review"), get(RD, "answers.what_may_the_documents_say_about_this_review"), src("review_documents_2026-10-08", "answers.what_may_the_documents_say_about_this_review"))
    add("money_spent_none", "Nothing was bought" in str(load("data_access").get("what", "")), "none" if "Nothing was bought" in str(load("data_access").get("what", "")) else "see results/data_access.json",
        src("data_access", "what: the sentence 'Nothing was bought and no account was created' (no results file has a money field; summary.costs lists the paid options, none of them used)"),
        "critic 1, item 18: the source is a sentence of the data-access record, not a number")
    add("data_access_checked_on", load("data_access").get("checked_on"), str(load("data_access").get("checked_on")), src("data_access", "checked_on"))
    # repository counts measured on this machine at run time
    section("9.2 Repository and hand-in package: counts measured on this machine at run time; placeholders")
    add("count_code_py_files", len(list(CODE.glob("*.py"))), str(len(list(CODE.glob("*.py")))), "file system: code/*.py (top level) at run time", "README.md of 8 Oct says 57; the three modules of 28 Oct (warning_policy.py, subdaily_inputs.py, fetch_usgs_iv.py) are among them")
    add("count_figures_png", len(list((ROOT / "figures").glob("*.png"))), str(len(list((ROOT / "figures").glob("*.png")))), "file system: figures/*.png at run time")
    add("count_preds_npz", len(list((RES / "preds").glob("*.npz"))), count(len(list((RES / "preds").glob("*.npz")))), "file system: results/preds/*.npz at run time")
    add("count_models_pt", len(list((RES / "models").glob("*.pt"))), count(len(list((RES / "models").glob("*.pt")))), "file system: results/models/*.pt at run time")
    add("count_review_suites", len([p for p in (CODE / "review").iterdir() if p.is_dir()]), str(len([p for p in (CODE / "review").iterdir() if p.is_dir()])), "file system: code/review/* at run time", "the reviewers' own check scripts, one folder per record")
    add("count_report_section_modules", len(list((CODE / "report_sections").glob("s*.py"))), str(len(list((CODE / "report_sections").glob("s*.py")))), "file system: code/report_sections/s*.py")
    man = ROOT / "handoff" / "MANIFEST.sha256"
    if man.exists():
        n_lines = sum(1 for _ in man.open(encoding="utf-8"))
        add("count_manifest_lines", n_lines, count(n_lines), "file system: handoff/MANIFEST.sha256 (lines)", "sha256 of every file of the hand-over zip")
    for folder in ("results/preds", "results/models", "figures", "data"):
        p = ROOT / folder
        if p.exists():
            b = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            add(f"size_{folder.replace('/', '_')}", b, size_text(b), f"file system: {folder} (sum of file sizes at run time)", f"{b:,} B")
    add("repo_url", "[REPO URL]", "[REPO URL]", "placeholder: the GitHub repository will be created later by the user (task brief of 2026-10-28)")
    add("data_bundle_url", "[RELEASE/DRIVE URL]", "[RELEASE/DRIVE URL]", "placeholder: the data bundle (results/preds, models, derived inputs) host is the user's call (GitHub Release or Google Drive)")
    add("web_deck_url", "a web page that is not published", "a web page that is not published", "HANDOFF.md section 3", "the proposal (mid-term) presentation, not a model demo; never pushed unless the user asks")


# ====================================================================================================================================
SEED_EXTENSION_FILES = ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input")     # the results files of scripts 15-18


def build_seed_extension(cfg):
    section("Five seeds: the extension of 30 October 2026 (seeds 3 and 4 of scripts 15-18; disclosure)")
    total, dates, scripts, per = 0, set(), [], []
    for name in SEED_EXTENSION_FILES:
        D = load(name)
        P = D.get("passes") or []
        ext = [p for p in P if (p.get("seeds_trained") and min(p["seeds_trained"]) >= 3) or p.get("seed_extension")]
        n_ext = sum(p.get("n_trainings", 0) for p in ext)
        # trainings of a stopped run that wrote no pass record (results/more_basins.json: 'passes accounts for 24 of the 25 trainings'):
        # the gap between n_trainings and the passes, less the gap the three-seed copy in results/archive/ already had
        gap = (D.get("n_trainings") or 0) - sum(p.get("n_trainings", 0) for p in P)
        arc = RES / "archive" / f"{name}_3seeds_2026-10-30.json"
        if arc.exists():
            A = json.loads(arc.read_text(encoding="utf-8"))
            gap -= (A.get("n_trainings") or 0) - sum(p.get("n_trainings", 0) for p in A.get("passes") or [])
        gap = max(0, gap) if ext else 0
        total += n_ext + gap
        for p in ext:
            m = re.match(r"\d{4}-\d{2}-\d{2}", str(p.get("finished_local_time", "")))
            if m:
                dates.add(m.group(0))
        sc = sorted(CODE.glob(f"[0-9][0-9]_{name}.py"))
        if ext and sc:
            scripts.append(sc[0].name[:2])
        per.append(f"{sc[0].name[:2] if sc else name}: {n_ext + gap}" + (f" ({gap} of them in a stopped run without a pass record)" if gap else ""))
    d0 = sorted(dates)
    files = "results/{" + ",".join(SEED_EXTENSION_FILES) + "}.json"
    add("seed_extension_date", d0[-1] if d0 else None, long_date(d0[-1]) if d0 else "n/a", files + " -> passes[*].finished_local_time of the seed-extension passes",
        "seeds 3 and 4 were added after the test scores of seeds 0-2 were known (each results file's SEED_NOTE)")
    scripts = sorted(set(scripts))
    add("seed_extension_scripts", scripts, (", ".join(scripts[:-1]) + " and " + scripts[-1]) if len(scripts) > 1 else "".join(scripts),
        files + " -> passes (a pass with seeds_trained [3, 4]); file system: code/NN_<name>.py")
    add("seed_extension_trainings", total, count(total),
        files + " -> passes[*].n_trainings of the seed-extension passes, plus the trainings of a stopped run without a pass record (n_trainings minus the passes, less the same gap of the three-seed copy in results/archive/)",
        "by script: " + ", ".join(per) + "; equal to the seed-3/4 networks results/models/*_s3.pt and *_s4.pt of these scripts; scripts 20-22 trained nothing on 30 October (rescored only)")
    add("seed_extension_machine", cfg["windows_pc"], cfg["windows_pc"], "code/final_report/config.json -> windows_pc", "seeds 0-2 were trained on the Mac (mac_machine) and kept, not retrained")
    AR = load("aorc_rain_to_midnight")
    ch = AR["choices_made_after_looking_at_test_scores"]
    i16 = max((i for i, c in enumerate(ch) if re.match(r"\s*Seeds 3 and 4", str(c))), default=None)
    if i16 is not None:
        sn = re.sub(r"\s+", " ", str(ch[i16])).strip()
        add("seed_note_16", sn, sn, src("aorc_rain_to_midnight", f"choices_made_after_looking_at_test_scores.{i16}"),
            "the SEED_NOTE of script 16, word for word: quote it whole where the five-seed AORC share is disclosed")
    arc = RES / "archive" / "aorc_rain_to_midnight_3seeds_2026-10-30.json"
    if arc.exists():
        o = get(json.loads(arc.read_text(encoding="utf-8")), "summary_test.sets.wqa.l1.onset")
        asrc = "results/archive/aorc_rain_to_midnight_3seeds_2026-10-30.json -> summary_test.sets.wqa.l1.onset"
        psrc, hsrc = asrc + ".pod", asrc + ".hits"
    else:
        h = get(load("review_aorc_rain_to_midnight"), "round_2.hindsight.three_seeds.aorc_all")
        o = {"pod": h / 611, "hits": h}
        psrc = hsrc = src("review_aorc_rain_to_midnight", "round_2.hindsight.three_seeds.aorc_all") + " (of 611 onsets)"
    add("aorc_share_three_seeds", o["pod"], pct(o["pod"]), psrc,
        f"lstm_wqa with seeds 0-2 ({o['hits']} of 611 onsets warned 1 day ahead), as quoted before 30 October 2026; now {pct(get(AR, 'summary_test.sets.wqa.l1.onset.pod'))} with five seeds")
    add("aorc_hits_three_seeds", o["hits"], count(o["hits"]), hsrc)
    LA, RT = load("lstm_all"), load("rain_timing")
    three = [f"lstm_{k}" for k in ("wq", "w", "wf", "wqf") if len(get(LA, f"sets.{k}.seeds")) == 3]
    if len(get(RT, "models.wqn.per_seed")) == 3:
        three.append("lstm_wqn")
    add("stored_sets_three_seeds_list", three, ", ".join(three) + " and the simple models" if three else "the simple models",
        src("lstm_all", "sets.*.seeds (length 3)") + "; " + src("rain_timing", "models.wqn.per_seed (length 3)") + "; results/baselines.json (the simple models, one fit each)",
        "the stored sets of scripts 02-10 keep three seeds: those scripts saved no networks, so extending them would mean retraining seeds 0-2 and replacing the reviewed forecasts "
        "(also three: lstm_wn, lstm_wn_older, lstm_wqn_older of script 10; two: the satellite variants of 08 and the comparison of 09)")


# ====================================================================================================================================
def build_targets_and_published():
    section("7.4 Mid-term targets (text constants from the live deck and the mid-term PPT) and their status")
    web = read_text("code/build_proposal_web.py")
    m = re.search(r"HEADLINE = dict\((.*?)\)\n", web, re.S)
    hl = dict(re.findall(r"(\w+)=([\d.]+)", m.group(1)))
    ln = next(i + 1 for i, l in enumerate(web.splitlines()) if l.startswith("HEADLINE = dict("))
    for k, v in hl.items():
        val = float(v) if "." in v else int(v)
        txt = f"{val}%" if k.startswith(("warned", "far")) else str(val)
        add(f"midterm_headline_{k}", val, txt, f"code/build_proposal_web.py:{ln} HEADLINE (the live mid-term deck, screens 2, 5 and 6)", "a number typed into the deck builder on 15 Oct 2026; the results files hold the same value")
    add("target_go_warned", 0.70, "at least 70%", src("key_figure", "criteria.go_min_warned") + "; live deck screen 6 tile 1 ('19% -> 48% -> >= 70%')", "TARGET, NOT MET: 48% in the real-time check (REALTIME_WORDING), 52% with AORC (five seeds; lstm_wqa_l1_onset_pod); even the perfect weather forecast gives 64%")
    add("target_go_false", 0.30, "at most 30%", src("key_figure", "criteria.go_max_false") + "; live deck screen 6 tile 2 ('59% -> 35% -> <= 30%')", "TARGET, NOT MET together with the first: see midterm_ppt_target_50_status for the 1-day false shares of the forecast model and of wqa (five seeds); 59% with the rain table")
    add("target_seeds", "3 -> 5", "3 -> 5", "live deck screen 6 tile 3 (agent4.md row 3); code/schedule.py WEEKS[1]", "the state on the build day: target_seeds_status")
    five = all(len(load(n).get("seeds") or []) == 5 for n in ("rain_to_midnight", "aorc_rain_to_midnight", "forecast_input", "within_day_inputs", "flow_state")) and \
        any(min(p.get("seeds_trained") or [0]) >= 3 for p in load("more_basins").get("passes") or [])
    st = ("done for every set of scripts 15-18 and for the new sets of 20-22 (five trainings); not for the baseline sets of scripts 02-10, which saved no networks "
          "(extending them would replace the reviewed forecasts)") if five else "PARTLY: see the seeds of each results file"
    add("target_seeds_status", st, st, "results/{rain_to_midnight,aorc_rain_to_midnight,forecast_input,within_day_inputs,flow_state}.json -> seeds (five); results/more_basins.json -> passes (seeds_trained [3, 4])",
        "the mid-term target row 'five trainings per model where the NSE decides'; seeds 3 and 4 were added on 30 October 2026 after the test scores of seeds 0-2 were known (seed_note_16)")
    add("target_scripts_reviewed", "100% of scripts", "100% of scripts", "live deck screen 6 tile 3; code/build_proposal_slides.py slide 5 table", "PARTLY: engine + scripts 15-18 have records; scripts 01-14 only the sentence in engine_review.json notes[1]; the modules of 28 Oct are unreviewed. Never write '100% of scripts reviewed'.")
    add("target_gis_layers", "12 of 12 basins as GIS layers", "12 of 12 basins as GIS layers", "live deck screen 6 tile 3; proposal deliverable D3", ("DONE 2026-10-29: statistics and maps for 12 of 12 and the layers exported (results/gis_layers/, code/27_gis_layers.py)" if (ROOT / "results" / "gis_layers.json").exists() else "PARTLY: statistics and maps for 12 of 12; the export as GIS layers NOT done (HANDOFF.md section 4 step 3)"))
    add("target_own_checks", "4 of 4", "4 of 4", "live deck screen 6 tile 3; code/schedule.py MILESTONES[1]", "0 of 4 on 2026-10-28 (team.json); only the members may change it")
    m = re.search(r"^STRETCH = \((.*?)\)\s*$", read_text("code/schedule.py"), re.M | re.S)
    stretch = "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", m.group(1))) if m else None
    add("target_stretch_stage4_network", stretch, "a network trained on the real-time radar product (Stage IV)", "code/schedule.py STRETCH", "NOT ATTEMPTED: the 128-200 GB download never started; the real-time check (REALTIME_WORDING) stands in its place. Full text in value.")
    m = re.search(r"about (\d+ GB)", stretch or "")
    add("stretch_download_size", m.group(1) if m else None, f"about {m.group(1)}" if m else "n/a", "code/schedule.py STRETCH ('needs its hourly archive for 2002-2014 (about N GB through the proxy)')",
        "the full-period Stage IV download the stretch goal needed for training; it never started (critic 1, item 19)")
    ps = read_text("code/build_proposal_slides.py")
    rows = re.findall(r"\[\"([^\"]+)\", \"([^\"]+)\", \"([^\"]+)\"\]", ps.split("head=[\"Indicator\", \"Baseline\", \"Target\"]")[1].split("]),")[0])
    ln = next(i + 1 for i, l in enumerate(ps.splitlines()) if "head=[\"Indicator\", \"Baseline\", \"Target\"]" in l)
    for i, (ind, base, tgt) in enumerate(rows, 1):
        add(f"midterm_ppt_target_{i}", {"indicator": ind, "baseline": base, "target": tgt}, f"{ind}: {base} -> {tgt}", f"code/build_proposal_slides.py:{ln + i} (slide 5 of the mid-term PPT, hand-in item 2)",
            "the mid-term PPT set interim targets ('above 45% with a network trained on real-time radar', 'below 50% false') beside the proposal's GO level; the report closes both sets (critic C11)")
    add("midterm_ppt_target_45_status", "number met, condition not", "exceeded only in the real-time check (48%, networks trained on AORC with Stage IV substituted), not by a radar-trained network", "agent4.md row 7b; " + src("stage4_realtime", "outcome"))
    far_f, far_a = get(load("forecast_input"), "summary_test.sets.wqaF_sub.l1.onset.far"), get(load("aorc_rain_to_midnight"), "summary_test.sets.wqa.l1.onset.far")
    add("midterm_ppt_target_50_status", "met by the forecast model only" if far_f < 0.5 <= far_a else "see the text",
        f"{'met' if far_f < 0.5 else 'not met'} at 1 day by the forecast model ({pct(far_f)} false), {'not ' if far_a >= 0.5 else ''}by the rain to midnight alone ({pct(far_a)})",
        "agent4.md row 7b; " + src("forecast_input", "summary_test.sets.wqaF_sub.l1.onset.far") + "; " + src("aorc_rain_to_midnight", "summary_test.sets.wqa.l1.onset.far"),
        "both shares built from the two sources (critic 3, M2: the typed '50%' was the three-seed share; the five-seed lstm_wqa share is 327 of 643)")
    add("milestone_m2_status", "not met with the river gauge", "not met with the river gauge", "PLAN.md Milestones (M2: the LSTM beats the best simple model); " + src("key_figure", "levels.2.others.base_linreg_wq.against_lstm.in_words"))
    add("week2_plan_status", "not started", "not started", "HANDOFF.md section 3 ('The week-2 plan (22-28 Oct) was NOT started'; 'Nothing in the project changed between 2026-10-19 and 2026-10-28')", "five seeds, GIS export, review records for every script, stretch goal: none run between 15 and 28 October")
    # published scores
    section("2.2/8.1 Published scores read at source (results/published_scores.json): 'of the same order as', never 'matches')")
    PS = load("published_scores")
    add("published_checked_by", PS["checked_by"], PS["checked_by"], src("published_scores", "checked_by"), "no team member has read these papers yet")
    add("published_how_to_word_it", PS["how_to_word_it"], PS["how_to_word_it"], src("published_scores", "how_to_word_it"))
    for key, vals in (("kratzert2019", ("lstm_with_basin_properties_nse_loss_single", "lstm_with_basin_properties_nse_loss_ensemble", "lstm_without_basin_properties_nse_loss_single")),
                      ("kratzert2019pub", ("lstm_basin_never_seen", "sac_sma_calibrated_on_each_basin", "national_water_model")),
                      ("feng2020", ("projection_no_observed_flow_671", "with_flow_observed_1_day_before_671", "with_flow_observed_3_days_before_671", "sac_sma_671"))):
        S = get(PS, f"sources.{key}")
        for v in vals:
            add(f"pub_{key}_{v}", S["median_nse"][v], n2(S["median_nse"][v]), src("published_scores", f"sources.{key}.median_nse.{v}"), f"{S['label']}: median NSE; {S['n_basins']} basins; read at {S['where_read'][:80]}")
        add(f"pub_{key}_n_basins", S["n_basins"], count(S["n_basins"]), src("published_scores", f"sources.{key}.n_basins"), S["label"])
        if "n_cells" in S:
            add(f"pub_{key}_n_cells", S["n_cells"], str(S["n_cells"]), src("published_scores", f"sources.{key}.n_cells"))
        if "n_seeds" in S:
            add(f"pub_{key}_n_seeds", S["n_seeds"], str(S["n_seeds"]), src("published_scores", f"sources.{key}.n_seeds"))
    S = get(PS, "sources.kratzert2018")
    add("pub_kratzert2018_mean_nse_lstm", S["mean_nse"]["lstm_one_per_basin"], n2(S["mean_nse"]["lstm_one_per_basin"]), src("published_scores", "sources.kratzert2018.mean_nse.lstm_one_per_basin"), S["label"] + ": mean NSE, one LSTM per basin")
    add("pub_kratzert2018_mean_nse_sac_sma", S["mean_nse"]["sac_sma_snow17"], n2(S["mean_nse"]["sac_sma_snow17"]), src("published_scores", "sources.kratzert2018.mean_nse.sac_sma_snow17"))
    add("pub_kratzert2018_n_basins", S["n_basins"], str(S["n_basins"]), src("published_scores", "sources.kratzert2018.n_basins"))
    PC = load("published_context")
    add("published_context_n_sources", len(PC.get("sources", {})), str(len(PC.get("sources", {}))), src("published_context", "sources (length)"), "sources read by an AI scout on 2026-10-07; none read by a team member")
    if "how_to_word_it" in PC:
        add("published_context_how_to_word_it", PC["how_to_word_it"], str(PC["how_to_word_it"]), src("published_context", "how_to_word_it"))
    SC = load("sources_checked")
    add("sources_checked_on", SC["checked_on"], SC["checked_on"], src("sources_checked", "checked_on"))
    add("sources_checked_how_to_word_it", SC["how_to_word_it"], SC["how_to_word_it"], src("sources_checked", "how_to_word_it"))
    # satellite inputs (section 6.2 of the report)
    section("6.2 Satellite and reanalysis inputs (08_satellite_inputs.py, two seeds, the satellite-era split)")
    SI = load("satellite_inputs")
    add("sat_n_trainings", SI["n_trainings"], str(SI["n_trainings"]), src("satellite_inputs", "n_trainings"))
    add("sat_training_seconds", SI["training_seconds"], minutes_text(SI["training_seconds"]), src("satellite_inputs", "training_seconds"), f"{SI['training_seconds']} s")
    add("sat_seeds", SI["seeds"], str(len(SI["seeds"])), src("satellite_inputs", "seeds"), "two seeds per variant; no p-value")
    add("sat_seed_noise_median", get(SI, "seed_noise_single_basin.median"), n2(get(SI, "seed_noise_single_basin.median"), 3), src("satellite_inputs", "seed_noise_single_basin.median"), "absolute difference of one basin's test NSE between two trainings of the same model")
    add("sat_seed_noise_rule", get(SI, "seed_noise_rule.rule"), get(SI, "seed_noise_rule.rule"), src("satellite_inputs", "seed_noise_rule.rule"))
    for grp in ("read_first_gains_1_day_ahead", "rain_from_satellite_instead_of_gauges", "satellite_rain_on_top_of_gauge_rain", "soil_wetness", "precipitable_water"):
        lst = get(SI, f"findings.{grp}")
        if lst:
            add(f"sat_finding_{grp}", lst[0], lst[0], src("satellite_inputs", f"findings.{grp}.0"))


# ====================================================================================================================================
def _flatten_scalars(prefix, obj, fname, path=""):
    """Top-level and one-level-down scalars of an optional results file, as e4x_<key> entries."""
    for k, v in obj.items():
        kk = re.sub(r"[^a-z0-9_]+", "_", str(k).lower()).strip("_")
        p = f"{path}.{k}" if path else k
        if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool) or isinstance(v, bool):
            if isinstance(v, bool):
                text = str(v)
            elif isinstance(v, float):
                text = pct(v) if 0 <= v <= 1 and ("pod" in kk or "far" in kk or "share" in kk) else (n2(v) if abs(v) < 10 else n2(v, 1))
            else:
                text = str(v)
            key = f"{prefix}_{kk}"
            if key not in REG and len(str(v)) <= 4000:
                add(key, v, text, src(fname, p), "OPTIONAL: present only because the file exists; needs its own review record before any document quotes it")


def build_optional_e4():
    section("E4 (optional, present only if the results files exist): warning policy (E4a), within-day inputs (E4b), flow state (E4e)")
    found = []
    for rel, prefix in (("results/warning_policy.json", "e4a"), ("results/within_day_inputs.json", "e4b"), ("results/flow_state.json", "e4e"),
                        ("results/review_warning_policy.json", "e4a_review"), ("results/review_within_day_inputs.json", "e4b_review"), ("results/review_flow_state.json", "e4e_review")):
        p = ROOT / rel
        add(f"{prefix}_file_present", p.exists(), "present" if p.exists() else "absent", rel, "an optional results file (E4 work order of 2026-10-28); absent = the experiment is future work")
        if not p.exists():
            continue
        found.append(rel)
        name = p.stem
        D = json.loads(p.read_text(encoding="utf-8"))
        if "diagnostic" in str(D.get("status", "")).lower() or "preliminary" in str(D.get("status", "")).lower():
            add(f"{prefix}_status", D.get("status"), "PRELIMINARY: not quotable", src(name, "status"), "a preliminary diagnostic: do not quote in any document")
            continue
        for k in ("outcome", "outcome_caveat", "outcome_full", "status", "question"):
            if k in D:
                add(f"{prefix}_{k}", D[k], str(D[k]), src(name, k), "needs results/review_<name>.json before it is quoted; label PRELIMINARY otherwise")
        for k in ("n_trainings", "training_seconds", "seeds"):
            if k in D:
                add(f"{prefix}_{k}", D[k], str(D[k]), src(name, k))
        if prefix.endswith("_review"):
            for k, v in (D.get("counts") or {}).items():
                add(f"{prefix}_{k}", v, str(v), src(name, f"counts.{k}"))
            for k in ("verdict", "who", "when"):
                if k in D:
                    add(f"{prefix}_{k}", D[k], str(D[k]), src(name, k))
            continue
        # the warning-policy table of warning_policy.py: sets.<set>.leads.<k>.rules.<rule>.{trigger,pod,far}
        sets = D.get("sets")
        if isinstance(sets, dict):
            for s, sv in sets.items():
                leads = sv.get("leads") if isinstance(sv, dict) else None
                if not isinstance(leads, dict):
                    continue
                for lk, lv in leads.items():
                    rules = lv.get("rules") if isinstance(lv, dict) else None
                    if not isinstance(rules, dict):
                        continue
                    for rk, rv in rules.items():
                        if not isinstance(rv, dict):
                            continue
                        for f_ in ("trigger", "pod", "far", "csi", "hits", "false_alarms", "misses"):
                            if f_ in rv:
                                v = rv[f_]
                                text = pct(v) if f_ in ("pod", "far") else (n2(v) if f_ == "csi" else (f"x{v:.2f}" if f_ == "trigger" else count(v)))
                                add(f"{prefix}_{s}_l{lk}_{rk}_{f_}", v, text, src(name, f"sets.{s}.leads.{lk}.rules.{rk}.{f_}"), "E4a operating point; quote only with its review record")
                        # added 2026-10-29 for the final report (sections 7.3 and 9.3): the scores of each rule on the validation and the test years,
                        # all high-flow days and onsets (sets.<set>.leads.<k>.rules.<rule>.{validation,test}.{all,onset}.*), and the verdicts
                        for split in ("validation", "test"):
                            SP = rv.get(split)
                            if not isinstance(SP, dict):
                                continue
                            for kind in ("all", "onset"):
                                T = SP.get(kind)
                                if not isinstance(T, dict):
                                    continue
                                for f_ in ("pod", "far", "csi", "hits", "misses", "false_alarms"):
                                    if f_ in T:
                                        v = T[f_]
                                        add(f"{prefix}_{s}_l{lk}_{rk}_{split}_{kind}_{f_}", v, pct(v) if f_ in ("pod", "far") else (n2(v) if f_ == "csi" else count(v)),
                                            src(name, f"sets.{s}.leads.{lk}.rules.{rk}.{split}.{kind}.{f_}"), "E4a operating point (test years scored once per rule); quote only with its review record")
                        for vk in ("verdict_onsets", "verdict_all_days"):
                            if vk in rv:
                                add(f"{prefix}_{s}_l{lk}_{rk}_{vk}", rv[vk], str(rv[vk]), src(name, f"sets.{s}.leads.{lk}.rules.{rk}.{vk}"), "E4a verdict of the fixed rule at this operating point")
        # summary_test of a training experiment (E4b, E4e): the same shape as scripts 15-18
        summ = opt(D, "summary_test.sets")
        if isinstance(summ, dict):
            for s, sv in summ.items():
                for L in LEADS:
                    S = sv.get(L) if isinstance(sv, dict) else None
                    if not isinstance(S, dict):
                        continue
                    if "nse_median" in S:
                        add(f"{prefix}_{s}_{L}_nse", S["nse_median"], n2(S["nse_median"]), src(name, f"summary_test.sets.{s}.{L}.nse_median"), "E4 result: quote only with its review record")
                    for kind in ("onset", "all"):
                        T = S.get(kind)
                        if isinstance(T, dict):
                            for f_ in ("pod", "far", "csi", "hits", "misses", "false_alarms"):
                                if f_ in T:
                                    v = T[f_]
                                    add(f"{prefix}_{s}_{L}_{kind}_{f_}", v, pct(v) if f_ in ("pod", "far") else (n2(v) if f_ == "csi" else count(v)), src(name, f"summary_test.sets.{s}.{L}.{kind}.{f_}"))
        # added 2026-10-29 for the final report (T7c, 7.3, 8.2, 9.1) and the deck: the operating points of the training experiments E4b and E4e
        # (operating_points.sets.<set>.l<k>.rules.<rule>.{trigger, validation, test, test_verdict}: the rules of E4a, chosen on the validation years
        # of the experiment's own split and scored once on the test years). The short names `_pod` / `_far` are the test-year onset scores, as
        # the writers and the deck name them beside `e4a_<set>_l<k>_far_cap_test_onset_pod`.
        ops = opt(D, "operating_points.sets")
        if isinstance(ops, dict):
            for s, sv in ops.items():
                if not isinstance(sv, dict):
                    continue
                for L, lv in sv.items():
                    rules = lv.get("rules") if isinstance(lv, dict) else None
                    if not isinstance(rules, dict):
                        continue
                    for rk, rv in rules.items():
                        if not isinstance(rv, dict):
                            continue
                        tag = prefix.upper()
                        if "trigger" in rv:
                            add(f"{prefix}_{s}_{L}_{rk}_trigger", rv["trigger"], f"x{rv['trigger']:.2f}", src(name, f"operating_points.sets.{s}.{L}.rules.{rk}.trigger"),
                                f"{tag} operating point, chosen on the validation years of this experiment's split; quote only with its review record")
                        if "reached" in rv:
                            add(f"{prefix}_{s}_{L}_{rk}_reached", rv["reached"], "yes" if rv["reached"] else "no", src(name, f"operating_points.sets.{s}.{L}.rules.{rk}.reached"),
                                "whether a trigger of the grid met the false-share cap on the validation years (no = the lowest-false trigger is given)")
                        for split in ("validation", "test"):
                            SP = rv.get(split)
                            if not isinstance(SP, dict):
                                continue
                            for kind in ("all", "onset"):
                                T = SP.get(kind)
                                if not isinstance(T, dict):
                                    continue
                                for f_ in ("pod", "far", "csi", "hits", "misses", "false_alarms"):
                                    if f_ in T:
                                        v = T[f_]
                                        text = pct(v) if f_ in ("pod", "far") else (n2(v) if f_ == "csi" else count(v))
                                        add(f"{prefix}_{s}_{L}_{rk}_{split}_{kind}_{f_}", v, text, src(name, f"operating_points.sets.{s}.{L}.rules.{rk}.{split}.{kind}.{f_}"),
                                            f"{tag} operating point (test years scored once per rule); quote only with its review record")
                                        if split == "test" and kind == "onset" and f_ in ("pod", "far"):
                                            add(f"{prefix}_{s}_{L}_{rk}_{f_}", v, text, src(name, f"operating_points.sets.{s}.{L}.rules.{rk}.test.onset.{f_}"),
                                                f"{tag} operating point: the test-year onset score at the {rk} level (short name); the test false share is the one that came out, not the cap; quote only with its review record")
                        TV = rv.get("test_verdict")
                        if isinstance(TV, dict):
                            for kind in ("onset", "all"):
                                if kind in TV:
                                    add(f"{prefix}_{s}_{L}_{rk}_test_verdict_{kind}", TV[kind], str(TV[kind]), src(name, f"operating_points.sets.{s}.{L}.rules.{rk}.test_verdict.{kind}"),
                                        "verdict of the fixed rule at this operating point, test years")
        # the paired onsets of E4e (readings.pairs.<first>_vs_<second>.paired_onsets.<lead>): warned only by the first-named set (`only_with`),
        # only by the second (`only_without`), by both, by neither
        pairs = opt(D, "readings.pairs")
        if isinstance(pairs, dict):
            for pr, pv in pairs.items():
                PO = pv.get("paired_onsets") if isinstance(pv, dict) else None
                if not isinstance(PO, dict):
                    continue
                for L, T in PO.items():
                    if not isinstance(T, dict):
                        continue
                    for f_, short in (("warned_only_by_first", "only_with"), ("warned_only_by_second", "only_without"), ("warned_by_both", "both"), ("warned_by_neither", "neither"), ("n_onsets", "n_onsets")):
                        if f_ in T:
                            add(f"{prefix}_paired_{pr}_{L}_{short}", T[f_], count(T[f_]), src(name, f"readings.pairs.{pr}.paired_onsets.{L}.{f_}"),
                                f"paired onsets of the test years, {T.get('first')} against {T.get('second')} at trigger 1.0; a description, the onsets of one storm are not independent")
        # the hourly record's coverage (E4e, data_coverage): basin-days read and the days whose columns fell back to the daily mean where the
        # record has gaps, for the midnight (24) and the 22:00 (22) variants, in all and in the test years, and per gauge
        DC = D.get("data_coverage")
        if isinstance(DC, dict) and isinstance(DC.get("per_basin"), dict):
            PB = DC["per_basin"]
            n_all = sum(b.get("days", 0) for b in PB.values())
            n_test = sum(b.get("test_days", 0) for b in PB.values())
            add(f"{prefix}_hourly_basin_days", n_all, count(n_all), src(name, "data_coverage.per_basin.<gauge>.days (sum over the basins)"), "basin-days of the hourly gauge record read for this split, all basins")
            add(f"{prefix}_hourly_basin_days_test_years", n_test, count(n_test), src(name, "data_coverage.per_basin.<gauge>.test_days (sum over the basins)"), "basin-days of the test years")
            for var in ("24", "22"):
                tot = opt(DC, f"fallback_days_total.{var}")
                if tot is not None:
                    add(f"{prefix}_fallback_basin_days_{var}", tot, count(tot), src(name, f"data_coverage.fallback_days_total.{var}"),
                        f"basin-days whose hourly-state columns fell back to the daily mean where the hourly record has gaps, record ending at {var}:00")
                tst = sum(b.get(f"fallback_days_{var}_test_years", 0) for b in PB.values())
                add(f"{prefix}_fallback_basin_days_{var}_test_years", tst, count(tst), src(name, f"data_coverage.per_basin.<gauge>.fallback_days_{var}_test_years (sum over the basins)"),
                    f"of those, basin-days in the test years, record ending at {var}:00")
                for g, b in PB.items():
                    if f"fallback_days_{var}_test_years" in b:
                        add(f"{prefix}_fallback_days_{var}_test_years_{g}", b[f"fallback_days_{var}_test_years"], count(b[f"fallback_days_{var}_test_years"]),
                            src(name, f"data_coverage.per_basin.{g}.fallback_days_{var}_test_years"), f"test-year basin-days that fell back to the daily mean at gauge {g}, record ending at {var}:00")
        _flatten_scalars(prefix, D, name)
    # the dates of the three independent reviews (results/review_*.json -> when, the first date each names), for "reviewed on ..." in the abstract and 1.1
    dates = []
    for rel in ("results/review_warning_policy.json", "results/review_within_day_inputs.json", "results/review_flow_state.json"):
        p = ROOT / rel
        if p.exists():
            m = re.search(r"\d{4}-\d{2}-\d{2}", str(json.loads(p.read_text(encoding="utf-8")).get("when", "")))
            if m:
                dates.append(m.group(0))
    if dates:
        ds = sorted(set(dates))
        texts = [long_date(d) for d in ds]
        tails = {t.split(" ", 1)[1] for t in texts}
        text = (" and ".join(t.split(" ", 1)[0] for t in texts) + " " + tails.pop()) if len(texts) > 1 and len(tails) == 1 else " and ".join(texts)
        add("date_e4_reviews", ds, text, "results/review_warning_policy.json, results/review_within_day_inputs.json, results/review_flow_state.json -> when (the first date each names)",
            "the dates of the independent reviews of E4a, E4b and E4e")
    for rel, prefix in (("results/within_day_inputs.json", "e4b"), ("results/flow_state.json", "e4e")):
        p = ROOT / rel
        if p.exists():
            D_ = json.loads(p.read_text(encoding="utf-8"))
            w = D_.get("written")
            if w:
                add(f"date_{prefix}_written", w, long_date(w), f"{rel} -> written", "the day the results file was written: the last networks and forecast files added to results/ (critic 1, item 7)")
            passes = D_.get("passes")
            if isinstance(passes, list) and passes and isinstance(passes[0], dict) and passes[0].get("finished_local_time"):
                t = passes[0]["finished_local_time"]
                add(f"{prefix}_training_finished", t, long_date(t), f"{rel} -> passes.0.finished_local_time",
                    f"the training run finished at {t} local time; the file's 'written' ({w}) is the rescore after the review, wording only (critic 2, item 5)")
    p = ROOT / "results/warning_policy.json"
    if p.exists():
        t = json.loads(p.read_text(encoding="utf-8")).get("finished_local_time")
        if t:
            add("e4a_finished_date", t, long_date(t), "results/warning_policy.json -> finished_local_time",
                f"{t} local time: the record as rescored after its review (no training; the record under review was written earlier that evening, results/review_warning_policy.json -> dates.record_written)")
    add("e4_files_found", found, ", ".join(found) if found else "none", "file system at run time", "which optional E4 results files existed when the registry was written")


# ====================================================================================================================================
def _two_dates_text(d0, d1):
    """'2026-10-28', '2026-10-29' -> '28 and 29 October 2026'; one date -> its long form."""
    if d0 == d1:
        return long_date(d0)
    a, b = long_date(d0), long_date(d1)
    return f"{a.split(' ', 1)[0]} and {b}" if a.split(" ", 1)[1] == b.split(" ", 1)[1] else f"{a} and {b}"


def build_hourly_flow():
    """The hourly reduction of the USGS 15-minute record (data/hourly_flow, fetched by code/fetch_usgs_iv.py through api.waterdata.usgs.gov),
    the input of E4e (code/22_flow_state.py): read from data/hourly_flow/usgs_iv_meta.json, results/flow_state.json data_coverage and the
    file system. Critic 2 of 2026-10-29, item 2: the data inventory (T5, T20, T3) must know this second USGS download; results/data_flow.json
    (written 2026-10-08) records only the January-June 2012 clock check."""
    section("4.1/Appendix B The hourly USGS flow record read for E4e (data/hourly_flow/usgs_iv_meta.json, results/flow_state.json data_coverage, file system)")
    meta_p = ROOT / "data" / "hourly_flow" / "usgs_iv_meta.json"
    add("hf_file_present", meta_p.exists(), "present" if meta_p.exists() else "absent", "data/hourly_flow/usgs_iv_meta.json (file system)",
        "the hourly reduction of the USGS 15-minute record, the input of E4e; absent = not fetched on this machine")
    if not meta_p.exists():
        return
    M = json.loads(meta_p.read_text(encoding="utf-8"))
    H = {g: v["hourly"] for g, v in M["gauges"].items() if isinstance(v, dict) and "hourly" in v}
    msrc = "data/hourly_flow/usgs_iv_meta.json -> "
    api = str(M.get("api", ""))
    add("hf_api", api, re.sub(r"^https?://([^/]+).*$", r"\1", api), msrc + "api", api)
    add("hf_gauges", len(H), str(len(H)), msrc + "gauges (count)", "all twelve study gauges")
    add("hf_values", sum(h["values"] for h in H.values()), count(sum(h["values"] for h in H.values())), msrc + "gauges.<id>.hourly.values (sum over the gauges)", "15-minute values reduced to hourly means and last values")
    add("hf_hours", sum(h["hours"] for h in H.values()), count(sum(h["hours"] for h in H.values())), msrc + "gauges.<id>.hourly.hours (sum over the gauges)")
    add("hf_hours_without_values", sum(h["hours_without_values"] for h in H.values()), count(sum(h["hours_without_values"] for h in H.values())), msrc + "gauges.<id>.hourly.hours_without_values (sum over the gauges)")
    add("hf_months_on_disk", sum(h["months_on_disk"] for h in H.values()), count(sum(h["months_on_disk"] for h in H.values())), msrc + "gauges.<id>.hourly.months_on_disk (sum over the gauges)", "monthly raw files under data/hourly_flow/raw")
    first = min(h["first_hour_end"] for h in H.values())
    last = max(h["last_hour_end"] for h in H.values())
    add("hf_first_hour", first, long_date(first), msrc + "gauges.<id>.hourly.first_hour_end (earliest)", f"{first} UTC, hour-ending stamp")
    add("hf_last_hour", last, long_date(last), msrc + "gauges.<id>.hourly.last_hour_end (latest)", f"{last} UTC, hour-ending stamp")
    add("hf_period", [first[:10], last[:10]], f"{first[:10]} to {last[:10]}", msrc + "gauges.<id>.hourly.first_hour_end (earliest) and last_hour_end (latest)", "hour-ending stamps in UTC; the record of a gauge may start later (per gauge in the file)")
    add("hf_period_long", [first[:10], last[:10]], f"{long_date(first)} to {long_date(last)}", msrc + "gauges.<id>.hourly.first_hour_end (earliest) and last_hour_end (latest)")
    approved = min(h["share_approved"] for h in H.values())
    add("hf_share_approved", approved, "approved values only" if approved >= 1 else pct(approved), msrc + "gauges.<id>.hourly.share_approved (minimum over the gauges)")
    hf_dir = ROOT / "data" / "hourly_flow"
    b_all = sum(f.stat().st_size for f in hf_dir.rglob("*") if f.is_file())
    b_npz = sum(f.stat().st_size for f in hf_dir.glob("*.npz"))
    add("hf_bytes_on_disk", b_all, size_text(b_all), "file system: data/hourly_flow (sum of file sizes at run time, the raw monthly files included)", f"{b_all:,} B")
    add("hf_bytes_npz", b_npz, size_text(b_npz), "file system: data/hourly_flow/usgs_iv_<gauge>.npz (sum of file sizes at run time)", f"{b_npz:,} B; the hourly files code/22_flow_state.py reads")
    logs = sorted((RES / "logs").glob("fetch_usgs_iv_*.log"))
    if logs:
        d0 = re.search(r"(\d{4}-\d{2}-\d{2})", logs[0].name).group(1)
        d1 = time.strftime("%Y-%m-%d", time.localtime(logs[-1].stat().st_mtime))
        add("hf_fetch_dates", [d0, d1], _two_dates_text(d0, d1), f"results/logs/{logs[0].name} (the date in the file name) and the last write time of results/logs/{logs[-1].name} (file system)",
            "the download by code/fetch_usgs_iv.py through api.waterdata.usgs.gov on the Windows PC, resumable month by month; the npz files carry the same last day")
    if (RES / "flow_state.json").exists():
        PB = opt(load("flow_state"), "data_coverage.per_basin")
        if isinstance(PB, dict) and PB:
            add("hf_basins_in_flow_state", len(PB), str(len(PB)), src("flow_state", "data_coverage.per_basin (count)"), "basins whose hourly record code/22_flow_state.py read")
            a2 = min(b.get("approved_share_after_2000", 0) for b in PB.values())
            add("hf_approved_share_after_2000", a2, "approved values only" if a2 >= 1 else pct(a2), src("flow_state", "data_coverage.per_basin.<gauge>.approved_share_after_2000 (minimum over the basins)"))
            zones = {}
            for b in PB.values():
                zones[b.get("zone")] = zones.get(b.get("zone"), 0) + 1
            add("hf_zones", zones, "; ".join(f"{k}: {v}" for k, v in sorted(zones.items(), key=lambda kv: -kv[1])), src("flow_state", "data_coverage.per_basin.<gauge>.zone (count per zone)"))


# ====================================================================================================================================
def build_handin_additions():
    """Keys added on 2026-10-29 for the final report builder (code/build_report_final.py): requirements.txt, the Windows hand-off date,
    the Flood_Cloud_Model package (the teammates' separate experiment; its numbers come from its own reference/results.json and are
    quoted in section 8.1 with the caveats of the package review, never without the persistence row)."""
    section("Hand-in additions of 2026-10-29 (4.3, 6.1, 8.1, Appendix A): requirements.txt, the Windows hand-off, the Flood_Cloud_Model package")
    req = read_text("requirements.txt").splitlines()
    pins = [l.split("#")[0].rstrip() for l in req if l.strip() and not l.lstrip().startswith("#")]
    add("requirements_txt_verbatim", pins, "\n".join(pins), "requirements.txt (every non-comment line in file order, inline comments cut)",
        "Appendix A prints it as a code block, one line per pin; the header comments and the inline comments of the file are left out "
        "(compliance audit P17 of 30 Oct 2026: an inline comment named the older build_report_paper.py)")
    m = re.search(r"^numpy==(\S+)", "\n".join(req), re.M)
    add("numpy_version", m.group(1) if m else None, m.group(1) if m else "n/a", "requirements.txt -> the line numpy==", "pinned as a pair with torch (requirements.txt header)")
    m = re.search(r"state on (\d{4}-\d{2}-\d{2})", read_text("HANDOFF.md"))
    add("date_windows_handoff", m.group(1), long_date(m.group(1)),
        "HANDOFF.md -> title line 'state on YYYY-MM-DD' (the day the project moved from the Mac to the Windows PC); also the comment at code/flood.py:436")
    tsrc = ("Claude Code session transcripts on the Windows PC (~/.claude/projects/<user>/*.jsonl that mention Project-Flood, "
            "fields 'version' and 'message.model'), read by the report fixer on 30 Oct 2026; the laptop sessions of 2-27 October did not "
            "travel with the hand-over, so their versions are not recorded")
    add("claude_code_versions_pc", ["2.1.310", "2.1.314"], "2.1.310 to 2.1.314", tsrc, "sessions of 28-30 October 2026 on the Windows PC")
    add("claude_models_pc", ["claude-opus-5-5", "claude-fable-5-1"], "claude-opus-5-5 and claude-fable-5-1", tsrc,
        "the models of the Claude Code sessions and their reviewer agents on the Windows PC, 28-30 October 2026")
    fcm = ROOT / "handoff" / "extras" / "Flood_Cloud_Model"
    fsrc = "handoff/extras/Flood_Cloud_Model/"
    add("fcm_package_present", fcm.exists(), "present" if fcm.exists() else "absent", fsrc, "the teammates' package (config.json flood_cloud_model_authors)")
    if not fcm.exists():
        return
    rt = json.loads((fcm / "runtime.json").read_text(encoding="utf-8"))
    add("flood_cloud_model_python", rt.get("python"), str(rt.get("python")), fsrc + "runtime.json -> python", "the package runs in its own environment (scikit-learn, no torch)")
    for pkg in ("numpy", "scikit-learn"):
        v = (rt.get("packages") or {}).get(pkg)
        add("flood_cloud_model_" + pkg.replace("-", "_"), v, str(v), fsrc + f"runtime.json -> packages.{pkg}")
    cfg = json.loads((fcm / "config.json").read_text(encoding="utf-8"))
    y0, y1 = cfg["periods"]["training"][0][:4], cfg["periods"]["testing"][1][:4]
    add("fcm_period_years", [y0, y1], f"{y0} to {y1}", fsrc + "config.json -> periods.training[0], periods.testing[1]",
        "hourly data of 2021-2025: training 2021-2022, calibration and selection 2023, test 2024-2025")
    for k in ("training", "calibration", "selection", "testing"):
        a, b = cfg["periods"][k]
        add(f"fcm_period_{k}", [a, b], f"{a[:10]} to {b[:10]}", fsrc + f"config.json -> periods.{k}")
    m = re.search(r"all\s*(\d+)\s*members", cfg.get("forecast", ""))
    add("fcm_gefs_members", int(m.group(1)) if m else None, m.group(1) if m else "n/a", fsrc + "config.json -> forecast ('all31 members'); README.md 'all 31 members'",
        "operational GEFS members read by the package (not the reforecast control member of E2)")
    add("fcm_primary_horizon_hours", cfg.get("primary_horizon"), str(cfg.get("primary_horizon")), fsrc + "config.json -> primary_horizon")
    res = json.loads((fcm / "reference" / "results.json").read_text(encoding="utf-8"))
    A = res["arms"]
    note = ("the teammates' package, a separate hourly gradient-boosting experiment on 2021-2025 data; its event (hourly flow above a high-flow level of another "
            "period) differs from this project's onsets, so shares are not comparable in level; never quote the headline detection or accuracy without the "
            "persistence row and the onset-only recall (package review of 2026-10-28)")
    h = str(cfg.get("primary_horizon", 24))
    g, pers = A["flow_observed_rain_gefs72"][h], A["persistence"][h]
    add("fcm_test_cases_24h", g["n"], count(g["n"]), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.n", note)
    leaf = next((b for b in (g.get("by_basin") or {}) if b.endswith("02472000")), None)      # Leaf River, MS: the basin nearly absent from the package's test cases
    if leaf:
        add("fcm_leaf_river_test_cases_24h", g["by_basin"][leaf]["n"], count(g["by_basin"][leaf]["n"]), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.by_basin.{leaf}.n",
            "the package's test cases at Leaf River (gauge 02472000): its result rests on the other eleven basins (package review of 2026-10-28, section 7.7; critic 1, item 14)")
    add("fcm_onset_cases_24h", g["known_low"]["n"], count(g["known_low"]["n"]), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.known_low.n",
        "cases with the river below the level at the issue time")
    add("fcm_positive_onset_cases_24h", g["known_low"]["tp"] + g["known_low"]["fn"], count(g["known_low"]["tp"] + g["known_low"]["fn"]),
        fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.known_low.tp + fn", "onset-positive cases at 24 h")
    add("fcm_scorable_starts_24h", g["event_starts"]["scorable_starts"], count(g["event_starts"]["scorable_starts"]),
        fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.event_starts.scorable_starts", note)
    for arm, lab in (("flow_only", "flow"), ("flow_observed_rain", "rain"), ("flow_observed_rain_gefs72", "gefs")):
        E = A[arm][h]["event_starts"]
        add(f"fcm_starts_warned_{lab}_24h", E["detected_before_start"], count(E["detected_before_start"]),
            fsrc + f"reference/results.json -> arms.{arm}.{h}.event_starts.detected_before_start", note)
        add(f"fcm_starts_share_{lab}_24h", E["recall"], pct(E["recall"]), fsrc + f"reference/results.json -> arms.{arm}.{h}.event_starts.recall", note)
        ml = E.get("median_lead_hours")
        add(f"fcm_median_lead_{lab}_24h", ml, "n/a" if _bad(ml) else f"{ml:.0f} hours", fsrc + f"reference/results.json -> arms.{arm}.{h}.event_starts.median_lead_hours", note)
        KL = A[arm][h]["known_low"]
        add(f"fcm_onset_only_recall_{lab}_24h", KL["recall"], pct(KL["recall"], 1), fsrc + f"reference/results.json -> arms.{arm}.{h}.known_low.recall",
            "recall on the cases where the river was still below the level")
        add(f"fcm_onset_only_false_share_{lab}_24h", KL["false_warning_share"], pct(KL["false_warning_share"], 1),
            fsrc + f"reference/results.json -> arms.{arm}.{h}.known_low.false_warning_share")
    add("fcm_headline_accuracy", g["accuracy"], pct(g["accuracy"], 1), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.accuracy (README.md headline 97.5%)", note)
    add("fcm_headline_detection", g["recall"], pct(g["recall"], 1), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.recall (README.md headline 68.7% detection)", note)
    add("fcm_headline_precision", g["precision"], pct(g["precision"], 1), fsrc + f"reference/results.json -> arms.flow_observed_rain_gefs72.{h}.precision", note)
    add("fcm_persistence_detection_24h", pers["recall"], pct(pers["recall"], 1), fsrc + f"reference/results.json -> arms.persistence.{h}.recall",
        "persistence (warn when the river is already above the level) scores this on the same cases: the headline counts cases already above the level")
    add("fcm_persistence_precision_24h", pers["precision"], pct(pers["precision"], 1), fsrc + f"reference/results.json -> arms.persistence.{h}.precision")


# ====================================================================================================================================
def write_outputs():
    out = {"_meta": {"what": "the numbers registry of the final report and deck: every number the documents print, with its source; built by code/final_report/numbers.py from results/*.json",
                     "root": str(ROOT), "n_keys": len(REG), "sections": [t for t, _ in SECTIONS], "formatting": "percentages f'{100x:.0f}%', NSE f'{x:.2f}', counts with thousands separators, points = percentage points",
                     "rules": ["numbers only from results/*.json (never results/diagnostics_2026-10-28/)", "fixed wordings word for word",
                               "the sets of scripts 15-18 and 20-22 are five-seed numbers (seeds 3-4 added on 30 October 2026 after the test scores of seeds 0-2 were known: seed_note_16); the stored sets of scripts 02-10 keep three (stored_sets_three_seeds_list); quote the n_seeds key of each set",
                               "E2b (_s) sets are comparable only inside the satellite-era split", "AORC is an analysis made after the fact, never 'observed'", "the team's own checks are not done (team.json)"],
                     "files_read": sorted(f"results/{k}.json" for k in _CACHE)},
           "_optional_sources": OPTIONAL_SOURCES}
    out.update(REG)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    # the key table for the writers
    lines = ["# NUMBERS_KEYS.md: the keys of results/final_report_numbers.json", "",
             "Generated by `python -X utf8 code/final_report/numbers.py` (rebuild, never hand-edit). Writers reference a number as `{{num:key}}`; the builder replaces it by the `text` field.",
             "Never type a number. Formatting of `text`: percentages as whole numbers (`19%`), NSE to two decimals (`0.59`), counts with thousands separators (`1,116`), differences in percentage points (`+32.6 points`),",
             "ranges as `49% to 55%` or `0.54 to 0.60` (lowest to highest single training). The sets of scripts 15-18 and 20-22 are five-seed numbers since 30 October 2026 (seeds 3-4 added after the test scores of seeds 0-2 were known: `seed_note_16`); the stored sets of scripts 02-10 keep three (`stored_sets_three_seeds_list`; see the `n_seeds` keys).",
             "The `source` column names the results file and the dotted JSON path. `note` carries the honesty constraint that travels with the number (fixed wording, 'another split', 'upper bound', PRELIMINARY).", "",
             f"Keys: {len(REG)}. Sections follow the report template. The per-set score family (section 7.1) is listed compactly: one row per prediction set and lead; its {len(SET_SUFFIXES)} fields are",
             "`<set>_l<N>_` + " + ", ".join(f"`{s}`" for s in SET_SUFFIXES) + ".", "",
             "Optional E4 keys (present only if the files exist; see the last section) are prefixed `e4a_` (results/warning_policy.json), `e4b_` (results/within_day_inputs.json), `e4e_` (results/flow_state.json),",
             "`e4a_review_` / `e4b_review_` (their review records). Expected names: `e4a_file_present`, `e4a_outcome`, `e4a_outcome_full`, `e4a_status`, `e4a_<set>_l<N>_far_cap_pod` / `_far` / `_trigger`",
             "(the operating point at the GO false share), `e4a_<set>_l<N>_csi_pod` ..., `e4b_outcome_full`, `e4b_<set>_l<N>_nse` / `_onset_pod` / `_onset_far` / `_onset_csi` / `_onset_hits`, `e4e_outcome_full`, `e4e_<set>_l<N>_onset_pod`,",
             "Added 2026-10-29 for the final report: `e4b_<set>_l<N>_far_cap_pod` / `_far` / `_trigger` and `e4e_<set>_l<N>_far_cap_pod` / `_far` / `_trigger` (the test-year onset scores at the level that caps the validation false share;",
             "also `_test_onset_*`, `_validation_onset_*`, `_test_verdict_onset`; the same for the `csi` and `plain` rules), `e4e_paired_<first>_vs_<second>_l<N>_only_with` / `_only_without` / `_both` / `_neither`, `e4e_hourly_basin_days`,",
             "`e4e_fallback_basin_days_24` / `_22` / `_24_test_years`, `e4e_fallback_days_24_test_years_<gauge>`, and `date_e4_reviews` (the dates of the three reviews).",
             "`e4a_review_verdict`, `e4a_review_checks_listed`, `e4a_review_failed`. A writer wraps them in an optional block: if `e4a_file_present` is absent or 'absent', print the future-work sentence instead.", ""]
    set_keys = {k for _, _, base in SET_TABLE for k in REG if k.startswith(base + "_")}
    for title, keys in SECTIONS:
        lines += [f"## {title}", ""]
        if title.startswith("7.1 Test-year scores"):
            lines += ["| set | lead | NSE | NSE range | onsets warned | range | false | CSI | hits/misses/false alarms | all days warned | all false | all CSI | onset verdict (fixed / chosen) | all-days verdict (fixed / chosen) | source file |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for name, L, base in SET_TABLE:
                g = lambda s: REG[f"{base}_{s}"]["text"]
                sf = REG[f"{base}_nse"]["source"].split(" -> ")[0]
                lines.append(f"| `{name}` | {L} | {g('nse')} | {g('nse_range')} | {g('onset_pod')} | {g('onset_pod_range')} | {g('onset_far')} | {g('onset_csi')} | {g('onset_hits')}/{g('onset_misses')}/{g('onset_false_alarms')} | {g('all_pod')} | {g('all_far')} | {g('all_csi')} | {g('onset_verdict')} / {g('onset_verdict_chosen')} | {g('all_verdict')} / {g('all_verdict_chosen')} | {sf} |")
            lines.append("")
            lines.append("Set labels: " + "; ".join(f"`{k}` = {v}" for k, v in SET_LABELS.items()) + ".")
            lines.append("")
            rest = [k for k in keys if k not in set_keys]
        else:
            rest = keys
        if rest:
            lines += ["| key | text | source |", "|---|---|---|"]
            for k in rest:
                e = REG[k]
                t = str(e["text"]).replace("|", "\\|").replace("\n", " ")
                if len(t) > 160:
                    t = t[:157] + "..."
                s = str(e["source"]).replace("|", "\\|")
                if e.get("note") and len(t) < 100:
                    n = str(e["note"]).replace("|", "\\|").replace("\n", " ")
                    s += " — " + (n if len(n) <= 120 else n[:117] + "...")
                lines.append(f"| `{k}` | {t} | {s} |")
            lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
    write_outputs()
    check = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    n = len([k for k in check if not k.startswith("_")])
    print(f"wrote {OUT_JSON} ({n} keys) and {OUT_MD}")
