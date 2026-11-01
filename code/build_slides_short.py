"""Build the FIVE-slide version of the 10-minute talk: Flood_Slides_5.html, Flood_Slides_5.pdf (16:9, one slide per page), Flood_Slides_5.pptx (the same
slides, editable, with speaker notes), slides_notes_5.md (the speaker notes) and results/slide_plan_5.json (who speaks when; the presentation plan of the short
form, code/build_presentation_plan_short.py, reads that file).

This is the hand-in form of the talk: only what is critical to show the work. The full deck (code/build_slides.py -> Flood_Slides.pdf, 13 slides and 7 backup
slides, with slides_notes.md) stays the detailed record and the place to look when a question goes beyond the five slides; nothing of the full deck's outputs
(Flood_Slides.*, slides_notes.md, results/slide_plan.json) is written or changed here.

Nothing is computed here. Every number on a slide and in the notes is read from results/*.json and results/*.csv (and the log of the GEFS download in
results/logs) and formatted inside an f-string, through the same readers the full deck uses: the module code/build_slides.py is IMPORTED and its functions are
reused (the readers _all_basins, _answer, _key, _need, _new, _reviews, _transformer, _windows; the spec functions s_cover, s_simple, s_all, s_verdict, s_key,
s_rain, s_forecast ... of which a slide here takes the title, the bullets, the caption or the sentences to say and the facts for questions; the HTML writers for the
layouts the full deck has, the helpers of its PPTX writer, the footer, the pace check) and the fixed wordings of code/report_sections/s0_summary.py (FINDING, SUPPORT,
AORC_WORDING, MOVED_POINT, INFO_NOT_NET, REALTIME_WORDING, HINDSIGHT_WORDING, NSE_FORECAST_WORDING, REVIEW_WORDING) are used word for word. Only what the full deck's
code does not allow is written here: the dataset slide (a new layout, 'stack': the flow diagram of results/data_flow.json across the slide, five lines under it, all read from
results/data_flow.json), the fourth slide (a new layout, 'split': two halves side by side, each with a small table and a few lines, for one speaker), a line of small print under a
table slide, the writers that know about these layouts (the PPTX loop, the notes, the plan), and the work-in-numbers count of the last slide.

The five slides and the time of each (600 s; each of the four members speaks 150 s in all):
  1  cover, the question, the data in a line, the fair test            the team lead          70 s
  2  the dataset: what, how obtained, the flow, how computed, imagery  RS/GIS member 2  150 s
  3  the key figure                                                    GNSS member 1    150 s
  4  the number to beat, and what more information did                 GNSS member 2    150 s
  5  conclusions, limits, next steps, the work in numbers, who did what  the team lead          80 s
Every slide is one part with one presenter, one time, its sentences to say, its hand-over sentence and its facts for questions (slide 4 has two halves to look at, but one speaker).
The pace rule of the full deck is kept for every part: at most PACE words a second with the hand-over counted, no sentence over MAX_SENTENCE words.

The builder warns on stderr (nothing is printed there when all is well) when a text is estimated to run past its slide, when the sentences of a part are too many for its time
or one is too long, when a crop was refused, when a result or a file is missing, when the PPTX it wrote does not pass its own check, when the seconds do not add up to ten
minutes or the four members do not speak the same time, or when notes and plan disagree.
Rebuild after any rerun:  python code/build_slides_short.py   (then python code/build_presentation_plan_short.py)
"""
import contextlib
import datetime
import html
import json
import re
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.util import Inches, Pt

import build_slides as BS
import team
from common import FIG, RES, ROOT
from report_common import LEAD_WORD, REFERENCES, count, load_results, n2, page, pct, print_pdf, signed
from report_sections import s0_summary as S0
from report_sections import s7_cost

NAME = "Flood_Slides_5"
ORDER = ("cover", "data", "key", "more", "end")                   # the five slides, in the order of the talk (their keys)
AT = {k: i for i, k in enumerate(ORDER, 1)}                           # key -> the slide's number in this deck
DECK = BS.DECK
LEADS, lk = BS.LEADS, BS.lk
TOTAL_SECONDS = 600                             # ten minutes
EACH_SECONDS = 150                              # every member speaks the same time in all: a quarter of the ten minutes (set at the team lead's request on 2026-10-08)
SLIDES = 5
Q, QC = BS.Q, BS.QC
UPPER, AORC_WORDING, MOVED_POINT, INFO_NOT_NET = BS.UPPER, S0.AORC_WORDING, S0.MOVED_POINT, S0.INFO_NOT_NET
REALTIME_WORDING, HINDSIGHT_WORDING, REVIEW_WORDING = S0.REALTIME_WORDING, S0.HINDSIGHT_WORDING, S0.REVIEW_WORDING
NO_CLEAR = BS.NO_CLEAR

# ---- the new layout 'split': two halves side by side (inches and points) ---------------------------------------------------
# Each half has a head (what it is about, and who speaks it), a compact table, an optional note under the table and a few lines. (The figures of the experiments were
# tried first, as cropped panels of the figures of the AORC and forecast experiments: at half the width of the slide their lettering came out at about 5 pt and the slide
# ran two inches past its frame, so the halves show the same numbers as small tables; the figures stay on the key-figure slide and in the full deck.)
SPLIT_GAP = 0.50                                # between the two halves (a hairline runs through the middle of it)
SPLIT_HEAD_PT, SPLIT_HEAD_LH = 18, 1.2          # the line above each half: what the half is about, and who speaks it
SPLIT_TABLE_PT, SPLIT_PAD = 13, (6, 2)          # the table of a half: font size; cell padding (side, top/bottom) in points (a half may ask for another size: 'table_pt')
SPLIT_WIDTHS = (0.40, 0.60)                     # the shares of the two halves' widths: the right half has the longer table (the whole ladder of what more information did)
SPLIT_NOTE_PT = 11                              # the note under a table (as the label under a figure)
SPLIT_BODY_PT = 14                              # the lines under each half's table (a half may ask for another size: 'body_pt')
STACK_GAP = 0.10                                # the dataset slide ('stack'): the space between the label of the figure and the lines under it, in inches
DATA_CROP = (0, 0, 1, 0.528)                    # the part of figures/data_flow_talk.png that the dataset slide shows: the legend, the product boxes and stages 1 to 5 (cut at the blank rows under them)
DATA_FIG_W, DATA_BODY_PT = 12.23, 14            # the widest the figure may be (the slide's text width, inches) and the size of the five lines under it (points)
MIN_LETTER_PT = 9                               # the builder warns when the smallest lettering of that figure on the slide is smaller than this (points)
SPLIT_HEAD_GAP, SPLIT_NOTE_GAP, SPLIT_BULLETS_GAP = 0.06, 0.05, 0.12
SMALL_PRINT_PT, SMALL_PRINT_LH, SMALL_PRINT_GAP = 11, 1.28, 0.0      # the small print under the lines of a table slide (a slide may have 'small_print', a list of paragraphs)
CSS_SPLIT = f"""
.slide.split h1{{position:absolute;left:{BS.MX}in;top:{BS.TOP}in;width:{BS.W - 2 * BS.MX:.3f}in;margin:0}}
.slide.split .hd{{position:absolute;margin:0;font-size:{SPLIT_HEAD_PT}pt;line-height:{SPLIT_HEAD_LH};color:var(--ink)}}
.slide.split .tb{{position:absolute;font-size:{SPLIT_TABLE_PT}pt}} .slide.split .tb table{{font-size:inherit;line-height:{BS.TABLE_LH};margin:0}}
.slide.split .tb th,.slide.split .tb td{{padding:{SPLIT_PAD[1]}pt {SPLIT_PAD[0]}pt}} .slide.split .tb tr:last-child td{{border-bottom:1px solid var(--grid)}}
.slide.split .nt{{position:absolute;margin:0;font-size:{SPLIT_NOTE_PT}pt;line-height:{BS.CAP_LH};color:var(--ink2)}}
.slide.split .bl{{position:absolute;font-size:{SPLIT_BODY_PT}pt}} .slide.split .bl ul.pts{{font-size:inherit}}
.slide.split .vr{{position:absolute;width:0;border-left:1px solid var(--grid)}}
.slide.stack h1{{position:absolute;left:{BS.MX}in;top:{BS.TOP}in;width:{BS.W - 2 * BS.MX:.3f}in;margin:0}}
.slide.stack .bl{{position:absolute}} .slide.stack .bl ul.pts{{font-size:{DATA_BODY_PT}pt}}
.slide .small5{{font-size:{SMALL_PRINT_PT}pt;line-height:{SMALL_PRINT_LH};color:var(--ink2);margin:{SMALL_PRINT_GAP}in 0 0}} .slide .small5 + .small5{{margin-top:0.04in}}
"""

LEFT_OUT = BS.LEFT_OUT                          # what was left out because a file or a key was missing: one line each on stderr (the list of the full deck's module)
_left_out = BS._left_out


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def _slash(vals):
    return " / ".join(vals)


def _plain(s):
    return BS._plain(s)


def _md(s):
    return BS._md(s)


def _num(x, dec=0):
    """A size in GB for a sentence: 4.5, 16, 48."""
    return f"{x:.{dec}f}"


def _said_years(span):
    """A period as it is said: ['1980-10-01', '2000-09-30'] -> 'from 1980 to 2000'."""
    return f"from {span[0][:4]} to {span[1][:4]}"


def segments(s):
    """The parts of a slide that one presenter speaks. Every slide is one part (slide 4 has two halves to look at, but one speaker): the slide itself has owner, seconds, notes (the
    sentences to say), handover, asked (facts for questions) and, optionally, if_late."""
    return [s]


def _view(seg):
    """A segment as the pace functions of the full deck's module read a slide: notes, handover, seconds."""
    return dict(notes=seg["notes"], handover=seg.get("handover", ""), seconds=seg["seconds"])


# ---- the facts that more than one slide needs -----------------------------------------------------------------------------------
def work_in_numbers(R, T):
    """The work of the project in numbers, every one read from a results file: the network trainings and their hours (the same count and sum as section 7 of the
    report, through its function _project_cost), the data read (results/aorc_data.json, results/stage4_realtime.json and the log results/logs/fetch_gefs.log),
    the independent AI reviews (every results/review_*.json: the rounds and the checks listed, the numbers the final critic compared) and whether anything was
    bought (results/data_access.json). The data read are those of results/data_flow.json (all_information.bytes_read: the recorded transfers of AORC, GEFS and Stage IV and the
    sizes of the stored files of the other products), the same 73 GB as the dataset slide; the log of the GEFS download only gives the megabytes a day. A part whose file is
    missing is None, with one line for stderr."""
    out = dict(trainings=None, hours=None, gb=None, gb_parts=None, rounds=None, checks=None, clean=None, compared=None, files=None, money=None)
    found = s7_cost._project_cost(R, T)
    if found:
        out.update(trainings=sum(n for _, _, _, n in found if n), hours=sum(s for _, _, s, _ in found) / 3600, runs=found)
    else:
        _left_out("no results file with a training count was found: the last slide does not count the trainings")
    F = data_facts(R)                                                  # the bytes read: results/data_flow.json adds up the recorded transfers and the sizes of the stored files
    out.update(gb=F["gb_total"], gb_parts=F["gb_parts"], gb_recorded=F["gb_recorded"], trainings_recorded=F["trainings_recorded"])
    log = RES / "logs" / "fetch_gefs.log"
    runs = [(int(d.replace(",", "")), int(m.replace(",", ""))) for d, m in
            re.findall(r"SUMMARY: exit \d+ \| ([\d,]+) days stored this run .*? \| ([\d,]+) MB", log.read_text(encoding="utf-8"))] if log.exists() else []
    if runs:
        out["gefs_mb_a_day"] = sum(m for _, m in runs) / sum(d for d, _ in runs)
    else:
        _left_out("results/logs/fetch_gefs.log has no SUMMARY line with a size: the notes of the last slide leave out the GEFS megabytes a day")
    recs = {k: v for k, v in R.items() if k.startswith("review_") and isinstance(v, dict) and not v.get("quick_mode")}
    if recs:
        blocks = {k: BS._counts_blocks(v) for k, v in recs.items()}
        cs = [c for b in blocks.values() for c in b]
        out.update(rounds=len(cs), checks=sum(c.get("checks_listed", 0) for c in cs), files=sorted(recs),
                   clean=all(c.get("failed", 0) == 0 and c.get("numerical_discrepancies", 0) == 0 for c in cs),
                   compared=sum(c.get("numbers_compared_in_all", 0) for c in cs) or None, per_record={k: (len(b), sum(c.get("checks_listed", 0) for c in b)) for k, b in blocks.items()})
    else:
        _left_out("no results/review_*.json was found: the last slide does not count the review rounds")
    out["money"] = "Nothing was bought" in str((R.get("data_access") or {}).get("what", ""))
    if not out["money"]:
        _left_out("results/data_access.json does not say that nothing was bought: the last slide leaves out 'money spent: none'")
    return out


def _revised():
    """The day this builder was last changed (the date of the file), as words: the day the five slides were last revised. Whoever changes the builder after Claude Code
    must also change the words 'also by Claude Code' where this date is printed."""
    return BS._date(datetime.date.fromtimestamp((ROOT / "code" / "build_slides_short.py").stat().st_mtime).isoformat())


# what each slide of the full deck is about, in a few words (no numbers): its own key -> label, for the lines 'where the detail is'
LABEL = {"cover": "the question and the short answer", "data": "the data and the rule that chose the rivers", "fair": "why the test is fair", "simple": "the number to beat",
         "all": "the LSTM against the simple models", "event": "one onset, day by day", "verdict": "what the warnings are worth", "fail": "where the onset warnings fail",
         "timing": "the rain-timing test", "rain": "the rain known up to the issue time", "forecast": "a real weather forecast", "need": "what it took, and what is missing",
         "key": "the key figure and the conclusion", "curves": "the learning curves", "memory": "how far back the memory reaches", "lobo": "a river never seen in training",
         "rivers": "more training rivers", "scores": "the warning scores at every lead time", "cost": "the Transformer and what training costs", "space": "satellite rain as an input"}


def _detail_keys():
    """Which slides of the full deck stand behind each single-part slide of the short form (the plan lists them; keys of the full deck's slides). The halves of slide 4
    carry their own list ('detail_keys')."""
    return {"cover": ["cover", "fair"], "data": ["data", "timing", "space"], "key": ["key", "timing"], "end": ["need", "rivers", "lobo"]}


# ---- the facts of the dataset slide: results/data_flow.json -------------------------------------------------------------------------
def data_facts(R):
    """results/data_flow.json (code/19_data_flow.py), the one source of the data text of the report and the slides, read by key: the numbers of the dataset slide, of its
    notes and of the data lines of the last slide. A file that is missing, is a smoke test or lacks a key stops the build with one line: the slide cannot be built without it."""
    DF = R.get("data_flow")
    if not isinstance(DF, dict) or not DF or DF.get("quick_mode"):
        raise SystemExit("build_slides_short: results/data_flow.json (code/19_data_flow.py) is missing or is a smoke test: the dataset slide cannot be built")
    try:
        st = {x["id"]: x for x in DF["flow"]["stages"]}
        c = {k: v["counts"] for k, v in st.items()}
        AI, hc, talk = DF["all_information"], DF["how_computed"], DF["figure"]["talk"]
        roles = [x["role"] for x in talk["sources"]]
        per, total = AI["bytes_read"]["per_product"], AI["bytes_read"]["total"]
        names = {"gefs": "GEFS", "stage4": "Stage IV", "aorc": "AORC", "camels_us": "CAMELS-US"}
        top = sorted(per.items(), key=lambda kv: -kv[1])[:4]
        gb_parts = {names.get(k, k): v / 1e9 for k, v in top}
        gb_parts["the rest"] = (total - sum(v for _, v in top)) / 1e9
        m = re.search(r"cfs x ([\d.]+) x (\d+) / area_m2 x (\d+)", hc["flow_mm_per_day"]["formula"])
        if not m:
            raise KeyError("how_computed.flow_mm_per_day.formula is not 'cfs x a x b / area_m2 x c'")
        note = DF["imagery_note"]
        join = re.search(r"\(stage (\d+)\)", talk["side"]["title"])                  # 'Extra inputs of the experiments join the daily inputs (stage 4)'
        n_st = len(DF["flow"]["stages"])
        rule, vals = AI["basins"]["rule"], hc["where_each_daily_value_ends"]["values"]
        utc = re.search(r"\d\d:\d\d", vals["imerg"]["documented"])                    # '19:00 (the UTC day)'
        srtm, nlcd = re.search(r"SRTM radar (?:flown in|mission of) ([A-Z][a-z]+ \d{4})", note), re.search(r"NLCD land-cover map of (\d{4})", note)
        F = dict(
            DF=DF, n_products=c["download"]["products"], gb_read=c["download"]["gb_read"], gb_kept=c["download"]["gb_kept_under_data"], gb_total=total / 1e9,
            gb_recorded=AI["bytes_read"]["recorded_transfers"] / 1e9, gb_parts=gb_parts, no_payment="No payment and no account is recorded" in st["download"]["words"],
            all_camels=c["audit"]["all_camels"], candidates=c["audit"]["candidates"], complete=c["audit"]["complete"], chosen=c["audit"]["chosen"],
            days=c["inputs"]["days_per_basin"], basin_days=c["inputs"]["basin_days"], missing=c["inputs"]["missing_values"], properties=c["inputs"]["properties"],
            train=c["split"]["train_days"], val=c["split"]["validation_days"], test=c["split"]["test_days"],
            train_only="training years only" in st["scaling"]["words"], n_sets=c["forecasts"]["prediction_sets"], networks=c["forecasts"]["saved_networks"],
            trainings_recorded=c["forecasts"]["trainings_recorded"], n_extra=roles.count("extra input"), n_check=roles.count("check or map"), n_stages=n_st,
            first_row=n_st // 2, first_title=re.sub(r"^the ", "", DF["flow"]["stages"][0]["title"].lower()), last_title=re.sub(r"^the ", "", DF["flow"]["stages"][n_st // 2 - 1]["title"].lower()),     # the figure has the stages in two rows
            join_stage=int(join.group(1)) if join else None,
            rule=dict(huc=[f"{int(h):02d}" for h in rule["huc_regions"]], snow=rule["max_snow_share"], arid=rule["max_aridity"], lo=rule["area_km2"][0], hi=rule["area_km2"][1],
                      per=rule["per_landscape"]), n_landscapes=len(AI["basins"]["chosen_by_landscape"]), blind="no flow statistic and no model score enters" in st["audit"]["words"],
            clocks=dict(daymet=vals["daymet"]["rounded"], nldas=vals["nldas"]["rounded"], utc=utc.group(0) if utc else None, river=str(vals["river_flow"]).startswith("local midnight")),
            cfs=m.group(1), secs=int(m.group(2)), mm=m.group(3),
            daymet_end=hc["where_each_daily_value_ends"]["values"]["daymet"]["rounded"], daymet_median=hc["where_each_daily_value_ends"]["values"]["daymet"]["median"],
            imagery=note, srtm=srtm.group(1) if srtm else "", nlcd=nlcd.group(1) if nlcd else "", unverified=DF.get("unverified") or [], rows=AI["records"],
            letter_pt=DF["figure"]["renderings"]["talk"]["smallest_lettering_pt"], design_w=DF["figure"]["renderings"]["talk"]["width_in"])
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as e:
        raise SystemExit(f"build_slides_short: results/data_flow.json lacks a key or is not in the expected form ({type(e).__name__}: {e}): the dataset slide cannot be built")
    return F


def _end(x):
    """A sentence of the record with a full stop at its end."""
    x = str(x).strip()
    return x if x.endswith((".", "!", "?", ".)", "!)", "?)")) else x + "."


def _size(nbytes):
    """A size in bytes as it is said: '48.5 GB', '86 MB'."""
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"


def _sq(x):
    """A text of the record with the angle brackets of a file name pattern ('<model>_<inputs>') turned into square ones, so that no viewer takes them for markup."""
    return re.sub(r"<([A-Za-z_][A-Za-z0-9_\-]*)>", r"[\1]", str(x))


def data_asked(F):
    """The facts for questions on the dataset slide, all from results/data_flow.json: the record itself, the ten stages, the eleven products with everything the record says of
    each (what it is, who made it, version, grid, time step, period, time stamp, real-time latency, how it was obtained, imagery), the formulas of how the data are computed, the
    counts of 'all the information', the imagery note in full and what the project could not verify. The text of the file is escaped as the full deck escapes a sentence of a results
    file."""
    DF, T_ = F["DF"], lambda x: BS._typeset(_sq(x))
    out = [T_(f"The record behind this slide, results/data_flow.json: {_end(DF['what'])}")]
    out += [T_(f"Stage {s['n']}, {s['title']}: {_end(s['words'])}") for s in DF["flow"]["stages"]]
    for x in DF["datasets"]:
        o = x["obtained"]
        out.append(T_(f"{x['name']}: {_end(x['one_line'])} Made by {_end(x['producer'])} Version: {_end(x['version'])} Grid: {_end(x['spatial_grid'])} Time step: {_end(x['time_step'])} "
                      f"Period used: {_end(x['period_used'])} Time stamp: {_end(x['time_stamp'])} Real-time latency: {_end(x['latency_real_time'])} Obtained: login: {o['login_needed']}; "
                      f"{_size(o['bytes_read'])} ({o['bytes_read_note']}); downloaded {o['date_downloaded']}; script: {o['script']}. Imagery: {_end(x['imagery'])}"))
    hc = DF["how_computed"]
    for k in ("flow_mm_per_day", "basin_mean_with_overlap_weights", "value_at_the_centroid", "local_day_sum", "where_each_daily_value_ends", "scaled_inputs", "high_flow_level_and_onset"):
        if k in hc:
            out.append(T_(f"How computed, {hc[k]['title'].lower()}: {_end(hc[k]['words'])}" + (f" Formula: {hc[k]['formula']}." if hc[k].get("formula") else "")))
    r = F["rows"]
    out.append(T_(f"All the information, in counts: {count(r['basin_days'])} basin-days, {count(r['values_weather_and_flow'])} values of weather and flow, {r['basin_property_values']} property "
                  f"values, {r['missing_values']} missing; {count(r['estimated_flow_days'])} flow days ({100 * r['estimated_share_of_all_basin_days']:.1f}% of all) are marked as estimated by the "
                  f"USGS and {count(r['zero_flow_days'])} are zero-flow days; the split is {count(F['train'])} / {count(F['val'])} / {count(F['test'])} days; {F['n_sets']} forecast sets are "
                  f"stored and {F['networks']} networks are saved."))
    out.append(T_(f"The imagery note in full: {_end(F['imagery'])}"))
    if F["unverified"]:
        out.append(T_("What the project could not verify, as the record lists it: " + " ".join(f"({i}) {x['dataset']}: {_end(x['fact'])}" for i, x in enumerate(F["unverified"], 1))))
    return out


# ---- the slides ------------------------------------------------------------------------------------------------------------------
def s_cover(R, T, RAW, put):
    """Slide 1: the title, the question, the data, the models, the fair test, and who did what in small print (the team lead, 70 s)."""
    c = RAW["cover"]
    A, split, first = R["data_audit"], R["baselines"]["split"], LEADS[0]
    n = len(A["selected"])
    start, stop = (datetime.date.fromisoformat(d) for d in A["period"])
    n_years = round(((stop - start).days + 1) / 365.25)
    ask = f"Can an LSTM forecast river flow {first} to {LEAD_WORD[LEADS[-1]]} better than simple models, and could its warnings be trusted?"
    ER = R["engine_review"]
    reviewed = ("Independent AI reviewer agents checked the shared engine and the experiments; a critic agent made the final checks. " if "AI reviewer" in ER["who"] else "")
    facts = [f"<b>Data, all public:</b> {n} rivers in the south-eastern United States, {n_years} years, {BS._years(A['period'])}: CAMELS-US {BS.cite('newman2015')}.",
             f"<b>Models:</b> one LSTM for all {n} rivers, with simple models beside it.",
             f"<b>A fair test:</b> train {BS._years(split['train'])}; every setting chosen on the validation years {BS._years(split['val'])} only; the test years "
             f"{BS._years(split['test'])} are only scored."]
    who = f"<b>Who did what:</b> Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction. {reviewed}{BS._team_check(R)[1]}"
    says = [f"Our question: {ask[0].lower() + ask[1:]}",
            "An LSTM is a neural network with a memory: it reads the days in order.",
            f"The data are public: {n} rivers in the south-eastern United States, {n_years} years of daily weather and river flow.",
            f"One LSTM learns all {n} rivers. Simple models run beside it: {Q}, ridge regression and a random forest.",
            f"The test is fair. The models learn {_said_years(split['train'])}, their settings are chosen on the validation years {_said_years(split['val'])[5:]}, and the test "
            f"years {_said_years(split['test'])[5:]} are only scored.",
            c["notes"][-1]]                                                    # 'To be plain: Claude Code wrote the code ... The team's own checks are not yet done.'
    d_ = ER.get("dates") or {}                                        # one check was reviewed and rerun a day later
    late = (f"the day-window check was reviewed and rerun on {BS._date(d_['day_window_check_rerun'])}; "
            if str(d_.get("day_window_check_rerun", ""))[:10] > str(d_.get("everything_else", ER["when"]))[:10] else "")
    audit_day, run_day = BS._days(R)
    when = (f"The data audit and the choice of the basins were made on {audit_day}, the experiments were run on {run_day}; " if audit_day and audit_day != run_day else
            f"The experiments were run on {run_day}; ")
    NWK = S0.new_work(R)
    four = f"; {BS._word(NWK['n'])} more experiments were run on {NWK['days']}" if NWK else ""
    N = BS._need(R)
    V = R["flood_analysis"]["verdict"]["services"]["lstm_wq"]
    v_on = {V[lk(k)]["onset"]["pooled"] for k in LEADS}
    short = (f"The short answer, in full: milestone 2 of our plan is {N['m2']} (slide {AT['more']}); the onset warnings are {BS._and(sorted(v_on))} at {BS._leads(LEADS)} "
             f"(slide {AT['more']}); the picture of the finding is slide {AT['key']}; where the data come from is slide {AT['data']}.")
    item0 = (f"{when}{late}these five slides were last revised on {_revised()}, also by Claude Code (the day code/build_slides_short.py was last changed), and built on "
             f"{BS._date(datetime.date.today().isoformat())}{four}. {BS._review_sentence(R)}")
    own = [put(x) for x in RAW["data"]["asked"] if x.startswith("In each map") or "LSTM" in x]          # the facts about the map and about the LSTM stay with the cover
    asked = ([item0] + own + [put(x) for x in RAW["fair"]["asked"]] + [short] + [put(x) for x in c["asked"][1:2] if x]
             + [f"The test: {n} river basins in the south-eastern United States. An LSTM is a neural network with a memory; the data are daily, {BS._years(A['period'])}. "
                f"The dataset has a slide of its own (slide {AT['data']})."])
    return dict(
        key="cover", layout="cover", owner="A", owners=["A"], seconds=70, image=c["image"], kicker=c["kicker"].replace("10-minute talk", f"10-minute talk, {BS._word(SLIDES)} slides"),
        title=DECK, sub=ask, small=facts + [c["small"][0], who], notes=says, handover="First: where do the data come from?", asked=asked,
        says=f"The question; the data ({n} rivers, {n_years} years, all public); the models; the fair test; who did what",
        figure_words="the map shows the study basins, coloured by landscape; the basins and the maps come from the remote-sensing and GIS part.")


def s_data(R, T, RAW, put):
    """Slide 2, the dataset (RS/GIS member 2, 150 s): the flow diagram of results/data_flow.json (figures/data_flow_talk.png, cut after the first row of stages: the full figure is in
    the report) and five lines: what, how obtained, the flow, how computed, imagery. Every number and every phrase is read from results/data_flow.json."""
    F = data_facts(R)
    n, w = F["n_products"], BS._word
    img = "data_flow_talk.png"
    if not (FIG / img).exists():
        raise SystemExit(f"build_slides_short: figures/{img} (code/19_data_flow.py) is missing: the dataset slide cannot be built")
    gb, arrow = F["gb_read"], " &rarr; "
    nofile = "none missing" if F["missing"] == 0 else f"{count(F['missing'])} missing"
    bullets = [f"<b>What:</b> every model reads CAMELS-US (USGS daily flow, Daymet weather as basin means, {F['properties']} basin properties); {w(F['n_extra'])} more products are extra inputs of "
               f"some experiments (green), {w(F['n_check'])} only checks and maps (grey); grid, step and period are in the boxes.",
               f"<b>How obtained:</b> all {n} are public" + ("; no account and no payment is recorded for any of them" if F["no_payment"] else "")
               + f"; {gb:.1f} GB were read in all, {F['gb_kept']:.1f} GB are kept.",
               f"<b>The flow:</b> {count(F['all_camels'])}{arrow}{F['candidates']}{arrow}{F['complete']}{arrow}{F['chosen']} basins (all, candidates, complete, chosen by rule); {count(F['days'])} days each, "
               f"{nofile}; {count(F['train'])} training, {count(F['val'])} validation and {count(F['test'])} test days; scaling on the training years only; {F['n_sets']} forecast sets stored.",
               f"<b>How computed:</b> flow in mm a day (cfs &times; {F['cfs']} &times; {count(F['secs'])} &divide; basin area in m&sup2; &times; {F['mm']}); gridded rain averaged over each basin with "
               f"overlap weights; hourly rain summed over the local wall-clock day; the Daymet day ends at about {F['daymet_end']}, as measured.",
               "<b>Imagery:</b> no optical scene was classified: capture date and sun angle do not apply; the facts per product are in the data record."]
    rule, clk = F["rule"], F["clocks"]
    if not (clk["utc"] and clk["river"]):
        raise SystemExit("build_slides_short: results/data_flow.json (how_computed.where_each_daily_value_ends) does not give the clocks in the form the dataset slide reads: it cannot be built")
    audit = [f"The audit goes from {count(F['all_camels'])} CAMELS basins to {F['candidates']} candidates: HUC regions {BS._and(rule['huc'])}, snow under {pct(rule['snow'])} of "
             f"precipitation, aridity under {rule['arid']:g} and {count(rule['lo'])} to {count(rule['hi'])} square kilometres.",
             f"Of these, {F['complete']} have a complete record, and {F['chosen']} are chosen: {w(rule['per'])} in each of {w(F['n_landscapes'])} landscapes, evenly spaced in area.",
             ("No flow statistic and no model score enters the choice, and each" if F["blind"] else "Each") + f" of the {F['chosen']} has {count(F['days'])} days"
             + (" with no value missing." if F["missing"] == 0 else ".")]
    clocks = [f"We measured where each day ends, in local standard time: Daymet at about {clk['daymet']}, NLDAS at about {clk['nldas']}, IMERG and MERRA-2 at {clk['utc']}, "
              f"the river flow at local midnight.",
              "So the afternoon rain of a Daymet day is filed under the next day."]
    says = [f"{BS._cap(w(n))} public products, one pipeline.",
            f"Every model reads CAMELS-US: the river flow measured daily by the US Geological Survey, the Daymet weather averaged over each basin, and {w(F['properties'])} basin properties.",
            f"{BS._cap(w(F['n_extra']))} more products are extra inputs of some experiments, in green; {w(F['n_check'])} are only checks and maps, in grey. Each box gives its grid, step and period.",
            ("All are public, with no account and no payment recorded." if F["no_payment"] else "All are public.") + f" We read {gb:.1f} GB and keep {F['gb_kept']:.1f} GB.",
            *audit,
            f"The split is by target day: {count(F['train'])} days to train, {count(F['val'])} to validate and {count(F['test'])} to test.",
            ("Inputs are scaled with the training years only, so nothing of the test years leaks in. " if F["train_only"] else "") + f"We store {F['n_sets']} sets of forecasts.",
            "How it is computed: the flow becomes millimetres a day, the unit of rain.",
            "Gridded rain is averaged over each basin, every cell weighted by its overlap; hourly rain is summed over the local wall-clock day.",
            *clocks,
            "No optical satellite scene was classified, so capture date and sun angle do not apply."
            + (f" The ready-made satellite products are IMERG rain, SRTM-format elevation tiles, mainly from the radar mission of {F['srtm']}, and a land-cover map of {F['nlcd']}."
               if F["srtm"] and F["nlcd"] else "")]
    public = ("All the data are public: no account was created and nothing was bought (results/data_access.json); each product is named, with its address and the day it was read, in the "
              "data record, results/data_flow.json.")
    asked = [public] + [put(x) for x in RAW["data"]["asked"] if not (x.startswith("In each map") or "LSTM" in x)] + data_asked(F)
    fr = F["first_row"]
    caption = (f"Top: the {n} data products by use. Below: stages 1 to {fr} of the {F['n_stages']}, with counts read from the files; the whole figure is figures/data_flow.png.")
    cut_off = f"stages {fr + 1} to {F['n_stages']}" + (f" and the extra inputs that join at stage {F['join_stage']}" if F["join_stage"] else "")
    return dict(
        key="data", layout="stack", owner="R2", owners=["R2"], seconds=150, title=f"Where the data come from: {w(n)} public products, one pipeline", image=img, crop=DATA_CROP,
        panel=f"the data products and stages 1 to {fr} of the flow, from the {F['first_title']} to the {F['last_title']} ({cut_off} are cut off)",
        caption=caption, caption_whole=f"The whole figure: the {n} data products by use, and the {F['n_stages']} stages with their counts, as read from the files.", fig_w=DATA_FIG_W,
        body_pt=DATA_BODY_PT, bullets=bullets, notes=[x for x in says if x], handover="That is the data. Now the picture of what we found.", asked=asked,
        letter_pt=F["letter_pt"], design_w=F["design_w"],
        says=f"The {n} public products: what they are, how obtained ({gb:.1f} GB read), the flow ({count(F['all_camels'])} to {F['chosen']} basins, the split), how computed, imagery",
        figure_words=f"the flow diagram of the data (results/data_flow.json, figures/data_flow_talk.png), cut after the first row of stages: the product boxes (blue: read by every model; "
                     f"green: extra input of some experiments; grey: a check or a map) with grid, time step and period, then stages 1 to {fr}, from the {F['first_title']} to the {F['last_title']}, "
                     f"with their counts; AORC (an analysis made after the fact), GEFSv12 (a weather forecast) and Stage IV (a real-time product, used only as a check) are the products the next "
                     f"slides use; the whole figure, with the later stages, is figures/data_flow.png, and the 3-page report lists all {F['n_stages']} stages.")


def s_key(R, T, RAW, put):
    """Slide 3: the key figure, with the finding as its title and the caption of results/key_figure.json (GNSS member 1, 150 s). It comes before the evidence, so its notes say what the
    NSE and an onset are."""
    k = RAW["key"]
    N, K, first = BS._need(R), BS._key(R), LEADS[0]
    NW = BS._new(R)
    on, sp = [pct(v) for v in K["on"]], K["spread"]
    hind = S0.hindsight_clause(NW["HS"])
    drawn = sorted(d for d, row in K["far"].items() if "wq" in row and "wqaF_sub" in row)
    far = {d: K["far"][d] for d in drawn}
    crit = BS._criteria(R)
    vo = set(NW["AO"]["verdict_onset"].values()) | set(NW["FC"]["verdict_onset"].values()) if NW["AO"] and NW["FC"] else set()
    bullets = [b for b in k["bullets"] if not b.startswith("Claude Code wrote")]
    nse = "1 is a perfect forecast, 0 is no better than always forecasting the average flow"
    says = [f"This is the picture of our finding: each row gives the LSTM more information.",
            f"Our score is the NSE: {nse}.",
            RAW["simple"]["notes"][2],                                        # 'For warnings the honest test is the onset: a high-flow day forecast while the river was still below the high-flow level.'
            f"Large marks are the LSTM. Small marks are the other kinds of model: ridge regression, random forest and a Transformer.",
            f"{BS._cap(LEAD_WORD[first])}, on the same inputs, {BS._word(sp['kinds'])} kinds of model lie within {n2(sp['nse'])} NSE and {100 * sp['onsets']:.0f} points of onsets warned of each other.",
            f"Now give the LSTM more information. With the rain up to the evening of the issue day, the onsets warned go from {on[0]} to {on[1]}.",
            f"With the rain up to midnight from AORC, {AORC_WORDING}, they reach {pct(K['aorc'])}, {hind}.",
            f"The two diamonds, {pct(K['real'][0])} and {pct(K['real'][1])}, put real-time radar in AORC&rsquo;s place: a check on the test years, not a service." if K["real"] else "",
            f"The last row is the perfect weather forecast, {UPPER}: {on[2]}. Even that stays below the {pct(crit[0])} that GO asks for." if K["on"][2] < crit[0] else
            f"The last row is the perfect weather forecast, {UPPER}: {on[2]}.",
            f"The lower block is {BS._leads(drawn)}. With the rain table alone the LSTM warns of {BS._and(pct(far[d]['wq'][0]) for d in drawn)} of the onsets." if drawn else "",
            f"Add the rain up to midnight and it warns of {BS._and(pct(far[d]['wqa'][0]) for d in drawn)}. Add a real forecast, one GEFS member, and it warns of "
            f"{BS._and(pct(far[d]['wqaF_sub'][0]) for d in drawn)}." if drawn else "",
            "The right-hand panels are the median NSE; it climbs the same steps.",
            f"So {S0.SUPPORT[0].lower() + S0.SUPPORT[1:]}." if NW and BS._answer(R)["support"] else "",
            f"{S0.FINDING}." if BS._answer(R)["limit"] else f"What we learned: milestone 2 of our plan is {N['m2']}.",
            f"For onsets, the warnings that matter, the verdict is still {next(iter(vo))}." if len(vo) == 1 else ""]
    asked = [put(x) for x in k["asked"] if not x.startswith("The headline of the figure")]          # that headline (results/key_figure.json) gives 51% and 48% without their wordings; the items below give both
    asked.append("Where the figure says mixed, the gain in NSE is not clear by our rule: rain up to the evening is "
                 f"higher in only {K['mixed'][0]} of {K['mixed'][1]} basins, and with the GEFS forecast the gain is mixed at {BS._leads(drawn)} (section 4.2 of the report). The evidence for each "
                 f"rung is on slide {AT['more']} and in the full deck." if K["mixed"] and drawn else "")
    return dict(
        key="key", layout="wide", owner="G1", owners=["G1"], seconds=150, image=k["image"], tx_w=k.get("tx_w", 4.7), title=k["title"], caption=k["caption"], bullets=bullets,
        notes=[x for x in says if x], handover="That is the finding. Now the evidence: first, what is the number to beat?", asked=[x for x in asked if x],
        figure_words=put(k["figure_words"]))


def s_more(R, T, RAW, put):
    """Slide 4 (GNSS member 2, 150 s, one part): the number to beat and the first finding, then what more information did. The slide has two halves to look at (a small table and a few
    lines each: on the left the number to beat; on the right the whole ladder of what more information did in one table, and the more-rivers result), but one speaker, one time, one
    list of sentences to say and one hand-over. Every number is read through the shared readers of the report (aorc_facts, stage4_facts, hindsight_facts, forecast_facts, rivers_facts)
    and the full deck's own (_all_basins, _transformer)."""
    N = BS._new(R)
    AO, S4, HS, FC, RV, W = N["AO"], N["S4"], N["HS"], N["FC"], N["RV"], N["W"]
    if not (AO and S4 and HS and FC and RV):
        raise SystemExit("build_slides_short: results/aorc_rain_to_midnight.json, stage4_realtime.json, aorc_hindsight.json, forecast_input.json or more_basins.json is "
                         "missing or not in the expected form: the fourth slide cannot be built")
    A, cmp_, cls, m2 = BS._all_basins(R)
    B = R["baselines"]
    M = B["models"]
    S = {(r["model"], r["lead"]): r for r in B["summary_table"]}
    best = {k: B["best_by_validation"][lk(k)]["without_weather_forecast"] for k in LEADS}
    simple = [m for m in M if not M[m]["uses_weather_forecast"]]
    top = {k: max(S[m, k]["onset_pod"] for m in simple) for k in LEADS}               # the simple model that warns of most onsets at each lead time
    Ls = A["sets"]["wq"]["test"]["summary"]
    first, rest = LEADS[0], tuple(LEADS[1:])
    k2, k3 = rest
    one, n_b = LEAD_WORD[first], len(A["basins"])
    days = lambda k: LEAD_WORD[k].replace(" ahead", "")                               # '1 day', '2 days'
    bare = lambda v: pct(v).rstrip("%")                                               # 0.48 -> '48': a share whose % sign stands once, after the last of several
    hind = S0.hindsight_clause(HS)
    b, w = AO["base"][first], AO["with_"][first]
    s4_on, s4_off = S4["stage4"] / S4["n"], S4["no_last"] / S4["n"]
    as_much = min(AO["gain"]["ridge"], AO["gain"]["forest"]) >= S0.AS_MUCH * AO["gain"]["lstm"]
    nse_w = S0.nse_forecast_words(FC)
    vs = set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values())
    verd = next(iter(vs)) if len(vs) == 1 else BS._and(sorted(vs))
    still = "still " if vs == {"NOT YET"} else ""
    pcts = lambda d: BS._and(f"{100 * d[k]:.0f}" for k in LEADS) + "%"                 # '41, 14 and 8%': a share for 1, 2 and 3 days ahead, as it is said
    n12, n11, n50 = RV["n_train"]["lstm_w"], RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
    rv_read = S0.reading_word(RV["nse_reading"][first])
    up = RV["unseen"]
    sat = R["forecast_input"]["split_satellite_era"]
    d9 = W["daymet"] if W else "09:00"
    wins = BS._and(cmp_("wq", k)["n_basins_lstm_higher"] for k in LEADS)
    tf = BS._transformer(R)
    V = R["flood_analysis"]["verdict"]["services"]["lstm_wq"]
    v_all, v_on = V[lk(first)]["all"]["pooled"], {V[lk(k)]["onset"]["pooled"] for k in LEADS}
    onsets = f"[[{next(iter(v_on))}]] at every lead time" if len(v_on) == 1 else BS._slash(f"[[{V[lk(k)]['onset']['pooled']}]]" for k in LEADS)
    gauge = m2.split(";")[0]                                           # 'not met with the river gauge (no clear difference)': the status of milestone 2, as the full deck words it
    tf_line = ("" if tf is None else
               (f" A Transformer shows {NO_CLEAR}" + (" either" if NO_CLEAR in gauge else "") + f", at {tf['times']} the training time." if tf["acc"] == NO_CLEAR else
                f" A Transformer: {tf['acc']}, at {tf['times']} the training time."))
    al, ve = RAW["all"], RAW["verdict"]
    tr_say = al["notes"][4].split(" What does change the forecast?")[0] if len(al["notes"]) > 4 else ""
    nse_best = BS._and(n2(best[k]["test"]["nse_median"]) for k in LEADS)
    nse_lstm = BS._and(n2(cmp_("wq", k)["lstm_nse_median"]) for k in LEADS)
    # ---- what is said: the number to beat and the first finding, then what more information did
    said = ["The best simple model, chosen on the validation years, sets the number to beat.",
            f"Its median NSE is {nse_best} at {BS._leads(LEADS)}.",
            f"One LSTM for all {n_b} rivers scores {nse_lstm}.",
            al["notes"][2],                                              # 'The LSTM is higher in 5, 5 and 9 of 12 basins: no clear difference. With the river gauge, milestone 2 of our plan is not met.'
            tr_say,                                                      # 'A Transformer shows no clear difference, and takes roughly 10 to 15 times as long.'
            ve["notes"][1],                                              # 'With the river gauge, 1 day ahead, the LSTM warns of 62% of the high-flow days: CAUTION.'
            ve["notes"][3].replace(", the warnings that matter,", ","),  # 'For onsets, the verdict is NOT YET at 1, 2 and 3 days ahead: 1 day ahead, 81% of the onsets get no warning.'
            "So what does change the forecast?",
            f"With the rain table alone, the onsets warned are {BS._and(pct(FC['base'][k]) for k in LEADS)} at {BS._leads(LEADS)}.",
            f"The rain up to midnight, from AORC, {AORC_WORDING}, lifts the first to {pct(w)}, {hind}.",
            f"Stage IV radar in its place gives {pct(s4_on)}, {pct(s4_off)} without its last hour. That is {REALTIME_WORDING}.",
            f"A real forecast, one GEFS member, lifts {BS._leads(rest)} to {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])}.",
            f"{BS._cap(nse_w)}." + (f" {BS._cap(one)}, for the warnings, it is {MOVED_POINT}." if FC["moved"] else
                                     f" {BS._cap(one)} it warns of {signed(FC['gain'][first], 0)} points of onsets ({S0.reading_word(FC['gain_reading'][first])})."),
            f"Retrained on its own, shorter split, the onsets go from {pcts(FC['s_rain'])} to {pcts(FC['s_fc'])}: comparable only inside that split.",
            (f"The simple models gain as many onsets: {INFO_NOT_NET}." if as_much else ""),
            f"With more rivers in training, never-seen rivers go from {n2(up['lobo_w_static'][first])} to {n2(up['unseen50_w_static'][first])} NSE.",
            f"Onsets are {still}{verd}."]
    # ---- the one agreed cut for a late talk: the whole slide, every number and every caveat of what is kept
    cut = (f"The best simple model, chosen on the validation years, sets the number to beat: a median NSE of {n2(best[first]['test']['nse_median'])} {one}. The LSTM scores "
           f"{n2(cmp_('wq', first)['lstm_nse_median'])}; milestone 2 of our plan is {gauge}. For onsets the verdict is {BS._and(sorted(v_on))}. "
           f"AORC, {AORC_WORDING}, takes the onsets warned {one} from {pct(b)} to {pct(w)}, {hind}. Stage IV radar in its place gives {pct(s4_on)}: {REALTIME_WORDING}. "
           f"A real forecast lifts {BS._leads(rest)} from {pct(FC['base'][k2])} and {pct(FC['base'][k3])} to {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])}. Onsets are {still}{verd}.")
    asked = ([put(x) for k in ("simple", "all", "verdict", "cost") for x in RAW[k]["asked"]]
             + [f"The control of AORC, the same rain over the table&rsquo;s own window, warns of {pct(AO['to09'])}: the hours after {d9} are worth {n2(AO['hours'], 1)} points."]
             + [put(x) for x in RAW["rain"]["asked"]] + [put(RAW["timing"]["asked"][0])] + [put(x) for x in RAW["forecast"]["asked"]]
             + [put(x) for x in RAW["rivers"]["asked"]] + [put(x) for x in RAW["lobo"]["asked"]])
    if not any(REALTIME_WORDING in x for x in asked):                           # the full wording of the Stage IV check stands in the notes (the sentences to say have the short form beside the number)
        asked.insert(0, f"What the {pct(s4_on)} is: {REALTIME_WORDING}.")
    # ---- what is on the slide: two halves
    rows_l = [[f"Median NSE, {days(k)}", n2(best[k]["test"]["nse_median"]), n2(cmp_("wq", k)["lstm_nse_median"])] for k in LEADS]
    rows_l.append([f"Onsets warned, {days(first)}", f"at most {pct(top[first])}", pct(Ls[lk(first)]["onset"]["pod"])])
    star = "own split*"
    rows_r = [["the rain table alone"] + [pct(FC["base"][k]) for k in LEADS],
              [f"+ AORC rain to midnight, {AORC_WORDING} ({hind})"] + [f"<b>{pct(FC['rain'][k])}</b>" if k == first else pct(FC["rain"][k]) for k in LEADS],
              ["Stage IV radar, a real-time product, in AORC&rsquo;s place: a check on the test years, not a service (whole issue day / without its last hour)",
               f"<b>{bare(s4_on)}</b> / {pct(s4_off)}", "&ndash;", "&ndash;"],
              [f"+ GEFS forecast beside it (one member; its rain correlates {_slash(n2(FC['corr'][k]) for k in LEADS)} with AORC)"]
              + [f"<b>{pct(FC['fc'][k])}</b>" if k in rest else pct(FC["fc"][k]) for k in LEADS],
              [f"{star}: rain to midnight / + forecast"] + [f"{bare(FC['s_rain'][k])} / <b>{pct(FC['s_fc'][k])}</b>" for k in LEADS]]
    note_r = f"* another split (train {BS._years(sat['train'])}, test {BS._years(sat['test'])}): comparable only inside it."
    left = dict(
        head="<b>The number to beat</b>", part_title="the number to beat",
        table=dict(header=["River gauge, test years", "Best simple model", "LSTM"], widths=[0.44, 0.31, 0.25], rows=rows_l, note=""),
        bullets=[f"Simple models: {Q}, ridge regression, random forest; the best on the validation years is the number to beat.",
                 f"The LSTM is higher in <b>{wins} of {n_b}</b> basins; milestone 2 of our plan is <b>{gauge}</b>.{tf_line}",
                 f"Warnings: all high-flow days [[{v_all}]]; onsets, the warnings that matter, {onsets}."],
        detail_keys=["simple", "all", "verdict", "scores", "cost"])
    right = dict(
        head="<b>What more information did</b>", part_title="what more information did",
        table=dict(header=["Onsets warned, with the river gauge"] + [days(k) for k in LEADS], widths=[0.58, 0.14, 0.14, 0.14], rows=rows_r, note=note_r),
        bullets=[x for x in (f"{BS._cap(nse_w)}; {one}, for the warnings: {MOVED_POINT}.",
                             (f"The simple models gain as many onsets: {INFO_NOT_NET}." if as_much else ""),
                             f"More rivers ({n50} instead of {n11} in training): never-seen rivers <b>{n2(up['lobo_w_static'][first])} &rarr; {n2(up['unseen50_w_static'][first])}</b> NSE; "
                             f"the same {n12} rivers, {rv_read} {one}. Onsets [[{verd}]].") if x],
        detail_keys=["rain", "forecast", "rivers", "lobo", "fail"])
    return dict(
        key="more", layout="split", owner="G2", owners=["G2"], seconds=150, title="The number to beat, and what more information did", halves=[left, right], widths=SPLIT_WIDTHS, bullets=[],
        notes=[x for x in said if x], handover="So what did it take, and what is missing?", asked=[x for x in asked if x], if_late=cut,
        says=f"The number to beat: best simple model {nse_best} against the LSTM {nse_lstm} (median NSE), {NO_CLEAR}. Then onsets warned {one}: {pct(b)} (rain table alone), {pct(w)} with AORC, "
             f"{AORC_WORDING} ({hind}), {pct(s4_on)} with Stage IV (a check on the test years, not a service)",
        figure_words=f"left half: median NSE of the {n_b} basins and the onsets warned {one}, with the river gauge, test years, from results/baselines.json and results/lstm_all.json; right half: "
                     f"onsets warned {_slash(str(k) for k in LEADS)} days ahead, with the river gauge, from results/aorc_rain_to_midnight.json, results/stage4_realtime.json and "
                     f"results/forecast_input.json; * the model retrained on its own, shorter split; the figures behind the rows are in the report ({BS.RAIN_REF}, {BS.FORECAST_REF}) and on "
                     f"the full deck's slides about the number to beat, the rain up to the issue time and the real weather forecast.")


def s_end(R, T, RAW, put):
    """Slide 5: the finding holds in both directions; what it took, what is missing, the limits, the work in numbers, who did what; thank you (the team lead, 80 s)."""
    N, NW = BS._need(R), BS._new(R)
    ans = N["ans"]
    AO, FC, RV = NW["AO"], NW["FC"], NW["RV"]
    if not (AO and FC and RV and ans["limit"]):
        raise SystemExit("build_slides_short: the closing slide needs the AORC, forecast and more-rivers results and the finding to hold; one of them is missing")
    Wn = work_in_numbers(R, T)
    first = LEADS[0]
    n_tr = BS._word(len(R["aorc_rain_to_midnight"]["seeds"]))
    A = R["lstm_all"]
    n_basins, n_test = len(A["basins"]), int(A["split"]["test"][1][:4]) - int(A["split"]["test"][0][:4])
    took = (f"onsets warned {LEAD_WORD[first]} <b>{pct(AO['base'][first])} &rarr; {pct(AO['with_'][first])}</b> ({S0.hindsight_clause(NW['HS'])}) with the rain up to midnight from AORC, "
            f"{AORC_WORDING}; <b>{pct(NW['S4']['stage4'] / NW['S4']['n'])}</b> with the real-time radar product in its place (a check on the test years, not a service); {BS._leads(LEADS[1:])}, <b>{pct(FC['base'][LEADS[1]])} / "
            f"{pct(FC['base'][LEADS[2]])} &rarr; {pct(FC['fc'][LEADS[1]])} / {pct(FC['fc'][LEADS[2]])}</b> with a weather forecast")
    missing = N["table"]["rows"][1][1].replace(", the real-time product", "")
    limits = f"{n_basins} basins, {n_test} test years, {n_tr} trainings of each model, one laptop"
    gbs = ", ".join(f"{k} {_num(v, 1 if v < 10 else 0)}" for k, v in (Wn["gb_parts"] or {}).items())
    work_bits = ([f"<b>{count(Wn['trainings'])}</b> trainings in {n2(Wn['hours'], 1)} hours on a two-core laptop"] if Wn["trainings"] else []) \
        + ([f"<b>{_num(Wn['gb'], 1)} GB</b> of open data read ({gbs})"] if Wn["gb"] else []) \
        + ([f"<b>{count(Wn['rounds'])}</b> review rounds by independent AI agents, <b>{count(Wn['checks'])}</b> checks" + (", none failed" if Wn["clean"] else "")] if Wn["rounds"] else []) \
        + (["money spent: <b>none</b>"] if Wn["money"] else [])
    work = "; ".join(work_bits) + "."
    who = (f"<b>Who did what:</b> Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction. {BS._cap(REVIEW_WORDING)}. {BS._team_check(R)[1]}")
    away = N["away"].split(": ", 1)[1] if ": " in N["away"] else N["away"]            # 'Taking information away lowers the skill: without the river gauge ...' -> what follows the colon
    rows = [["<b>Information taken away</b>", away], ["<b>Information given: two inputs</b>", took],
            ["<b>Still missing; limits</b>", f"{missing}. <b>Limits:</b> {limits}"], ["<b>The work in numbers</b>", work]]
    says = [N["say"][0], N["say"][1], N["say"][2],
            f"Limits: {n_basins} basins, {n_test} test years, {n_tr} trainings of each model, one laptop.",
            ("The work in numbers: " + ", ".join(
                ([f"{count(Wn['trainings'])} trainings"] if Wn["trainings"] else []) + ([f"{_num(Wn['gb'], 1)} GB of open data"] if Wn["gb"] else [])
                + ([f"{BS._word(Wn['rounds'])} review rounds with {count(Wn['checks'])} checks"] if Wn["rounds"] else []) + (["and no money spent"] if Wn["money"] else [])) + ".")
            if work_bits else "",
            f"Claude Code wrote the code and ran the experiments under the team lead&rsquo;s direction. Independent AI reviewers checked the engine and every experiment; a final critic agent "
            f"verified the last checks.",
            BS._check_say(R), "Thank you. We welcome your questions."]
    per = Wn.get("per_record") or {}
    single = next((n for f, _, _, n in (Wn.get("runs") or []) if f == "lstm_single"), None)
    trainings_note = (f"The data record, results/data_flow.json, lists {count(Wn['trainings_recorded'])} trainings; section 7 also counts the {count(single)} of the one-river LSTM of section 3, "
                      f"which the data record does not list: {count(Wn['trainings'])} in all. " if single and Wn["trainings"] == Wn["trainings_recorded"] + single else "")
    itemized = (f"The work in numbers, itemised. Trainings, as section 7 of the report adds them from the results files: "
                + "; ".join(f"{label.replace('this comparison', 'the Transformer comparison of section 7').replace('the ', '', 1) if label.startswith('the ') else label.replace('this comparison', 'the Transformer comparison of section 7')}: {n}" for _, label, _, n in Wn["runs"] if n)
                + f" ({count(Wn['trainings'])} in {n2(Wn['hours'], 1)} hours, not counting the waiting for the processor, test runs and repeated runs). Data read: "
                + (", ".join(f"{k} {_num(v, 1 if v < 10 else 0)} GB" for k, v in Wn["gb_parts"].items()) if Wn["gb_parts"] else "not counted")
                + f" (results/data_flow.json, all_information.bytes_read: {_num(Wn['gb_recorded'])} GB of the {_num(Wn['gb'], 1)} GB are transfers recorded in a results file or a log, for AORC, GEFS and "
                  "Stage IV; for the other products the transfer size was not recorded and the size of the stored files is counted; the Stage IV figure includes a batch fetched again after a stop, "
                  "and the GEFS figure is the size of the byte ranges read, about " + (f"{Wn['gefs_mb_a_day']:.0f}" if Wn.get("gefs_mb_a_day") else "?") + " MB a day, not of the files). "
                + trainings_note + "Review rounds and checks listed, "
                  "from every results/review_*.json: " + "; ".join(f"{k.replace('review_', '')}: {r} round{'s' if r != 1 else ''}, {c} checks" for k, (r, c) in per.items())
                + f" - together {Wn['rounds']} rounds and {count(Wn['checks'])} checks: the four reviews of the experiments {sum(r for k, (r, c) in per.items() if k in S0.NEW_REVIEWS)} rounds and "
                  f"{sum(c for k, (r, c) in per.items() if k in S0.NEW_REVIEWS)} checks, the final critic's review of the documents {sum(r for k, (r, c) in per.items() if k not in S0.NEW_REVIEWS)} rounds and "
                  f"{sum(c for k, (r, c) in per.items() if k not in S0.NEW_REVIEWS)} checks"
                + (f"; the final critic made {count(Wn['compared'])} comparisons of the documents' numbers and their recomputations against the results files" if Wn["compared"] else "")
                + ". These are AI reviewers, not team members. " + ("Money: results/data_access.json says that nothing was bought and no account was created; the downloads were open "
                                                                    "archives." if Wn["money"] else "")) if (Wn["runs"] and per) else ""
    asked = [x for x in [f"{BS._review_sentence(R)} {BS._team_check(R)[1]} &lsquo;We&rsquo; means the project, never a check made by a team member.", itemized, put(N["upper"])]
             + [put(x) for x in N["need_asked"][1:]] if x]
    return dict(
        key="end", layout="table", owner="A", owners=["A"], seconds=80, title=N["title"].split(":")[0],
        table=dict(header=["Conclusions and limits", "What our results say"], widths=[0.24, 0.76], rows=[[put(c) for c in r] for r in rows]),
        bullets=["<b>Thank you.</b> We welcome your questions; the team lead leads them."], small_print=[who],
        says=("What it took, what is missing, the limits; the work in numbers ("
              + ", ".join(([f"{count(Wn['trainings'])} trainings"] if Wn["trainings"] else []) + ([f"{_num(Wn['gb'], 1)} GB"] if Wn["gb"] else [])
                          + ([f"{count(Wn['checks'])} checks"] if Wn["rounds"] else []) + (["no money spent"] if Wn["money"] else []))
              + "); who did what"),
        notes=[x for x in says if x], handover="", asked=asked)


# ---- the deck -----------------------------------------------------------------------------------------------------------------------
def _hindsight(x, num, clause):
    """A text of the notes (a sentence reused from the full deck or from a results file) with the wording of the review beside the share that AORC gives: where `num` ('51%') stands without
    the word 'hindsight' near it, the stored wording (S0.hindsight_clause: 'hindsight at most about 3 points') is put after it, in brackets (after a comma where the share stands
    inside a bracket already). A text that has the word near the share is not changed."""
    if not isinstance(x, str) or num not in x:
        return x
    out, last = [], 0
    for m in re.finditer(re.escape(num) + r"(?!\d)", x):
        if "hindsight" in x[max(0, m.start() - 60): m.end() + 90]:
            continue
        out.append(x[last:m.end()] + (f", {clause}" if x[m.end(): m.end() + 1] == ")" else f" ({clause})"))
        last = m.end()
    return "".join(out) + x[last:]


def build_deck(R, T):
    """The five slides, and the full deck's slides (their keys and numbers) that stand behind each. The full deck's own specs are built once (RAW) and their sentences reused;
    a reference such as [slide:rain] in one of them is written as 'full-deck slide 10'."""
    full = BS.slides(R, T)
    at = {s["key"]: (f"full-deck backup slide B{s['backup']}" if s.get("backup") else f"full-deck slide {i}") for i, s in enumerate(full, 1)}
    put = lambda x: BS.SLIDE_REF.sub(lambda m: at.get(m.group(1), "another slide of the full deck"), x) if isinstance(x, str) else x
    RAW = {k: f(R, T) for k, f in (("cover", BS.s_cover), ("data", BS.s_data), ("fair", BS.s_fair), ("simple", BS.s_simple), ("all", BS.s_all), ("verdict", BS.s_verdict),
                                   ("key", BS.s_key), ("timing", BS.s_timing), ("rain", BS.s_rain), ("forecast", BS.s_forecast), ("need", BS.s_need))}
    for k, f in (("rivers", BS.b_rivers), ("lobo", BS.b_lobo), ("cost", BS.b_cost)):
        RAW[k] = f(R, T) or dict(asked=[])
    make = dict(cover=s_cover, data=s_data, key=s_key, more=s_more, end=s_end)
    deck = [make[k](R, T, RAW, put) for k in ORDER]
    NW = BS._new(R)
    w51, clause = pct(NW["AO"]["with_"][LEADS[0]]), S0.hindsight_clause(NW["HS"])                   # the share AORC gives 1 day ahead, and the words that belong beside it
    for s in deck:
        for seg in segments(s):
            for k in ("notes", "asked", "bullets", "small"):
                if seg.get(k):
                    seg[k] = [_hindsight(put(x), w51, clause) for x in seg[k]]
        for h in s.get("halves", []):                                       # the lines under each half's table are text of the slide too
            h["bullets"] = [_hindsight(put(x), w51, clause) for x in h["bullets"]]
        for k in ("title", "caption", "figure_words", "handover"):
            if s.get(k):
                s[k] = put(s[k])
    where = {s["key"]: ((f"B{s['backup']}", i) if s.get("backup") else (str(i), i)) for i, s in enumerate(full, 1)}
    behind = lambda keys: [dict(key=x, slide=where[x][1], backup=full[where[x][1] - 1].get("backup"), label=LABEL.get(x, x), title=_plain(full[where[x][1] - 1]["title"])) for x in keys if x in where]
    keys_of = _detail_keys()
    for s in deck:
        if s["layout"] == "split":                                          # the slide with two halves: each half has its own list; the slide's list is both, without repeats
            for h in s["halves"]:
                h["detail"] = behind(h.pop("detail_keys"))
            s["detail"] = [d for j, h in enumerate(s["halves"]) for d in h["detail"] if d["key"] not in {y["key"] for g in s["halves"][:j] for y in g["detail"]}]
        else:
            s["detail"] = behind(keys_of[s["key"]])
    assert len(deck) == SLIDES
    return deck, full


# ---- geometry of the split layout -------------------------------------------------------------------------------------------
def split_geometry(s):
    """Where things go on the slide with two halves, in inches: the title across the top and, in each half, its head, its table, the note under the table and its lines, one under the
    other; a hairline in the gap. The halves have the widths of s['widths'] (shares of the width left by the gap). The HTML uses the same boxes and lets the text flow."""
    full = BS.W - 2 * BS.MX
    shares_ = s.get("widths", (0.5, 0.5))
    widths = [(full - SPLIT_GAP) * f / sum(shares_) for f in shares_]
    xs = [BS.MX, BS.MX + widths[0] + SPLIT_GAP]
    g = dict(title_w=full, title_y=BS.TOP, mid=BS.MX + widths[0] + SPLIT_GAP / 2)
    g["title_h"] = BS._lines(s["title"], full, BS.TITLE_PT, bold=True) * BS.TITLE_PT * BS.TITLE_LH / 72
    g["body_y"] = BS.TOP + g["title_h"] + BS.TITLE_GAP
    halves = []
    for j, h in enumerate(s["halves"]):
        x, colw = xs[j], widths[j]
        tpt, bpt = h.get("table_pt", SPLIT_TABLE_PT), h.get("body_pt", SPLIT_BODY_PT)
        head_h = BS._lines(h["head"], colw, SPLIT_HEAD_PT) * SPLIT_HEAD_PT * SPLIT_HEAD_LH / 72
        tb = h["table"]
        pad = 2 * SPLIT_PAD[0] / 72
        col_w = [colw * f for f in tb["widths"]]
        row_h = [(max(BS._lines(c, w - pad, tpt, bold=(i == 0)) for c, w in zip(row, col_w)) * tpt * BS.TABLE_LH + 2 * SPLIT_PAD[1] + 2) / 72
                 for i, row in enumerate([tb["header"]] + tb["rows"])]
        ty = g["body_y"] + head_h + SPLIT_HEAD_GAP
        note_h = BS._lines(tb["note"], colw, SPLIT_NOTE_PT) * SPLIT_NOTE_PT * BS.CAP_LH / 72 if tb.get("note") else 0.0
        ny = ty + sum(row_h) + SPLIT_NOTE_GAP
        by = ny + note_h + SPLIT_BULLETS_GAP
        bh = sum(BS._lines(b, colw, bpt, indent=BS.BULLET_IN) * bpt * BS.BODY_LH + BS.BULLET_AFTER for b in h["bullets"] if b) / 72
        halves.append(dict(x=x, w=colw, head_h=head_h, ty=ty, col_w=col_w, row_h=row_h, ny=ny, note_h=note_h, by=by, bh=bh, tpt=tpt, bpt=bpt))
    g["bullets_y"] = min(h["by"] for h in halves)
    g["end"] = max(h["by"] + h["bh"] for h in halves)                    # the lowest end of any half is what counts for the overflow check
    g["bullets_h"] = g["end"] - g["bullets_y"]
    g["halves"] = halves
    return g


def stack_geometry(s):
    """The dataset slide: the title, the flow diagram across the slide (as wide as the lines under it leave room for, at most s['fig_w']) with its label, and the lines (s['body_pt'])
    full width under it. The figure is made as large as the height allows: the more the lines say, the smaller the figure."""
    full = BS.W - 2 * BS.MX
    g = dict(title_w=full, title_y=BS.TOP, tw=full)
    g["title_h"] = BS._lines(s["title"], full, BS.TITLE_PT, bold=True) * BS.TITLE_PT * BS.TITLE_LH / 72
    g["body_y"] = BS.TOP + g["title_h"] + BS.TITLE_GAP
    pt = s.get("body_pt", BS.BODY_PT)
    g["bullets_h"] = sum(BS._lines(b, full, pt, indent=BS.BULLET_IN) * pt * BS.BODY_LH + BS.BULLET_AFTER for b in s["bullets"] if b) / 72
    fw, fh, ch = BS._place(s, min(full, s.get("fig_w", full)), BS.BOTTOM - g["body_y"] - g["bullets_h"] - STACK_GAP)
    g.update(fw=fw, fh=fh, cap_h=ch, fx=BS.MX + (full - fw) / 2, fy=g["body_y"])
    g["bullets_y"] = g["fy"] + fh + ch + STACK_GAP
    g["end"] = g["bullets_y"] + g["bullets_h"]
    return g


def geometry(s):
    """The full deck's geometry, or that of the slide with two halves; for a slide with small print also where the small print goes (small_y, small_h) and where the slide's content ends."""
    if s["layout"] == "split":
        return split_geometry(s)
    if s["layout"] == "stack":
        return stack_geometry(s)
    g = BS._geometry(s)
    g["end"] = g["bullets_y"] + g["bullets_h"]
    if s.get("small_print"):
        g["small_y"] = g["end"] + SMALL_PRINT_GAP
        g["small_h"] = sum(BS._lines(x, g["tw"], SMALL_PRINT_PT) * SMALL_PRINT_PT * SMALL_PRINT_LH / 72 + 0.04 for x in s["small_print"])
        g["end"] = g["small_y"] + g["small_h"]
    return g


def overflow(deck):
    """Slides whose text is estimated to run past the lowest content edge: [(slide number, inches too long)]."""
    out = []
    for i, s in enumerate(deck, 1):
        g = geometry(s)
        over = g.get("end", g["bullets_y"] + g["bullets_h"]) - BS.BOTTOM
        if over > 0:
            out.append((i, round(over, 2)))
    return out


# ---- HTML (and from it the PDF) ----------------------------------------------------------------------------------------------
def split_html(s, i, deck):
    g = geometry(s)
    left, right = BS._foot(s, i, deck)
    parts = [f"<h1>{s['title']}</h1>"]
    for h, gh in zip(s["halves"], g["halves"]):
        tb = h["table"]
        grid = BS.table([BS._html(c) for c in tb["header"]], [[BS._html(c) for c in row] for row in tb["rows"]], widths=[f"{100 * f:.1f}%" for f in tb["widths"]])
        parts.append(f'<p class="hd" style="left:{gh["x"]:.3f}in;top:{g["body_y"]:.3f}in;width:{gh["w"]:.3f}in">{h["head"]}</p>')
        parts.append(f'<div class="tb" style="left:{gh["x"]:.3f}in;top:{gh["ty"]:.3f}in;width:{gh["w"]:.3f}in;font-size:{gh["tpt"]}pt">{grid}</div>')
        if tb.get("note"):
            parts.append(f'<p class="nt" style="left:{gh["x"]:.3f}in;top:{gh["ny"]:.3f}in;width:{gh["w"]:.3f}in">{tb["note"]}</p>')
        parts.append(f'<div class="bl" style="left:{gh["x"]:.3f}in;top:{gh["by"]:.3f}in;width:{gh["w"]:.3f}in;font-size:{gh["bpt"]}pt">{BS._ul(h["bullets"])}</div>')
    parts.append(f'<div class="vr" style="left:{g["mid"]:.3f}in;top:{g["body_y"]:.3f}in;height:{BS.BOTTOM - g["body_y"]:.3f}in"></div>')
    return f'<section class="slide split">{"".join(parts)}<div class="ft"><span>{left}</span><span>{right}</span></div></section>'


def stack_html(s, i, deck):
    """The dataset slide: the title, the figure (the cut of the flow diagram) with its label, the lines under it, the footer."""
    g = geometry(s)
    left, right = BS._foot(s, i, deck)
    return (f'<section class="slide stack"><h1>{s["title"]}</h1>'
            f'<div class="fg" style="left:{g["fx"]:.3f}in;top:{g["fy"]:.3f}in;width:{g["fw"]:.3f}in">{BS._figure_html(s, g)}</div>'
            f'<div class="bl" style="left:{BS.MX}in;top:{g["bullets_y"]:.3f}in;width:{g["tw"]:.3f}in">{BS._ul(s["bullets"])}</div>'
            f'<div class="ft"><span>{left}</span><span>{right}</span></div></section>')


def slide_html(s, i, deck):
    if s["layout"] == "split":
        return split_html(s, i, deck)
    if s["layout"] == "stack":
        return stack_html(s, i, deck)
    out = BS.slide_html(s, i, deck)
    if s.get("small_print"):                                            # after the lines of the slide, inside its text block (which flows)
        small = "".join(f'<p class="small5">{x}</p>' for x in s["small_print"])
        out = out.replace('</div><div class="ft">', f'{small}</div><div class="ft">', 1)
    return out


def deck_html(deck):
    return page(f"{DECK} - five slides", "\n".join(slide_html(s, i, deck) for i, s in enumerate(deck, 1)), BS.CSS_SLIDES + CSS_SPLIT)


# ---- PPTX ---------------------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _table_frame(x, w, pt=SPLIT_TABLE_PT):
    """The full deck's _table draws a table across the slide, at its left margin, in its table font; for a half of slide 4 the margin, the width, the font and the
    padding are set for the length of the call (the four module constants _table reads) and put back after it."""
    saved = BS.MX, BS.W, BS.TABLE_PT, BS.CELL_PAD
    BS.MX, BS.W, BS.TABLE_PT, BS.CELL_PAD = x, 2 * x + w, pt, SPLIT_PAD
    try:
        yield
    finally:
        BS.MX, BS.W, BS.TABLE_PT, BS.CELL_PAD = saved


def split_pptx(slide, s, g):
    """The shapes of the slide with two halves: for each half its head, its table, the note under it and its lines; and the hairline between the halves."""
    for h, gh in zip(s["halves"], g["halves"]):
        BS._textbox(slide, gh["x"], g["body_y"], gh["w"], gh["head_h"], [h["head"]], SPLIT_HEAD_PT, lh=SPLIT_HEAD_LH)
        with _table_frame(gh["x"], gh["w"], gh["tpt"]):
            BS._table(slide, h["table"], dict(body_y=gh["ty"], col_w=gh["col_w"], row_h=gh["row_h"]))
        if h["table"].get("note"):
            BS._textbox(slide, gh["x"], gh["ny"], gh["w"], max(gh["note_h"], 0.2), [h["table"]["note"]], SPLIT_NOTE_PT, color=BS.INK2, lh=BS.CAP_LH)
        BS._textbox(slide, gh["x"], gh["by"], gh["w"], max(gh["bh"], 0.4), [b for b in h["bullets"] if b], gh["bpt"], bullet=True, after=BS.BULLET_AFTER)
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(g["mid"]), Inches(g["body_y"]), Inches(g["mid"]), Inches(BS.BOTTOM))
    line.line.color.rgb, line.line.width = BS._rgb(BS.GRID), Pt(0.75)


def notes_text(s, i, deck):
    """The speaker notes of one slide as plain text, for the notes pane of the PPTX: every segment with its presenter, time, sentences, hand-over and facts for questions."""
    clock, t = lambda sec: f"{sec // 60}:{sec % 60:02d}", timeline(deck)
    lines = []
    for seg_t in [x for x in t if x["slide"] == i]:
        seg = seg_t["spec"]
        who, what = BS.ROLE[seg["owner"]]
        lines += [f"Slide {i}. Presenter: {_plain(who)} ({what}). Time: {seg['seconds']} s ({clock(seg_t['start_s'])} to {clock(seg_t['end_s'])}).", "", "Say:"]
        lines += [f"- {_plain(x)}" for x in seg["notes"] if x]
        if seg.get("handover"):
            lines += ["", "Hand-over: " + _plain(seg["handover"])]
        if any(seg.get("asked", [])):
            lines += ["", "If asked:"] + [f"- {_plain(x)}" for x in seg["asked"] if x]
        if seg.get("figure_words"):
            lines += ["", "Reading the slide: " + _plain(seg["figure_words"])]
        lines += ["", ""]
    return "\n".join(lines).strip()


def deck_pptx(deck, path):
    """The same slides as editable shapes. The loop of the full deck's deck_pptx with two more layouts (the dataset slide and the slide with two halves); every shape comes from the full deck's own helpers."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(BS.W), Inches(BS.H)
    cp = prs.core_properties
    cp.title, cp.subject = DECK, "AI course group project: the five slides of the 10-minute presentation"
    cp.author = "built by code/build_slides_short.py (written by Claude Code, the team's coding assistant, under the team lead's direction)"
    cp.last_modified_by, cp.created, cp.modified = "code/build_slides_short.py", datetime.datetime.now(), datetime.datetime.now()
    layout = prs.slide_layouts[5]                                     # 'Title Only': the title stays a real title for the outline view
    for i, s in enumerate(deck, 1):
        slide = prs.slides.add_slide(layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = BS._rgb(BS.SURFACE)
        g = geometry(s)
        cover = s["layout"] == "cover"
        title = slide.shapes.title
        title.left, title.top, title.width, title.height = Inches(BS.MX), Inches(g["title_y"]), Inches(g["title_w"]), Inches(g["title_h"])
        BS._para(BS._frame(title.text_frame), s["title"], BS.COVER_PT if cover else BS.TITLE_PT, True, bold=True, lh=BS.COVER_LH if cover else BS.TITLE_LH)
        if cover:
            BS._textbox(slide, BS.MX, g["kicker_y"], g["kicker_w"], g["kicker_h"], [s["kicker"]], BS.SMALL_PT, color=BS.INK2, lh=BS.SMALL_LH)
            bar = slide.shapes.add_shape(BS.MSO_SHAPE.RECTANGLE, Inches(BS.MX), Inches(g["rule_y"]), Inches(1.1), Inches(0.05))
            bar.fill.solid()
            bar.fill.fore_color.rgb = BS._rgb(BS.BLUE)
            bar.line.fill.background()
            bar.shadow.inherit = False
            BS._textbox(slide, BS.MX, g["sub_y"], g["tw"], g["sub_h"], s["sub"].split("<br>"), BS.SUB_PT, lh=BS.SUB_LH)
            for x, (y, h) in zip(s["small"], g["small"]):
                BS._textbox(slide, BS.MX, y, g["tw"], h, [x], BS.SMALL_PT, color=BS.INK2, lh=BS.SMALL_LH)
        if s["layout"] == "table":
            BS._table(slide, s["table"], g)
            if s["table"].get("note"):
                BS._textbox(slide, BS.MX, g["note_y"], g["tw"], max(g["note_h"], 0.3), [s["table"]["note"]], BS.SMALL_PT, color=BS.INK2, lh=BS.NOTE_LH)
        if s["layout"] == "split":
            split_pptx(slide, s, g)
        if "fx" in g and s["layout"] != "split":
            BS._picture(slide, s, g)
        bullets = [b for b in s.get("bullets", []) if b]
        if bullets and s["layout"] != "split":
            BS._textbox(slide, BS.MX, g["bullets_y"], g["tw"], max(g["bullets_h"], 0.4), bullets, s.get("body_pt", BS.BODY_PT), bullet=True, after=BS.BULLET_AFTER)
        if s.get("small_print"):
            BS._textbox(slide, BS.MX, g["small_y"], g["tw"], g["small_h"], s["small_print"], SMALL_PRINT_PT, color=BS.INK2, lh=SMALL_PRINT_LH, after=3)
        BS._footer(slide, s, i, deck)
        slide.notes_slide.notes_text_frame.text = notes_text(s, i, deck)
    prs.save(str(path))


def check_pptx(path, deck):
    """The full deck's check (slide count, titles, shapes inside the slide, no picture distorted, notes on every slide) and: two tables on the slide with two halves; the footer and the
    notes of every slide name its presenter."""
    bad = BS.check_pptx(path, deck)
    prs = Presentation(str(path))
    for i, (slide, s) in enumerate(zip(prs.slides, deck), 1):
        if s["layout"] == "split":
            tables = [sh for sh in slide.shapes if sh.has_table]
            if len(tables) != len(s["halves"]):
                bad.append(f"slide {i}: {len(tables)} tables instead of {len(s['halves'])}")
        text = " ".join(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)
        if BS.ROLE[s["owner"]][0] not in text:
            bad.append(f"slide {i}: the footer does not name the presenter")
        if BS.ROLE[s["owner"]][0] not in slide.notes_slide.notes_text_frame.text:
            bad.append(f"slide {i}: the notes do not name the presenter")
    return bad


# ---- the timeline, the totals, the plan -----------------------------------------------------------------------------------------------
def timeline(deck):
    """Every part in the order of the talk: slide number, part (always '': a slide is one part), presenter, seconds, start and end (seconds), and the part itself."""
    out, t = [], 0
    for i, s in enumerate(deck, 1):
        for seg in segments(s):
            out.append(dict(slide=i, part="", owner=seg["owner"], seconds=seg["seconds"], start_s=t, end_s=t + seg["seconds"], spec=seg))
            t += seg["seconds"]
    return out


def totals(deck):
    """{presenter key: seconds in all} for the presenters who speak."""
    tl = timeline(deck)
    return {k: sum(x["seconds"] for x in tl if x["owner"] == k) for k in BS.ROLE if any(x["owner"] == k for x in tl)}


def checkpoints(deck):
    """The clock at the start of every part that another presenter takes over, with the slide it belongs to."""
    tl = timeline(deck)
    return [x for i, x in enumerate(tl) if i and x["owner"] != tl[i - 1]["owner"]]


def late_cut(deck):
    """The one agreed cut for a late talk, if a segment has one (key if_late): the first such segment, at its start plus BS.LATE seconds."""
    tl = timeline(deck)
    for x in tl:
        seg = x["spec"]
        if seg.get("if_late"):
            words = len(_plain(seg["if_late"]).split()) + len(_plain(seg.get("handover", "")).split())
            return dict(x, words=words, saves=5 * int((seg["seconds"] - words / BS.PACE) // 5))
    return None


def slide_plan(deck, full, R):
    """The plan of the talk as data (results/slide_plan_5.json): per slide its presenters, time, figures, parts (each with sentences to say, hand-over, what it says and the
    cut for a late talk), the full deck's slides behind it; the presenters with their seconds; the clock checkpoints; the one agreed cut. Plain text, no markup."""
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    plain = lambda x: _md(x).replace("**", "")
    tl = timeline(deck)
    roles = {d["key"]: d["role"] for d in team.DEFAULT}
    rows = []
    for i, s in enumerate(deck, 1):
        mine = [x for x in tl if x["slide"] == i]
        parts = []
        for x in mine:
            seg = x["spec"]
            parts.append(dict(part=x["part"], presenter=seg["owner"], presenter_label=team.label(seg["owner"]), job=BS.JOB[seg["owner"]], seconds=seg["seconds"], start_s=x["start_s"],
                              end_s=x["end_s"], start=clock(x["start_s"]), end=clock(x["end_s"]), says=plain(seg["says"]) if seg.get("says") else plain(s["title"]),
                              say=[plain(y) for y in seg["notes"] if y], handover=plain(seg["handover"]) if seg.get("handover") else "",
                              if_late=plain(seg["if_late"]) if seg.get("if_late") else "", figure=seg.get("image") or s.get("image"), has_table=bool(seg.get("table") or seg.get("halves")),
                              figure_part=plain(seg["panel"]) if seg.get("panel") and seg.get("crop") else None,
                              full_deck=[dict(key=d["key"], slide=d["slide"], backup=d["backup"], label=d["label"], title=d["title"]) for d in seg.get("detail", s["detail"])]))
        rows.append(dict(slide=i, key=s["key"], kind="talk", title=plain(s["title"]), presenters=[p["presenter"] for p in parts], shared=len(parts) > 1, seconds=s["seconds"],
                         start_s=mine[0]["start_s"], end_s=mine[-1]["end_s"], start=clock(mine[0]["start_s"]), end=clock(mine[-1]["end_s"]), has_table=bool(s.get("table") or s.get("halves")),
                         parts=parts, full_deck=[dict(key=d["key"], slide=d["slide"], backup=d["backup"], label=d["label"], title=d["title"]) for d in s["detail"]]))
    total = sum(x["seconds"] for x in tl)
    tot = totals(deck)
    presenters = [dict(key=k, role=roles[k], label=team.label(k), job=BS.JOB[k], slides=sorted({x["slide"] for x in tl if x["owner"] == k}),
                       parts=[f"{x['slide']}{' ' + x['part'] if x['part'] else ''}" for x in tl if x["owner"] == k], seconds=tot[k]) for k in BS.ROLE if k in tot]
    cps = [dict(slide=x["slide"], part=x["part"], presenter=x["owner"], presenter_label=team.label(x["owner"]), clock=clock(x["start_s"])) for x in checkpoints(deck)]
    cut = late_cut(deck)
    cut = None if cut is None else dict(slide=cut["slide"], part=cut["part"], presenter=cut["owner"], presenter_label=team.label(cut["owner"]), clock_over=clock(cut["start_s"] + BS.LATE),
                                        seconds_over=cut["start_s"] + BS.LATE, say=plain(cut["spec"]["if_late"]), saves_seconds=cut["saves"], then_handover=bool(cut["spec"].get("handover")))
    return dict(written_by="code/build_slides_short.py", built=datetime.date.today().isoformat(), deck="short", total_seconds=total, talk_slides=len(deck), backup_slides=0,
                full_deck_slides=len([s for s in full if not s.get("backup")]), full_deck_backup_slides=len([s for s in full if s.get("backup")]), words_per_second=BS.PACE,
                late_threshold_seconds=BS.LATE, presenters=presenters, slides=rows, clock_checkpoints=cps, late_cut=cut)


# ---- speaker notes as Markdown -----------------------------------------------------------------------------------------------------------
def _shows(seg, s):
    """What a part shows: the figure (and the part of it), a table, two small tables side by side (the slide with two halves), or text."""
    if seg.get("halves"):
        sides = " and ".join(f"{side}: {_md(h['part_title'])}" for side, h in zip(("left", "right"), seg["halves"]))
        return f"**On the slide:** two small tables side by side, each with a few lines under it ({sides})" + (f"; {_md(seg['figure_words'])}" if seg.get("figure_words") else ".")
    return BS._shows(seg)


def before_you_speak(deck, R, plan):
    """The half page a presenter reads before the talk: how numbers and names are said, the clock at every hand-over, the one agreed cut, and every presenter's first words."""
    name = lambda k: _md(BS.ROLE[k][0])
    score = _md(n2(BS._all_basins(R)[1]("wq", LEADS[0])["lstm_nse_median"]))
    aloud = " ".join(BS.DIGIT.get(c, c) for c in score)
    Wd, hour = BS._windows(R), ""
    if Wd and re.fullmatch(r"\d\d:00", Wd["daymet"]) and int(Wd["daymet"][:2]) in BS.NUMBER:
        hour = f" {Wd['daymet']} is said ‘{BS.NUMBER[int(Wd['daymet'][:2])]} o’clock’."
    out = ["## Before you speak", "",
           f"**Saying it aloud.** A score such as {score} is said digit by digit: ‘{aloud}’. NSE and LSTM are spelled out, letter by letter, and so are AORC, GEFS and NLDAS; "
           f"ERA5 is said ‘E R A five’ and Stage IV ‘Stage four’.{hour}", ""]
    cps = plan["clock_checkpoints"]
    if cps:
        out += ["**Clock checkpoints.** " + " ".join(
            f"When slide {c['slide']}{' (' + c['part'] + ')' if c['part'] else ''} starts ({name(c['presenter'])}), the clock should show {c['clock']}." for c in cps), ""]
    cut = plan["late_cut"]
    if cut:
        out += [f"**If the talk runs late.** If the clock shows more than {cut['clock_over']} when slide {cut['slide']}{' (' + cut['part'] + ')' if cut['part'] else ''} starts, "
                f"{name(cut['presenter'])} says only: “{cut['say']}”" + (" - and then the hand-over sentence." if cut["then_handover"] else "")
                + f" That saves about {cut['saves_seconds']} s. No other part is shortened: the other sentences carry the caveats.", ""]
    out += ["**Your first words.** So that each presenter can start without reading:", ""]
    for k in dict.fromkeys(x["owner"] for x in timeline(deck)):
        out.append(f"- {name(k)}")
        out += [f"  - slide {x['slide']}{' (' + x['part'] + ')' if x['part'] else ''}: “{BS._first_sentence(x['spec']['notes'][0])}”" for x in timeline(deck) if x["owner"] == k and x["spec"]["notes"]]
    return out + [""]


def notes_md(deck, full, R, plan, text_sources):
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    name = lambda k: _md(BS.ROLE[k][0])
    tl = timeline(deck)
    total = sum(x["seconds"] for x in tl)
    _, check = BS._team_check(R)
    audit_day, run_day = BS._days(R)
    when = (f"data audit and choice of the basins: {audit_day}; experiments: {run_day}" if audit_day and audit_day != run_day else f"experiments: {run_day}")
    NWK = S0.new_work(R)
    if NWK:
        when += f"; {BS._word(NWK['n'])} more experiments: {NWK['days']}"
    n_full, n_back = plan["full_deck_slides"], plan["full_deck_backup_slides"]
    example = _md(n2(BS._all_basins(R)[1]("wq", LEADS[0])["lstm_nse_median"]))
    out = [f"# Speaker notes: {DECK} (five slides)", "",
           f"Built on {BS._date(datetime.date.today().isoformat())} by `python code/build_slides_short.py` together with `{NAME}.html`, `{NAME}.pdf` and `{NAME}.pptx`, from one list of slides. "
           f"Every number below is read from `results/*.json` and `results/*.csv`; do not edit this file by hand, rebuild it after a rerun.", "",
           f"**Present from `{NAME}.pdf` in full-screen mode.** It is the version that was looked at page by page when the slides were last revised ({_revised()}, by Claude Code). "
           f"`{NAME}.pptx` is the editable copy with the speaker notes: the same {len(deck)} slides, positions, sizes and typeface (the builder checks this), but it was not opened in "
           f"PowerPoint on the build machine, and its verdicts are text marks instead of labels. If the PPTX is used, open every slide once before the talk.", "",
           f"**This is the short form of the talk, for the hand-in.** The full deck, `Flood_Slides.pdf` ({n_full} slides and {_word_(n_back)} backup slides; its notes are `slides_notes.md`), "
           f"stays the detailed record: when a question goes beyond the five slides, open the slide of the full deck that stands under *If asked* (the plan lists them slide by slide).", "",
           f"**{len(deck)} slides, {clock(total)} minutes.** Under *Say* are the short sentences of each part, with the numbers. They fit the time at a calm pace: at most {BS.PACE:g} words a "
           f"second with the hand-over sentence counted (a number such as {example} counts as three words), and no sentence longer than {BS.MAX_SENTENCE} words. *Hand-over* is the one "
           f"sentence that passes the talk to the next presenter. Under *If asked* are facts for questions; they are not part of the 10 minutes.", "",
           f"**Slide {AT['more']} has two halves to look at and one speaker.** {name(deck[AT['more'] - 1]['owner'])} speaks for both, in this order: "
           f"{deck[AT['more'] - 1]['halves'][0]['part_title']} (left half), then {deck[AT['more'] - 1]['halves'][1]['part_title']} (right half); there is no hand-over inside the slide.", "",
           f"**Slide {AT['data']} is the dataset.** {name(deck[AT['data'] - 1]['owner'])} explains what the data are, how they were obtained, the flow, how they are computed and the imagery "
           f"facts, with the flow diagram of `results/data_flow.json`; every product, with its grid, time step, period, time stamp and the day it was read, is under *If asked*.", "",
           _md(f"**Who did what - say it plainly if asked.** Claude Code, the team&rsquo;s coding assistant, wrote all code and ran all experiments under the team lead&rsquo;s direction ({when}; "
               f"these slides were last revised on {_revised()}, also by Claude Code). "
               + (f"{BS._review_sentence(R)} The review of the shared engine is `results/engine_review.json`. " if BS._review_sentence(R) else
                  "Independent AI reviewer agents checked the shared engine and every script (`results/engine_review.json`). ")
               + f"{check} The presenters follow the roles of `PLAN.md` where the plan names one, and the project lead&rsquo;s choice where it does not: a presenter speaks for a slide "
                 f"and answers questions on it; it does not mean that this member wrote or checked it."), "",
           "| Slide | Presenter (role in the plan) | Time | Runs to | The slide’s message |", "|---|---|---|---|---|"]
    for i, s in enumerate(deck, 1):
        mine = [x for x in tl if x["slide"] == i]
        out.append(f"| {i} | {name(mine[0]['owner'])} | {mine[0]['seconds']} s | {clock(mine[0]['end_s'])} | {_md(s['title'])} |")
    out += [f"| | | **{total} s** | **{clock(total)}** | |", ""]        # the per-presenter totals and shares were removed on 2026-10-08 at the team lead's request
    out += before_you_speak(deck, R, plan)
    for i, s in enumerate(deck, 1):
        mine = [x for x in tl if x["slide"] == i]
        out += [f"## Slide {i}. {_md(s['title'])}", ""]
        for x in mine:
            seg = x["spec"]
            who, what = BS.ROLE[seg["owner"]]
            nxt = next((y for y in tl if y["start_s"] == x["end_s"]), None)
            head = f"**Presenter:** {_md(who)} - {what}. **Time:** {seg['seconds']} s ({clock(x['start_s'])} to {clock(x['end_s'])})."
            out += [head, ""]
            out += ["**Say:**", ""] + [f"{j}. {_md(y)}" for j, y in enumerate([y for y in seg["notes"] if y], 1)]
            if seg.get("handover"):
                inside = nxt is not None and nxt["slide"] == x["slide"]
                to = f" to {name(nxt['owner'])}" if nxt and nxt["owner"] != seg["owner"] else ""
                out += ["", f"**Hand-over{to}{' (inside the slide)' if inside else ''}:** {_md(seg['handover'])}"]
            elif nxt and nxt["owner"] != seg["owner"]:
                out += ["", f"**Hand-over to {name(nxt['owner'])}:** the last sentence above."]
            else:
                out += ["", "**Hand-over:** to the questions; " + name(seg["owner"]) + " leads them."] if x is tl[-1] else []
            if any(seg.get("asked", [])):
                out += ["", "**If asked:**", ""] + [f"- {_md(y)}" for y in seg["asked"] if y]
            out += ["", _shows(seg, s), ""]
        fmt = lambda ds: "; ".join(f"{'backup slide B' + str(d['backup']) if d['backup'] else 'slide ' + str(d['slide'])} ({d['label']})" for d in ds)
        where = "; ".join(f"{('left half', 'right half')[j]}: {fmt(h['detail'])}" for j, h in enumerate(s["halves"])) if s["layout"] == "split" else fmt(s["detail"])
        out += [f"**Where the detail is (full deck, `Flood_Slides.pdf` and `slides_notes.md`):** {where}.", ""]
    cited = [k for k, (short, _) in REFERENCES.items() if short in text_sources]
    if cited:
        out += ["## Sources cited on the slides and in these notes", ""] + [f"- {_md(BS._full_reference(k))}" for k in sorted(cited, key=lambda k: REFERENCES[k][1])] + [""]
    return "\n".join(out)


def _word_(n):
    return BS._word(n)


# ---- checks and main ------------------------------------------------------------------------------------------------------------------------
def all_text(deck):
    """Every string of the five slides and their notes, to find the sources they cite."""
    parts = []
    for s in deck:
        for k in ("title", "sub", "caption"):
            parts.append(str(s.get(k, "")))
        parts += [str(x) for x in s.get("small", []) + s.get("bullets", [])]
        for seg in segments(s):
            parts += [str(x) for x in seg.get("notes", []) + seg.get("asked", []) + seg.get("bullets", [])] + [str(seg.get("caption", ""))]
        if s.get("table"):
            parts += [str(c) for row in s["table"]["rows"] for c in row]
    return html.unescape(" ".join(parts))


def main():
    R, T = load_results()
    deck, full = build_deck(R, T)
    tl = timeline(deck)
    total, tot = sum(x["seconds"] for x in tl), totals(deck)
    if total != TOTAL_SECONDS:
        print(f"the time budgets of the five slides add up to {total} s, not {TOTAL_SECONDS} s", file=sys.stderr)
    if set(tot) != set(BS.ROLE) or set(tot.values()) != {EACH_SECONDS}:
        print("the four members do not speak the same time: " + ", ".join(f"{_md(BS.ROLE[k][0])} {sec} s" for k, sec in tot.items()) + f" (each should speak {EACH_SECONDS} s)", file=sys.stderr)
    out = ROOT / f"{NAME}.html"
    out.write_text(deck_html(deck), encoding="utf-8")
    print("wrote", out, f"({len(deck)} slides, {total} s)")
    if print_pdf(out, out.with_suffix(".pdf")):
        print("wrote", out.with_suffix(".pdf"))
    deck_pptx(deck, out.with_suffix(".pptx"))
    print("wrote", out.with_suffix(".pptx"))
    plan = slide_plan(deck, full, R)
    notes = notes_md(deck, full, R, plan, all_text(deck))
    (ROOT / "slides_notes_5.md").write_text(notes, encoding="utf-8")
    print("wrote", ROOT / "slides_notes_5.md")
    (RES / "slide_plan_5.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", RES / "slide_plan_5.json", "(" + ", ".join(f"{p['label']} {p['seconds']} s" for p in plan["presenters"]) + ")")
    said = lambda k: _md(BS.ROLE[k][0])                                # the notes and the plan file are made by two small pieces of code: say so if they ever differ
    for c in plan["clock_checkpoints"]:
        if f"When slide {c['slide']}{' (' + c['part'] + ')' if c['part'] else ''} starts ({said(c['presenter'])}), the clock should show {c['clock']}." not in notes:
            print(f"slide_plan_5.json and slides_notes_5.md disagree on the clock checkpoint of slide {c['slide']}", file=sys.stderr)
    if plan["late_cut"] and f"more than {plan['late_cut']['clock_over']} when slide {plan['late_cut']['slide']}" not in notes:
        print("slide_plan_5.json and slides_notes_5.md disagree on the cut for a late talk", file=sys.stderr)
    for problem in check_pptx(out.with_suffix(".pptx"), deck):
        print("PPTX check:", problem, file=sys.stderr)
    for i, over in overflow(deck):
        print(f"slide {i}: the text is estimated to run {over} in past the content area - shorten it", file=sys.stderr)
    for i, s in enumerate(deck, 1):
        if s.get("image") and s.get("crop") and not BS._image(s)[2]:
            print(f"slide {i}: the crop of figures/{s['image']} was refused - the whole figure is shown; check the slide", file=sys.stderr)
    print("slide / part: words to say / words the time allows")
    for x in tl:
        v = _view(x["spec"])
        label = f"slide {x['slide']}{' ' + x['part'] if x['part'] else ''}"
        print(f"  {label}: {BS.spoken(v)}/{round(10 * BS.PACE) * v['seconds'] // 10}")
        if BS.too_fast(v):
            print(f"{label}: {BS.spoken(v)} words to say in {v['seconds']} s - more than {BS.PACE:g} a second, shorten the notes", file=sys.stderr)
        for words, sentence in BS.long_sentences(v):
            print(f"{label}: a sentence to say has {words} words, more than {BS.MAX_SENTENCE}: '{sentence}'", file=sys.stderr)
    for i, s in enumerate(deck, 1):
        if s.get("image") and s["layout"] != "cover":
            g = geometry(s)
            print(f"slide {i}: figures/{s['image']} at {g['fw']:.1f} x {g['fh']:.1f} in ({100 * g['fw'] / BS.W:.0f}% of the slide's width)")
            if s.get("letter_pt") and s.get("design_w"):
                pt = s["letter_pt"] * g["fw"] / s["design_w"]
                print(f"  smallest lettering of that figure on the slide: {pt:.1f} pt")
                if pt < MIN_LETTER_PT:
                    print(f"slide {i}: the figure is shown so small that its smallest lettering is {pt:.1f} pt, below {MIN_LETTER_PT} pt - shorten the lines under it", file=sys.stderr)
    for line in LEFT_OUT:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
