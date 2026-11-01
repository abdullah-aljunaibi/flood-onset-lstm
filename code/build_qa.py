"""Build the professor's-questions sheet (PLAN.md, week 8): Professor_QA.md, Professor_QA.html and Professor_QA.pdf.

Questions a demanding professor could ask about this project (their number is counted, not typed). Each has an answer that
begins with what to say first, 'the trap' (the wrong or over-confident answer, and why it is wrong) and a pointer to the report
section and the results file or script behind it. After the last question come, on a page of their own, what each team member
should open, run or verify before the hand-in, and then the sources named in the sheet.
Nothing is computed here: every number is read from results/*.json or results/*.csv and formatted in place, and wording that
depends on an outcome is chosen from the stored values. After a rerun of any experiment:  python code/build_qa.py
Answers that need results/lstm_lobo.json or results/rain_timing.json say 'result pending' until those files exist; the files that
came last (day_windows.json, sources_checked.json, floods_2004.json) are read by key inside guarded blocks, so the sheet builds
with or without them. The same holds for what was added on 4 October 2026 from the stored results: key_figure.json (with
figures/key_figure.png), training_curves.json, uncertainty.json, same_misses.json and published_scores.json. Where one of them or
one of its keys is missing, its sentence or its question is left out and one line on stderr says so.
The names of the team and the state of each member's own check come from team.json through code/team.py; no builder writes it.

Page 1 is what a member reads in the last ten minutes: the finding, then the ten questions most likely to be asked, each with a
two-sentence answer and the numbers of the full answers. Page 2 is the summary 'If asked what did you find?' and the words used.

A question is referred to by its key, X('onsets'), never by a typed number: questions can be added without renumbering by hand,
and a reference to a key that does not exist stops the build. New questions are added at the end of their group, so that no
existing number moves.
The network, its loss and its optimiser are fixed properties of code/flood.py, not results: they are worded here and checked
against that file every time the sheet is built (code_facts); a clause that the code no longer supports is left out and reported.
The same is done for the Transformer of the comparison (the source of code/09_model_comparison.py).

Final state of 30 October 2026 (rebuilt for the hand-in before 2 November and the talk of 2 November 2026). Every pointer names the
final report, hand-in item 3 (Handin_2026-11-02/3_Final_Project_Report_Flood.pdf), with a section, table, figure or appendix that its
docx holds, and, where the pointer is given for a statement, only when that part of the report holds the words it is given for
(code/qa_final.py: FinalReport.where / inline; report_says asks the same of the final report). A pointer that fails is left out with one
line on stderr. The long report of 8 October 2026 (Flood_Report.pdf) is named only as that earlier record. Group I (code/qa_since_midterm.py)
holds what was done after the mid-term talk (E4a, E4b, E4e, the five seeds of scripts 15 to 18, the GIS layers, the teammates' package,
the final documents); its numbers are read from the registry of the final report (results/final_report_numbers.json), so that the sheet
prints them as the report and the deck do. The review records are read round by round and tolerantly (their counts are named differently
across rounds). The member page names the members with their roles (code/final_report/config.json) and Group #__, their slides and seconds
(the speaker notes of the final deck) and their own check (team_checks/team_check.py); nothing on it says that anyone has done anything.
"""
import html as html_lib
import inspect
import re
import sys
from datetime import datetime

import team
from common import FIG, ROOT
from flood import FORCING, Cfg, LSTMNet, fit
from report_common import CITED, LEAD_WORD, REFERENCES, cite, count, load_results, n2, page, pct, print_pdf, signed
# the fixed wordings and the verified readers of the experiments of 7 and 8 October 2026 come from page 1 of the report, so that the sheet and the
# report say the same words and quote the same numbers (each reader returns None, with one line on stderr, while its results file is missing)
from report_sections import s0_summary as S0
from report_sections.s0_summary import (AORC_WORDING, INFO_NOT_NET, MOVED_POINT, REALTIME_WORDING, aorc_facts, clock_facts, era5_facts, forecast_facts, four_experiments,
                                         hindsight_clause, hindsight_facts, new_work, nse_forecast_numbers, nse_forecast_words, reading_word, review_sentence, rivers_facts,
                                         stage4_facts)
from report_sections.s8_limits import _minutes

import qa_final as QF                            # the final report's text (pointers), its registry, the deck's timing, the team checks
import qa_since_midterm as QI                    # group I (after the mid-term talk) and the member page of the final state

LEADS = sorted(LEAD_WORD)
P_CUT = 0.05                                    # the usual line for 'p is small', as in report section 4
LOW_LEVEL = 0.7                                 # the lower warning level quoted as an example in D4 (a multiple of the high-flow level)
FAR_ELEV = 25                                   # metres: a basin whose mean elevation differs from CAMELS by more than this is named
TT, TT_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
UPPER = "an upper bound that cannot be had in practice"
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
FAMILY = {"linreg": "ridge regression", "rf": "random forest", "persistence": TT, "climatology": "the calendar-day average"}
PROP = {"log_area": "area", "elev_mean": "elevation", "slope_mean": "slope", "p_mean": "mean rain", "aridity": "dryness of the climate",
        "p_seasonality": "season of the rain", "frac_forest": "forest share", "soil_depth_pelletier": "soil depth", "sand_frac": "sand in the soil",
        "geol_permeability": "permeability of the rock"}
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
STORM = {"HU": "hurricane", "TS": "tropical storm", "TD": "tropical depression"}        # status codes of the storm-track record (HURDAT2)
# where the sheet sends a reader: the final report (hand-in item 3), checked against its docx text. W(...) is the 'Where' pointer, IN(...) a pointer
# inside a sentence; an item is a part ('5.3', 'T6', 'F8', 'AppC') or (part, words that part must hold). The long report of 8 October 2026
# (Flood_Report.pdf) and its section numbers are no longer used.
FR = QF.final_report()
W, IN = FR.where, FR.inline
# the final report's sections for the topics of the old section constants of the long report of 8 October 2026 (located in the final report's
# text on 30 October 2026 with qa_final.FinalReport: data 4.1, network settings 5.2, comparison and verdicts 7.1, figures 7.2, robustness 7.3,
# day ends 4.2, the seed rule 6.3, interpretation 8.1, limits 8.2, contributions 10.2, resources 10.3, the map Appendix B)
S_DATA, S_ONE, S_ALL, S_NET, S_UNSEEN, S_TIMING, S_WARN = "4.1", "5.2", "7.1", "7.1", "7.3", "4.2", "7.1"
S_MAPS, S_SAT, S_COST, S_LIMITS, S_WHO = "4.1 and Appendix B", "5.4", "10.3", "8.2", "10.2"
S_RULE, S_SKILL = "6.3", "8.1"
S_RAIN, S_MORE, S_FORECAST = "7.1", "7.3", "7.2"
# the finding and its supporting line: fixed sentences, the same words in the report, the slides and this sheet (imported from page 1 of the report;
# the report's FINDING has no full stop, this sheet's has)
FINDING = S0.FINDING + "."
SUPPORT = S0.SUPPORT
FIGURE_KEY = "key_figure.png"                   # the key figure of the main conclusion (code/11_key_figure.py)
SAME_SIZE = 0.10                                # NSE: a published score and ours are called 'of the same size' only within this distance
FEW_HUNDREDTHS = 0.05                           # 'a few hundredths at most': every end of every range closer to zero than this (as the seed rule's ranges)
GROUPS = {"A": "The four things every member must be able to explain", "B": "Data and set-up", "C": "The model and its training", "D": "Results",
          "E": "Remote sensing, GIS and GNSS", "F": "Transformer and cost", "G": "Honesty and limits",
          "H": "The four experiments of the last days (E1, E1b, E3, E2): rain to midnight, a forecast, more rivers",      # the days are put in by build(): GROUPS["H"] says 'of 7 and 8 October' once they are read
          "I": QI.GROUP_TITLE.format(when="")}                                                                           # the day of the mid-term talk is put in by build()
CSS_QA = """
body{font-size:9.4pt;line-height:1.38}
h1{margin-bottom:2pt} h2{margin:11pt 0 5pt;padding-top:3pt;border-top:1.5px solid var(--axis)} h2.first{break-before:page;border-top:none;margin-top:0;padding-top:0}
.qa{break-inside:avoid;margin:0 0 8.5pt}
.qa .q{font-weight:600;font-size:10.1pt;margin:0 0 2pt} .qa .q span{color:var(--blue);margin-right:5pt}
.qa .a{margin:0 0 3pt}
.qa .trap{border-left:3px solid var(--orange);padding:0 0 0 7pt;margin:0 0 2.5pt;font-size:8.9pt}
.qa .where{font-size:8pt;color:var(--ink2);margin:0}
.lead{font-size:9.2pt;margin:0 0 4pt} .gloss{font-size:8.7pt;color:var(--ink2);margin:0 0 4pt}
.story{border:1.6px solid var(--blue);border-radius:8px;padding:5pt 10pt 3pt;margin:5pt 0 6pt;font-size:9.2pt} .story ol{margin:2pt 0 2pt}
.find{font-size:11.2pt;line-height:1.34;margin:1pt 0 6pt} .find b{font-weight:600} .find span{font-size:9.2pt;color:var(--ink2)}
.top{border:1.6px solid var(--blue);border-radius:8px;padding:5pt 10pt 3pt;margin:5pt 0 6pt;font-size:9.1pt;line-height:1.36}
.top .h{font-size:10.1pt;font-weight:600;margin:0 0 3pt} .top .h span{font-weight:400;font-size:8.6pt;color:var(--ink2)}
.top ol{margin:0 0 2pt} .top li{margin-bottom:3.2pt;break-inside:avoid} .top .to{color:var(--blue);font-weight:600;white-space:nowrap}
.top .also{font-size:8.7pt;color:var(--ink2);margin:0 0 2pt}
.front2,.keep{break-inside:avoid}
.check{font-size:8.9pt;break-before:page} .check h2{border-top:none;margin-top:0;padding-top:0} .check h3{margin:6pt 0 2pt;font-size:9.6pt} .check ul{margin:0 0 3pt} .check li{margin-bottom:1.5pt}
.cols{column-count:2;column-gap:14pt} .cols > div{break-inside:avoid}
.refs{column-count:2;column-gap:14pt;font-size:7.6pt} .refs li{break-inside:avoid} .src{break-inside:avoid} .src h2{margin-top:9pt}
@page{@bottom-left{content:'Predicting river floods with AI \\00B7  the professor\\2019s questions';font:7.5pt 'Helvetica Neue',sans-serif;color:#898781}
@bottom-right{content:counter(page) ' / ' counter(pages);font:7.5pt 'Helvetica Neue',sans-serif;color:#898781}}
"""                                             # the last rule: a running title and page numbers in the bottom margin (this sheet only, not the shared CSS)
# .front2 (the summary box when page 1 holds the ten questions) is kept in one piece and so starts page 2; it is not a forced page break, so that
# nothing is left on a nearly empty page if the box of the ten questions should ever run over page 1


# ---- small helpers -------------------------------------------------------------------------------------------------------
def lk(k):
    return f"l{k}"


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def cap(s):
    return s[0].upper() + s[1:]


def lower_first(s):
    return s[0].lower() + s[1:]


def join(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def slash(vals, fmt=n2):
    """Three values for 1, 2 and 3 days ahead: '0.59 / 0.32 / 0.25'."""
    return " / ".join(fmt(v) for v in vals)


def leads(ks):
    """[1] -> '1 day ahead'; [2, 3] -> '2 and 3 days ahead'."""
    ks = sorted(ks)
    return LEAD_WORD[ks[0]] if len(ks) == 1 else join(ks) + " days ahead"


def years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def n_years(span):
    return round((datetime.fromisoformat(span[1]) - datetime.fromisoformat(span[0])).days / 365.25)


def date(iso):
    d = datetime.fromisoformat(iso[:10])
    return f"{d.day} {MONTH[d.month - 1]} {d.year}"


def river(short):
    """'French Broad, NC' -> 'French Broad'."""
    return short.split(",")[0]


def note_with(notes, text):
    """The first note of a results file that contains `text` (or '')."""
    return next((n for n in notes if text in n), "")


def X(*keys):
    """A reference to other questions by their keys: X('timing', 'onsets') becomes 'B6, D3' once every question has its number."""
    return ", ".join(f"⟦{k}⟧" for k in keys)


def resolve(text, number):
    """Replace the tokens of X() by the question numbers; a key that no question carries stops the build (KeyError)."""
    return re.sub("⟦(\\w+)⟧", lambda m: number[m.group(1)], text)


def about(v):
    """A share in words where a plain fraction is near: 0.25 -> 'about a quarter'; otherwise the percentage. The same bands as the report uses
    (report_sections.s0_summary.share_word), so that the two documents choose the same words for the same number."""
    for lo, hi, words in ((0.22, 0.28, "about a quarter"), (0.30, 0.36, "about a third"), (0.45, 0.55, "about half"), (0.64, 0.69, "about two thirds"),
                          (0.72, 0.78, "about three quarters")):
        if lo <= v <= hi:
            return words
    return pct(v)


def report_shows(name):
    """True if the final report shows figures/<name>: one of its section sources (code/final_report/sections/*.md) places that figure. Asked
    before the sheet sends a reader to a figure of the report; a figure that only the project folder holds is named by its file."""
    try:
        return any(f"figures/{name}" in p.read_text(encoding="utf-8") for p in (ROOT / "code" / "final_report" / "sections").glob("*.md"))
    except OSError:
        return False


def report_says(part, text):
    """True if a part of the final report (a section '7.3', a table 'T6', a figure 'F8', an appendix 'AppC') holds `text` in its docx. Asked before
    the sheet names a figure, a table or a paragraph of the report, so that the sheet never sends a reader to something the final report does not
    have; without it the answer names the section only, or nothing."""
    return FR.says(part, text)


def seed_rule(s):
    """A shared sentence that names 'the rule of section 4.2' of the long report of 8 October 2026 -> the seed rule of final report Section 6.3."""
    return s.replace("by the rule of section 4.2", f"by the seed rule ({IN(('6.3', 'The seed rule'))})").replace(
        "the rule of section 4.2", f"the seed rule ({IN(('6.3', 'The seed rule'))})")


def plan_quotes():
    """Two passages of PLAN.md that the answer on the unmet milestone quotes word for word: milestone 2 as the plan states it, and the first
    clause of the fallback the plan gives for the risk 'LSTM does not beat the baselines'. ('' for a passage that is not found.)"""
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    m2 = re.search(r"^- M2 \([^)]*\): (.*?)\. \*\*", plan, flags=re.M)
    risk = re.search(r"^\| *LSTM does not beat the baselines *\| *(.*?) *\|", plan, flags=re.M)
    return (m2.group(1).strip() if m2 else ""), (risk.group(1).split(",")[0].strip() if risk else "")


def signed0(v, d=2):
    """signed(), but a value that rounds to zero is printed as 0.00, without a sign."""
    return n2(0.0, d) if round(v, d) == 0 else signed(v, d)


def as_clause(s):
    """'The team's own checks ...' -> 'the team's own checks ...' (for a sentence that continues another), with the apostrophes of the sheet."""
    s = html_lib.escape(s, quote=False)                            # a name from team.json may hold any character
    return (s[0].lower() + s[1:]).replace("'", "&rsquo;")


def resamples(share):
    """A share of the resamples of results/uncertainty.json in words: 1.0 -> 'every resample'; a share that would round up to 100% is
    not printed as 100%."""
    return "every resample" if share >= 1 else "more than 99% of the resamples" if share > 0.995 else f"{pct(share)} of the resamples"


def code_facts():
    """What the sheet says about the network, checked against code/flood.py itself (its docstrings and the source of fit).
    These are fixed properties of the code, not results; a fact that is no longer found is reported and its clause is left out."""
    src, doc = inspect.getsource(fit), " ".join((fit.__doc__ or "").split())
    net_src = inspect.getsource(LSTMNet)
    # the start value of the forget gates: PyTorch keeps the four blocks of an LSTM in the order input, forget, new value, output, so the
    # second block of the bias vector is the forget gate (None if LSTMNet no longer sets it)
    forget = re.search(r"bias_hh_l0\[cfg\.hidden:2 \* cfg\.hidden\]\.fill_\(([\d.]+)\)", net_src)
    facts = {"forget_bias": float(forget.group(1)) if forget else None,
             "one_layer": "One LSTM layer" in (LSTMNet.__doc__ or "") and "num_layers" not in net_src,
             "linear_head": "nn.Linear(cfg.hidden, len(LEADS))" in net_src,
             "dropout_before_head": "self.head(self.drop(" in net_src,
             "loss_per_basin": "0.1 x the spread of all training basins" in doc and "/ sc[\"q_sd\"] + 0.1) ** 2" in src,
             "adam": "torch.optim.Adam(" in src,
             "clip": "clip_grad_norm_(" in src,
             "random_sequences": "rng.integers(0, len(ids), cfg.batch)" in src,
             "log_flow": "np.log1p(d.q_mm.values)" in inspect.getsource(sys.modules[fit.__module__]._design),
             "not_below_zero": "np.clip(" in inspect.getsource(sys.modules[fit.__module__]._to_target_day)}
    for name, ok in facts.items():
        if not ok:
            print(f"build_qa: code/flood.py no longer shows '{name}': its clause in the question on the network was left out - check that answer", file=sys.stderr)
    return facts


def reading(c, p_cut=P_CUT):
    """One LSTM-against-simple-model comparison of lstm_all.json, by the rule stated in report section 4.2: the LSTM is called
    better only when the simple model is below every single seed, the LSTM is higher in most basins and p (two-sided) is below 0.05."""
    if not c.get("like_for_like", True):
        return "nocall"
    most, few = c["n_basins_lstm_higher"] > c["n_basins"] / 2, c["n_basins_lstm_higher"] < c["n_basins"] / 2
    small_p, below = c["wilcoxon_p"] < p_cut, c["baseline_against_seed_spread"] == "below every seed"
    if most and small_p:
        return "lstm" if below else "lstm_avg"
    if few and small_p:
        return "simple"
    return "lstm_med" if most and below else "tie"


PHRASE = {"lstm": "the LSTM is better", "lstm_avg": "only the average forecast of the LSTM seeds is ahead", "lstm_med": f"the LSTM scores higher, but p is above {P_CUT:g}",
          "tie": "there is no clear difference", "simple": "the simple model is better", "nocall": "no winner is called"}
# the same readings for the upper bound, where even the comparable lead time is not exactly like-for-like (the two kinds of model read the
# same days of weather, not the same variables): 'scores higher', never 'is better' - the words of report section 4.2
PHRASE_UB = dict(PHRASE, lstm="the LSTM scores higher", simple="the simple model scores higher")


def say(classes):
    """{lead: reading} -> 'there is no clear difference at any of the three lead times' or '1 day ahead ...; 2 and 3 days ahead ...'."""
    groups = {}
    for k in LEADS:
        groups.setdefault(classes[k], []).append(k)
    if len(groups) == 1:
        c = next(iter(groups))
        return PHRASE[c] + (" at any of the three lead times" if c == "tie" else " at all three lead times")
    return "; ".join(f"{leads(ks)} {PHRASE[c]}" for c, ks in groups.items())


def open_decisions():
    """The numbered items under 'Decisions still needed from the group' in PLAN.md ([] if the heading is not found)."""
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    m = re.search(r"## Decisions still needed from the group\s*\n(.*?)(?:\n## |\Z)", plan, flags=re.S)
    return [re.sub(r"^\d+\.\s*", "", ln).rstrip(".") for ln in m.group(1).splitlines() if re.match(r"\d+\.", ln)] if m else []


# ---- the experiments of 7 and 8 October 2026: what the questions of group H and the updated answers quote -------------------------
NEW_FILES = ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input")     # E1, E1b, E3 and E2: report sections 4.6 to 4.8
NEW_LABELS = {"rain_to_midnight": "E1 (ERA5 rain to midnight)", "aorc_rain_to_midnight": "E1b (AORC rain to midnight)", "more_basins": "E3 (more rivers)",
              "forecast_input": "E2 (the forecast)"}


def _file(R, name):
    """results/<name>.json when it can be quoted (it exists and is no smoke test), else None."""
    X = R.get(name)
    return X if isinstance(X, dict) and not X.get("quick_mode") else None


def _check(R, review, cid):
    """The 'numbers' of one check (by id) of an independent review record; KeyError if the record or the check is not there."""
    rev = _file(R, review)
    if rev is None:
        raise KeyError(f"results/{review}.json is not there")
    return next(c for c in rev["checks"] if c["id"] == cid)["numbers"]


def new_work_facts(R, T):
    """What the questions of group H and the updated answers quote from the experiments of 7 and 8 October 2026 (report sections 4.6 to 4.8). Every number is
    read by key from the results files: through the verified readers of page 1 of the report (s0_summary: era5_facts, aorc_facts, hindsight_facts,
    stage4_facts, clock_facts, rivers_facts, forecast_facts, and the sentences of facts()['new']) and through a few more reads of the records behind them.
    A dict of parts; a part is None, with one line on stderr, while its results file is missing or not in the expected form: the questions that need it are
    left out, and so are the sentences of other answers that quote it."""
    def part(what, make):
        try:
            return make()
        except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError, StopIteration) as err:
            print(f"build_qa: {what} was not read ({err!r}): what needs it is left out", file=sys.stderr)
            return None

    NW = dict(E1=era5_facts(R), AO=aorc_facts(R), HS=hindsight_facts(R), S4=stage4_facts(R), CK=clock_facts(R), RV=rivers_facts(R), FC=forecast_facts(R))
    NW["PAGE1"] = part("the sentences of page 1 of the report (s0_summary.facts)", lambda: S0.facts(R, T)["new"])
    NW["PAGE1"] = NW["PAGE1"] if NW["PAGE1"] and NW["PAGE1"].get("rungs") else None

    def clock():                                                   # the clock of AORC: the review's proofs and the stamp
        A, rev = R["aorc_rain_to_midnight"], R["review_aorc_rain_to_midnight"]
        b3, c2 = _check(R, "review_aorc_rain_to_midnight", "b3"), _check(R, "review_aorc_rain_to_midnight", "c2")
        what_b3 = next(c for c in rev["checks"] if c["id"] == "b3")["what"]
        plus = re.search(r"\+(\d+) mm added", what_b3)                # the rain that the proof adds after the end of each window
        gauge_day = re.search(r"gauge day that ends at (\d+) UTC", A["outcome_caveat"])           # the day on which AORC's daily totals are made
        table_end = next(c["day_ends_local"] for c in A["inputs"]["clock"] if str(c["input"]).startswith("aorc_to_09"))                    # the rain table's own window ends at ...
        utc_end = re.search(r"ends (\d\d:\d\d)", A["figure_labels"]["wqa_utc"])                                                           # the UTC-day control ends at ...
        run_utc = re.search(r"the (\d\d) UTC run", R["gefs_basins"]["rule"]["run"])                                                       # the forecast run that the input uses
        early = re.search(r"hours before (\d{4})", A["outcome_caveat"])                                                                  # before which year AORC's hours are disaggregated daily totals
        if not (plus and gauge_day and table_end and utc_end and run_utc and early):
            raise ValueError("the perturbation of check b3, the gauge day of AORC, the windows of its controls or the run of the forecast is not found in the records")
        sc = A["stamp_convention_check"]["sets"]["wqa"]["by_lead"]["l1"]
        return dict(cases=b3["n_cases"], cases_ok=bool(b3["all_ok"]), r_lag0=c2["r_lag0_pooled_78_days"], r_off=max(c2["r_lag_minus1_and_plus1_pooled"]),
                    plus_mm=int(plus.group(1)), gauge_day_utc=int(gauge_day.group(1)), table_end=table_end, utc_end=utc_end.group(1), run_utc=run_utc.group(1), early=early.group(1),
                    end_of_hour=bool(c2["stamp_is_end_of_hour_in_the_data"]), opp_points=sc["difference_points"], opp_from=sc["onset_warned_documented_convention"],
                    opp_to=sc["onset_warned_opposite_convention"], opp_hits=(sc["hits_documented"], sc["hits_opposite"]))
    NW["CLOCK"] = part("the clock checks of results/review_aorc_rain_to_midnight.json, results/aorc_rain_to_midnight.json and results/gefs_basins.json", clock)

    def hind():                                                    # the independent review's tests of hindsight in AORC
        d3, d4 = (_check(R, "review_aorc_rain_to_midnight", c) for c in ("d3", "d4"))
        planted = d3["positive_control_leak_of_a_share_of_aorc_own_after_midnight_rain_put_into_the_data_78_days"]
        share = max(planted, key=float)                           # the largest planted leak: a share of the night rain put into the hours before midnight
        dry = d4["dry_before_wet_after"]
        covered = re.match(r"(\d+) of (\d+)", NW["HS"]["covered"])
        return dict(basin_days=d3["sample"]["basin_days"], my_days=d3["sample"]["my_days"], night_mine=d3["sample"]["mean_stage_iv_mm_after_midnight_mine"],
                    night_builder=d3["sample"]["mean_stage_iv_mm_after_midnight_builder"], planted_share=float(share), planted_b=planted[share]["b2"],
                    planted_seen=bool(planted[share]["interval_above_zero"]), dry_days=dry["basin_days"], dry_after=dry["mean_stage_iv_mm_after_midnight"],
                    dry_before=dry["mean_aorc_mm_before_midnight"], n_covered=int(covered.group(1)), n_onsets=int(covered.group(2)))
    NW["HIND"] = part("the hindsight checks of results/review_aorc_rain_to_midnight.json (d3, d4)", hind) if NW["HS"] else None

    def hours():                                                   # the onsets warned only by the rain to midnight, and only by its matched-window control
        o = R["aorc_rain_to_midnight"]["value_of_hours_09_to_24"]["overlap_of_warned_onsets"]
        return dict(only_rain=o["warned_only_by_first"], only_control=o["warned_only_by_second"], hits=(o["hits_first"], o["hits_second"]))
    NW["HOURS"] = part("the overlap of warned onsets in results/aorc_rain_to_midnight.json", hours)

    def fcx():                                                     # the forecast: the 1-day result, the controls, the splits
        X = R["forecast_input"]
        L1, OP = X["lead1_facts"], X["value_of_the_forecast"]["operating_points"]["l1"]
        ctl, ret = X["forecast_vs_climatology_control"]["by_lead"], X["retrained_vs_substituted"]["retrained_against_its_control"]
        Y = X["yardstick"]["lstm_against_simple"]
        budget_keys = sorted(int(m.group(1)) for k in OP["equal_false_warning_budgets"] for m in [re.fullmatch(r"points_at_most_(\d+)", k)] if m)
        pub = R["published_context"]["sources"]["guan2022"]["numbers"]                         # the two forecast-hour windows of the published Brier skill (5 mm rows)
        i5 = next(j for j, x in enumerate(pub) if ">= 5 mm" in x["quantity"])
        windows = [re.search(r"(\d+)-(\d+)", pub[i5]["quantity"]).groups(), re.search(r"(\d+)-(\d+)", pub[i5 + 1]["quantity"]).groups()]
        if budget_keys != [40, 50]:
            raise ValueError(f"the false-warning budgets of the results file are {budget_keys}, not the two the answer words")
        return dict(perfect_slots=L1["warned"]["same_networks_given_the_perfect_rain"], lost=L1["lost_onsets"]["n"],
                    budgets_pct=budget_keys, box_deg=R["gefs_basins"]["grid"]["box_size_deg"], brier_windows=windows,
                    yard_nse=[Y[f"lstm_wqaF_sub__base_{m}_wqaF"]["l1"]["reading_in_words"] for m in ("linreg", "rf")],
                    lost_even=L1["lost_onsets"]["missed_even_with_the_perfect_rain"], lost_even_share=L1["lost_onsets"]["share_missed_even_with_the_perfect_rain"],
                    small_n=L1["little_rain_on_the_target_day"]["n"], small_rule=L1["little_rain_on_the_target_day"]["rule"],
                    small_without=L1["little_rain_on_the_target_day"]["no_slots"], small_with=L1["little_rain_on_the_target_day"]["forecast"],
                    fewer_basins=L1["by_basin"]["basins_where_the_forecast_model_warns_of_fewer"],
                    ratio_onset_days=L1["forecast_on_the_onset_days"]["mean_ratio_to_the_aorc_rain"],
                    chosen=OP["level_chosen_on_the_validation_years"], budgets=OP["equal_false_warning_budgets"],
                    ap_ci=OP["trigger_free"]["average_precision_difference_ci95"], boot=OP["block_bootstrap_fixed_level"],
                    above_ctrl={k: ctl[f"l{k}"]["onsets"]["difference_points"] for k in LEADS},
                    above_ctrl_s={k: ret[f"l{k}"]["onsets"]["difference_points"] for k in LEADS},
                    split_main=X["split_main"], split_sat=X["split_satellite_era"])
    NW["FCX"] = part("the 1-day facts and the controls of results/forecast_input.json", fcx) if NW["FC"] else None

    def unc():                                                     # the ranges of the new headline numbers (13_uncertainty.py, added on 2026-10-08)
        U = R["uncertainty"]
        S = U["stats"]
        rng = lambda name: (S[name]["value"], S[name]["both"]["lo"], S[name]["both"]["hi"])
        return dict(rain=rng("onsets_warned.lstm_wqa.l1"), fc2=rng("onsets_warned.lstm_wqaF_sub.l2"), fc3=rng("onsets_warned.lstm_wqaF_sub.l3"),
                    B=int(U["resampling"]["B"]))
    NW["UNC"] = part("the ranges of the new rungs in results/uncertainty.json", unc)

    def cost():                                                    # what the experiments took: downloads, trainings, computing time, money
        A, S, DA = R["aorc_rain_to_midnight"], R["stage4_realtime"], R["data_access"]
        dl = A["aorc_data"]["download"]
        g = DA["summary"]["archived_weather_forecasts"]["gefsv12_reforecast_aws"]["download_done"]
        mm = re.search(r"(\d+):(\d+)-(\d+):(\d+)", g["when"])
        s4 = S["data"]["fetch_runs"]
        mins = {k: _minutes(R, k) for k in NEW_FILES}
        if any(v is None for v in mins.values()):
            raise KeyError("a results file of the four experiments has no timing")
        kept = re.search(r"(\d+) MB", g["kept_on_disk"])
        if not (mm and kept):
            raise ValueError("the download record of the GEFS reforecast has no clock time or no size of what was kept")
        return dict(aorc_gb=dl["gb"], aorc_h=dl["hours"], gefs_gb=g["bytes_read_GB"], kept_mb=int(kept.group(1)),
                    gefs_min=int(mm.group(3)) * 60 + int(mm.group(4)) - int(mm.group(1)) * 60 - int(mm.group(2)),
                    s4_gb=s4["gb"], s4_h=s4["hours"], trainings={k: R[k]["n_trainings"] for k in NEW_FILES}, hours=sum(mins.values()) / 60,
                    nothing_bought="Nothing was bought and no account was created" in DA["what"],
                    cluster_note="largest paid item is a few US dollars" in DA["summary"]["costs"]["no_cluster_needed"],
                    saved=len(list((ROOT / "results" / "models").glob("*.pt"))))
    NW["COST"] = part("the download and training records (aorc_data, data_access, stage4_realtime and the four results files)", cost)

    def reviews():                                                 # the independent AI reviews of the seven experiments, every round read tolerantly
        out = QI.review_facts(R)                                   # rounds, listed checks and mismatches over all rounds; the latest (five-seed) round
        for k in NEW_FILES:                                        # the first round of the four experiments of 7 and 8 October: its wording items, its leak verdict
            rev = R[f"review_{k}"]
            out[k].update(must=len(rev.get("must_fix") or []), should=len(rev.get("should_fix") or []),
                          leak=bool(re.search(r"no leakage|no way for information from after the issue time|no hour after a window's end enters any input",
                                              str(rev.get("verdict", "")))))
        return out
    NW["REV"] = part("the review records results/review_*.json of the seven experiments (every round, read tolerantly)", reviews)

    def spent():                                                   # what was decided after the test-year scores of earlier experiments were known
        out = {}
        for k in NEW_FILES:
            c = R[k]["choices_made_after_looking_at_test_scores"]
            out[k] = len(c)
        return out
    NW["SPENT"] = part("choices_made_after_looking_at_test_scores of the four results files", spent)
    return NW


# ---- the questions ---------------------------------------------------------------------------------------------------------
def build(R, T):
    """-> (story, items, checklist, pending, page1): every string is built from the results files R (json) and T (csv).
    page1: the finding as one line ('' unless the results carry it), the ten most likely questions as (question, short answer, numbers of the
    full answers), and the line 'Also likely'."""
    B, LS, LA, FA, G, SD, SI, MC = (R[k] for k in ("baselines", "lstm_single", "lstm_all", "flood_analysis", "basin_gis", "satellite_data", "satellite_inputs", "model_comparison"))
    ER, FT, DA = R["engine_review"], R["fair_test"], R["data_audit"]
    LOBO, RT = R.get("lstm_lobo"), R.get("rain_timing")            # the two results that finished last (None while a run is still going)
    if LOBO is not None and LOBO.get("quick_mode"):
        LOBO = None                                                # a smoke-test file is not a result
    DW, SC, F04 = R.get("day_windows"), R.get("sources_checked"), R.get("floods_2004")     # added last; each is read by key inside a guarded block
    if DW is not None and DW.get("quick_mode"):
        DW = None
    # added on 4 October 2026 from the stored results (no new training); each is read by key inside a guarded block, and a smoke-test file is not a result
    KF, TC, UN, SM, PS = ((None if isinstance(d, dict) and d.get("quick_mode") else d)
                          for d in (R.get(k) for k in ("key_figure", "training_curves", "uncertainty", "same_misses", "published_scores")))
    key_fig = KF is not None and (FIG / FIGURE_KEY).exists()      # the key figure of the main conclusion is there, with its numbers
    if not key_fig:
        print(f"build_qa: figures/{FIGURE_KEY} or results/key_figure.json was not found: the sheet does not point to the key figure", file=sys.stderr)
    # where it is shown: the final report (the figure whose caption names results/key_figure.json) and the ladder slide of the final deck (its title in
    # the speaker notes), each named only if it is there
    key_in_report = report_says("F3", "key_figure.json")
    ladder_slide = next((s for s, title, _, _ in QF.deck_timing() if "ladder" in title.lower()), None)
    key_places = (" and ".join(([IN("F3")] if key_in_report else []) + ([f"slide {ladder_slide} of the final deck"] if ladder_slide else []))
                  or f"figures/{FIGURE_KEY}")
    CODE = code_facts()
    # the members' own checks as team.json records them (filled in by the team, by hand; no builder sets a check to done)
    TS = team.checks_state()
    none_done, all_done = TS["n_done"] == 0, TS["n_done"] == TS["n"]

    # ---- the experiments of 7 and 8 October 2026 (report sections 4.6 to 4.8): the numbers that group H and the updated answers quote ------------------
    # NW holds the parts that could be read (None where a results file is missing); H_OK names the questions of group H whose parts are all there. No
    # other answer refers to a question that will not exist: Hx() and Hp() name only those in H_OK.
    NW = new_work_facts(R, T)
    E1, AO, HS, S4, CK, RV, FC, P1 = (NW[k] for k in ("E1", "AO", "HS", "S4", "CK", "RV", "FC", "PAGE1"))
    NEW = new_work(R)                                                  # the days of the experiments, from the dates in the four review records
    NEW_DAYS = NEW["days"] if NEW else ""
    NEW_SHORT = NEW_DAYS.rsplit(" ", 2)[0] + " " + NEW_DAYS.rsplit(" ", 2)[1] if NEW_DAYS else "the last days"         # '7 and 8 October' (the year dropped)
    GROUPS["H"] = f"The four experiments of {NEW_SHORT} (E1, E1b, E3, E2): rain to midnight, a forecast, more rivers"
    H_NEED = dict(aorc=("AO", "S4", "HS", "CLOCK"), hindsight=("AO", "HS", "HIND", "S4", "CLOCK"), hours=("AO", "E1", "HOURS", "CLOCK"), design=("FC", "FCX", "CLOCK"), fcskill=("FC", "FCX"),
                  moved=("FC", "FCX"), shuffled=("FC", "FCX"), rivers=("RV",), holds=("AO", "FC", "RV", "E1", "S4", "PAGE1"), service=("AO", "S4", "FC", "CLOCK"),
                  newcost=("COST",), reviewed=("REV",), spent=("SPENT",))
    in_report = all(report_says("T14", e) for e in ("E1b AORC rain to midnight", "E2a GEFS forecast", "E3 sixty-three rivers"))      # the final report has the four experiments
    if not in_report:
        print("build_qa: Table 14 of the final report does not list E1b, E2a and E3: group H is left out", file=sys.stderr)
    H_OK = {k for k, need in H_NEED.items() if in_report and all(NW.get(x) for x in need) and (k != "reviewed" or NEW)}
    Hx = lambda *keys: X(*[k for k in keys if k in H_OK])              # a reference to questions of group H, only to those that will exist ('' if none does)
    Hp = lambda *keys: f" ({Hx(*keys)})" if any(k in H_OK for k in keys) else ""      # the same as an appended parenthesis
    NOT_YET = lambda verdicts: "NOT YET" if set(verdicts.values()) == {"NOT YET"} else join(sorted(set(verdicts.values())))     # {lead: verdict} -> the one verdict, or all
    for missing in sorted(set(H_NEED) - H_OK):
        print(f"build_qa: the question '{missing}' of group H was left out: a results file or key it needs is missing (see the lines above)", file=sys.stderr)
    # the fixed wordings that stand beside the numbers of the new rungs (s0_summary): the hindsight bound beside every 51% (HIND_W; '' while the review record of the hindsight check is
    # missing), what the Stage IV figure is (REALTIME_WORDING), and the caveat that it is not established whether the Stage IV archive holds first postings or later versions
    HIND_W = hindsight_clause(HS)
    hw = f" ({HIND_W})" if HIND_W else ""                                  # to go right after a 51%

    def Hi(*keys):
        """'; hindsight at most about 3 points; H1': the bound and the pointers, to go inside a parenthesis that is already open."""
        ref = Hx(*keys) if any(k in H_OK for k in keys) else ""
        return (f"; {HIND_W}" if HIND_W else "") + (f"; {ref}" if ref else "")

    Hh = lambda *keys: f" ({Hi(*keys)[2:]})" if Hi(*keys) else ""      # the same in a parenthesis of its own
    S48 = f"{pct(S4['stage4'] / S4['n'])} with Stage IV: {REALTIME_WORDING}" if S4 else ""        # the Stage IV figure and what it is
    SIDE = f"{HIND_W + '; ' if HIND_W else ''}{S48}"                       # what stands in the parenthesis after a 51%
    # more rivers lower the share of the gauged rivers' onsets warned, with fewer false warnings and the same CSI: a moved operating point (MOVED_POINT), by the test of the report
    RV_MOVED = bool(RV) and S0.moved_operating_point(RV["onset"]["lstm_wq"][1], RV["onset"]["lstm63_wq"][1], RV["false_alarms"][0] / (RV["hits"][0] + RV["false_alarms"][0]),
                                                     RV["false_alarms"][1] / (RV["hits"][1] + RV["false_alarms"][1]), RV["csi"][0], RV["csi"][1])
    S4_ARCHIVE = bool(S4) and any("first postings" in p for p in R["stage4_realtime"]["honesty_points"])
    if S4 and not S4_ARCHIVE:
        print("build_qa: results/stage4_realtime.json has no honesty point on first postings or later versions of the Stage IV archive: H1 does not state the caveat", file=sys.stderr)

    # ---- after the mid-term talk (group I): the facts are read first, so that every answer refers only to the questions of group I that will exist (Ix) ---
    REG = QF.Registry()                                                # the registry of the final report: numbers as the report and the deck print them
    FI = QI.facts(R, REG, FR)
    I_OK = QI.ok_keys(FI)
    Ix = lambda *keys: X(*[k for k in keys if k in I_OK])              # '' when none of them exists
    Ip = lambda *keys: f" ({Ix(*keys)})" if any(k in I_OK for k in keys) else ""
    proposal_day = REG.t("date_proposal_talk") if REG.has("date_proposal_talk") else ""
    GROUPS["I"] = QI.GROUP_TITLE.format(when=f" of {proposal_day}" if proposal_day else "")
    E4E, SEEDS5 = FI.get("E4E"), FI.get("SEEDS")

    items = []

    def Q(group, key, q, a, trap, where):
        """One question. key: the name other answers use to point here (X(key)); the number follows from the order of the calls."""
        assert all(i["key"] != key for i in items), f"two questions with the key {key}"
        number = f"{group}{sum(i['group'] == group for i in items) + 1}"
        where = where.replace("<code>results/", "<code>").replace("<code>code/", "<code>")           # the folders are named once, in the introduction
        items.append(dict(group=group, key=key, number=number, q=q, a=" ".join(a.split()), trap=" ".join(trap.split()), where=" ".join(where.split())))

    ids = list(LA["basins"])
    n = len(ids)
    short = lambda g: LA["basins"][g]["short"]
    split = LA["split"]
    TRAIN, VAL, TEST = (f"{name} years ({years(split[p])})" for name, p in (("training", "train"), ("validation", "val"), ("test", "test")))
    top5 = pct(sum(h["share_train_days_above"] for h in B["high_flow"].values()) / n)              # '5%': the share of training days above the level
    n_seeds = len(LA["sets"]["wq"]["seeds"])

    # ---- the all-river LSTM against the simple models ----------------------------------------------------------------------
    C, LV, ST = LA["comparison"], LA["decomposition"]["levels"], LA["decomposition"]["steps"]
    best = lambda inp, k: C[inp]["best_baseline_by_validation"][lk(k)]
    cmp_ = lambda inp, k, name=None: C[inp]["baselines"][name or best(inp, k)][lk(k)]
    family = lambda name: FAMILY[LA["baselines"][name]["model"]]
    cls = {inp: {k: reading(cmp_(inp, k)) for k in LEADS} for inp in C}
    gauge_tie = all(v == "tie" for v in cls["wq"].values())
    gauge_better = all(v == "lstm" for v in cls["wq"].values())
    wq_lstm, wq_base = [cmp_("wq", k)["lstm_nse_median"] for k in LEADS], [cmp_("wq", k)["baseline_nse_median"] for k in LEADS]
    wq_wins = [cmp_("wq", k)["n_basins_lstm_higher"] for k in LEADS]
    pers_wins = [cmp_("wq", k, "base_persistence")["n_basins_lstm_higher"] for k in LEADS]
    like_leads = {inp: [k for k in LEADS if cmp_(inp, k).get("like_for_like", True)] for inp in C}
    m2 = "met" if gauge_better else "not met"
    w_cls = cls["w"]
    w_wins = [cmp_("w", k)["n_basins_lstm_higher"] for k in LEADS]
    w_avg_only = all(v == "lstm_avg" for v in w_cls.values())      # past weather only: ahead as a seed average, the simple model inside the seed spread
    # of the two forecasts that could be run in practice (past weather; past weather + river gauge) the one with the gauge scores higher
    stronger = all(LV["wq"][lk(k)]["nse_median"] > LV["w"][lk(k)]["nse_median"] for k in LEADS)
    practice = ("the stronger of the two forecasts that could be run in practice" if stronger else "one of the two forecasts that could be run in practice")
    # the largest gap between the LSTM and the best simple model where the results file calls the comparison like-for-like (as report section 4.3)
    model_gap = max(abs(cmp_(inp, k)["lstm_nse_median"] - cmp_(inp, k)["baseline_nse_median"]) for inp in C for k in LEADS if cmp_(inp, k).get("like_for_like") is not False)
    two_sided = "two-sided" in note_with(LA["notes"], "Wilcoxon")  # how 04_lstm_all.py computes p (the default of scipy.stats.wilcoxon)
    p_kind = "two-sided" if two_sided else "see the notes of results/lstm_all.json for its kind"

    # ---- warnings (flood analysis) -----------------------------------------------------------------------------------------
    SK, FAIL, VD = FA["skill"], FA["failures"], FA["verdict"]
    crit = VD["criteria"]
    go, ca = re.search(r">= ([\d.]+) and \w+ <= ([\d.]+)", crit["GO"]), re.search(r">= ([\d.]+)", crit["CAUTION"])
    go_warned, go_false, ca_index = float(go.group(1)), float(go.group(2)), float(ca.group(1))

    def verdict_of(c):
        """GO / CAUTION / NOT YET for one stored warning table, by the criteria stored in flood_analysis.json (verdict.criteria)."""
        return "GO" if c["pod"] >= go_warned and c["far"] <= go_false else "CAUTION" if c["csi"] >= ca_index else "NOT YET"

    plain = lambda m, k, kind: SK[m]["plain"]["test"]["summary"][lk(k)][kind]
    n_high = plain("lstm_wq", 1, "all")["hits"] + plain("lstm_wq", 1, "all")["misses"]
    n_onset = {k: plain("lstm_wq", k, "onset")["hits"] + plain("lstm_wq", k, "onset")["misses"] for k in LEADS}
    on_wq = [plain("lstm_wq", k, "onset") for k in LEADS]
    on_ub = [plain("lstm_wqf", k, "onset") for k in LEADS]
    all_wq1, all_pers1 = plain("lstm_wq", 1, "all"), plain("base_persistence", 1, "all")
    later_days = n_high - n_onset[1]                               # high-flow days that are the second or a later day of an event
    peak = FAIL["peak_size"]["lstm_wq"]["all_events"]["median"]
    land = FAIL["by_basin"]["lstm_wq"]["median_by_landscape"]["tuned"]
    land_lo, land_hi = min(land, key=lambda k: land[k]["onset_pod"]), max(land, key=lambda k: land[k]["onset_pod"])

    # ---- the two results that finished last: read behind a guard; 'result pending' until the file is there and complete ---------
    try:
        L1 = LOBO["summary"]["l1"]
        nm, pos = L1["nse_median"], L1["n_basins_positive_nse"]
        loss, rng = L1["loss_from_not_seeing_the_basin"]["with_properties"], L1["in_training_seed_range_median"]
        loss_no = L1["loss_from_not_seeing_the_basin"]["without_properties"]              # the same loss for the model that reads the weather only
        bp_out, bp_in = L1["basin_properties"]["held_out"], L1["basin_properties"]["in_training"]
        lf = L1["lstm_against_forest_held_out"]["with_properties"]
        worst = max((r for r in LOBO["comparison"] if r["lead"] == 1), key=lambda r: r["loss_with_properties"])        # the basin that loses most when held out
        one_seed = "lstm_w_s0" in note_with(LOBO["notes"], "in_training =")        # 'in training' is one seed of the model of report section 4.2, not its seed average
        avg_w = LV["w"]["l1"]["nse_median"]                                        # the same model as the three-seed average (report tables 4.1 and 4.3)
        differs = one_seed and n2(avg_w) != n2(nm["in_training"])
        # the paired loss comes first wherever the result is summarised: the two medians come from different basins, the loss of the typical basin does not
        unseen_worse = loss["median_nse_loss"] > 0 and loss["n_basins_lower_when_held_out"] > n / 2
        unseen_pair = (f"the typical basin loses {n2(loss['median_nse_loss'])} NSE {LEAD_WORD[1]} (lower in {loss['n_basins_lower_when_held_out']} of {n} basins); the median "
                       f"falls from {n2(nm['in_training'])} to {n2(nm['held_out_with_properties'])}")
        unseen_short = ((f"And a river it has never seen is forecast worse: without a river gauge, from past weather and the basin properties, {unseen_pair} "
                         f"({'one seed each; ' if one_seed else ''}{X('unseen')}).") if unseen_worse and nm["held_out_with_properties"] < nm["in_training"] else
                        (f"And on a river it has never seen, forecast without a river gauge from past weather and the basin properties, its median NSE {LEAD_WORD[1]} goes from "
                         f"{n2(nm['in_training'])} to {n2(nm['held_out_with_properties'])} ({'one seed each; ' if one_seed else ''}{X('unseen')})."))
    except (KeyError, TypeError, IndexError, ValueError) as err:
        if LOBO is not None:
            print(f"results/lstm_lobo.json is not in the expected form ({err!r}): treated as pending", file=sys.stderr)
        LOBO, unseen_short = None, f"On a river it has never seen: result pending ({X('unseen')})."

    # ---- the rain-timing test (report section 4.5). The firm result is the onsets warned; NSE follows, with the basin count beside it -------
    # How a gain is judged here, as section 4.5 does: two LSTMs that differ in one input are compared by their single seeds (every seed with the
    # extra rain above every seed without it AND above every seed of the control) and by the basin count (higher in most basins).
    try:
        CL = RT["closure"]
        row = lambda fam, k: next(r for r in CL[fam]["by_lead"] if r["lead"] == k)
        r1 = row("wqn", LEADS[0])                                                                     # the LSTM with the river gauge, 1 day ahead
        r1n = row("wn", LEADS[0]) if "wn" in CL else None                                             # the same test without the river gauge
        above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                      # every single seed of a above every single seed of b
        on_clear = lambda r: all(above(r["onset_seed_range_with_nldas"], r[f"onset_seed_range_{o}"]) for o in ("base", "control"))
        share = lambda v: pct(0.0 if abs(v) < 0.005 else v).replace("-", "&minus;")
        on_gap = r1["onset_share_of_gap_closed"]                                                      # share of the gap in onsets warned (to the upper bound) closed by the rain of the issue day
        s1 = r1["share_of_gap_closed"]                                                                # the same share for the median NSE: quoted only with the basin count beside it
        stamp_gap = on_clear(r1) and 0 < on_gap < 1                                                   # the onsets warned show a time-stamp gap, clear of every seed
        both_fams = stamp_gap and r1n is not None and on_clear(r1n) and 0 < r1n["onset_share_of_gap_closed"] < 1      # ... with and without the river gauge
        on_move = lambda r: f"from {pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])}"
        on_refs = lambda r: f"the control, the same rain made one day older: {pct(r['onset_warned_control'])}; upper bound: {pct(r['onset_warned_perfect_forecast'])}"
        # NSE with the river gauge, read as the report reads it (s0_summary.nse_reading: the rule of section 4.2 as far as this file stores what
        # it needs): clear = every seed with the extra rain above every seed without it AND higher in most basins; mixed = the seeds say so,
        # the basins do not; none = the single seeds overlap
        nse_seeds = above(r1["seed_range_with_nldas"], r1["seed_range_base"])
        nse_most = r1["basins_better"] > n / 2
        nse_move = f"the median {'rises' if r1['gain'] > 0 else 'falls'} from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}"
        if nse_seeds and nse_most:
            nse_word, rt_nse = "clear", f"In NSE the gain is clear as well: {nse_move}, and the forecast is higher in {r1['basins_better']} of {n} basins"
        elif nse_seeds:
            nse_word, rt_nse = "mixed", f"In NSE the result is mixed: {nse_move}, but the forecast is higher in only {r1['basins_better']} of {n} basins"
        else:
            nse_word, rt_nse = "none", f"In NSE there is no clear difference: {nse_move}, and the single seeds overlap"
        # what is left 2 and 3 days ahead, for the model the sentences name (with the river gauge)
        closed_nse = [row("wqn", k)["share_of_gap_closed"] for k in LEADS[1:]]
        closed_on = [row("wqn", k)["onset_share_of_gap_closed"] for k in LEADS[1:]]
        left_nse, left_on = min(1.0, 1 - max(closed_nse)), min(1.0, 1 - max(closed_on))
        left_word = "almost all" if min(left_nse, left_on) >= 0.9 else "most" if min(left_nse, left_on) > 0.5 else "part"
        rt_later = (f"{leads(LEADS[1:])} {left_word} of the gap remains (with the river gauge {share(left_nse)} or more of the gap in NSE and "
                    f"{share(left_on)} or more of the gap in onsets warned)")
        # the onset warnings with that rain, as counts, and their verdict by the stored criteria (both models at the high-flow level itself)
        on_base_t, on_rain_t = (RT["models"][s]["test"]["summary"]["l1"]["onset"] for s in ("wq", "wqn"))
        rain_verdict = verdict_of(on_rain_t)
        stamp_nldas = RT["stamps"]["median"]["daymet_vs_nldas"]                                       # the rain table's day against a product stamped by UTC day
        try:                                                                                          # basin by basin (differences of stored scores, checked against the stored count)
            new_b, old_b = RT["models"]["wqn"]["test"]["basins"], RT["models"]["wq"]["test"]["basins"]
            change = {g: new_b[g]["l1"]["nse"] - old_b[g]["l1"]["nse"] for g in new_b}
            ups, downs = [v for v in change.values() if v > 0], [v for v in change.values() if v <= 0]
            if len(ups) != r1["basins_better"] or not ups or not downs:
                raise KeyError("the per-basin scores do not match basins_better")
            order = sorted(old_b, key=lambda g: old_b[g]["l1"]["nse"])
            middle = [change[g] for g in order[len(order) // 2 - 1:len(order) // 2 + 1]] if len(order) % 2 == 0 else [change[order[len(order) // 2]]]
            means = [RT["models"][s]["test"]["summary"]["l1"]["nse_mean"] for s in ("wq", "wqn")]
        except (KeyError, TypeError) as err:
            print(f"results/rain_timing.json: the basin-by-basin changes were not read ({err!r}); the answer gives the counts only", file=sys.stderr)
            change = None
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError, StopIteration) as err:
        if RT is not None:
            print(f"results/rain_timing.json is not in the expected form ({err!r}): treated as pending", file=sys.stderr)
        RT, stamp_gap, both_fams = None, False, False

    # ---- three small files that came last; each is read by key, and the sheet stands without them ----------------------------------
    try:                                                           # the day windows measured against hourly rain (check_day_windows.py)
        dm, nl = DW["day_ends"]["daymet"], DW["day_ends"]["nldas"]
        off = sorted(DW["day_ends"]["imerg"]["hours_late_in_this_check"].values())         # how far the check misplaces a day whose end is documented
        dw = dict(daymet=dm["day_ends_local_rounded"], daymet_readings=dm["range_over_readings_local"], misfits=list(dm["basins_that_do_not_fit"]),
                  nldas=nl["day_ends_local_rounded"], nldas_readings=nl["range_over_readings_local"], off=off,
                  n_readings=len(dm["median_by_reading"]), n_references=len(DW["windows"]))       # readings per product; hourly references (reanalyses)
    except (KeyError, TypeError, AttributeError) as err:
        if DW is not None:
            print(f"results/day_windows.json is not in the expected form ({err!r}): left out", file=sys.stderr)
        dw = None
    try:                                                           # what the two papers behind the reading of the rain table's day say (looked up, not yet read by the team)
        src = {s["key"]: s for s in SC["sources"]}
        sources = dict(n=len(src), v4=any("version 4" in s for s in src["thornton2021"]["what_it_says"]), shifts=bool(src["kratzert2021"]["what_it_says"]),
                       camels_unchecked="was not checked" in SC["how_to_word_it"], unread="no team member has read" in SC["what"], when=SC["checked_on"])
    except (KeyError, TypeError) as err:
        if SC is not None:
            print(f"results/sources_checked.json is not in the expected form ({err!r}): left out", file=sys.stderr)
        sources = None
    pending = []
    if LOBO is None:
        pending.append("the leave-one-basin-out test (<code>results/lstm_lobo.json</code>, written by <code>code/05_lobo.py</code>)")
    if RT is None:
        pending.append("the rain-timing test (<code>results/rain_timing.json</code>, written by <code>code/10_rain_timing.py</code>)")
    for p in pending:
        print("result pending:", re.sub(r"<[^>]+>", "", p), file=sys.stderr)

    # ======================================================================================================================
    # A. the four things every member must be able to explain
    # ======================================================================================================================
    mem = LS["memory"]
    hist, base_hist = LS["tuning"]["chosen"]["warmup"], LS["tuning"]["base"]["warmup"]
    ctx = [c for c in dict.fromkeys((mem["contexts_days"][0], hist, base_hist)) if c in mem["contexts_days"]]
    mem_nse = lambda inp, c: mem["test_nse_by_lead"][inp][str(c)][0]
    paid = (f"It did not make the LSTM the better forecaster: with the river gauge there is no clear difference from the best simple model ({X('beat')})." if gauge_tie else
            f"Whether it paid off: with the river gauge {say(cls['wq'])} ({X('beat')}).")
    Q("A", "why_lstm", "Why an LSTM, and not just a regression?",
      f"""A river has a memory: today&rsquo;s flow depends on the rain of the past days to months, which the basin stores as wet soil and groundwater. An LSTM (a long
      short-term memory network) reads the days one after another and carries a memory from each day to the next, so it can learn that store from the data, which is
      how LSTMs are used for river flow {cite('kratzert2018')}. The simple models are handed the store ready-made, as rain sums over fixed numbers of days. The
      memory is used: {LEAD_WORD[1]} with past weather only, the one-river LSTM scores NSE {join(n2(mem_nse('w', c)) for c in ctx)} when it may read only the last
      {join(ctx)} days. {paid}""",
      f"""&ldquo;Because deep learning is more accurate.&rdquo; {'Our own numbers do not show that' if not gauge_better else 'That is a result to show, not a reason'}.
      The reason for the choice is the river&rsquo;s memory; whether it paid off is a result, and the result with the river gauge is
      {'no clear difference from ' + join(sorted({family(best('wq', k)) for k in LEADS})) if gauge_tie else 'that ' + say(cls['wq'])}.""",
      f"""{W(('5.2', 'one-river tuning'), ('7.1', 'no clear difference'), 'T5')} &middot; <code>results/lstm_single.json</code> (memory), <code>results/lstm_all.json</code> (comparison) &middot;
      <code>code/03_lstm.py</code>, <code>code/04_lstm_all.py</code>""")

    BM = B["models"]
    SB = {(r["model"], r["lead"]): r for r in B["summary_table"]}
    b_best = {k: B["best_by_validation"][lk(k)]["without_weather_forecast"] for k in LEADS}
    b_like = {k: B["best_by_validation_same_information"]["w"][lk(k)] for k in LEADS}
    gauge_word = lambda name: f"{FAMILY[BM[name]['model']]}{' with the river gauge' if BM[name]['uses_river_gauge'] and BM[name]['model'] != 'persistence' else ''}"
    by_model = {}
    for k in LEADS:
        by_model.setdefault(b_best[k]["model"], []).append(k)
    to_beat = "; ".join(join(f"{n2(b_best[k]['test']['nse_median'])} {LEAD_WORD[k]}" for k in ks) + f" ({gauge_word(m)})" for m, ks in by_model.items())
    no_fc = [m for m in BM if not BM[m]["uses_weather_forecast"]]
    top1 = max(no_fc, key=lambda m: SB[m, 1]["nse_median"])
    not_top = (f", and the validation-chosen model is not even the top test scorer {LEAD_WORD[1]} ({gauge_word(top1)}: {n2(SB[top1, 1]['nse_median'])})"
               if top1 != b_best[1]["model"] else "")
    sums = sorted(int(c[len("rain_sum"):]) for c in B["features"]["w"] if c.startswith("rain_sum"))
    trees = sorted({BM[m]["settings"]["n_trees"] for m in BM if BM[m]["model"] == "rf"})
    Q("A", "baselines", "What are the baselines, and what is the number to beat?",
      f"""{cap(word(len(BM)))} simple models, each fitted per basin and lead time. {TT_CAP} repeats today&rsquo;s gauge reading and the calendar-day average uses no
      live data; ridge regression (a weighted sum) and random forest (the average of {join(trees)} decision trees) read {len(B['features']['w'])} hand-made features of
      the past weather, such as the rain sums of the last {sums[0]} to {sums[-1]} days, with and without the river gauge and, as an upper bound, with the perfect
      weather forecast. The number to beat is the test-years median NSE of the simple model that the validation years chose: {to_beat}. Without a river gauge it is
      {join(f"{n2(b_like[k]['test']['nse_median'])}" for k in LEADS)}. {TT_CAP} alone scores {n2(SB['base_persistence', 1]['nse_median'])} {LEAD_WORD[1]} and
      {n2(SB['base_persistence', 3]['nse_median'])} {LEAD_WORD[3]}.""",
      f"""Setting the LSTM only against {TT}, which it beats in {join(pers_wins)} of {n} basins. The bar is the best simple model given the same inputs{not_top}.""",
      f"""{W(('5.2', 'Simple models'), ('5.4', 'Baselines'), 'T5', 'T17')} &middot; <code>results/baselines.json</code> (best_by_validation, summary_table) &middot; <code>code/02_baselines.py</code>,
      <code>code/features.py</code>""")

    exp, mea, cnt = FT["expected"]["past_weather_only"], FT["measured"]["past_weather_only"], ER["counts"]
    zero = lambda v: n2(0.0 if round(v, 3) == 0 else v, 3)
    clim = re.search(r"\(([a-z_, ]+):", note_with(LA["notes"], "climate averages"))
    clim_codes = [c.strip() for c in clim.group(1).split(",")] if clim else []
    clim_years = re.search(r"(\d{4}) to \w+ (\d{4})", note_with(LA["notes"], "climate averages"))
    clim_text = (f"; and {word(len(clim_codes))} of the {len(LA['static_inputs'])} basin properties are climate averages of {clim_years.group(1)}&ndash;{clim_years.group(2)}, "
                 f"years that reach into the validation and test years" if clim_codes and clim_years else "")
    hf = B["high_flow"]
    val_share, test_share = [h["share_val_days_above"] for h in hf.values()], [h["share_test_days_above"] for h in hf.values()]
    v_nse, t_nse = (LA["sets"]["wq"][p]["summary"]["l1"]["nse_median"] for p in ("val", "test"))
    no_leak = bool(note_with(ER["notes"], "No reviewer found a leak"))
    harder = (f" The validation years are also the harder sample: with the river gauge {LEAD_WORD[1]} the LSTM&rsquo;s median NSE is {n2(v_nse)} there and {n2(t_nse)} in the test "
              f"years, although nothing was chosen on the test years." if t_nse > v_nse else "")
    # how the major findings of the review were dealt with (engine_review.json: major_findings, all_fixed, all_fixed_note). A finding that is a limit
    # of the measurements (the training times) was eased, not removed: the cost comparison was NOT timed again on an idle machine.
    eased = [m for m in ER["major_findings"] if "NOT re-timed" in m] if "limit of the measurements" in ER.get("all_fixed_note", "") else []
    n_faults = cnt["major"] - len(eased)
    if ER["all_fixed"] and eased:
        fixed_long = (f"Of the {cnt['major']} major problems they did find, {word(n_faults)} were faults in the code and were fixed before the experiments they affect; "
                      f"the {'other is' if len(eased) == 1 else 'others are'} a limit of the measurements, not a fault: training times taken on a busy laptop are not a "
                      f"clean cost measure. That was eased by logging the load and by giving the cost ratios as a range; the cost comparison was not timed again "
                      f"on an idle machine ({X('cost')})")
        fixed_short = (f"every fault in the code was fixed before the affected experiments ran; {word(len(eased))} major finding, the training times taken on a busy "
                       f"laptop, is a limit of the measurements that was eased, not removed")
    elif ER["all_fixed"]:
        fixed_long = f"The {cnt['major']} major problems they did find were fixed before the experiments they affect"
        fixed_short = "all fixed before the affected experiments ran"
    else:
        fixed_long, fixed_short = f"The {cnt['major']} major problems they did find are not all fixed yet", "not all fixed yet"
    Q("A", "fair", "How is the test kept fair?",
      f"""By two rules written once in the shared code. A forecast belongs to the period of the day it forecasts, not of the day it was made; and each period has one
      job: the {TRAIN} fit the weights, the scalers and the high-flow levels, the {VAL} choose the settings and where training stops, the {TEST} are only scored. A
      check on made-up rivers with a known answer backs this: a model that reads the weather up to day t can reach an NSE of at most {slash(exp)} for 1 / 2 / 3
      days ahead and measures {slash(mea, zero)}; a leak would push it higher. {cap(word(len(ER['reviews'])))} AI reviewer agents then tried to break that code:
      {count(cnt['checks_passed'])} checks passed{' and no leak was found' if no_leak else ''}. {fixed_long}. The limits of the test itself: all {n} basins are in
      all three periods, so it is a test of later years of rivers already seen{clim_text}.""",
      f"""&ldquo;The test years were never touched&rdquo; or &ldquo;scored once&rdquo;. They were scored again at every re-scoring pass; what is true is that nothing
      was chosen from them.""",
      f"""{W(('4.2', 'Leakage controls'), ('5.3', 'Split by target day'), ('6.3', 'Independent review'), 'T19')} &middot; <code>results/fair_test.json</code>, <code>results/engine_review.json</code> &middot; <code>code/flood.py</code>,
      <code>code/test_flood.py</code>""")

    fail_new = (f" The inputs added on {NEW_SHORT} lift the share of onsets warned, to {pct(AO['with_'][1])} {LEAD_WORD[1]} with the rain up to midnight{hw}, but not the verdict, "
                f"which stays {NOT_YET(AO['verdict_onset'])}{Hp('holds')}." if AO else "")
    Q("A", "fails", "Where does the model fail?",
      f"""On the warnings that matter: the onsets, high-flow days forecast while the river was still below the high-flow level. {cap(LEAD_WORD[1])} with the river
      gauge the LSTM warns of {pct(on_wq[0]['pod'])} of the {count(n_onset[1])} onsets, and {pct(on_wq[0]['far'])} of its onset warnings are false;
      {leads(LEADS[1:])} it warns of {join(pct(o['pod']) for o in on_wq[1:])}. It fails most in the {land_lo.lower()}, where the median basin warns of
      {pct(land[land_lo]['onset_pod'])} of its onsets ({land_hi.lower()}: {pct(land[land_hi]['onset_pod'])}). {'It forecasts peaks too low: in' if peak < 1 else 'In'} the median event the
      forecast made the day before is {pct(peak)} of the observed peak. {unseen_short}{fail_new}""",
      f"""Quoting the overall share warned ({pct(all_wq1['pod'])}) as flood-warning skill. {count(later_days)} of the {count(n_high)} high-flow days are the second or
      a later day of an event, and {TT} warns of {pct(all_pers1['pod'])} with nothing fitted."""
      + "".join(f" (The report also counts {count(n_onset[k])} onsets {LEAD_WORD[k]}: the same number by chance, not a copy error.)"
                for k in LEADS[1:] if n_onset[k] == later_days),
      f"""{W(('2.3', 'A flood onset'), ('7.3', 'Where the warnings fail'), 'F4', 'T16')} &middot; <code>results/flood_analysis.json</code> (skill, failures), <code>results/flood_events.csv</code>,
      <code>results/lstm_lobo.json</code> &middot; <code>code/06_flood_analysis.py</code>""")

    # ======================================================================================================================
    # B. data and set-up
    # ======================================================================================================================
    AUD = T["basin_audit"]
    per_land = {}
    for b in DA["selected"]:
        per_land[b["landscape"]] = per_land.get(b["landscape"], 0) + 1
    each = f"{word(next(iter(per_land.values())))} from each" if len(set(per_land.values())) == 1 else join(per_land.values()) + " from"
    Q("B", "basins", f"Why these {n} basins?",
      f"""They were chosen by a rule, not by hand. A basin is the land area whose rain drains to one river gauge. From the {count(DA['n_all_camels'])} basins of
      CAMELS-US {cite('newman2015')} the rule takes the
      {count(DA['n_candidates'])} in two south-eastern river regions that are humid, fed by rain and of middle size ({count(AUD.area_gages2.min())} to
      {count(AUD.area_gages2.max())} km&sup2;). It rejects {count(DA['n_with_gaps'])} with gaps in the flow record, sorts the {count(DA['n_complete'])} complete ones
      into {word(len(per_land))} landscapes and takes {each}, evenly spaced in rank from the smallest to the largest area. No flow statistic and no model score enters
      the rule. The number {n} was fixed in our plan; it is small, and we say so ({X('untested')}).""",
      f"""&ldquo;We picked representative basins.&rdquo; Nobody judged that: the rule is mechanical. And with {word(next(iter(per_land.values())))} basins per landscape,
      which also differ in size and rainfall, no difference between landscapes can be called a cause.""",
      f"""{W(('4.2', 'Basin selection'), 'F5')} &middot; <code>results/data_audit.json</code>, <code>results/basin_audit.csv</code> &middot; <code>code/01_data_audit.py</code>""")

    period = DA["period"]
    pers_basin = {g: v["l1"]["nse"] for g, v in BM["base_persistence"]["test"]["basins"].items()}
    slow = max(pers_basin, key=pers_basin.get)
    Q("B", "period", f"Why {period[0][:4]}&ndash;{period[1][:4]}, and why {n_years(split['train'])} / {n_years(split['val'])} / {n_years(split['test'])} years?",
      f"""It is the period of the CAMELS-US files; we use its {n_years(period)} full water years (October to September), {date(period[0])} to {date(period[1])}:
      {count(AUD.n_days.max())} days per basin, with no missing day in any of the {n} basins. The first {n_years(split['train'])} years train, the next
      {n_years(split['val'])} validate and the last {n_years(split['test'])} test, in time order, so every test day lies after everything the model learned from. The
      weak part is the validation period: {word(n_years(split['val']))} years, in which the share of days above the high-flow level runs from
      {pct(min(val_share), 1)} to {pct(max(val_share), 1)} between basins (test years: {pct(min(test_share), 1)} to {pct(max(test_share), 1)}), so settings
      chosen there are fair but noisy.{harder}""",
      f"""&ldquo;We could as well have split the days at random.&rdquo; Neighbouring days are nearly copies of each other ({TT} alone reaches NSE
      {n2(pers_basin[slow])} in {river(short(slow))}), so a random split would put the neighbours of every test day into training: a leak.""",
      f"""{W(('5.3', 'Split by target day'), ('4.1', 'CAMELS-US'))} &middot; <code>results/data_audit.json</code>, <code>results/baselines.json</code> (split, high_flow) &middot;
      <code>code/flood.py</code>, <code>code/camels.py</code>""")

    VC = LS["variance_concentration"]["val"]
    fast = min(pers_basin, key=pers_basin.get)
    wq_basin = [v["l1"]["nse"] for v in LA["sets"]["wq"]["test"]["basins"].values()]
    Q("B", "nse", "What does NSE mean, and why can it mislead?",
      f"""NSE, the Nash-Sutcliffe efficiency {cite('nash1970')}, sets the squared error of the forecast against the squared swings of the flow around its average: 1 is
      a perfect forecast, 0 is no better than always forecasting the average flow, below 0 is worse. It can mislead in three ways that our results show. A few flood
      days carry it: in the validation years of the {river(LS['basin']['short'])} River, {word(len(VC['days']))} days hold
      {pct(VC['share_of_variance_in_top_days'])} of those squared swings. A slow river flatters every model: {TT} scores {n2(pers_basin[slow])} in
      {river(short(slow))} and {n2(pers_basin[fast])} in {river(short(fast))} with nothing fitted. And a fair NSE says little about warnings: the LSTM with the river
      gauge has a median NSE of {n2(wq_lstm[0])} {LEAD_WORD[1]} and still warns of only {pct(on_wq[0]['pod'])} of the onsets.""",
      f"""&ldquo;NSE {n2(wq_lstm[0])} means {pct(wq_lstm[0])} accurate.&rdquo; NSE is not a share of correct forecasts, and the median of {n} basins hides a range
      from {n2(min(wq_basin))} to {n2(max(wq_basin))}.""",
      f"""{W(('6.3', 'Metrics.'), 'T5')} &middot; <code>results/lstm_single.json</code> (variance_concentration), <code>results/baselines.json</code>,
      <code>results/lstm_all.json</code> &middot; <code>code/flood.py</code> (nse)""")

    # the median over the basins against the mean: both are stored for the main comparison; two stored cases show how one basin moves a median
    wq_mean, base_mean = LA["sets"]["wq"]["test"]["summary"]["l1"]["nse_mean"], b_best[1]["test"]["nse_mean"]
    mv1 = MC["comparison"]["val"]["l1"]
    mbig = mv1["largest_basin_difference"]
    drag = (f"On never-seen rivers, for example, one basin at NSE {n2(worst['held_out_with_properties'])} makes the mean loss {n2(loss['mean_nse_loss'])} where the "
            f"median loss is {n2(loss['median_nse_loss'])} ({X('unseen')}). " if LOBO is not None else "")
    seed_spread1 = cmp_("wq", 1)["lstm_seed_median_max"] - cmp_("wq", 1)["lstm_seed_median_min"]
    same_story = cls["wq"][1] == "tie" and abs(wq_mean - base_mean) < seed_spread1                 # by the mean, too, the two lie closer than the seed spread
    main_text = (f"For our main result the choice {'does not matter' if same_story else 'matters'}: with the river gauge {LEAD_WORD[1]} the LSTM and the best simple model "
                 f"score {n2(wq_lstm[0])} and {n2(wq_base[0])} in the median, {n2(wq_mean)} and {n2(base_mean)} in the mean"
                 + (f", both closer than the seed spread ({n2(seed_spread1)})." if same_story else "."))
    cases = [f"in the Transformer comparison the gap on the validation years {LEAD_WORD[1]} is {n2(abs(mv1['nse_median_transformer_minus_lstm']), 3)} NSE with "
             f"{river(mbig['basin'])} and {n2(abs(mbig['nse_median_transformer_minus_lstm_without_it']), 3)} without it ({X('transformer')})"]
    if RT is not None:
        cases.append(f"in the rain-timing test the median rises by {signed(r1['gain'])} while the median of the {n} basin changes is "
                     f"{signed(r1['median_basin_gain'])} ({X('typical')})")
    count_text = f" ({r1['basins_better']} of {n} in the rain-timing test)" if RT is not None else ""
    Q("B", "median", f"Why the median of {n} basins and not the mean? How much can one basin move it?",
      f"""NSE has no lower end, so one failed basin can pull a mean far down, while a median still says what the middle basin does. {drag}{main_text} The price of the
      median: with {n} basins it is the average of the {n // 2}th and {n // 2 + 1}th value, so one or two basins near the middle can move it. {cap(word(len(cases)))}
      {'cases' if len(cases) > 1 else 'case'} in our results: {'; and '.join(cases)}. So wherever a claim rests on a median, the basin count belongs beside it.""",
      f"""&ldquo;The median is robust, so nothing more is needed.&rdquo; It is robust against one extreme basin, not against a move of the two in the middle. Quote
      the basin count with it{count_text}.""",
      f"""{W(('6.3', 'median over the 12 basins'), ('7.1', 'Transformer'), 'T5')} &middot; <code>results/lstm_all.json</code> (sets: summary), <code>results/baselines.json</code>
      (best_by_validation), <code>results/model_comparison.json</code> (comparison, notes), <code>results/rain_timing.json</code> (closure) &middot;
      <code>code/flood.py</code> (evaluate)""")

    ev = FA["events"]
    few = min(ev["by_basin"], key=ev["by_basin"].get)
    share_above = sum(h["share_train_days_above"] for h in hf.values()) / n
    Q("B", "level", f"Why the {n2(100 * (1 - share_above), 0)}th percentile? Is a high-flow day a flood?",
      f"""A high-flow day is a day with flow above the level that was exceeded on {top5} of days in the training years. Each basin has its own level, computed from the
      training years only, so nothing from the test years enters it. It is a statistical level, not an official flood stage: our &lsquo;flood warnings&rsquo; are
      warnings of high-flow days. At this level the test years hold {count(n_high)} high-flow days in {count(ev['n_events'])} events,
      {min(ev['by_basin'].values())} to {max(ev['by_basin'].values())} per basin: enough to score the {n} basins together, few for a single basin (in
      {river(few)} one event moves the share of onsets warned by about {n2(100 / ev['by_basin'][few], 0)} percentage points). No other level was tested.""",
      f"""&ldquo;We predict floods.&rdquo; We predict days above a basin&rsquo;s own top-{top5} level; whether such a day floods anything, our data do not say.""",
      f"""{W(('2.3', 'high-flow level (Q95)'), ('4.2', 'Labels are computed'), 'T18')} &middot; <code>results/flood_analysis.json</code> (events), <code>results/baselines.json</code> (thresholds, high_flow)
      &middot; <code>code/flood.py</code> (thresholds)""")

    only3 = like_leads["wqf"] == [LEADS[-1]]
    fallen = (f"part of the gain is rain that had already fallen but was not yet in the model&rsquo;s inputs, {about(on_gap)} of the gap in onsets warned "
              f"({X('rainday', 'timing')})" if stamp_gap else
              f"some of the rain that the upper bound reads had already fallen when the forecast was made ({X('rainday')})"
              + (f"; our test did not show how much of the gain that explains ({X('timing')})" if RT is not None else ""))
    # with the perfect weather forecast the two kinds of model read the same DAYS of weather only at the lead time(s) the results file marks as
    # like-for-like, and even there not the same variables (the features of the simple models: baselines.json; the LSTM: all of flood.FORCING)
    ahead = sorted({c.split("_ahead")[0] for c in B["features"]["wf"] if "_ahead" in c})
    ahead_words = join({"rain": "rain", "temp": "temperature"}.get(a, a) for a in ahead)
    vars_text = (f"the LSTM reads all {word(len(FORCING))} weather variables of those days and the simple models only {ahead_words}" if 0 < len(ahead) < len(FORCING) else "")
    like_text = (f"against the simple models, both read the same days of weather only {leads(like_leads['wqf'])}, because the LSTM reads the weather of all "
                 f"{word(len(LEADS))} coming days at every lead time and the simple models only up to the day being forecast" if only3 else
                 f"against the simple models, both read the same days of weather {leads(like_leads['wqf'])}") + (f" (and even there {vars_text})" if vars_text else "")
    # a real forecast has errors: one was tested on 8 October 2026 (the control member of a public reforecast); before that the answer said that none had been
    real_fc = (f"A real weather forecast has errors, so a real system lands in between: the one we tested, a public forecast (one ensemble member), lifts the onsets warned "
               f"{leads(LEADS[1:])} from {slash([FC['rain'][k] for k in LEADS[1:]], pct)} with the rain to midnight alone to {slash([FC['fc'][k] for k in LEADS[1:]], pct)} with the "
               f"forecast beside it, where the perfect rain of those days gives {slash([FC['perfect'][k] for k in LEADS[1:]], pct)}{Hp('fcskill', 'design')}." if FC else
               "A real weather forecast has errors, so a real system would land in between; we did not test one and cannot say where.")
    Q("B", "upper", "What is the &lsquo;perfect weather forecast&rsquo;, and why show something nobody can have?",
      f"""It is the actual weather recorded for the days being forecast, given to the model as if it were a forecast without error. Nobody can have it in practice: it
      is an upper bound that shows what the weather of those days is worth. With it the LSTM with the river gauge goes from a median NSE of {n2(LV['wq']['l1']['nse_median'])}
      to {n2(LV['wqf']['l1']['nse_median'])} {LEAD_WORD[1]} and from {pct(on_wq[0]['pod'])} to {pct(on_ub[0]['pod'])} of onsets warned. {real_fc}
      Two cautions: {LEAD_WORD[1]}, {fallen}; and {like_text}.""",
      f"""&ldquo;With a weather forecast we would reach these scores.&rdquo; The upper bound reads the true weather. Even so it misses
      {pct(1 - on_ub[0]['pod'])} of the onsets {LEAD_WORD[1]}, and {pct(on_ub[0]['far'])} of its onset warnings are false.""",
      f"""{W(('5.4', 'Ablation by information'), 'T5', 'T15', ('T18', 'Upper bound'))} &middot; <code>results/lstm_all.json</code> (decomposition, comparison, notes) &middot;
      <code>code/flood.py</code>, <code>code/features.py</code>""")

    im = SD["rain_check_summary"]["imerg"]
    sd_notes = " ".join(SD["notes"])
    utc_end = re.search(r"(\d\d:\d\d) local standard time", sd_notes)
    utc_other = re.search(r"\((\d\d:\d\d) for ([^)]+)\)", sd_notes)                   # the one basin in another time zone
    not_checked = "not checked against the Daymet documentation" in sd_notes
    if RT is not None:
        dn = stamp_nldas
        second = (f", and against a second rain product stamped by UTC day (NLDAS) the pattern is the same: {n2(dn['same_day'])} for the same day, "
                  f"{n2(dn['day_before'])} for the day before, {n2(dn['day_after'])} for the day after")
    else:
        second = ""
    if dw:                                                         # the day windows measured against hourly rain
        misfit = f"; {join(dw['misfits'])} {'does' if len(dw['misfits']) == 1 else 'do'} not fit" if dw["misfits"] else ""
        measured = (f" A check against hourly rain from {word(dw['n_references'])} weather reanalyses puts an hour on it: the table&rsquo;s &lsquo;day d&rsquo; ends at about "
                    f"{dw['daymet']} local standard time on day d ({word(dw['n_readings'])} readings: {dw['daymet_readings'][0]} to {dw['daymet_readings'][1]}{misfit}); that check is itself good to about "
                    f"{n2(dw['off'][0], 0)} to {n2(dw['off'][-1], 0)} hours.")
    else:
        measured = ""
    # how the reading is known: an inference from our own data; what the two papers say is recorded in sources_checked.json (decision of the lead:
    # report and question sheet both use its 'how_to_word_it')
    if sources:
        inference = (f"This reading is an inference from our own data, not a documented property of the table. It is consistent with a time-of-observation effect that "
                     f"the Daymet authors describe and correct {'from version 4 on' if sources['v4'] else 'in a later version'} {cite('thornton2021')}"
                     + (f", and with the one-day shifts reported between the rain products of CAMELS {cite('kratzert2021')}" if sources["shifts"] else "")
                     + ("; the documentation of the Daymet version inside CAMELS was not checked." if sources["camels_unchecked"] else "."))
        source_trap = (f" And do not say &lsquo;the Daymet documentation says so&rsquo;: the reading is our inference. What the {word(sources['n'])} papers say was looked up by Claude "
                       f"Code on {date(sources['when'])}" + ("; no team member has read them yet (list at the end)." if sources["unread"] else "."))
    elif not_checked:
        inference = "This reading is an inference from these correlations; it was not checked against the Daymet documentation."
        source_trap = " And the morning reading is inferred from these correlations; it was not checked against the Daymet documentation."
    else:
        inference, source_trap = "This reading is an inference from the data.", ""
    whole_day = f" With the whole day up to midnight, from {AORC_WORDING}, they rise to {pct(AO['with_'][1])}{Hh('aorc')}." if AO else ""
    had_fallen = (f"{cap(LEAD_WORD[1])} part of it had fallen: given rain stamped up to the evening of the issue day, the onsets warned rise {on_move(r1)}, "
                  f"{about(on_gap)} of the gap to the upper bound ({X('timing')}).{whole_day}" if stamp_gap else
                  f"{cap(LEAD_WORD[1])} part of it had fallen and was only stamped on the next day.")
    Q("B", "rainday", "When did the rain of &lsquo;day d&rsquo; in your rain table fall?",
      f"""Not during calendar day d. The data indicate that the gauge-based rain table (Daymet) stamps rain by a day that ends in the morning, as a table built from
      rain gauges that observers read in the morning would: the rain stamped &lsquo;day d&rsquo; fell mostly between the morning of day d&minus;1 and the morning of
      day d. The table&rsquo;s rain of day d matches satellite rain of UTC day d ({n2(im['median_correlation'])}) and of UTC day d&minus;1
      ({n2(im['median_correlation_satellite_one_day_earlier'])}) about equally, and of UTC day d+1 hardly at all
      ({n2(im['median_correlation_satellite_one_day_later'])}; correlations, medians of {n} basins){second}.{measured} So at the end of day t, when a forecast is
      issued, the rain of the last half day and more has fallen but is not yet in the model&rsquo;s inputs. It appears under day t+1, inside what the &lsquo;perfect
      weather forecast&rsquo; models read. {inference}""",
      f"""&ldquo;The model misses floods because the rain had not fallen yet.&rdquo; {had_fallen} Say &lsquo;rain that was not yet in the model&rsquo;s
      inputs&rsquo;.{source_trap}""",
      f"""{W(('2.3', 'filed under the next day'), ('4.2', 'Clocks'), 'F1', ('2.2', 'Daymet V4'))} &middot; <code>results/satellite_data.json</code> (rain_check_summary, notes),
      <code>results/rain_timing.json</code> (stamps)"""
      + (", <code>results/day_windows.json</code> (day_ends)" if dw else "") + (", <code>results/sources_checked.json</code>" if sources else "")
      + """ &middot; <code>code/10_rain_timing.py</code>, <code>code/fetch_satellite.py</code>"""
      + (", <code>code/check_day_windows.py</code> &middot; <code>figures/day_windows.png</code>" if dw else ""))

    # ======================================================================================================================
    # C. the model and its training
    # ======================================================================================================================
    cfg = LA["config"]
    spread1 = {inp: LS["comparison"][inp]["verdict"]["l1"]["seed_spread_used"] for inp in ("w", "wq")}
    older = {inp: mem_nse(inp, base_hist) - mem_nse(inp, hist) for inp in ("w", "wq")}
    adds = {inp: older[inp] > spread1[inp] for inp in older}
    mem_text = (f"weather older than {hist} days "
                + (f"still adds {n2(older['w'])} NSE without the river gauge" if adds["w"] else "adds nothing clear without the river gauge")
                + (f" and {n2(older['wq'])} with it" if adds["wq"] else " and nothing clear with it"))
    default = "history length" in note_with(LA["notes"], "engine defaults")

    # the network itself: fixed properties of code/flood.py (checked by code_facts), with the settings of results/lstm_all.json
    props = LA["static_inputs"]
    engine = Cfg()                                                 # the engine's defaults, for the one setting no results file stores (the gradient clip)
    n_weights = next(c for c in LA["setting_check"]["table"] if c["chosen"])["n_params"]             # the model of the setting check reads the river gauge
    arch = ((f"One LSTM layer of {cfg['hidden']} memory cells" if CODE["one_layer"] else f"An LSTM of {cfg['hidden']} memory cells") + " reads the days one after another"
            + (f"; after each day a linear layer turns the cell values into {word(len(LEADS))} numbers, the flow {join(LEADS)} days ahead" if CODE["linear_head"] else "")
            + f" ({count(n_weights)} weights with the river gauge).")
    inputs_text = (f"The daily inputs are the {word(len(FORCING))} weather variables, the {word(len(props))} basin properties and, where it is used, the river gauge"
                   + (" as log(1 + flow)" if CODE["log_flow"] else "")
                   + "; each is put on one scale with its mean and standard deviation (weather and flow: of the training years; the properties: over the basins).")
    loss_text = ((f"The loss, the error that training reduces, is the squared error of all {word(len(LEADS))} lead times together, with each basin&rsquo;s error divided "
                  f"by (the standard deviation of that basin&rsquo;s flow + 0.1 &times; the standard deviation of all basins&rsquo; flow) squared. That is close to "
                  f"dividing by the basin&rsquo;s own flow variance, so wet and dry rivers count about alike.") if CODE["loss_per_basin"] else
                 "The loss, the error that training reduces, is a squared error of the flow.")
    # the published name of this loss (published_scores.json records its form and its constant as read at the source); said only if the constant is ours
    try:
        nse_loss = CODE["loss_per_basin"] and abs(PS["sources"]["kratzert2019"]["loss_epsilon"] - 0.1) < 1e-9 and "NSE loss" in PS["sources"]["kratzert2019"]["loss"]
    except (KeyError, TypeError):
        nse_loss = False
    if nse_loss:
        loss_text += (f" It has the form of the basin-averaged NSE loss of the literature {cite('kratzert2019')}, with the same constant: a weighted squared error, "
                      f"not the NSE score itself.")
    share_text = (f" So {word(n)} rivers share one loss in this way: every update draws {cfg['batch']} sequences of {count(cfg['warmup'] + cfg['block'])} days at "
                  f"random from all basins, and this weight is all that differs between the rivers." if CODE["random_sequences"] and CODE["loss_per_basin"] else "")
    optim = ((f"The optimiser is Adam with a learning rate of {cfg['lr']:g}" if CODE["adam"] else f"The learning rate is {cfg['lr']:g}")
             + (f"; gradients are clipped to a norm of {engine.clip:g}" if CODE["clip"] and engine.clip else "")
             + (f"; a dropout of {cfg['dropout']:g} stands between the LSTM and the output layer" if CODE["dropout_before_head"] else "") + ".")
    zero = " A forecast below zero is set to zero." if CODE["not_below_zero"] else ""
    Q("C", "network", f"Describe the network exactly: layers, loss, optimiser. How do {word(n)} rivers share one loss?",
      f"""{arch} {inputs_text} {loss_text}{share_text} {optim} Training runs for at most {count(cfg['updates'])} updates and stops early on the validation years
      ({X('settings')}).{zero} The one-river LSTM ({IN(('5.2', 'one-river tuning'), 'T17')}) is the same network with the settings its own tuning chose.""",
      f"""&ldquo;It minimises the squared error&rdquo; or &ldquo;it is trained on NSE&rdquo;. Neither is exact: the squared error is weighted per basin, or the
      wettest rivers would dominate it, and NSE only decides where training stops. And it is not a deep network: one layer of {cfg['hidden']} cells; for the
      all-river model nothing larger was tried ({X('settings')}).""" if CODE["loss_per_basin"] and CODE["one_layer"] else
      f"""Calling it a deep network: for the all-river model nothing larger than {cfg['hidden']} cells was tried ({X('settings')}).""",
      f"""{W(('5.2', 'Equation (1)'), 'T17')} &middot; <code>results/lstm_all.json</code> (config, setting_check, notes) &middot; <code>code/flood.py</code> (LSTMNet,
      Cfg, fit)""")

    Q("C", "seq", "Why sequence-to-sequence training, and why a warm-up year?",
      f"""Sequence-to-sequence means that the model gives a forecast on every day of a sequence, not only at its end. One training sequence is
      {count(cfg['warmup'] + cfg['block'])} days of one basin: the model first reads {cfg['warmup']} days only to fill its memory, which
      starts empty (the warm-up), and then, on each of the next {cfg['block']} days, gives the flow {join(LEADS)} days ahead. So one pass yields {cfg['block']}
      &times; {len(LEADS)} scored forecasts instead of one, and the model is trained the way it is used: reading on, day after day. How much history counts, the
      memory test of the one-river LSTM says: {LEAD_WORD[1]}, {mem_text}."""
      + (f" The year of warm-up in the all-river model is the engine&rsquo;s default and was not tuned." if default else ""),
      f"""&ldquo;A year of warm-up is needed because rivers remember a year.&rdquo; We did not show that: for the one-river LSTM the validation years chose
      {hist} days of history{', and the year in the all-river model is an untuned default' if default else ''}.""",
      f"""{W(('5.2', '365 warm-up'), 'T17')} &middot; <code>results/lstm_single.json</code> (memory, tuning), <code>results/lstm_all.json</code> (config, notes) &middot;
      <code>code/flood.py</code> (Cfg, fit)""")

    tun, chk = LS["tuning"], LA["setting_check"]
    knobs, ch = tun["per_knob"], tun["chosen"]
    edge = [k for k, v in knobs.items() if v["chosen"] in (min(v["values"]), max(v["values"]))]
    cand = chk["table"]
    chosen = next(c for c in cand if c["chosen"])
    gap = chosen["val_nse_mean"] - max(c["val_nse_mean"] for c in cand if not c["chosen"])
    spread = max(chk["chosen_val_nse_mean_by_seed"]) - min(chk["chosen_val_nse_mean_by_seed"])
    hid, lrs = sorted({c["hidden"] for c in cand}), sorted({c["lr"] for c in cand})
    cap_note = note_with(LA["notes"], "capped")
    cores = re.search(r"\((\d+) CPU cores, no GPU\)", cap_note)
    laptop = f"the laptop ({word(int(cores.group(1)))} processor cores, no graphics card)" if cores else "the laptop"
    capped = bool(cap_note)
    size_text = (f"{chosen['hidden']} cells is a cap set by {laptop}: larger models were not tried" if capped else f"{chosen['hidden']} cells is the largest size we tried")
    hk = knobs["hidden"]
    edge_text = (f"; {word(len(edge))} of these values are the smallest or largest we tried, so a better setting may lie outside the ranges" if edge else "")
    one_every = tun["final_budget"]["eval_every"]                  # the one-river LSTM is checked more often than the all-river model
    every_text = (f"every {cfg['eval_every']} updates (every {one_every} for the one-river LSTM)" if one_every != cfg["eval_every"] else f"every {cfg['eval_every']} updates")
    Q("C", "settings", f"How were the settings chosen, and why {chosen['hidden']} memory cells?",
      f"""On the validation years only; the test years chose no setting. For the one-river LSTM {word(len(knobs))} settings were changed one at a time
      ({len(tun['runs'])} tuning runs): {ch['hidden']} memory cells, learning rate {ch['lr']:g}, dropout {ch['dropout']:g} and {ch['warmup']} days of
      history{edge_text}. For the all-river LSTM {word(len(cand))} candidates were compared ({join(hid[:-1])} or {hid[-1]} cells, learning rate
      {join(f'{v:g}' for v in lrs[:-1])} or {lrs[-1]:g}): the chosen one leads by {n2(gap, 3)} NSE, {'less' if gap < spread else 'more'} than the {n2(spread, 3)} by
      which {word(len(chk['chosen_val_nse_mean_by_seed']))} seeds of one setting differ, so the choice
      {'is not a finding' if gap < spread else 'stands clear of the seed spread'}. {cap(size_text)}. Where training stops is chosen the same way:
      {every_text} the model is scored on the validation years, the best state is kept, and training ends after {cfg['patience']} checks without
      improvement (early stopping).""",
      f"""&ldquo;{chosen['hidden']} was the best size.&rdquo;
      {'The candidates differ by less than the seed spread' if gap < spread else 'It was the best of the ' + word(len(cand)) + ' candidates; larger ones were not tried'}.
      In the one-river tuning the validation scores by size were {join(n2(v) for v in hk['val_nse_mean'])} for {join(hk['values'])} cells.""",
      f"""{W(('5.2', 'three candidates'), 'T17')} &middot; <code>results/lstm_single.json</code> (tuning), <code>results/hparam_log.csv</code>,
      <code>results/lstm_all.json</code> (setting_check, notes) &middot; <code>code/03_lstm.py</code>, <code>code/04_lstm_all.py</code>""")

    c_g1, c_f1, c_w1 = cmp_("wq", 1), cmp_("wf", 1), cmp_("w", 1)
    avg_above = c_f1["lstm_nse_median"] > c_f1["lstm_seed_median_max"]
    flatter = (f" (with the perfect weather forecast, the upper bound, and no gauge: {n2(c_f1['lstm_nse_median'])} for the average, "
               f"{n2(c_f1['lstm_seed_median_min'])} to {n2(c_f1['lstm_seed_median_max'])} for its seeds)" if avg_above else "")
    Q("C", "seeds", "What is the seed spread, and what do you do with it?",
      f"""A seed is one training of a model from its own random start; the seed spread is how far the scores of several such trainings lie apart. For the all-river
      LSTM with the river gauge, {LEAD_WORD[1]}, the median NSE runs from {n2(c_g1['lstm_seed_median_min'])} to {n2(c_g1['lstm_seed_median_max'])} over its
      {word(n_seeds)} seeds.""" + (f""" That stored set keeps {word(n_seeds)}; every set of scripts {SEEDS5['scripts']} and of E4b and E4e has
      {word(SEEDS5['n5'])}{Ip('moreseeds')}.""" if SEEDS5 else "") + f""" A difference smaller than the seed spread is &lsquo;no clear difference&rsquo;, because one
      more training could land on either side ({X('clear')}). Two cautions: our LSTM score is the day-by-day average forecast of its seeds, which usually scores a little higher than a single seed{flatter}
      and so flatters the LSTM against single-model baselines; and the random forest was fitted from one seed, so its own spread is not known.""",
      f"""&ldquo;With past weather only, the LSTM wins in {c_w1['n_basins_lstm_higher']} of {n} basins.&rdquo; That is the {word(n_seeds)}-seed average. Single seeds
      win in {min(c_w1['n_basins_lstm_higher_by_seed'])} to {max(c_w1['n_basins_lstm_higher_by_seed'])} basins {LEAD_WORD[1]}, and the simple model lies
      {c_w1['baseline_against_seed_spread']}.""",
      f"""{W(('5.2', 'Each input set is five seeds'), ('5.3', 'Seeds.'), ('6.3', 'The seed rule'))} &middot; <code>results/lstm_all.json</code> (comparison, notes) &middot; <code>code/04_lstm_all.py</code>""")

    # the rule for a 'clear' difference: stated once (report section 4.2) and applied wherever a results file stores what it needs; the two experiments
    # that store less say which part they use (4.5: seed ranges and basin counts; 6.2: the stored flag larger_than_seed_noise)
    rule = SI["seed_noise_rule"]
    rates = sorted(rule["share_of_comparisons_flagged_when_there_is_no_real_difference"].values())
    twice = "2 x seed_noise" in rule["rule"] and "same side" in rule["rule"]
    sat_seeds = len(SI["seeds"])
    sat_where = IN(("T14", "Satellite inputs"))
    sat_rule = (f"The satellite experiment ({sat_where}) has only {word(sat_seeds)} seeds per model, so it uses a flag stored with every comparison: the change must exceed "
                f"twice the seed noise and every seed of one model must lie on the same side of every seed of the other" if twice else
                f"The satellite experiment ({sat_where}) has only {word(sat_seeds)} seeds per model, so it uses a flag stored with every comparison")
    sat_rule += f"; by a simulation stored in the file, that flag comes up in {pct(rates[0])} to {pct(rates[-1])} of comparisons when nothing has changed."
    if RT is not None:
        on_word = "is clear of every seed" if stamp_gap else "is not clear of the seeds"
        nse_reason = (f"every seed is higher, but only {r1['basins_better']} of {n} basins are" if nse_word == "mixed" else
                      "the single seeds overlap" if nse_word == "none" else f"every seed and {r1['basins_better']} of {n} basins are higher")
        rt_rule = (f"The rain-timing test (the NLDAS row of {IN(('T5', 'lstm_wqn'))}) sets two LSTMs that differ in one input against each other by their single seeds "
                   f"and by the basin count, with no p stored: there the gain in onsets warned {on_word}, and the NSE gain with the river gauge is "
                   f"{'clear' if nse_word == 'clear' else 'not clear'} ({nse_reason}; {X('typical')}).")
    else:
        rt_rule = ("The rain-timing test sets two LSTMs that differ in one input against each other by their single seeds and by the basin count (result pending).")
    smallest_p = re.search(r"smallest possible p is ([\d.]+)", note_with(LA["notes"], "Wilcoxon"))
    Q("C", "clear", "What exactly is a &lsquo;clear&rsquo; difference? Is it the same rule in every section?",
      f"""One rule, the seed rule of {IN(('6.3', 'The seed rule'))}. Model A is called clearly better than model B only when three checks agree: B lies below every single seed of
      A; A scores higher in most basins; and p is below {P_CUT:g}{', two-sided' if two_sided else ''} ({X('pvalue')}). The rule is applied wherever the results
      file stores all three, above all to the LSTM against the simple models ({X('beat')}). Two experiments store less and use the part of the rule they can.
      {rt_rule} {sat_rule} Whatever fails its rule is &lsquo;no clear difference&rsquo;.""",
      f"""&ldquo;Significant&rdquo;, or a winner called from one of the three checks alone. {cap(word(n))} basins that share storms and many comparisons without
      correction do not carry that word: say &lsquo;higher in {wq_wins[-1]} of {n} basins&rsquo; and give the seed spread.""",
      f"""{W(('6.3', 'The seed rule'), ('T18', 'Seed rule'), ('T14', 'Satellite inputs'))} &middot; <code>results/lstm_all.json</code> (comparison, notes), <code>results/rain_timing.json</code>
      (closure), <code>results/satellite_inputs.json</code> (seed_noise_rule, notes) &middot; <code>code/04_lstm_all.py</code>""")

    p_small = f"With {n} basins the smallest possible p is {smallest_p.group(1)}. " if smallest_p else ""
    p_examples = (f"with the river gauge {LEAD_WORD[1]}, p is {n2(c_g1['wilcoxon_p'])} and the LSTM is higher in {c_g1['n_basins_lstm_higher']} of {n} basins; with past "
                  f"weather only, p is {n2(c_w1['wilcoxon_p'])}, yet the simple model lies {c_w1['baseline_against_seed_spread']}")
    Q("C", "pvalue", f"Is your p one-sided or two-sided, and what is it worth with {word(n)} basins?",
      f"""{cap(p_kind)}: the Wilcoxon signed-rank test over the {n} basins. It asks how likely differences this lopsided between two models are if the two are
      equally good, whichever of them is ahead. {p_small}It describes and does not prove: neighbouring basins share storms, so they are not {n} independent
      trials, and many comparisons were made without a correction. So p never decides alone ({X('clear')}): {p_examples}.""",
      f"""&ldquo;p below {P_CUT:g}, so the LSTM is significantly better.&rdquo; With past weather only p is below {P_CUT:g} at all three lead times and a single
      training is still not clearly ahead ({X('seeds')}). And do not call the test one-sided: it is {p_kind}.""" if w_avg_only and two_sided else
      f"""Reading p as proof. The basins are not independent and the comparisons are not corrected: p is one of three checks ({X('clear')}).""",
      f"""{W(('6.3', 'Wilcoxon signed-rank test'))} &middot; <code>results/lstm_all.json</code> (comparison: wilcoxon_p; notes) &middot; <code>code/04_lstm_all.py</code>
      (scipy.stats.wilcoxon)""")

    calls = [LS["comparison"][inp]["verdict"][lk(k)]["nse_call"] for inp in ("w", "wq") for k in LEADS]
    n_ahead = sum(c.lower().startswith("lstm ahead") for c in calls)
    v1 = LS["comparison"]["wq"]["verdict"]["l1"]
    behind = v1["nse_call"].startswith("baseline ahead")
    diag = sorted(LS["diagnostic_fixed_length"]["wq"], key=lambda d: d["updates"])
    mean = lambda v: sum(v) / len(v)
    memorised = mean(diag[-1]["train_nse"]) > mean(diag[0]["train_nse"]) and mean(diag[-1]["val_nse"]) < mean(diag[0]["val_nse"])     # better where fitted, worse elsewhere
    fb = LS["basin"]["gauge_id"]
    fb_name = river(LS["basin"]["short"])
    all_fb = [LA["sets"]["wq"]["test"]["basins"][fb][lk(k)]["nse"] for k in LEADS]
    one_fb = [LS["comparison"]["wq"]["verdict"][lk(k)]["lstm_nse"] for k in LEADS]
    all_seeds = {k: [v["test_nse_by_basin"][fb][k - 1] for v in LA["sets"]["wq"]["per_seed"].values()] for k in LEADS}
    one_seeds = {k: LS["finals"]["wq"]["test_nse_by_seed"][lk(k)] for k in LEADS}
    above = [k for k in LEADS if min(all_seeds[k]) > max(one_seeds[k])]
    seeds_text = (f"{leads(above)} all its seeds are above those of the one-river model" if above else "the seeds of the two models overlap at every lead time")
    still = (f"Even so, {LEAD_WORD[1]} it stays below the best simple model fitted to that river alone ({n2(v1['best_baseline_nse'])})."
             if all_fb[0] < v1["best_baseline_nse"] else f"{cap(LEAD_WORD[1])} it is level with or above the best simple model fitted to that river alone ({n2(v1['best_baseline_nse'])}).")
    Q("C", "allrivers", "Why one model for all rivers instead of one per river?",
      f"""{'Because the one-river LSTM fell short' if n_ahead == 0 else 'To give the model more rivers to learn from'}: on the {fb_name} River
      ({n_years(split['train'])} training years) it was ahead of the best simple model in {word(n_ahead) if n_ahead else 'none'} of its {word(len(calls))} comparisons"""
      + (f", and with the river gauge {LEAD_WORD[1]} it was clearly behind (NSE {n2(v1['lstm_nse'])} against {n2(v1['best_baseline_nse'])})" if behind else "")
      + f""". {'It memorised its training years almost at once: trained' if memorised else 'Trained'} on from update {diag[0]['updates']} to update
      {diag[-1]['updates']}, its NSE went from {n2(mean(diag[0]['train_nse']))} to {n2(mean(diag[-1]['train_nse']))} in the training years and from
      {n2(mean(diag[0]['val_nse']))} to {n2(mean(diag[-1]['val_nse']))} in the validation years. One model for many rivers is the approach of recent work
      {cite('kratzert2019')}; here it sees {n} rivers instead of one. On the {fb_name} River the all-river model scores NSE {slash(all_fb)} against {slash(one_fb)}; {seeds_text}. {still}""",
      f"""&ldquo;More rivers made the LSTM beat the simple models.&rdquo; They made it better than the one-river LSTM. Against the simple models, with the river gauge,
      {say(cls['wq'])} ({X('beat')}).""",
      f"""{W(('2.2', 'should never be trained on a single basin'), ('5.2', 'one-river tuning'))} &middot; <code>results/lstm_single.json</code> (comparison, diagnostic_fixed_length), <code>results/lstm_all.json</code>
      &middot; <code>code/03_lstm.py</code>, <code>code/04_lstm_all.py</code>""")

    props = LA["static_inputs"]
    GC = G["comparison_with_camels"]
    ratio = 1 + GC["slope_mean"]["median_difference"] / 100
    rho = [GC[k]["rank_correlation"] for k in ("area_vs_gages2", "elev_mean", "slope_mean", "forest_share")]
    cell = re.search(r"(\d+) m", G["sources"]["land_cover"]["resolution"]).group(1)                    # the rasters' cell size in metres
    if LOBO is not None:
        same_v = bp_in["verdict"] == bp_out["verdict"]
        help_text = (f"Do they help? By our rule, {bp_in['verdict'] if same_v else 'it depends'}: {LEAD_WORD[1]} the median NSE changes by "
                     f"{signed(bp_in['median_nse_difference'])} with them for basins in training{'' if same_v else ' (' + bp_in['verdict'] + ')'} and by "
                     f"{signed(bp_out['median_nse_difference'])} for a never-seen basin{'' if same_v else ' (' + bp_out['verdict'] + ')'}, against a typical seed spread of "
                     f"{n2(rng)}.")
        prop_trap = (f"On a never-seen river the properties make {bp_out['verdict']}: the forecast is higher with them in {bp_out['n_basins_higher']} of {n} basins, and in "
                     f"{river(worst['basin'])} it falls to NSE {n2(worst['held_out_with_properties'])} with them ({n2(worst['held_out_without_properties'])} without, "
                     f"{n2(worst['in_training'])} in training).")
    else:
        help_text = f"Whether they help, for basins in training and for a never-seen basin: result pending ({X('unseen')})."
        prop_trap = f"That is the leave-one-basin-out question ({X('unseen')}, result pending). With {n} basins the model has met only {n} combinations of properties."
    clim_words = join(PROP[c] for c in clim_codes) if clim_codes else ""
    clim_sentence = (f"{word(len(clim_codes))} of them ({clim_words}) are CAMELS climate averages of {clim_years.group(1)}&ndash;{clim_years.group(2)}, years that "
                     f"reach into the validation and test years; and " if clim_codes and clim_years else "")
    some = [p for p in ("log_area", "slope_mean", "p_mean", "frac_forest", "soil_depth_pelletier") if p in props]
    Q("C", "props", "What do the basin properties do?",
      f"""They are {word(len(props))} fixed numbers per basin from the CAMELS attribute table {cite('addor2017')}, such as {join(PROP[p] for p in some)}. They are
      repeated on every day as extra inputs, so that one model can tell the rivers apart; none is computed from the flow record. {help_text}
      {'Two cautions' if clim_sentence else 'One caution'}: {clim_sentence}when four of them were recomputed from {cell} m rasters, the values differed (mean slope
      {n2(ratio, 1)} times the CAMELS value) while the order of the basins stayed nearly the same (rank correlation {n2(min(rho))} to {n2(max(rho))}). The model was not retrained with those values.""",
      f"""&ldquo;The basin properties let the model work on any river.&rdquo; {prop_trap}""",
      f"""{W(('4.2', 'climate averages'), ('5.2', '10 static properties'), ('AppB', 'STATIC'))} &middot; <code>results/lstm_all.json</code> (static_inputs, notes), <code>results/lstm_lobo.json</code>,
      <code>results/basin_gis.json</code> &middot; <code>code/flood.py</code>, <code>code/05_lobo.py</code>, <code>code/07_basin_maps.py</code>""")

    # the training target: every scored day counts in the loss, and onsets are few of them (counts of the test years, all basins together)
    all_days = LA["sets"]["wq"]["test"]["summary"]["l1"]["all"]
    n_days = sum(all_days[c] for c in ("hits", "misses", "false_alarms", "correct_neg"))
    same_loss = (f"What our results do show: with the same loss, the LSTM given the weather of the days being forecast (the upper bound) warns of "
                 f"{pct(on_ub[0]['pod'])} of the onsets {LEAD_WORD[1]} instead of {pct(on_wq[0]['pod'])}, so the loss does not stop a model from warning when the "
                 f"information is there" + (f" (given the rain up to midnight it warns of {pct(AO['with_'][1])}{Hi('aorc')})" if AO else "")
                 if on_ub[0]["pod"] > on_wq[0]["pod"] else "")
    Q("C", "target", "You care about onsets, but you trained on the squared error of every day. Why, and what would another training target change?",
      f"""A squared error is the standard training target for a regression, and NSE, our score, is built on it. It counts every scored day and every lead time
      alike ({X('network')}), and onsets are rare: in the test years {count(n_onset[1])} of {count(n_days)} basin-days ({pct(n_onset[1] / n_days, 1)}). A squared
      error does weigh large errors heavily, and flood days carry the largest; but a forecast that rises one day late pays for it on a single day. What another
      target would change was not tested: a loss that gives high-flow days or onsets more weight, or a network trained to give the warning itself. {same_loss};
      and moving the warning level afterwards only trades misses for false warnings ({X('warnlevel')}). Whether another target would get more out of today&rsquo;s
      inputs is open.""",
      f"""&ldquo;A flood-weighted loss would fix the onsets.&rdquo; Not tested; and no loss can supply rain that is not in the model&rsquo;s inputs
      ({X('onsets')}).""",
      f"""{W(('5.2', 'Equation (1)'), ('9.3', 'Open.'))} &middot; <code>results/lstm_all.json</code> (sets, decomposition),
      <code>results/flood_analysis.json</code> (skill) &middot; <code>code/flood.py</code> (fit)""")

    # overfitting and the learning curves (training_curves.json, written by 12_training_curves.py from the curves every training stored; the report
    # shows them as Figure 3.2). Every statement about a curve is checked on the stored numbers before it is printed. Two things the answer must
    # not blur: the one-river curve is the BASE setting of the tuning (one training), not the chosen one-river model; and an update is not a pass
    # through the data, so 'twelve rivers peak later' is true in updates only.
    fig32 = report_says("F8", "Learning curves")                     # the learning curves are Figure 8 of the final report (Appendix C)
    try:
        if TC is None:
            raise KeyError("the file is not there")
        one = TC["panels"]["one_river"]["per_training"][0]
        all_q, all_f = TC["panels"]["all_wq"]["per_training"], TC["panels"]["all_wqf"]["per_training"]
        peaks_q, peaks_f = sorted(t["best_update"] for t in all_q), sorted(t["best_update"] for t in all_f)
        span = lambda v: f"update {count(v[0])}" if v[0] == v[-1] else f"updates {count(v[0])} to {count(v[-1])}"
        falls_after = lambda t: t["val_nse_at_last"] < t["val_nse_at_best"] and t["train_loss_last"] < t["train_loss_at_best"]     # worse on new years while it fits its own better
        one_overfits = one["last_update"] > one["best_update"] and falls_after(one)
        base_note = (" (the base setting of the tuning, one training" + (f"; not the chosen {ch['hidden']}-cell model" if ch["hidden"] != tun["base"]["hidden"] else "")
                     + ")") if tun["chosen"] != tun["base"] else ""
        same_net = (f"The same network settings ({TC['settings']['hidden']} memory cells, learning rate {TC['settings']['lr']:g}) were used on one river and on all "
                    f"{n}. " if TC["same_settings_one_river_and_all"] else "")
        curves = (f"The learning curves show it{' (' + IN('F8') + ')' if fig32 else ''}. {same_net}On one river{base_note} the validation NSE peaks at update "
                  f"{count(one['best_update'])} ({n2(one['val_nse_at_best'])}) and has fallen to {n2(one['val_nse_at_last'])} by update {count(one['last_update'])}, while the "
                  f"training loss falls further ({n2(one['train_loss_at_best'])} to {n2(one['train_loss_last'])}): the network fits its training years better and the "
                  f"validation years worse, which is overfitting. " if one_overfits else
                  f"The learning curves{' (' + IN('F8') + ')' if fig32 else ''} show where each training was stopped. {same_net}On one river{base_note} the state was "
                  f"kept at update {count(one['best_update'])} of {count(one['last_update'])}. ")
        curves += (f"On all {n} rivers with the river gauge the kept states are at {span(peaks_q)}; given the weather of the days being forecast, an upper bound, at "
                   f"{span(peaks_f)}." + (" There too the training loss has fallen further by the last check; that the validation score is lower there than at the kept state "
                                          "follows from the stopping rule." if all(falls_after(t) for t in all_q + all_f) else ""))
        # updates against passes through the training years (an update scores the same number of days in every set-up)
        p_one, p_lo, p_hi = one.get("passes_at_best"), TC["panels"]["all_wq"].get("passes_at_best_min"), TC["panels"]["all_wq"].get("passes_at_best_max")
        if p_one and p_lo and p_hi and one["best_update"] > 0 and peaks_q[0] > one["best_update"]:
            r_lo, r_hi = f"{peaks_q[0] / one['best_update']:.0f}", f"{peaks_q[-1] / one['best_update']:.0f}"
            as_many = "about as many" if p_lo <= 1.25 * p_one and p_hi >= 0.8 * p_one else "more" if p_lo > p_one else "fewer"       # the reading of report section 3
            passes_trap = (f" &ldquo;On {word(n)} rivers it learns {r_lo if r_lo == r_hi else r_lo + ' to ' + r_hi} times as long before it overfits&rdquo;: in updates, yes, "
                           f"but an update scores the same number of days in every set-up, and counted in passes through the training years the kept states come "
                           f"after {as_many} passes (about {p_lo:.0f} to {p_hi:.0f} on {n} rivers, about {p_one:.0f} on one).")
        else:
            passes_trap = ""
    except (KeyError, TypeError, IndexError, ValueError) as err:
        print(f"results/training_curves.json was not read ({err!r}): the answer on overfitting gives no learning curve", file=sys.stderr)
        TC, one_overfits, curves, passes_trap = None, False, "", ""
    hk_steady = hk["val_nse_mean"] in (sorted(hk["val_nse_mean"]), sorted(hk["val_nse_mean"], reverse=True))      # the validation score rises or falls steadily with size
    by_size = (f"on one river the validation scores were {join(n2(v) for v in hk['val_nse_mean'])} for {join(hk['values'])} cells"
               + ("" if hk_steady else ", with no steady trend") + f" ({X('settings')})")
    if memorised or one_overfits:
        o_head = "Trained on, yes, and most clearly on one river; early stopping keeps the state before that."
        if not curves:
            curves = f"Trained on past its stopping point, the one-river LSTM scored higher in the training years and lower in the validation years ({X('allrivers')})."
    else:
        o_head = "Not clearly."
    no_drop = (f" And the kept all-river model does not score lower on the years that chose nothing: with the river gauge {LEAD_WORD[1]} its median NSE is {n2(v_nse)} in "
               f"the validation years and {n2(t_nse)} in the test years ({X('period')})." if t_nse >= v_nse else "")
    Q("C", "overfit", "Did the network overfit? Show me a learning curve.",
      f"""{o_head} {curves} What is done against it: early stopping on the validation years (a check every {cfg['eval_every']} updates, the best state is kept,
      and training ends after {cfg['patience']} checks without a better score; {X('settings')}), a dropout of {cfg['dropout']:g}, and one model for all {n}
      rivers instead of one per river ({X('allrivers')}).{no_drop}""",
      f"""&ldquo;No, because we used dropout&rdquo;: the evidence is the validation curve, not the method.{passes_trap} And &ldquo;it needs a smaller
      network&rdquo;: {by_size}.""",
      f"""{W(('5.2', 'learning curves: Figure 8'), 'F8') if fig32 and TC else W(('5.2', 'kept states'))} &middot; """
      + ("<code>results/training_curves.json</code> (panels, notes), " if TC else "")
      + """<code>results/lstm_single.json</code> (tuning, diagnostic_fixed_length), <code>results/lstm_all.json</code> (config; sets: curve) &middot; """
      + ("<code>code/12_training_curves.py</code>, " if TC else "") + """<code>code/flood.py</code> (fit)"""
      + (" &middot; <code>figures/training_curves.png</code>" if TC and (FIG / "training_curves.png").exists() else ""))

    # inside the LSTM cell: fixed properties of torch.nn.LSTM and of LSTMNet in code/flood.py (the start value of the forget gates is read from its
    # source by code_facts); the count of the weights is written out and must equal the count the training stored
    n_in = len(FORCING) + 1 + len(props)                           # daily inputs of the all-river model with the river gauge: weather, the gauge, the basin properties
    cells, n_out = cfg["hidden"], len(LEADS)
    counted = 4 * cells * (n_in + cells + 2) + n_out * cells + n_out           # four blocks (three gates and the new value), each with a weight per input and per cell and two bias terms; the output layer
    if counted == n_weights:
        weights_text = (f" The {count(n_weights)} weights of the all-river model with the river gauge are quickly counted: four blocks (the three gates and the new "
                        f"value), each with {cells} cells that read {n_in} inputs and {cells} cell outputs and have two bias terms, 4 &times; {cells} &times; ({n_in} + {cells} + 2); "
                        f"plus the output layer, {n_out} &times; {cells} + {n_out}.")
    else:
        print(f"build_qa: the weights of the LSTM counted from its formula ({counted}) are not the stored {n_weights}: the count was left out of the question on the cell",
              file=sys.stderr)
        weights_text = ""
    forget_text = (f" In our code the forget gates start open (a bias of {CODE['forget_bias']:g} is set before training), so the untrained network starts by "
                   f"remembering." if CODE["forget_bias"] else "")
    Q("C", "cell", "What is inside an LSTM cell, and why can it remember?",
      f"""Each memory cell carries one number from day to day, its state. Three gates, each a value between 0 and 1 computed from the day&rsquo;s inputs and
      the cell outputs of the day before, act on it: the forget gate says how much of yesterday&rsquo;s state is kept, the input gate how much of today&rsquo;s
      new value is written, the output gate how much of the state is passed on. Read as hydrology, the state is a store of water: the forget gate lets it
      drain, the input gate lets today&rsquo;s rain in. From one day to the next the state is changed only by a multiplication and an addition, so a signal
      can survive hundreds of days, and so can the error signal that training sends back; in a plain recurrent network both fade
      {cite('hochreiter1997')}.{forget_text}{weights_text}""",
      f"""&ldquo;The LSTM remembers the last {cfg['warmup']} days.&rdquo; The {cfg['warmup']} days are what it is given to fill its memory ({X('seq')}); what it
      keeps is decided by the forget gates, day by day. And the memory test says how much old weather counts: {LEAD_WORD[1]}, {mem_text}.""",
      f"""{W(('2.2', 'gates carry the state'), ('5.2', 'The LSTM.'))} &middot;
      <code>results/lstm_all.json</code> (config, setting_check), <code>results/lstm_single.json</code> (memory) &middot; <code>code/flood.py</code> (LSTMNet)""")

    # ======================================================================================================================
    # D. results
    # ======================================================================================================================
    # one forecast as a picture: the event that 06_flood_analysis.py picked by rule (flood_analysis.json: event_shown); the final report does not show it, so the
    # answer names the figure file and the final report's figure of where the onset warnings fail
    try:
        EV = FA["event_shown"]
        storm = (f", when {STORM.get(EV['cyclone_status'], 'tropical cyclone')} {EV['cyclone_name'].title()} passed {n2(EV['cyclone_km'], 0)} km from the gauge"
                 if EV.get("cyclone_name") else "")
        month_of = datetime.fromisoformat(EV["onset_date"])
        picked = "missed" in EV["rule"] and "caught" in EV["rule"]    # the rule of the pick: an onset missed with the river gauge and caught with the upper bound
        outcome = ("The forecast of the LSTM with the river gauge stays below the level: no warning. The LSTM given the weather of the days being forecast, the upper "
                   "bound, warns. " if picked else "")
        pattern = (f"{count(EV['onsets_with_this_pattern'])} of the {count(EV['onsets'])} onsets of the test years went this way {LEAD_WORD[1]}. " if picked else "")
        big_day = LS["variance_concentration"]["test"]["days"][0]      # the largest flow day of the test years on the one-river basin (the second picture)
        first_pic = ("Open the event figure of the final report (<code>figures/flood_event.png</code>)" if report_shows("flood_event.png") else
                     f"Open <code>figures/flood_event.png</code> (in the project folder; the final report does not show it, its {IN(('F4', 'Where the onset warnings fail'))} "
                     f"shows where the onset warnings fail)")
        second_pic = ("A second picture, in the final report, is <code>figures/lstm_single_hydrograph.png</code>" if report_shows("lstm_single_hydrograph.png")
                      else "A second picture, also only in the project folder, is <code>figures/lstm_single_hydrograph.png</code>")
        Q("D", "show", "Show me one forecast.",
          f"""{first_pic}: {EV['basin']}, {MONTH[month_of.month - 1]} {month_of.year}{storm}.
          Every forecast is drawn on the day it is for and was made the evening before. On {date(EV['onset_date'])} the river crosses its high-flow level
          ({n2(EV['high_flow_level_mm_day'], 1)} mm per day): the onset of an event of {EV['duration_days']} days. The rain table stamps
          {n2(EV['rain_target_day_mm'], 0)} mm on that day and {n2(EV['rain_issue_day_mm'], 0)} mm on the day before. {outcome}{pattern}The event was picked by a
          rule written in the script, not by eye. {second_pic}: the largest flow
          day of the test years on the {fb_name} River, {date(big_day['day'])} ({n2(big_day['flow_mm'], 0)} mm per day), with the forecasts made the day
          before. Look at how far each forecast rises against the observed peak.""",
          f"""Presenting the picture as typical skill: it is one event picked by a rule, an illustration and not a score. And &lsquo;the rain had not fallen
          yet&rsquo;: the rain was stamped on the onset day and was not yet in the model&rsquo;s inputs; how much of it had fallen by the evening before, a daily
          table cannot say for a single event ({X('rainday', 'timing')}).""",
          f"""{W(('7.3', 'Where the warnings fail'), 'F4')} &middot; <code>results/flood_analysis.json</code> (event_shown, failures), <code>results/flood_events.csv</code> &middot;
          <code>code/06_flood_analysis.py</code>, <code>code/03_lstm.py</code> &middot; <code>figures/flood_event.png</code>,
          <code>figures/lstm_single_hydrograph.png</code>""")
    except (KeyError, TypeError, IndexError, ValueError) as err:
        print(f"results/flood_analysis.json: the example event was not read ({err!r}); the question 'show me one forecast' was left out", file=sys.stderr)

    head = "Not where it counts." if gauge_tie else "Yes, at every lead time." if gauge_better else "Only in part."
    if w_avg_only:
        w_text = (f"With past weather only, the other forecast that could be run in practice, the {word(n_seeds)}-seed average is ahead (in {join(w_wins)} of {n} "
                  f"basins), but the simple model lies inside the seed spread: a single training is not clearly ahead.")
    else:
        w_text = f"With past weather only, the other forecast that could be run in practice, {say(w_cls)} (LSTM higher in {join(w_wins)} of {n} basins)."
    k3 = like_leads["wqf"][-1]
    c3 = cmp_("wqf", k3)
    ub_text = (f"With the perfect weather forecast (the upper bound) the two kinds of model read the same days of weather only {leads(like_leads['wqf'])}; there, with "
               f"the river gauge, {PHRASE_UB[cls['wqf'][k3]]} ({n2(c3['lstm_nse_median'])} against {n2(c3['baseline_nse_median'])}; in "
               f"{c3['n_basins_lstm_higher']} of {n} basins, p = {n2(c3['wilcoxon_p'])})" + (f", though {vars_text}." if vars_text else "."))
    c2 = cmp_("wq", 2)
    forest_above = c2["baseline_against_seed_spread"] == "above every seed"
    c9 = max((cmp_("wq", k) for k in LEADS), key=lambda c: c["n_basins_lstm_higher"])
    same9 = (f"the two medians are the same to two decimals ({n2(c9['lstm_nse_median'])})" if n2(c9["lstm_nse_median"]) == n2(c9["baseline_nse_median"]) else
             f"the medians differ by {n2(abs(c9['lstm_nse_median'] - c9['baseline_nse_median']))} NSE")
    k9 = next(k for k in LEADS if cmp_("wq", k) is c9)
    # with the rain up to midnight (7 and 8 October 2026) the LSTM ends ahead in median NSE but not in onsets: said once here, in full in H9
    new_beat = (f" With the rain up to midnight as one more input the LSTM is ahead in median NSE ({n2(AO['nse_with'][1])} against {n2(AO['nse_simple']['ridge'])} and "
                f"{n2(AO['nse_simple']['forest'])} {LEAD_WORD[1]}), but not in the onsets warned, where ridge regression warns of {pct(AO['warned']['ridge'])} against "
                f"{pct(AO['warned']['lstm'])}{Hp('holds')}." if AO and AO["warned"]["ridge"] > AO["warned"]["lstm"] else "")
    Q("D", "beat", "Does the LSTM really beat the simple models? Where not?",
      f"""{head} With past weather + river gauge, {practice}, {say(cls['wq'])}: median NSE {slash(wq_lstm)} against {slash(wq_base)}
      for the best simple model, the LSTM higher in {join(wq_wins)} of {n} basins, so milestone 2 of our plan is {m2} there. {w_text} {ub_text} The clear win
      is over {TT}: higher in {join(pers_wins)} of {n} basins.{new_beat}""",
      (f"""&ldquo;The LSTM is better in most basins&rdquo; ({c9['n_basins_lstm_higher']} of {n}, {LEAD_WORD[k9]}, with the gauge). There p is {n2(c9['wilcoxon_p'])} and
      {same9}"""
       + (f"; and {LEAD_WORD[2]} the {family(best('wq', 2))} is above every single LSTM seed." if forest_above else ".")) if cls["wq"][k9] == "tie" else
      f"""Calling a winner from a basin count alone. Our rule asks three things at once: the simple model below every single seed of the LSTM, the LSTM higher in most
      basins, and p below {P_CUT:g}.""",
      f"""{W(('7.1', 'no clear difference'), 'T5', ('9.1', 'RQ1'))} &middot; <code>results/lstm_all.json</code> (comparison, findings), <code>results/lstm_single.json</code> (comparison)
      &middot; <code>code/04_lstm_all.py</code>""")

    protocol = (f"Each basin is held out in turn; an LSTM trained on the other {n - 1} forecasts it from its past weather alone, because a river nobody has measured "
                f"has no gauge.")
    if LOBO is not None:
        held, held_no = nm["held_out_with_properties"], nm["held_out_without_properties"]
        clear_loss = loss["n_basins_loss_larger_than_seed_range"] > n / 2
        skill = ("no better than always forecasting the average flow" if held <= 0 else
                 f"still better than always forecasting the average flow in {pos['held_out_with_properties']} of {n} basins"
                 + (", but clearly worse than on a river it has seen" if clear_loss else ""))
        forest_clear = abs(lf["median_nse_difference"]) > rng                                        # beyond the typical seed spread
        forest_word = ((", and it beats the simple model" if lf["n_basins_higher"] > n / 2 else ", and the simple model does better") if forest_clear else
                       ", and it does not beat the simple model" if lf["n_basins_higher"] <= n / 2 else ", with no clear difference from the simple model")
        forest = (f"A random forest fitted on the pooled days of the other {n - 1} basins scores {n2(nm['forest_held_out'])}: the LSTM with the properties is higher in "
                  f"{lf['n_basins_higher']} of {n} basins, with a median difference of {signed(lf['median_nse_difference'])} NSE"
                  f"{'' if forest_clear else ', less than the seed spread'}.")
        # the basin that scores higher when held out, and the shift that goes with every loss (lstm_lobo.json: higher_when_held_out, flow_per_rain)
        try:
            HW = LOBO["higher_when_held_out"][0]
            hn, hs, hval = HW["nse_1_day_ahead"], HW["share_of_rain_reaching_the_river"], HW["nse_validation_years_1_day_ahead"]
            hvol = HW["forecast_volume_error_pct_1_day_ahead"]["test_years"]
            ratios = [b["test_over_training"] for b in LOBO["flow_per_rain"]["basins"].values()]
            n_less = sum(r < 1 for r in ratios)
            vol = lambda v: f"{n2(abs(v), 0)}% too {'much' if v > 0 else 'little'}"
            n_higher = len(LOBO["higher_when_held_out"])
            usual = (f" In the validation years the order is the usual one ({n2(hval['in_training'])} in training, {n2(hval['held_out_with_properties'])} held out)."
                     if hval["in_training"] > hval["held_out_with_properties"] else "")
            shift = (f" And it is not only this river: in {n_less} of the {n} basins a smaller share of the rain reached the river in the test years than in the "
                     f"training years ({n2(min(ratios))} to {n2(max(ratios))} of the training share), so every loss from not seeing a basin is measured together "
                     f"with this shift and is approximate." if n_less > n / 2 else "")
            heldout = (f"""No. It is {HW['basin']}: NSE {n2(hn['held_out_with_properties'])} held out against {n2(hn['in_training'])} in training, {LEAD_WORD[1]}
                       (single seeds in training: {n2(hn['in_training_seed_min'])} to {n2(hn['in_training_seed_max'])}). The river changed between the periods:
                       {pct(hs['training_years'])} of the rain reached it in the training years and {pct(hs['test_years'])} in the test years. So the model that
                       had learned the river forecasts {vol(hvol['in_training'])} water in the test years; the held-out model,
                       {vol(hvol['held_out_with_properties'])}.{usual}{shift}""",
                       f"""&ldquo;A basin can score higher when held out, so the model works on rivers it has not seen.&rdquo; The higher score goes with a change
                       in the river between the two periods; it is not skill for unseen rivers.""")
            more = f" ({X('heldout')} for the {'one basin that scores' if n_higher == 1 else word(n_higher) + ' basins that score'} higher)"
            more_own = f"; {'one basin scores' if n_higher == 1 else word(n_higher) + ' basins score'} higher when held out ({X('heldout')})"
        except (KeyError, IndexError, TypeError, ValueError) as err:
            print(f"results/lstm_lobo.json: the basin that scores higher when held out was not read ({err!r}); its question was left out", file=sys.stderr)
            heldout, more, more_own = None, "", ""
        seed_note = (f" (one seed each; the {word(n_seeds)}-seed average of the model in training scores {n2(avg_w)}, the value in final report section {S_RULE})" if differs else
                     " (one seed each)" if one_seed else "")
        # the paired loss first (the number to quote), the two medians second; then the same without the basin properties
        if unseen_worse and held < nm["in_training"]:
            first = (f"It is forecast worse: {unseen_pair}{seed_note}. Quote the loss of the typical basin, not the difference of the two medians: it is the "
                     f"median of the {n} losses, basin by basin, and the two medians come from different basins. {loss['n_basins_loss_larger_than_seed_range']} of "
                     f"the {n} losses are larger than the typical seed spread of {n2(rng)}{more_own}; the worst is {river(worst['basin'])}, from "
                     f"{n2(worst['in_training'])} to {n2(worst['held_out_with_properties'])}. With weather only, without the basin properties, the typical basin "
                     f"loses {n2(loss_no['median_nse_loss'])} (the median goes from {n2(nm['in_training_no_properties'])} to {n2(held_no)}).")
        else:
            first = (f"{cap(LEAD_WORD[1])} its median NSE is {n2(held)} with the basin properties and {n2(held_no)} with weather only; the same basins score "
                     f"{n2(nm['in_training'])} and {n2(nm['in_training_no_properties'])} when they are in training{seed_note}. For the typical basin, not having seen "
                     f"it changes the NSE by {signed(-loss['median_nse_loss'])} with the properties (the median of the {n} changes): "
                     f"{loss['n_basins_lower_when_held_out']} of {n} basins score lower{more}.")
        # what hundreds of training basins make possible, as published (published_scores.json; read at the source by an AI agent): not comparable in level
        try:
            pub_k = PS["sources"]["kratzert2019pub"]
            pub_trap = (f" And the opposite, &lsquo;an LSTM cannot forecast a river it has not seen&rsquo;: trained on several hundred of {count(pub_k['n_basins'])} basins, a published LSTM "
                        f"reached a median NSE of {n2(pub_k['median_nse']['lstm_basin_never_seen'])} on basins it never saw {cite('kratzert2019pub')}; ours had "
                        f"{n - 1}. That score is for the flow of a day from the weather up to that day, so its level cannot be set beside ours ({X('published')}).")
        except (KeyError, TypeError):
            pub_trap = ""
        more_unseen = (f" A later test with {RV['n_train']['unseen50_w_static']} training rivers instead of {RV['n_train']['lobo_w_static']} raised the median NSE {LEAD_WORD[1]} from "
                       f"{n2(RV['unseen']['lobo_w_static'][1])} to {n2(RV['unseen']['unseen50_w_static'][1])}, and the loss from not having seen the river shrank from {n2(RV['loss'][0])} to "
                       f"{n2(RV['loss'][1])} NSE{Hp('rivers')}." if RV else "")
        Q("D", "unseen", "What happens on a river the model has never seen?",
          f"""{protocol} {first} {forest} So on an unseen river this model is {skill}{forest_word}; with one seed per held-out model and {n - 1} training basins it
          is a small test.{more_unseen}""",
          f"""&ldquo;It was tested on unseen years, so it works on unseen rivers.&rdquo; Later years of rivers already seen is the easier test, and all our main scores
          are of that kind.{pub_trap}""",
          f"""Final report section {S_UNSEEN} &middot; <code>results/lstm_lobo.json</code> (summary, findings, notes)"""
          + (", <code>results/published_scores.json</code>" if pub_trap else "") + """ &middot; <code>code/05_lobo.py</code>""")
        if heldout:
            Q("D", "heldout", "One river scores higher when it is held out than when it is in training. Is that a leak?", heldout[0], heldout[1],
              f"""Final report section {S_UNSEEN} &middot; <code>results/lstm_lobo.json</code> (higher_when_held_out, flow_per_rain, notes) &middot; <code>code/05_lobo.py</code>""")
    else:
        Q("D", "unseen", "What happens on a river the model has never seen?",
          f"""<b>Result pending:</b> the leave-one-basin-out run had not finished when this sheet was built; rebuild it with <code>python code/build_qa.py</code>. What
          can be said now is the method. {protocol} The held-out score is set against the same basin&rsquo;s score when it was in training, with and without the
          {word(len(props))} basin properties, and against a random forest fitted on the pooled days of the other {n - 1} basins.""",
          f"""&ldquo;It was tested on unseen years, so it works on unseen rivers.&rdquo; Later years of rivers already seen is the easier test; all our main scores are
          of that kind.""",
          f"""Final report section {S_UNSEEN} &middot; <code>results/lstm_lobo.json</code> (not yet written) &middot; <code>code/05_lobo.py</code>""")

    RTM = FAIL["by_rain_timing"]["lstm_wq"]["tuned"]
    cls_hi, cls_lo = "75-100%", "0-25%"
    n_hi = RTM[cls_hi]["onset"]["hits"] + RTM[cls_hi]["onset"]["misses"]
    resc = FAIL["perfect_forecast_rescue"]["tuned"]["all_onsets"]
    stamped_late = RTM[cls_hi]["onset"]["pod"] < RTM[cls_lo]["onset"]["pod"] and n_hi > n_onset[1] / 3      # the misses sit where the rain is stamped on the target day
    onset_verdict1 = VD["services"]["lstm_wq"]["l1"]["onset"]["pooled"]                    # the verdict for onsets today, 1 day ahead with the river gauge
    # (1) why onsets are missed: where the rain of an onset is stamped, and what the upper bound catches; the rain-timing test follows as its own question
    if RT is not None and stamp_gap:
        link = (f"&lsquo;Not yet in the inputs&rsquo; is not &lsquo;not yet fallen&rsquo;: {LEAD_WORD[1]}, {about(on_gap)} of the gap in onsets warned closes with rain "
                f"that had fallen by the evening of the issue day ({X('timing')}).")
        onsets_trap = (f"Part of that rain had fallen, and {about(on_gap)} of the gap in onsets warned closes {LEAD_WORD[1]} without any forecast ({X('timing')}). Say "
                       f"&lsquo;rain that was not yet in the model&rsquo;s inputs&rsquo;.")
    elif RT is not None:
        link = (f"&lsquo;Not yet in the inputs&rsquo; stays the right wording; how much of that rain had already fallen, our test did not show clearly "
                f"({X('timing')}).")
        onsets_trap = (f"The rain table&rsquo;s day stamp means that part of that rain had fallen ({X('rainday')}). Say &lsquo;rain that was not yet in the "
                       f"model&rsquo;s inputs&rsquo;.")
    else:
        link = (f"&lsquo;Not yet in the inputs&rsquo; is not &lsquo;not yet fallen&rsquo;: {LEAD_WORD[1]} part of that rain had fallen by the evening of the issue day "
                f"({X('timing')}).")
        onsets_trap = (f"{cap(LEAD_WORD[1])} part of the missing rain had already fallen and was only stamped on the next day ({X('rainday')}). Say &lsquo;rain that "
                       f"was not yet in the model&rsquo;s inputs&rsquo;.")
    if P1:                                                         # the experiments of 7 and 8 October: what the rain of the rest of the day, and a forecast, add
        link += f" What the later rain and a forecast add was tested next{Hp('aorc', 'moved')}: {lower_first(P1['rungs'])}"
    Q("D", "onsets", "Why are onsets missed?",
      f"""{'Mainly because the rain that makes the river rise was not yet in the model&rsquo;s inputs when the forecast was made' if stamped_late else
           'We sorted the onsets by where their rain is stamped'}. In {count(n_hi)} of the
      {count(n_onset[1])} onsets {LEAD_WORD[1]}, {cls_hi.replace('-', '&ndash;')} of the two days&rsquo; rain is stamped on the target day; the LSTM with the river gauge
      warns of {pct(RTM[cls_hi]['onset']['pod'])} of those, against {pct(RTM[cls_lo]['onset']['pod'])} where most of the rain was already in its inputs. Given the
      weather of the target days (the upper bound), the same LSTM catches {count(resc['of_those_caught_with_perfect_forecast'])} of the
      {count(resc['missed_with_gauge'])} onsets it missed; {count(resc['missed_by_both'])} are missed by both. {link}""",
      f"""&ldquo;The rain had not fallen yet, so only a weather forecast can help.&rdquo; {onsets_trap}""",
      f"""Final report sections {S_WARN} and {S_TIMING} &middot; <code>results/flood_analysis.json</code> (failures) &middot; <code>code/06_flood_analysis.py</code> &middot;
      <code>figures/flood_failures.png</code>""")

    # (2) how much of the gap is rain that had already fallen: the rain-timing test, onsets first, NSE as it is, then what is left
    timing_q = "How much of the gap to the upper bound is rain that had already fallen when the forecast was issued?"
    timing_where = (f"""Final report section {S_TIMING} &middot; <code>results/rain_timing.json</code> (closure, models, notes)"""
                    + (", <code>results/day_windows.json</code> (day_ends)" if dw else "")
                    + """ &middot; <code>code/10_rain_timing.py</code> &middot; <code>figures/rain_timing.png</code>""")
    if RT is not None and stamp_gap:                               # the onsets warned show a time-stamp gap, clear of every seed
        if both_fams:
            other = (f" Without the river gauge it is the same: {pct(r1n['onset_warned_base'])} to {pct(r1n['onset_warned_with_nldas'])} (control "
                     f"{pct(r1n['onset_warned_control'])}, upper bound {pct(r1n['onset_warned_perfect_forecast'])}), {share(r1n['onset_share_of_gap_closed'])} of that gap.")
        elif r1n is not None:
            other = (f" Without the river gauge the onsets warned go {on_move(r1n)} (control {pct(r1n['onset_warned_control'])}), which is not clear of the single "
                     f"seeds.")
        else:
            other = ""
        ends = (f"the added rain ends at about {dw['nldas']} local standard time by our hourly check, so the last hours of the issue day are in neither model" if dw else
                "the added rain ends in the early evening, so the last hours of the issue day are in neither model")
        later_test = f" That rest was tested next{Hp('aorc', 'moved')}: {lower_first(P1['rungs'])}" if P1 else ""
        still = "is still" if rain_verdict == onset_verdict1 else "becomes"
        remains = (f"{about(1 - on_gap)} of that gap remains, and with that rain the verdict for onsets {still} {rain_verdict} ({pct(on_rain_t['pod'])} warned, "
                   f"{pct(on_rain_t['far'])} of the warnings false)")
        medians = (f"; and the {share(s1)} of the NSE gap that closes is a difference of two medians in which only {r1['basins_better']} of {n} basins gain"
                   if nse_word == "mixed" else "")
        Q("D", "timing", timing_q,
          f"""Rain known by the early evening of the issue day closes {about(on_gap)} of it, {LEAD_WORD[1]}, measured on the onsets warned. Given rain stamped up to the
          evening of the issue day as one more input, the LSTM with the river gauge warns of {pct(r1['onset_warned_with_nldas'])} of the onsets instead of
          {pct(r1['onset_warned_base'])} ({on_refs(r1)}): {share(on_gap)} of the gap, and every single seed agrees.{other} The price: {pct(r1['onset_false_with_nldas'])}
          instead of {pct(r1['onset_false_base'])} of the onset warnings are false, and their number goes from {count(on_base_t['false_alarms'])} to
          {count(on_rain_t['false_alarms'])}. {rt_nse} ({X('typical')}). The rest of the gap needs later rain, most of it after the forecast is issued ({ends});
          {rt_later}.{later_test}""",
          f"""There is one on each side. &ldquo;Only a weather forecast can help&rdquo;: {about(on_gap)} of the gap in onsets warned closes without one"""
          + (", with and without the river gauge" if both_fams else "")
          + f""". And &ldquo;most of the gap is a time stamp&rdquo;, or &ldquo;rain observations up to the hour would fix the warnings&rdquo;: {remains}{medians}.""",
          timing_where)
    elif RT is not None:                                           # the onsets warned do not stand clear of the single seeds
        Q("D", "timing", timing_q,
          f"""Our test did not show it clearly. With rain stamped up to the evening of the issue day as one more input, the LSTM with the river gauge warns of
          {pct(r1['onset_warned_with_nldas'])} of the onsets {LEAD_WORD[1]} instead of {pct(r1['onset_warned_base'])} ({on_refs(r1)}), and the single seeds overlap
          those of the model without that rain or of the control.""",
          f"""&ldquo;Rain observations up to the hour would fix the warnings.&rdquo; The rain table&rsquo;s day stamp means that part of the rain had fallen
          ({X('rainday')}); whether such observations would help, our test did not show.""",
          timing_where)
    else:
        Q("D", "timing", timing_q,
          f"""<b>Result pending:</b> the rain-timing test had not finished when this sheet was built. What can be said now: {LEAD_WORD[1]}, part of the rain that the
          upper bound reads had fallen by the evening of the issue day and was only stamped on the next day ({X('rainday')}).""",
          f"""&ldquo;Rain observations up to the hour would fix the warnings.&rdquo; How much of the gap they close is exactly what the pending test measures.""",
          timing_where)

    # the rain-timing test in NSE: a difference of two medians against what the typical basin does (rain_timing.json: closure and the scores by basin)
    if RT is not None:
        pct1 = lambda v: pct(v, 1)
        seeds_on = (f"single seeds {pct1(r1['onset_seed_range_with_nldas'][0])} to {pct1(r1['onset_seed_range_with_nldas'][1])} with that rain, "
                    f"{pct1(r1['onset_seed_range_base'][0])} to {pct1(r1['onset_seed_range_base'][1])} without it and "
                    f"{pct1(r1['onset_seed_range_control'][0])} to {pct1(r1['onset_seed_range_control'][1])} for the control")
        if change:
            by_basin = (f"the forecast is higher in {len(ups)} of {n} basins, by up to {signed(max(ups))}, and lower in the other {len(downs)}, by at most "
                        f"{n2(-min(downs))}; the median of the {n} changes is {signed(r1['median_basin_gain'])}, and the mean over the basins moves from "
                        f"{n2(means[0])} to {n2(means[1])}")
            mid = (f" The median of {n} basins is the average of the {n // 2}th and {n // 2 + 1}th value, and here the two basins in the middle of the ranking are among "
                   f"those that gained ({join(signed(v) for v in middle)}): the median moves although the typical basin does not."
                   if len(middle) == 2 and all(v > 0 for v in middle) and r1["median_basin_gain"] < min(middle) else
                   f" The median of {n} basins is the average of its two middle values, so it can move when few basins do.")
        else:
            by_basin = f"the forecast is higher in {r1['basins_better']} of {n} basins, and the median of the {n} changes is {signed(r1['median_basin_gain'])}"
            mid = f" The median of {n} basins is the average of its two middle values, so it can move when few basins do."
        if nse_word == "mixed":
            typical_q = (f"In your rain-timing test the median NSE rises, yet the forecast is not better in {n - r1['basins_better']} of {n} basins. What does the typical "
                         f"basin do?")
            typical_head = "It hardly changes." if abs(r1["median_basin_gain"]) < 0.02 else f"It changes by {signed(r1['median_basin_gain'])} NSE."
            verdict_nse = f"So by the basin count the NSE gain is not clear, although every single seed is higher ({X('clear')})."
        else:
            typical_q = "In your rain-timing test, does the typical basin gain as much as the median says?"
            typical_head = f"Its change is {signed(r1['median_basin_gain'])} NSE."
            verdict_nse = f"By our rule the NSE gain is {'clear' if nse_word == 'clear' else 'not clear'} ({X('clear')})."
        Q("D", "typical", typical_q,
          f"""{typical_head} With rain stamped up to the evening of the issue day the median NSE {LEAD_WORD[1]} with the river gauge goes from {n2(r1['nse_base'])} to
          {n2(r1['nse_with_nldas'])}, which is {share(s1)} of the gap to the upper bound ({n2(r1['nse_perfect_forecast'])}); but {by_basin}.{mid} {verdict_nse}
          {'The firm result of the test is the onsets warned:' if stamp_gap else 'The onsets warned are not a firm result either: they go'} {on_move(r1)}
          ({on_refs(r1)}); {seeds_on}.""",
          f"""Quoting &lsquo;{share(s1)} of the gap closed&rsquo; alone. It is a difference of two medians: the basin count ({r1['basins_better']} of {n}) and the
          control ({signed(r1['gain_control'])} NSE) belong beside it. Lead with the onsets ({X('timing')}).""",
          f"""Final report section {S_TIMING} &middot; <code>results/rain_timing.json</code> (closure: basins_better, median_basin_gain, onset_seed_range; models: scores by
          basin) &middot; <code>code/10_rain_timing.py</code> &middot; <code>figures/rain_timing.png</code>""")

    trig = [SK["lstm_wq"]["tuned"][lk(k)]["trigger"] for k in LEADS]
    low = LOW_LEVEL
    i_low = FA["warning_levels"].index(low)
    sweep = SK["lstm_wq"]["tradeoff_test"]["l1"]["onset"]
    no_lower = min(trig) >= 1
    Q("D", "warnlevel", "What does the warning level do? Would a lower one not catch more floods?",
      f"""A warning is given when the forecast is above a warning level: the high-flow level itself, or a multiple of it chosen on the validation years for each model
      and lead time. """ + (f"The LSTM forecasts peaks too low ({pct(peak)} of the observed peak in the median event), so a lower level sounds right. " if peak < 1 else "")
      + (f"But the validation years did not ask for one: they chose {slash(trig)} times the high-flow level for 1 / 2 / 3 days ahead. " if no_lower else
         f"The validation years chose {slash(trig)} times the high-flow level for 1 / 2 / 3 days ahead. ")
      + f"""On the test years, as a description only, a level of {n2(low)} times would warn of {pct(sweep['pod'][i_low])} of the onsets {LEAD_WORD[1]} instead of
      {pct(on_wq[0]['pod'])}, with {pct(sweep['far'][i_low])} of those warnings false instead of {pct(on_wq[0]['far'])}. The level trades misses for false warnings;
      it adds no information. The same trade-off explains one result of the forecast experiment: {LEAD_WORD[1]} the forecast lowers the share of onsets warned at the fixed level
      and is right more often, {MOVED_POINT}{Hp('moved')}.""",
      f"""&ldquo;Just lower the level until the floods are caught.&rdquo; Choosing the level on the test years would be tuning on the test; and at {n2(low)} times the
      level {pct(sweep['far'][i_low])} of the onset warnings would be false.""",
      f"""Final report section {S_WARN} &middot; <code>results/flood_analysis.json</code> (skill, findings) &middot; <code>code/06_flood_analysis.py</code>""")

    SV = VD["services"]
    before = "before the scores were looked at" if note_with(FA["notes"], "fixed before") else "in the script, not after the fact"
    verdicts = lambda m, kind: [SV[m][lk(k)][kind]["pooled"] for k in LEADS]
    g1 = SV["lstm_wq"]["l1"]["all"]
    pers_v = VD["other_models_pooled"]["base_persistence"]["l1"]["all"]
    pers_csi = SK["base_persistence"]["tuned"]["l1"]["test"]["all"]["csi"]
    same = (f"{TT_CAP} gets the same {pers_v}: that verdict says that a high river will stay high, not that a low river will rise." if pers_v == g1["pooled"]
            else f"{TT_CAP} gets {pers_v} there (index {n2(pers_csi)} against {n2(g1['csi'])}).")
    no_gauge = [v for kind in ("all", "onset") for v in verdicts("lstm_w", kind)]
    # the upper bound: a GO of the seed-averaged forecast that not every single seed reaches is named, with the number of seeds that do reach it
    KIND = {"all": "all high-flow days", "onset": "onsets"}
    ub_cells, soft = [], []
    for kind in KIND:
        for k in LEADS:
            c = SV["lstm_wqf"][lk(k)][kind]
            seed_v = list(c["single_seeds"].values())
            ub_cells.append(c["pooled"])
            n_same = sum(v == c["pooled"] for v in seed_v)
            if c["pooled"] == "GO" and n_same < len(seed_v):
                others = sorted(set(seed_v) - {"GO"})
                soft.append(f"its GO for {KIND[kind]} {LEAD_WORD[k]} is reached by {'none' if n_same == 0 else word(n_same)} of the {word(len(seed_v))} single seeds"
                            + (f" (all {others[0]})" if n_same == 0 and len(others) == 1 else "")
                            + (", and only at the warning level chosen on the validation years, not at the high-flow level itself"
                               if c.get("at_plain_level", "GO") != "GO" else ""))
    n_go = sum(v == "GO" for v in ub_cells)

    def by_verdict(vs):
        """['CAUTION', 'NOT YET', 'NOT YET'] -> 'CAUTION 1 day ahead and NOT YET 2 and 3 days ahead'."""
        groups = {}
        for k, v in zip(LEADS, vs):
            groups.setdefault(v, []).append(k)
        return f"{vs[0]} at every lead time" if len(groups) == 1 else join(f"{v} {leads(ks)}" for v, ks in groups.items())

    def _new_stability():
        """How often the NOT YET for onsets of the sets added on 7-8 October holds when test years and basins are drawn again (results/uncertainty.json), 1 day ahead;
        empty when the file or the cells are missing. The gauge-only sets hold it in every draw; the new sets do not, so the sentence must say so."""
        UNC = R.get("uncertainty") or {}
        VS = UNC.get("verdict_stability") or {}
        cells = [("lstm_wqa.onset.l1", "with the rain up to midnight"), ("lstm_wqaF_sub.onset.l1", "with the forecast")]
        held = [(VS[c]["share_of_resamples_with_each_verdict"].get(VS[c]["verdict"], 0.0), VS[c]["share_of_resamples_with_each_verdict"].get("GO", 0.0), name)
                for c, name in cells if c in VS and VS[c].get("verdict") == "NOT YET"]
        if not held:
            return ""
        return (" (when the test years and basins are drawn again it holds " + join(f"in {pct(sh)} of the draws {name}" for sh, go, name in held)
                + f" {LEAD_WORD[1]}" + ("; no draw reaches GO" if all(go == 0 for _, go, _ in held) else "") + ")")

    Q("D", "verdict", "GO, CAUTION or NOT YET: what is the verdict, and who set the criteria?",
      f"""The criteria are our own, written into the analysis script {before}; they are not an official standard. GO is at least {pct(go_warned)} warned and
      at most {pct(go_false)} false warnings; CAUTION is not GO, but hits make up at least {pct(ca_index)} of all hits, misses and false warnings;
      otherwise NOT YET. On the test years, at the warning level chosen on the validation years, the LSTM with the river gauge is {by_verdict(verdicts('lstm_wq', 'all'))}
      for all high-flow days ({pct(g1['pod'])} warned and {pct(g1['far'])} false warnings {LEAD_WORD[1]}), and {by_verdict(verdicts('lstm_wq', 'onset'))} for onsets.
      {same} """ + ("Without a river gauge every cell is NOT YET" if set(no_gauge) == {"NOT YET"} else f"Without a river gauge the cells are {join(sorted(set(no_gauge)))}")
      + f"""; the upper bound reaches GO in {word(n_go)} of its {word(len(ub_cells))} cells"""
      + (f". That holds for the forecast averaged over its seeds: {'; '.join(soft)}." if soft else ".")
      + (f" With the inputs added on {NEW_SHORT}, the rain up to midnight and a public forecast, the verdict for onsets is {NOT_YET(AO['verdict_onset'])} at every lead time{_new_stability()}: "
         f"{pct(AO['false'][1])} of the warnings are false {LEAD_WORD[1]} with the rain, and {slash([FC['false_fc'][k] for k in LEADS[1:]], pct)} {leads(LEADS[1:])} with the forecast."
         if AO and FC and set(AO["verdict_onset"].values()) == {"NOT YET"} and set(FC["verdict_onset"].values()) == {"NOT YET"} else ""),
      (f"&ldquo;{g1['pooled']} means the warnings are fairly good.&rdquo; The same rule gives {TT} {pers_v}. " if pers_v == g1["pooled"] != "NOT YET" else
       f"Reading the verdict for all high-flow days ({g1['pooled']}) as the verdict for the warnings that matter. ")
      + f"For onsets, {pct(1 - on_wq[0]['pod'])} get no warning {LEAD_WORD[1]}.",
      f"""Final report section {S_WARN} &middot; <code>results/flood_analysis.json</code> (verdict) &middot; <code>code/06_flood_analysis.py</code>""")

    # published LSTM results (published_scores.json: each number was read at the source by an AI agent, not yet by a team member). They are set
    # beside our upper-bound rows only, never beside the forecast rows, in the words the file asks for: 'of the same size', never 'reproduced'
    ub_w = [LV["wf"][lk(k)]["nse_median"] for k in LEADS]
    try:
        if PS is None:
            raise KeyError("the file is not there")
        K19, F20 = PS["sources"]["kratzert2019"], PS["sources"]["feng2020"]
        k_one, k_avg = (K19["median_nse"][f"lstm_with_basin_properties_nse_loss_{v}"] for v in ("single", "ensemble"))
        f_none, f_day, f_three = (F20["median_nse"][v] for v in ("projection_no_observed_flow_671", "with_flow_observed_1_day_before_671",
                                                                 "with_flow_observed_3_days_before_671"))
        own = lambda s, k: LA["sets"][s]["test"]["summary"][lk(k)]["nse_median"]
        far_lead = LEADS[-1]                                           # the lead time at which our model reads the weather up to the day being forecast and no further
        our_none, our_day, our_three = own("wf", far_lead), own("wqf", LEADS[0]), own("wqf", far_lead)
        pairs = [(k_one, our_none), (f_none, our_none), (f_day, our_day), (f_three, our_three)]
        same_size = all(abs(a - b) <= SAME_SIZE for a, b in pairs)
        pub_date = date(PS["checked_on"])
        pub_unread = "No team member has read these papers yet" in PS["checked_by"]
        # the papers whose numbers the documents quote (the file lists them row by row), for the checklist
        pub_papers = list(dict.fromkeys(PS["sources"][p.split(".")[0]]["citation_key"] for row in PS["rows_of_this_project_to_set_beside_them"].values()
                                        for p in row["published"]))
        pub_range = (min(k_one, f_none, f_day, f_three), max(k_avg, f_none, f_day, f_three), min(our_none, our_day, our_three), max(our_none, our_day, our_three))
        pub_head = ("Given the same kind of information, our scores are of the same size as the published ones; it is no like-for-like test." if same_size else
                    "Our scores can be set beside the published ones only where the information is of the same kind, and even there it is no like-for-like test.")
        level = f", where there is no clear difference from the best simple model ({X('beat')})" if gauge_tie else ""
        other_rain = all(str(x.get("forcing", "")).lower() != "daymet" for x in (K19, F20))       # the published set-ups use other rain tables than ours
        Q("D", "published", "How do your scores compare with published LSTM results on CAMELS? Which of your rows is the comparable one?",
          f"""{pub_head} Published LSTM scores on CAMELS are for estimating the flow of a day from the weather up to and including that day: in our terms the
          perfect weather forecast, our upper bound. Three pairs of median NSE. Without observed flow: published {n2(k_one)} for one LSTM and {n2(k_avg)} for the
          average of {K19['n_seeds']} seeds on {count(K19['n_basins'])} basins {cite('kratzert2019')}, and {n2(f_none)} on {count(F20['n_basins'])} basins
          {cite('feng2020')}; ours {n2(our_none)}, {LEAD_WORD[far_lead]} without a gauge, where the model reads the weather up to the day being forecast and no
          further. With the flow observed the day before: published {n2(f_day)} {cite('feng2020')}; ours {n2(our_day)}, {LEAD_WORD[1]} with the river gauge. With
          flow three days old: published {n2(f_three)}; ours {n2(our_three)}, {LEAD_WORD[far_lead]}. What differs: {n} south-eastern basins against several hundred
          across the United States, other years, {'another rain product, ' if other_rain else ''}{cfg['hidden']} memory cells against {K19['n_cells']}, and
          {LEAD_WORD[1]} our model reads
          {word(len(LEADS) - 1)} more days of weather. What the published set-ups do not do is withhold the weather of the day being estimated. Our forecast rows
          do: {n2(LV['w']['l1']['nse_median'])} from past weather and {n2(LV['wq']['l1']['nse_median'])} with the river gauge {LEAD_WORD[1]}{level}.""",
          f"""&ldquo;So you reproduced the papers.&rdquo; No: other basins, years{', rain product' if other_rain else ''} and network size; &lsquo;of the same
          size&rsquo; is all that can be said. And never set a published score beside our forecast rows ({n2(LV['w']['l1']['nse_median'])} and {n2(LV['wq']['l1']['nse_median'])}): that is
          another task. The published numbers were read at the source by an AI agent on {pub_date}"""
          + ("; no team member has opened the papers yet (list at the end)." if pub_unread else "."),
          f"""Final report sections {S_SKILL}{' (the note &lsquo;How these scores relate to published results&rsquo;)' if report_says('s4_all', 'How these scores relate to published results') else ''}
          and {S_UNSEEN} &middot; <code>results/published_scores.json</code> (each entry names its table or page), <code>results/lstm_all.json</code> (sets: wf, wqf;
          decomposition) &middot; <code>code/flood.py</code> (Cfg: future)""")
    except (KeyError, TypeError, ValueError, IndexError) as err:
        print(f"results/published_scores.json was not read ({err!r}): the answer on published results quotes no published score", file=sys.stderr)
        PS = None
    if PS is None:
      Q("D", "published", "How do your scores compare with published LSTM results on CAMELS? Which of your rows is the comparable one?",
      f"""We quote no published score and did not run a published set-up, so we claim nothing side by side. What we can say is which of our rows is of the same kind.
      The LSTM studies we cite ({cite('kratzert2018')[1:-1]}; {cite('kratzert2019')[1:-1]}) model rainfall and runoff: they estimate the flow of a day from the weather
      up to and including that day, and the later one trains one model on hundreds of basins. A forecast is a harder task: {LEAD_WORD[1]}, our model must give
      tomorrow&rsquo;s flow from the weather up to today. So the row of the same kind is &lsquo;past weather + perfect weather forecast&rsquo; without the river
      gauge, our upper bound: median NSE {slash(ub_w)} for 1 / 2 / 3 days ahead, and strictly its value {LEAD_WORD[LEADS[-1]]}, where the model reads the weather
      up to the day being forecast and no further. Our forecast rows, {n2(LV['w']['l1']['nse_median'])} from past weather and
      {n2(LV['wq']['l1']['nse_median'])} with the river gauge {LEAD_WORD[1]}, answer a different question, on {n} basins.""",
      f"""Quoting a published NSE from memory, or setting it against our {n2(LV['w']['l1']['nse_median'])}: a different task, other basins, other years. And no team
      member has yet checked this reading of the papers&rsquo; set-up (list at the end).""",
      f"""Final report sections {S_ALL} and {S_LIMITS} &middot; <code>results/lstm_all.json</code> (decomposition, notes) &middot; <code>code/flood.py</code> (Cfg: future)""")

    # how sure the headline numbers are (uncertainty.json, written by 13_uncertainty.py): the range of a number when the test years and the basins are
    # drawn again (95%). It is not the seed spread and says nothing about other regions; this answer is the one place where the sheet says so.
    try:
        if UN is None:
            raise KeyError("the file is not there")
        if RT is None or not stamp_gap:
            raise KeyError("the rain-timing result that the answer sets the ranges beside is not there")
        US, UV = UN["stats"], UN["verdict_stability"]
        both = lambda name: (US[name]["both"]["lo"], US[name]["both"]["hi"])
        u_on = {s: US[f"onsets_warned.lstm_{s}.l1"] for s in ("wq", "wqn", "wqf")}
        for u_set, u_stored in (("wq", r1["onset_warned_base"]), ("wqn", r1["onset_warned_with_nldas"]), ("wqf", r1["onset_warned_perfect_forecast"])):
            if abs(u_on[u_set]["value"] - u_stored) > 0.0005:          # the resampling script must reproduce the share that the documents print
                raise ValueError(f"onsets warned of lstm_{u_set}: {u_on[u_set]['value']:.4f} in uncertainty.json, {u_stored:.4f} in rain_timing.json")
        on_rng = {s: f"{pct(u_on[s]['both']['lo'])} to {pct(u_on[s]['both']['hi'])}" for s in u_on}
        u_gap = US["onsets_gap_share_closed.wq.l1"]
        if abs(u_gap["value"] - on_gap) > 0.0005:
            raise ValueError("the share of the gap closed differs from results/rain_timing.json")
        gap_rng = f"{pct(u_gap['both']['lo'])} to {pct(u_gap['both']['hi'])}"
        over_control = US["onsets_gain.evening_rain_minus_control.wq.l1"]["both"]["share_above_zero"]
        u_diff = [US[f"nse_median_of_diffs.lstm_wq_minus_best_simple.{lk(k)}"] for k in LEADS]
        diff_text = join(f"{signed0(d['value'])} ({signed0(d['both']['lo'])} to {signed0(d['both']['hi'])})" for d in u_diff)
        either = all(d["both"]["lo"] < 0 < d["both"]["hi"] for d in u_diff)                         # every range reaches to both sides of zero
        hundredths = max(abs(v) for d in u_diff for v in (d["both"]["lo"], d["both"]["hi"])) < FEW_HUNDREDTHS    # ... and every end is closer to zero than this
        shares = lambda cell: UV[cell]["share_of_resamples_with_each_verdict"]
        onset_now = [UV[f"lstm_wq.onset.{lk(k)}"]["verdict"] for k in LEADS]
        if len(set(onset_now)) != 1:
            raise ValueError("the onset verdicts of uncertainty.json differ between the lead times")
        onset_stay = min(shares(f"lstm_wq.onset.{lk(k)}")[onset_now[0]] for k in LEADS)
        rain_stay = shares("lstm_wqn.onset.l1").get(rain_verdict)
        all_now = UV["lstm_wq.all.l1"]["verdict"]
        all_other = ", ".join(f"{v} in {pct(s)}" for v, s in shares("lstm_wq.all.l1").items() if v != all_now and s > 0)
        # the same shares at the warning levels of the verdict table (chosen on the validation years), where the file stores them
        table_cells = [f"lstm_wq.onset.{lk(k)}" for k in LEADS] + ["lstm_wq.all.l1"]
        at_table = [UV[c].get("at_level_chosen_on_validation_years") for c in table_cells]
        same_at_table = all(a and a["verdict"] == UV[c]["verdict"] and all(abs(a["share_of_resamples_with_each_verdict"][v] - x) < 0.005 for v, x in shares(c).items())
                            for a, c in zip(at_table, table_cells))
        level_note = ("For these verdicts a warning is a forecast above the high-flow level itself; at the warning levels of the verdict table the shares are the same."
                      if same_at_table else "For these verdicts a warning is a forecast above the high-flow level itself.")
        rain_too = ("" if rain_stay is None or rain_verdict != onset_now[0] else ", also with rain up to the evening" if rain_stay >= 1 and onset_stay >= 1 else
                    f" (with rain up to the evening, {LEAD_WORD[1]}: in {resamples(rain_stay)})")
        if "2.5th to 97.5th" not in UN["resampling"]["interval"]:
            raise ValueError("the ranges of uncertainty.json are not the middle 95% of the resamples")
        n_draws = UN["resampling"]["B"]
        n_test = n_years(split["test"])
        if UN["resampling"].get("n_years", n_test) != n_test or UN["resampling"].get("n_basins", n) != n:
            raise ValueError("uncertainty.json resampled other test years or basins than those of the report")
        UNC = NW["UNC"]                                            # the ranges of the new rungs (13_uncertainty.py, added on 2026-10-08); None while they are missing
        sure_new = f", {pct(UNC['rain'][0])}" if UNC else ""
        unc_text = (f" The rungs added on {NEW_SHORT} have ranges of the same width: with the rain to midnight from AORC, {AORC_WORDING}, {pct(UNC['rain'][0])} "
                    f"({pct(UNC['rain'][1])} to {pct(UNC['rain'][2])}{'; ' + HIND_W if HIND_W else ''}); with a public forecast beside it, {pct(UNC['fc2'][0])} ({pct(UNC['fc2'][1])} to {pct(UNC['fc2'][2])}) and "
                    f"{pct(UNC['fc3'][0])} ({pct(UNC['fc3'][1])} to {pct(UNC['fc3'][2])}) {leads(LEADS[1:])}. None of the three ranges "
                    f"reaches the {pct(go_warned)} that GO asks for." if UNC and UNC["rain"][2] < go_warned and UNC["fc2"][2] < go_warned and UNC["fc3"][2] < go_warned else "")
        Q("D", "sure", f"How sure are you of {pct(r1['onset_warned_base'])}, {pct(r1['onset_warned_with_nldas'])}{sure_new} and {pct(r1['onset_warned_perfect_forecast'])}, "
                       f"and of &lsquo;no clear difference&rsquo;?",
          f"""Each headline number has a range: how far it moves when the {word(n_test)} test years and the {n} basins are drawn again, {count(n_draws)} times
          (95%). Onsets warned {LEAD_WORD[1]} with the river gauge: {pct(u_on['wq']['value'])} ({on_rng['wq']}); with rain stamped up to the evening of the issue
          day: {pct(u_on['wqn']['value'])} ({on_rng['wqn']}); with the perfect weather forecast, {UPPER}: {pct(u_on['wqf']['value'])} ({on_rng['wqf']}).{unc_text} The share of
          the gap that the rain up to the evening closes is {about(on_gap)} ({gap_rng}), and its gain over the control is above zero in {resamples(over_control)}. &lsquo;No clear
          difference&rsquo;: with the river gauge, the LSTM&rsquo;s NSE minus that of the best simple model in the typical basin is {diff_text} for
          {join(LEADS)} days ahead"""
          + (": a few hundredths at most, in either direction. " if either and hundredths else ". ")
          + f"""The verdict {onset_now[0]} for onsets holds in {resamples(onset_stay)} at every lead time{rain_too}"""
          + f"""; the {all_now} for all high-flow days {LEAD_WORD[1]} holds in {resamples(shares('lstm_wq.all.l1')[all_now])}"""
          + (f" ({all_other})" if all_other else "")
          + f""". {level_note} These ranges are not the seed spread, and they say nothing about other regions.""",
          f"""&ldquo;Our {word(n_seeds)} seeds agree, so the number is sure.&rdquo; The seed spread says how far a new training moves a number ({X('seeds')}); it
          does not say what other years and other basins would give. And the ranges are wide, because {n} basins and {word(n_test)} test years are a small
          sample: quote them as they are.""",
          f"""Final report sections {S_LIMITS}{' (&lsquo;How sure are these numbers?&rsquo;)' if report_says('s8_limits', 'How sure are these numbers') else ''} and
          {S_RULE} &middot; <code>results/uncertainty.json</code> (stats, verdict_stability, how_to_read) &middot; <code>code/13_uncertainty.py</code>""")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as err:
        print(f"results/uncertainty.json was not read ({err!r}): the question on how sure the numbers are was left out", file=sys.stderr)

    # do the kinds of model miss the same onsets (same_misses.json, written by check_same_misses.py from the stored forecasts)? Pairwise shares are
    # not printed: most onsets are missed by every model, so part of any such agreement is the base rate
    same_short, same_where = "", ""
    try:
        if SM is None:
            raise KeyError("the file is not there")
        sm1 = SM["l1"]
        if sm1["n_onsets"] != n_onset[1]:
            raise ValueError(f"{sm1['n_onsets']} onsets in same_misses.json, {n_onset[1]} in flood_analysis.json")
        kinds = list(SM["kinds"])                                  # the plain names of the kinds of model that read past weather + river gauge
        lstm_set = SM["kinds"]["LSTM"]
        m_all, m_any, of_all = sm1["missed_by_all_kinds"], sm1["warned_by_any_kind"], sm1["of_missed_by_all"]
        up_share, eve_share = of_all["warned_with_perfect_forecast"]["share"], of_all["warned_with_rain_to_evening"]["share"]
        n_tf_seeds = len(SM["prediction_sets"][SM["kinds"]["Transformer"]]["seeds"])
        indep = (f" If the misses of the {word(len(kinds))} had nothing to do with each other, {pct(m_all['share_if_independent'])} of the onsets would be missed by "
                 f"all of them; {pct(m_all['share'])} are." if m_all.get("share_if_independent") is not None and m_all["share_if_independent"] < m_all["share"] else "")
        mostly = m_all["share"] > 0.5
        Q("D", "samemiss", "Do the models miss the same floods?",
          f"""{'Mostly, yes' if mostly else 'In part'}: they miss the same onsets. {cap(LEAD_WORD[1])} with the river gauge, {pct(m_all['share'])} of the
          {count(sm1['n_onsets'])} onsets are missed by all {word(len(kinds))} kinds of model ({join(kinds)}), and a warning from any one of them would cover
          {pct(m_any['share'])}, with {pct(m_any['far'])} false warnings.{indep} Of the onsets that all of them miss, the LSTM given the weather of the days being
          forecast, {UPPER}, warns of {pct(up_share)}; given rain stamped up to the evening of the issue day, of {pct(eve_share)}. So, among the models tried,
          another kind of model does not catch these onsets; more information does. (Warnings at the high-flow level itself; the Transformer is the
          {word(n_tf_seeds)}-seed model of final report section {S_COST}.)""",
          f"""&ldquo;Then combine the models.&rdquo; A warning from any of the {word(len(kinds))} covers {pct(m_any['share'])} of the onsets, against
          {pct(sm1['models'][lstm_set]['pod'])} for the LSTM alone, and {pct(m_any['far'])} of those warnings are false. And do not quote how often two models
          miss the same onset as proof: most onsets are missed by every model, so two models that seldom warn agree for that reason alone; the results file
          stores the share to expect from that beside every such share.""",
          f"""Final report section {S_LIMITS}{' (conclusions)' if report_says('s8_limits', 'same_misses') else ''} &middot; <code>results/same_misses.json</code>
          (l1, how_to_read, notes) &middot; <code>code/check_same_misses.py</code>""")
        same_short = (f" The models also miss the same onsets: {LEAD_WORD[1]} with the river gauge, {pct(m_all['share'])} of the {count(sm1['n_onsets'])} onsets are "
                      f"missed by all {word(len(kinds))} kinds of model, and the LSTM given the perfect weather forecast, {UPPER}, warns of {pct(up_share)} of those "
                      f"({X('samemiss')}).")
        same_where = ", <code>results/same_misses.json</code>"
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as err:
        print(f"results/same_misses.json was not read ({err!r}): the question on the same misses and its sentence were left out", file=sys.stderr)
        SM = None

    # ======================================================================================================================
    # E. remote sensing, GIS and GNSS
    # ======================================================================================================================
    GB = G["basins"]
    far_elev = {river(b["short"]): b["difference"]["elev_mean_m"] for b in GB.values() if abs(b["difference"]["elev_mean_m"]) > FAR_ELEV}
    inside = sum(b["gauge"]["inside_outline"] for b in GB.values())
    voids = sum(b["n_elevation_voids"] for b in GB.values())
    grid = GC["slope_mean_by_grid"]
    fine, coarse = list(grid)[0], list(grid)[-1]
    shrinks = abs(grid[coarse]["median_difference"]) < abs(grid[fine]["median_difference"])
    n_groups = len(G["sources"]["land_cover"]["groups"])
    # the member's own check of this part, as team.json records it (today's sentence while it is not entered there)
    gis_status = team.check_status("R2")
    gis_check = ("the check by the RS/GIS members is still to do" if gis_status == "still to be done" else
                 f"the own check of {html_lib.escape(team.label('R2'), quote=False)} is entered in team.json "
                 f"({html_lib.escape(gis_status.replace(' (entered by the member in team.json)', ''), quote=False)})")
    elev_text = (f"mean elevation agrees in the median ({signed(GC['elev_mean']['median_difference'], 1)} m) but is {join(n2(v, 0) for v in sorted(far_elev.values()))} m "
                 f"higher in {word(len(far_elev))} mountain basins, and the cause was not found" if far_elev and min(far_elev.values()) > 0 else
                 f"mean elevation agrees in the median ({signed(GC['elev_mean']['median_difference'], 1)} m)")
    Q("E", "rasters", "What did you compute from the rasters, and how was it checked?",
      f"""The script lays each basin outline over two {cell} m rasters, elevation and the land cover of {G['sources']['land_cover']['year']}, and computes area,
      elevation, slope, shape and the share of {word(n_groups)} land-cover groups. Checks inside the script: three ways of measuring the area agree, {inside} of {n}
      gauges lie inside their outline, and there {'are no' if voids == 0 else f'are {voids}'} empty elevation cells. Check against the CAMELS table: area agrees
      (largest difference {signed(GC['area_vs_gages2']['largest_difference'], 1)}%); {elev_text}; forest share is {n2(GC['forest_share']['mean_absolute_difference'], 1)}
      percentage points away on average. Our mean slope is {n2(ratio, 1)} times the CAMELS value{', a scale effect' if shrinks else ''}: from {fine} cells to {coarse} cells the median
      difference {'shrinks' if shrinks else 'goes'} from {signed(grid[fine]['median_difference'], 0)}% to {signed(grid[coarse]['median_difference'], 0)}%. Claude Code wrote and ran the
      script and an AI reviewer agent checked it; {gis_check}.""",
      f"""&ldquo;We used SRTM and NLCD 2011.&rdquo; The elevation tiles are a composite whose cells do not name their source, and the land cover is the
      {G['sources']['land_cover']['year']} layer of the Annual NLCD series, not the NLCD 2011 release. And slope has no value without its grid size.""",
      f"""Final report section {S_MAPS} &middot; <code>results/basin_gis.json</code> (comparison_with_camels, notes, sources) &middot; <code>code/07_basin_maps.py</code> &middot;
      <code>figures/basin_maps.png</code>, <code>figures/basin_gis_check.png</code>""")

    A_EXP = SI["experiments"]["a_rain_source"]
    aa = A_EXP["against_base"]
    sa = A_EXP["split"]
    clear_leads = lambda d: [k for k in LEADS if d[lk(k)]["nse_median"]["larger_than_seed_noise"]]
    iw, iq = clear_leads(aa["imerg_w"]), clear_leads(aa["imerg_wq"])
    know = SI["what_inputs_know_about_next_daymet_day"]
    k_sat, k_gauge = (next(v for kk, v in know.items() if kk.startswith(p))["correlation_with_daymet_rain_of_the_next_day"] for p in ("satellite rain", "gauge-based rain"))
    late = re.search(r"about ([\d.]+) months late", SD["imerg_run"])
    period_sat = SD["rain_check_period"]
    no_gauge_text = ("without the river gauge it makes no clear difference" if not iw else f"without the river gauge it changes the median NSE clearly {leads(iw)}")
    # the scores of this experiment are the mean of its single seeds, not the score of the seed-averaged forecast as in the main results (notes of the file)
    seed_mean = f", the mean of the {word(sat_seeds)} single seeds" if "single trainings, not the seed average" in note_with(SI["notes"], "Seeds:") else ""
    day_end = ((f"at {utc_end.group(1)} local standard time on the issue day" + (f" ({utc_other.group(1)} at {utc_other.group(2)})" if utc_other else ""))
               if utc_end else "in the evening of the issue day")
    if iq:
        d1 = aa["imerg_wq"][lk(iq[0])]
        sat_gain = (d1["nse_median"]["base"], d1["nse_median"]["variant"], iq[0])        # kept for the page-1 answer
        gauge_text = (f"with the river gauge it raises the median NSE clearly {leads(iq)} ({n2(d1['nse_median']['base'])} to {n2(d1['nse_median']['variant'])} "
                      f"{LEAD_WORD[iq[0]]}{seed_mean}; better in {d1['n_basins_improved']} of {n} basins)")
        stamp = (f"That gain is probably a time stamp, not better rain: the satellite&rsquo;s day is the UTC day, which ends {day_end}, and its rain of day t "
                 f"correlates {n2(k_sat)} with the rain the gauge table stamps on day t+1 (the gauge table&rsquo;s own rain of day t: {n2(k_gauge)})."
                 + (" This experiment cannot separate better timing from better amounts." if "cannot separate the two" in " ".join(SI["notes"]) else ""))
    else:
        gauge_text, stamp, sat_gain = "with the river gauge it makes no clear difference either", "", None
    final_run = (f"And it is no real-time result: this is the gauge-corrected &lsquo;Final Run&rsquo;, published about {late.group(1)} months late." if late else
                 "And it is no real-time result: the product is computed after the fact.")
    Q("E", "imerg", "IMERG against rain gauges: can satellite rain replace them?",
      f"""As rain it agrees {'only in part' if im['median_correlation'] < 0.8 else 'closely'} with the gauge-based table. Over
      {period_sat[0][:4]}&ndash;{period_sat[1][:4]} the day-by-day correlation is {n2(im['correlation_min'])} to {n2(im['correlation_max'])} (median {n2(im['median_correlation'])}), and the totals differ by {signed(im['bias_percent_min'], 0)}% to
      {signed(im['bias_percent_max'], 0)}%. As a model input, on its own split ({years(sa['train'])} / {years(sa['val'])} / {years(sa['test'])}, not comparable with the
      main results): {no_gauge_text}; {gauge_text}. {stamp} {final_run}""",
      (f"""&ldquo;Satellite rain beats rain gauges.&rdquo; The agreement is moderate; what probably helped is that the satellite&rsquo;s day ends in the evening of
      the issue day ({X('rainday', 'timing') if stamp_gap else X('rainday')}). And the real-time runs of the product were not tested.""" if iq else
       f"""&ldquo;Satellite rain can replace rain gauges.&rdquo; We saw no clear difference, with an after-the-fact product; its real-time runs were not tested."""),
      f"""Final report section {S_SAT} &middot; <code>results/satellite_data.json</code> (rain_check_summary), <code>results/satellite_inputs.json</code> (experiments, findings)
      &middot; <code>code/fetch_satellite.py</code>, <code>code/08_satellite_inputs.py</code>""")

    bb = SI["experiments"]["b_extra_inputs"]["against_base"]
    sw, sq = clear_leads(bb["soil_w"]), clear_leads(bb["soil_wq"])
    coarse_grid = re.search(r"([\d.]+) \(latitude\) x ([\d.]+) \(longitude\) degree", SD["grid"]["finding"]["GWETTOP"]["grid_assumed"])
    grid_text = f"a coarse grid ({coarse_grid.group(1)}&deg; &times; {coarse_grid.group(2)}&deg;)" if coarse_grid else "a coarse grid"
    k_soil = next(v for kk, v in know.items() if "soil wetness" in kk)["correlation_with_daymet_rain_of_the_next_day"]
    if sw:
        d1 = bb["soil_w"][lk(sw[0])]
        soil_text = (f"Without the river gauge, soil wetness raises the median NSE clearly {leads(sw)} ({n2(d1['nse_median']['base'])} to "
                     f"{n2(d1['nse_median']['variant'])}{seed_mean}; better in {d1['n_basins_improved']} of {n} basins)")
    else:
        soil_text = "Without the river gauge, soil wetness makes no clear difference"
    soil_text += ("; with the gauge it makes no clear difference at any lead time." if not sq else f"; with the gauge it changes the median NSE clearly {leads(sq)}.")
    soil_stamp = (f"Part of that gain may again be the time stamp: the day-to-day change of the top-layer wetness correlates {n2(k_soil)} with the gauge table&rsquo;s rain "
                  f"of the next day." if sw and k_soil > k_gauge else "")
    Q("E", "reanalysis", "Are your soil wetness and water vapour satellite data?",
      f"""No: both come from MERRA-2 {cite('gelaro2017')}, a reanalysis, that is a weather model that assimilates satellite and other observations; it is model
      output, not a direct satellite measurement. Its rain over land is corrected with rain-gauge analyses and drives the land model that computes the soil wetness,
      so soil wetness is not independent of the rain gauges; and it comes on {grid_text}, so each basin gets the value of the one cell that holds its centre.
      {soil_text} {soil_stamp}""",
      f"""&ldquo;We added satellite soil moisture.&rdquo; We added a modelled soil wetness from a reanalysis that is published weeks after the fact; the gain it brings
      without a river gauge is not proof that water stored in the soil adds skill.""",
      f"""Final report section {S_SAT} &middot; <code>results/satellite_data.json</code> (notes, grid), <code>results/satellite_inputs.json</code> (experiments, notes) &middot;
      <code>code/08_satellite_inputs.py</code>""")

    pw = [bb[v][lk(k)] for v in ("pw_w", "pw_wq") for k in LEADS]
    pw_nse = sum(d["nse_median"]["larger_than_seed_noise"] for d in pw)
    pw_on = [(v, k) for v in ("pw_w", "pw_wq") for k in LEADS if bb[v][lk(k)]["onset_pod"]["larger_than_seed_noise"]]
    if pw_on:
        v0, k0 = pw_on[0]
        o = bb[v0][lk(k0)]["onset_pod"]
        on_text = (f"the share of onsets warned rose clearly in {word(len(pw_on))} ({'with' if v0.endswith('wq') else 'without'} the river gauge, {LEAD_WORD[k0]}: "
                   f"{pct(o['base'])} to {pct(o['variant'])})")
    else:
        on_text = "the share of onsets warned rose clearly in none"
    hint = ("One clear change among so many comparisons is a hint to follow up, not a result." if len(pw_on) + pw_nse == 1 else
            "No clear change: no evidence that it helps." if len(pw_on) + pw_nse == 0 else "A few clear changes among so many comparisons are a hint to follow up, not a result.")
    rates = list(SI["seed_noise_rule"]["share_of_comparisons_flagged_when_there_is_no_real_difference"].values())
    # a basin count near all basins carries weight (notes of the file) and must not be left out, even where the median NSE does not change clearly
    pq1 = bb["pw_wq"]["l1"]
    k_pw = next(v for kk, v in know.items() if kk.startswith("precipitable water"))["correlation_with_daymet_rain_of_the_next_day"]
    all_basins = (f" One count must be named: with the river gauge {LEAD_WORD[1]} the forecast is higher in {pq1['n_basins_improved']} of {n} basins, although the "
                  f"change of the median NSE ({signed(pq1['nse_median']['difference'])}) is inside the seed noise"
                  + ("; the vapour is a UTC-day value, so this too may be the time stamp." if k_pw > k_gauge else ".")
                  if pq1["n_basins_improved"] >= n - 1 and not pq1["nse_median"]["larger_than_seed_noise"] else "")
    Q("E", "gnss", "What could GNSS water vapour add, and what did you actually test?",
      f"""Water vapour delays the GNSS signal. A receiver measures that delay, not the vapour; from the wet part of the delay, with the pressure and temperature
      at the station, the precipitable water above the station is derived {cite('bevis1992')}: the water vapour in the whole air column, the moisture a storm can
      draw on. We did not use GNSS data; we tested a stand-in, the daily precipitable water of the MERRA-2 reanalysis (model output), one coarse cell per basin, as
      an extra input. It changed the median NSE clearly in {word(pw_nse) if pw_nse else 'none'} of the {word(len(pw))} comparisons; {on_text}. {hint}{all_basins}
      What GNSS could add is what the stand-in lacks, a value derived at the station many times a day: a daily mean cannot show a rise within hours before a
      storm, and that was not tested.""",
      f"""&ldquo;GNSS water vapour improves flood warnings.&rdquo; Not shown: no GNSS data were used, and one clear change can be chance: by our rule
      {pct(min(rates))} to {pct(max(rates))} of comparisons are flagged as clear when nothing has changed.""",
      f"""Final report section {S_SAT} &middot; <code>results/satellite_inputs.json</code> (experiments, findings, seed_noise_rule) &middot;
      <code>code/08_satellite_inputs.py</code>""")

    # ======================================================================================================================
    # F. Transformer and cost
    # ======================================================================================================================
    CT, CV = MC["comparison"]["test"], MC["comparison"]["val"]
    tsk, vsk = CT["skill"], CV["skill"]
    over = MC["cost_of_skill"]["transformer_over_lstm"]
    r_lo, r_hi = sorted((over["seconds_per_model"], over["seconds_per_update"]))
    times = f"{n2(r_lo, 0)} to {n2(r_hi, 0)} times"
    tie = tsk["seed_ranges_overlap"]
    big = CV["l1"]["largest_basin_difference"]
    one_basin = abs(big["nse_median_transformer_minus_lstm_without_it"]) < abs(CV["l1"]["nse_median_transformer_minus_lstm"]) / 2
    val_ahead = "LSTM" if vsk["transformer_minus_lstm"] < 0 else "Transformer"
    val_text = (f"On the validation years, where a choice has to be made, the {val_ahead}&rsquo;s median is ahead by {n2(abs(vsk['transformer_minus_lstm']), 3)}"
                + (f", but most of that is one basin {LEAD_WORD[1]} ({river(big['basin'])}: {n2(big['lstm'], 3)} against {n2(big['transformer'], 3)}; without it the "
                   f"gap {LEAD_WORD[1]} is {n2(abs(big['nse_median_transformer_minus_lstm_without_it']), 3)}, not "
                   f"{n2(abs(CV['l1']['nse_median_transformer_minus_lstm']), 3)}), and in the mean over the basins the gap is "
                   f"{n2(abs(vsk['mean_nse_over_basins']['transformer_minus_lstm']), 3)}." if one_basin else "."))
    t_ahead = not tie and tsk["transformer_minus_lstm"] > 0        # the Transformer clearly ahead in the test years
    f_head = "For this set-up it is not better." if not t_ahead else "In the test years it scores higher, at a much higher cost."
    not_lstm = ", not &lsquo;the LSTM is more accurate&rsquo;" if val_ahead == "LSTM" else ""
    f_verdict = (f"So the verdict is &lsquo;not better, and roughly {times} dearer to train, for this set-up&rsquo;{not_lstm}." if not t_ahead else
                 f"So the verdict is &lsquo;ahead by {n2(tsk['transformer_minus_lstm'], 3)} NSE in the test years, at roughly {times} the training time&rsquo;.")
    Q("F", "transformer", "Why not a Transformer? Is it better than the LSTM?",
      f"""{f_head} In the test years {'there is no clear difference between the two' if tie else 'the two differ'}: NSE {n2(tsk['lstm'], 3)} for the LSTM and {n2(tsk['transformer'], 3)} for the Transformer
      (mean of the three lead times, median of {n} basins), while two seeds of one model differ by up to {n2(tsk['largest_seed_spread'], 3)}. {val_text} {f_verdict}
      It rests on {word(len(MC['seeds']))} seeds per model, one learning-rate check each, {n} basins and sequences of {MC['settings']['sequence_days']} days: it is not
      a verdict on Transformers.""",
      (f"""&ldquo;The LSTM is more accurate&rdquo;, read from the validation lead, or &ldquo;Transformers are worse for rivers&rdquo;. The test years show
      {'no clear difference' if tie else 'no lead of the LSTM'}, and only the learning rate was checked for each model.""" if not t_ahead else
       f"""Treating the Transformer&rsquo;s lead as settled. It rests on {word(len(MC['seeds']))} seeds per model and one learning-rate check each."""),
      f"""Final report section {S_COST} &middot; <code>results/model_comparison.json</code> (comparison, answer, notes) &middot; <code>code/09_model_comparison.py</code>""")

    ML, MT = MC["models"]["lstm"], MC["models"]["transformer"]
    secs = sorted(v["cost"]["seconds"] for v in ML["per_seed"].values())
    CA = {k: v for k, v in MC["causality"].items() if isinstance(v, dict)}                             # one entry per network type
    nets = sum(v["networks_tested"] for v in CA.values())
    unchanged = max(v["largest_change_up_to_day_t"] for v in CA.values()) == 0
    mask_off = CA["transformer"].get("largest_change_up_to_day_t_without_mask")
    not_retimed = bool(eased)                                      # the review's finding on training times was eased by logging the load, not by timing again
    Q("F", "cost", "What does each cost to train, and how do you know the Transformer cannot look ahead?",
      f"""One model on the laptop: the Transformer has {count(MT['n_params'])} weights against {count(ML['n_params'])} ({n2(over['parameters'], 1)} times), needs
      {n2(over['operations_per_update'], 1)} times the multiply-adds per update (counted from the formulas) and trained for {count(MT['cost']['seconds_per_model'])} s
      against {count(ML['cost']['seconds_per_model'])} s. Call that roughly {times} ({n2(over['seconds_per_model'], 1)} times in total training time,
      {n2(over['seconds_per_update'], 1)} times for a typical update): the laptop was busy, and the {word(len(secs))} LSTM seeds took
      {join(n2(s, 0) for s in secs)} s for the same {count(MC['settings']['shared']['updates'])} updates{'; the comparison was not timed again on an idle machine' if not_retimed else ''}.
      The energy ({n2(MT['cost']['energy_wh_per_model_assumed'])} Wh
      against {n2(ML['cost']['energy_wh_per_model_assumed'])} Wh) is seconds times an assumed {n2(MC['settings']['assumed_power_watt'], 0)} W; nothing was measured.
      Looking ahead: in all {nets} trained networks the inputs after a day t were replaced by random numbers, and
      {'no forecast issued up to day t changed' if unchanged else 'forecasts issued up to day t changed'}"""
      + (f"; with the mask taken off, the same test fails (the forecasts move by {n2(mask_off)})." if mask_off else "."),
      f"""Quoting {n2(over['seconds_per_model'], 1)} times as an exact cost ratio, or the watt-hours as measured. And &ldquo;the mask guarantees it&rdquo;: a mask can be
      wrong; the test is the evidence, and the trainer now refuses any network that fails it.""",
      f"""Final report section {S_COST} &middot; <code>results/model_comparison.json</code> (cost_of_skill, causality, formulas, notes) &middot;
      <code>code/09_model_comparison.py</code>, <code>code/flood.py</code>""")

    # what the Transformer of the comparison consists of: its settings from model_comparison.json; its fixed parts (encoder layers, input projection,
    # position code, causal mask, output layer) are checked against the source of 09_model_comparison.py, and a part that is not found is left out
    TFS = MC["settings"]["transformer"]
    try:
        src09 = (ROOT / "code" / "09_model_comparison.py").read_text(encoding="utf-8")
    except OSError:
        src09 = ""
    tf_code = {"encoder": "nn.TransformerEncoderLayer(" in src09 and "nn.TransformerEncoder(" in src09, "projection": "nn.Linear(n_in, d)" in src09,
               "position": "torch.sin(pos * div)" in src09 and "torch.cos(pos * div)" in src09, "mask": "generate_square_subsequent_mask" in src09,
               "head": "nn.Linear(d, len(F.LEADS))" in src09}
    for name, ok in tf_code.items():
        if not ok:
            print(f"build_qa: code/09_model_comparison.py no longer shows '{name}': its clause in the question on the Transformer was left out - check that answer",
                  file=sys.stderr)
    tf_fixed = "Fixed without a search" in " ".join(MC["notes"])                           # depth, width and heads were set, not searched (notes of the file)
    tf_parts = [f"A small encoder-only Transformer {cite('vaswani2017')}" + (", built from PyTorch&rsquo;s standard encoder layers." if tf_code["encoder"] else "."),
                (f"A linear layer turns each day&rsquo;s {MT['n_inputs']} inputs into {TFS['hidden']} numbers" if tf_code["projection"] else
                 f"Each day&rsquo;s {MT['n_inputs']} inputs become {TFS['hidden']} numbers")
                + (", and a sinusoidal position code adds the day&rsquo;s number within the sequence." if tf_code["position"] else "."),
                f"Then come {word(TFS['layers'])} layers, each with {word(TFS['heads'])} attention heads and a feed-forward block of width {TFS['feed_forward']}, with a "
                f"dropout of {TFS['dropout']:g}. In each layer every day is compared with the days before it and takes a weighted average of what they hold: that is "
                f"attention."]
    if tf_code["mask"]:
        tf_parts.append(f"A causal mask gives the later days zero weight, so the forecast issued on day t reads only the days up to t ({X('cost')}).")
    if tf_code["head"]:
        tf_parts.append(f"A last linear layer gives the flow {join(LEADS)} days ahead, as in the LSTM.")
    tf_parts.append(f"That makes {count(MT['n_params'])} weights against the LSTM&rsquo;s {count(ML['n_params'])}. Data, trainer, loss and test are the same for "
                    f"both; the sequences are {MC['settings']['sequence_days']} days long, and of its settings only the learning rate was checked "
                    f"({TFS['lr']:g}; {X('transformer')}).")
    net_table = report_says("s4_all", "The three networks at a glance")
    Q("F", "tfspec", "What exactly is your Transformer?",
      " ".join(tf_parts),
      (f"""&ldquo;Attention sees the whole year at once.&rdquo; With the mask a forecast sees only the days up to its issue day, and that was tested, not assumed
      ({X('cost')}). """ if tf_code["mask"] else "")
      + (f"""{'And it' if tf_code['mask'] else 'It'} is a small Transformer, sized for the laptop, with its depth, width and number of heads set without a search:
      the result is not a verdict on Transformers ({X('transformer')}).""" if tf_fixed else
         f"""Reading the result as a verdict on Transformers: this one is small, and only its learning rate was checked ({X('transformer')})."""),
      f"""Final report section {S_COST}{' and the table &lsquo;The three networks at a glance&rsquo; in section ' + S_NET if net_table else ''} &middot;
      <code>results/model_comparison.json</code> (settings, models, causality, notes) &middot; <code>code/09_model_comparison.py</code> (TransformerNet)""")

    # ======================================================================================================================
    # G. honesty and limits
    # ======================================================================================================================
    crit_n = "no critical" if cnt["critical"] == 0 else f"{cnt['critical']} critical"
    not_done = ER["human_review"].startswith("not yet done")
    each_script = " and one for each experiment script" if note_with(ER["notes"], "Each experiment script") else ""
    # the state of the members' own checks: today's sentences while team.json records none; once a member has entered one, the words of team.py
    g1_role = html_lib.escape(team.label("G1"), quote=False)
    g1_done = team.check_status("G1") != "still to be done"
    if none_done and not_done:
        human = f"The checks our plan gives to team members, for example the check of the scoring and of the fair-test split by {g1_role}, have not been done yet."
        verification, checks_now = "the team&rsquo;s own verification is still open", "the team&rsquo;s own checks are still open"
    elif not none_done:
        human = (f"{cap(as_clause(team.checks_sentence()))} (final report section {S_WHO}; a check counts as done when the member has entered it in "
                 f"<code>team.json</code>).")
        verification = ("each team member has entered an own check in team.json" if all_done else
                        f"{TS['n_done']} of the {TS['n']} team members {'has' if TS['n_done'] == 1 else 'have'} entered an own check in team.json")
        checks_now = as_clause(team.checks_sentence(short=True))
    else:
        human = f"The team&rsquo;s own check: {ER['human_review']}."
        verification, checks_now = "the team&rsquo;s own verification is recorded in results/engine_review.json", "the team&rsquo;s own checks are recorded in results/engine_review.json"
    # what was added on 4 October 2026 from the stored results: one sentence of the review record, in the long who-did-what statement only
    late_additions = ER.get("late_additions")
    if isinstance(late_additions, str) and late_additions.strip():
        late_additions = " " + html_lib.escape(late_additions.strip(), quote=False)
    else:
        print("results/engine_review.json has no key late_additions: the sentence on the analyses added on 4 October was left out", file=sys.stderr)
        late_additions = ""
    # the days of the work (engine_review.json, key dates): the data audit and the choice of the basins on one day, everything else on the next
    try:
        d_audit, d_rest = date(ER["dates"]["data_audit_and_basin_selection"]), date(ER["dates"]["everything_else"])
        when_text = f"the data audit and the choice of the {n} basins on {d_audit}, everything else on {d_rest}" if d_audit != d_rest else f"all of it on {d_rest}"
        when_short = f"the data audit on {d_audit}, the experiments on {d_rest}" if d_audit != d_rest else f"on {d_rest}"
    except (KeyError, TypeError, ValueError):
        when_text, when_short = f"the experiments on {date(ER['when'])}", f"the experiments on {date(ER['when'])}"
    try:                                                         # one check was reviewed and rerun a day later (engine_review.json, key dates)
        if ER["dates"]["day_window_check_rerun"][:10] > ER["dates"]["everything_else"][:10]:
            when_text += f" (the day-window check was reviewed and rerun on {date(ER['dates']['day_window_check_rerun'])})"
    except (KeyError, TypeError, ValueError):
        pass
    if NEW:                                                      # the four experiments of 7 and 8 October 2026 (report sections 4.6 to 4.8)
        when_text += f", and the experiments of final report sections {S_RAIN} to {S_FORECAST} on {NEW['days']}"
        when_short += f" and the experiments of final report sections {S_RAIN} to {S_FORECAST} on {NEW['days']}"
    new_reviews = (f" {four_experiments(R)}. {review_sentence()}{Hp('reviewed')}." if NEW else "")        # the four experiments named once (E1, E1b, E3, E2); who checked what, in the words of the project
    Q("G", "who", "What did the AI assistant do, and what did the team do?",
      f"""Our plan names Claude Code as the team&rsquo;s coding assistant, and it did the technical work: all code was written and all experiments were run by Claude
      Code under the team lead&rsquo;s direction, {when_text}. The code was then checked by independent AI reviewer agents, not by people:
      {word(len(ER['reviews']))} for the shared engine ({count(cnt['checks_passed'])} checks passed; {crit_n}, {cnt['major']} major and {cnt['minor']} minor problems
      found; {fixed_short}){each_script}.{late_additions}{new_reviews} {human} So the honest
      sentence today is: the team lead directed, Claude Code built and ran, AI agents reviewed, and {verification}. What anyone can rerun
      is the fair-test check (<code>python code/test_flood.py</code>) and the rebuild of every number in the report from the results files.""",
      (f"""&ldquo;We wrote the code and the assistant helped&rdquo;, or &ldquo;{g1_role} checked the scoring&rdquo;. Neither is true today. Say what report section
      {S_WHO} says, and add only the checks you have done yourself since (list at the end).""" if not g1_done else
       f"""&ldquo;We wrote the code and the assistant helped.&rdquo; That is not true. Say what final report section {S_WHO} says, and name only the checks that are
      entered in <code>team.json</code> (list at the end)."""),
      f"""Final report sections {S_WHO} and {S_DATA} &middot; <code>PLAN.md</code>, <code>results/engine_review.json</code>, <code>team.json</code>""")

    rf_one = bool(note_with(B["notes"], "Only this one seed"))
    if FC and AO and S4 and RV:                                    # since 8 October 2026 a real forecast and a real-time rain product have been tested, in part
        first_untested = (f"A real forecast in full: one was tested, the control member of a public reforecast{Hp('fcskill', 'design')}, so the other {word(FC['members'] - 1)} ensemble "
                          f"members, the forecast&rsquo;s temperature and the encoder&ndash;decoder form that reads a forecast day by day are untested. Real-time rain: AORC is "
                          f"{AORC_WORDING}, and Stage IV, a real-time product, was put in only at forecast time and only on the issue day of the onsets{Hp('aorc')}; a network trained "
                          f"on Stage IV, and its false warnings on days without an onset, were not tested. The satellite rain of final report section {S_SAT} is an after-the-fact product "
                          f"whose real-time runs were not tested, the water vapour is a reanalysis stand-in, and real GNSS water vapour was not tested.")
        more_rivers = (f"More rivers: {RV['n_train']['lstm_w']}, {RV['n_train']['lstm63_w']} and {RV['n_train']['unseen50_w_static']} basins of one region were tried{Hp('rivers')}, not "
                       f"the whole data set, and a median can still move with a single basin ({X('median')}).")
        forecast_trap = (f"&ldquo;a weather forecast would fix it&rdquo;: one was tested, and the verdict for onsets is {NOT_YET(AO['verdict_onset'])} with every input tried.")
    else:
        first_untested = ("A real weather forecast as an input: we have only the upper bound, so we cannot say where a real system would land. Rain as it is at hand on the day, "
                          "and real GNSS water vapour: the satellite rain is an after-the-fact product whose real-time runs were not tested, and the water vapour is a reanalysis stand-in.")
        more_rivers = f"More rivers: {n} basins of one region, in which a median can move with a single basin ({X('median')})."
        forecast_trap = "&ldquo;a weather forecast would fix it&rdquo;: a forecast was never an input."
    Q("G", "untested", "What was not tested?",
      f"""{first_untested}
      Larger models and wider searches:
      {chosen['hidden']} cells is {'a cap' if capped else 'the largest size tried'}, each network of the Transformer comparison had only its learning rate checked"""
      + (", and the random forest was fitted from one seed" if rf_one else "")
      + f""". Another training target than the squared error ({X('target')}). {more_rivers} And the model was not retrained with our own GIS values, no official flood
      stage was used, and all data are daily.""",
      f"""&ldquo;We tested satellite and GNSS inputs.&rdquo; We tested an after-the-fact satellite product and a reanalysis stand-in for GNSS. And {forecast_trap}""",
      f"""Final report section {S_LIMITS} &middot; the notes of every results file, above all <code>results/lstm_all.json</code>, <code>results/satellite_inputs.json</code>,
      <code>results/model_comparison.json</code>""")

    info_gain = ST["perfect_forecast_with_gauge"]["l1"]["nse_median_gain"]
    # 'not the network': the evidence is that different models given the same inputs score alike, while each gains when it is given more information
    # (test years, 1 day ahead; the simple model is the one the validation years chose for each set of inputs)
    simple_seq = [B["best_by_validation_same_information"][s]["l1"]["test"]["nse_median"] for s in ("w", "wq", "wqf")]
    lstm_seq = [LV[s]["l1"]["nse_median"] for s in ("w", "wq", "wqf")]
    each_gains = all(a < b for seq in (simple_seq, lstm_seq) for a, b in zip(seq, seq[1:]))
    same_inputs = (f"with the river gauge {LEAD_WORD[1]}: ridge regression {n2(SB['base_linreg_wq', 1]['nse_median'])}, random forest "
                   f"{n2(SB['base_rf_wq', 1]['nse_median'])}, LSTM {n2(wq_lstm[0])}")
    models_tie = gauge_tie and tie                                 # LSTM against the simple models with the gauge, and Transformer against LSTM in the test years
    tied = f"with the inputs that can be had in practice there is no clear difference between the models ({same_inputs}; none between the Transformer and the LSTM, {X('transformer')})"
    lstm_gains = (f"the LSTM {LEAD_WORD[1]}: {n2(lstm_seq[0])} from past weather, {n2(lstm_seq[1])} with the river gauge, {n2(lstm_seq[2])} with the weather of the "
                  f"days being forecast, {UPPER}")
    not_network = f"{tied}, while each gains when it is given more information ({lstm_gains}; the best simple model: {join(n2(v) for v in simple_seq)})"
    not_network_short = f"{tied}, and each gains when it is given more ({lstm_gains}; {X('change')})"                   # for the summary at the top
    seeds_one = LS["finals"]["wq"]["seeds"]
    sv_val = {s: mean([v["val"][lk(k)]["nse"] for k in LEADS]) for s, v in seeds_one.items()}
    sv_test = {s: mean([v["test"][lk(k)]["nse"] for k in LEADS]) for s, v in seeds_one.items()}
    best_val = max(sv_val, key=sv_val.get)
    poor_judge = sv_test[best_val] == min(sv_test.values())
    judge = (f" (on the one-river LSTM the seed with the best validation score has the lowest test score: {n2(sv_test[best_val])} against "
             f"{join(n2(v) for s, v in sv_test.items() if s != best_val)})" if poor_judge else "")
    still_word = ("is still" if rain_verdict == onset_verdict1 else "becomes") if RT is not None else ""
    moved = (f"rain stamped up to the evening of the issue day lifts it to {pct(r1['onset_warned_with_nldas'])}, {about(on_gap)} of that gap, and the verdict "
             f"{still_word} {rain_verdict}; the larger part needs a weather forecast, which was not tested" if stamp_gap and on_gap < 0.5 else
             f"rain stamped up to the evening of the issue day lifts it to {pct(r1['onset_warned_with_nldas'])}, and a real weather forecast was not tested" if stamp_gap else
             "a real weather forecast was not tested, and our rain-timing test did not show a clear time-stamp gap" if RT is not None else
             "a real weather forecast or rain up to the hour could move it, and we did not test how far")
    if AO and FC and S4 and stamp_gap:                             # since 8 October 2026 the first thing on the list, better inputs, has been tried
        moved = (f"rain stamped up to the evening of the issue day lifts it to {pct(r1['onset_warned_with_nldas'])}, {about(on_gap)} of that gap; the rain of the whole day up to midnight, "
                 f"{AORC_WORDING}, to {pct(AO['with_'][1])} ({SIDE}); and a public weather forecast, {leads(LEADS[1:])}, from "
                 f"{slash([FC['rain'][k] for k in LEADS[1:]], pct)} with the rain to midnight alone to {slash([FC['fc'][k] for k in LEADS[1:]], pct)} with the forecast beside it{Hp('aorc', 'moved')}; with every one of these inputs the verdict for "
                 f"onsets stays {NOT_YET(AO['verdict_onset'])}")
    real_share = (f"And a real forecast delivers part of the upper bound, not all of it: {slash([FC['fc'][k] for k in LEADS[1:]], pct)} {leads(LEADS[1:])}, against "
                  f"{slash([FC['perfect'][k] for k in LEADS[1:]], pct)} for the perfect rain of those days." if FC else
                  "And we have not shown how much of the upper bound a real forecast delivers.")
    inputs_first = info_gain > model_gap and models_tie and each_gains      # more information moves every model; no model stands out given the same information
    tie_word = "the lack of a clear difference between the LSTM and the simple models with the river gauge" if gauge_tie else "the comparison of the LSTM with the simple models"
    network = (f" What we do not expect to change it, among the models tried, is the network: {not_network}. Where both read the same days of weather, the LSTM "
               f"and the best simple model are never more than {n2(model_gap)} NSE apart.{same_short}" if inputs_first else "")
    Q("G", "change", "What would change the conclusions?",
      f"""Three things could. First, better inputs: our verdict for onsets ({by_verdict(verdicts('lstm_wq', 'onset'))}) is a verdict on the inputs we used, and the
      upper bound shows the room ({pct(on_wq[0]['pod'])} against {pct(on_ub[0]['pod'])} of onsets warned {LEAD_WORD[1]}); {moved}. Second, more basins and more seeds:
      {tie_word} rests on {n} basins and {word(n_seeds)} seeds, and differences inside the seed spread could fall either way. Third, the yardsticks: another high-flow
      level, or other validation years than the {word(n_years(split['val']))} uneven ones that chose every setting{judge}.{network}""",
      (f"""&ldquo;Nothing, the results are robust&rdquo;, or &ldquo;a bigger model would fix it&rdquo;. Among the models tried, the limit we found is the information
      in the inputs, not the network. Keep the first four words: a larger network was not tried ({X('untested')}). {real_share}""" if inputs_first else
       f"""&ldquo;Nothing, the results are robust.&rdquo; They rest on {n} basins, {word(n_seeds)} seeds and one validation period."""),
      f"""Final report sections {S_ALL}, {S_TIMING}, {S_WARN} and {S_LIMITS} &middot; <code>results/lstm_all.json</code> (decomposition, comparison),
      <code>results/baselines.json</code> (best_by_validation_same_information), <code>results/flood_analysis.json</code> (verdict),
      <code>results/rain_timing.json</code> (closure)""" + (same_where if inputs_first else "")
      + (f" &middot; <code>figures/{FIGURE_KEY}</code> (the key figure{': report page 1' if key_in_report else ''})" if key_fig and inputs_first else ""))

    # what it would take: a weather forecast is needed at every lead time (the larger part of the gap even 1 day ahead); rain observations up to the
    # hour add a part 1 day ahead, and with that rain the verdict for onsets does not change
    forecast_step = (f"a real weather forecast as an input, at every lead time: it is the larger part of the gap even {LEAD_WORD[1]}, and it would show where a real "
                     f"system lands between today&rsquo;s verdict and the upper bound" if stamp_gap and on_gap < 0.5 else
                     "a real weather forecast as an input, to see where a real system lands between today&rsquo;s verdict and the upper bound")
    rain_step = (f"rain observations up to the hour of the forecast, which exist today: {LEAD_WORD[1]}, rain stamped up to the evening already closes {about(on_gap)} "
                 f"of the gap in onsets warned ({pct(r1['onset_warned_base'])} to {pct(r1['onset_warned_with_nldas'])}), though the verdict for onsets {still_word} "
                 f"{rain_verdict} ({X('timing')})" if stamp_gap else
                 f"a cleaner test of rain observations up to the hour of the forecast, which exist today ({X('timing')})" if RT is not None else
                 f"rain observations up to the hour of the forecast as an input, which exist today (the rain-timing test is pending, {X('timing')})")
    own_first = ("First our own checks: each of us runs and verifies the part the plan gives us (list below); that needs no new experiment. Then, in the order our "
                 "results suggest:" if none_done else
                 f"First the rest of our own checks ({TS['n_done']} of {TS['n']} are entered in team.json; list below). Then, in the order our results suggest:"
                 if not all_done else "In the order our results suggest:")
    if FC and AO and S4:                                           # the experiments of 7 and 8 October did the three things that came first; the next steps of report section 8
        next_text = (f"Since the first version, the experiments of {NEW_DAYS} did the three things that came first: rain known up to the issue time, a real forecast as an input and more "
                     f"rivers{Hp('aorc', 'moved', 'rivers')}. {own_first} five trainings in place of {word(n_seeds)} where the NSE decides; the other {word(FC['members'] - 1)} ensemble members in place of the control member alone, and the forecast&rsquo;s temperature; "
                     f"a network trained on Stage IV, the real-time product, and a test of its false warnings on days without an onset; the encoder&ndash;decoder form that reads a forecast "
                     f"day by day {cite('nearing2024')}; satellite soil moisture as one more input; and hourly rain and flow, and a training target aimed at peaks and onsets. For the "
                     f"GNSS pair: real GNSS water vapour at sub-daily sampling in place of the reanalysis stand-in. Why five trainings: with {word(n_seeds)}, {nse_forecast_words(FC)}: "
                     f"{nse_forecast_numbers(FC)}.")
    else:
        next_text = (f"{own_first} {forecast_step}; {rain_step}; more basins, for the unseen-river test and to {'settle the river-gauge comparison' if gauge_tie else 'firm up the comparison'} with the "
                     f"simple models; and more seeds, with a wider search for the one-river LSTM{', whose chosen settings sit at the edge of the ranges tried' if edge else ''}. For the "
                     f"GNSS pair: real GNSS water vapour at sub-daily sampling in place of the reanalysis stand-in.")
    Q("G", "next", "What would you do with two more weeks?",
      next_text,
      (f"""&ldquo;Train a bigger network, or switch to a Transformer.&rdquo; The weather of the days being forecast adds {signed(info_gain)} NSE {LEAD_WORD[1]}; the
      largest gap between the LSTM and the best simple model, where both read the same days of weather, is {n2(model_gap)}, and the Transformer
      {'showed no clear difference from' if not t_ahead else 'cost far more than'} the LSTM. Among the models tried, the next gain is in the inputs.""" if inputs_first else
       f"""&ldquo;Train a bigger network, or switch to a Transformer.&rdquo; Our results do not say that a bigger network would help: nothing larger than
      {chosen['hidden']} cells was tried."""),
      f"""Final report section {S_LIMITS} &middot; <code>PLAN.md</code> (risks and fallbacks) &middot; <code>results/rain_timing.json</code>,
      <code>results/lstm_lobo.json</code>""")

    # a negative result is a result: the plan allowed for it (its milestone 2 and its risk table are quoted word for word from PLAN.md)
    if not gauge_better:
        m2_plan, risk_plan = (html_lib.escape(t, quote=False) for t in plan_quotes())
        if not (m2_plan and risk_plan):
            print("build_qa: milestone 2 or the risk 'LSTM does not beat the baselines' was not found in PLAN.md: the answer on the unmet milestone does not quote it",
                  file=sys.stderr)
        allowed = ("Our plan allowed for it: " + " and ".join(
            ([f"milestone 2 reads &ldquo;{m2_plan}&rdquo;"] if m2_plan else []) + ([f"its risk table says of this outcome &ldquo;{risk_plan}&rdquo;"] if risk_plan else []))
                   + ". ") if m2_plan or risk_plan else ""
        learned = (f"What we learned instead is the finding: among the models tried, the limit is the information in the inputs, not the network ({X('change')})."
                   if inputs_first else f"What we learned instead is where the skill comes from ({X('change')}).")
        what_next = (f"The same result said what to do next, and the experiments of {NEW_DAYS} did it: rain up to the issue time and a weather forecast moved the onsets warned; "
                     f"more rivers moved the median NSE for rivers never seen, from {n2(RV['unseen']['lobo_w_static'][1])} to {n2(RV['unseen']['unseen50_w_static'][1])}, but not the onsets warned "
                     f"for the {RV['n_train']['lstm_w']} gauged rivers, which went from {pct(RV['onset']['lstm_wq'][1])} to {pct(RV['onset']['lstm63_wq'][1])} ({MOVED_POINT}); the kind of "
                     f"model changed little{Hp('holds', 'rivers')}; what is still open is in {X('next')}." if FC and AO and NEW and RV_MOVED else
                     f"The same result said what to do next, and the experiments of {NEW_DAYS} did it: rain up to the issue time and a weather forecast moved the onsets warned, the kind "
                     f"of model changed little{Hp('holds')}; what is still open is in {X('next')}." if FC and AO and NEW else
                     f"The same result says what to do next: a weather forecast as an input, rain observations up to the hour, more basins ({X('next')}).")
        Q("G", "failed", "The LSTM did not beat your simple models. Did the project fail?",
          f"""No. It tested an expectation and found out why it did not hold. {allowed}With the river gauge the milestone is not met: between the LSTM and the best
          simple model {say(cls['wq'])} (median NSE {n2(wq_lstm[0])} against {n2(wq_base[0])} {LEAD_WORD[1]}). {learned} {what_next}""",
          (f"""Rescuing the LSTM (&ldquo;without the gauge it is ahead in {w_wins[0]} of {n} basins&rdquo;: only as a {word(n_seeds)}-seed average; a single training
          is not clearly ahead), or apologising. """ if w_avg_only else "Apologising, or looking for a comparison in which the LSTM is ahead. ")
          + "Say that the milestone is not met with the river gauge, then say what the result shows.",
          f"""Final report sections {S_RULE}, {S_SKILL} and {S_LIMITS} &middot; <code>PLAN.md</code> (milestones; risks and fallbacks) &middot;
          <code>results/lstm_all.json</code> (comparison, decomposition)""")

    # what the project teaches about applying AI: the lessons of report section 8, each tied to a result and printed only where that result holds
    lessons = []
    if inputs_first and not t_ahead:
        lessons.append(f"<b>Ask what the model is given before asking which model.</b> The same LSTM, run with {word(len(C))} sets of inputs up to an upper bound, showed "
                       f"where the room is; a newer network did not: the Transformer took {times} as long to train for no clear gain ({X('transformer')}).")
    if gauge_tie:
        lessons.append(f"<b>The yardstick is a simple model given the same inputs, and more than one training.</b> With the river gauge the LSTM is higher than {TT} in "
                       f"{join(pers_wins)} of {n} basins; against the best simple model given the same inputs there is no clear difference, and {word(n_seeds)} "
                       f"trainings of the same LSTM differ by {n2(seed_spread1)} NSE {LEAD_WORD[1]} ({X('beat', 'seeds')}).")
    lessons.append(f"<b>Score what the user needs.</b> {pct(g1['pod'])} of all high-flow days are warned {LEAD_WORD[1]}, yet {pct(1 - LV['wq']['l1']['onset_pod'])} of "
                   f"the onsets get no warning" + (", and a model that only repeats today&rsquo;s flow gets the same verdict" if pers_v == g1["pooled"] else "")
                   + f" ({X('verdict')}).")
    if stamp_gap:
        lessons.append(f"<b>Know the clock of every input.</b> Which hours a daily value covers decides what a model can know when the forecast is issued; here rain "
                       f"stamped up to the evening of the issue day closes {about(on_gap)} of the gap in onsets warned {LEAD_WORD[1]} ({X('timing')})"
                       + (f", and the whole day up to midnight, from {AORC_WORDING}, takes it to {pct(AO['with_'][1])}{hw} while the same hours over the rain table&rsquo;s own window leave it at "
                          f"{pct(AO['to09'])}{Hp('hours')}" if AO else "") + ".")
    if len(lessons) > 1:
        ub_clause = f" With the upper bound&rsquo;s information {PHRASE_UB[cls['wqf'][k3]]} {LEAD_WORD[k3]}." if cls["wqf"][k3] == "lstm" else ""
        teach = report_says("s8_limits", "What this teaches about applying AI")
        Q("G", "lesson", "What does this project teach about applying AI?",
          f"""{cap(word(len(lessons)))} lessons, each from a result of ours. {' '.join(lessons)}""",
          f"""&ldquo;Simple models are as good as deep learning&rdquo;, or &ldquo;LSTMs do not work for floods&rdquo;. Our result holds for {n} basins,
          {n_years(split['train'])} training years and a network of {cfg['hidden']} memory cells on a laptop.{ub_clause} The published studies train one model on
          hundreds of basins ({X('published')}). The lesson is a method: find what limits the forecast before changing the model.""",
          f"""Final report section {S_LIMITS}{' (&lsquo;What this teaches about applying AI&rsquo;)' if teach else ''} &middot; <code>results/lstm_all.json</code>
          (comparison, decomposition), <code>results/model_comparison.json</code>, <code>results/flood_analysis.json</code> (verdict),
          <code>results/rain_timing.json</code> (closure)""")

    # ======================================================================================================================
    # H. the experiments of 7 and 8 October 2026 (report sections 4.6 to 4.8): rain to midnight, a forecast, more rivers
    # Every number comes from the results files through new_work_facts(); a question whose facts are missing is left out (H_OK), and no other
    # answer then refers to it (Hx). The fixed wordings (AORC_WORDING, MOVED_POINT, INFO_NOT_NET, REALTIME_WORDING and, through the helpers of s0_summary, the hindsight, NSE and review
    # wordings: hindsight_clause, nse_forecast_words, nse_forecast_numbers, review_sentence, four_experiments) are those of the report, used word for word.
    # ======================================================================================================================
    before = {i["key"] for i in items}                                # the questions of A to G, which an answer of this group may point to
    beforex = lambda *keys: X(*[k for k in keys if k in before])
    CLOCK = NW["CLOCK"]                                               # the clock facts: H1, H2 and H3 quote them (None while a record is missing; those questions need it)
    if "aorc" in H_OK:
        n_on = S4["n"]
        Q("H", "aorc", "Why is AORC rain allowed, when it is made after the fact?",
          f"""Because the question is what the model could read at midnight, and the input was built to read nothing later. But AORC, an hourly rain product of NOAA, is
          {AORC_WORDING} {cite('aorc')}: its daily totals come from rain gauges and are spread over the hours with radar, with corrections about {word(AO['lag_days'])} days after the hour, and its hours
          before {CLOCK['early']}, which are all the training years, are only disaggregated daily totals. So it is not the rain a service would have, and the result says what rain up
          to midnight, known as well as AORC knows it ({AORC_WORDING}), is worth, not what a service delivers. Four checks stand behind it. <i>The clock:</i> every hour belongs to the wall-clock day of the gauge, so the
          last hour ends at local midnight, the issue time; the reviewer rebuilt the clock with its own code and added {CLOCK['plus_mm']} mm after the end of each window in
          {CLOCK['cases']} cases (random basin-days): no value of day t moved. <i>The stamp:</i> AORC stamps the end of the hour (r = {n2(CLOCK['r_lag0'])} against hourly Stage IV
          at lag 0, {n2(CLOCK['r_off'])} one hour off), and the opposite convention would cost {n2(abs(CLOCK['opp_points']), 1)} points ({pct(CLOCK['opp_from'])} to
          {pct(CLOCK['opp_to'])}). <i>Hindsight:</i> in the {pct(AO['with_'][1])}, {HIND_W}{Hp('hindsight')}. <i>Real time:</i> with Stage IV {cite('stage4')}, the radar-and-gauge
          product that is about an hour behind real time, in place of AORC on the issue day of all {count(n_on)} onsets, the same networks warn of {count(S4['stage4'])}
          ({pct(S4['stage4'] / n_on)}) against {count(S4['aorc'])} ({pct(S4['aorc'] / n_on)}) with AORC and {count(S4['table'])} ({pct(S4['table'] / n_on)}) with the rain table alone:
          {pct(S4['share'])} of the effect survives (95% interval {pct(S4['interval'][0])} to {pct(S4['interval'][1])}), and {count(S4['no_last'])}
          ({pct(S4['no_last'] / n_on)}) with the last hour of the issue day taken out. These two figures are {REALTIME_WORDING}: a service with these inputs might reach about
          {pct(S4['no_last'] / n_on)} to {pct(S4['stage4'] / n_on)}, not {pct(AO['with_'][1])}. {'Whether the Stage IV archive holds each hour as it was first posted or in a later, corrected version is not established (<code>stage4_realtime.json</code>, honesty_points). ' if S4_ARCHIVE else ''}And the verdict for onsets stays {NOT_YET(AO['verdict_onset'])} ({pct(AO['false'][1])} of the warnings false).""",
          f"""&ldquo;AORC is observed rain&rdquo;, or &ldquo;a service would warn of {pct(AO['with_'][1])}&rdquo;. AORC is {AORC_WORDING}; with Stage IV in its place the figure is
          {pct(S4['no_last'] / n_on)} to {pct(S4['stage4'] / n_on)}, and that is {REALTIME_WORDING}. And say what the real-time test did not do: the networks were trained on AORC and Stage IV was put in at forecast time
          only (nothing is said about training on it), and false warnings were counted only on the issue days of the {n} basins ({pct(S4['far_stage4'])} with Stage IV against
          {pct(S4['far_aorc'])} with AORC), not on days without an onset.""",
          f"""Final report section {S_RAIN} &middot; <code>results/aorc_rain_to_midnight.json</code> (inputs: clock; stamp_convention_check; outcome_caveat),
          <code>results/stage4_realtime.json</code>, <code>results/review_aorc_rain_to_midnight.json</code> (checks b3, c2) &middot; <code>code/16_aorc_rain_to_midnight.py</code>,
          <code>code/fetch_aorc.py</code>, <code>code/check_stage4_realtime.py</code> &middot; <code>figures/aorc_rain_to_midnight.png</code>, <code>figures/stage4_realtime.png</code>""")

    if "hindsight" in H_OK:
        HIND = NW["HIND"]
        Q("H", "hindsight", "Could the model peek at tomorrow&rsquo;s rain through AORC?",
          f"""The channel exists by construction, so it was measured and not argued away. AORC&rsquo;s daily totals sit on a gauge day that ends at {CLOCK['gauge_day_utc']} UTC, in the morning after
          the issue time, and in the test years are spread over the hours by the shape of the Stage IV radar: an hour before midnight could carry a factor set by a day that
          includes rain after the issue time. The independent review regressed the AORC rain before midnight on the Stage IV rain of the same hours and of the hours after midnight, on
          {HS['days_review']} test-year days ({count(HIND['basin_days'])} basin-days; the {HIND['my_days']} days the reviewer added were picked for onsets and night storms and hold
          {n2(HIND['night_mine'], 1)} mm of Stage IV rain after midnight on average, against {n2(HIND['night_builder'], 1)} mm in the builder&rsquo;s own {HS['days_builder']}): the
          coefficient on the later rain is {signed(HS['coef_review'], 3)} mm per mm (95% interval {signed(HS['ci_review'][0], 3)} to {signed(HS['ci_review'][1], 3)}), nothing
          positive. The {HIND['dry_days']} basin-days that were dry before midnight and wet after (on average {n2(HIND['dry_after'], 1)} mm after) hold {n2(HIND['dry_before'], 3)} mm of
          AORC rain before midnight. A leak of {pct(HIND['planted_share'])} of the night rain, planted in the data, is found (coefficient {signed(HIND['planted_b'], 3)}), so the test
          can see one. And with Stage IV in place of AORC on the issue day of the {HIND['n_covered']} onsets of those days, the saved networks warn of {HS['raw']} to
          {HS['matched']} against {HS['aorc']} with AORC (the rain table alone: {HS['table']}). So the best estimate of the hindsight in the {pct(AO['with_'][1])} is
          {n2(HS['best'], 0)} points, and {HIND_W} ({HS['bound_hits']} of {AO['hits']} hits).""",
          f"""&ldquo;The review proved that AORC holds no hindsight.&rdquo; It measured it on {HS['days_review']} days and on {HIND['n_covered']} of the {count(HIND['n_onsets'])} onsets
          ({pct(HIND['n_covered'] / HIND['n_onsets'])}); the builder&rsquo;s own {HS['days_builder']} days held too little night rain to exclude a leak of up to
          {pct(HIND['planted_share'])} (coefficient {signed(HS['coef_builder'], 2)}, interval {signed(HS['ci_builder'][0], 2)} to {signed(HS['ci_builder'][1], 2)}). Say &lsquo;best
          estimate {n2(HS['best'], 0)} points; {HIND_W}&rsquo;. The later test on all {count(S4['n'])} onsets ({Hx('aorc')}) measures the real-time figure; it does not prove
          the absence of hindsight.""",
          f"""Final report section {S_RAIN} &middot; <code>results/review_aorc_rain_to_midnight.json</code> (checks d3 to d7), <code>results/aorc_hindsight.json</code> (numbers),
          <code>results/stage4_realtime.json</code> &middot; <code>code/check_aorc_hindsight.py</code>, <code>code/check_stage4_realtime.py</code>, <code>code/review/aorc_rain_to_midnight/</code>""")

    if "hours" in H_OK:
        HOURS = NW["HOURS"]
        Q("H", "hours", f"Rain to midnight from ERA5 warned of {pct(E1['with_'])} of the onsets and from AORC of {pct(AO['with_'][1])}. How do you know the gain comes from the hours, "
                        f"and not from a second rain product?",
          f"""By controls that differ from the input in one thing only, the same design in both experiments. The same AORC hours summed over the window of the rain table itself (the 24 hours ending at {CLOCK['table_end']}), a second product with no new hour, warn of {pct(AO['to09'])} of the onsets: flat against {pct(AO['base'][1])} with the rain table alone ({signed(AO['second_product'], 1)}
          points). Summed over the UTC day, which ends about {CLOCK['utc_end']}, they warn of {pct(AO['utc'])}; over the whole local day, up to midnight, of {pct(AO['with_'][1])}{hw}. So the hours after
          about {CLOCK['table_end']} are worth {n2(AO['hours'], 1)} points (every training above every control training), {n2(AO['evening'], 1)} of them after {CLOCK['utc_end']}; {HOURS['only_rain']} onsets are
          warned only with the rain to midnight, {HOURS['only_control']} only by the control. ERA5 gives the same picture at a smaller size: {pct(E1['to09'])} over the table&rsquo;s window,
          {pct(E1['utc'])} over the UTC day, {pct(E1['with_'])} to midnight. The product counts as well: over the same hours AORC warns of {n2(AO['era5'], 1)} points more than ERA5, model rain
          against an analysis anchored to gauges and radar. The success rule fixed in advance, clearly above the {pct(E1['nldas'])} of the rain up to the evening with a flat control, was
          {'met' if AO['met'] else 'not met'} by AORC and {'met' if E1['met'] else 'not met'} by ERA5.""",
          f"""&ldquo;A better rain product explains it.&rdquo; Over the table&rsquo;s own window the better product adds nothing ({signed(AO['second_product'], 1)} points); the gain appears
          only when the hours after about {CLOCK['table_end']} are in. And &ldquo;ERA5 and AORC disagree, so one of them is wrong&rdquo;: both show the gain of the hours, and they differ in size because they
          are different products. Quote the two numbers of the control together, {pct(AO['to09'])} and {pct(AO['with_'][1])}.""",
          f"""Final report section {S_RAIN} &middot; <code>results/aorc_rain_to_midnight.json</code> (value_of_hours_09_to_24, second_rain_product_alone, aorc_against_era5),
          <code>results/rain_to_midnight.json</code> (e1_success, value_of_hours_09_to_24) &middot; <code>code/15_rain_to_midnight.py</code>,
          <code>code/16_aorc_rain_to_midnight.py</code> &middot; <code>figures/aorc_rain_to_midnight.png</code>""")

    if "design" in H_OK:
        FCX = NW["FCX"]
        sm, ss = FCX["split_main"], FCX["split_sat"]
        Q("H", "design", f"The forecast archive starts in {FC['years'][0]}. How was it given to a network that was trained on {years(sm['train'])}?",
          f"""Two designs, each with its own test. <i>E2a, substitution:</i> the network is trained on {years(sm['train'])} with three extra inputs, the AORC rain of the days being forecast, a
          perfect rain forecast, because no forecast archive of this resolution covers those years; where it stops is chosen on {years(sm['val'])}, which lie inside the archive, with the
          forecast in those slots; and it is scored on the usual test years {years(sm['test'])} with the forecast in the slots, so its numbers stand beside every earlier row. <i>E2b,
          retraining:</i> the network is trained on {years(ss['train'])}, validated on {years(ss['val'])} and tested on {years(ss['test'])} (the satellite-era split of report section
          {S_SAT}), so that it learns the forecast&rsquo;s own errors. The forecast is the control member of the GEFSv12 reforecast {cite('gefsv12')}, basin means by exact cell overlaps, from the
          {CLOCK['run_utc']} UTC run of the UTC date t+1, the latest run that exists at the issue time ({FC['hours_old'][0]} to {FC['hours_old'][1]} hours old), with one bias factor per basin and lead
          fitted on the validation years only. E2b&rsquo;s numbers are its own: its base is weaker (the rain to midnight warns of {slash([FC['s_rain'][k] for k in LEADS], pct)} there, against
          {slash([FC['rain'][k] for k in LEADS], pct)} on the main split) and it has {slash([FC['n_sat'][k] for k in LEADS], count)} onsets against
          {slash([FC['n'][k] for k in LEADS], count)}.""",
          f"""Setting E2a and E2b side by side as one result. They differ in training years, test years, base and number of onsets: the contrast between their gains
          ({slash([FC['gain'][k] for k in LEADS], lambda v: signed(v, 0))} points for E2a, {slash([FC['s_gain'][k] for k in LEADS], lambda v: signed(v, 0))} for E2b, at the fixed warning
          level) mixes two splits and two bases and is not a measurement.""",
          f"""Final report section {S_FORECAST} &middot; <code>results/forecast_input.json</code> (split_main, split_satellite_era, inputs: clock, retrained_vs_substituted),
          <code>results/gefs_basins.json</code> (rule, clock) &middot; <code>code/fetch_gefs.py</code>, <code>code/gefs_to_basins.py</code>, <code>code/18_forecast_input.py</code>
          &middot; <code>figures/forecast_input.png</code>""")

    if "fcskill" in H_OK:
        FCX = NW["FCX"]
        Q("H", "fcskill", "How good is the weather forecast itself?",
          f"""Moderate, and worse every day ahead. The forecast is the control member of the GEFSv12 reforecast {cite('gefsv12')}, averaged over each basin by exact overlaps with its
          grid cells ({FC['n_cells'][0]} to {FC['n_cells'][1]} cells of {n2(FCX['box_deg'], 2)} degree), summed over the local days t+1 to t+3. Against AORC daily rain its correlation (median of the
          {FC['n_basins']} basins) is {slash([FC['corr'][k] for k in LEADS])} at {slash(LEADS, str)} days; its mean rain is {slash([FC['ratio'][k] for k in LEADS])} of AORC&rsquo;s (1 day ahead:
          {n2(FC['ratio_summer'])} in summer, {n2(FC['ratio_winter'])} in winter), and on the onset days it carries only {n2(FCX['ratio_onset_days'])} of the rain: it is light where the
          onsets are. The Brier skill for days with at least {FC['brier_event']} of rain, with the single member&rsquo;s forecast turned into a probability by a monotone map fitted on the validation years and scored on the test years, is {slash([FC['brier'][k] for k in LEADS])}; the published numbers for the whole ensemble are
          {n2(FC['published'][0])} and {n2(FC['published'][1])} at {FCX['brier_windows'][0][0]}&ndash;{FCX['brier_windows'][0][1]} h and {FCX['brier_windows'][1][0]}&ndash;{FCX['brier_windows'][1][1]} h {cite('guan2022')}: of the same order, but computed differently and not reproduced. One bias
          factor per basin and lead, fitted on the validation years, corrects the mean, not the seasons or the heavy-rain days.""",
          f"""&ldquo;Our forecast is as skilful as the published GEFSv12 forecast.&rdquo; It is one member against the whole ensemble, the Brier skill was computed in another way, and no team member
          has opened that paper yet. And &ldquo;a correlation of {n2(FC['corr'][1])} means the forecast is {pct(FC['corr'][1])} right&rdquo;: it is a correlation of daily rain, not a share of
          correct forecasts.""",
          f"""Final report section {S_FORECAST} &middot; <code>results/gefs_basins.json</code> (skill, bias, realtime_margin), <code>results/forecast_input.json</code> (forecast_data),
          <code>results/published_context.json</code> (Guan et al.) &middot; <code>code/gefs_to_basins.py</code>""")

    if "moved" in H_OK:
        FCX = NW["FCX"]
        ch, bu, bo = FCX["chosen"], FCX["budgets"], FCX["boot"]
        Q("H", "moved", f"Why does the forecast lower the onsets warned {LEAD_WORD[1]}, from {pct(FC['pod'][0])} to {pct(FC['pod'][1])}?",
          f"""At the fixed warning level it does, and for the warnings that is {MOVED_POINT}. The model given the forecast beside the rain to midnight warns of {pct(FC['pod'][1])} of the onsets instead of
          {pct(FC['pod'][0])} ({signed(FC['gain'][1], 1)} points; {reading_word(FC['gain_reading'][1])} by the seed rule), but only {pct(FC['far'][1])} of its onset warnings are false instead of
          {pct(FC['far'][0])}, and its CSI (hits, misses and false warnings together) is {n2(FC['csi'][1], 3)} against {n2(FC['csi'][0], 3)}: fewer warnings, more often right. The NSE is another matter,
          since it has no warning level that could move. {cap(nse_forecast_words(FC))}: {nse_forecast_numbers(FC)}. At the
          warning level that the validation years choose it warns of {pct(ch['onsets_warned_with'])}, {signed(FC['chosen_gain'][1], 1)} points more than {pct(ch['onsets_warned_without'])}; at equal
          false-warning budgets it is {signed(bu['points_at_most_' + str(FCX['budgets_pct'][0])], 1)} points ahead (at most {FCX['budgets_pct'][0]}% of the onset warnings false) and
          {signed(bu['points_at_most_' + str(FCX['budgets_pct'][1])], 1)} (at most {FCX['budgets_pct'][1]}%);
          and without any warning level, the average precision of its onset warnings is {n2(FC['ap'][1][1])} against {n2(FC['ap'][1][0])}. Retrained with the forecast (E2b, its own split) it gains
          {signed(FC['s_gain'][1], 1)} points at the fixed level, with a harmless control: no loss. Why the fixed level shows fewer is not isolated. That the network trusts slots that were exact in
          training too much is a hypothesis: for it, the same networks warn of {pct(FCX['perfect_slots'])} given the perfect rain and {pct(FC['ctrl'][1])} given a shuffled forecast; against it, or not
          separating it from other causes, {FCX['lost_even']} of the {FCX['lost']} onsets that are lost are lost even with the perfect rain in the slots, the difference sits in the {FCX['small_n']}
          onsets with little rain on the target day ({pct(FCX['small_without'])} warned without the forecast, {pct(FCX['small_with'])} with it) and in {FCX['fewer_basins']} of {FC['n_basins']} basins, and the
          forecast carries only {n2(FCX['ratio_onset_days'])} of the AORC rain on the onset days.""",
          f"""&ldquo;The forecast makes the 1-day warning worse&rdquo;, or &ldquo;it loses nothing at 1 day.&rdquo; At the fixed level the share of onsets warned falls by {n2(abs(FC['gain'][1]), 1)}
          points, which is borderline by a block bootstrap over the test days ({signed(bo['pod_points_ci95'][0], 1)} to {signed(bo['pod_points_ci95'][1], 1)} points) and is above zero in
          only {bo['water_years_above_zero']} of {bo['n_water_years']} test years; at the level chosen on the validation years it rises by {n2(FC['chosen_gain'][1], 1)}. Say that the
          warning point moved, that the 1-day NSE is lower too (which is not a moved point), and do not explain it by over-trust as if that had been shown. {leads(LEADS[1:]).capitalize()} the forecast adds {slash([FC['gain'][k] for k in LEADS[1:]], lambda v: signed(v, 0))}
          points at the fixed level, clear in every training.""",
          f"""Final report section {S_FORECAST} &middot; <code>results/forecast_input.json</code> (value_of_the_forecast: operating_points; lead1_facts; retrained_vs_substituted;
          outcome_caveat) &middot; <code>code/18_forecast_input.py</code> &middot; <code>figures/forecast_input.png</code>""")

    if "shuffled" in H_OK:
        FCX = NW["FCX"]
        Q("H", "shuffled", f"With the forecast shuffled within the month the onsets warned fall to {slash([FC['ctrl'][k] for k in LEADS], pct)}, far below the model that has no forecast at all. "
                           f"Why does that control collapse, and what does it show?",
          f"""The control gives the network a forecast of the right climate and the wrong day: each day&rsquo;s value was taken from another year of the same month. The network of E2a was trained on exact
          rain in those three slots, so it learned to trust them, and given noise it is misled: onsets warned {slash([FC['ctrl'][k] for k in LEADS], pct)} against
          {slash([FC['rain'][k] for k in LEADS], pct)} with no slots at all, median NSE {slash([FC['nse_ctrl'][k] for k in LEADS])} against {slash([FC['nse_rain'][k] for k in LEADS])}. So this control
          measures harm, not information: it is not a neutral baseline, and being {slash([FCX['above_ctrl'][k] for k in LEADS], lambda v: signed(v, 0))} points above it says little. The clean control is
          E2b&rsquo;s: a network trained and run on shuffled slots warns of {slash([FC['s_ctrl'][k] for k in LEADS], pct)}, the same as the model without any forecast
          ({slash([FC['s_rain'][k] for k in LEADS], pct)}), so the {slash([FCX['above_ctrl_s'][k] for k in LEADS], lambda v: signed(v, 0))} points that the forecast adds over it there are the
          forecast&rsquo;s information.""",
          f"""&ldquo;The forecast is worth {n2(FCX['above_ctrl'][2], 0)} points more than its control.&rdquo; Not as a measure of information: that comparison is with a network that has been
          misled. Read the information from E2b&rsquo;s control, and say that the E2a control measures harm.""",
          f"""Final report section {S_FORECAST} &middot; <code>results/forecast_input.json</code> (value_of_a_misleading_input, forecast_vs_climatology_control, retrained_vs_substituted;
          outcome_caveat) &middot; <code>code/18_forecast_input.py</code> &middot; <code>figures/forecast_input.png</code>""")

    if "rivers" in H_OK:
        n12, n63, n11, n50 = RV["n_train"]["lstm_w"], RV["n_train"]["lstm63_w"], RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
        fst = lambda d: slash([d[k] for k in LEADS])
        Q("H", "rivers", f"Why do {n63} training rivers not help the forecast for the {n12} gauged rivers, but help rivers never seen? Is that the effect of scale?",
          f"""Because the two forecasts lack different things. For a gauged river the gauge already says how wet the basin is; what is missing is tomorrow&rsquo;s rain, and no number of other rivers
          supplies it. With {n63} training rivers instead of {n12}, the median NSE of the gauged forecast shows {S0.by_lead_reading(RV['nse_reading'])} ({slash([RV['nse']['lstm_wq'][k] for k in LEADS])} to
          {slash([RV['nse']['lstm63_wq'][k] for k in LEADS])}; {LEAD_WORD[1]} it is higher in {RV['nse_basins'][1]} of {RV['nse_n']} rivers, but not clear of the seeds); the onsets warned go from {pct(RV['onset']['lstm_wq'][1])} to
          {pct(RV['onset']['lstm63_wq'][1])} because the operating point moved ({RV['hits'][0] - RV['hits'][1]} fewer hits, {RV['false_alarms'][0] - RV['false_alarms'][1]} fewer false warnings, CSI
          {n2(RV['csi'][0], 3)} against {n2(RV['csi'][1], 3)}; at the level chosen on the validation years {pct(RV['chosen'][1], 1)} against {pct(RV['chosen'][0], 1)}). For a river without a gauge the
          network must learn from other rivers what such a basin does, and there more rivers matter: the median NSE {LEAD_WORD[1]} rises from {n2(RV['unseen']['lobo_w_static'][1])} with {n11} training
          rivers to {n2(RV['unseen']['unseen50_w_static'][1])} with {n50} (from {n2(RV['unseen']['lobo_w_nostatic'][1])} to {n2(RV['unseen']['unseen50_w_nostatic'][1])} without the basin properties),
          and the loss from not having seen the river shrinks from {n2(RV['loss'][0])} to {n2(RV['loss'][1])} NSE. Whether that is scale cannot be said: the number, the variety and the nearness of the
          training rivers changed together (the nearest training gauge is a median of {n2(RV['km'][0], 0)} km away with {n11} rivers, {n2(RV['km'][1], 0)} km with {n50}), and the gain does not rank
          with the distance (rank correlation {signed(RV['rho'][0], 2)}, p = {n2(RV['rho'][1])}), so it can be credited neither to scale nor to nearness alone. Not done: a buffered leave-out, or a
          dose-response over subsets of the {n50}.""",
          f"""&ldquo;More rivers always help&rdquo;, or &ldquo;the {n63}-river run shows that more data are useless.&rdquo; More rivers did not lift the gauged forecast {LEAD_WORD[1]}; they lifted
          rivers never seen, in a comparison with an earlier test of one training per river, against a {word(RV['n_seeds'])}-seed average here (each single {n50}-river training is
          {signed(min(RV['single_gain']), 2)} to {signed(max(RV['single_gain']), 2)} above, p = {n2(min(RV['single_p']))} to {n2(max(RV['single_p']))}). And never say that scale, or nearness, caused the gain.""",
          f"""Final report sections {S_MORE} and {S_UNSEEN} &middot; <code>results/more_basins.json</code> (summary_test, unseen, proximity, gain_rule, outcome_caveat),
          <code>results/lstm_lobo.json</code> &middot; <code>code/17_more_basins.py</code>, <code>code/05_lobo.py</code> &middot; <code>figures/more_basins.png</code>""")

    if "holds" in H_OK:
        P1 = NW["PAGE1"]
        Q("H", "holds", "Does your main finding still hold after the four new experiments?",
          f"""Yes, in both directions. Taking information away lowered the skill: without the river gauge the LSTM loses {n2(ST['river_gauge_without_forecast'][lk(1)]['nse_median_gain'])} NSE
          {LEAD_WORD[1]}, and a river never seen loses {n2(RV['loss'][0])} NSE with {RV['n_train']['lobo_w_static']} training rivers. Giving it raised the skill: the rain to midnight takes the onsets warned
          {LEAD_WORD[1]} from {pct(AO['base'][1])} to {pct(AO['with_'][1])} ({HIND_W + '; ' if HIND_W else ''}{pct(E1['with_'])} with ERA5; {S48}), and the forecast takes them {leads(LEADS[1:])} from
          {slash([FC['rain'][k] for k in LEADS[1:]], pct)} with the rain to midnight alone to {slash([FC['fc'][k] for k in LEADS[1:]], pct)} with the forecast beside it. And the kind of model still changed little. {P1['yard']} Ridge regression warns of
          {pct(AO['warned']['ridge'])} of the onsets with the rain to midnight, with {pct(AO['far']['ridge'])} of its warnings false, against {pct(AO['warned']['lstm'])} and
          {pct(AO['far']['lstm'])} for the LSTM: a different operating point, not a better model. The verdict for onsets stays {NOT_YET(AO['verdict_onset'])} with every input tried.""",
          f"""&ldquo;The LSTM is now better than the simple models.&rdquo; In median NSE with the rain to midnight it is ahead ({n2(AO['nse_with'][1])} against {n2(AO['nse_simple']['ridge'])} and
          {n2(AO['nse_simple']['forest'])}); in onsets ridge regression warns of more, and with the forecast the NSE readings against the simple models show no clear difference. The finding is
          confirmed by what the simple models gain, not by what the LSTM gains: {INFO_NOT_NET}.""",
          f"""Final report sections {S_RAIN}, {S_FORECAST} and {S_LIMITS} (conclusions) &middot; <code>results/aorc_rain_to_midnight.json</code> (yardstick), <code>results/forecast_input.json</code> (yardstick),
          <code>results/lstm_all.json</code> (decomposition) &middot; <code>figures/{FIGURE_KEY}</code> (the key figure{': report page 1' if key_in_report else ''})""")

    if "service" in H_OK:
        Q("H", "service", "What would a real service need that this project does not have?",
          f"""Four things, in the order of what we measured. First, rain up to midnight from a real-time product and not from AORC: Stage IV or a similar radar product, about an hour behind; with Stage
          IV in place of AORC the same networks warn of {pct(S4['stage4'] / S4['n'])} of the onsets {LEAD_WORD[1]}, {pct(S4['no_last'] / S4['n'])} with the last hour out, and both figures are
          {REALTIME_WORDING}. Second, the forecast in time: the run
          used is the {NW['CLOCK']['run_utc']} UTC run of the UTC date t+1, {FC['hours_old'][0]} to {FC['hours_old'][1]} hours old at the issue time; in Eastern summer its last files appear about
          {n2(abs(FC['late']), 0)} minutes after the issue time, a margin read from {word(FC['margin_days'])} days of posting times of the operational control member, not from the archive. Third, the
          ensemble: only the control member of {word(FC['members'])} was used, and no temperature forecast and no encoder&ndash;decoder form that reads a forecast day by day {cite('nearing2024')}. Fourth, a
          network trained on the real-time product, and a test of its false warnings on days without an onset: neither was done. The project itself would also need five
          trainings in place of {word(n_seeds)} where the NSE decides: {nse_forecast_words(FC)}. And even then the verdict for onsets is {NOT_YET(AO['verdict_onset'])} with
          every input tried ({pct(AO['false'][1])} of the warnings false with the rain to midnight; {pct(FC['false_fc'][2])} and {pct(FC['false_fc'][3])} {leads(LEADS[1:])} with the forecast).""",
          f"""&ldquo;The {pct(AO['with_'][1])} and the {pct(FC['fc'][2])} / {pct(FC['fc'][3])} are what a service would deliver.&rdquo; They are what these inputs are worth in a test on past years: the
          rain is {AORC_WORDING} ({HIND_W}), the forecast is one member of a reforecast, and with Stage IV in place of AORC the figure is {pct(S4['no_last'] / S4['n'])} to {pct(S4['stage4'] / S4['n'])}, {REALTIME_WORDING}.""",
          f"""Final report sections {S_RAIN}, {S_FORECAST} and {S_LIMITS} (next steps) &middot; <code>results/stage4_realtime.json</code>, <code>results/gefs_basins.json</code> (realtime_margin),
          <code>results/data_access.json</code> (gefs_reforecast_record) &middot; <code>code/check_stage4_realtime.py</code>, <code>code/gefs_to_basins.py</code>""")

    if "newcost" in H_OK:
        CO = NW["COST"]
        n_tr = sum(CO["trainings"].values())
        Q("H", "newcost", "What did the new experiments cost?",
          f"""No money and no account: {'nothing was bought and no account was created' if CO['nothing_bought'] else 'see results/data_access.json'}, the data are open. The downloads: AORC
          {n2(CO['aorc_gb'], 1)} GB in {n2(CO['aorc_h'], 1)} hours, the GEFSv12 reforecast {n2(CO['gefs_gb'], 0)} GB read in {CO['gefs_min']} minutes (of which {CO['kept_mb']} MB were kept), and Stage IV
          {n2(CO['s4_gb'], 0)} GB in {n2(CO['s4_h'], 1)} hours; every file was read, reduced to basin means and dropped. The computing: {n_tr} trainings of the network on {laptop}, one at a
          time, {n2(CO['hours'], 1)} hours of training in all ({join(f'{CO["trainings"][k]} for {NEW_LABELS[k]}' for k in CO['trainings'])}), and all {CO['saved']} trained networks were saved (results/models); each results file records a reload check, in which a network rebuilt from its file alone gives
          back its stored forecasts. {'No cluster was needed; the largest paid option the cost research found was a few US dollars.' if CO['cluster_note'] else ''}""",
          f"""&ldquo;It needed a cluster&rdquo;, or &ldquo;it is free, so anyone can rerun it in an hour.&rdquo; Neither. The three download scripts are not part of <code>run_all.sh</code>; the one for AORC needs the
          <code>zstd</code> tool and the one for the forecast a separate environment with a GRIB reader; and the four training steps add about {n2(CO['hours'], 1)} hours to a full run.""",
          f"""Final report sections {S_RAIN} to {S_FORECAST} and {S_WHO} (how to reproduce) &middot; <code>results/aorc_data.json</code> (download), <code>results/data_access.json</code> (summary:
          download_done, costs), <code>results/stage4_realtime.json</code> (data: fetch_runs), the four results files (n_trainings, runtime_s) &middot; <code>code/fetch_aorc.py</code>,
          <code>code/fetch_gefs.py</code>, <code>run_all.sh</code>""")

    if "reviewed" in H_OK:
        RE = NW["REV"]
        KEYS = list(RE)                                                # the seven experiments with a review record, E1 to E4e (qa_since_midterm.review_facts)
        again = [RE[k]["label"] for k in KEYS if RE[k].get("same_agent")]
        same_text = f" ({join(again)} by the same reviewer agent as an earlier review, started again with a new brief)" if again else ""
        total = sum(int(RE[k]["checks"]) for k in KEYS)
        values = sum(int(RE[k].get("last_values") or 0) for k in KEYS)
        mism = sum(int(RE[k].get("mismatches") or 0) for k in KEYS)
        failed_last = [(RE[k]["label"], int(RE[k].get("last_failed") or 0)) for k in KEYS if int(RE[k].get("last_failed") or 0)]
        all_disclosed = all(RE[k].get("disclosed") for k in KEYS)
        rounds_text = join(f'{RE[k]["label"]} {"once" if int(RE[k]["rounds"]) == 1 else word(int(RE[k]["rounds"])) + " times"}' for k in KEYS)
        later_text = (f"Over all rounds the {word(len(KEYS))} reviews list {total} checks. In their latest rounds the reviewers recomputed {values:,} values from the stored forecasts with "
                      f"their own code: " + ("no number differed" if mism == 0 else f"{mism} differed (see the records)") + ". "
                      + (f"The checks that failed in those rounds ({join(f'{lab}: {n}' for lab, n in failed_last)}) asked for wording: the disclosure of the seeds added after the "
                         f"test scores were known, and scripts that read the newest review round"
                         + ("; every results file now carries that disclosure." if all_disclosed else "; not every results file carries the disclosure yet.")
                         if failed_last else "No check failed in the latest rounds."))
        stage4_review = (f" The E1b review recomputed a smaller Stage IV substitution, on {NW['HIND']['n_covered']} of the {count(NW['HIND']['n_onsets'])} onsets." if NW.get("HIND") else "")
        Q("H", "reviewed", "Who checked the new experiments?",
          f"""Claude Code wrote and ran all {word(len(KEYS))} under the group leader&rsquo;s direction: the first four on {NEW_DAYS}, E4a, E4b and E4e after the mid-term talk. {review_sentence()}. Each was checked by an independent AI reviewer agent (Claude) with a brief of its own, allowed to edit only its own record and its own scripts:
          {rounds_text}{same_text}.
          {later_text}{stage4_review} Every record says that the team&rsquo;s own check of its experiment had not been done when it was written (<code>team.json</code> untouched by the review); today {checks_now}.""",
          f"""&ldquo;The new experiments were checked by the team&rdquo;, or &ldquo;independent means checked by people.&rdquo; Independent here means a separate AI agent with a separate brief, the same
          kind of system as the one that wrote the code; it recomputed the numbers with its own code, but it is not one of us. And do not say &lsquo;every script was reviewed independently&rsquo;: what only the final critic agent verified is named in the first sentence. Name only a check that you have run yourself (list at the end).""",
          f"""{W("6.3", "AppE")} &middot; <code>results/review_*.json</code> (every round; the latest is the five-seed round) &middot; <code>code/review/</code> (the reviewers&rsquo; own scripts)""")

    if "spent" in H_OK:
        SP = NW["SPENT"]
        Q("H", "spent", "The test years were already used for the earlier results. Were the new experiments tuned on them?",
          f"""No setting was chosen on the test years, but the designs of the later experiments were fixed after the earlier test scores were known, and the results files say so
          (<code>choices_made_after_looking_at_test_scores</code>: {join(f'{SP[k]} entr{"y" if SP[k] == 1 else "ies"} in {NEW_LABELS[k].split()[0]}' for k in NEW_FILES)}). Every choice made from data, the stopping point, the
          warning level and the bias factors, was made on the validation years; the success rules were written before the runs (<code>NEXT_STEPS_FOR_AGENTS.md</code>, <code>ROADMAP.md</code>). What was decided
          after looking: the matched-window controls, the combined set and the simple models of E1 after its first twelve trainings; the design of E1b after the scores of E1; that of E2 after those of E1
          and E1b; the descriptive analyses of the 1-day result of E2 after its review; and, in E3, a scoring of the test years of four sets made once while a run waited for the
          processor, to check the scoring code and the figure. None of them changed an input set, a setting, a seed, a stopping rule, a warning level or a success rule after the
          first training of its experiment, and no training was dropped. The limit that remains is {word(n_seeds)} trainings per set and one test period of {n_years(split['test'])} years, scored many times:
          the ranges of {beforex('sure') or 'the question on how sure we are'} are the guard.""",
          f"""&ldquo;The test years were scored once.&rdquo; They were scored at every re-scoring pass, and the later designs follow earlier results ({X('fair')} says the same of the first version). What is true is that
          nothing was chosen from them, and that each design decision taken after looking is listed in the results file of its experiment.""",
          f"""Final report sections {S_DATA}, {S_RAIN} to {S_FORECAST} &middot; <code>NEXT_STEPS_FOR_AGENTS.md</code> (section 3), the four results files (choices_made_after_looking_at_test_scores, passes) &middot;
          <code>results/uncertainty.json</code>""")

    # ---- group I: the work after the mid-term talk (code/qa_since_midterm.py) ------------------------------------------------
    QI.add_questions(Q, X, FI, FR, dict(HIND_W=HIND_W, about=about, n=n, I_OK=I_OK, proposal_day=proposal_day))

    # ---- the story, for the top of the sheet -----------------------------------------------------------------------------------
    slides = sorted(ROOT.glob("*.pptx"))
    m1 = "met, the number to beat is known" if b_best else "open"
    w_most = all(w > n / 2 for w in w_wins)
    w_m2 = (f"; with past weather only the {word(n_seeds)}-seed average is ahead in {'most' if w_most else 'some'} basins, a single training is not clearly ahead"
            if w_avg_only else f"; with past weather only {say(w_cls)}")
    m2_long = ("met" if gauge_better else f"not met with the river gauge ({'no clear difference' if gauge_tie else say(cls['wq'])}){w_m2}; what we learned instead "
               f"is why not")
    m3 = ("done" if "verdict" in FA and all((ROOT / "figures" / f).exists() for f in ("flood_skill.png", "basin_maps.png")) else "open")
    m4 = (("done, by Claude Code" if slides else "the report is drafted, by Claude Code; no slides file is in the project folder yet")
          if (ROOT / "Flood_Report.pdf").exists() else "open")
    onset_all_not_yet = set(verdicts("lstm_wq", "onset")) == {"NOT YET"}
    warn_story = (f"{g1['pooled']} for all high-flow days {LEAD_WORD[1]} with a river gauge"
                  + (f", the same verdict as {TT}" if pers_v == g1["pooled"] else "")
                  + (f"; NOT YET for onsets, the warnings that matter ({pct(on_wq[0]['pod'])} warned {LEAD_WORD[1]})" if onset_all_not_yet else
                     f"; for onsets {slash(verdicts('lstm_wq', 'onset'), str)}"))
    w_story = (f" With past weather only, the {word(n_seeds)}-seed average is ahead in {join(w_wins)} of {n} basins, but a single training is not clearly ahead."
               if w_avg_only else f" With past weather only, {say(w_cls)}.")
    # why: the evidence that fits the claim (models given the same inputs tie, each gains with more information), then the rain-timing test with the
    # onsets first and NSE as mixed, then what is left
    why_head = (f"Among the models tried, the limit is the information in the inputs, not the network: {not_network_short}." if inputs_first else
                f"The weather of the days being forecast, {UPPER}, would lift the onsets warned {LEAD_WORD[1]} from {pct(on_wq[0]['pod'])} to {pct(on_ub[0]['pod'])}.")
    why = ""                                                       # what follows the first sentence: the rain-timing test, onsets first
    if stamp_gap:
        why += (f" Part of the gap to that upper bound is rain that had fallen but was not yet in the model&rsquo;s inputs: rain stamped up to the evening of the issue day lifts the "
                f"onsets warned {LEAD_WORD[1]} with the river gauge {on_move(r1)} (control {pct(r1['onset_warned_control'])}, upper bound "
                f"{pct(r1['onset_warned_perfect_forecast'])}), {about(on_gap)} of the gap ({X('timing')}). {rt_nse}. The rest needs later rain, most of it after the "
                f"forecast is issued; {rt_later.split(' (')[0]}.")
    elif RT is not None:
        why += f" The rain-timing test did not show a clear time-stamp gap ({X('timing')})."
    else:
        why += f" How much of the gap is rain that had fallen but was not yet in the model&rsquo;s inputs: result pending ({X('timing')})."
    if P1:                                                         # the experiments of 7 and 8 October: what the rain of the rest of the day, and a forecast, add
        why += (f" The next experiments gave the model that later rain and a forecast{Hp('aorc', 'moved')}: {lower_first(P1['rungs'])} The simple models given the same new inputs "
                f"gain as many onsets as the LSTM: {INFO_NOT_NET}{Hp('holds')}.")
    if stamp_gap and on_gap < 0.5:
        take = (f"A weather forecast as an input at every lead time: it is the larger part of the gap even {LEAD_WORD[1]}, and it was not tested. Rain observations up "
                f"to the hour, which exist today, add a part {LEAD_WORD[1]} ({about(on_gap)} of the gap in onsets warned here); with that rain the verdict for onsets "
                f"{still_word} {rain_verdict}. More basins for unseen rivers.")
    else:
        take = ("A weather forecast as an input (not tested); rain observations up to the hour of the forecast, which exist today (their worth is not shown by our "
                "test); more basins for unseen rivers.")
    take_head = "What it would take."
    if P1 and P1.get("took"):                                      # the experiments took two new inputs: page 1 of the report says what it took and what is missing
        take_head, take = "What it took, and what is missing.", P1["took"]
    if P1 and P1.get("verdict"):                                   # the verdict for onsets with every new input (the clause of finding 3 on page 1)
        warn_story += P1["verdict"]
    # the finding comes first, then its evidence; the forecasts, the warnings and the milestone that was not met follow, said plainly
    key_ref = f"key figure: {key_places}; " if key_fig else ""
    rest = [
        f"<b>The forecasts.</b> With the river gauge, {practice}, between the LSTM and the best simple model {say(cls['wq'])} (median NSE {n2(wq_lstm[0])} against "
        f"{n2(wq_base[0])} {LEAD_WORD[1]}; LSTM higher in {join(wq_wins)} of {n} basins): milestone 2 of our plan is {m2} there.{w_story}",
        f"<b>The warnings.</b> {cap(warn_story)}.",
        f"<b>{take_head}</b> {take}",
        f"<b>The plan&rsquo;s milestones.</b> 1 (baselines scored): {m1}. 2 (the LSTM beats them in most basins, or we learn why not): {m2_long}. 3 (warning table and "
        f"maps): {m3}. 4 (report and slides drafted): {m4}.",
        f"<b>Who did it.</b> Claude Code wrote and ran everything under the team lead&rsquo;s direction ({when_short}); AI reviewer agents checked it; {checks_now}."]
    if inputs_first:
        story = [f"<b>The finding.</b> {FINDING} {SUPPORT} ({key_ref}{X('change')}).", f"<b>The evidence.</b> {cap(not_network_short)}.{why}"] + rest
        finding = (f"<b>What we found: {FINDING[0].lower() + FINDING[1:]}</b> <span>{SUPPORT} ({key_ref}the summary with the numbers is on page 2).</span>")
    else:                                                          # the old order: nothing is called 'the finding' unless the results carry it
        story = rest[:2] + [f"<b>Why.</b> {why_head}{why}"] + rest[2:]
        finding = ""

    # ---- page 1: the ten questions most likely to be asked. Two sentences each, built from the values above; the numbers of the full answers follow -----
    have = {i["key"] for i in items}
    to = lambda *keys: X(*[k for k in keys if k in have])
    top = []
    lost = "Your LSTM did not beat the simple models. So what did you find?" if not gauge_better else "Does your LSTM beat the simple models? What did you find?"
    if gauge_tie and inputs_first:
        top.append((lost, f"{FINDING} With the river gauge the LSTM and the best simple model show no clear difference (median NSE {n2(wq_lstm[0])} against "
                          f"{n2(wq_base[0])} {LEAD_WORD[1]}; LSTM higher in {wq_wins[0]} of {n} basins), so milestone 2 of our plan is not met there.",
                    to("beat", "change", "failed")))
    else:
        top.append((lost, f"With the river gauge, between the LSTM and the best simple model {say(cls['wq'])} (median NSE {n2(wq_lstm[0])} against {n2(wq_base[0])} "
                          f"{LEAD_WORD[1]}): milestone 2 of our plan is {m2} there." + (f" {FINDING}" if inputs_first else ""), to("beat", "change", "failed")))
    top.append(("What did the AI assistant do, and what did you do?",
                f"Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction; AI reviewer agents checked it; {checks_now}. If you are asked what "
                f"you did yourself, " + ("name only a check you have run yourself; if there is none yet, say that it is still open." if none_done else
                                         "name only the check you have run yourself and entered in team.json."), to("who")))
    ft_e, ft_m = FT["expected"]["past_weather_only"], FT["measured"]["past_weather_only"]
    top.append(("How do you know the test is fair?",
                f"Each period has one job: the training years fit, the validation years choose, the test years are only scored. A check on made-up rivers with a "
                f"known answer measured NSE {n2(ft_m[0], 3)} {LEAD_WORD[1]} where a model without a leak can reach at most {n2(ft_e[0])}; a leak would push it higher.",
                to("fair")))
    top.append(("Describe the network. Did it overfit?",
                (f"{'One LSTM layer' if CODE['one_layer'] else 'An LSTM'} of {cfg['hidden']} memory cells" + (" and a linear output layer" if CODE["linear_head"] else "")
                 + f" for the {word(len(LEADS))} lead times; " + ("Adam, " if CODE["adam"] else "")
                 + ("a squared error weighted per basin, " if CODE["loss_per_basin"] else "a squared error, ")
                 + f"a dropout of {cfg['dropout']:g} and early stopping on the validation years. "
                 + (f"Trained on one river alone it memorised its training years (trained on past its stopping point, its NSE went from "
                    f"{n2(mean(diag[0]['train_nse']))} to {n2(mean(diag[-1]['train_nse']))} in the training years and from {n2(mean(diag[0]['val_nse']))} to "
                    f"{n2(mean(diag[-1]['val_nse']))} in the validation years), so one model is trained on all {n} rivers."
                    if memorised else f"One model is trained on all {n} rivers.")),
                to("network", "settings", "allrivers", "overfit", "cell")))
    papers = (f"Those studies train one model on hundreds of basins and estimate the flow of a day from the weather up to and including that day; we forecast tomorrow "
              f"from the weather up to today, on {n} basins. ")
    top.append(("Papers say LSTMs are the best models for river flow. Why not here?",
                papers + (f"Given the weather of the day itself, an upper bound, our LSTM scores {n2(pub_range[2])} to {n2(pub_range[3])}, of the same size as the published "
                          f"{n2(pub_range[0])} to {n2(pub_range[1])}." if PS is not None and same_size else
                          "And the simple models are handed the river&rsquo;s memory ready-made, as rain sums of the last days."), to("published", "why_lstm")))
    top.append((f"What does an NSE of {n2(wq_lstm[0])} mean? Is it good?",
                f"1 is a perfect forecast, 0 is no better than always forecasting the average flow; it is not a percentage. {n2(wq_lstm[0])} is the median of {n} basins "
                f"that run from {n2(min(wq_basin))} to {n2(max(wq_basin))}, and {TT} alone scores {n2(SB['base_persistence', 1]['nse_median'])} with nothing fitted.",
                to("nse", "baselines")))
    real_top = (f"A real forecast has errors: the public one we tested lifts the onsets warned {leads(LEADS[1:])} from {slash([FC['rain'][k] for k in LEADS[1:]], pct)} with the rain to "
                f"midnight alone to {slash([FC['fc'][k] for k in LEADS[1:]], pct)} with the forecast beside it, where perfect rain would give "
                f"{slash([FC['perfect'][k] for k in LEADS[1:]], pct)}." if FC else
                "A real forecast has errors and would land in between; none was tested.")
    top.append(("What is the perfect weather forecast? Is that not cheating?",
                f"It is the weather recorded for the days being forecast, given to the model as if it were a forecast: {UPPER}, shown to measure what that information "
                f"is worth (onsets warned {LEAD_WORD[1]}: {pct(on_wq[0]['pod'])} without it, {pct(on_ub[0]['pod'])} with it). {real_top}", to("upper", "fcskill")))
    why_missed = ("Mainly because the rain behind the rise was not yet in the model&rsquo;s inputs" if stamped_late else
                  "The onsets were sorted by where their rain is stamped, and no single cause stands out")
    if stamp_gap and P1:                                           # since 8 October 2026 the answer holds the whole ladder: rain to the evening, to midnight, a forecast
        top.append(("Why are the onsets missed, and what did the rain and forecast experiments show?",
                    f"{why_missed}: rain stamped up to the evening of the issue day lifts the onsets warned {on_move(r1)} (control {pct(r1['onset_warned_control'])}, upper bound "
                    f"{pct(r1['onset_warned_perfect_forecast'])}), about a quarter of the gap; the rain of the whole day up to midnight from AORC, {AORC_WORDING}, to "
                    f"{pct(AO['with_'][1])}{hw}; a public forecast beside it, {leads(LEADS[1:])}, from "
                    f"{slash([AO['base'][k] for k in LEADS[1:]], pct)} with the rain table alone to {slash([FC['fc'][k] for k in LEADS[1:]], pct)}. The verdict for onsets stays "
                    f"{NOT_YET(AO['verdict_onset'])}; the day stamp of the rain table is an inference from our own data; the {pct(S4['stage4'] / S4['n'])} with Stage IV is "
                    f"{REALTIME_WORDING}; still open are five trainings in place of {word(n_seeds)} where the NSE decides, the ensemble and a network trained on Stage IV.",
                    to("onsets", "timing", "aorc", "moved", "rainday")))
    elif stamp_gap:
        top.append(("Why are the onsets missed, and what is the rain-timing result?",
                    f"{why_missed}: rain stamped up to the evening of the issue day lifts the onsets warned {on_move(r1)} (control {pct(r1['onset_warned_control'])}, "
                    f"upper bound {pct(r1['onset_warned_perfect_forecast'])}), which closes {about(on_gap)} of the gap in onsets warned {LEAD_WORD[1]}. "
                    + (f"In NSE the result is mixed (higher in only {r1['basins_better']} of {n} basins), and the day stamp is an inference from our own data."
                       if nse_word == "mixed" else "The day stamp of the rain table is an inference from our own data."), to("onsets", "timing", "rainday")))
    else:
        top.append(("Why are the onsets missed?", f"{why_missed}. How much of that rain had already fallen when the forecast was issued, our test did not show clearly.",
                    to("onsets", "timing", "rainday")))
    if inputs_first and tie:
        top.append(("Would a bigger network or a Transformer fix it?",
                    f"Not among the models tried: given the same inputs, the Transformer showed no clear difference from the LSTM in the test years (NSE {n2(tsk['transformer'], 3)} against "
                    f"{n2(tsk['lstm'], 3)}) and took {times} as long to train; a larger network was not tried. The weather of the days being forecast, an upper bound, adds {signed(info_gain)} "
                    f"NSE {LEAD_WORD[1]}; the largest like-for-like gap between the LSTM and the best simple model is {n2(model_gap)}.",
                    to("transformer", "tfspec", "change")))
    else:
        top.append(("Would a bigger network or a Transformer fix it?",
                    f"{f_head} Training the Transformer took {times} as long, and a larger network was not tried.", to("transformer", "tfspec", "change")))
    sat_text = (f"with the river gauge it raises the median NSE from {n2(sat_gain[0])} to {n2(sat_gain[1])} {LEAD_WORD[sat_gain[2]]}, probably because its day ends "
                f"later, not because the rain is better" if sat_gain else "with the river gauge it makes no clear difference")
    vapour = "made no clear change in NSE" if pw_nse == 0 else f"changed the median NSE clearly in {word(pw_nse)} of {word(len(pw))} comparisons"
    top.append(("What did remote sensing and GNSS add?",
                f"Basin maps with elevation and land cover from {cell} m rasters, a check of the CAMELS basin properties, and satellite rain as an input: {sat_text}. No "
                f"GNSS data were used: a reanalysis stand-in for water vapour {vapour}; real GNSS-derived vapour was not tested.", to("rasters", "imerg", "gnss")))
    also = "Also likely: " + "; ".join(f"{text} ({refs})" for text, refs in (
        ("what is new since the mid-term", to("midterm", "riverstate")), ("why the full model is not GO", to("sixtyeight", "ownsplit")),
        ("seeds added after the test scores", to("moreseeds")), ("the teammates&rsquo; model", to("cloud")),
        (f"how sure are you with {n} basins", to("clear", "pvalue", "sure")), ("a river never seen", to("unseen")), ("what is inside an LSTM cell", to("cell")),
        ("did the project fail", to("failed")), ("what would you do next", to("next")),
        ("AORC made after the fact, and hindsight", to("aorc", "hindsight")), ("a forecast that lowers the 1-day onsets", to("moved")),
        ("more rivers", to("rivers")), ("cost and review of the new work", to("newcost", "reviewed"))) if refs) + "."

    # ---- check before the hand-in ----------------------------------------------------------------------------------------------
    H = T["hparam_log"]
    group_a = [i["number"] for i in items if i["group"] == "A"]
    decisions = open_decisions()
    both_run = "satellite_inputs" in R and "model_comparison" in R
    purposes = H.purpose.str.split(" ").str[0].value_counts()
    worst_elev = max(far_elev, key=lambda k: abs(far_elev[k])) if far_elev else None
    floods = [d["day"] for d in VC["days"] if d["day"].startswith("2004")]
    hurricane = "hurricane" in " ".join(LS["notes"])
    # the two floods of September 2004 on the one-river basin: the storm-track check of check_2004_floods.py (floods_2004.json) is the source
    try:
        near = [c for d in floods for c in F04["cyclones_within_500_km"] if c["date"] == d]              # the storm that was closest on each of those days
        if len(near) != len(floods) or not near:
            raise KeyError("no storm of the track record matches the flood days")
        status = {STORM.get(c["status_then"], "tropical cyclone") for c in near}
        remnant = any("remnant of the hurricane" in x for x in F04["notes"])
        what04 = f"{'the remnants of hurricanes' if remnant else 'the tropical cyclones'} {join(c['name'] for c in near)}"
        floods_item = (f"Know the source for the floods of {join(date(d) for d in floods)} on the {fb_name} River, which the report links to {what04}: "
                       f"<code>results/floods_2004.json</code>, from the storm-track record HURDAT2 {cite('landsea2013')}. The storms passed "
                       f"{join(n2(c['closest_km'], 0) for c in near)} km from the gauge on those days"
                       + (f", {'both' if len(near) == 2 else 'all'} as {next(iter(status))}s" if len(status) == 1 else "")
                       + ("; a storm nearby is strong circumstantial evidence, not proof, that its rain caused the flood."
                          if any("not proof" in x for x in F04["notes"]) else "."))
    except (KeyError, TypeError, IndexError) as err:
        if F04 is not None:
            print(f"results/floods_2004.json is not in the expected form ({err!r}): the old checklist item is kept", file=sys.stderr)
        floods_item = (f"The report calls the floods of {join(date(d) for d in floods)} on the {fb_name} River hurricane floods: find a source for that word, or ask for "
                       f"it to be dropped." if hurricane and floods else
                       "Look at the training curves stored in <code>results/lstm_all.json</code>: where each training stopped and why.")
    if sources:
        rain_day_item = (f"Read the {word(sources['n'])} passages behind {X('rainday')} yourself: the paper of Daymet{' version 4' if sources['v4'] else ''} {cite('thornton2021')}, on rain reports that are not taken "
                         f"from midnight to midnight, and the lag analysis of the rain products of CAMELS {cite('kratzert2021')}. "
                         f"<code>results/sources_checked.json</code> records what Claude Code found in them on {date(sources['when'])}"
                         + ("; no team member has read them yet" if sources["unread"] else "")
                         + (", and the documentation of the Daymet version inside CAMELS was not checked." if sources["camels_unchecked"] else "."))
    elif not_checked:
        rain_day_item = (f"Check the reading of the rain table&rsquo;s day ({X('rainday')}) against the Daymet documentation, starting from the paper the report names "
                         f"for it {cite('thornton2021')}: <code>results/satellite_data.json</code> says this was not done.")
    else:
        rain_day_item = "Read the notes of <code>results/satellite_data.json</code> on the day stamps."
    windows_item = ([f"Open <code>figures/day_windows.png</code> and <code>results/day_windows.json</code> (day_ends, notes): the check against hourly rain that puts the "
                     f"end of the rain table&rsquo;s day at about {dw['daymet']} and of the added rain of the rain-timing test at about {dw['nldas']} local standard "
                     f"time. It was added last: read its notes and its limits before quoting an hour."] if dw else [])
    order_item = ([f"Say the rain-timing result in this order ({X('timing', 'typical')}): first the onsets warned, {pct(r1['onset_warned_base'])} to "
                   f"{pct(r1['onset_warned_with_nldas'])}, {about(on_gap)} of the gap; then NSE as {nse_word}"
                   + (f", higher in only {r1['basins_better']} of {n} basins" if nse_word == "mixed" else "")
                   + f". Never &lsquo;{share(s1)} of the gap closed&rsquo; alone."] if stamp_gap else [])
    members = {m["key"]: m for m in team.load()["members"]}

    def member(key, plain, tasks):
        """The heading of one member's list: today's words while team.json gives no name, 'Name (field): tasks' once it does."""
        return f"{plain} ({tasks})" if not members[key]["name"] else f"{html_lib.escape(team.label(key), quote=False)}: {tasks}"

    def own_check(key):
        """One line under a member's heading once that member has entered an own check in team.json; nothing while it is still to be done."""
        status = team.check_status(key)
        return [] if status == "still to be done" else [f"<b>Own check:</b> {html_lib.escape(status, quote=False)}."]

    # the published scores behind the answer on published results: to be confirmed at the source by a team member
    if PS is not None:
        published_item = (f"Open the {word(len(pub_papers))} papers and confirm the numbers of <code>results/published_scores.json</code> (each entry names its table or "
                          f"page): {'; '.join(cite(k)[1:-1] for k in pub_papers)}. They stand in {X('published')} and in final report sections {S_SKILL} and {S_UNSEEN}; an AI "
                          f"agent read them at the source on {pub_date}" + ("; no team member has yet." if pub_unread else "."))
    else:
        published_item = (f"Before anyone quotes a published score, open the paper {cite('kratzert2019')} and confirm what {X('published')} says about its set-up: the "
                          f"flow of a day from the weather up to that day, one model for hundreds of basins. This sheet quotes no published score.")
    # the experiments of 7 and 8 October 2026: what each member can open, run or verify. These are things to do; nothing here says that anyone has done them
    new_all, new_a, new_r2, new_g1, new_g2 = [], [], [], [], []
    if "aorc" in H_OK and FC and E1 and RV:
        new_all = [f"Say the ladder of the new experiments in this order{Hp('aorc', 'moved', 'holds')}: onsets warned {LEAD_WORD[1]}, {pct(AO['base'][1])} with the rain table, "
                   f"{pct(E1['nldas'])} with rain up to the evening, {pct(AO['with_'][1])} with the rain to midnight from AORC, {AORC_WORDING} ({HIND_W + '; ' if HIND_W else ''}{pct(S4['stage4'] / S4['n'])} with "
                   f"Stage IV and {pct(S4['no_last'] / S4['n'])} with its last hour out: {REALTIME_WORDING}), {pct(AO['upper'][1])} with the perfect weather forecast, an upper bound; {leads(LEADS[1:])}, "
                   f"{slash([AO['base'][k] for k in LEADS[1:]], pct)} with the rain table, {slash([AO['with_'][k] for k in LEADS[1:]], pct)} with the rain to midnight and "
                   f"{slash([FC['fc'][k] for k in LEADS[1:]], pct)} with a public forecast beside the rain to midnight. Never quote the {pct(AO['with_'][1])} without &lsquo;{AORC_WORDING}&rsquo;"
                   f"{' and &lsquo;' + HIND_W + '&rsquo;' if HIND_W else ''}, and never call the forecast numbers what a service would deliver."]
        new_a = [f"Open the four review records, {join(f'<code>results/review_{k}.json</code>' for k in NEW_FILES)}: who reviewed what, their must-fix and should-fix items and how each was "
                 f"dealt with. Every record says that the team&rsquo;s own check of its experiment had not been done when it was written; read page 1 and section {S_LIMITS} of the final report once "
                 f"more and check that they say no more than the records do. Open <code>results/data_access.json</code>: what was downloaded, from where, and that nothing was bought{Hp('newcost', 'reviewed')}."]
        new_r2 = [f"Read how AORC is made, as the reviewer quotes the methods document (<code>results/review_aorc_rain_to_midnight.json</code>, check d1): daily totals on a gauge day that "
                  f"ends at {NW['CLOCK']['gauge_day_utc']} UTC, spread over the hours with radar, and before {NW['CLOCK']['early']} disaggregated daily totals, so that AORC is {AORC_WORDING}. Open <code>figures/stage4_realtime.png</code> "
                  f"and <code>results/stage4_realtime.json</code> and say in your own words what the Stage IV test shows and what it does not{Hp('aorc', 'hindsight')}. Pick one onset in "
                  f"its per_onset list and compare its AORC and Stage IV rain of the issue day with the hourly Stage IV file of the Iowa Mesonet archive (the address is in "
                  f"<code>results/aorc_rain_to_midnight.json</code>, aorc_data: time_stamps)."]
        new_g1 = [f"Check the first rung by hand: the one-line check of <code>README.md</code> with <code>lstm_wqa</code> in place of <code>lstm_wq</code> must print the NSE and the share of onsets "
                  f"warned of the AORC model, {n2(AO['nse_with'][1])} and {n2(AO['with_'][1])}; recompute its {AO['hits']} hits of the {AO['n']} onsets from <code>results/preds/</code> in "
                  f"your own way and compare them with <code>results/aorc_rain_to_midnight.json</code>. The independent reviews did the same with their own code; a check by you is a "
                  f"separate one."]
    if "newcost" in H_OK and "shuffled" in H_OK:
        new_g2 = [f"Open <code>results/models/</code>: one file per training ({NW['COST']['saved']} in all, {join(f'{v} for {NEW_LABELS[k].split()[0]}' for k, v in NW['COST']['trainings'].items())}); "
                  f"the field <code>reload_check</code> of each of the four results files says that a network rebuilt from its file alone reproduces its stored forecasts. In each of them read "
                  f"<code>choices_made_after_looking_at_test_scores</code>, and be ready for {Hx('spent', 'shuffled')}: what was decided after the test scores were known, and why the "
                  f"shuffled control of E2a measures harm."]
    checklist = [
        ("Everyone", [
            f"Read final report section {S_WHO} (who did what) and {X('who')} above, and say it as it is. Explain {group_a[0]} to {group_a[-1]} aloud once without this sheet"
            + (f", then answer the {word(len(top))} questions of page 1." if top else "."),
            (f"The {word(len(decisions))} decisions our plan left to the group when it was written (confirm the first two against the professor&rsquo;s "
             f"instructions): {'; '.join(d[0].lower() + d[1:] for d in decisions)}"
             f"{' (both were run)' if both_run else ''}." if decisions else "Read the open decisions at the end of <code>PLAN.md</code> and settle them.")]
            + order_item + new_all
            + [(f"Rebuild this sheet when the pending results exist ({'; '.join(re.sub(r'<[^>]+>', '', p).split(' (')[0] for p in pending)}): "
                f"<code>python code/build_qa.py</code>." if pending else
                "After any rerun, rebuild the report and this sheet: <code>python code/build_report.py</code>, <code>python code/build_qa.py</code>.")]),
        (member("A", "The team lead", "project folder, data audit, report build"), own_check("A") + [
            f"Run <code>python code/01_data_audit.py</code> and follow the rule in <code>results/data_audit.json</code> and <code>results/basin_audit.csv</code>: "
            f"{count(DA['n_candidates'])} candidates, {count(DA['n_complete'])} complete, {n} selected.",
            "Run <code>python code/build_report.py</code> and read the PDF once from the first page to the last; check that no sentence says or implies that a team "
            "member did or checked something.",
            f"Read <code>results/engine_review.json</code>: who reviewed the code, the {cnt['major']} major findings and how each was dealt with"
            + (f" ({word(n_faults)} fixed in the code; the training times eased, not timed again)." if ER["all_fixed"] and eased else ".")] + new_a),
        (member("R2", "RS/GIS member 2", "basin maps, satellite inputs"), own_check("R2") + [
            f"Run <code>python code/07_basin_maps.py</code> (about {n2(G['runtime_s'], 0)} s with the stored rasters) and confirm that <code>results/basin_gis.json</code> "
            f"does not change; open the three basin figures.",
            (f"Check one basin by hand in a GIS: {worst_elev}, where our mean elevation is {n2(far_elev[worst_elev], 0)} m above the CAMELS value and the cause was not found."
             if worst_elev else "Check one basin by hand in a GIS: area, mean elevation and forest share."),
            rain_day_item]
            + windows_item
            + ["Know the sources as the results file names them: a composite elevation tile set, not &lsquo;SRTM&rsquo;; the Annual NLCD layer, not &lsquo;NLCD "
               f"{G['sources']['land_cover']['year']}&rsquo;."] + new_r2),
        (member("G1", "GNSS member 1", "scoring, fair-test split, flood metrics"), own_check("G1") + [
            f"Run <code>python code/test_flood.py</code>: all {len(FT['checks'])} checks must pass. This is the check that <code>results/engine_review.json</code> lists as "
            f"{'not yet done' if not_done else 'the team&rsquo;s own'}.",
            f"For one basin, {LEAD_WORD[1]}, recompute the NSE and the warning counts (hits, misses, false warnings) yourself, in a spreadsheet or a few lines of "
            f"Python, from the forecasts stored in <code>results/preds/</code>; compare them with <code>results/lstm_all_scores.csv</code>. Do not set "
            f"<code>FLOOD_QUICK=1</code> for this: in quick mode the loader takes the smoke-test forecasts in <code>results/quick/preds/</code> first.",
            f"Confirm the three periods in <code>code/flood.py</code>, read the verdict criteria in <code>results/flood_analysis.json</code> (verdict) and check one "
            f"cell of the verdict table of final report section {S_WARN}."] + new_g1),
        (member("G2", "GNSS member 2", "training, hyper-parameter log, model comparison"), own_check("G2") + [
            f"Open <code>results/hparam_log.csv</code>: {len(H)} runs ({join(f'{v} {k}' for k, v in purposes.items())}); check the tuning table of final report section {S_ONE} "
            f"against it.",
            f"In <code>results/model_comparison.json</code>, check the cost numbers of final report section {S_COST} (the item &lsquo;Cost&rsquo; and the right half of its figure), know its two assumptions "
            f"({n2(MC['settings']['assumed_power_watt'], 0)} W; times measured on a busy laptop) and confirm that the look-ahead test moved no earlier forecast.",
            floods_item,
            published_item,
            f"Be ready for {X('gnss')}: what a GNSS receiver measures (the delay of the signal, not the water vapour itself), and that no GNSS data were used."] + new_g2)]
    # the member page of the final state (qa_since_midterm.member_page): the team check sheet, each member's slides and seconds, the open items
    _mp = QI.member_page(FI, FR, REG, team, X, {i["key"] for i in items}, dict(group_a=group_a, n_top=len(top), extra={}))
    if _mp and _mp[2]:
        checklist = _mp[2]
        global CHECK_HEAD, CHECK_LEAD                                  # the member page's title and lead of the final state (Group #__, the talk)
        CHECK_HEAD, CHECK_LEAD = _mp[0], _mp[1]
    # every reference to another question gets its number now that all questions exist
    number = {i["key"]: i["number"] for i in items}
    for i in items:
        for part in ("q", "a", "trap", "where"):
            i[part] = resolve(i[part], number)
    story = [resolve(s, number) for s in story]
    checklist = [(who, [resolve(t, number) for t in todo]) for who, todo in checklist]
    top = [(q, resolve(a, number), resolve(refs, number)) for q, a, refs in top]
    return story, items, checklist, pending, dict(finding=finding, top=top, also=resolve(also, number))


# ---- the two documents -------------------------------------------------------------------------------------------------------
TITLE = "Predicting river floods with AI: the professor&rsquo;s questions"
STORY_HEAD = "If asked &lsquo;what did you find?&rsquo;"
TOP_HEAD, TOP_SUB = "The {n} questions you will most likely get", "two sentences each; the full answer is under the number given"
CHECK_HEAD, CHECK_LEAD = "Check before the hand-in", "What each member should open, run or verify personally, by the roles of our plan. None of it needs a new experiment."
REFS_HEAD = "Sources named in this sheet"


def team_line():
    """'team: Member 1 (RS/GIS, team lead), RS/GIS member 2, ...; coding assistant: Claude Code' - the names as team.json gives them, the roles of the plan until it does."""
    return f"team: {html_lib.escape(team.names_line(), quote=False)}; coding assistant: Claude Code"


def front(R, items, pending, built):
    """The pieces above the questions, as HTML strings: sub line, how to use the sheet, the note on pending results ('' if none), the words used."""
    B = R["baselines"]
    split = B["split"]
    top5 = pct(sum(h["share_train_days_above"] for h in B["high_flow"].values()) / len(B["high_flow"]))
    number = {i["key"]: i["number"] for i in items}
    two_sided = "two-sided" in note_with(R["lstm_all"]["notes"], "Wilcoxon")
    course = team.course_line()                                   # the course and the university, once team.json gives them ('' until then)
    sub = (f"AI course group project{' &middot; ' + html_lib.escape(course, quote=False) if course else ''} &middot; rehearsal sheet (plan, week 8) &middot; "
           f"built {built} &middot; {team_line()}")
    use = (f"{len(items)} questions a demanding professor could ask, in {word(len({i['group'] for i in items}))} groups. Each answer begins with what to say first; the rest is for the "
           f"questions that follow. <b>The trap</b> is the wrong "
           f"or over-confident answer, and why it is wrong. <b>Where</b> names the report section, the results files (folder <code>results/</code>) and the scripts "
           f"(folder <code>code/</code>) behind the numbers. This sheet was written by Claude Code, the team&rsquo;s coding assistant; every number in it is read from "
           f"the results files when the sheet is built (<code>python code/build_qa.py</code>), none is typed by hand. Differences are taken before rounding, so a "
           f"printed difference can be 0.01 away from the difference of the two numbers printed beside it. In the answers &lsquo;we&rsquo; means the project, "
           f"never a check made by a team member ({number['who']}). Check the answers, then say them in your own words.")
    wait = (f"<b>Result pending.</b> Still being computed when this sheet was built: {'; '.join(pending)}. The answers that need "
            f"{'it' if len(pending) == 1 else 'them'} say so." if pending else "")
    # the words that the experiments of 7 and 8 October 2026 added (only while their questions are in the sheet)
    new_words = ""
    if "aorc" in number:
        new_words += (f" <i>Rain to midnight</i>: the rain of the whole issue day, 00:00 to 24:00 by the wall clock of the gauge. <i>AORC</i>: an hourly rain product, {AORC_WORDING} "
                      f"({number['aorc']}); <i>Stage IV</i>: the radar-and-gauge product that is about an hour behind real time; "
                      f"what it was used for here is a check on past years, not a service ({number['aorc']}).")
    if "moved" in number:
        new_words += f" <i>Operating point</i>: the pair of shares (warned, false) that one warning level gives ({number['moved']})."
    words = (f"<b>Words used.</b> <i>Basin</i>: the land area whose rain drains to one river gauge. <i>NSE</i>: 1 is a perfect forecast, 0 is no better than always "
             f"forecasting the average flow. <i>High-flow day</i>: a day with flow "
             f"above the level that was exceeded on {top5} of days in the training years; each basin has its own level. A model <i>warns</i> when its forecast is above "
             f"that level. <i>Share warned</i>: the share of high-flow days for which a warning was given. <i>False warnings</i>: the share of warnings not followed by a "
             f"high-flow day. <i>Onset</i>: a high-flow day forecast while the river was still below the high-flow level. <i>Seed</i>: one training of a model from its "
             f"own random start; <i>seed spread</i>: how far the scores of several seeds lie apart. <i>Perfect weather forecast</i>: the actual weather of the days being "
             f"forecast, {UPPER}. <i>Lead time</i>: 1, 2 or 3 days ahead. <i>Issue day</i>: the day at whose end a forecast is made; <i>target day</i>: the day being "
             f"forecast.{new_words} <i>p</i>: how likely differences this lopsided are if two models are equally good (Wilcoxon signed-rank test over the basins"
             f"{', two-sided' if two_sided else ''}; it describes, it does not prove; {number['pvalue']}). A <i>clear</i> difference: one that passes the rule of "
             f"{number['clear']}. Training years "
             f"({years(split['train'])}), validation years ({years(split['val'])}), test years ({years(split['test'])}).")
    return sub, use, wait, words


def references_html():
    """The sources cited in this sheet, in alphabetical order; a year that the text gives with a letter (2019a) carries the same letter here."""
    def full(k):
        short, ref = REFERENCES[k]
        lettered = re.search(r"(\d{4})([a-z])$", short)
        return ref.replace(f"({lettered.group(1)})", f"({lettered.group(1)}{lettered.group(2)})", 1) if lettered else ref
    return [full(k) for k in sorted(CITED, key=lambda k: REFERENCES[k][1])]


def contents(items):
    """Where each group of questions is: 'A1&ndash;A4 The four things ... &middot; B1&ndash;B7 Data and set-up ...' (the numbers follow from the questions)."""
    parts = []
    for g, title in GROUPS.items():
        nums = [i["number"] for i in items if i["group"] == g]
        if nums:
            parts.append(f"{nums[0]}{'&ndash;' + nums[-1] if len(nums) > 1 else ''} {title}")
    return (f"<b>The questions.</b> {' &middot; '.join(parts)} &middot; then, on a page of their own, the checks before the hand-in &middot; and the sources "
            f"named in this sheet.")


def to_html(R, story, items, checklist, pending, page1, built):
    """Page 1 is for the last ten minutes: the finding, the ten questions most likely to be asked, each with a two-sentence answer, and where the
    questions are. Page 2 is the summary (what to say if asked what was found, how to use the sheet, the words used); the questions follow from
    page 3; the checks before the hand-in are a page of their own, and the sources named in the sheet stand last."""
    sub, use, wait, words = front(R, items, pending, built)
    arrow = lambda refs: f' <span class="to">&rarr;&nbsp;{refs}</span>' if refs else ""
    top = (f'<div class="top"><p class="h">{TOP_HEAD.format(n=word(len(page1["top"])))} <span>&ndash; {TOP_SUB}</span></p><ol>'
           + "".join(f"<li><b>{q}</b> {a}{arrow(refs)}</li>" for q, a, refs in page1["top"]) + f'</ol><p class="also">{page1["also"]}</p></div>') if page1["top"] else ""
    where = f'<p class="gloss keep">{contents(items)}</p>'
    head = (f'<h1>{TITLE}</h1><p class="sub">{sub}</p>' + (f'<p class="find">{page1["finding"]}</p>' if page1["finding"] else "") + top + (where if top else "")
            + f'<div class="story{" front2" if top else ""}"><b>{STORY_HEAD}</b><ol>{"".join(f"<li>{s}</li>" for s in story)}</ol></div>'
            + f'<p class="lead">{use}</p>' + (f'<div class="note">{wait}</div>' if wait else "") + f'<p class="gloss">{words}</p>' + ("" if top else where))
    body = []
    for n_group, (g, title) in enumerate(g_t for g_t in GROUPS.items() if any(i["group"] == g_t[0] for i in items)):         # a group without a question (its results are missing) has no heading
        body.append(f'<h2{" class=first" if n_group == 0 else ""}>{g}. {title}</h2>')
        for i in (i for i in items if i["group"] == g):
            body.append(f'<div class="qa"><p class="q"><span>{i["number"]}</span>{i["q"]}</p><p class="a">{i["a"]}</p>'
                        f'<p class="trap"><b>The trap:</b> {i["trap"]}</p><p class="where"><b>Where:</b> {i["where"]}</p></div>')
    blocks = "".join(f"<div><h3>{who}</h3><ul>{''.join(f'<li>{t}</li>' for t in todo)}</ul></div>" for who, todo in checklist)
    tail = f'<div class="check"><h2>{CHECK_HEAD}</h2><p>{CHECK_LEAD}</p><div class="cols">{blocks}</div></div>'
    # the sources stand last, as one block: the checklist keeps its own page whatever the last page of questions holds
    refs = f'<div class="src"><h2>{REFS_HEAD}</h2><ul class="refs">{"".join(f"<li>{r}</li>" for r in references_html())}</ul></div>' if CITED else ""
    return page("Professor's questions", head + "\n".join(body) + tail + refs, CSS_QA)


def md(s):
    """The small set of HTML used in this sheet -> Markdown."""
    s = re.sub(r"</?b>", "**", s)
    s = re.sub(r"</?i>", "*", s)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(html_lib.unescape(s).split())


def to_md(R, story, items, checklist, pending, page1, built):
    sub, use, wait, words = front(R, items, pending, built)
    out = [f"# {md(TITLE)}", "", md(sub), ""] + ([md(page1["finding"]), ""] if page1["finding"] else [])
    if page1["top"]:                                               # the same order as in the PDF: the ten questions first, then the summary
        out += [f"## {TOP_HEAD.format(n=word(len(page1['top'])))}", "", f"{TOP_SUB[0].upper() + TOP_SUB[1:]}.", ""]
        out += [f"{j}. **{md(q)}** {md(a)}" + (f" (→ {refs})" if refs else "") for j, (q, a, refs) in enumerate(page1["top"], 1)] + ["", md(page1["also"]), ""]
        out += [md(contents(items)), ""]
    out += [f"## {md(STORY_HEAD)}", ""] + [f"{j}. {md(s)}" for j, s in enumerate(story, 1)] + [""]
    out += [md(use), ""] + ([f"> {md(wait)}", ""] if wait else []) + [md(words), ""] + ([] if page1["top"] else [md(contents(items)), ""])
    for g, title in (g_t for g_t in GROUPS.items() if any(i["group"] == g_t[0] for i in items)):
        out += [f"## {g}. {title}", ""]
        for i in (i for i in items if i["group"] == g):
            out += [f"### {i['number']}. {md(i['q'])}", "", md(i["a"]), "", f"**The trap:** {md(i['trap'])}", "", f"**Where:** {md(i['where'])}", ""]
    out += [f"## {CHECK_HEAD}", "", CHECK_LEAD, ""]
    for who, todo in checklist:
        out += [f"**{md(who)}**", ""] + [f"- {md(t)}" for t in todo] + [""]
    if CITED:                                                      # the same order as in the PDF: the checks after the last question, the sources last
        out += [f"## {REFS_HEAD}", ""] + [f"- {md(r)}" for r in references_html()] + [""]
    return "\n".join(out)


if __name__ == "__main__":
    R, T = load_results()
    now = datetime.now()
    built = f"{now.day} {MONTH[now.month - 1]} {now.year}"       # the day of the build; the clock time is not printed
    story, items, checklist, pending, page1 = build(R, T)
    parts = (R, story, items, checklist, pending, page1, built)
    (ROOT / "Professor_QA.md").write_text(to_md(*parts), encoding="utf-8")
    out = ROOT / "Professor_QA.html"
    out.write_text(to_html(*parts), encoding="utf-8")
    print(f"wrote {ROOT / 'Professor_QA.md'} and {out} ({len(items)} questions, {len(pending)} results pending)")
    if print_pdf(out, out.with_suffix(".pdf")):
        print("wrote", out.with_suffix(".pdf"))
