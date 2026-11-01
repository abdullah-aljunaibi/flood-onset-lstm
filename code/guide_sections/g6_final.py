"""Guide, steps 17 to 23 - the work after the mid-term talk of 15 October 2026, up to the state of the final hand-in (30 October 2026).

  17  A second model from the team, and what it pointed to   the Flood_Cloud_Model package of the other members: gradient boosting, why a high
                                                              accuracy misleads on rare events, what counts (the cases still below the level), what it pointed to
  18  A stricter warning level                                E4a: no training; the stored forecasts at two levels chosen on the validation years; a decision
                                                              layer re-arranges the trade-off and adds no skill (the box 'operating point' of step 16 is reused)
  19  How the day's rain fell, hour by hour                   E4b: six within-day AORC columns against a control one day older; the rule is not met
  20  The river at the issue time                             E4e: the gauge's hourly record; why it needed the shorter split; the ladder on that split (Table 6
                                                              of the final report); what is and is not established; delivery delay
  21  Five trainings instead of three                         seeds 3 and 4 of scripts 15-18, added after the test scores were known; what they changed
  22  The basins as GIS layers                                the export, its checks against CAMELS and step 12, the two review rounds
  23  Where the project ended                                 the final ladder, GO not met, what a service would need, what was handed in, the members' checks,
                                                              who did what, what is still open before sending

Written for readers who are new to machine learning, in the style of the earlier parts: short sentences, plain words, a new idea in a box where it is first
used (four new concepts: 'boosting' and 'accuracy' in step 17, 'latency' in step 20, 'after_looking' in step 21; every other technical word was explained before).

Where the numbers come from (each read by key, none typed in): the registry of the final report, results/final_report_numbers.json, whose entries each hold the
value, the text the final report prints and the source file and key (_v, _t); for logic and for keys the registry lacks, the results files themselves
(flow_state.json, within_day_inputs.json, warning_policy.json, gis_layers.json, review_*.json, the five_seed_changes blocks of the experiment files and the
three-seed copy in results/archive/), the verified readers of report_sections/s0_summary.py, team.json through team.py, and three hand-in files that are read,
never written: the final report (.docx for its text, .pdf for its page count), the speaker notes of the final deck and README_HANDIN.md. Nothing is computed
beyond differences, ratios of counts and formatting. The fixed wordings (AORC_WORDING, REALTIME_WORDING, the hindsight clause, FINDING, 'an upper bound that
cannot be had in practice', 'no clear difference') are used word for word.

Every paragraph that states an outcome is built inside _try() and first checks (_need) that the stored numbers still say what the sentence says; otherwise it is left
out with one line on stderr. A step whose results file is missing is left out whole, and build_guide.py then says so.
Layout: as in g5_next: a picture is a full-width float that moves to the top of the next page when it does not fit; the boxes and the end of each step clear it.
Three pictures are shown without their own long header and footnote (s0_summary.fig_cropped): the captions here repeat what those said.
"""
import datetime
import json
import re
import sys

import guide_common as G
import team
from common import FIG, ROOT
from flood import LEADS
from report_common import LEAD_WORD, cite, count, fig, pct, signed, table, tag
from report_sections.s0_summary import (AORC_WORDING, FINDING, REALTIME_WORDING, READING, _and, _day, _esc, _word, aorc_facts, fig_cropped, forecast_facts,
                                         hindsight_clause, hindsight_facts, rivers_facts, stage4_facts)

FIRST = LEADS[0]
UPPER = "an upper bound that cannot be had in practice"                 # the fixed wording beside every perfect-forecast or perfect-rain number
HANDIN = ROOT.parent / "Handin_2026-11-02"                              # the hand-in folder of the final documents (read only)
REPORT_DOCX, REPORT_PDF = HANDIN / "3_Final_Project_Report_Flood.docx", HANDIN / G.FINAL_REPORT
NOTES = HANDIN / "4_Final_Project_Presentation_notes.md"
UPPER_SETS = ("lstm_wqf", "lstm_wqaP", "lstm_wqaP_s")                  # the sets fed the observed weather or rain of the target days
STYLE = ("<style>.g6{clear:both} .g6 .ff{float:left;width:100%;break-inside:avoid} .g6 .ff figure{margin-top:2pt} "
         ".g6 h2,.g6 .concept,.g6 .remember,.g6 .keep,.g6 .who,.g6 ul.short,.g6 ol.short,.g6 table{clear:both} .g6 .keep{break-inside:avoid} "
         ".g6 ol.short,.g6 ul.short{margin:2pt 0 6pt;padding-left:17pt} .g6 .short li{margin-bottom:2pt} .g6 p.intro{break-after:avoid;margin-bottom:2pt} "
         ".g6 table.ladder{margin:3pt 0 6pt} .g6 table.ladder td,.g6 table.ladder th{padding:2.2pt 6pt} .g6 table.ladder td.grp{font-weight:600;color:var(--ink2);padding-top:5pt} "
         ".g6 table.slides td,.g6 table.slides th{padding:2pt 6pt}</style>")


# ---- reading ------------------------------------------------------------------------------------------------------------------
def _try(what, make):
    """make() builds one paragraph (or box, caption, table) from the files; if a file or key is missing, or the stored numbers no longer support the sentence (_need),
    the piece is left out and one line on stderr says so."""
    try:
        return make()
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError, OSError, StopIteration) as e:
        print(f"g6_final: {what} left out ({type(e).__name__}: {e})", file=sys.stderr)
        return ""


def _need(holds, outcome):
    if not holds:
        raise ValueError(f"the results files no longer show that {outcome}")


def _reg(R):
    X = R.get("final_report_numbers")
    if not isinstance(X, dict) or "_meta" not in X:
        raise KeyError("results/final_report_numbers.json (the registry of the final report)")
    return X


def _v(R, key):
    """The value of a registry entry."""
    return _reg(R)[key]["value"]


def _t(R, key):
    """The text of a registry entry, as the final report prints it."""
    return _esc(_reg(R)[key]["text"])


_CONFIG = {}


def config():
    """code/final_report/config.json, read once: the group number, the group leader, the members with their roles and student ids, the course, the instructor."""
    if "c" not in _CONFIG:
        _CONFIG["c"] = json.loads((ROOT / "code" / "final_report" / "config.json").read_text(encoding="utf-8"))
    return _CONFIG["c"]


def _years(span):
    return f"{str(span[0])[:4]}&ndash;{str(span[1])[:4]}"


def _cap(s):
    return s[0].upper() + s[1:]


_DOCX = {}


def _read_report():
    """Read the final report (.docx) once: its body paragraphs and the text of its table cells, as plain text."""
    if "p" not in _DOCX:
        try:
            import docx
            d = docx.Document(str(REPORT_DOCX))
            _DOCX["p"] = [p.text.strip() for p in d.paragraphs]
            _DOCX["t"] = [c.text.strip() for tb in d.tables for row in tb.rows for c in row.cells]
        except Exception as e:                                                     # any reader failure leaves the sentences that quote the report out
            print(f"g6_final: the final report {REPORT_DOCX.name} could not be read ({type(e).__name__}: {e}): what quotes it is left out", file=sys.stderr)
            _DOCX["p"], _DOCX["t"] = [], []
    return _DOCX


def report_paragraphs():
    """The body paragraphs of the final report (.docx) as plain text; [] (and one line on stderr) when the file or python-docx is missing."""
    return _read_report()["p"]


def report_says(*phrases):
    """True when the final report's text (paragraphs and table cells) holds every phrase, whatever the form of its apostrophes (the sentences that state an
    open item check this first)."""
    plain = lambda s: s.replace("’", "'").replace("‘", "'")
    D = _read_report()
    text = plain("\n".join(D["p"] + D["t"]))
    return bool(text) and all(plain(p) in text for p in phrases)


def report_references():
    """The number of entries in section 11 (References) of the final report."""
    ps = report_paragraphs()
    i = ps.index("11. References")
    j = next(k for k in range(i + 1, len(ps)) if ps[k] == "Appendices")
    n = sum(bool(re.match(r"\[\d+\] ", x)) for x in ps[i + 1:j])
    if not n:
        raise ValueError("section 11 of the final report lists no numbered reference")
    return n


def declaration():
    """The three paragraphs of Appendix E of the final report that say who did what, word for word: [(label, rest)] ('AI-assistance declaration.',
    'What the AI did not do.', 'The teammates' package.'); [] when the report cannot be read."""
    out = []
    for label in ("AI-assistance declaration.", "What the AI did not do.", "The teammates’ package.", "The teammates' package."):
        p = next((x for x in report_paragraphs() if x.startswith(label)), None)
        if p and not any(lab == label.replace("'", "’") or lab.replace("’", "'") == label for lab, _ in out):
            out.append((label, p[len(label):].strip()))
    return out


def _slides():
    """The timing table of the speaker notes of the final deck: [(number, title, speaker, seconds)], read from the hand-in folder."""
    rows = []
    for line in NOTES.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3), int(m.group(4))))
    if not rows:
        raise ValueError(f"{NOTES.name} holds no timing table")
    return rows


def _seed_counts(R):
    """(three, five): the trainings per set before and after the extension of 30 October 2026 (aorc_rain_to_midnight.json five_seed_changes)."""
    S = R["aorc_rain_to_midnight"]["five_seed_changes"]["seeds"]
    return len(S["three_seed_file"]), len(S["this_file"])


def last_day(R):
    """The last day of work that the results record: the latest review date of the latest rounds of the review records and the written dates of the files
    after the mid-term ('30 October 2026')."""
    isos = []
    for k, X in R.items():
        if k.startswith("review_") and isinstance(X, dict):
            rounds = [X] + [X[r] for r in X if re.fullmatch(r"round_\d+", r) and isinstance(X[r], dict)]
            for blk in rounds:
                isos += re.findall(r"\d{4}-\d{2}-\d{2}", json.dumps(blk.get("dates") or {}) + " " + str(blk.get("when", "")))
    for k in ("flow_state", "within_day_inputs", "gis_layers"):
        isos += re.findall(r"\d{4}-\d{2}-\d{2}", str((R.get(k) or {}).get("written", "")))
    isos = [x for x in isos if x <= datetime.date.today().isoformat()]
    if not isos:
        raise ValueError("no review or results file gives a date")
    return _day(max(isos))


def available(R):
    """True when the files that steps 17 to 23 stand on can all be read (the registry of the final report, the three E4 results files, the GIS layers and
    their reviews); the steps before point to these steps only while this holds."""
    try:
        _reg(R)
        return all(isinstance(R.get(k), dict) and not R[k].get("quick_mode") for k in ("flow_state", "within_day_inputs", "warning_policy", "gis_layers",
                                                                                         "review_flow_state", "review_within_day_inputs", "review_warning_policy",
                                                                                         "review_gis_layers"))
    except KeyError:
        return False


def _picture(name, caption, top=0.0, bottom=0.0):
    """A picture of figures/ as a full-width float; top/bottom: the shares of its height not shown (its own header and footnote, which the caption repeats)."""
    if not (FIG / name).exists():
        raise ValueError(f"figures/{name} is missing")
    figure = fig_cropped(name, caption, top=top, bottom=bottom) if (top or bottom) else fig(name, caption)
    return f'<div class="ff">{figure}</div>'


def _review_line(R, name, what):
    """One sentence on the independent review of one experiment, from its record (results/review_<name>.json): the rounds, the values recomputed in the
    last round, the mismatches, the verdict, and whether the results file says that the wording fixes were applied."""
    X = R[f"review_{name}"]
    rounds = sorted((k for k in X if re.fullmatch(r"round_\d+", k)), key=lambda k: int(k[6:]))
    last = X[rounds[-1]] if rounds else X
    c = last["counts"]
    _need(c.get("numerical_mismatches", 1) == 0, f"the last review round of {what} found no number that differed")
    applied = "applied" in str((R.get(name) or {}).get("status", ""))
    return (f"An independent AI reviewer agent checked {what} in {_word(1 + len(rounds))} rounds; the last, with all trainings, recomputed {count(c['values_compared'])} "
            f"values from the stored forecasts and found none that differed: &lsquo;{_esc(last['verdict'])}&rsquo;"
            + (", and the wording fixes it asked for were made." if applied else "."))


# ---- the part's opening and step 17: the package of the other members -----------------------------------------------------------
def _intro(R):
    """The paragraph in front of step 17: what the last part of the guide covers, and who did it."""
    return (f'<h2 class="plain">After the mid-term talk</h2><p>Steps 1 to 16 are the project as it stood at the mid-term talk of {_t(R, "date_proposal_talk")}. On '
            f"{_t(R, 'date_windows_handoff')} the project moved to a second machine, a Windows PC, and the work went on until {last_day(R)}. Steps 17 to 22 follow "
            f"that work in order, and step 23 says where the project ended. Claude Code did the work under the direction of the group leader, "
            f"{_t(R, 'group_leader')}, except the package of step 17, which the other members built. Independent AI reviewer agents checked every new experiment; "
            f"the team members&rsquo; own checks are still to be done.</p>")


def _step17(R):
    pkg = ROOT / "handoff" / "extras" / "Flood_Cloud_Model"
    if not pkg.is_dir():
        raise ValueError("handoff/extras/Flood_Cloud_Model is missing")
    n = len(R["lstm_all"]["basins"])
    h = _t(R, "fcm_primary_horizon_hours")
    review = sorted((ROOT / "handoff" / "extras").glob("Flood_Cloud_Model_REVIEW_*.md"))
    when = _day(re.search(r"(\d{4}-\d{2}-\d{2})", review[-1].name).group(1)) if review else None

    def opening():
        return (f"<p>After the mid-term talk a second experiment on the same {n} rivers was built: the package <b>Flood_Cloud_Model</b>, the work of "
                f"{_t(R, 'flood_cloud_model_authors')}. It is not a version of the LSTM. It forecasts hour by hour, for the years {_t(R, 'fcm_period_years')}, from "
                f"the river&rsquo;s hourly flow, radar rain and all {_t(R, 'fcm_gefs_members')} runs of the US weather forecast (GEFS) as they were issued at the time, "
                f"and it answers yes or no: will the river pass its high-flow level within the next hours? Its model is gradient boosting.</p>")

    def box_boosting():
        return G.concept("boosting", here="The package uses it to say yes or no for each forecast window. Each of its models was fitted once, with no repeated "
                                          "trainings (step 7), and its test years had been used while the package was developed (step 3 says why that matters). "
                                          "So its shares cannot stand beside those of the LSTM.")

    def headline():
        _need(when, "an AI agent reviewed the package (handoff/extras/Flood_Cloud_Model_REVIEW_<date>.md)")
        return (f"<p>On {when} an AI agent (Claude Code) reviewed the package and recomputed its numbers from its own stored results. The package&rsquo;s summary leads "
                f"with {_t(R, 'fcm_headline_accuracy')} accuracy and {_t(R, 'fcm_headline_detection')} detection, {h} hours ahead. Both numbers are right, and both "
                f"mislead. Most hours have no flood, so being right most of the time is easy (the box). And many of the cases it detects are hours when the river "
                f"was already above its high-flow level: &lsquo;tomorrow = today&rsquo; alone detects {_t(R, 'fcm_persistence_detection_24h')} of them. These are the "
                f"two traps of steps 4 and 10.</p>")

    def box_accuracy():
        total, low, pos = _v(R, "fcm_test_cases_24h"), _v(R, "fcm_onset_cases_24h"), _v(R, "fcm_positive_onset_cases_24h")
        _need(0 < pos < low < total, "the package's test cases split into cases above and below the level")
        return G.concept("accuracy", here=f"Of the package&rsquo;s {count(total)} test cases {h} hours ahead, {count(total - low)} began with the river already above "
                                          f"its high-flow level. Of the {count(low)} that began below it, only {count(pos)} rose above it: a model that never warned would "
                                          f"be right on {pct(1 - pos / low)} of them.")

    def what_counts():
        a, b, c = (_v(R, f"fcm_onset_only_recall_{k}_24h") for k in ("flow", "rain", "gefs"))
        _need(c > b >= a and b - a < 0.05, "the forecast lifts the share warned most and the observed rain adds almost nothing")
        return (f"<p>What counts are the cases that began below the level: the onsets of step 4. Of these the package warned, {h} hours ahead, of "
                f"{_t(R, 'fcm_onset_only_recall_flow_24h')} with the river&rsquo;s flow alone, {_t(R, 'fcm_onset_only_recall_rain_24h')} with the observed rain as well "
                f"and {_t(R, 'fcm_onset_only_recall_gefs_24h')} with the weather forecast. As in step 16, the forecast helped most. These shares are not comparable in "
                f"level with those of the LSTM: no repeated trainings, test years used in development, and an hourly event in place of a daily one.</p>"
                f"<p>The observed rain added almost nothing once the model saw the river&rsquo;s hourly flow: the rain of the last hours was already in the rising "
                f"river. The LSTM read the gauge only as the mean flow of each day, which hides what the river does in the evening. So the package pointed at the "
                f"input the LSTM lacked one day ahead: the state of the river at the issue time. Step 20 tests it.</p>")

    def statement():
        _need(report_says("no statement of how AI was used in it has been supplied yet"), "the authors of the package have not yet supplied a statement of AI use")
        return ("<p>Its authors have not yet written how they used AI to build it. The final report says so, and they are to add that statement.</p>")

    remember = ("A high accuracy says little when the event is rare: count the events that were warned of, and the false warnings. Two experiments can be set "
                "side by side only when they are scored the same way.")
    return f"""
{G.keep(_try("the opening of the last part", lambda: _intro(R)), G.step(17, "The other team members built a model of their own for the same rivers. Read with care, its numbers taught two lessons, and it pointed at an input that the LSTM had never been given."), _try("step 17, the package", opening))}
{_try("step 17, the box 'gradient boosting'", box_boosting)}
{_try("step 17, the headline numbers", headline)}
{_try("step 17, the box 'accuracy'", box_accuracy) or G.concept("accuracy")}
{_try("step 17, what counts and what it pointed to", what_counts)}
{G.keep(_try("step 17, the statement of AI use", statement), G.remember(remember), G.full_detail("Sections 8.1 and 8.3, and Appendix E"))}"""


# ---- step 18: a stricter warning level (E4a) ----------------------------------------------------------------------------------
def _step18(R, hind):
    P = R["warning_policy"]
    cap = _t(R, "go_max_false")

    def opening():
        return ("<p>The first experiment after the talk, E4a, trained nothing. It took the stored forecasts of the earlier steps and moved only the warning level: how "
                "far a forecast must rise before it gives a warning (the box &lsquo;Operating point&rsquo; of step 16). For each model two levels were chosen on the "
                "validation years. The first scores best by the share that CAUTION counts in step 4, the hits among all hits, misses and false warnings, called the "
                f"CSI. The second warns of as many onsets as it can while at most {cap} of its warnings are false, the limit that GO allows.</p>")

    def result():
        met = P["go_met_by"]
        _need(set(met) <= set(UPPER_SETS) and not any(" l1 " in f" {x} " for x in P["go_met_at"]),
              "no set that a service could have meets GO at any level, and no set at all 1 day ahead")
        return (f"<p>With the AORC rain to midnight ({AORC_WORDING}), the first level warns of {_t(R, 'e4a_lstm_wqa_ll1_csi_test_onset_pod')} of the onsets "
                f"{LEAD_WORD[FIRST]}" + (f" ({hind})" if hind else "") + f", with {_t(R, 'e4a_lstm_wqa_ll1_csi_test_onset_far')} of its warnings false. The second "
                f"keeps only {_t(R, 'e4a_lstm_wqa_ll1_far_cap_test_onset_pod')} warned, with {_t(R, 'e4a_lstm_wqa_ll1_far_cap_test_onset_far')} false on the test years. "
                f"Even the model with a perfect weather forecast keeps only {_t(R, 'e4a_lstm_wqf_ll1_far_cap_test_onset_pod')} there, {UPPER}. No model that a "
                f"service could have reaches GO at either level. A rule on top of the forecast, a decision layer, re-arranges the trade-off between warned onsets and "
                f"false warnings; it adds no skill.</p>")

    def picture():
        rows = P["summary"][f"l{FIRST}"]
        inside = [r["set"] for r in rows for k in ("csi", "far_cap", "stack") if isinstance(r.get(k), dict) and r[k].get("warned") is not None
                  and r[k]["warned"] >= _v(R, "go_min_warned") and r[k]["false"] <= _v(R, "go_max_false")]
        _need(not inside, "no mark of the picture lies inside the GO box")
        both = [r for r in rows if r["csi"].get("reached") and r["far_cap"].get("reached")]
        down_left = all(r["far_cap"]["warned"] <= r["csi"]["warned"] and r["far_cap"]["false"] <= r["csi"]["false"] + 1e-9 for r in both)
        cap_text = (f"<b>Where the warning sits.</b> <b>How to read it:</b> Up: the share of the onsets warned {LEAD_WORD[FIRST]}; right: the share of the warnings "
                    f"that were false. The blue box at the top left is GO. Each colour is one model of the earlier steps, from the rain table with the river gauge "
                    f"(grey) to the perfect weather (red). Its dot is the level with the best CSI, its square the level that keeps at most {cap} false on the "
                    f"validation years; the line between them is the trade-off."
                    + (" Every line runs down to the left: fewer false warnings, fewer onsets warned." if down_left else "")
                    + " Triangles: a small extra rule fitted on the validation years; hollow marks: the shorter test of step 20; thin bars: the range when the test "
                      "days are drawn again (step 14). No mark enters the box.")
        return _picture("warning_policy.png", cap_text, top=0.045, bottom=0.135)

    remember = "A warning level trades warned onsets for false warnings. It moves a model along its trade-off; it cannot add what the model does not know."
    return f"""
{G.keep(G.step(18, "Before any new training, the stored forecasts were asked what the warning rule alone can change."), _try("step 18, the opening", opening))}
{_try("step 18, the picture", picture)}
{_try("step 18, the result", result)}
{G.keep(_try("step 18, the review", lambda: f"<p>{_review_line(R, 'warning_policy', 'E4a')}</p>"), G.remember(remember),
         G.full_detail("Section 7.3 (&lsquo;A second operating point&rsquo;) and Section 9.3"))}"""


# ---- step 19: how the day's rain fell (E4b) -------------------------------------------------------------------------------------
def _step19(R, hind):
    W = R["within_day_inputs"]
    S = W["e4b_success"]
    cols = W["design"]["sets"]["wqaW"]["extra_columns"]
    hours = [int(m) for c in cols for m in re.findall(r"aorc_last(\d+)h", c)]
    n_seeds = _word(len(W["seeds"]))

    def opening():
        _need(len(hours) == 3 and {"aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours"} <= set(cols), "the columns are the rain of the last hours, the wettest hour "
                                                                                                    "(basin and cell) and the wet hours")
        return (f"<p>Step 16 gave the model the rain of the issue day up to midnight as one total. But a storm that drops it all in the last hours before midnight is "
                f"more dangerous than the same total spread over the day. E4b added {_word(len(cols))} numbers that describe how the issue day&rsquo;s AORC rain "
                f"fell: the rain of the last {_and(hours)} hours before midnight, the wettest hour over the basin, the wettest hour in any one small cell of the rain "
                f"grid, and the number of wet hours. It is the same rain described hour by hour, not more rain. The control is the same {_word(len(cols))} numbers one "
                f"day older: they hold no news about the issue day. Every set was trained {n_seeds} times.</p>")

    def result():
        _need(S["met"] is False and S["control_flat"], "the rule fixed before training is not met and the control is flat")
        return (f"<p>{_cap(LEAD_WORD[FIRST])} the onsets warned went from {_t(R, 'e4b_wqa_l1_onset_pod')}" + (f" ({hind})" if hind else "")
                + f" to {_t(R, 'e4b_wqaw_l1_onset_pod')}, and the false warnings from {_t(R, 'e4b_wqa_l1_onset_far')} to {_t(R, 'e4b_wqaw_l1_onset_far')}. The control "
                f"reached {_t(R, 'e4b_wqaw_older_l1_onset_pod')}. The rule fixed before training asked that every training with the new numbers lie above every "
                f"training without them and above every training of the control. It is not met, so there is no clear gain. At the stricter level of step 18 the new "
                f"numbers warn of {_t(R, 'e4b_wqaw_l1_far_cap_pod')}, the control of {_t(R, 'e4b_wqaw_older_l1_far_cap_pod')}. One more limit: the hindsight of AORC "
                f"was measured for its daily total only, not for these hour-by-hour numbers.</p>")

    def picture():
        _need(not S["every_seed_of_wqaQ_above_every_seed_of_wqa"], "the flow measured against the high-flow level is no clear gain")
        cap = (f"<b>How the day&rsquo;s rain fell, as inputs (E4b).</b> <b>How to read it:</b> Top: the share of the {_t(R, 'onsets_l1')} onsets warned "
               f"{LEAD_WORD[FIRST]}; bottom: the median NSE. A bar is the average forecast of the {n_seeds} trainings, a dot one training. Green: the AORC rain to "
               f"midnight alone ({AORC_WORDING}); the dashed line repeats its level. Blue: the same with the day&rsquo;s flow measured against the high-flow level "
               f"(no clear gain), with the {_word(len(cols))} hour-by-hour numbers, and the control, hollow, with the same numbers one day older. Look at the third bar "
               f"against the hollow one: their dots overlap. Orange: the same with the weather forecast of step 16. Yellow and the dotted line: the perfect rain of "
               f"the next days, {UPPER}.")
        return _picture("within_day_inputs.png", cap, top=0.17, bottom=0.16)

    remember = "Describing the same information in more detail is not new information. Here the hour-by-hour rain gave no clear gain over its control."
    return f"""
{G.keep(G.step(19, "The model already had the whole day&rsquo;s rain. Would it help to know how that rain fell over the hours?"), _try("step 19, the opening", opening))}
{_try("step 19, the picture", picture)}
{_try("step 19, the result", result)}
{G.keep(_try("step 19, the review", lambda: f"<p>{_review_line(R, 'within_day_inputs', 'E4b')}</p>"), G.remember(remember),
         G.full_detail("Section 7.3 (&lsquo;Within-day inputs&rsquo;) and Table 14 in Appendix C"))}"""


# ---- step 20: the river at the issue time (E4e) -----------------------------------------------------------------------------------
def _step20(R):
    F = R["flow_state"]
    S = F["e4e_success"]
    sets = F["design"]["sets"]
    cols = sets["wqaH_s"]["extra_columns"]
    h24, h22 = sets["wqaH_s"]["last_available_hour"], sets["wqaH22_s"]["last_available_hour"]
    n5 = len(F["seeds"])
    gap = _word(h24 - h22)
    t = lambda key: _t(R, f"e4e_{key}")
    v = lambda key: _v(R, f"e4e_{key}")

    def opening():
        minutes = re.search(r"(\d+)-minute record", F["outcome"]).group(1)
        changes = [int(x) for c in cols for x in re.findall(r"^dq(\d+)$", c)]
        _need(len(cols) == 5 and len(changes) == 2, "the river's state is five numbers, two of them changes over hours")
        return (f"<p>Since step 8 the model has read the river gauge as one number a day: the mean flow of the day. At midnight, a river that began to rise in the "
                f"evening still looks calm in that mean. The US Geological Survey keeps the flow of every gauge every {minutes} minutes. That record was fetched for "
                f"all {F['n_basins']} gauges and reduced to hours. From it, E4e gave the model {_word(len(cols))} numbers for the river at midnight: its flow against "
                f"the high-flow level, its change over the last {_and(changes)} hours, the day&rsquo;s highest hourly flow, and the hours it spent above the "
                f"high-flow level.</p>")

    def own_split():
        first = sorted(int(b["first_hour_end_utc"][:4]) for b in F["data_coverage"]["per_basin"].values())
        main_train = R["lstm_all"]["split"]["train"]
        _need(first[0] > int(str(main_train[0])[:4]), "every hourly record begins after the training years of the main test begin")
        sp = F["split"]
        return (f"<p>Why a test of its own? The hourly records begin between {first[0]} and {first[-1]}, depending on the gauge: after the training years of the main "
                f"test begin, in {str(main_train[0])[:4]}. So E4e could not use the years of steps 3 to 16. It used the shorter split of the satellite inputs (step 12) "
                f"and of the retrained forecast (step 16): training {_years(sp['train'])}, validation {_years(sp['val'])}, test {_years(sp['test'])}, with "
                f"{_t(R, 'onsets_sat_l1')} onsets. Its numbers can be compared with each other, never with the {_t(R, 'onsets_l1')} onsets of the main test.</p>")

    def ladder():
        verdict = t("wqawhf_s_l1_plain_test_verdict_onset")
        go_w, go_f = _v(R, "go_min_warned"), _v(R, "go_max_false")
        full_w, full_f = v("wqawhf_s_l1_onset_pod"), v("wqawhf_s_l1_onset_far")
        _need(verdict == "CAUTION" and (full_w < go_w or full_f > go_f), "the full model is CAUTION and not GO")
        why = _and(([f"below {_t(R, 'go_min_warned')} warned"] if full_w < go_w else []) + ([f"above {_t(R, 'go_max_false')} false"] if full_f > go_f else []))
        return (f'<p class="intro"><b>The ladder on this split</b>, {LEAD_WORD[FIRST]}: the share of the {_t(R, "onsets_sat_l1")} onsets warned, and the share of the '
                f"warnings that were false. Every rung reads the AORC rain to midnight, {AORC_WORDING}.</p>"
                f'<ol class="short"><li>The base, AORC rain to midnight and the river gauge: <b>{t("wqa_s_l1_onset_pod")}</b> warned, {t("wqa_s_l1_onset_far")} false.</li>'
                f'<li>Plus the river at midnight: <b>{t("wqah_s_l1_onset_pod")}</b>, {t("wqah_s_l1_onset_far")} false.</li>'
                f'<li>Plus the hour-by-hour rain of step 19: <b>{t("wqawh_s_l1_onset_pod")}</b>, {t("wqawh_s_l1_onset_far")} false.</li>'
                f'<li>Plus the weather forecast of step 16, the full model: <b>{t("wqawhf_s_l1_onset_pod")}</b>, {t("wqawhf_s_l1_onset_far")} false. By the rule fixed in '
                f"advance this is {tag(verdict)} (CSI {t('wqawhf_s_l1_onset_csi')}); it is not GO: {why}.</li></ol>"
                f"<p>Beside them: the base with the forecast alone, tested before the mid-term (step 16), warns of {t('wqaf_s_l1_onset_pod')} with "
                f"{t('wqaf_s_l1_onset_far')} false; the perfect rain of the next days of {t('wqap_s_l1_onset_pod')} with {t('wqap_s_l1_onset_far')} false, {UPPER}.</p>")

    def state():
        far = [v(f"{s}_l1_onset_far") for s in ("wqa_s", "wqah_s", "wqawh_s", "wqawhf_s")]
        cap = [v(f"{s}_l1_far_cap_test_onset_pod") for s in ("wqa_s", "wqah_s", "wqawh_s", "wqawhf_s")]
        drops = [far[i] - far[i + 1] for i in range(3)]
        steps = [cap[i + 1] - cap[i] for i in range(3)]
        _need(drops[0] == max(drops) and steps[0] == max(steps), "the river's state cuts the false share most and gives the biggest step at the stricter level")
        three = len(S["first_three_seeds"])
        _need(S["met"] is False and S["met_with_first_three_seeds"] is True and S["every_seed_of_wqaWH_s_above_every_seed_of_wqa_s"] is True,
              "the rule holds with the first three trainings and not with five, and holds with the hour-by-hour rain beside it")
        _need("before any test score was seen" in F["choices_made_after_looking_at_test_scores"][0], "the five trainings were fixed before any test score was seen")
        return (f"<p><b>What the river&rsquo;s state did.</b> It cut the false warnings most, from {t('wqa_s_l1_onset_far')} to {t('wqah_s_l1_onset_far')}. At the "
                f"stricter level of step 18 it gave the biggest single step: from {t('wqa_s_l1_far_cap_test_onset_pod')} to {t('wqah_s_l1_far_cap_test_onset_pod')} of "
                f"the onsets warned, against {t('wqawh_s_l1_far_cap_test_onset_pod')} with the hour-by-hour rain beside it and {t('wqawhf_s_l1_far_cap_test_onset_pod')} "
                f"for the full model. But its gain in onsets warned at the usual level is not established. The rule fixed before training asks that every training "
                f"with the new numbers lie above every training without them. That holds for the first {_word(three)} of its {_word(n5)} trainings and not for all "
                f"{_word(n5)}; the {_word(n5)} ran together from the start, before any test score was seen, so the rule is judged on all of them. With the "
                f"hour-by-hour rain beside it, every training lies above every training of the base.</p>")

    def rise():
        base, alone, full = v("wqa_s_l1_onset_pod"), v("wqaf_s_l1_onset_pod"), v("wqawhf_s_l1_onset_pod")
        _need(alone - base > full - alone > 0, "most of the rise from the base to the full model comes from the forecast")
        return (f"<p>Most of the rise from {t('wqa_s_l1_onset_pod')} to {t('wqawhf_s_l1_onset_pod')} comes from the weather forecast, which was tested before the "
                f"mid-term: alone it lifts the base to {t('wqaf_s_l1_onset_pod')}. The new inputs mainly cut false warnings.</p>")

    def service():
        _need(v("wqawhf22_s_l1_onset_far") > _v(R, "go_max_false"), "the full model with the earlier record end is not GO")
        _need(any("approved" in x for x in F["what_was_not_done"]), "the study used the approved gauge record with no delivery delay")
        return (f"<p><b>What a service would have.</b> Every rung reads inputs that a real service does not have in this form: the AORC rain ({AORC_WORDING}), the "
                f"archived forecast, and the approved gauge record with no delivery delay (the box). How much worse a service would do was not measured. One test was "
                f"made: with the gauge record ending {gap} hours earlier, the full model warns of {t('wqawhf22_s_l1_onset_pod')} with {t('wqawhf22_s_l1_onset_far')} "
                f"false, so GO is still not met.</p>")

    def box():
        S4 = stage4_facts(R)
        return G.concept("latency", here=f"A service receives the gauge record as provisional values, some time after they are measured; this study used the approved "
                                         f"record as if it arrived at once, and as one test a record that ends {gap} hours before the issue time. Stage IV, the radar "
                                         f"rain of step 16, comes about an hour late: without its last hour the networks warn of {pct(S4['no_last'] / S4['n'])} of the "
                                         f"onsets instead of {pct(S4['stage4'] / S4['n'])}, {REALTIME_WORDING}.")

    def picture():
        _need(all(x > _v(R, "go_max_false") for x in S["onset_false"].values()), "every bar of the picture has more false warnings than GO allows")
        cap = (f"<b>The river at the issue time (E4e), on its own split.</b> <b>How to read it:</b> Top: the share of the {_t(R, 'onsets_sat_l1')} onsets warned "
               f"{LEAD_WORD[FIRST]}, with the share of false warnings under each number; bottom: the median NSE. A bar is the average forecast of the {_word(n5)} "
               f"trainings, a dot one training, the white diamond the onsets warned at the stricter level of step 18. Green: the base. Blue: with the river at "
               f"midnight (the record to {h24}:00, or to {h22}:00), then with the hour-by-hour rain. Orange: with the weather forecast, alone (hatched, tested before "
               f"the mid-term) and with the new inputs. Yellow, hatched: the perfect rain, {UPPER}. The grey line is the {_t(R, 'go_min_warned')} that GO asks for; "
               f"every bar has more than {_t(R, 'go_max_false')} false warnings. AORC is {AORC_WORDING}.")
        return _picture("flow_state.png", cap, top=0.19, bottom=0.15)

    remember = ("A daily mean can hide what matters at the moment of the forecast. The river&rsquo;s state cut false warnings, but by the rule fixed in advance its "
                "gain in warned onsets is not established; most of the rise came from the forecast, and GO is still not met.")
    return f"""
{G.keep(G.step(20, "A daily mean hides what the river is doing at midnight, when the forecast is made. The gauge&rsquo;s own hourly record shows it."), _try("step 20, the opening", opening))}
{_try("step 20, why its own split", own_split)}
{_try("step 20, the picture", picture)}
{_try("step 20, the ladder", ladder)}
{_try("step 20, what the river's state did", state)}
{_try("step 20, where the rise comes from", rise)}
{_try("step 20, what a service would have", service)}
{_try("step 20, the box 'delivery delay'", box) or G.concept("latency")}
{G.keep(_try("step 20, the review", lambda: f"<p>{_review_line(R, 'flow_state', 'E4e')}</p>"), G.remember(remember),
         G.full_detail("Sections 5.3 and 7.3, Table 6 in Section 7.1, and Section 8.2"))}"""


# ---- step 21: five trainings instead of three -----------------------------------------------------------------------------------
def _step21(R, HS):
    three, five = _seed_counts(R)

    def opening():
        _need(five > three, "the experiment sets gained trainings")
        return (f"<p>Step 7 showed why one training is not enough: trainings with different seeds give different networks, and a difference between two models "
                f"counts only if it is larger than that. The experiments of step 16 were trained {_word(three)} times each, and at the mid-term talk the team promised "
                f"{_word(five)} trainings where the NSE decides. On {_t(R, 'seed_extension_date')} {_word(five - three)} more trainings were added to every set of "
                f"scripts {_t(R, 'seed_extension_scripts')}: {_t(R, 'seed_extension_trainings')} trainings on the second machine. They were added after the test "
                f"scores of the first {_word(three)} were known. Nothing else changed: the settings, the inputs, the split, the stopping rule, the warning level and "
                f"the success rules stayed as they were.</p>")

    def box():
        new = [len(R[k]["seeds"]) for k in ("within_day_inputs", "flow_state")]
        _need(all(n == five for n in new), "the experiments after the mid-term had five trainings from the start")
        return G.concept("after_looking", here=f"Every results file lists the choices made after its test scores were seen, and the final report discloses the added "
                                               f"trainings; the files with {_word(three)} trainings are kept unchanged in <code>results/archive/</code>. The experiments "
                                               f"of steps 19 and 20 had {_word(five)} trainings from the start, fixed before any test score was seen; step 18 trains "
                                               f"nothing.")

    def changes():
        files = ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input")
        changed = [c for k in files for c in R[k]["five_seed_changes"].get("verdicts_and_success_rules_changed", [])]
        _need(not [c for c in changed if ".onset." in c["where"] or c["where"].endswith(".met")], "no success rule and no verdict for the onsets changed")
        A = next(c for c in R["aorc_rain_to_midnight"]["five_seed_changes"]["scores_changed_lead_1"] if c["set"] == "lstm_wqa")
        old_bound = _v(R, "hindsight_bound_three_seeds_points")
        items = [f"The AORC rain to midnight (step 16): {pct(A['onset_pod'][0])} of the onsets warned became {pct(A['onset_pod'][1])}, and its hindsight bound went "
                 f"from about {round(old_bound)} to about {round(HS['bound'])} points."]
        RV = rivers_facts(R)
        old = json.loads((ROOT / R["more_basins"]["five_seed_changes"]["compared_with"]).read_text(encoding="utf-8"))
        p3 = old["gain_rule"]["lstm63_wq_against_lstm_wq"][f"l{FIRST}"]["onset"]["paired"]["mcnemar_p"]
        _need(RV["onset_reading"] == "clear loss" and RV["mcnemar"] < 0.05 <= p3 and "move of the operating point" in old["outcome_caveat"]
              and RV["chosen"][1] > RV["chosen"][0], "the 63-river onsets read as a moved operating point with three trainings and as a clear loss with five")
        items.append(f"More rivers (step 16): with {RV['n_train']['lstm63_w']} training rivers the network warns of fewer onsets at the usual level, "
                     f"{pct(RV['onset']['lstm63_wq'][FIRST])} against {pct(RV['onset']['lstm_wq'][FIRST])}. With {_word(three)} trainings that was read as a moved "
                     f"operating point; with {_word(five)}, the onset-by-onset count separates the two networks (p = {_t(R, 'e3_mcnemar_p')}), so it reads as a clear "
                     f"loss. The level chosen on the validation years reverses it.")
        nse = next(c for c in R["forecast_input"]["five_seed_changes"]["readings_changed"] if c["where"] == f"gain_rule.wqaF_sub_against_wqa.l{FIRST}.nse.reading")
        _need(nse["three_seeds"] in READING and nse["five_seeds"] in READING, "the readings of the forecast's 1-day NSE are words of the rule of step 9")
        items.append(f"The weather forecast (step 16): by the rule of step 9 its change of the 1-day NSE went from &lsquo;{nse['three_seeds']}&rsquo; to "
                     f"&lsquo;{nse['five_seeds']}&rsquo;.")
        S = R["flow_state"]["e4e_success"]
        _need(S["met"] is False and S["met_with_first_three_seeds"] is True, "the E4e rule holds with three trainings and not with five")
        items.append(f"The river at the issue time (step 20) had {_word(five)} trainings from the start: its rule would be met with the first {_word(three)} and is not "
                     f"with all {_word(five)}.")
        return ('<p class="intro"><b>What changed.</b> No success rule and no verdict for the onsets changed. But some readings moved:</p><ul class="short">'
                + "".join(f"<li>{x}</li>" for x in items) + "</ul>")

    def kept():
        n04 = len(R["lstm_all"]["sets"]["wq"]["seeds"])
        n_sat, n_cmp = len(R["satellite_inputs"]["seeds"]), len(R["model_comparison"]["seeds"])
        _need(n04 == three and n_sat == n_cmp, "the stored sets keep three trainings and the satellite and Transformer comparisons the same number")
        _need(_v(R, "review_five_seed_mismatches") == 0, "the reviewers found no five-training number that differed")
        return (f"<p>The networks of steps 8 to 11 keep their {_word(n04)} trainings: they were never saved, and new trainings would have replaced the forecasts that "
                f"had been reviewed. The satellite-input and Transformer comparisons of steps 12 and 13 keep {_word(n_sat)}. Independent AI reviewer agents "
                f"recomputed every number of the {_word(five)} trainings from the stored forecasts, in a new round of all {_word(_v(R, 'review_records'))} experiment "
                f"review records: {_t(R, 'review_five_seed_values_compared')} values, {_t(R, 'review_five_seed_mismatches')} different.</p>")

    remember = ("More trainings make a result firmer, and they can move it. A choice made after seeing the test scores must be written down and disclosed, as it is "
                "here.")
    return f"""
{G.keep(G.step(21, f"The plan promised {_word(five)} trainings where the NSE decides. They were added at the very end, after the test scores were known, and the report says so."), _try("step 21, the opening", opening))}
{_try("step 21, the box 'a choice made after seeing the test scores'", box) or G.concept("after_looking")}
{_try("step 21, what changed", changes)}
{G.keep(_try("step 21, what kept its trainings", kept), G.remember(remember),
         G.full_detail("Sections 5.3 and 8.2, and &lsquo;Choices made after looking at test scores&rsquo; in Appendix C"))}"""


# ---- step 22: the basins as GIS layers --------------------------------------------------------------------------------------------
def _step22(R):
    GL, RV = R["gis_layers"], R["review_gis_layers"]
    n = len(GL["basins"])
    over = next(k for k in GL["check_table"][0] if re.fullmatch(r"elev_differs_from_camels_by_more_than_\d+_m", k))
    metres = re.search(r"(\d+)_m$", over).group(1)

    def opening():
        kinds = " ".join(f["kind"] for f in GL["files"])
        _need(all(k in kinds for k in ("GeoPackage", "GeoJSON", "GeoTIFF")) and "centre lies inside the outline" in GL["parameters"]["clipping"],
              "the layers are GeoPackage, GeoJSON and GeoTIFF files clipped by the cell-centre rule")
        return (f"<p>Step 12 laid each basin outline over two rasters: the height of the ground and the land cover. For the hand-in, the {n} basins were exported as "
                f"GIS layers, {_t(R, 'gis_layers_files')} files in <code>results/gis_layers/</code>: the outlines and the river gauges as vector layers (GeoPackage and "
                f"GeoJSON), and for each basin its heights and its land cover, clipped to the outline, as raster files (GeoTIFF). A cell belongs to a basin when its "
                f"centre lies inside the outline, the rule of step 12.</p>")

    def checks():
        Sm = GL["summary"]
        big = [r for r in GL["check_table"] if r[over]]
        mountain = all(str(GL["basins"][r["gauge_id"]]["landscape"]).lower().startswith("mountain") for r in big)
        _need(Sm["basins_agreeing_with_basin_gis"] == n and big and all(r["elev_vs_camels_m"] > 0 for r in big) and Sm["forest_vs_camels_points"]["median"] < 0,
              "the layers agree with step 12 in every basin, the heights that differ by much are all higher than the table, and the forest share is lower")
        return (f"<p>Each layer was checked twice. Against the numbers of step 12 it agrees in all {n} basins. Against the table of the data set (CAMELS) the areas agree "
                f"closely: the typical difference is {_t(R, 'gis_area_vs_gages2_median_pct')}, the largest {_t(R, 'gis_area_vs_gages2_largest_pct')}. The mean heights "
                f"agree within a few metres in most basins (typically {_t(R, 'gis_elev_median_diff_m')}), but in {_word(len(big))} {'mountain ' if mountain else ''}"
                f"basins ours are higher by more than {metres} m ({_t(R, 'gis_elev_over_25m_range')}), and the documentation of the data set does not say why. The "
                f"forest share is lower in our maps (typically {signed(Sm['forest_vs_camels_points']['median'], 0)} points). The picture shows the comparison.</p>")

    def review():
        r1, r2 = RV, RV["round_2"]
        status = {i["item"][:30]: i["status"] for i in r2["round_1_items_status"]}
        must = [k for k in status if k.startswith("round 1 must-fix")]
        _need(r1["counts"]["must_fix"] == len(must) and all(status[k] == "closed" for k in must) and r2["counts"]["numerical_mismatches"] == 0,
              "the must-fix items of the first round are closed and the second round found no number that differed")
        c2 = r2["counts"]
        return (f"<p>An independent AI reviewer agent recomputed the layers in {_word(_v(R, 'gis_layers_review_rounds'))} rounds. The first found that the gauge points "
                f"had been rounded, so their positions and heights were slightly off, and that a licence mark was lost in the description of the files. Both were "
                f"fixed. The second round found no number that differed ({count(c2['values_compared'])} values, and "
                f"{count(c2['raster_cells_compared_with_their_sources_not_in_values_compared'])} raster cells equal to their sources): "
                f"&lsquo;{_t(R, 'gis_layers_review_verdict')}&rsquo;. The files claim no licence of their own; the terms of their sources are in Table 3 of the final "
                f"report.</p>")

    def picture():
        cap = (f"<b>The basin statistics against the table of the data set, one dot per basin.</b> <b>How to read it:</b> Each panel compares one property; a dot on "
               f"the grey line means that both agree, and the colour is the landscape. Area agrees closely in every basin. In the height panel the blue dots well "
               f"above the line are the mountain basins whose heights in the table are lower. Slope depends on the size of the grid cells, as step 12 said: dots and "
               f"rings are two cell sizes. The forest share is lower in our maps in most basins. The picture was drawn in step 12; the layers give back its numbers "
               f"for all {n} basins.")
        return _picture("basin_gis_check.png", cap)

    remember = "A map is data. Export it in a standard format, check it against an independent table, and say where the two disagree."
    return f"""
{G.keep(G.step(22, "The maps of step 12 became files that any GIS program can open."), _try("step 22, the opening", opening))}
{_try("step 22, the picture", picture)}
{_try("step 22, the checks", checks)}
{G.keep(_try("step 22, the review", review), G.remember(remember),
         G.full_detail("Table 1 in Section 1, objective O4 in Section 3.1, Table 7 in Section 7.4, and Appendix B"))}"""


# ---- step 23: where the project ended ---------------------------------------------------------------------------------------------
def _step23(R, AO, S4, FC, hind):
    checks = team.checks_state()
    three, five = _seed_counts(R)

    def answer():
        _need(set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) == {"NOT YET"}, "every set a service could have is NOT YET for onsets on the main test")
        sat = {k[4:-len("_l1_plain_test_verdict_onset")]: e["value"] for k, e in _reg(R).items() if k.startswith("e4e_") and k.endswith("_l1_plain_test_verdict_onset")}
        _need(sat and "GO" not in sat.values() and sat.get("wqawhf_s") == "CAUTION", "no set of the shorter test is GO and the full model is CAUTION")
        return (f"<p>The finding of step 15 stands: <b>{FINDING}.</b> The table gathers the ladder of steps 16 and 20 in its final form, {LEAD_WORD[FIRST]}. GO, at "
                f"least {_t(R, 'go_min_warned')} of the onsets warned with at most {_t(R, 'go_max_false')} of the warnings false, was met by no set of inputs that a "
                f"service could have. On the main test every such set is {tag('NOT YET')} for onsets, at every lead time. On the shorter test the full model reaches "
                f"{tag('CAUTION')}, with inputs that a service does not have in this form.</p>")

    def ladder():
        main = [("Past weather and the river gauge (the rain table)", "lstm_wq_l1_onset_pod", "lstm_wq_l1_onset_far"),
                ("+ rain up to the evening (step 11)", "lstm_wqn_l1_onset_pod", "lstm_wqn_l1_onset_far"),
                (f"+ rain up to midnight, AORC, {AORC_WORDING} (step 16)", "lstm_wqa_l1_onset_pod", "lstm_wqa_l1_onset_far"),
                (f"&nbsp;&nbsp;&nbsp;Stage IV in its place, {REALTIME_WORDING}", "s4_stage4_share", None),
                (f"+ perfect weather forecast, {UPPER}", "lstm_wqf_l1_onset_pod", "lstm_wqf_l1_onset_far")]
        sat = [("AORC rain to midnight and the river gauge", "e4e_wqa_s_l1_onset_pod", "e4e_wqa_s_l1_onset_far"),
               ("+ the river at midnight (step 20)", "e4e_wqah_s_l1_onset_pod", "e4e_wqah_s_l1_onset_far"),
               ("+ the hour-by-hour rain (step 19)", "e4e_wqawh_s_l1_onset_pod", "e4e_wqawh_s_l1_onset_far"),
               ("+ the weather forecast: the full model", "e4e_wqawhf_s_l1_onset_pod", "e4e_wqawhf_s_l1_onset_far"),
               (f"The perfect rain of the next days, {UPPER}", "e4e_wqap_s_l1_onset_pod", "e4e_wqap_s_l1_onset_far")]
        row = lambda label, w, f: [label, _t(R, w) + (f" ({hind})" if w == "lstm_wqa_l1_onset_pod" and hind else ""), _t(R, f) if f else "&ndash;"]
        test_main = R["lstm_all"]["split"]["test"]
        rows = ([[f'<td class="grp" colspan="3">Main test, {_years(test_main)}, {_t(R, "onsets_l1")} onsets</td>']] + [row(*x) for x in main]
                + [[f'<td class="grp" colspan="3">Shorter test of step 20, {_years(R["flow_state"]["split"]["test"])}, {_t(R, "onsets_sat_l1")} onsets, '
                    f"comparable only inside it</td>"]] + [row(*x) for x in sat])
        body = "".join("<tr>" + (r[0] if len(r) == 1 else "".join(f"<td>{c}</td>" for c in r)) + "</tr>" for r in rows)
        return (f'<table class="ladder"><colgroup><col style="width:62%"><col style="width:23%"><col style="width:15%"></colgroup>'
                f"<tr><th>Set of inputs, {LEAD_WORD[FIRST]}</th><th>Onsets warned</th><th>Warnings false</th></tr>{body}</table>")

    def service():
        _need(not any(" l1 " in f" {x} " for x in R["warning_policy"]["go_met_at"]), "no warning rule reaches GO 1 day ahead, not even with perfect inputs")
        return (f"<p><b>What a service would need.</b> The gauge&rsquo;s hourly record and radar rain up to the moment of issue, and a weather forecast, as operational "
                f"flood systems use {cite('nearing2024')}. The Stage IV check of step 16 kept {pct(S4['share'])} of the effect of the AORC rain, {REALTIME_WORDING}. "
                f"But with daily inputs GO is out of reach {LEAD_WORD[FIRST]} by any warning rule: the perfect rain of the next days warns of "
                f"{_t(R, 'lstm_wqap_l1_onset_pod')} with {_t(R, 'lstm_wqap_l1_onset_far')} false, the perfect weather forecast of {_t(R, 'lstm_wqf_l1_onset_pod')} with "
                f"{_t(R, 'lstm_wqf_l1_onset_far')} false, both {UPPER}.</p>")

    def handed_in():
        readme = (HANDIN / "README_HANDIN.md").read_text(encoding="utf-8")
        audits = re.search(r"(\w+) audits \(compliance, quality, references and originality\)", readme)
        _need(audits and report_says("every entry with a DOI was checked against its Crossref or DataCite record"),
              "the final report was audited for compliance, quality, and references and originality, and its references were checked against their records")
        from report_common import pdf_pages
        pages = pdf_pages(REPORT_PDF)
        _need(pages, f"{REPORT_PDF.name} can be read")
        return (f"<p><b>What was handed in.</b> The hand-in is due {_t(R, 'submission_deadline')}, and the talk is on {_t(R, 'final_presentation')}. Besides the "
                f"proposal and the mid-term deck it holds the final report on the professor&rsquo;s template ({pages} pages), audited {audits.group(1)} times by AI "
                f"agents: for compliance with his instructions, for academic quality, and for its references and originality; each of its {report_references()} "
                f"references was checked against its online record. It also holds the final deck with its speaker notes, and a page of datasets and links.</p>")

    def deck():
        rows = _slides()
        secs = {}
        for _, _, who, s in rows:
            secs[who] = secs.get(who, 0) + s
        same = len(set(secs.values())) == 1
        total = sum(s for *_, s in rows)
        body = [[str(k), _esc(title), _esc(who), f"{s} s"] for k, title, who, s in rows]
        return (f'<p class="intro">The final deck has {_word(len(rows))} slides for {round(total / 60)} minutes'
                + (f", {next(iter(secs.values()))} seconds for each of the {_word(len(secs))} speakers" if same else "") + ":</p>"
                + table(["Slide", "Title", "Speaker", "Time"], body, cls="slides", widths=["8%", "56%", "24%", "12%"]))

    def members():
        _need(checks["n_done"] < checks["n"], "some of the team's own checks are still open")
        return (f"<p><b>The team&rsquo;s own checks.</b> The hand-in also holds a check sheet for the members, <code>team_checks/TEAM_CHECK_SHEET.pdf</code>, with its "
                f"script <code>team_checks/team_check.py</code>: one command per member, which recomputes that member&rsquo;s headline numbers from the stored "
                f"forecasts with numpy and pandas only, and prints AGREE or DIFFER beside each. A member&rsquo;s own check counts only when that member records it in "
                f"<code>team.json</code>. {checks['n_done']} of {checks['n']} are recorded so far.</p>")

    def who():
        _need(report_says("no team member has yet read those papers"), "no team member has yet read the cited papers")
        return (f"<p><b>Who did what.</b> Claude Code wrote all the code, ran all the experiments and drafted the documents, under the direction of the group leader, "
                f"{_t(R, 'group_leader')}. Independent AI reviewer agents checked the shared engine and every experiment ({_t(R, 'review_records')} review records); "
                f"for the numbers of the {_word(five)} trainings they recomputed {_t(R, 'review_five_seed_values_compared')} values, with "
                f"{_t(R, 'review_five_seed_mismatches')} different. AI agents read the cited papers at their sources; no team member has read them yet. The other "
                f"members built the package of step 17. {_cap(_esc(team.checks_sentence()))}.</p>")

    def still_open():
        items = []
        if "[" in _t(R, "repo_url") or "[" in _t(R, "data_bundle_url"):
            items.append("the links to the code repository and to the data, still placeholders in the report;")
        items.append("the date on the cover, which is to be the day of sending;")
        if checks["n_done"] < checks["n"]:
            items.append(f"the members&rsquo; own checks ({checks['n_done']} of {checks['n']} recorded);")
        if report_says("Signed with the following exception"):
            items.append(f"the {_word(checks['n'])} signatures of Appendix E, which has a sentence for signing before the checks are done;")
        if report_says("no statement of how AI was used in it has been supplied yet"):
            items.append("a statement of how AI was used, from the authors of the package of step 17;")
        if report_says("no team member has yet read those papers"):
            items.append("the members&rsquo; own reading of the cited papers;")
        if report_says("a licence for the project's own code and results is still to be chosen"):
            items.append("a licence for the code and the results, still to be chosen by the group;")
        if report_says("rehearsal: N"):
            items.append(f"and before the talk on {_t(R, 'final_presentation')}, the rehearsal.")
        _need(len(items) > 2, "items are still open before sending")
        return ('<p class="intro"><b>Still open before sending:</b></p><ul class="short">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
                f"<p>Experiments not done: a probability of high flow beside the forecast, all members of the forecast ensemble, the LSTM on the inputs of the "
                f"package of step 17, the river&rsquo;s state on the main test, {_word(five)} trainings for the networks of steps 8 to 11, a Stage IV measurement of "
                f"the hindsight of the hour-by-hour rain, provisional gauge values with measured delays, and a network trained on Stage IV.</p>")

    remember = (f"{FINDING}: what a forecast can know at the moment it is issued. GO was not met, and the report says so. A result is finished only when the "
                f"people who sign it have checked it.")
    return f"""
{G.keep(G.step(23, "The final answer of the project, what a service would need, what was handed in, and what is still open before it is sent."), _try("step 23, the answer", answer))}
{_try("step 23, the table of the final ladder", ladder)}
{_try("step 23, what a service would need", service)}
{_try("step 23, what was handed in", handed_in)}
{G.keep(_try("step 23, the deck", deck))}
{_try("step 23, the team's own checks", members)}
{_try("step 23, who did what", who)}
{G.keep(_try("step 23, still open", still_open), G.remember(remember),
         G.full_detail("Section 7.4 (Table 7), Sections 9.1 to 9.3 and 10, and Appendix E"))}"""


def html(R, T):
    """Steps 17 to 23 as HTML; '' (and one line on stderr) while the files they stand on are missing. A step whose own file is missing is left out alone."""
    if not available(R):
        print("g6_final: the registry of the final report or the results files of the work after the mid-term are missing: steps 17 to 23 are left out", file=sys.stderr)
        return ""
    HS = hindsight_facts(R)
    hind = hindsight_clause(HS)
    AO, S4, FC = aorc_facts(R), stage4_facts(R), forecast_facts(R)
    parts = [_try("step 17", lambda: _step17(R)), _try("step 18", lambda: _step18(R, hind)), _try("step 19", lambda: _step19(R, hind)),
             _try("step 20", lambda: _step20(R)), _try("step 21", lambda: _step21(R, HS)), _try("step 22", lambda: _step22(R)),
             _try("step 23", lambda: _step23(R, AO, S4, FC, hind))]
    return f'{STYLE}<div class="g6">' + "\n".join(parts) + "</div>"
