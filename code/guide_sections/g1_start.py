"""Guide, part 1 - steps 1 to 5: the question, the data, the fair test, the score, the simple models.

Written for a reader who has never trained a model: short sentences, plain words, each idea of an AI course in a concept box
(guide_common.WORDS) at the place where the project first needs it.

Nothing is computed here beyond counts, differences, rounding and formatting. Every number is read by key from results/*.json
and results/basin_audit.csv; the lead times and the names of the daily weather values come from code/flood.py, as in section 1
of the report. A sentence whose file or key is missing is left out, and one line on stderr says which (say()). A picture that
is not in figures/ yet is left out in the same way (picture()).

Reads: data_audit.json, basin_audit.csv, floods_2004.json, baselines.json, fair_test.json, engine_review.json,
flood_analysis.json (the verdict criteria, through s0_summary.criteria), lstm_all.json (the list of basin properties) and
guide_figures.json (the two real examples that the pictures of steps 1 and 4 show, and the rules that picked them).
"""
import datetime
import re
import sys

import team
from common import FIG
from flood import FORCING, LEADS
from guide_common import concept, final_detail_for, keep, lettering_width, remember, small_figure, step, try_it
from report_common import LEAD_WORD, cite, count, fig, img, n2, pct, tag
from report_sections.s0_summary import _word, criteria
from report_sections.s1_data import FORCING_WORD, _split_bar

QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
STATES = {"NC": "North Carolina", "VA": "Virginia", "GA": "Georgia", "FL": "Florida", "MS": "Mississippi"}
KIND = {"linreg": "ridge regression", "rf": "the random forest"}       # the two simple models that are fitted to the training years
READ = "<b>How to read it:</b>"
# Layout only (no colour, no size of type), for this part alone. A large picture is a full-width float: when it does not fit on the rest of
# a page it moves to the top of the next page and the text after it fills the page instead of leaving it empty (as in the report,
# s0_summary.LAYOUT). A step's heading stays with its first paragraph and starts below the pictures of the step before; 'What to
# remember' and the line 'Full detail' end a step below its pictures, as one block; the boxes are laid out as blocks of their own, so
# that none slides under a picture. The two overview pictures were drawn narrower than the text: they are shown at the width at which
# their lettering matches the other pictures, with the caption beside them (guide_common.small_figure).
LAYOUT = ("<style>.g1 .ff{float:left;width:100%;break-inside:avoid} .g1 .keep{break-inside:avoid;clear:both} .g1 .who,.g1 .remember{clear:both} "
          ".g1 .concept,.g1 .remember,.g1 .try{display:flow-root} .g1 .v3{margin:0 0 3pt 10pt}</style>")
_SAID = set()


def _note(message):
    """One line on stderr, once per build."""
    if message not in _SAID:
        _SAID.add(message)
        print(f"g1_start: {message}", file=sys.stderr)


def say(what, make):
    """The text that make() builds from the results files. If a file or a key it reads is missing, the text is left out ('') and one
    line on stderr says so."""
    try:
        return make()
    except (KeyError, IndexError, TypeError, AttributeError, ValueError, ZeroDivisionError) as e:
        _note(f"{what}: left out, a results file or key is missing or not as expected ({type(e).__name__}: {e})")
        return ""


def picture(name, caption, in_text=False):
    """A picture of figures/ at the full text width, or nothing (and one line on stderr) while the file does not exist. A picture that
    was drawn narrower than the text is shown at the width at which its lettering matches the other pictures, its caption beside it.
    in_text: the picture is not a float: it stays between the two paragraphs it stands between (for a small picture)."""
    if not (FIG / name).exists():
        _note(f"figures/{name} does not exist yet: the picture is left out")
        return ""
    if not caption:
        return ""
    width = lettering_width(name)
    figure = small_figure(img(name), caption, width, beside=True) if width < 100 else fig(name, caption)
    return figure if in_text else f'<div class="ff">{figure}</div>'


def para(*sentences):
    """One paragraph from the sentences that exist."""
    text = " ".join(s for s in sentences if s)
    return f"<p>{text}</p>" if text else ""


def opening(n, lead, *sentences):
    """The heading of step n, its lead and its first paragraph, kept on one page."""
    return keep(step(n, lead), para(*sentences))


def closing(text, sections):
    """The end of a step: 'What to remember' and the pointer to the report, kept together below the pictures of the step."""
    return keep(remember(text), final_detail_for(sections))


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _date(iso):
    d = datetime.date.fromisoformat(str(iso)[:10])
    return f"{d.day} {d:%B %Y}"


def _n_years(span):
    a, b = (datetime.date.fromisoformat(str(x)[:10]) for x in span)
    return round(((b - a).days + 1) / 365.25)


def _years(span):
    """('1980-10-01', '2000-09-30') -> '1980&ndash;2000'."""
    return f"{str(span[0])[:4]}&ndash;{str(span[1])[:4]}"


def _zero(v):
    """A score to two decimals; a value that rounds to zero is printed without a sign."""
    return n2(0.0 if round(v, 2) == 0 else v)


FIRST = LEAD_WORD[LEADS[0]]                                   # '1 day ahead'
AHEAD = f"{_and(LEADS)} days ahead"                           # '1, 2 and 3 days ahead'
N_LEADS = _word(len(LEADS))                                   # 'three'


# ---- step 1: the question ---------------------------------------------------------------------------------------------------
def _step1(R, T):
    def flood_days():
        F4 = R["floods_2004"]
        days = sorted(F4["largest_flow_days"][:2], key=lambda d: d["date"])
        return F4, days, [datetime.date.fromisoformat(d["date"]) for d in days]

    def rise():
        F4, days, (a, b) = flood_days()
        top = max(d["flow_mm_day"] for d in days)
        return (f"On {a.day} and {b.day} {b:%B} heavy rain sent the river far above its high-flow level. That is a flow which the river passes "
                f"on only a few days in a hundred. At the higher peak the flow was {n2(top, 0)} millimetres per day, about "
                f"{n2(top / F4['high_flow_level_mm_day'], 0)} times that level.")

    def storms():
        F4, days, _ = flood_days()
        near = {c["date"]: c for c in F4["cyclones_within_500_km"]}
        if not all(d["date"] in near for d in days):
            return ""
        hit = [near[d["date"]] for d in days]
        remains = any("remnant of the hurricane" in x for x in F4.get("notes", [])) and {c["status_then"] for c in hit} <= {"TD", "EX", "LO"}
        what = "the remains of hurricanes" if remains else "the tropical storms"
        return (f"On these two days {what} {_and(c['name'] for c in hit)} passed {_and(count(c['closest_km']) for c in hit)} kilometres "
                f"from the river {cite('landsea2013')}.")

    def hydro_caption():
        F4, _, (a, b) = flood_days()
        river, state = F4["basin"].split(", ")
        return (f"<b>Rain in, river flow out: the {river} River, {STATES.get(state, state)}, {b:%B %Y}.</b> {READ} The grey bars that hang from "
                f"the top are the rain that the data set gives for each day. The blue line is the flow of the river, and the orange line is the "
                f"river&rsquo;s high-flow level. Look at {a.day} and {b.day} {b:%B}: the longest bars and the two peaks fall on the same days, and "
                f"each peak needs weeks to come down.")

    def example():
        """What the lower half of the path picture shows: the one real forecast that 14_guide_figures.py picked by a rule, and how
        favourable it is (results/guide_figures.json)."""
        E, n_rivers = R["guide_figures"]["example_of_the_pipeline"], R["guide_figures"]["facts"]["n_basins"]
        n_warn, n_fc = sum(bool(w) for w in E["warning"]), len(E["forecast_mm_day"])
        if n_fc != len(LEADS) or E["rule_fallback"] or "most rain" not in E["rule"] or not 0 < E["n_onsets_warned"] < E["n_onsets"] / 2:
            raise ValueError("the example of the path picture is not the warned rise with the most rain, or most rises are warned of")
        weeks = round(len(E["days"]) / 7)
        above = (f"{_word(n_warn).capitalize()} of them lie above the dashed high-flow level, so {_word(n_warn)} warnings are given."
                 if 1 < n_warn < n_fc else "A warning is given for each forecast above the dashed high-flow level.")
        return (f" Below the boxes is one real forecast: on the left the rain and the river gauge of the last {_word(weeks)} weeks, on the "
                f"right the {_word(n_fc)} forecasts. {above} The example was picked by a rule, the warned rise with the most rain, and it is a "
                f"favourable case. In the test years the {_word(n_rivers)} rivers rose above their high-flow levels {count(E['n_onsets'])} "
                f"times, and only {count(E['n_onsets_warned'])} of these rises were warned of a day ahead (step 10).")

    pipeline_caption = (f"<b>From inputs to a warning: the path of every experiment.</b> {READ} Read the row of boxes from left to right. The "
                        f"inputs go into a model, and the model gives the flow of the river for the next {N_LEADS} days. A warning is given "
                        f"where a forecast is above the river&rsquo;s high-flow level."
                        + (say("step 1, the example in the path picture", example)
                           or " Below the boxes is one real forecast, picked by a rule: it shows the path, not how often a warning comes in "
                              "time (step 10 counts that)."))

    return f"""{opening(1, "What the project wants to find out, and why a river is not easy to forecast.",
        "A flood warning is only worth something if it comes before the river rises. A day or more of notice is what lets people act. So the "
        "project is about forecasting: saying today what a river will do in the next days.")}
{para("The picture shows one of the rivers of the project in September 2004. Rain is drawn at the top and "
      "the flow of the river below, both in millimetres per day (step 2 explains the unit).", say("step 1, the two flood days", rise),
      say("step 1, the storms of September 2004", storms), "The river answered the rain within a day. Then it needed weeks to come back down.")}
{picture("overview_hydrograph.png", say("step 1, the caption of the rain and flow picture", hydro_caption))}
<p>The project asks two things. Can a model forecast the flow of a river {AHEAD} better than simple methods can? And could its warnings of
high flow be trusted?</p>
{concept("model", here="a model turns the weather of the past days and months into the flow of a river on the next days. The project compares "
                       "several kinds of model: simple ones first (step 5), then larger ones (steps 6 and 13).")}
{concept("inputs_target", here=f"the inputs are the weather of the past days and some fixed facts about the river. One version of each model "
                               f"is also given the flow measured in the river so far. The target is the flow on each of the next {N_LEADS} days.")}
{concept("lead_time", here=f"a forecast is made at the end of a day, for each of the next {N_LEADS} days. So every model is scored at {N_LEADS} "
                           f"lead times: {_and(LEADS)} days.")}
<p>The picture of the flood shows why the first question is hard. A model must react at once to heavy rain. And a model must remember the
rain of earlier weeks, because the ground stores that water and passes it on slowly.</p>
<p>The second picture shows the path that every experiment of the project follows. Each later step of this guide looks at one part of that
path: the data, the test, the score, the models, and what was found.</p>
{picture("guide_pipeline.png", pipeline_caption)}
{closing(f"The project forecasts the flow of rivers {AHEAD}. It asks whether a model that learns from data does this better than simple "
         f"methods, and whether its warnings could be trusted. A warning counts only if it comes before the river rises.",
         "page 1 and the introduction in front of section 1")}"""


# ---- step 2: the data -------------------------------------------------------------------------------------------------------
def _step2(R, T):
    def source():
        A = R["data_audit"]
        return (f"The project takes them from CAMELS-US, a public data set of {count(A['n_all_camels'])} river basins in the United States "
                f"{cite('newman2015')}.")

    def basin_box():
        sel = R["data_audit"]["selected"]
        area = [b["area_gages2"] for b in sel]
        return (f"each of the {_word(len(sel))} rivers has its basin; the smallest is about {count(round(min(area), -1))} square kilometres, the "
                f"largest about {count(round(max(area), -2))}. The point is a river gauge, an instrument that measures the flow.")

    def rule():
        A = R["data_audit"]
        sel = A["selected"]
        per = {}
        for b in sel:
            per[b["landscape"]] = per.get(b["landscape"], 0) + 1
        if len(set(per.values())) != 1:
            raise ValueError("the landscapes do not hold the same number of basins")
        return (f"The {_word(len(sel))} basins were chosen by a rule, not by hand. The rule starts from the {count(A['n_candidates'])} basins of two "
                f"south-eastern river regions that are humid and fed by rain. It keeps the {count(A['n_complete'])} whose record has no missing day. "
                f"It sorts these into {_word(len(per))} landscapes and takes {_word(next(iter(per.values())))} from each, evenly spaced from the "
                f"smallest to the largest. No model score enters the rule, so easy rivers could not be picked. Two pictures show the "
                f"{_word(len(sel))}: a map of where they lie, and each basin drawn inside its outline.")

    def checked():
        A, C = R["data_audit"], T["basin_audit"]
        gap = int(C.longest_gap_days.max())
        where = "their flow record" if int(C.forcing_missing.sum()) == 0 else "their record"
        return (f"The other {count(A['n_with_gaps'])} candidates have gaps in {where}; in the worst one, {count(gap)} days in a row are "
                f"missing, about {_word(round(gap / 365.25))} years. A model cannot learn from days without a measurement. That is why the "
                f"records were checked before any model was built.")

    def given():
        A, C = R["data_audit"], T["basin_audit"]
        n_static = len(R["lstm_all"]["static_inputs"])
        return (f"The record runs for {_n_years(A['period'])} years, {A['period'][0][:4]} to {A['period'][1][:4]}: "
                f"{count(int(C.n_days.max()))} days for each river. For every day a model is given {_word(len(FORCING))} weather values, averaged over "
                f"the basin: {_and(FORCING_WORD[c] for c in FORCING)}. A model that serves all rivers is also given {_word(n_static)} fixed "
                f"properties of each basin, such as area, slope and forest cover. One version of each model also reads the river "
                f"gauge: the flow measured up to the day of the forecast.")

    def map_caption():
        sel = R["data_audit"]["selected"]
        return (f"<b>Where the {_word(len(sel))} basins are.</b> {READ} Each dot is the river gauge of one basin, on a map of the south-eastern "
                f"United States. The colour gives the landscape: blue for the mountains, green for the northern lowlands, orange for the southern "
                f"lowlands. The number in brackets says how many basins each landscape has.")

    def maps_caption():
        sel = R["data_audit"]["selected"]
        return (f"<b>The {_word(len(sel))} basins, each drawn inside its outline (step 12 tells how the maps were made).</b> {READ} Each shape is one "
                f"basin, and the small circle is its river gauge: the point that all the rain inside the shape runs to. Dark blue is the high "
                f"ground of a basin and light blue its low ground. Above each map stand its name with the state (NC is North Carolina), its area "
                f"and its lowest and highest point.")

    return f"""{opening(2, "Where the examples come from, how the rivers were chosen, and what a model is given.",
        "A model that learns from examples needs many of them.", say("step 2, the data set", source),
        "For every basin and every day it holds the weather over the basin and the flow measured in the river.")}
{concept("basin", here=say("step 2, the basin box", basin_box))}
{picture("basin_maps.png", say("step 2, the caption of the basin maps", maps_caption))}
{para(say("step 2, the rule that chose the basins", rule))}
{picture("overview_map.png", say("step 2, the caption of the map", map_caption), in_text=True)}
{para(say("step 2, why the data were checked first", checked))}
{para(say("step 2, what a model is given", given))}
<p>What a model must produce is the flow of each of the next {N_LEADS} days. Flow is given in millimetres per day (mm/day), like rain. It
is the water that passes the gauge in a day, spread as a layer over the whole basin. So rain and flow can be compared, and large
and small rivers share one scale.</p>
{closing(f"The data are public, and they were checked before any model was built. The rivers were chosen by a rule, so easy rivers could not "
         f"be picked. A model is given daily weather, in one version also the river gauge, and must produce the flow of the next {N_LEADS} days.",
         "section 1")}"""


# ---- step 3: a fair test ----------------------------------------------------------------------------------------------------
def _step3(R, T):
    def cut():
        return f"So the {_n_years(R['data_audit']['period'])} years are cut into three periods, and each period has one job."

    def split_box():
        S = R["baselines"]["split"]
        tr, va, te = (f"{_years(S[p])} ({_n_years(S[p])} years)" for p in ("train", "val", "test"))
        return (f"the training years are {tr}, the validation years {va} and the test years {te}. Unless the text says otherwise, every score in "
                f"this guide is from the test years.")

    def bar():
        return (f"<figure>{_split_bar()}<figcaption><b>The fair test: three periods, one job each.</b> {READ} The bar is the "
                f"{_n_years(R['data_audit']['period'])} years from left to right, drawn to scale; each shade is one period. Under each period stands "
                f"its one job. &lsquo;Fit&rsquo; covers everything that is worked out from data: the adjustable numbers inside a model, each "
                f"river&rsquo;s high-flow level and the scale of the inputs.</figcaption></figure>")

    def leak_box():
        FT = R["fair_test"]
        if FT["passed"] is not True:
            raise ValueError("the fairness checks did not pass")
        return (f"a leak would be a model that reads weather or flow from after the evening of the forecast, or anything from the test years. "
                f"A test on made-up rivers looked for one: all {len(FT['checks'])} of its checks passed.")

    def made_up():
        FT = R["fair_test"]
        exp, mea = FT["expected"]["past_weather_only"], FT["measured"]["past_weather_only"]
        if len(exp) != len(LEADS) or len(mea) != len(LEADS) or len(set(exp[1:])) != 1:
            raise ValueError("the expected scores are not one value for the first lead time and one for the later ones")
        w = re.search(r"([\d.]+) \* rain\(d-1\) \+ ([\d.]+) \* rain\(d\)", FT["rule"])       # the weights of the made-up rule, as the test stores them
        if not w or float(w.group(1)) <= float(w.group(2)):
            raise ValueError("the made-up rule is not 'more weight on the rain of the day before than on the rain of the same day'")
        twice = abs(float(w.group(1)) / float(w.group(2)) - 2) < 1e-9
        rule = ("The rule gives the rain of the day before twice the weight of the rain of the same day." if twice else
                "The rule gives the rain of the day before more weight than the rain of the same day.")
        return (f"{rule} A forecast for tomorrow knows the first, today&rsquo;s rain, and not the second. Worked out from the rule, an honest "
                f"model can then reach {n2(exp[0])} at most, on a scale where 1 is a perfect forecast (step 4). For {_and(LEADS[1:])} days ahead "
                f"the rain that matters has not yet fallen: at most {_zero(exp[1])}. A higher score would prove a leak. The scores were "
                f"{_and(_zero(v) for v in mea)}.")

    def review():
        ER = R["engine_review"]
        if not any("No reviewer found a leak" in x for x in ER["notes"]):
            raise ValueError("the review does not record that no leak was found")
        own = (" The team members&rsquo; own check of this part is still to be done." if team.checks_state()["n_done"] == 0 else "")
        return f"{_word(len(ER['reviews'])).capitalize()} AI reviewer agents also read the shared code, each on its own, and found no leak.{own}"

    def catch():
        n = len(R["data_audit"]["selected"])
        return (f"One limit remains. All {_word(n)} rivers are in all three periods. So the test shows how a model does in later years of rivers "
                f"it has already seen. A river it has never seen is a different test (step 9).")

    def no_leak():
        if R["fair_test"]["passed"] is not True:
            raise ValueError("the fairness checks did not pass")
        return " A test on made-up rivers, whose right answer is known, found no leak."

    return f"""{opening(3, "How the project makes sure that a good score means a good forecast.",
        "A model can look good for the wrong reason. Scored on the same days it learned from, a model can simply repeat what it has stored. "
        "That says nothing about a day it has not seen.", say("step 3, the three periods", cut))}
{concept("split", here=say("step 3, the box on the three periods", split_box))}
{say("step 3, the period bar", bar)}
<p>Why whole years, and not days picked at random? Neighbouring days are much alike: with random days, a model could score well on a test
day by remembering the day before and the day after. And why three parts, not two? Choosing is also a kind of learning. Whatever is
chosen on the validation years scored well there partly by luck, so its score there is a little too good. Only years that took no part
in any choice give an honest score. And they give it once: whoever changes a model after seeing its test score has begun to choose on
the test years.</p>
<p>The second rule is about time. Every forecast is made at the end of a day. It may use what is known up to that evening, and nothing
later. And a forecast is counted in the period of the day it is for, not of the day it was made.</p>
{concept("leak", here=say("step 3, the leak box", leak_box))}
{para("How can a leak be ruled out? The code of the project was tested on made-up rivers that follow a known rule. Their flow depends only on "
      "the rain of the same day and of the day before. So the best possible forecast is known in advance.",
      say("step 3, the scores on made-up rivers", made_up))}
{para(say("step 3, the review of the shared code", review))}
{para(say("step 3, the limit of the test", catch))}
{try_it("<code>python code/test_flood.py</code> runs these checks on made-up rivers. It needs no data download. It prints the scores named above, "
        "ends with <code>all checks passed</code> and writes <code>results/fair_test.json</code> again.")}
{closing("A score counts only on data that the model has never seen. The project learns on the training years, chooses on the validation "
         "years and scores on the test years." + say("step 3, what to remember of the test on made-up rivers", no_leak), "section 1")}"""


# ---- step 4: the score ------------------------------------------------------------------------------------------------------
def _step4(R, T):
    def pooled():
        """The warning table of 'tomorrow = today' for the first lead time in the test years, all basins together, with the number of
        high-flow days and of onsets in it (the same for every model)."""
        s = R["baselines"]["models"]["base_persistence"]["test"]["summary"][f"l{LEADS[0]}"]
        return s, s["all"]["hits"] + s["all"]["misses"], s["onset"]["hits"] + s["onset"]["misses"]

    def one_number():
        return (f"For {_word(_n_years(R['baselines']['split']['test']))} years of daily forecasts a single number is needed. For river flow the "
                f"usual one is the Nash-Sutcliffe efficiency, NSE for short {cite('nash1970')}.")

    def nse_box():
        n = len(R["baselines"]["high_flow"])
        return (f"one NSE is worked out for every river, lead time and model, over the test years. The guide mostly quotes the median of the "
                f"{_word(n)} rivers: the middle value, with half of the rivers above it and half below.")

    def level():
        HF = R["baselines"]["high_flow"]
        share = sum(v["share_train_days_above"] for v in HF.values()) / len(HF)
        return (f"The second way of judging looks only at high water. Each river gets its own high-flow level: the flow that was passed on "
                f"{pct(share)} of the days of the training years.")

    def high_box():
        _, n_high, n_on = pooled()
        n = len(R["baselines"]["high_flow"])
        return (f"the test years hold {count(n_high)} high-flow days in the {_word(n)} rivers together. For a forecast {FIRST}, {count(n_on)} "
                f"of them are onsets.")

    def why_onsets():
        _, n_high, n_on = pooled()
        return (f"Why look at the onsets separately? On {pct((n_high - n_on) / n_high)} of the high-flow days the river was already high the day before. "
                f"To say that a high river stays high is easy, and it helps nobody. The onset is the day a warning is for.")

    def rare():
        s, n_high, _ = pooled()
        days = sum(s["all"][c] for c in ("hits", "misses", "false_alarms", "correct_neg"))
        return (f"Why not simply count the days on which a model was right? Because high flow is rare. A model that never warns is right on "
                f"{pct(1 - n_high / days)} of the days, and it is useless.")

    def verdicts():
        c = criteria(R)
        rows = [("GO", f"At least {pct(c['go_pod'])} of the high-flow days are warned of, and at most {pct(c['go_far'])} of the warnings are false."),
                ("CAUTION", f"Not GO, but the hits are at least {pct(c['ca_csi'])} of all hits, misses and false warnings together."),
                ("NOT YET", "Neither of the two.")]
        return ("<p>When could warnings be trusted? The answer was fixed before the flood analysis was run, so that the rule could not be changed "
                "afterwards to fit the result. There are three verdicts.</p>\n" + "\n".join(f'<p class="v3">{tag(v)}&nbsp; {text}</p>' for v, text in rows))

    def stretch():
        """What the picture of the NSE shows: the stretch of days that 14_guide_figures.py picked by a rule, where its error lies and how
        favourable it is (results/guide_figures.json). Every wording that depends on the stored forecast is checked against it here."""
        S = R["guide_figures"]["stretch_of_the_score_picture"]
        obs, fc, C = S["observed_mm_day"], S["forecast_mm_day"], S["candidates_nse"]
        if "closest to the median NSE" not in S["rule"] or S["nse"]["average"] != 0 or n2(S["nse"]["forecast"]) == n2(S["nse"]["perfect"]):
            raise ValueError("the stretch of the score picture is not the one the caption describes")
        top = max(range(len(obs)), key=lambda i: obs[i])
        late = max(range(len(fc)), key=lambda i: fc[i]) == top + 1 and max(fc) < obs[top]
        err = [(f - o) ** 2 for f, o in zip(fc, obs)]
        few = sum(err[max(top - 1, 0):top + 2]) > 0.5 * sum(err)                    # the day of the peak and its two neighbours
        where = (" Look at the peak in the middle panel: the forecast rises a day late and too low." if late else
                 " Look at the peak in the middle panel, where the blue line leaves the black one.")
        most = " Errors are squared, so these few days make most of the error." if few else ""
        picked = (f" The stretch was picked by a rule and is a favourable case: of {S['n_candidates']} stretches around a rise of this river, "
                  f"the middle one scores only {n2(C['median'])}." if S["nse"]["forecast"] > C["median"] else
                  f" The stretch was picked by a rule from {S['n_candidates']} stretches around a rise of this river.")
        return (f" On the right the forecast is a flat line at the average of these {S['n_days']} days: it scores exactly 0.{where}{most}{picked}")

    nse_caption = (f"<b>The NSE in one picture.</b> {READ} The black line is the measured flow (&lsquo;observed&rsquo; in the picture), the blue "
                   f"line the forecast, and the shaded area between them the error. The closer the blue line follows the black one, the nearer "
                   f"the NSE is to 1."
                   + (say("step 4, the stretch in the picture of the NSE", stretch)
                      or " On the right the forecast is a flat line at the average of the days shown: it scores 0."))

    return f"""{opening(4, "Two ways to judge a forecast: how close it is on all days, and whether it warns on the days that matter.",
        "One forecast for one day is easy to judge: compare it with what the river did.", say("step 4, one number for many days", one_number))}
{concept("nse", here=say("step 4, the NSE box", nse_box))}
<p>In words: the NSE compares the errors of the forecast with the errors of the simplest forecast of all, the river&rsquo;s average flow on
every day. Large errors count far more than small ones, so the few days of high water decide much of this score.</p>
{picture("guide_nse.png", nse_caption)}
{para(say("step 4, the high-flow level", level))}
{concept("high_flow", here=say("step 4, the high-flow box", high_box))}
{para(say("step 4, why the onsets are singled out", why_onsets))}
{concept("hit_miss", here="a model warns of a day when its forecast for that day is above the high-flow level. Two shares are then counted: "
                          "the share of high-flow days that were warned of, and the share of warnings that were false.")}
{para("An AI course calls the share of high-flow days that were warned of &lsquo;recall&rsquo;, and the share of warnings that were right "
      "&lsquo;precision&rsquo;.", say("step 4, why the right days are not simply counted", rare))}
{say("step 4, the three verdicts", verdicts)}
{closing("The NSE says how close the forecasts are on all days: 1 is perfect, 0 is no better than the average. Warnings are judged by their "
         "hits, misses and false warnings, above all at the onsets. The rule for GO, CAUTION and NOT YET was fixed before the results were known.",
         "sections 2 and 5")}"""


# ---- step 5: simple models first --------------------------------------------------------------------------------------------
def _step5(R, T):
    def table():
        B = R["baselines"]
        return B, {(r["model"], r["lead"]): r for r in B["summary_table"]}

    def baseline_box():
        B = R["baselines"]
        kinds = {m["model"] for m in B["models"].values()}
        beat = "; ".join(f"{LEAD_WORD[k]}, {n2(B['best_by_validation'][f'l{k}']['without_weather_forecast']['test']['nse_median'])}" for k in LEADS)
        return f"{_word(len(kinds))} kinds of simple model were scored. With the river gauge, the NSE to beat is: {beat}."

    def calendar():
        _, S = table()
        return (f"<li><b>Calendar-day average.</b> The forecast is the average flow of that day of the year in the training years. Its NSE is "
                f"{n2(S['base_climatology', LEADS[0]]['nse_median'])}: no better than the average.</li>")

    def ridge():
        F = R["baselines"]["features"]["w"]
        sums = sorted(int(c[len("rain_sum"):]) for c in F if c.startswith("rain_sum"))
        return (f"<li><b>Ridge regression.</b> {len(F)} summaries of the past weather are made by hand, such as the rain of the last {sums[0]} to "
                f"{sums[-1]} days. Each summary is multiplied by a number, and the results are added up. The numbers are found from the training "
                f"years. &lsquo;Ridge&rsquo; means that these numbers are kept small, so that they do not fit the accidents of those years.</li>")

    def forest():
        trees = {m["settings"]["n_trees"] for m in R["baselines"]["models"].values() if m["model"] == "rf"}
        if len(trees) != 1:
            raise ValueError("the forests do not all have the same number of trees")
        n = next(iter(trees))
        return (f"<li><b>Random forest.</b> {n} decision trees are built from the same summaries. A tree is a chain of yes/no questions that "
                f"ends in a value for the flow. Each tree is built from a random sample of the training days, so the trees differ. The "
                f"forecast is the average of the {n} answers, which evens out their errors.</li>")

    def named(m, both_gauge=False):
        """'ridge regression with the river gauge' from a model record of baselines.json (the gauge is left out when the sentence names it)."""
        return KIND[m["model"]] + ("" if both_gauge else " with the river gauge" if m["uses_river_gauge"] else " without the river gauge")

    def to_beat():
        B, S = table()
        k = LEADS[0]
        best = B["best_by_validation"][f"l{k}"]["without_weather_forecast"]
        no_fc = [n for n, m in B["models"].items() if not m["uses_weather_forecast"]]
        top = max(no_fc, key=lambda n: S[n, k]["nse_median"])
        rule = ("The number to beat is the test score of the simple model that was best on the validation years: choosing by the test score "
                "would be a leak (step 3).")
        a, b = B["models"][best["model"]], B["models"][top]
        if top == best["model"] or n2(S[top, k]["nse_median"]) == n2(best["test"]["nse_median"]):
            return rule
        gauge = a["uses_river_gauge"] and b["uses_river_gauge"]
        return (f"{rule} {'With the river gauge, the' if gauge else 'The'} number to beat {FIRST} is therefore the "
                f"{n2(best['test']['nse_median'])} of {named(a, gauge)}, although {named(b, gauge)} scores {n2(S[top, k]['nse_median'])} in the "
                f"test years.")

    def gauge():
        _, S = table()
        k = LEADS[0]
        pers, lw, fw, lq, fq = (S[m, k]["nse_median"] for m in ("base_persistence", "base_linreg_w", "base_rf_w", "base_linreg_wq", "base_rf_wq"))
        if not (pers > max(lw, fw) and lq > lw and fq > fw):
            raise ValueError(f"{FIRST} the river gauge is not worth more than the past weather")
        return (f"First, the river gauge is worth a lot {FIRST}. {QUOTED_CAP} reads nothing but the gauge and reaches an NSE of {n2(pers)}. "
                f"Ridge regression and the random forest reach {n2(lw)} and {n2(fw)} from the past weather alone, and {n2(lq)} and {n2(fq)} with "
                f"the gauge.")

    def onsets():
        B, S = table()
        no_fc = [n for n, m in B["models"].items() if not m["uses_weather_forecast"]]
        if any(S["base_persistence", k]["onset_pod"] != 0 for k in LEADS):
            raise ValueError("'tomorrow = today' warns of an onset")
        top, name, k = max((S[n, k]["onset_pod"], n, k) for n in no_fc for k in LEADS)       # the largest share of onsets warned, and who reaches it
        return (f"Second, {QUOTED} can never warn of an onset: its forecast is high only when the river already is. For {FIRST} it warns of "
                f"{pct(S['base_persistence', LEADS[0]]['pod'])} of all high-flow days and of none of the onsets. None of the simple models above "
                f"warns of more than {pct(top)} of the onsets: that best case is {named(B['models'][name])}, {LEAD_WORD[k]}.")

    caption = (f"<b>The simple models in the test years.</b> {READ} Start at the top left: a large marker is the median NSE of one model, a small "
               f"dot is one river. Filled circles are models that read the river gauge, and hollow circles are models that do not; {FIRST} the "
               f"filled circles are higher. The grey square is the calendar-day average. The two panels on the right show the warnings, and "
               f"the text explains the panel at the bottom left.")

    return f"""{opening(5, "What very simple methods reach: their score is the number that every later model has to beat.",
        "A complicated model is only worth its cost if it forecasts better than a simple one. So the simple ones came first.")}
{concept("baseline", here=say("step 5, the baseline box", baseline_box))}
<ul>
<li><b>{QUOTED_CAP}.</b> The forecast is the flow that the river gauge shows today. Nothing is learned.</li>
{say("step 5, the calendar-day average", calendar)}
{say("step 5, ridge regression", ridge)}
{say("step 5, the random forest", forest)}
</ul>
{para("The last two were built without the river gauge and with it.", say("step 5, the number to beat", to_beat))}
{picture("baselines.png", caption)}
{para("The simple models already teach two things.", say("step 5, what the river gauge is worth", gauge))}
{para(say("step 5, the onsets", onsets))}
<p>The panel at the bottom left gives the last two models a perfect weather forecast: the weather really recorded for the days being
forecast. Nobody has that when a forecast is made. So these scores are a ceiling: an upper bound that cannot be had in practice (step 8
explains the term).</p>
{closing(f"Build the simple models first: they give the number to beat. Here they also show two things. The river gauge is worth a lot "
         f"{FIRST}, and repeating today&rsquo;s flow can never warn of an onset.", "section 2")}"""


def html(R, T):
    steps = "\n".join(part(R, T) for part in (_step1, _step2, _step3, _step4, _step5))
    return f'{LAYOUT}<div class="g1">\n{steps}\n</div>'
