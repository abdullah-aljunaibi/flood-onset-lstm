"""Guide, step 16 - what the project did next, and what it showed (the experiments of 7 and 8 October 2026; report sections 4.6 to 4.8).

  16  What we did next, and what it showed   the question (is the limit really the information?), the ladder of information, what a control is for
                                             (a matched-window control separates the hours from a second product; a shuffled control can measure
                                             harm instead of information), a real forecast and the operating point, more rivers, and what the new
                                             numbers do not say

Written for readers who are new to machine learning: short sentences, plain words, one idea each. The two new concept boxes are
'real_time' (real-time and after-the-fact data) and 'operating_point' (the warning level and the operating point); every other technical word was
explained in a box of an earlier step. The step stands between step 15 and the list of concepts; g4_concepts.html() asks for it (html(R, T) here).

Where the numbers come from (each read by key, none typed in), through the verified readers of page 1 of the report (report_sections/s0_summary.py):
  aorc_rain_to_midnight.json     the AORC rain to midnight, its controls, the simple models   (aorc_facts)
  rain_to_midnight.json          the same with ERA5 model rain                               (era5_facts)
  stage4_realtime.json           Stage IV, the real-time product, in place of AORC           (stage4_facts)
  aorc_hindsight.json, review_aorc_rain_to_midnight.json   the hindsight check and its bound  (hindsight_facts)
  forecast_input.json, gefs_basins.json   a real weather forecast as an input, its skill and its controls   (forecast_facts)
  more_basins.json               63 training rivers instead of 12, and rivers never seen    (rivers_facts)
The fixed wordings (AORC_WORDING, MOVED_POINT, INFO_NOT_NET, REALTIME_WORDING and, through the helpers of s0_summary, the wordings on hindsight, on the NSE of the forecast and on the
reviews) are those of the report, used word for word.
Nothing is computed beyond differences, counts and formatting.

Every paragraph that states an outcome is built inside _try(). It first checks that the stored numbers still say what the sentence says (_need). If a file
or a key is missing, or a rerun has changed the outcome, the paragraph is left out and one line goes to stderr: the guide never prints a sentence that the
results files do not support.

Layout: as in step 15 and in g3_findings: the picture is a full-width float that moves to the top of the next page if it does not fit, the paragraphs after
it fill the page, and the boxes and the end of the step start below it.
"""
import re
import sys

import guide_common as G
import team
from common import FIG
from flood import LEADS
from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed
from report_sections.s0_summary import (AORC_WORDING, INFO_NOT_NET, MOVED_POINT, REALTIME_WORDING, _and, _esc, _leads, _word, aorc_facts, era5_facts, forecast_facts,
                                         hindsight_clause, hindsight_facts, nse_forecast_numbers, nse_forecast_words, new_work, review_sentence, rivers_facts,
                                         stage4_facts)

FIRST, MORE = LEADS[0], LEADS[1:]                              # the lead time the ladder starts from (1 day ahead), and the two that follow
lk = lambda k: f"l{k}"
PIC = "What the rain of the whole day adds"                    # the name of the picture: its caption starts with it, and the text speaks of it
SAME = 0.02                                                    # a clean control 'equals' the model without the forecast within two points of onsets warned
STYLE = ("<style>.g5{clear:both} .g5 .ff{float:left;width:100%;break-inside:avoid} .g5 .ff figure{margin-top:2pt} "
         ".g5 h2,.g5 .concept,.g5 .remember,.g5 .keep,.g5 .who,.g5 ul.short,.g5 ol.short{clear:both} .g5 .keep{break-inside:avoid} "
         ".g5 ol.short,.g5 ul.short{margin:2pt 0 6pt;padding-left:17pt} .g5 .short li{margin-bottom:2pt}</style>")


def _try(what, make):
    """make() builds one paragraph (or box, or caption) from the results files. If a file or a key is missing, or the stored numbers no longer support
    the sentence (_need), the piece is left out and one line on stderr says so."""
    try:
        return make()
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError) as e:
        print(f"g5_next: {what} left out ({type(e).__name__}: {e})", file=sys.stderr)
        return ""


def _need(holds, outcome):
    """Stop a sentence whose outcome the stored numbers no longer show (caught by _try)."""
    if not holds:
        raise ValueError(f"the results files no longer show that {outcome}")


def _slash(by_lead, ks=LEADS, fmt=pct):
    """The values of the leads ks, 'a / b / c'."""
    return " / ".join(fmt(by_lead[k]) for k in ks)


def _two(by_lead, ks=MORE, fmt=pct):
    """The values of two leads as 'a and b' (the form the report uses for 2 and 3 days ahead)."""
    return _and(fmt(by_lead[k]) for k in ks)


def _named_picture():
    return f"The picture &lsquo;{PIC}&rsquo;"


def _clock(R):
    """The hours that the text names, read from the records: where the window of the rain table ends (the control of the third rung is summed over it), where the UTC day ends,
    and the hour (UTC) of the forecast run. ValueError if a record does not hold them (_try then leaves the piece out)."""
    A = R["aorc_rain_to_midnight"]
    table = next(c["day_ends_local"] for c in A["inputs"]["clock"] if str(c["input"]).startswith("aorc_to_09"))
    utc = re.search(r"ends (\d\d:\d\d)", A["figure_labels"]["wqa_utc"])
    run = re.search(r"the (\d\d) UTC run", R["gefs_basins"]["rule"]["run"])
    if not (table and utc and run):
        raise ValueError("the windows of the controls or the run of the forecast are not in the records")
    early = re.search(r"hours before (\d{4})", A["outcome_caveat"])
    if not early:
        raise ValueError("the year before which AORC's hours are disaggregated daily totals is not in the caveat of the AORC results")
    return dict(table=table, utc=utc.group(1), run=f"{run.group(1)}:00", early=early.group(1))


def _lead_words(ks):
    return _leads(list(ks))


def _for_guide(text):
    """The shared wordings of the report speak of basins and of section 4.2; the guide speaks of rivers and of step 9, where the rule for a clear difference is taught."""
    return text.replace(" basins", " rivers").replace("the rule of section 4.2", "the rule of step 9")


def _seeds(R):
    """How many times each network was trained, as a word (results/aorc_rain_to_midnight.json, seeds)."""
    return _word(len(R["aorc_rain_to_midnight"]["seeds"]))


def _reviews(R):
    """What the four review records of the experiments add up to over ALL their rounds (a record keeps its second and third round under round_2 and round_3, each with counts of its
    own): dict(n, checks, multi, clean, team_open); multi: [(experiment, number of rounds)] of the records with more than one round. ValueError while a record is missing."""
    recs = [R.get(f"review_{k}") for k in ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input")]
    if not all(isinstance(r, dict) and not r.get("quick_mode") and isinstance(r.get("counts"), dict) for r in recs):
        raise ValueError("the four review records of the experiments are not all there")
    later = [[r[x] for x in sorted(r) if re.fullmatch(r"round_\d+", x)] for r in recs]
    blocks = [r["counts"] for r in recs] + [x.get("counts") or {} for rounds in later for x in rounds]
    num = lambda b, *keys: next((int(b[k]) for k in keys if isinstance(b.get(k), (int, float))), 0)          # the rounds name their counts differently
    mism = sum(num(b, "numerical_discrepancies", "numerical_mismatches") for b in blocks)
    failed = sum(num(b, "failed") for b in blocks)
    return dict(n=len(recs), checks=sum(num(b, "checks_listed") for b in blocks), multi=[(name, 1 + len(rounds)) for name, rounds in zip(("E1", "E1b", "E3", "E2"), later) if rounds],
                clean=(failed == 0 and mism == 0), mismatches=mism, failed=failed,
                team_open=all(str(r.get("human_review", "")).startswith("not yet done") for r in recs))


def available(R):
    """The facts that step 16 stands on, as a dict (AO, E1, S4, FC, RV, HS), when the results of the four experiments of 7 and 8 October 2026 can all be quoted; else None. The
    other steps point to step 16 only while this holds (HS, the hindsight check, is optional: its sentence is left out without it)."""
    AO, E1, S4, FC, RV, HS = (fn(R) for fn in (aorc_facts, era5_facts, stage4_facts, forecast_facts, rivers_facts, hindsight_facts))
    return dict(AO=AO, E1=E1, S4=S4, FC=FC, RV=RV, HS=HS) if AO and E1 and S4 and FC and RV else None


def html(R, T):
    """Step 16 as HTML; '' (and one line on stderr) while the results of the experiments are missing, so that the guide is then the guide of fifteen steps."""
    A = available(R)
    if A is None:
        print("g5_next: the results of the experiments of 7 and 8 October 2026 are missing or not in the expected form: step 16 is left out", file=sys.stderr)
        return ""
    AO, E1, S4, FC, RV, HS = (A[k] for k in ("AO", "E1", "S4", "FC", "RV", "HS"))
    n_on = AO["n"]
    hind = hindsight_clause(HS)                                # 'hindsight at most about 3 points', to stand beside the AORC share (empty while the hindsight record is missing)
    hind_note = f" ({hind}; see the end of this step)" if hind else ""

    # ---- the ladder --------------------------------------------------------------------------------------------------------
    def opening():
        _need(AO and E1 and S4 and FC, "the four experiments have their results files")
        nw = new_work(R)
        _need(nw, "the four review records give the days of the experiments")
        return ("<p>If the limit is the information in the inputs, then more information should lift the warnings, and the simple models should gain as much as the "
                "LSTM. A claim like that can be tested. "
                f"On {nw['days']} the project did so, one change at a time, with a control for each change. As before, every network was trained {_seeds(R)} times.</p>"
                f"<p>There were {_word(nw['n'])} experiments, called E1, E1b, E3 and E2 in the report. E1 gave the model the rain of the whole issue day, up to midnight, from a weather "
                "model (ERA5). E1b gave it the same rain from AORC, an hourly rain product, and then put Stage IV, a product that a real service could read at midnight, "
                "in the place of AORC, as a check. E3 taught the network on more rivers. E2 gave it a real weather forecast.</p>")

    def ladder():
        _need(AO["verdict_onset"][FIRST] == "NOT YET" and E1["nldas"] > AO["base"][FIRST], "the rungs rise one by one and the verdict is NOT YET")
        up = AO["upper"]
        return (f"<p><b>The ladder of information.</b> Each rung gives the model one more thing. The numbers are the share of the {count(n_on)} onsets that were warned of "
                f"{LEAD_WORD[FIRST]}, with the river gauge.</p>"
                f'<ol class="short"><li>The rain table alone, which is what the model had in steps 8 to 10: <b>{pct(AO["base"][FIRST])}</b>.</li>'
                f"<li>Rain up to the evening of the issue day, a second rain product (step 11): <b>{pct(E1['nldas'])}</b>.</li>"
                f"<li>The rain of the whole issue day, up to midnight, from AORC: <b>{pct(AO['with_'][FIRST])}</b>{hind_note}. AORC is an hourly rain product of the US weather "
                f"service {cite('aorc')}, and it is {AORC_WORDING} (see the box below). Stage IV is a radar product that a real service could read at midnight "
                f"{cite('stage4')}. It gives "
                f"<b>{pct(S4['stage4'] / S4['n'])}</b> in the place of AORC, and <b>{pct(S4['no_last'] / S4['n'])}</b> if its last hour is left out, because it is about an hour "
                f"behind. These two figures are {REALTIME_WORDING}.</li>"
                f"<li>A perfect weather forecast, the upper bound of step 8: <b>{pct(up[FIRST])}</b>.</li></ol>"
                f"<p>For {_lead_words(MORE)}, the rain table alone warns of {_two(AO['base'])} of the onsets. The rain up to midnight lifts that to {_two(AO['with_'])}. "
                f"A public weather forecast beside it, which is a real forecast with real errors, lifts it to {_two(FC['fc'])}. A perfect weather forecast would "
                f"give {_two(up)}.</p>")

    def real_time_box():
        _need(AO["lag_days"] is not None and S4["share"] > 0.8, "AORC is corrected some days after the hour and Stage IV keeps most of its effect")
        K = _clock(R)
        _need(any("first postings" in p for p in R["stage4_realtime"]["honesty_points"]),
              "the Stage IV record says that it is not established whether its archive holds first postings or later versions of each hour")
        return G.concept("real_time",
                         here=f"AORC, the rain of the third rung, is {AORC_WORDING}: its daily totals come from rain gauges and are spread over the hours with radar, with "
                              f"corrections about {_word(AO['lag_days'])} days after the hour, and before {K['early']} its hours are daily totals split by a rule. Stage IV is the "
                              f"real-time product. Where AORC warns of {pct(AO['with_'][FIRST])}{f' ({hind})' if hind else ''}, Stage IV in its place warns of "
                              f"{pct(S4['stage4'] / S4['n'])}, and of {pct(S4['no_last'] / S4['n'])} with its last hour left out. These figures are {REALTIME_WORDING}. A service "
                              f"with these inputs might reach about {pct(S4['no_last'] / S4['n'])} to {pct(S4['stage4'] / S4['n'])}, not {pct(AO['with_'][FIRST])}. And it is not "
                              f"established whether the archive that was used holds each hour as it was first posted or in a later, corrected version.")

    # ---- the controls ---------------------------------------------------------------------------------------------------------
    def control():
        _need(AO["control_flat"] and AO["hours_reading"] == "clear gain", "the control over the table's own window is flat and the hours are a clear gain")
        K = _clock(R)
        return ("<p><b>What does a gain come from?</b> A gain proves little until a control says where it comes from. Take the third rung. AORC is a different product from "
                f"the rain table. So the gain could come from the new hours, or only from having a second, better rain product. A control separates the two: the same AORC "
                f"rain, added up over the 24 hours that the rain table seems to cover, the day that ends at {K['table']} (step 11). It is a second product that holds no new hour. It warns of "
                f"{pct(AO['to09'])} of the onsets, as few as the rain table alone ({pct(AO['base'][FIRST])}). Added up to the evening, about {K['utc']}, the same rain warns of "
                f"{pct(AO['utc'])}; up to midnight, of {pct(AO['with_'][FIRST])}. So the gain comes from the hours after about {K['table']} of the issue day, and the last ones "
                f"count most. A better product alone adds nothing. {_named_picture()} shows it: look at the diamond below the AORC rung.</p>")

    def picture():
        _need(AO and E1, "the rungs of the picture have their numbers")
        K = _clock(R)
        if not (FIG / "aorc_rain_to_midnight.png").exists():
            raise ValueError("figures/aorc_rain_to_midnight.png is missing")
        cap = (f"<b>{PIC}: AORC against ERA5.</b> <b>How to read it:</b> Top: the share of the onsets warned {LEAD_WORD[FIRST]}. Bottom: the median NSE. A filled dot is a model with the "
               f"river gauge, and its bar runs from the lowest to the highest of three trainings. The hollow marks are controls: the diamond is the same rain over the window of the "
               f"rain table, the day that ends at {K['table']}, the triangle the same rain over the day that ends at about {K['utc']}, the circle the same rain one day older. The shaded rungs are the new ones. Look at the "
               f"AORC rung: {pct(AO['with_'][FIRST])}{hind_note} against {pct(AO['to09'])} for its diamond. ERA5, rain from a weather model, reaches {pct(E1['with_'])} with the same "
               f"hours, so the product counts as well. From left to right the rungs are: the rain table alone; the evening rain of step 11 (NLDAS); the rain to midnight from "
               f"ERA5 and from AORC; the AORC rain in place of the rain table; the evening and midnight rains together; and the upper bound.")
        return f'<div class="ff">{fig("aorc_rain_to_midnight.png", cap)}</div>'

    # ---- a real forecast, and the operating point ---------------------------------------------------------------------------------
    def forecast():
        K = _clock(R)
        _need(FC["success"] and all(FC["gain"][k] > 0 for k in MORE), "the forecast lifts the onsets warned 2 and 3 days ahead")
        X = R["forecast_input"]
        y = lambda part, key: f"{X[part][key][0][:4]}&ndash;{X[part][key][1][:4]}"
        return (f"<p><b>A real forecast.</b> It comes from a public archive of past forecasts {cite('gefsv12')}: one member of a set of forecast runs that differ a little, the "
                f"control member of a NOAA ensemble. The run used was made at midnight by the world&rsquo;s reference clock ({K['run']} UTC), {_word(FC['hours_old'][0])} to "
                f"{_word(FC['hours_old'][1])} hours before the forecast is issued. Against the AORC rain of the same days ({AORC_WORDING}) "
                f"its correlation is {_slash(FC['corr'], fmt=n2)} at {_and(LEADS)} days. A correlation of 1 would be a perfect match, and 0 no match at all. So the "
                f"forecast is useful, and it is not exact.</p>"
                f"<p>The archive starts in {FC['years'][0]}, so the forecast was given to the network in two ways. In the first, the network was trained as before, on "
                f"{y('split_main', 'train')}, with {_word(len(LEADS))} extra inputs: the AORC rain ({AORC_WORDING}) of the {_word(len(LEADS))} days to be forecast, which is a perfect rain forecast. Only when it "
                f"was scored on the usual test years, {y('split_main', 'test')}, was the real forecast put in their place. Beside the rain to midnight, it lifts the onsets warned {_lead_words(MORE)} from "
                f"{_two(FC['rain'])} to {_two(FC['fc'])}, a clear gain in every training. In the second, the network was trained with the real forecast, on "
                f"{y('split_satellite_era', 'train')}, and scored on {y('split_satellite_era', 'test')}: a shorter test, with scores of its own that do not stand beside the others.</p>")

    def one_day():
        _need(FC["moved"] and FC["gain"][FIRST] < 0 and FC["chosen_gain"][FIRST] > 0, "1 day ahead the share warned falls at the usual level and the point has moved")
        return (f"<p>{_cap(LEAD_WORD[FIRST])} something else happened. At the usual warning level the share warned went down, from {pct(FC['pod'][0])} to {pct(FC['pod'][1])}. "
                f"But its warnings were right more often: {pct(FC['far'][1])} of them were false, against {pct(FC['far'][0])}. The model had moved along a trade-off: "
                f"fewer warnings, and more of them right.</p>")

    def operating_box():
        _need(FC["s_gain"][FIRST] > 0, "retrained with the forecast the share warned rises 1 day ahead")
        return G.concept("operating_point",
                         here=f"At the warning level chosen on the validation years, the same model warns of {pct(FC['rain'][FIRST] + FC['chosen_gain'][FIRST] / 100)} of the onsets "
                              f"{LEAD_WORD[FIRST]}, more than the {pct(FC['rain'][FIRST])} without the forecast. And when the network is trained with the forecast as an input, "
                              f"on a shorter set of years of its own, the share warned rises from {pct(FC['s_rain'][FIRST])} to {pct(FC['s_fc'][FIRST])}.")

    def moved():
        _need(FC["nse_gain"][FIRST] < 0, "1 day ahead the forecast lowers the median NSE")
        return (f"<p>So the fall in the share warned was {MOVED_POINT}. The NSE is another matter, because it has no warning level that could move. "
                f"{_cap(nse_forecast_words(FC))}: {_for_guide(nse_forecast_numbers(FC))}. Why the model warns of fewer onsets at the usual level was not found out.</p>")

    def shuffled():
        ctrl = FC["s_ctrl"]
        _need(FC["s_ctrl_reading"][FIRST] == "clear gain", "the forecast of the second design is a clear gain over its own control")
        _need(all(abs(ctrl[k] - FC["s_rain"][k]) <= SAME for k in LEADS) and all(FC["ctrl"][k] < FC["rain"][k] for k in LEADS),
              "the shuffled control collapses in E2a and equals the model without the forecast in E2b")
        ahead = {k: FC["s_fc"][k] - ctrl[k] for k in LEADS}
        return (f"<p><b>A control that measures harm.</b> A control for the forecast is a forecast with the right climate and the wrong day: each day&rsquo;s values come from "
                f"another year. Given that, the network&rsquo;s onsets warned collapse to {_slash(FC['ctrl'])}. With no forecast at all they were {_slash(FC['rain'])}. The network "
                f"had been trained on exact rain in those places. It had learned to trust it, and noise misleads it. So this control shows how much a misleading input hurts. It does "
                f"not show how much a real input helps.</p>"
                f"<p>A clean control needs a network that is trained and run on shuffled values, and the second way gives one. It warns of {_slash(ctrl)}, the same as with no forecast "
                f"on those years ({_slash(FC['s_rain'])}). The real forecast adds {_slash(ahead, fmt=lambda v: signed(100 * v, 0))} points above it. So that gain is the "
                f"forecast&rsquo;s information. A control must differ in the one thing that is tested. Otherwise it tests something else.</p>")

    # ---- more rivers ---------------------------------------------------------------------------------------------------------------
    def rivers():
        n12, n63, n11, n50 = RV["n_train"]["lstm_w"], RV["n_train"]["lstm63_w"], RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
        _need(RV["nse_reading"][FIRST] == "no clear difference" and all(v == "clear gain" for v in RV["unseen_reading"]["w_static"].values()),
              "63 rivers change nothing clear 1 day ahead for the 12 and 50 rivers clearly help rivers never seen")
        return (f"<p><b>More rivers.</b> The third experiment trained the network on {n63} rivers instead of {n12}. For the same {n12} rivers, with their gauge, nothing clear changed "
                f"{LEAD_WORD[FIRST]}: the median NSE went from {n2(RV['nse']['lstm_wq'][FIRST])} to {n2(RV['nse']['lstm63_wq'][FIRST])}. What is missing there is tomorrow&rsquo;s rain, "
                f"and other rivers do not hold it. For a river the network has never seen it mattered: with {n50} training rivers instead of {n11}, the median NSE went from "
                f"{n2(RV['unseen']['lobo_w_static'][FIRST])} to {n2(RV['unseen']['unseen50_w_static'][FIRST])}. Why is not known: the number, the variety and the nearness of the "
                f"training rivers changed together.</p>")

    # ---- what these numbers do not say ------------------------------------------------------------------------------------------
    def limits():
        g, gs = AO["gain"], FC["gain_simple"]
        _need(min(g["ridge"], g["forest"]) >= 0.9 * g["lstm"] and all(min(gs["ridge regression"][k], gs["random forest"][k]) >= 0.9 * gs["lstm"][k] for k in MORE),
              "the simple models gain as much as the LSTM from the new inputs")
        _need(set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) == {"NOT YET"}, "the verdict for onsets is NOT YET with every new input")
        lo, hi = AO["seeds"]
        seeds = _seeds(R)
        if not HS:
            print("g5_next: the hindsight check (results/aorc_hindsight.json, results/review_aorc_rain_to_midnight.json) is missing: its sentence is left out of 'What these numbers do not say'",
                  file=sys.stderr)
        items = [f"<b>Hindsight.</b> AORC was made after the day was over, so its hours could hold a trace of the next day&rsquo;s rain. A check against Stage IV on {HS['days_review']} "
                 f"storm days found no sign of it. So the {pct(AO['with_'][FIRST])} comes with a limit: {hind} (the best estimate is "
                 f"{n2(HS['best'], 0)} points)." if HS else "",
                 f"<b>{_cap(seeds)} trainings.</b> The {pct(AO['with_'][FIRST])} is the average of {seeds} networks. Alone, they warned of {pct(lo)} to {pct(hi)}. "
                 f"With so few trainings, {nse_forecast_words(FC)} ({_for_guide(nse_forecast_numbers(FC, medians=False))}).",
                 f"<b>The simple models gain as many onsets.</b> With the rain to midnight, ridge regression and the random forest gain {signed(g['ridge'], 0)} and {signed(g['forest'], 0)} "
                 f"points of onsets warned, the LSTM {signed(g['lstm'], 0)}. With the forecast {LEAD_WORD[MORE[0]]}, they gain {signed(gs['ridge regression'][MORE[0]], 0)} and "
                 f"{signed(gs['random forest'][MORE[0]], 0)}, the LSTM {signed(gs['lstm'][MORE[0]], 0)}. So the gain is the information, not the network.",
                 f"<b>The warnings still cannot be trusted.</b> With the rain to midnight, {pct(AO['false'][FIRST])} of the onset warnings are false; with the forecast, "
                 f"{_two(FC['false_fc'])} {_lead_words(MORE)}. The verdict for onsets is NOT YET at every lead time.",
                 _try("step 16, who checked the experiments,", who_checked)]
        return '<p class="intro"><b>What these numbers do not say.</b></p><ul class="short">' + "".join(f"<li>{x}</li>" for x in items if x) + "</ul>"

    def who_checked():
        """The honest statement on who ran and who checked the four experiments: Claude Code ran them; what the independent AI reviewer agents checked and what only the final critic
        agent verified is the wording of the report (review_sentence); the checks are those of ALL rounds of the four records (the first experiment was reviewed three times); the team's
        own checks are as team.json says."""
        V = _reviews(R)
        _need(V["mismatches"] == 0, "no review found a number that differed")
        _need(V["team_open"], "every review record says that the team's own check was not done")
        nw, checks = new_work(R), team.checks_state()
        own = ("The team members&rsquo; own checks of the code are still to be done." if checks["n_done"] == 0 else f"{_esc(team.checks_sentence())}.")
        found = ("no check failed and no number differed" if V["failed"] == 0 else
                 f"no number differed; the {_word(V['failed'])} checks that failed, in the later rounds, asked for wording, above all the disclosure that seeds were added "
                 f"after the test scores were known")
        return (f"<b>Who checked it.</b> Claude Code ran the {_word(V['n'])} experiments (E1, E1b, E3 and E2) under the group leader&rsquo;s direction on "
                f"{nw['days'] if nw else 'the days named in the report'}. {review_sentence()}. The {_word(V['n'])} reviews list {V['checks']} checks over all their rounds"
                f"{' (' + _and(f'{name} was reviewed in {_word(n)} rounds' for name, n in V['multi']) + ', the others once)' if V['multi'] else ''}: "
                f"{found}. {own}")

    def still_open():
        """What is still open: the next steps of section 8 of the report, in their order (the team's own checks first while they are not all done)."""
        _need(FC and AO and S4, "the experiments have their results files")
        checks = team.checks_state()
        first_ = "First come the team members&rsquo; own checks of the code. Then, in the order of the report:" if checks["n_done"] < checks["n"] else ""
        members = FC["members"] - 1 if FC.get("members") else None
        others = f"the other {_word(members)} members of the forecast ensemble" if members else "the other members of the forecast ensemble"
        items = [f"five trainings in place of {_seeds(R)}, where the NSE decides.", f"{others}, and the forecast&rsquo;s temperature.",
                 "A network trained on Stage IV, with a test of its false warnings on days without an onset.",
                 f"The form of network that reads a forecast day by day, as an operational flood model does {cite('nearing2024')}.",
                 "Satellite soil moisture as one more input.", "Hourly rain and flow, and a training target aimed at peaks and onsets."]
        first_p = f"<p>{first_.replace('members&rsquo;own', 'members&rsquo; own').strip()}</p>" if first_ else ""
        return (f'<p class="intro"><b>What is still open.</b></p>{first_p}<ul class="short">' + "".join(f"<li>{x[0].upper() + x[1:]}</li>" for x in items) + "</ul>")

    keep_text = ("More information lifted the warnings, for the simple models as much as for the LSTM. A control that differs in one thing says where a gain comes from; a "
                 "control that misleads the network measures harm, not information. Ask of every input: could a real service have it at that hour? Even with the best new input "
                 "the verdict for onsets stays NOT YET.")
    lead = "The finding of step 15 is a claim about information, so the project tested it: it gave the model more, one thing at a time."

    first = _try("step 16, the question,", opening)
    return f"""{STYLE}<div class="g5">
{G.keep(G.step(16, lead), first)}
{_try("step 16, the ladder,", ladder)}
{_try("step 16, the picture of the rain to midnight,", picture)}
{_try("step 16, the box 'real-time and after-the-fact data',", real_time_box) or G.concept("real_time")}
{_try("step 16, the control over the table's own window,", control)}
{_try("step 16, a real forecast,", forecast)}
{_try("step 16, the 1-day result,", one_day)}
{_try("step 16, the box 'operating point',", operating_box) or G.concept("operating_point")}
{_try("step 16, the moved operating point,", moved)}
{_try("step 16, the shuffled control,", shuffled)}
{_try("step 16, more rivers,", rivers)}
{_try("step 16, what is still open,", still_open)}
{G.keep(_try("step 16, what the numbers do not say,", limits), G.remember(keep_text), G.final_detail_for("sections 4.6 to 4.8 and 8"))}
</div>"""


def _cap(s):
    return s[0].upper() + s[1:]
