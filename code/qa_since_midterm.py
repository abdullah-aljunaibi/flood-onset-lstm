"""Group I of the professor's-questions sheet (code/build_qa.py): what was done after the mid-term talk of 15 October 2026 - the river's state at the
issue time (E4e), the hour-by-hour rain of the issue day (E4b), a stricter warning level (E4a), five trainings instead of three for every set of scripts
15 to 18, the GIS layers, the teammates' Flood_Cloud_Model package and the final documents - and the parts of the sheet that the final state changes:
the member page and the facts that page 1 quotes.

Every number is read, none is typed: from the registry of the final report (results/final_report_numbers.json, whose 'text' field is the number as the
report and the final deck print it) or from the results files, through qa_final. Every pointer to the final report is checked against its docx text
(qa_final.FinalReport). The fixed wordings of the project (AORC_WORDING, REALTIME_WORDING, the hindsight clause, UPPER, 'no clear difference') stand
word for word where they apply. A question whose facts are missing is left out, with one line on stderr, and no other answer refers to it (I_OK).
"""
import json
import re

from report_common import LEAD_WORD, n2, pct, signed
from report_sections.s0_summary import AORC_WORDING, REALTIME_WORDING, moved_operating_point

import qa_final as QF

UPPER = "an upper bound that cannot be had in practice"
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
ERRORS = (KeyError, TypeError, IndexError, ValueError, AttributeError, StopIteration, ZeroDivisionError, OSError)
LABEL = {"rain_to_midnight": "E1", "aorc_rain_to_midnight": "E1b", "more_basins": "E3", "forecast_input": "E2", "warning_policy": "E4a",
         "within_day_inputs": "E4b", "flow_state": "E4e"}                 # the seven experiments with a review record of their own
GROUP_TITLE = "Since the mid-term talk{when}: the river at the issue time, within-day rain, a stricter warning level, five seeds, GIS layers, the teammates&rsquo; model"
# the questions of group I and the parts of facts() each one needs
I_NEED = dict(midterm=("E4E", "E4B", "E4A", "SEEDS", "GIS", "FCM", "DOCS"), riverstate=("E4E",), ownsplit=("E4E",), sixtyeight=("E4E",), stricter=("E4A",),
              withinday=("E4B",), moreseeds=("SEEDS", "REV"), cloud=("FCM",), numbers=("NUM", "REV"), gislayers=("GIS",))


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def join(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def points(v):
    """A difference of two shares in percentage points: 0.087 -> '+8.7 points'."""
    return f"{signed(100 * v, 1)} points"


def _part(what, make):
    try:
        return make()
    except ERRORS as err:
        QF.warn(f"{what} was not read ({err!r}): the questions that need it are left out")
        return None


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _choices(R, name):
    """The text of choices_made_after_looking_at_test_scores of one results file, as one string."""
    return " ".join(str(x) for x in (R[name].get("choices_made_after_looking_at_test_scores") or []))


# ---- the independent review records of the seven experiments, read tolerantly ---------------------------------------------------
def review_facts(R):
    """What the question on the reviews quotes, for the seven experiments with a review record (E1, E1b, E3, E2, E4a, E4b, E4e): rounds, checks,
    mismatches over every round, and the latest round (the five-seed round of 30 October 2026) with its values, failed checks and must-fix items;
    and whether each results file now discloses the seed extension that the latest rounds asked to be disclosed."""
    out = {}
    for name, label in LABEL.items():
        rec = R[f"review_{name}"]
        blocks = [rec] + [rec[k] for k in QF.rounds_of(rec)]
        counts = [QF.round_counts(b) for b in blocks]
        last_name, last = QF.latest_round(rec)
        lc = QF.round_counts(last)
        if lc["values"] is None or lc["mismatches"] is None:
            raise KeyError(f"the latest round of results/review_{name}.json records no values compared or no mismatches")
        text = _choices(R, name)
        out[name] = dict(label=label, rounds=len(blocks), checks=sum(c["checks"] or 0 for c in counts), mismatches=sum(c["mismatches"] or 0 for c in counts),
                         last=last_name, last_when=str(last.get("when", ""))[:10], last_values=lc["values"], last_failed=lc["failed"] or 0,
                         last_failed_list=QF.failed_checks(last), last_must=lc["must"] or 0, last_verdict=str(last.get("verdict", "")),
                         same_agent="the same independent AI reviewer agent" in str(rec.get("who", "")),
                         team_open=all(str(b.get("human_review", "not")).startswith("not") for b in blocks),
                         disclosed=("30 October 2026" in text and "eeds 3 and 4" in text and "after" in text))
    return out


# ---- the facts of group I ----------------------------------------------------------------------------------------------------
def facts(R, REG, FR):
    """A dict of parts (None, with one line on stderr, where a file or a key is missing)."""
    t, v = REG.t, REG.v
    F = {}

    def e4e():
        X = R["flow_state"]
        S = X["e4e_success"]
        cols = X["design"]["sets"]["wqaH_s"]["extra_columns"]
        hours = sorted(int(m.group(1)) for c in cols for m in [re.fullmatch(r"dq(\d+)", c)] if m)
        meta = _json(QF.ROOT / "data" / "hourly_flow" / "usgs_iv_meta.json")
        starts = sorted(int(g["hourly"]["first_hour_end"][:4]) for g in meta["gauges"].values())
        sets = ("wqa_s", "wqah_s", "wqawh_s", "wqawhf_s", "wqaf_s", "wqap_s", "wqah22_s", "wqawhf22_s")
        launch = _choices(R, "flow_state")
        if not ("five seeds were chosen at launch" in launch and "before any test score was seen" in launch):
            raise ValueError("results/flow_state.json does not say that its five seeds ran in one pass before any test score was seen")
        if not hours or len(starts) != len(meta["gauges"]):
            raise ValueError("the hourly columns or the start of the hourly records are not found")
        val = {s: v(f"e4e_{s}_l1_onset_pod") for s in sets}
        valfar = {s: v(f"e4e_{s}_l1_onset_far") for s in sets}
        hour_now, hour_early = (X["design"]["sets"][s]["last_available_hour"] for s in ("wqaH_s", "wqaH22_s"))
        return dict(n_cols=len(cols), hours=hours, start=(starts[0], starts[-1]), main_start=R["lstm_all"]["split"]["train"][0][:4],
                    pod={s: t(f"e4e_{s}_l1_onset_pod") for s in sets}, far={s: t(f"e4e_{s}_l1_onset_far") for s in sets},
                    cap={s: t(f"e4e_{s}_l1_far_cap_pod") for s in sets}, capfar={s: t(f"e4e_{s}_l1_far_cap_far") for s in sets}, val=val, valfar=valfar,
                    earlier=word(hour_now - hour_early),
                    csi_full=t("e4e_wqawhf_s_l1_onset_csi"), verdict_full=v("e4e_wqawhf_s_l1_plain_test_verdict_onset"),
                    verdict_full22=v("e4e_wqawhf22_s_l1_plain_test_verdict_onset"),
                    met=bool(S["met"]), met3=bool(S["met_with_first_three_seeds"]), n_seeds=S["n_seeds"]["wqaH_s"], first3=len(S["first_three_seeds"]),
                    not_above=S["seeds_of_wqaH_s_not_above_the_best_wqa_s_seed"], wh_all_above=bool(S["every_seed_of_wqaWH_s_above_every_seed_of_wqa_s"]),
                    only_with=t("e4e_paired_wqah_s_vs_wqa_s_l1_only_with"), only_without=t("e4e_paired_wqah_s_vs_wqa_s_l1_only_without"),
                    n=t("e4e_paired_wqah_s_vs_wqa_s_l1_n_onsets"), split=(t("split_sat_train_dates"), t("split_sat_val_dates"), t("split_sat_test_dates")),
                    gain_fc=val["wqaf_s"] - val["wqa_s"], gain_cols=val["wqawhf_s"] - val["wqaf_s"], gain_h=val["wqah_s"] - val["wqa_s"],
                    date_e2=t("date_review_e2"), hf_period=t("hf_period"), main_wqa=t("kf_level_wqa_onset_warned"),
                    go_warned=t("go_min_warned"), go_false=t("go_max_false"), go_v=(v("go_min_warned"), v("go_max_false")))
    F["E4E"] = _part("the E4e facts (results/flow_state.json, the registry, data/hourly_flow/usgs_iv_meta.json)", e4e)

    def e4b():
        X = R["within_day_inputs"]
        cols = X["design"]["sets"]["wqaW"]["extra_columns"]
        last = sorted(int(m.group(1)) for c in cols for m in [re.fullmatch(r"aorc_last(\d+)h", c)] if m)
        if not last or "measured for the daily total, not for these columns" not in X["outcome_full"]:
            raise ValueError("the within-day columns or the clause on their hindsight are not found in results/within_day_inputs.json")
        S = X["e4b_success"]
        return dict(n_cols=len(cols), last=last, base=t("e4b_wqa_l1_onset_pod"), with_=t("e4b_wqaw_l1_onset_pod"), older=t("e4b_wqaw_older_l1_onset_pod"),
                    far_base=t("e4b_wqa_l1_onset_far"), far_with=t("e4b_wqaw_l1_onset_far"), cap_with=t("e4b_wqaw_l1_far_cap_pod"),
                    cap_older=t("e4b_wqaw_older_l1_far_cap_pod"), cap_base=t("e4b_wqa_l1_far_cap_pod"), met=bool(S["met"]), n_seeds=len(v("e4b_seeds")))
    F["E4B"] = _part("the E4b facts (results/within_day_inputs.json and the registry)", e4b)

    def e4a():
        out = R["warning_policy"]["outcome"]
        if "does not create skill" not in out or "no model whose inputs can be had in practice meets it" not in out:
            raise ValueError("the outcome of results/warning_policy.json no longer says that the decision layer creates no skill and that no practical model meets GO")
        k = "e4a_lstm_wqa_ll1"
        return dict(csi_pod=t(f"{k}_csi_test_onset_pod"), csi_far=t(f"{k}_csi_test_onset_far"), cap_pod=t(f"{k}_far_cap_test_onset_pod"),
                    cap_far=t(f"{k}_far_cap_test_onset_far"), cap_v=v(f"{k}_far_cap_test_onset_pod"), wqf_cap=t("e4a_lstm_wqf_ll1_far_cap_test_onset_pod"),
                    go_false=t("go_max_false"))
    F["E4A"] = _part("the E4a facts (results/warning_policy.json and the registry)", e4a)

    def seeds():
        note = _choices(R, "aorc_rain_to_midnight")
        planned = re.search(r"planned on (\d{1,2} [A-Z][a-z]+ \d{4})", note)
        if not planned or "for every set alike, whatever the direction of its result" not in note:
            raise ValueError("the seed note of results/aorc_rain_to_midnight.json does not say when and for which sets the extra seeds were planned")
        arc = QF.ROOT / "results" / "archive"
        mb3 = _json(arc / "more_basins_3seeds_2026-10-30.json")
        fi3 = _json(arc / "forecast_input_3seeds_2026-10-30.json")

        def e3(X):                                                     # the 1-day onsets of the 63-river gauge model against the 12-river one
            T = X["summary_test"]["sets"]
            a, b = T["lstm_wq"]["l1"]["onset"], T["lstm63_wq"]["l1"]["onset"]
            far = lambda o: o["false_alarms"] / (o["hits"] + o["false_alarms"])
            g = X["gain_rule"]["lstm63_wq_against_lstm_wq"]["l1"]["onset"]
            return dict(moved=moved_operating_point(a["pod"], b["pod"], far(a), far(b), a["csi"], b["csi"]), mcnemar=g["paired"]["mcnemar_p"],
                        csi=(a["csi"], b["csi"]), pod=(a["pod"], b["pod"]))
        e3_3, e3_5 = e3(mb3), e3(R["more_basins"])
        nse2 = lambda X: X["gain_rule"]["wqaF_sub_against_wqa"]["l1"]["nse"]["reading"]
        five, three = (sorted(int(x) for x in re.findall(r"\d+", str(v(k)))) for k in ("e4e_seeds", "seeds_list"))
        added = [s for s in five if s not in three]
        if not added or len(three) != v("seeds_per_set"):
            raise ValueError("the seed lists of the registry (e4e_seeds, seeds_list) do not show which seeds were added")
        return dict(added=join(str(s) for s in added), orig=f"{three[0]} to {three[-1]}", n_added=len(added), n5=len(five),
                    date=t("seed_extension_date"), scripts=t("seed_extension_scripts"), trainings=t("seed_extension_trainings"), planned=planned.group(1),
                    machine=t("windows_pc"), e3_3=e3_3, e3_5=e3_5, e2_nse=(nse2(fi3), nse2(R["forecast_input"])),
                    aorc=(t("aorc_share_three_seeds"), t("kf_level_wqa_onset_warned")), hind=(t("hindsight_three_seed_wording"), t("fw_hindsight")),
                    e4=(len(v("e4b_seeds")), len(v("e4e_seeds"))), stored=t("stored_sets_three_seeds_list"), stored_n=v("seeds_per_set"),
                    two=tuple(len(x) if isinstance(x, list) else int(x) for x in (v("cmp_seeds"), v("sat_seeds"))), values=t("review_five_seed_values_compared"), mism=t("review_five_seed_mismatches"),
                    archived=len(list(arc.glob("*_3seeds_*.json"))))
    F["SEEDS"] = _part("the seed facts (the registry, the seed notes of the results files, results/archive/)", seeds)

    def fcm():
        if not FR.says("8.3", "no statement of AI use yet"):
            raise ValueError("final report Section 8.3 no longer says that the package has no statement of AI use")
        authors = re.search(r"\(([^)]*)\)", REG.v("flood_cloud_model_authors"))
        names = authors.group(1).split(", ") if authors else []
        return dict(acc=t("fcm_headline_accuracy"), det=t("fcm_headline_detection"), pers=t("fcm_persistence_detection_24h"), horizon=t("fcm_primary_horizon_hours"),
                    r_flow=t("fcm_onset_only_recall_flow_24h"), r_rain=t("fcm_onset_only_recall_rain_24h"), r_gefs=t("fcm_onset_only_recall_gefs_24h"),
                    years=t("fcm_period_years"), members=t("fcm_gefs_members"), authors=QF.esc(join(names)) if names else "the other members of the team")
    F["FCM"] = _part("the facts of the Flood_Cloud_Model package (the registry and final report Section 8.3)", fcm)

    def num():
        return dict(keys=f"{R['final_report_numbers']['_meta']['n_keys']:,}", records=t("review_records"), rounds=t("review_rounds"),
                    checks=t("review_checks_listed"), values=t("review_five_seed_values_compared"), mism=t("review_five_seed_mismatches"),
                    passes=t("document_review_passes"), last_entries=t("rev_documents_2026_10_30_registry_entries"),
                    last_mism=t("rev_documents_2026_10_30_registry_value_mismatches"), team=t("team_checks_done"))
    F["NUM"] = _part("the facts of the number checks (the registry and its _meta)", num)

    def gis():
        G = R["gis_layers"]
        rv = R["review_gis_layers"]
        r_name, r_last = QF.latest_round(rv)
        rc = QF.round_counts(r_last)
        kinds = sorted({k.split()[0] for k in (f["kind"] for f in G["files"]) if k.split()[0].startswith("Geo")})
        inside = bool(G["verification"]["checks"]["gpkg_gauges_inside_outlines"]["ok"])
        unverified = all(str(x).startswith("UNVERIFIED") for k, x in G["licences"].items() if k != "note")
        return dict(files=t("gis_layers_files"), kinds=kinds, area_med=t("gis_area_vs_gages2_median_pct"), area_max=t("gis_area_vs_gages2_largest_pct"),
                    elev_rho=t("gis_elev_rank_correlation"), elev_n=t("gis_elev_basins_over_25m"), elev_range=t("gis_elev_over_25m_range"),
                    elev_names=join(QF.esc(x.split(",")[0]) for x in G["summary"]["basins_with_elev_more_than_25_m_from_camels"]), forest=t("gis_forest_share_mean_diff_points"), inside=inside, unverified=unverified,
                    rounds=len(QF.rounds_of(rv)) + 1, verdict=QF.esc(r_last["verdict"]), values=f"{rc['values']:,}",
                    mism=rc["mismatches"], elev_max_basin=t("gis_elev_largest_diff_basin"), elev_max=t("gis_elev_largest_diff_m"),
                    land_year=t("gis_land_cover_year"))
    F["GIS"] = _part("the GIS facts (results/gis_layers.json, results/review_gis_layers.json and the registry)", gis)

    def docs():
        timing = QF.deck_timing()
        pages = QF.pdf_pages(QF.FINAL_PDF)
        if not timing or not pages:
            raise ValueError("the final deck's timing table or the final report's page count is not there")
        readme = QF.HANDIN_README.read_text(encoding="utf-8")
        template = "NSFC-style template" in readme
        return dict(pages=pages, slides=len(timing), seconds=sum(s for *_, s in timing), audits=QF.esc(QF.handin_audits()), refs=FR.n_references,
                    template=template, datasets=(QF.HANDIN / "5_Datasets_and_Repository_Links_Flood.pdf").exists(), sheet=QF.CHECK_SHEET.exists())
    F["DOCS"] = _part("the facts of the final documents (the hand-in folder)", docs)
    F["REV"] = _part("the review records results/review_*.json of the seven experiments", lambda: review_facts(R))
    return F


def ok_keys(F):
    """The questions of group I whose facts are all there."""
    have = {k for k, need in I_NEED.items() if all(F.get(p) for p in need)}
    for k in sorted(set(I_NEED) - have):
        QF.warn(f"the question '{k}' of group I was left out: a file or key it needs is missing (see the lines above)")
    return have


# ---- the questions of group I ------------------------------------------------------------------------------------------------
def cap(s):
    return s[0].upper() + s[1:] if s else s


def word_or(text):
    """A registry count as a word where it is small ('4' -> 'four')."""
    return word(int(text)) if str(text).isdigit() else str(text)


def add_questions(Q, ref, F, FR, C):
    """Adds the questions of group I through Q (the sheet's question adder). ref(*keys) refers to questions by key (only to those that exist).
    C: values of the sheet that the answers share - HIND_W (the hindsight clause), about (a share in words), n (basins), I_OK (the questions of this
    group that will exist), proposal_day (the day of the mid-term talk)."""
    E, B, A, S, G, M, N, D = (F.get(k) for k in ("E4E", "E4B", "E4A", "SEEDS", "GIS", "FCM", "NUM", "DOCS"))
    HIND_W, about = C["HIND_W"], C["about"]
    ok = C["I_OK"]
    aorc_every = (f"every row reads the AORC rain to midnight ({AORC_WORDING}; {HIND_W}, a bound measured for the daily total only)" if HIND_W else
                  f"every row reads the AORC rain to midnight ({AORC_WORDING})")

    if "midterm" in ok:
        Q("I", "midterm", "What is new since the mid-term?",
          f"""Since the talk of {C['proposal_day']}: <i>the river&rsquo;s state at the issue time</i> (E4e): {word(E['n_cols'])} columns from each gauge&rsquo;s own hourly
          record take the onsets warned {LEAD_WORD[1]} from {E['pod']['wqa_s']} to {E['pod']['wqah_s']} on a split of their own, comparable only inside it, and with
          within-day rain and the forecast beside them to {E['pod']['wqawhf_s']}, {E['verdict_full']}, still not GO ({ref('riverstate', 'sixtyeight')}). <i>The
          hour-by-hour rain of the issue day</i> (E4b): {B['base']} ({HIND_W}) to {B['with_']}, control {B['older']}: no clear gain ({ref('withinday')}). <i>A stricter
          warning level</i> (E4a): at the level that holds false warnings at {A['go_false']}, {A['cap_pod']} of the onsets are warned ({ref('stricter')}).
          <i>{cap(word(S['n5']))} trainings instead of {word(S['stored_n'])}</i> for every set of scripts {S['scripts']}, added on {S['date']} after the test scores were
          known, and disclosed ({ref('moreseeds')}). <i>GIS layers</i> of all {C['n']} basins, {G['files']} files, checked against the CAMELS tables
          ({ref('gislayers')}). <i>The teammates&rsquo; Flood_Cloud_Model package</i>, reviewed and written into the report&rsquo;s discussion ({ref('cloud')}).
          <i>The final documents</i>: the final report ({D['pages']} pages{', on the professor&rsquo;s NSFC-style template' if D['template'] else ''}), the
          {word(D['slides'])}-slide final deck, the datasets and links document and the team check sheet. Each new experiment has its own AI review record; the
          team&rsquo;s own checks are {N['team'] if N else 'still open'}.""",
          f"""Setting the new shares beside the old ones as gains of one experiment. E4e has its own split ({ref('ownsplit')}), and most of its rise to
          {E['pod']['wqawhf_s']} is the forecast, tested before the mid-term ({ref('sixtyeight')}). And &lsquo;the extra seeds confirmed everything&rsquo;: they moved
          some readings ({ref('moreseeds')}).""",
          f"""{FR.where(('1.1', 'Three experiments reviewed'), ('7.3', 'The river at the issue time (E4e)'), ('9.3', 'Done in the last week'), 'T6', 'T7',
                         ('8.1', 'Flood_Cloud_Model'), context=" of the question on what is new")} &middot; <code>results/flow_state.json</code>,
          <code>results/within_day_inputs.json</code>, <code>results/warning_policy.json</code>, <code>results/gis_layers.json</code> &middot;
          <code>code/20_warning_policy.py</code>, <code>code/21_within_day_inputs.py</code>, <code>code/22_flow_state.py</code>, <code>code/27_gis_layers.py</code>""")

    if "riverstate" in ok:
        hours = join(str(x) for x in E["hours"])
        rule = (f"holds with the first {word(E['first3'])} of its {word(E['n_seeds'])} trainings and not with all {word(E['n_seeds'])}" if E["met3"] and not E["met"] else
                "is met" if E["met"] else "is not met")
        Q("I", "riverstate", "Why the river&rsquo;s hourly state, what did it do, and why do you call its gain &lsquo;not established&rsquo;?",
          f"""<i>Why:</i> a daily mean flow hides where the river stands at midnight, the issue time; the teammates&rsquo; hourly package pointed at it ({ref('cloud')}).
          <i>What:</i> the USGS 15-minute record of all {C['n']} gauges ({E['hf_period']}), reduced to hours, gives {word(E['n_cols'])} columns at midnight: the flow against
          the high-flow level Q95, its change over the last {hours} hours, the day&rsquo;s highest hourly flow and the hours above Q95. Beside the AORC rain to midnight,
          on the satellite-era split ({ref('ownsplit')}), the onsets warned {LEAD_WORD[1]} go from {E['pod']['wqa_s']} to {E['pod']['wqah_s']} and the false share from
          {E['far']['wqa_s']} to {E['far']['wqah_s']}; at the level that holds false warnings at {E['go_false']}, from {E['cap']['wqa_s']} to {E['cap']['wqah_s']}. It
          cut false warnings the most and gave the biggest step at that level. <i>Not established:</i> the success rule fixed before training (every training with the
          columns above every training without them) {rule}; all {word(E['n_seeds'])} ran together from the start, in one pass before any test score was seen. The
          paired onsets ({E['only_with']} warned only with the columns, {E['only_without']} only without) point the same way but were not a test fixed in advance.
          With the within-day rain beside it, every training {'is' if E['wh_all_above'] else 'is not'} above every base training ({ref('withinday')}). {cap(aorc_every)}.""",
          f"""&ldquo;The river state was our biggest gain.&rdquo; In false warnings and at the {E['go_false']}-false level, yes; in onsets warned at the fixed level its
          gain is not established by the rule fixed in advance. And it read the approved record with no delivery delay: with the record ending {E['earlier']} hours
          earlier the river-only model warns of {E['pod']['wqah22_s']}, inside the seed spread ({ref('sixtyeight')}).""",
          f"""{FR.where(('7.3', 'The river at the issue time (E4e)'), 'T6', ('9.1', 'not established'), ('8.2', 'hourly state (E4e)'), ('T14', 'E4e'),
                         context=" of the question on the river's state")} &middot; <code>results/flow_state.json</code> (e4e_success, readings, operating_points,
          data_coverage), <code>data/hourly_flow/usgs_iv_meta.json</code> &middot; <code>code/fetch_usgs_iv.py</code>, <code>code/22_flow_state.py</code> &middot;
          <code>figures/flow_state.png</code>""")

    if "ownsplit" in ok:
        Q("I", "ownsplit", "Why does E4e have a split of its own, and can its numbers stand beside the main split&rsquo;s?",
          f"""Because the hourly gauge records start between {E['start'][0]} and {E['start'][1]}, gauge by gauge, after the main split&rsquo;s training years begin
          ({E['main_start']}): the main split cannot be trained with them. So E4e uses the satellite-era split (training {E['split'][0]}, validation {E['split'][1]}, test
          {E['split'][2]}; {E['n']} test onsets), with its own high-flow levels. The name only records where the satellite rain record starts; E4e reads no satellite
          data. Its numbers are comparable only inside it: its base, the AORC rain to midnight, warns of {E['pod']['wqa_s']} of the onsets there, against
          {E['main_wqa']} on the main split ({HIND_W}), so only gains inside the split carry over.""",
          f"""Setting E4e&rsquo;s {E['pod']['wqawhf_s']} beside the main split&rsquo;s {E['main_wqa']}, or calling the split &lsquo;satellite data&rsquo;. Other years,
          other levels, a weaker base. A main-split E4e was not run ({FR.inline(('9.3', 'the main-split E4e'), context=' on the main-split E4e')}).""",
          f"""{FR.where(('5.3', 'The satellite-era split'), 'T6', ('4.1', 'E4e'), context=" of the question on the split of E4e")} &middot;
          <code>results/flow_state.json</code> (split, data_coverage), <code>data/hourly_flow/usgs_iv_meta.json</code> &middot; <code>code/22_flow_state.py</code>""")

    if "sixtyeight" in ok:
        miss_w, miss_f = E["val"]["wqawhf_s"] < E["go_v"][0], E["valfar"]["wqawhf_s"] > E["go_v"][1]
        misses = "both counts" if miss_w and miss_f else "the share warned" if miss_w else "the false share"
        Q("I", "sixtyeight", f"Your best model warns of {E['pod']['wqawhf_s']} of the onsets. Why is that not GO, and not what a service would get?",
          f"""Not GO: GO needs at least {E['go_warned']} warned and at most {E['go_false']} false together; the full model (river state, within-day rain and forecast, on
          the satellite-era split) warns of {E['pod']['wqawhf_s']} with {E['far']['wqawhf_s']} false, CSI {E['csi_full']}: {E['verdict_full']} by the rule fixed in
          advance. At the level that holds false warnings at {E['go_false']} it keeps {E['cap']['wqawhf_s']}, against {E['cap']['wqap_s']} for perfect rain ({UPPER}).
          Not a service: {aorc_every}, the archived GEFS reforecast and the approved gauge record with no delivery delay, inputs a service does not have; how much worse a
          service would do was not measured, and with the gauge record ending {E['earlier']} hours earlier the full model warns of {E['pod']['wqawhf22_s']} with
          {E['far']['wqawhf22_s']} false. And most of the rise from {E['pod']['wqa_s']} to {E['pod']['wqawhf_s']} is the forecast, tested before the mid-term: retrained
          with the forecast alone (E2b, {E['date_e2']}) the model warns of {E['pod']['wqaf_s']} ({points(E['gain_fc'])}); the new columns add {points(E['gain_cols'])}
          and mainly cut false warnings ({E['far']['wqaf_s']} to {E['far']['wqawhf_s']}).""",
          f"""&ldquo;{E['pod']['wqawhf_s']} is nearly GO&rdquo;, or &ldquo;a service would warn of {E['pod']['wqawhf_s']}&rdquo;. It misses GO on {misses}, its
          inputs are not a service&rsquo;s, and it stands on its own split ({ref('ownsplit')}); on the main split every input set that could be had in practice is NOT
          YET for onsets. And the river state was not the biggest gain in onsets warned ({ref('riverstate')}).""",
          f"""{FR.where(('7.1', 'the full model (river state, within-day rain and forecast) is CAUTION'), 'T6', ('7.3', 'GO is not met'), ('8.2', 'hourly state (E4e)'),
                         ('9.3', 'What could run today'), ('3.2', 'Criteria fixed in advance'), context=" of the question on the best model")} &middot;
          <code>results/flow_state.json</code> (summary_test, delivery, operating_points), <code>results/forecast_input.json</code> (the satellite-era sets of E2b)
          &middot; <code>code/22_flow_state.py</code>""")

    if "stricter" in ok:
        Q("I", "stricter", "Would a stricter warning level solve the false alarms?",
          f"""No: it trades warnings for false alarms and adds no skill. E4a trains nothing; on the stored forecasts it chooses each level on the validation years. For
          the main-split model with the AORC rain to midnight ({AORC_WORDING}) the level with the best CSI warns of {A['csi_pod']} of the onsets ({HIND_W}) with
          {A['csi_far']} false; the level chosen to hold false warnings at {A['go_false']} keeps only {A['cap_pod']}, with {A['cap_far']} false on the test years. Even
          the perfect-weather model keeps only {A['wqf_cap']} there ({UPPER}). A decision layer re-arranges the trade-off and does not create skill; no model whose
          inputs can be had in practice meets GO.""",
          f"""&ldquo;Just raise the threshold.&rdquo; At the {A['go_false']}-false level {about(1 - A['cap_v'])} of the onsets go unwarned; and the level is chosen on
          the validation years, never on the test years ({ref('warnlevel')}).""",
          f"""{FR.where(('7.3', 'A second operating point (E4a)'), ('9.3', 'Ceilings'), 'T6', ('T14', 'E4a'), context=" of the question on a stricter level")} &middot;
          <code>results/warning_policy.json</code> (outcome, sets) &middot; <code>code/20_warning_policy.py</code>, <code>code/warning_policy.py</code> &middot;
          <code>figures/warning_policy.png</code>""")

    if "withinday" in ok:
        last = join(str(x) for x in B["last"])
        cap_level = f"the level that holds false warnings at {A['go_false']}" if A else "the level that caps the false share"
        Q("I", "withinday", "Did the hour-by-hour rain of the issue day help?",
          f"""Not clearly. E4b adds {word(B['n_cols'])} columns that describe how the issue day&rsquo;s AORC rain fell hour by hour: the rain of the last {last} hours,
          the wettest hour (basin mean and wettest cell) and the wet hours. The base already had the day&rsquo;s total, so this is how the rain fell, not more rain. On
          the main split, {LEAD_WORD[1]}, the onsets warned go from {B['base']} ({HIND_W}) to {B['with_']}, the false share from {B['far_base']} to {B['far_with']}; a
          control with the same columns one day late gives {B['older']}. The rule fixed before training (every training above every training of the base and of the
          control) is {'met' if B['met'] else 'not met'}: {'a gain' if B['met'] else 'no clear gain'}. At {cap_level} the columns give {B['cap_with']} against
          {B['cap_older']} for the control. The hindsight of these columns was not measured: they share AORC&rsquo;s channel with no bound of their own.""",
          f"""&ldquo;More rain made it better.&rdquo; The base already had the day&rsquo;s total; these columns only say when it fell, and {B['with_']} is not a clear gain
          over its control ({B['older']}). Beside the river state on the satellite-era split the same columns do lift every training ({ref('riverstate')}).""",
          f"""{FR.where(('7.3', 'Within-day inputs (E4b)'), ('8.2', 'The within-day columns'), ('9.1', 'the within-day rain'), ('T14', 'E4b'),
                         context=" of the question on within-day rain")} &middot; <code>results/within_day_inputs.json</code> (e4b_success, operating_points) &middot;
          <code>code/21_within_day_inputs.py</code>, <code>code/subdaily_inputs.py</code> &middot; <code>figures/within_day_inputs.png</code>""")

    if "moreseeds" in ok:
        e3w = (f"E3&rsquo;s one-day onsets (more training rivers), a moved operating point with {word(S['stored_n'])} seeds (McNemar p = {n2(S['e3_3']['mcnemar'])}),"
               f" are a clear loss by the seed rule with {word(S['n5'])} (CSI {n2(S['e3_5']['csi'][0], 3)} against {n2(S['e3_5']['csi'][1], 3)}, McNemar p ="
               f" {n2(S['e3_5']['mcnemar'], 3)})" if S["e3_3"]["moved"] and not S["e3_5"]["moved"] else "E3&rsquo;s one-day onsets are a clear loss by the seed rule")
        e2w = (f"E2&rsquo;s one-day NSE went from &lsquo;{S['e2_nse'][0]}&rsquo; to &lsquo;{S['e2_nse'][1]}&rsquo;" if S["e2_nse"][0] != S["e2_nse"][1] else "")
        changes = [e3w] + ([e2w] if e2w else []) + [f"the AORC share went from {S['aorc'][0]} to {S['aorc'][1]}, and the clause beside it from "
                                                    f"&lsquo;{S['hind'][0]}&rsquo; to &lsquo;{S['hind'][1]}&rsquo;"]
        Q("I", "moreseeds", "You added seeds after seeing the test scores. Is that p-hacking?",
          f"""It is a choice made after looking, so it is disclosed, not hidden: seeds {S['added']} were added to every set of scripts {S['scripts']} on {S['date']},
          after the test scores of seeds {S['orig']} were known ({S['trainings']} trainings on a second machine). What keeps it from p-hacking: the extra seeds were
          planned on {S['planned']}, before the mid-term talk, for every set alike, whatever the direction of its result; no setting, input, stopping rule, warning
          level or success rule changed and no training was dropped; the {word(S['stored_n'])}-seed files are kept ({S['archived']} in <code>results/archive/</code>);
          and AI reviewers recomputed every {word(S['n5'])}-seed number ({S['values']} values, {S['mism']} mismatches). The seeds did move readings, one against the
          model: {join(changes)}. E4b and E4e had {word(S['e4'][1])} trainings from launch, E4a trains nothing; the stored sets ({S['stored']}) keep
          {word(S['stored_n'])}, """ + (f"the satellite-input and Transformer comparisons {word(S['two'][0])}" if S['two'][0] == S['two'][1] else
                                f"the Transformer comparison {word(S['two'][0])} and the satellite-input comparison {word(S['two'][1])}") + ".",
          f"""&ldquo;The extra seeds only confirmed our results&rdquo;, or leaving them out of the talk. They changed readings, and the change of E3 went against the
          model. And not &lsquo;{word(S['n5'])} seeds everywhere&rsquo;: the stored sets keep {word(S['stored_n'])} ({ref('spent')}).""",
          f"""{FR.where(('5.3', 'Choices after seeing test scores'), ('8.2', 'Seeds.'), ('AppC', 'Choices made after looking at test scores'), ('T7', 'seeds 3 and 4'),
                         context=" of the question on the added seeds")} &middot; <code>choices_made_after_looking_at_test_scores</code> of every results file,
          <code>results/archive/*_3seeds_2026-10-30.json</code>, the latest round of every <code>results/review_*.json</code>""")

    if "cloud" in ok:
        Q("I", "cloud", f"Your teammates&rsquo; model reports {M['acc']} accuracy. Why not use it?",
          f"""Because that number, and its {M['det']} detection, count cases that need no forecast. Flood_Cloud_Model, built by {M['authors']}, is a separate
          gradient-boosting experiment on the same basins, {M['years']}, with hourly USGS flow, MRMS radar rain and {M['members']} GEFS members. Most hours have no
          flood, so accuracy is high by default; and the detection counts rivers already above Q95, where persistence alone detects {M['pers']}. On cases still below
          the level its {M['horizon']}-hour recall is {M['r_flow']} with flow only, {M['r_rain']} with observed rain and {M['r_gefs']} with GEFS. Its numbers are not
          comparable in level with the LSTM&rsquo;s: no repeated trainings, test years used in development, a different hourly event. What it gave the project: it
          pointed at the missing one-day input, the river&rsquo;s state at the issue time, which E4e then tested ({ref('riverstate')}).""",
          f"""&ldquo;{M['acc']} beats our numbers&rdquo;, or dismissing the package. Different measures, years and events; its authors are on the team, its numbers
          were recomputed by AI agents and it is in the report&rsquo;s discussion. Its statement of AI use is still missing: its authors are to supply one.""",
          f"""{FR.where(('8.1', 'A second experiment on the same basins'), ('8.3', 'no statement of AI use yet'), ('AppE', 'The teammates'), 'T10',
                         context=" of the question on the teammates' model")} &middot; <code>handoff/extras/Flood_Cloud_Model/</code> (reference/results.json),
          <code>handoff/extras/Flood_Cloud_Model_REVIEW_2026-10-28.md</code>""")

    if "numbers" in ok:
        Q("I", "numbers", "How do you know the numbers in the report are right?",
          f"""Layer by layer, and the last layer is still open. No number is typed: the report, the deck and this sheet read <code>results/*.json</code> through code,
          and the report&rsquo;s registry (<code>results/final_report_numbers.json</code>, {N['keys']} keys) records the file and key of every value. Independent AI
          reviewer agents recomputed the experiments from the stored forecasts with their own code: {N['records']} records, {N['rounds']} rounds, {N['checks']} listed
          checks; in the {word(S['n5']) if S else 'five'}-seed rounds of {S['date'] if S else 'the last day'}, {N['values']} values with {N['mism']} mismatches
          ({ref('reviewed')}). {cap(word_or(N['passes']))} critic passes compared the documents with the registry; the last checked {N['last_entries']} entries, with
          {N['last_mism']} value mismatches. And the team&rsquo;s own checks: {N['team']}; each member&rsquo;s check recomputes the numbers of his or her slides from the
          stored forecasts (<code>team_checks/team_check.py</code>; list at the end).""",
          f"""&ldquo;The numbers were checked by the team&rdquo;, or &ldquo;independent means people&rdquo;. Every reviewer is an AI agent, the same kind of system that
          wrote the code; a recomputation shows that the code does what it says, not that the design is right. Say {N['team']} until a member has recorded a check in
          <code>team.json</code>.""",
          f"""{FR.where(('6.1', 'One source of numbers'), ('6.3', 'Independent review'), 'T19', ('AppC', 'Reproducibility.'), ('T8', 'Verified?'),
                         context=" of the question on the numbers")} &middot; <code>results/final_report_numbers.json</code>, <code>results/review_*.json</code>,
          <code>team_checks/TEAM_CHECK_SHEET.pdf</code>, <code>team.json</code>""")

    if "gislayers" in ok:
        Q("I", "gislayers", "What do the GIS layers contain, and what did they check?",
          f"""The {C['n']} basins as GIS layers in <code>results/gis_layers/</code>, {G['files']} files ({join(G['kinds'])}): outlines and gauges, and for each basin
          the elevation and the land cover of {G['land_year']} clipped to its outline. Checked against the CAMELS tables: area {G['area_med']} in the median (largest
          {G['area_max']}); mean elevation in the same order (rank correlation {G['elev_rho']}), but {G['elev_n']} mountain basins lie {G['elev_range']} above the
          CAMELS value ({G['elev_names']}), cause not found; forest share {G['forest']} on average.{' Every gauge lies inside its outline.' if G['inside'] else ''} An
          independent AI reviewer rebuilt them in {word(G['rounds'])} rounds: &lsquo;{G['verdict']}&rsquo;; its last round compared {G['values']} values, with
          {G['mism']} mismatches. The layers carry no licence of their own{' (the licences of their sources are marked UNVERIFIED in the layer record)' if G['unverified'] else ''};
          the team&rsquo;s own check of them is open.""",
          f"""&ldquo;Our layers show that CAMELS is wrong&rdquo;, or &ldquo;our RS/GIS members checked them&rdquo;. The differences are reported, not explained
          ({ref('rasters')}); the check was an AI agent&rsquo;s. The own check of slide 2 reads the check table (list at the end).""",
          f"""{FR.where(('T1', 'GIS layer export'), ('T7', 'GIS layers'), ('AppB', 'Imagery'), 'F5', context=" of the question on the GIS layers")} &middot;
          <code>results/gis_layers.json</code> (summary, verification, files), <code>results/review_gis_layers.json</code> (round_2) &middot;
          <code>code/27_gis_layers.py</code>, <code>code/07_basin_maps.py</code>""")


# ---- the member page ---------------------------------------------------------------------------------------------------------
KEY_OF = {"The team lead": "A", "Member 2": "R2", "Member 3": "G1", "Member 4": "G2"}     # names -> the keys of team.json


def open_items(FR, REG, n_members, authors):
    """What is still open before the e-mail goes, each item kept only while the final report's text (or team.json) says it is open."""
    items = []
    if FR.has("[REPO URL]") or FR.has("[RELEASE/DRIVE URL]"):
        items.append(f"the repository and data links (still <code>{REG.t('repo_url')}</code> and <code>{REG.t('data_bundle_url')}</code> in the report)")
    items.append("the cover date, which must be the day of sending")
    if REG.has("team_checks_done"):
        items.append(f"the members&rsquo; own checks ({REG.t('team_checks_done')} in <code>team.json</code>)")
    if FR.says("AppE", "Signed with the following exception"):
        items.append(f"the {word(n_members)} signatures (Appendix E holds an exception sentence for signing before the checks are recorded)")
    if FR.says("8.3", "no statement of AI use yet"):
        items.append(f"a statement of AI use from the authors of Flood_Cloud_Model ({authors})")
    if FR.has("no team member has yet read"):
        items.append("the members&rsquo; own reading of the cited papers (the report says that no member has read them yet)")
    lic = re.search(r"\((the MIT licence for code and CC BY [\d.]+ for results) are proposed\)", FR.parts.get("8.3", ""))
    if FR.says("8.3", "is still to be chosen by the group"):
        items.append("a licence for the project&rsquo;s own code and results" + (f" ({QF.esc(lic.group(1))} are proposed)" if lic else ""))
    return items


def member_page(F, FR, REG, team, X, have, C):
    """(head, lead, [(heading, [items])]): the member page of the final state. Every member is named with the role of code/final_report/config.json;
    the slides and seconds come from the speaker notes of the final deck, the command, the part and the figure from team_checks/team_check.py. These are
    things to do: nothing here says that anyone has done them."""
    group, members = QF.team_members()
    group_text = REG.t("group_number") if REG.has("group_number") else (f"Group #{group}" if group else "the group")
    timing = QF.deck_timing()
    checks = QF.team_check_members()
    ref = lambda *keys: X(*[k for k in keys if k in have])
    talk = REG.t("final_presentation") if REG.has("final_presentation") else "the talk"
    head = f"Check before the hand-in and the talk &middot; {group_text}"
    lead = (f"What each member of {group_text} should open, run or verify personally before the e-mail goes and before the talk of {talk}. None of it needs a new "
            f"experiment, and nothing on this page says that anyone has done it.")
    state = team.checks_state()
    S, M = F.get("SEEDS"), F.get("FCM")
    everyone = [f"Say who did what as the final report does ({FR.inline(('8.3', 'AI-generated content'), ('10.2', 'Claude Code wrote all code'), 'AppE')}; {ref('who')}): "
                f"Claude Code wrote and ran everything under the group leader&rsquo;s direction, AI agents reviewed it, and the team&rsquo;s own checks are "
                f"{state['n_done']} of {state['n']}. Explain {C['group_a'][0]} to {C['group_a'][-1]} aloud once without this sheet, then answer the "
                f"{word(C['n_top'])} questions of page 1."]
    if checks:
        everyone.append("Your own check, by yourself (<code>team_checks/TEAM_CHECK_SHEET.pdf</code>): run your command below on this PC; record it in your own entry of "
                        "<code>team.json</code> only if every line says AGREE, in your own words; nobody else records it for you, and a line that says DIFFER goes "
                        "to the group leader, not into <code>team.json</code>.")
    if timing:
        per = {}
        for _, _, who, sec in timing:
            per[who] = per.get(who, 0) + sec
        each = (f"{next(iter(per.values()))} s for each of the {word(len(per))} speakers" if len(set(per.values())) == 1 else
                ", ".join(f"{QF.esc(w)} {s} s" for w, s in per.items()))
        everyone.append(f"The talk: {word(len(timing))} slides, {sum(s for *_, s in timing)} s in all, {each}. Rehearse it once with a clock before {talk}; the "
                        f"&lsquo;If asked&rsquo; lines of the speaker notes (<code>4_Final_Project_Presentation_notes.md</code>) are not spoken.")
    if C.get("ladder"):
        everyone.append(C["ladder"])
    if S:
        everyone.append(f"Quote the final report, never <code>Flood_Report.pdf</code>, the earlier long report of 8 October 2026: it predates the "
                        f"{word(S['n5'])} seeds and still says {S['aorc'][0]} and &lsquo;{S['hind'][0]}&rsquo;.")
    items = open_items(FR, REG, len(members), M["authors"] if M else "its three authors")
    if items:
        everyone.append("Still open before sending (the group leader sends; everyone should know): " + "; ".join(items) + "; and, before the talk, the rehearsal.")
    everyone.append("After a check is recorded, the group leader rebuilds as <code>README_HANDIN.md</code> of the hand-in folder says (the registry, the report, the "
                    "deck), then this sheet: <code>python -X utf8 code/build_qa.py</code>.")
    blocks = [("Everyone", everyone)]
    extra = C["extra"]                                               # member-specific items, by team key (A, R2, G1, G2)
    for m in members:
        key = KEY_OF.get(m["name"])
        short = {"A": "lead", "R2": "member2", "G1": "member3", "G2": "member4"}.get(key, "")   # the --member key of team_checks/team_check.py
        mine = [(s, title, sec) for s, title, who, sec in timing if who == m["name"]]
        todo = []
        status = team.check_status(key) if key else "still to be done"
        if status != "still to be done":
            todo.append(f"<b>Own check:</b> {QF.esc(status)}.")
        if mine:
            todo.append(f"<b>Your part of the talk:</b> slide{'s' if len(mine) > 1 else ''} {join(str(s) for s, _, _ in mine)} "
                        f"({'; '.join(QF.esc(t) for _, t, _ in mine)}), {sum(sec for *_, sec in mine)} s.")
        c = checks.get(short)
        if c:
            todo.append(f"<b>Your own check</b> ({QF.esc(c['part'])}): <code>python -X utf8 team_checks/team_check.py --member {short}</code>; then open "
                        f"<code>{QF.esc(c['figure'])}</code> and read its one sentence on the check sheet.")
        todo += extra.get(key, [])
        blocks.append((f"{QF.esc(m['name'])} ({QF.esc(m['role'])})", todo))
    return head, lead, blocks
