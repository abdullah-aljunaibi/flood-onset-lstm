"""Flood_Report_3pages_dense.html and Flood_Report_3pages_dense.pdf: the dense version of the report in exactly three A4 pages.

Since 8 October 2026 the hand-in form is the formal paper written by code/build_report_paper.py (Flood_Report_3pages.docx and .pdf); this builder writes the _dense names, so that the two
never overwrite each other.

The full report (Flood_Report.pdf, 31 pages) stays the detailed record; this one keeps what shows the work and its result. Page 1: the question, what was
found, how it was done, and the dataset (what it is, how it was obtained, the flow of the data, how it is computed, the imagery facts: one table, the stages of the flow figure and four short paragraphs, every
fact read from results/data_flow.json, the single source written by code/19_data_flow.py). Page 2: the key figure, the ladder of inputs as two small tables (with the lowest and highest single training and the range when
the test years and basins are drawn again) and four short paragraphs on what the numbers mean. Page 3: what it means, the limits, the work in numbers, who
did what, how to reproduce, the sources.

Nothing is computed here beyond differences, sums, counts and formatting. Every number is read from results/*.json through the shared readers of
report_sections/s0_summary.py (facts, aorc_facts, era5_facts, stage4_facts, hindsight_facts, forecast_facts, rivers_facts, timing_facts, clock_facts,
lobo_facts, day_windows, key_figure, key_caption, who) or by key from the results files (lstm_all, flood_analysis, data_audit, fair_test, uncertainty, the three
ladders, the review records, data_access, aorc_data, stage4_realtime, data_flow). The computing time and the number of trainings come from the reader of section 7 of the
report (s7_cost._project_cost); the roles come from PLAN.md (s8_limits._plan). The only counts taken from the file system are the number and size of the scripts
in code/ (like 'what exists now' in section 9 of the report).

The fixed wordings are imported from s0_summary and used word for word: FINDING and SUPPORT, AORC_WORDING, HINDSIGHT_WORDING (beside every 51%),
REALTIME_WORDING (beside every 48% of Stage IV), INFO_NOT_NET, MOVED_POINT, NSE_FORECAST_WORDING and REVIEW_WORDING (inside who()). The perfect forecast is
always 'an upper bound that cannot be had in practice', AORC always 'an analysis made after the fact', and the numbers of E2b are always marked as another
split. Wording that depends on an outcome is chosen from the stored numbers and readings; where they no longer support the condensed wording, the paragraph
falls back to the sentence of page 1 of the full report and one line says so on stderr.

The honesty lines of the project stand: Claude Code wrote and ran everything under the team lead's direction; the independent reviewers are AI agents, not team
members; the team's own checks are still open. This script reads team.json (through team.py) and never writes it.

Layout: three containers, each one A4 page, each starting a new page, so that the pages are independent. Each ends with an invisible spacer of SAFETY_MM: if a page and its spacer
do not fit, it prints as more than one page. fit_plan() fits the pages one by one with test prints of one page (the largest body size of SIZES at which page 1 fits with its flow figure at
FLOW_FLOOR % or more, page 2 fits with the key figure at FIG_FLOOR % or more and page 3 fits; then each figure as wide as its page allows), so every page has at least SAFETY_MM to spare (about 5 mm on the
last ink), and prints the three pages together; it says on stderr if the result is not three pages. The brief allowed 8.6 to 9.4 pt; pages 1 and 2 decide: page 1 holds the findings and the dataset block (its
flow figure is the two rows of stages of figures/data_flow_talk.png, cropped; the whole figure, 9.5 in tall, cannot be legible in the room), page 2 the key figure, two tables and four paragraphs.
Rebuild after any rerun with: python code/build_report_short.py
"""
import html as _html
import re
import struct
import sys
import tempfile
from pathlib import Path

import team
from common import FIG, ROOT
from flood import LEADS
from report_common import CITED, LEAD_WORD, REFERENCES, cite, count, img, load_results, n2, page, pct, pdf_pages, print_pdf, signed, tag
from report_sections import s0_summary as S0
from report_sections import s7_cost, s8_limits

NAME = "Flood_Report_3pages_dense"                                        # the output: Flood_Report_3pages_dense.html / .pdf (Flood_Report_3pages.docx / .pdf belong to build_report_paper.py)
UPPER = "an upper bound that cannot be had in practice"                   # the fixed wording for every perfect forecast (as in build_overview.py, build_slides.py, build_qa.py)
SAFETY_MM = 4                                                             # every page must end at least this far above the foot of its text area
# What the layout may change, in this order of preference (the brief: type size 8.6 to 9.4 pt, margins, the figure widths, trimmed prose). The three pages are independent (each starts a new
# page), so fit_plan() fits them one by one with test prints of one page: the largest body size in SIZES at which page 1 fits with its flow figure at FLOW_FLOOR % of the text width or more,
# page 2 fits with the key figure at FIG_FLOOR % or more, and page 3 fits; the two figures are then made as wide as their pages allow. A plan is (body size in pt, key figure width in %,
# flow figure width in %).
SIZES = (9.0, 8.8, 8.6)
FIG_WIDTHS = tuple(range(100, 59, -2))                                       # % of the text width, widest first
FIG_FLOOR = 80                                                               # the key figure (page 2) is not made narrower than this: a smaller body size is tried first
FLOW_FLOOR = 80                                                              # nor the flow figure (page 1)
FOOT = "font:7.2pt 'Helvetica Neue',sans-serif;color:#898781"
first, k2, k3 = LEADS
lk = lambda k: f"l{k}"
n1 = lambda v: n2(v, 1)
s1 = lambda v: signed(v, 1)
word = S0._word
slash = lambda d, f=pct: "/".join(f(d[k]) for k in LEADS)                  # 51%/21%/12%: the three lead times in a row
LEADS23 = S0._leads([k2, k3])                                              # '2 and 3 days ahead'


def say(msg):
    """One line on stderr (once per build): the stored numbers no longer support a condensed wording."""
    S0.note(f"build_report_short: {msg}")


def rng(r, share=True):
    """[lowest, highest] -> '49-55%' (shares) or '0.70-0.77' (NSE) with en dashes, one number where both ends round alike (as ladder_cell does); '' without a range."""
    if not r:
        return ""
    lo, hi = (f"{100 * x:.0f}" for x in r) if share else (n2(x) for x in r)
    return (lo if lo == hi else f"{lo}&ndash;{hi}") + ("%" if share else "")


def cap(s):
    return s[0].upper() + s[1:]


# ---- everything the three pages read -----------------------------------------------------------------------------------------
def gather(R, T):
    """Every number and reading the three pages quote, by key; None (with a line on stderr) if a results file the pages need is missing or the finding is not supported."""
    need = ("lstm_all", "flood_analysis", "data_audit", "fair_test", "uncertainty", "rain_to_midnight", "aorc_rain_to_midnight", "forecast_input", "more_basins", "key_figure",
            "rain_timing", "stage4_realtime", "aorc_hindsight", "onset_rain_clock", "engine_review", "data_access", "aorc_data", "day_windows", "data_flow")
    gone = [k for k in need if k not in R or (isinstance(R[k], dict) and R[k].get("quick_mode"))]
    if gone:
        print("build_report_short: results files missing or smoke tests: " + ", ".join(gone), file=sys.stderr)
        return None
    f = S0.facts(R, T)
    G = dict(f=f, R=R, T=T, AO=S0.aorc_facts(R), E1=S0.era5_facts(R), S4=S0.stage4_facts(R), HS=S0.hindsight_facts(R), FC=S0.forecast_facts(R), RV=S0.rivers_facts(R),
             TF=S0.timing_facts(R), C=S0.clock_facts(R), LF=S0.lobo_facts(R), W=S0.day_windows(R), K=S0.key_figure(R))
    absent = [k for k in ("AO", "E1", "S4", "HS", "FC", "RV", "TF", "C", "LF", "W", "K") if not G[k]]
    if absent or not f["story"]:
        print("build_report_short: the stored numbers do not support the finding as worded, or a reader found no file (" + ", ".join(absent or ["story"]) + "); not built",
              file=sys.stderr)
        return None
    A, F = R["lstm_all"], R["flood_analysis"]
    best = A["comparison"]["wq"]["best_baseline_by_validation"][lk(first)]
    cmp1 = A["comparison"]["wq"]["baselines"][best][lk(first)]                          # the LSTM against the best simple model with the gauge, 1 day ahead
    G.update(A=A, D=R["data_audit"], FT=R["fair_test"], UNC=R["uncertainty"], FIN=R["forecast_input"], DF=R["data_flow"], cmp1=cmp1,
             spread=cmp1["lstm_seed_median_max"] - cmp1["lstm_seed_median_min"],
             g1=A["decomposition"]["steps"]["river_gauge_without_forecast"][lk(first)]["nse_median_gain"],
             on0=A["decomposition"]["levels"]["wq"][lk(first)]["onset_pod"], up=A["decomposition"]["levels"]["wqf"][lk(first)]["onset_pod"],
             hind=S0.hindsight_clause(G["HS"]), v_all=F["verdict"]["services"]["lstm_wq"][lk(first)]["all"]["pooled"],
             v_on={F["verdict"]["services"]["lstm_wq"][lk(k)]["onset"]["pooled"] for k in LEADS},
             pers=F["verdict"]["other_models_pooled"].get("base_persistence", {}).get(lk(first), {}).get("all"),
             mixed_rows=R["key_figure"]["figure"]["drawn_words"].get("mixed_rows") or {})
    G["vs"] = set(G["AO"]["verdict_onset"].values()) | set(G["FC"]["verdict_onset"].values()) | set(G["FC"]["verdict_onset_chosen"].values())
    G["share48"] = G["S4"]["stage4"] / G["S4"]["n"]                                    # Stage IV in place of AORC: always printed with REALTIME_WORDING
    G["share45"] = G["S4"]["no_last"] / G["S4"]["n"]
    G["go"] = S0.criteria(R)
    G["verdict_words"] = next(iter(G["vs"])) if len(G["vs"]) == 1 else " / ".join(sorted(G["vs"]))
    return G


def redraw(G, stat, name, k):
    """The 95% range of results/uncertainty.json when the test years and the basins are drawn again together (design 'both'), or None where the file has none."""
    e = G["UNC"]["stats"].get(f"{stat}.{name}.l{k}") if name else None
    b = e.get("both") if e else None
    return [b["lo"], b["hi"]] if b else None


# ---- the two ladders -----------------------------------------------------------------------------------------------------------
def cell(G, v, seeds, name, stat, k, share=True, mixed=False):
    """A table cell: the value (in a box of fixed width, so that the values of a column line up), then in small grey type the lowest-highest single training and the range when years
    and basins are drawn again (the % sign once, at the end), then 'mixed' where the key figure prints it."""
    parts = [x for x in (rng(seeds, share), rng(redraw(G, stat, name, k), share)) if x]
    small = " &middot; ".join(x.removesuffix("%") if share and i < len(parts) - 1 else x for i, x in enumerate(parts))
    big = pct(v) if share else n2(v)
    return (f'<span class="v">{big}</span>' + (f'<span class="sr">{small}</span>' if small else "") + (' <i class="mx">mixed</i>' if mixed else ""))


def tables(G):
    """(Table 1, Table 2): the ladder 1 day ahead and the ladder 2 and 3 days ahead, from the ladders of the results files."""
    R, AO, S4, FC, W = G["R"], G["AO"], G["S4"], G["FC"], G["W"]
    X_A, X_E, X_F = R["aorc_rain_to_midnight"], R["rain_to_midnight"], G["FIN"]
    a = {r["set"]: r["by_lead"] for r in X_A["ladder"]["wq"]["rungs"]}
    e = {r["set"]: r["by_lead"] for r in X_E["ladder"]["wq"]["rungs"]}
    utc = {lk(r["lead"]): dict(onset_warned=r["onset_warned_with"], onset_seed_range=r["onset_seed_range_with"], nse_median=r["nse_with"], nse_seed_range=r["seed_range_nse_with"],
                               onset_false=r["onset_false_with"]) for r in X_A["closure"]["wqa_utc"]["by_lead"]}
    fm = {r["set"]: r["by_lead"] for r in X_F["ladder"]["main"]["rungs"]}
    fs = {r["set"]: r["by_lead"] for r in X_F["ladder"]["sat"]["rungs"]}
    mixed = lambda block, key, k: bool(G["mixed_rows"].get(f"{block}.{key}.l{k}"))        # the rows beside which the key figure prints 'mixed'

    # ---- Table 1: 1 day ahead
    def row1(label, by_lead, name, cls="", prefix="", key=None):
        g = lambda x: by_lead[lk(first)].get(prefix + x)
        return (f'<tr class="{cls}"><td>{label}</td><td>{cell(G, g("onset_warned"), g("onset_seed_range"), name, "onsets_warned", first)}</td>'
                f'<td>{cell(G, g("nse_median"), g("nse_seed_range"), name, "nse_median", first, share=False, mixed=bool(key and mixed("upper", key, first)))}</td>'
                f'<td>{pct(g("onset_false"))}</td></tr>')
    rows1 = [row1("Rain table alone (past weather + river gauge)", a["lstm_wq"], "lstm_wq"),
             row1(f"+ NLDAS rain, day ends about {W['nldas']}", e["lstm_wqn"], "lstm_wqn", key="wqn"),
             row1("+ ERA5 rain to midnight (model rain)", e["lstm_wqh"], "lstm_wqh"),
             row1(f"+ AORC rain to midnight, {S0.AORC_WORDING} ({G['hind']})", a["lstm_wqa"], "lstm_wqa"),
             row1(f"control: AORC over the table&rsquo;s window, {W['daymet']} to {W['daymet']}", a["lstm_wqa"], "lstm_wqa_to09", "ctl", "control_"),
             row1(f"control: AORC over the UTC day, ends about {W['imerg']}", utc, "lstm_wqa_utc", "ctl"),
             (f'<tr><td>Stage IV in place of AORC &Dagger;</td><td><span class="v">{pct(G["share48"])}</span><span class="sr">{pct(G["share45"])} with the last hour out</span></td>'
              f'<td>&ndash;</td><td>&ndash;</td></tr>'),
             (f'<tr><td>Ridge regression / random forest, given the AORC rain</td><td><span class="v w">{pct(AO["warned"]["ridge"])} / {pct(AO["warned"]["forest"])}</span>'
              f'<span class="sr">{rng(redraw(G, "onsets_warned", "base_linreg_wqa", first))} / {rng(redraw(G, "onsets_warned", "base_rf_wqa", first))}</span></td>'
              f'<td><span class="v w">{n2(AO["nse_simple"]["ridge"])} / {n2(AO["nse_simple"]["forest"])}</span></td><td>{pct(AO["far"]["ridge"])} / {pct(AO["far"]["forest"])}</td></tr>'),
             row1("Perfect weather forecast &dagger;", a["lstm_wqf"], "lstm_wqf")]
    head1 = '<tr><th>One day ahead, with the river gauge</th><th>Onsets warned</th><th>Median NSE</th><th>False warnings</th></tr>'
    t1 = (f'<table class="num tl t1"><colgroup><col style="width:42%"><col style="width:23%"><col style="width:26%"><col style="width:9%"></colgroup>{head1}{"".join(rows1)}</table>')

    # ---- Table 2: 2 and 3 days ahead
    leads = (k2, k3)

    def row2(label, by_lead, name, cls=""):
        cells = [cell(G, by_lead[lk(k)]["onset_warned"], by_lead[lk(k)]["onset_seed_range"], name, "onsets_warned", k) for k in leads]
        cells += [cell(G, by_lead[lk(k)]["nse_median"], by_lead[lk(k)]["nse_seed_range"], name, "nse_median", k, share=False) for k in leads]
        return f'<tr class="{cls}"><td>{label}</td>' + "".join(f'<td class="{"g" if i in (0, 2) else ""}">{c}</td>' for i, c in enumerate(cells)) + "</tr>"
    years = lambda s: f"{X_F[s]['test'][0][:4]}&ndash;{X_F[s]['test'][1][:4]}"
    rows2 = [row2("Rain table alone", fm["lstm_wq"], "lstm_wq"),
             row2(f"+ AORC to midnight, {S0.AORC_WORDING}", fm["lstm_wqa"], "lstm_wqa"),
             row2(f"+ GEFS forecast, next {word(len(LEADS))} days, substituted (E2a)", fm["lstm_wqaF_sub"], "lstm_wqaF_sub"),
             row2(f"perfect rain of the next {word(len(LEADS))} days &dagger;", fm["lstm_wqaP"], "lstm_wqaP"),
             row2(f"perfect weather of the next {word(len(LEADS))} days &dagger;", fm["lstm_wqf"], "lstm_wqf"),
             row2("E2b, retrained with it: <b>own split, not comparable</b> &sect;", fs["lstm_wqaF_s"], None, "alt")]
    head2 = ('<tr><th>Two and three days ahead, with the river gauge</th><th>Onsets, 2 days</th><th>Onsets, 3 days</th><th class="g">Median NSE, 2 days</th><th>Median NSE, 3 days</th></tr>')
    ns = lambda d: ", ".join(count(d[k]) for k in LEADS[:-1]) + " and " + count(d[LEADS[-1]])
    gl = f"({cite('gefsv12')[1:-1]}; {cite('guan2022')[1:-1]})"
    t2 = (f'<table class="num tl t2"><colgroup><col style="width:36%"><col style="width:15%"><col style="width:15%"><col style="width:17%"><col style="width:17%"></colgroup>{head2}{"".join(rows2)}</table>')
    unc = G["UNC"]["resampling"]
    notes = (f'<p class="cap"><b>Tables 1 and 2.</b> Large: average of {word(G["f"]["n_seeds"])} trainings. Small grey: lowest&ndash;highest single training '
             f'&middot; 95% range when the {unc["n_years"]} test years and the {G["f"]["n"]} basins are drawn again together ({count(unc["B"])} times). False warnings: share of the onset '
             f'warnings that were false. &dagger; {cap(UPPER)}. &Dagger; {cap(S0.REALTIME_WORDING)}. &sect; <b>Another split</b>: test years {years("split_satellite_era")}, {ns(FC["n_sat"])} onsets, a weaker base, '
             f'no second range ({LEAD_WORD[first]}: {pct(FC["s_fc"][first])}). Table 2: the GEFS forecast {gl} is put into a network trained on the AORC rain of the days being forecast; {S0.nse_forecast_words(FC)}.</p>')
    return t1, t2 + notes


# ---- page 1 ----------------------------------------------------------------------------------------------------------------------
def findings(G):
    """The three findings of page 1 of the full report, condensed: [(heading, text)]. Where the stored numbers do not support the condensed wording, the full text."""
    f, AO, S4, FC, TF, LF, RV = G["f"], G["AO"], G["S4"], G["FC"], G["TF"], G["LF"], G["RV"]
    full = f["finds"]
    cls, n = f["cls"], f["n"]
    # 1 the kind of model
    nums = f"{n2(G['cmp1']['lstm_nse_median'])} against {n2(G['cmp1']['baseline_nse_median'])}"
    if all(v == "tie" for v in cls["wq"].values()) and f["acc"] == "no clear difference in accuracy":
        t1 = (f"With past weather + river gauge the LSTM&rsquo;s median NSE is {nums} for the best simple model: {S0.reading_word('no clear difference')} at any of the {word(len(LEADS))} lead times, so milestone 2 of the plan, "
              f"&lsquo;the LSTM beats the baselines&rsquo;, is <b>{f['m2'].split(' (')[0]}</b>. A Transformer in place of the LSTM: {f['acc']}, at {f['t_lo']:.0f} to {f['t_hi']:.0f} times the training time.")
    else:
        say("the readings of the model comparison changed: finding 1 is the full text of page 1 of the report")
        t1 = full[0][1]
    # 2 the information
    r = TF["gauge"][first]
    read = S0.nse_reading(r, TF["n"])
    nse_bit = {"clear": f"in NSE the gain is clear as well (higher in {r['basins_better']} of {TF['n']} basins)", "mixed": f"in NSE that result is mixed, higher in only {r['basins_better']} of {TF['n']} basins",
               "none": "in NSE there is no clear difference"}[read]
    evening = (f"Rain stamped up to the evening of the issue day closes {S0.share_word(r['onset_share_of_gap_closed'])} of that gap ({pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])}; "
               f"control {pct(r['onset_warned_control'])}; {nse_bit})" if S0.onset_clear(r) else
               f"Rain stamped up to the evening of the issue day takes them from {pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} (control {pct(r['onset_warned_control'])}), but the single seeds "
               f"overlap: no clear gain; {nse_bit}")
    t2 = (f"The river gauge adds {signed(G['g1'])} NSE to the LSTM. A perfect weather forecast, {UPPER}, would lift the onsets warned from {pct(G['on0'])} to {pct(G['up'])}. {evening}; with the rain of the whole issue day up to "
          f"midnight from AORC, {S0.AORC_WORDING}, {pct(AO['with_'][first])} of the onsets are warned {LEAD_WORD[first]} (control {pct(AO['to09'])}; {G['hind']}; {pct(G['share48'])} with Stage IV: "
          f"{S0.REALTIME_WORDING}), and with a public weather forecast beside it (one ensemble member) {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} {LEADS23}, against {pct(FC['base'][k2])} and "
          f"{pct(FC['base'][k3])} with the rain table alone.")
    # 3 the warnings
    v_on, vs = G["v_on"], G["vs"]
    same = f", the same verdict as {S0.QUOTED}" if G["pers"] == G["v_all"] else ""
    on3 = (f"{tag(next(iter(v_on)))} for onsets, the warnings that matter, at every lead time" if len(v_on) == 1 else
           f"for onsets, the warnings that matter, {' / '.join(sorted(v_on))} at the {word(len(LEADS))} lead times")
    false = AO["false"][first]
    false_txt = f"{S0.share_word(false)} of the warnings false" + (f", {pct(false)}," if S0.share_word(false) != pct(false) else "")
    stay = f"the verdict for onsets {'stays' if vs == {'NOT YET'} else 'is'} {G['verdict_words']} with every new input ({false_txt} with the rain up to midnight)"
    lobo = (f"A river never seen in training loses {n2(LF['loss'])} NSE {LEAD_WORD[first]} (median {n2(LF['seen'])} to {n2(LF['unseen'])}, lower in {LF['n_lower']} of {LF['n']} basins); with "
            f"{RV['n_train']['unseen50_w_static']} training rivers instead of {RV['n_train']['lobo_w_static']} the loss shrinks to {n2(RV['loss'][1])} and the median rises from "
            f"{n2(RV['unseen']['lobo_w_static'][first])} to {n2(RV['unseen']['unseen50_w_static'][first])} (number, variety and nearness of the training rivers not separated).")
    t3 = (f"By criteria fixed before the analysis: {tag(G['v_all'])} for all high-flow days {LEAD_WORD[first]} with the river gauge{same}; {on3}: {LEAD_WORD[first]} {pct(G['on0'])} of the "
          f"{count(f['n_on'])} onsets are warned; {stay}. {lobo}")
    return [(full[0][0], t1), (full[1][0], t2), (full[2][0], t3)]


# ---- the dataset block of page 1: one source, results/data_flow.json (written by code/19_data_flow.py) -----------------------------------------
def df_item(DF, key):
    """A dataset of results/data_flow.json (or one of the parts of CAMELS-US) by its key."""
    for e in DF["datasets"]:
        if e["key"] == key:
            return e
        for part in e.get("parts") or []:
            if part["key"] == key:
                return part
    raise KeyError(key)


def host_of(text):
    """The host of the first address in a text ('https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/...' -> 'noaa-gefs-retrospective.s3.amazonaws.com')."""
    m = re.search(r"https?://([^/\s:?;]+)", str(text))
    return m.group(1) if m else ""


def size_of(nbytes):
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.0f} MB"


def login_of(text):
    """What the record says about an account, in two or three words: 'no login', 'no key', 'no account recorded'."""
    t = str(text)
    return "no account recorded" if t.startswith("no account recorded") else "no key" if t.startswith("no key") else "no login" if t.startswith("no") else t


def breakable(host):
    """A host with a line break allowed after each dot."""
    return host.replace(".", ".<wbr>")


def span_of(d):
    """'159&ndash;2,649' from the smallest and the largest value of a dict."""
    v = list(d.values())
    return f"{count(min(v))}&ndash;{count(max(v))}"


def dataset_rows(G):
    """The rows of the table of the data products that models read and the extra inputs of the experiments: [(role, [cells])]; a cell may be (text, rowspan). Every number is read from
    results/data_flow.json (the strings of the datasets, all_information, how_computed and flow) or from the readers of s0_summary."""
    DF, FC = G["DF"], G["FC"]
    ai, hc = DF["all_information"], DF["how_computed"]
    it = lambda key: df_item(DF, key)
    fl = lambda key: it(key)["figure_line"]                                   # [name, 'grid · step · period']
    ob = lambda key: it(key)["obtained"]
    mark = lambda key: "" if "recorded in a results file" in str(ob(key).get("bytes_read_kind", "")) else "*"        # a size marked * is that of the stored files: the transfer size was not recorded
    got = lambda key: f"{breakable(host_of(ob(key)['url_pattern']))} &middot; {size_of(ob(key)['bytes_read'])}{mark(key)} &middot; {login_of(ob(key)['login_needed'])}"
    ends = hc["where_each_daily_value_ends"]["values"]
    rec, records = ai["hourly_and_other_records"], ai["records"]
    flowclock = ai["clocks"]["river_flow_day"]["standard_time_in_force"]
    n_gauges = len(ai["basin_table"])
    lag0 = ai["clocks"]["aorc_stamp_check_correlation_by_lag"]["0"]
    km = hc["value_at_the_centroid"]["worked"]["era5_point_km_from_centroid"].values()
    f_lo, f_hi = hc["gefs_bias_factor_and_issue_time"]["worked"]["factor_range"]
    clim = ai["static_properties"]["climate_properties"]
    c_from, c_to = (x[:4] for x in ai["static_properties"]["climate_averaging_period"])
    st = {x["id"]: x["counts"] for x in DF["flow"]["stages"]}
    cam = it("camels_us")
    q = hc["flow_mm_per_day"]["formula"].replace(" x ", " &times; ").replace("area_m2", "area_m&sup2;")
    est = f"{pct(records['estimated_share_per_basin_min'], 1)} to {pct(records['estimated_share_per_basin_max'], 1)}"
    terms = "the page asks the downloader to accept its terms of use" in cam["obtained"]["login_needed"]
    zips = any("address the zip files were downloaded from was not recorded" in u for u in cam["unverified"])
    camels_got = (f"data set page {breakable(host_of(cam['obtained']['url_pattern']))}{' (the address of the zip files was not recorded)' if zips else ''} &middot; {size_of(cam['obtained']['bytes_read'])}* &middot; by hand; "
                  f"{login_of(cam['obtained']['login_needed'])}{', the page asks to accept its terms of use' if terms else ''}", 2)
    return [
        ("every", [f"<b>{fl('camels_flow')[0]}</b> (CAMELS-US): {est} of the days estimated, {records['missing_values']} missing &middot; {fl('camels_flow')[1]}",
                   f"local midnight to midnight ({flowclock['n_gauges_best_midnight_standard_time']} of {n_gauges} gauges fit best in standard time)", camels_got, q]),
        ("every", [f"<b>{fl('camels_daymet')[0]}</b> (stations interpolated): {fl('camels_daymet')[1]}; <b>{fl('camels_nldas')[0]}</b>: {fl('camels_nldas')[1]}; <b>{st['inputs']['properties']} basin properties</b>, fixed",
                   f"Daymet rain day ends about {ends['daymet']['rounded']} (median basin {ends['daymet']['median']}); NLDAS-2 about {ends['nldas']['rounded']} (median {ends['nldas']['median']})", None,
                   f"the {st['inputs']['weather_columns']} Daymet columns as shipped; NLDAS-2 rain as an extra column (control: one day older); {len(clim)} of the {st['inputs']['properties']} properties are climate averages of {c_from}&ndash;{c_to}"]),
        ("extra", [f"<b>{fl('aorc')[0]}</b>, {S0.AORC_WORDING}, anchored on rain gauges and radar: {fl('aorc')[1]}", f"hour-ending UTC (r = {n2(lag0)} against Stage IV at lag 0)", got("aorc"),
                   f"mean of the {span_of(rec['aorc_cells_per_basin'])} cells inside the outline; local-day sums: rain to midnight, controls"]),
        ("extra", [f"<b>{fl('era5')[0]}</b>, weather-model reanalysis: {fl('era5')[1]}", "hour-ending UTC (rain of the preceding hour)", got("era5"),
                   f"the point {min(km):.0f}&ndash;{max(km):.0f} km from the centroid; local-day sums: rain to midnight (E1), controls"]),
        ("extra", [f"<b>{fl('gefs')[0]}</b>, weather-model rain forecasts re-run for past dates: {fl('gefs')[1]}, control member",
                   f"UTC; the run of date t+1, {FC['hours_old'][0]} to {FC['hours_old'][1]} hours old at the issue time", got("gefs"),
                   f"area-weighted mean of {span_of(rec['gefs_boxes_per_basin'])} boxes; local days t+1&hellip;t+3; bias factor {n2(f_lo)}&ndash;{n2(f_hi)}"]),
        ("extra", [f"<b>{fl('imerg')[0].replace(' (POWER)', '')}</b>, satellite rain: {fl('imerg')[1]}; <b>{fl('merra2')[0].replace(' (POWER)', '')}</b>, model soil wetness, vapour: {fl('merra2')[1]}",
                   f"the UTC day (ends {ends['imerg']['documented'].split(' ')[0]} local standard time)",
                   f"{breakable(host_of(ob('imerg')['url_pattern']))} &middot; {size_of(ob('imerg')['bytes_read'])}* + {size_of(ob('merra2')['bytes_read'])}* &middot; {login_of(ob('imerg')['login_needed'])}",
                   f"area-weighted mean of {span_of(rec['imerg_cells_per_basin'])} cells (IMERG); the centroid cell (MERRA-2)"]),
    ]


def dataset_table(G):
    head = "<tr><th>Product, what it is &middot; grid &middot; step &middot; period</th><th>Time stamp, as measured</th><th>Obtained: host &middot; size &middot; login</th><th>Becomes the input</th></tr>"
    body = ""
    for role, cells in dataset_rows(G):
        tds = ""
        for c in cells:
            if c is None:
                continue                                                  # covered by the rowspan of the cell above
            text, span = c if isinstance(c, tuple) else (c, 1)
            tds += f'<td{f" rowspan={span}" if span > 1 else ""}>{text}</td>'
        body += f'<tr class="r-{role}">{tds}</tr>'
    return f'<table class="dt"><colgroup><col style="width:31%"><col style="width:19%"><col style="width:20%"><col style="width:30%"></colgroup>{head}{body}</table>'


def crop_picture(name, width, top=0.0, bottom=0.0):
    """A figure of figures/ at `width` % of the text width, shown without its top `top` and bottom `bottom` fractions (the file is not changed; as s0_summary.fig_cropped, but
    the caption stands outside)."""
    with open(FIG / name, "rb") as fh:
        w, h = struct.unpack(">II", fh.read(26)[16:24])
    keep = 1 - top - bottom
    inner = img(name).replace("<img ", f'<img style="position:absolute;left:0;width:100%;top:-{100 * top / keep:.3f}%" ', 1)
    return (f'<div style="position:relative;overflow:hidden;width:{width}%;margin:0 auto;padding-bottom:{width * keep * h / w:.3f}%">{inner}</div>')


# The ten stages of figures/data_flow_talk.png (2640 x 1231 px): its two rows of boxes. Left out: the band of products above (content rows 96 to 410) and the band of the extra inputs between the
# rows (715 to 905); the table has both. The arrow from stage 5 to stage 6 (x = 2390) is cut with it: its head is kept above row 2.
FLOW_FIG, FLOW_H = "data_flow_talk.png", 1231
FLOW_ROWS = ((414, 654), (950, 1212))                                        # first and last pixel row of the two crops


def flow_picture(width):
    return "".join(crop_picture(FLOW_FIG, width, top=a / FLOW_H, bottom=(FLOW_H - b) / FLOW_H) for a, b in FLOW_ROWS)


def flow_clauses(G):
    """One clause per stage of the flow, with its counts: the 'title', 'count' and 'detail' strings of the drawing of results/data_flow.json (figure.report.stages), whose own number audit
    checks them. The detail is given where it says how a stage works (the audit, the split, the scaling and the high-flow level); the leak test of the split is added from results/fair_test.json."""
    FT = G["FT"]
    z = lambda v: n2(0.0 if abs(v) < 0.0005 else v, 3)                                        # 0.797, -0.002, 0.000 (no '-0.000')
    leak = (f"; leak test on made-up rivers: {', '.join(n2(v) for v in FT['expected']['past_weather_only'])} is the most an honest model can reach, scored "
            f"{', '.join(z(v) for v in FT['measured']['past_weather_only'])} ({word(len(FT['checks']))} checks {'passed' if FT['passed'] else 'NOT all passed'})")
    out = []
    for s in G["DF"]["figure"]["report"]["stages"]:
        out.append(f"<b>{s['n']}</b> {s['title']}: {s['count']}" + (f" ({s['detail']}{leak if s['n'] == 5 else ''})" if s["n"] in (2, 5, 6) else ""))
    return "; ".join(out) + "."


def how_computed(G):
    """'How computed': the flow in mm/day, the basin mean with overlap-area weights, the sum over the local wall-clock day and where the Daymet day ends, from results/data_flow.json (how_computed)."""
    hc = G["DF"]["how_computed"]
    q = hc["flow_mm_per_day"]["formula"].replace(" x ", " &times; ").replace("area_m2", "area_m&sup2;")
    chk = re.search(r"all ([\d,]+) basin-days: the largest difference from the cache is (\d+)", hc["flow_mm_per_day"]["check"])
    w = hc["basin_mean_with_overlap_weights"]["formula"].split(", with ")[0].replace("sum_i (w_i x_i) / sum_i w_i", "&Sigma; w_i x_i / &Sigma; w_i")
    w_def = hc["basin_mean_with_overlap_weights"]["formula"].split(", with ")[1].split(" (")[0]
    day = hc["local_day_sum"]["formula"].replace("sum of", "the sum of")
    dm = hc["where_each_daily_value_ends"]["values"]["daymet"]
    return (f"<b>How computed.</b> Flow in mm/day: {q}" + (f" (recomputed for all {chk.group(1)} basin-days: largest difference {chk.group(2)})" if chk else "") + f". Basin mean: {w}, {w_def} "
            f"(IMERG, GEFS), or the plain mean of the cells inside the outline (AORC, Stage IV). Local day: {day}, by the gauge&rsquo;s wall clock, summer time included. Measured against hourly rain, "
            f"the Daymet rain day ends about {dm['rounded']} (median basin {dm['median']}; basins {dm['basin_range'][0]} to {dm['basin_range'][1]}).")


def dataset_block(G, plan):
    DF = G["DF"]
    cam = df_item(DF, "camels_us")
    st = {x["id"]: x["counts"] for x in DF["flow"]["stages"]}
    legend = ('<span class="leg"><span class="k every"></span>read by every model <span class="k extra"></span>extra input of an experiment</span>')
    days = DF["all_information"]["download_dates"]["days"]
    kind = cam["kind"].split(":")[0].replace("river basins", f"{st['audit']['all_camels']:,} river basins")          # 'a public data set of 671 river basins'
    intro = (f'<p class="ds" id="b-intro"><b>CAMELS-US</b> {cite("newman2015")} is {kind} (a <i>basin</i> is the land whose rain drains to one river gauge); the project uses {st["audit"]["chosen"]} of them, '
             f'each with {st["inputs"]["days_per_basin"]:,} days, and adds products of other sources as extra inputs.</p>')
    checks = [d["short"] + (f" ({size_of(d['obtained']['bytes_read'])})" if d["key"] == "stage4" else "") for d in DF["datasets"] if d["role_code"] == "check or map"]
    caption = (f'<p class="ds" id="b-cap">Obtained {days[0]} (by hand) and {", ".join(days[1:-1])} and {days[-1]} (by script); no payment, no account recorded; * size of the stored files (the transfer size was not recorded). '
               f'{word(len(checks)).capitalize()} more products are checks or maps, not model inputs: {", ".join(checks)}. '
               f'{len(DF["unverified"])} facts are not verified, all listed in <code>results/data_flow.json</code>; here: the Daymet version inside CAMELS, the IMERG release.</p>')
    return (f'<h2 id="b-h2">The dataset: what, how obtained, how computed {legend}</h2>{intro}<div id="b-table">{dataset_table(G)}</div>{caption}'
            f'<figure class="flow" id="b-fig">{flow_picture(plan[2])}<figcaption id="b-flow"><b>The flow of the data</b> (stages of <code>figures/data_flow_talk.png</code>; the whole figure: <code>figures/data_flow.png</code>): '
            f'{flow_clauses(G)}</figcaption></figure>'
            f'<p class="ds" id="b-how">{how_computed(G)}</p>'
            f'<p class="ds" id="b-img"><b>Imagery.</b> {_html.escape(DF["imagery_note"], quote=False)}</p>')


def methods_para(G):
    """'How it was done' in one short paragraph: the model, the simple models and the words of the scores (the data, the split and the leak test are in the dataset block)."""
    f, A = G["f"], G["A"]
    cfg = A["config"]
    return (f"<p class=\"ds how\" id=\"b-methods\"><b>How it was done.</b> One <i>LSTM</i>, a neural network that reads the days in order and keeps a memory {cite('kratzert2019')}, for all {word(f['n'])} rivers together "
            f"({cfg['hidden']} memory cells, {word(f['n_seeds'])} trainings averaged), beside &lsquo;tomorrow = today&rsquo;, ridge regression and a random forest, the simple models to beat. "
            f"<i>NSE</i>: 1 is a perfect forecast, 0 is no better than always forecasting the average flow. <i>High-flow day</i>: flow above the level exceeded on {pct(1 - S0.HIGH_Q)} of the days of the training years. "
            f"<i>Onset</i>: a high-flow day forecast while the river was still below that level; the warnings that matter. {S0.criteria_line(G['R'])}</p>")


def page1(G, plan):
    f = G["f"]
    (h1, t1), (h2, t2), (h3, t3) = findings(G)
    course = "".join(f"{S0._esc(x)} &middot; " for x in team.course_line("\n").split("\n") if x)
    byline = (f"{course}three-page report &middot; built on {f['built']} &middot; team: {S0._esc(team.names_line())} &middot; coding assistant: Claude Code")
    return f"""<div class="pg" id="p1">
<h1 id="b-head">Predicting river floods with AI</h1>
<p class="sub">{byline}</p>
<div class="verdict" id="b-box"><p><b>The question.</b> {S0.QUESTION}</p>
<p><b>What we found.</b> <b>{S0.FINDING}.</b> {S0.SUPPORT}.</p>
<p><b>1. {h1}</b> {t1}</p>
<p><b>2. {h2}</b> {t2}</p>
<p><b>3. {h3}</b> {t3}</p></div>
{methods_para(G)}
{dataset_block(G, plan)}
<div class="safety"></div></div>"""


# ---- page 2 ----------------------------------------------------------------------------------------------------------------------
def key_caption(G):
    """The caption of the plain key figure: the shared caption of page 1 of the full report (read from results/key_figure.json), without its criteria line (page 1 of this report
    has it), with the real-time check and AORC in their fixed wordings and the sections pointing to the full report."""
    c = S0.key_caption(G["R"], G["K"], plain=True, short=True, terms=False)
    cut = c.find(" <i>Verdicts</i>")
    if cut < 0:
        say("the caption of the key figure has no criteria line to cut; it is printed whole")
    else:
        c = c[:cut]
    old = "Hollow diamonds: Stage IV radar, a real-time product, in place of AORC (right), without its last hour (left)."
    new = f"Hollow diamonds, Stage IV radar (right; without its last hour, left): {S0.REALTIME_WORDING}."
    if old in c:
        c = c.replace(old, new)
    else:
        say("the caption of the key figure no longer holds the sentence on the hollow diamonds; the fixed wording is added at the end")
        c += " " + new
    c = c.replace(", section 7)", ")").replace(" (section 4.2)", "")
    return c + f" AORC is {S0.AORC_WORDING} ({cite('aorc')[1:-1]}; {G['hind']})."


def paragraphs2(G):
    """The four short paragraphs of page 2: the hours, hindsight and the real-time check, the forecast, more rivers."""
    f, AO, E1, S4, HS, FC, RV, C, W = (G[k] for k in ("f", "AO", "E1", "S4", "HS", "FC", "RV", "C", "W"))
    D9, U19 = W["daymet"], W["imerg"]
    # what the hours are worth
    flat = " (no gain)" if AO["control_flat"] else " (the control is not flat)"
    hours = (f"<p><b>What the hours are worth.</b> The rain table&rsquo;s day ends at about {D9}, so {pct(C['hidden'])} of the rain around the onsets, which falls between then and midnight of the issue day, "
             f"is hidden from the model. With the same AORC rain over matched windows the model warns of {pct(AO['to09'])} of the onsets when the rain ends at the table&rsquo;s own {D9}{flat}, of {pct(AO['utc'])} "
             f"at about {U19} (the UTC day) and of {pct(AO['with_'][first])} at midnight ({G['hind']}): the hours after {D9} are worth {n1(AO['hours'])} points, {n1(AO['evening'])} of them after {U19}. "
             f"Product quality counts too: over about the same window NLDAS warns of {pct(E1['nldas'])}, ERA5 (model rain) of {pct(E1['utc'])}.</p>")
    # hindsight and the real-time check
    lo, hi = S4["interval"]
    real = (f"<p><b>Hindsight and the real-time check.</b> AORC is {S0.AORC_WORDING}, so its {pct(AO['with_'][first])} could hold hindsight. The independent review bounds it: {G['hind']} "
            f"({count(HS['bound_hits'])} of the {count(AO['hits'])} hits). The decisive test is {S0.REALTIME_WORDING}: with Stage IV radar rain {cite('stage4')} on the issue day of all {count(S4['n'])} onsets the saved "
            f"networks warn of {count(S4['stage4'])} ({pct(G['share48'])}) against {count(S4['aorc'])} with AORC and {count(S4['table'])} with the rain table alone, so {pct(S4['share'])} of the AORC effect survives "
            f"({pct(lo)} to {pct(hi)}, 95% interval); {count(S4['no_last'])} ({pct(G['share45'])}) with the last hour of the issue day left out, as the delay of Stage IV would.</p>")
    # the forecast
    moved = (f"{cap(LEAD_WORD[first])} it warns of fewer onsets ({s1(FC['gain'][first])} points, to {pct(FC['fc'][first])}) with far fewer false warnings ({pct(FC['false_fc'][first])} against "
             f"{pct(FC['false_rain'][first])}): {S0.MOVED_POINT}, said of the warnings only. " if FC["moved"] else
             f"{cap(LEAD_WORD[first])} it warns of {s1(FC['gain'][first])} points of onsets ({S0.reading_word(FC['gain_reading'][first])}). ")
    harm = all(FC["ctrl"][k] < FC["rain"][k] for k in LEADS)
    clean = all(abs(FC["s_ctrl"][k] - FC["s_rain"][k]) < 0.02 for k in LEADS)
    if harm and clean:
        control = (f"The shuffled control measures harm in E2a ({slash(FC['ctrl'])}) and is clean in E2b ({slash(FC['s_ctrl'])} against {slash(FC['s_rain'])} without the forecast; retraining with the "
                   f"forecast gives {slash(FC['s_fc'])} on <b>another split</b>).")
    else:
        say("the shuffled controls of E2a and E2b no longer read as harm and clean; the sentence gives the numbers only")
        control = f"The shuffled controls warn of {slash(FC['ctrl'])} (E2a) and {slash(FC['s_ctrl'])} (E2b, another split)."
    corr = ", ".join(n2(FC["corr"][k]) for k in LEADS[:-1]) + " and " + n2(FC["corr"][LEADS[-1]])
    forecast = (f"<p><b>The forecast.</b> One real forecast was tested, the control member (one of {word(FC['members'])}) of the GEFSv12 reforecast {cite('guan2022')}: its rain correlates {corr} with AORC daily rain "
                f"at {S0._and(LEADS)} days. Put into the AORC-trained network (E2a) it lifts the onsets warned {LEADS23} from {pct(FC['rain'][k2])} and {pct(FC['rain'][k3])} to {pct(FC['fc'][k2])} and "
                f"{pct(FC['fc'][k3])}; {S0.NSE_FORECAST_WORDING}. {moved}{control}</p>")
    # more rivers
    RVn = RV["n_train"]
    rivers = (f"<p><b>More rivers.</b> {RVn['lstm63_w']} training rivers instead of {RVn['lstm_w']} give, for the same {word(f['n'])} gauged rivers, {S0.by_lead_reading(RV['nse_reading'])} in median NSE "
              f"({' / '.join(n2(RV['nse']['lstm_wq'][k]) + ' to ' + n2(RV['nse']['lstm63_wq'][k]) for k in LEADS)}). For rivers never seen, {RVn['unseen50_w_static']} training rivers instead of "
              f"{RVn['lobo_w_static']} lift the median NSE {LEAD_WORD[first]} from {n2(RV['unseen']['lobo_w_static'][first])} to {n2(RV['unseen']['unseen50_w_static'][first])} with basin properties "
              f"({n2(RV['unseen']['lobo_w_nostatic'][first])} to {n2(RV['unseen']['unseen50_w_nostatic'][first])} without), but the number, variety and nearness of the training rivers changed together and were "
              f"not separated, so the gain is credited neither to scale nor to proximity alone.</p>")
    return hours + real + forecast + rivers


def page2(G, plan):
    t1, t2 = tables(G)
    picture = img("key_figure_plain.png").replace("<img ", f'<img style="width:{plan[1]}%;display:block;margin:0 auto" ', 1)      # the picture at plan[1]% of the text width, its caption at the full width
    figure = f'<figure class="key">{picture}<figcaption>{key_caption(G)}</figcaption></figure>'
    return f"""<div class="pg" id="p2">
{figure}
{t1}
{t2}
{paragraphs2(G)}
<div class="safety"></div></div>"""


# ---- page 3 ----------------------------------------------------------------------------------------------------------------------
def reviews(R):
    """(records, rounds, listed checks, failed, numbers that differed) summed over every results/review_*.json, the rounds counted as review_clause counts them (the top-level counts and
    every round_N block with counts)."""
    recs = {k: v for k, v in R.items() if k.startswith("review_") and isinstance(v, dict)}
    rounds = checks = failed = differed = 0
    for rec in recs.values():
        cs = [rec.get("counts") or {}] + [v["counts"] for k, v in rec.items() if k.startswith("round_") and isinstance(v, dict) and isinstance(v.get("counts"), dict)]
        rounds += len(cs)
        checks += sum(c.get("checks_listed", 0) for c in cs)
        failed += sum(c.get("failed", 0) for c in cs)
        differed += sum(c.get("numerical_discrepancies", 0) for c in cs)
    return len(recs), rounds, checks, failed, differed


def scripts():
    """(own files, own lines, files by the numbered experiment scripts, reviewers' files, reviewers' lines): the Python files of code/ (not the archive), by counting."""
    code = ROOT / "code"
    size = lambda ps: sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in ps)
    own = [p for p in list(code.glob("*.py")) + list((code / "report_sections").glob("*.py")) + list((code / "guide_sections").glob("*.py")) if p.stat().st_size > 0]
    rev = [p for p in (code / "review").rglob("*.py") if p.stat().st_size > 0]
    return len(own), size(own), sum(bool(re.match(r"\d\d_", p.name)) for p in own), len(rev), size(rev)


def work_table(G):
    R = G["R"]
    found = s7_cost._project_cost(R, G["T"])
    hours = sum(s for _, _, s, _ in found) / 3600
    n_train = sum(n for _, _, _, n in found if n)
    uncounted = [name for name, _, _, n in found if n is None and name not in s7_cost.NOT_NETWORKS]
    dl = next(x for x in G["DF"]["flow"]["stages"] if x["id"] == "download")["counts"]                                   # the totals of page 1 (stage 1 of the flow of the data)
    per = G["DF"]["all_information"]["bytes_read"]["per_product"]
    gb = {k: per[k] / 1e9 for k in ("aorc", "gefs", "stage4")}
    n_rec, rounds, checks, failed, differed = reviews(R)
    ER = R["engine_review"]
    n_own, l_own, n_num, n_rev, l_rev = scripts()
    bought = "nothing was bought and no account was created" in str(R["data_access"].get("what", "")).lower()
    if not bought:
        say("results/data_access.json no longer says that nothing was bought: the row 'money spent' makes no claim")
    pairs = [("Scripts written", f"<b>{n_own} Python files, {count(l_own)} lines</b>, by Claude Code (the engine, {n_num} numbered scripts, checks, downloads, builders); "
                                f"{n_rev} more files ({count(l_rev)} lines) by the independent reviewers, to recompute the numbers"),
             ("Trainings run", f"<b>{'at least ' if uncounted else ''}{n_train} networks</b>, counted in {word(sum(1 for _, _, _, n in found if n))} results files"),
             ("Downloads made", f"<b>{dl['gb_read']:.0f} GB read</b> in {dl['products']} products (the transfer recorded for {n1(dl['gb_read_recorded_transfers'])} GB): AORC {n1(gb['aorc'])} GB, the GEFSv12 reforecast {gb['gefs']:.0f} GB, Stage IV {gb['stage4']:.0f} GB"),
             ("Independent reviews", f"<b>{rounds} rounds, {count(checks)} listed checks</b>: {failed} failed, {differed} numbers differed (AI agents, not team members); the shared engine: "
                                     f"{len(ER['reviews'])} reviewers, {ER['counts']['checks_passed']} checks passed"
                                     + (f", {ER['counts']['major']} major findings, all fixed" if ER.get("all_fixed") and ER["counts"]["major"] else "")),
             ("Laptop hours", f"<b>{n1(hours)} hours</b> of training on a two-core laptop, no graphics card"),
             ("Money spent", "<b>none</b>" + (": nothing was bought and no account was created (<code>results/data_access.json</code>)" if bought else ""))]
    rows = ""
    for i in range(0, len(pairs), 2):
        (a, b), (c, d) = pairs[i], pairs[i + 1]
        rows += f"<tr><th>{a}</th><td>{b}</td><th>{c}</th><td>{d}</td></tr>"
    return f'<table class="work"><colgroup><col style="width:9%"><col style="width:41%"><col style="width:9%"><col style="width:41%"></colgroup>{rows}</table>'


def limits(G):
    f, A, AO, S4, HS, FC, W = G["f"], G["A"], G["AO"], G["S4"], G["HS"], G["FC"], G["W"]
    n_test_years = G["UNC"]["resampling"]["n_years"]
    ar = ", and whether its archive holds first-posted or later-corrected hours is not established" if S4["archive_unestablished"] else ""
    margin = (f"the forecast run is {FC['hours_old'][0]} to {FC['hours_old'][1]} hours old at the issue time, but its last files appear about {abs(round(FC['late']))} minutes after the Eastern summer issue time "
              f"(a margin read from {word(FC['margin_days'])} days of posting times of the operational member, not from the archive)" if FC["late"] < 0 else
              "the forecast run exists before the issue time (a margin read from the posting times of the operational member, not from the archive)")
    yr = re.search(r"hours before (\d{4}), which are all the training years", str(G["R"]["aorc_rain_to_midnight"].get("outcome_caveat", "")))      # the year from which AORC holds radar hours
    if yr:
        before = f"its hours before {yr.group(1)}, all the training years, are disaggregated daily totals"
    else:
        say("results/aorc_rain_to_midnight.json no longer says from which year AORC holds radar hours: the limit says only that the training years hold disaggregated daily totals")
        before = "its hours in the training years are disaggregated daily totals"
    pk = G["R"]["flood_analysis"]["failures"]["peak_size"]
    peaks = (f"<b>Peaks are forecast too low:</b> the median event is forecast at {pct(pk['lstm_wq']['all_events']['median'])} of the observed peak the day before ({pct(pk['lstm_wqf']['all_events']['median'])} "
             f"with the perfect weather forecast, {UPPER})." if "lstm_wq" in pk and "lstm_wqf" in pk else "")
    hf = G["R"]["baselines"]["high_flow"]
    share = sum(v["share_test_days_above"] for v in hf.values()) / len(hf)
    levels = (f"<b>Fewer high-flow days in the test years</b> ({pct(share, 1)} of the days, not {pct(1 - S0.HIGH_Q)}); the level is a statistic of each river, not an official flood stage.")
    items = [f"<b>{cap(word(f['n']))} basins, {word(n_test_years)} test years, {word(f['n_seeds'])} trainings.</b> The same rivers are in all three periods; every median rests on {f['n']} basins; "
             f"{word(f['n_seeds'])} trainings of one LSTM differ by {n2(G['spread'])} NSE {LEAD_WORD[first]}, as much as most differences between models; the ranges of page 2 do not cover other regions.",
             f"<b>AORC is {S0.AORC_WORDING}:</b> {before}; its {pct(AO['with_'][first])} carries {G['hind']} ({count(HS['bound_hits'])} of the "
             f"{count(AO['hits'])} hits); the Stage IV check is not a service{ar}.",
             f"<b>One ensemble member.</b> The control member of {word(FC['members'])}: where a real system lands is known only for that forecast ({pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} against "
             f"{pct(FC['perfect'][k2])} and {pct(FC['perfect'][k3])} for perfect rain, {UPPER}); E2b is another split with a weaker base.",
             f"<b>The issue-time margin:</b> {margin}.",
             peaks, levels,
             f"<b>Open checks.</b> {cap(S0._esc(team.checks_sentence()))}; the reviews so far are by AI agents."]
    items = [x for x in items if x]
    return "<ul class='lim'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def means(G):
    f, AO, S4, FC, RV, C, LF = G["f"], G["AO"], G["S4"], G["FC"], G["RV"], G["C"], G["LF"]
    both = AO["met"] and FC["success"]
    away = (f"without the river gauge the LSTM loses {n2(G['g1'])} NSE {LEAD_WORD[first]}, and a river never seen loses {n2(LF['loss'])} NSE with {RV['n_train']['lobo_w_static']} training rivers "
            f"({n2(RV['loss'][1])} with {RV['n_train']['unseen50_w_static']})")
    give = (f"onsets warned {LEAD_WORD[first]}: {pct(G['on0'])} with today&rsquo;s inputs, {pct(AO['with_'][first])} with the rain up to midnight from AORC, {S0.AORC_WORDING} ({G['hind']}; "
            f"{pct(G['share48'])} with Stage IV: {S0.REALTIME_WORDING}), {pct(G['up'])} with a perfect weather forecast, {UPPER}; {LEADS23} a real forecast beside that rain takes them from "
            f"{pct(FC['rain'][k2])} and {pct(FC['rain'][k3])} to {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])}")
    g = AO["gain"]
    as_much = min(g["ridge"], g["forest"]) >= S0.AS_MUCH * g["lstm"]
    head = ("The finding holds in both directions: taking information away lowered the skill, giving it raised it." if both else "Taking information away and giving it: what the inputs changed.")
    p1 = (f"<p><b>{head}</b> {cap(away)}. {cap(give)}. Changing the kind of model changed little"
          + (f" (the simple models given the same inputs gain as many onsets: {S0.INFO_NOT_NET})" if as_much else "") + f". Yet the verdict for the onsets is {G['verdict_words']} at every lead time with every input "
          f"tried ({pct(AO['false'][first])} of the warnings false {LEAD_WORD[first]} with the rain up to midnight).</p>")
    took = f["new"]["took"]
    missing = took.split("Missing: ", 1)[1] if "Missing: " in took else took
    if FC["members"] and "the ensemble," in missing:                                             # 'the ensemble' = the other members of the five
        missing = missing.replace("the ensemble,", f"the other {word(FC['members'] - 1)} ensemble members,", 1)
    p2 = (f"<p><b>What a service would need.</b> The rain of the whole day up to the moment of the forecast, from radar (Stage IV, tested on page 2), and a weather forecast for the rain still to come: "
          f"{pct(C['future'])} of the rain around the onsets falls after the forecast is issued, and an operational flood model reads weather forecasts ({cite('nearing2024')[1:-1]}). "
          f"<b>What is still missing:</b> {missing[0].lower() + missing[1:]}</p>")
    return p1 + p2


def roles_para(G):
    roles = s8_limits._plan()[0]
    clean = lambda o: re.sub(r" [(]week \d+[)]", "", o)                    # 'model comparison (week 6)' -> 'model comparison': the week of the plan is of no use here
    owns = "; ".join(f"<b>{team.label(m)}</b>: {clean(o)[0].lower() + clean(o)[1:]}" for m, o in roles)
    open_ = all(team.check_status(m) == "still to be done" for m, _ in roles)
    return (f"<p>Roles in the plan (<code>PLAN.md</code>): {owns}. "
            + ("None of these parts has been checked by its owner yet; <code>team.json</code> is where a member records a finished check." if open_ else "The status of each member&rsquo;s own check is in <code>team.json</code>.") + "</p>")


def reproduce(G):
    n_full = pdf_pages(ROOT / "Flood_Report.pdf")
    full = f"<code>Flood_Report.pdf</code> ({n_full} pages; sections 4.6 to 4.8 and 8)" if n_full else "<code>Flood_Report.pdf</code> (sections 4.6 to 4.8 and 8)"
    return (f"<p><code>./run_all.sh</code> runs every step from the raw CAMELS-US download (the GEFS and Stage IV downloads are separate scripts) and writes <code>results/</code> and <code>figures/</code>; "
            f"<code>FLOOD_QUICK=1 ./run_all.sh</code> is a fast smoke test. <code>python code/build_report.py</code>, <code>build_slides.py</code> and <code>build_report_short.py</code> rebuild the documents from "
            f"them, and every number in them is read from <code>results/</code>. The detail: {full} and <code>ROADMAP.md</code>.</p>")


def refs_html():
    """The cited sources as one flowing paragraph (compact): the full references of the keys cited so far, in alphabetical order, with the letter of a short form such as
    'Kratzert et al., 2019a' carried into the reference."""
    keys = sorted(CITED, key=lambda k: REFERENCES[k][1].lower())
    items = []
    for key in keys:
        short, full = REFERENCES[key]
        m = re.search(r"(\d{4})([a-z])$", short)
        if m:
            full = full.replace(f"({m.group(1)})", f"({m.group(1)}{m.group(2)})", 1)
        items.append(full)
    return '<p class="refs">' + " &middot; ".join(items) + "</p>"


def page3(G, plan):
    return f"""<div class="pg" id="p3">
<h2>What it means</h2>
{means(G)}
<h2>Limits</h2>
{limits(G)}
<h2>The work in numbers</h2>
{work_table(G)}
<h2>Who did what</h2>
<p>{S0.who(G['R'], short=True).replace('(section 9)', '(report, section 9)').replace('for sections 4.6 to 4.8', 'for sections 4.6 to 4.8 of the report')}</p>
{roles_para(G)}
<h2>How to reproduce</h2>
{reproduce(G)}
<h2>Sources</h2>
{refs_html()}
<div class="safety"></div></div>"""


# ---- the page ----------------------------------------------------------------------------------------------------------------------
CSS_EXTRA = """
@page{size:A4;margin:6.5mm 11.5mm 9mm;@bottom-left{content:'Predicting river floods with AI \\00B7  three-page report \\00B7  AI course group project';@FOOT@}@bottom-right{content:counter(page) ' / ' counter(pages);@FOOT@}}
body{font-size:@PT@pt;line-height:@LH@}
h1{font-size:17pt;margin:0 0 1pt}
h2{font-size:@H2@pt;margin:6pt 0 2pt;border-left:4px solid var(--blue);padding-left:6pt;break-after:avoid}
p{margin:0 0 @PG@pt} .sub{margin:0 0 3pt;font-size:@SMALL@pt} code{font-size:@SMALL@pt}
.tagline{font-size:10.8pt;line-height:1.28;font-weight:600;color:var(--blue);margin:1pt 0 2pt}
.verdict{font-size:@PT@pt;line-height:@LH@;padding:5pt 10pt 3pt;margin:3pt 0 2pt} .verdict p{margin:0 0 2.4pt}
.pg{break-after:page} .pg:last-child{break-after:auto} .safety{height:@SAFETY@mm}
.two{gap:12pt} .how p{margin:0 0 3pt}
.cap,figcaption{font-size:@SMALL@pt;line-height:1.28;margin:1.5pt 0 4pt;color:var(--ink2)}
figure.key{margin:1pt 0 2pt} figure.key figcaption{margin:2pt 0 0}
table.tl{font-size:@TAB@pt;margin:2pt 0 0} table.t2{margin-top:4pt} table.tl th,table.tl td{padding:0.55pt 3pt;line-height:1.15;text-align:left}
table.tl .v{display:inline-block;min-width:2.35em;text-align:right;margin-right:3pt} table.tl .v.w{min-width:0}
table.tl .sr{font-size:@SR@pt;color:var(--muted);white-space:nowrap} table.tl i.mx{font-size:@SR@pt;color:var(--muted);font-style:italic}
table.tl tr.ctl td:first-child{padding-left:10pt} table.tl tr.ctl td{color:var(--ink2)} table.tl tr.alt td{background:#f6f3ee}
table.dt{font-size:@DTT@pt;table-layout:fixed;margin:1pt 0 0} table.dt th,table.dt td{padding:0.35pt 2.5pt;line-height:1.1;text-align:left;vertical-align:top;border-top:1px solid var(--grid)}
table.dt th{font-weight:600;color:var(--muted);font-size:@DTT@pt;line-height:1.12} table.dt td:first-child{border-left:3px solid var(--axis)} table.dt tr.r-every td:first-child{border-left-color:var(--blue)}
table.dt tr.r-extra td:first-child{border-left-color:var(--aqua)} table.dt sub{font-size:70%;line-height:0}
.leg{font-size:@DT@pt;font-weight:400;color:var(--ink2);margin-left:8pt} .k{display:inline-block;width:7pt;height:7pt;border-radius:1pt;vertical-align:-1pt;margin:0 2pt 0 4pt;background:var(--axis)} .k.every{background:var(--blue)} .k.extra{background:var(--aqua)}
p.ds{font-size:@DT@pt;line-height:1.2;margin:0 0 2pt} p.ds sub{font-size:70%;line-height:0} p.ds.how{font-size:@DT2@pt;line-height:1.24;margin:3pt 0 2pt}
h2#b-h2{margin-top:4pt} figure.flow{margin:2pt 0 2pt} figure.flow figcaption{font-size:@DT@pt;line-height:1.2;margin:1pt 0 0;color:var(--ink)} figure.flow code,p.ds code{font-size:@DT@pt}
table.work{font-size:@TAB@pt;margin:1pt 0 3pt} table.work th{width:auto;color:var(--ink2);font-weight:600;padding:1.6pt 4pt;line-height:1.22} table.work td{padding:1.6pt 4pt;line-height:1.22}
ul.lim{margin:0 0 2pt;padding-left:12pt} ul.lim li{margin-bottom:1.2pt}
p.refs{font-size:@REF@pt;line-height:1.25;color:var(--ink2);margin:0} .h2s{font-weight:400;font-size:@SMALL@pt;color:var(--ink2)}
"""


def css(plan):
    pt = plan[0]
    sub = dict(PT=pt, LH=1.3, PG=2.6, H2=round(pt + 1.3, 1), SMALL=round(pt - 1.1, 1), TAB=round(pt - 1.0, 1), SR=round(pt - 2.4, 1), DT=round(pt - 1.4, 1), DTT=round(pt - 1.1, 1), DT2=round(pt - 1.1, 1), REF=round(pt - 2.0, 1), SAFETY=SAFETY_MM, FOOT=FOOT)
    out = CSS_EXTRA
    for k, v in sub.items():
        out = out.replace(f"@{k}@", str(v))
    return S0.LADDER_STYLE.replace("<style>", "").replace("</style>", "") + out


def render(G, plan, only=None):
    """The HTML of the three pages (or of page `only`, 1 to 3, for a test print) for one plan: (body type size in pt, key figure width in %, table of steps on page 1)."""
    pages = (page1, page2, page3)
    body = "".join(f(G, plan) for i, f in enumerate(pages, 1) if only in (None, i))
    return page("Predicting river floods with AI: three-page report", body, css_extra=css(plan))


def fit_plan(G, out, pdf):
    """Fit the three pages (see SIZES, FIG_WIDTHS, FIG_FLOOR, FLOW_FLOOR) and write Flood_Report_3pages_dense.html and .pdf (NAME) with the plan found: (plan, number of pages of the print).
    The plan is None, with a line on stderr, when no body size fits; the barest plan is then printed so that the files show how far the text is over. No browser: (plan, None)."""
    def one_page(plan, which):
        """True if page `which` alone prints as one page (with its spare of SAFETY_MM), False if not, None without a browser."""
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / f"page{which}.html"
            src.write_text(render(G, plan, only=which), encoding="utf-8")
            if not print_pdf(src, src.with_suffix(".pdf"), quiet=True):
                return None
            return pdf_pages(src.with_suffix(".pdf")) == 1

    def widest(pt, which, floor):
        """The widest figure width of FIG_WIDTHS (down to `floor`) at which page `which` (1: the flow figure, 2: the key figure) fits; the floor is known to fit."""
        make = (lambda w: (pt, 100, w)) if which == 1 else (lambda w: (pt, w, 100))
        lo, hi = 0, FIG_WIDTHS.index(floor)
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if one_page(make(FIG_WIDTHS[mid]), which) else (mid + 1, hi)
        return FIG_WIDTHS[lo]

    found = None
    for pt in SIZES:
        narrow2 = one_page((pt, FIG_FLOOR, 100), 2)
        if narrow2 is None:
            break                                                      # no browser
        if not narrow2 or not one_page((pt, 100, 100), 3) or not one_page((pt, 100, FLOW_FLOOR), 1):
            continue                                                   # a page over even with its narrowest figure: a smaller body size
        found = (pt, widest(pt, 2, FIG_FLOOR), widest(pt, 1, FLOW_FLOOR))
        break
    plan = found or (SIZES[-1], FIG_FLOOR, FLOW_FLOOR)
    out.write_text(render(G, plan), encoding="utf-8")
    if not print_pdf(out, pdf):
        return plan, None
    n = pdf_pages(pdf)
    if not found or n != 3:
        print(f"build_report_short: no plan prints as three pages (the barest, {plan}, has {n}): shorten the text", file=sys.stderr)
    return plan, n


def main():
    R, T = load_results()
    G = gather(R, T)
    if G is None:
        sys.exit(1)
    out = ROOT / f"{NAME}.html"
    pdf = out.with_suffix(".pdf")
    plan, n_pages = fit_plan(G, out, pdf)
    print("pages:", n_pages, "| plan (type pt, key figure %, flow figure %):", plan)
    print("wrote", out, "and", pdf if n_pages is not None else "(no PDF: no browser)")


if __name__ == "__main__":
    main()
