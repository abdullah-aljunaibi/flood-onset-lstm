"""d9: builds the review record of the final critic from the outputs of d1-d5 (and d2a) and the findings file, into OUT/review_documents_2026-10-08.json.

    python code/review/documents_2026-10-08/d9_assemble_review.py                         # writes OUT/review_documents_2026-10-08.json only
    python code/review/documents_2026-10-08/d9_assemble_review.py --write-into-project    # also writes results/review_documents_2026-10-08.json

The record is the one that the documents cite (REVIEW_WORDING in code/report_sections/s0_summary.py: 'the real-time check on all onsets, the rain-clock diagnostic, the resampling ranges and the key figure
of 8 October 2026 were verified by the final critic agent only'). Every number in 'checks' and 'numbers' is read from the outputs of the check scripts, not typed; the sentences around them are written here.
The script refuses to assemble when a check script's output says that something differed (all_equal False), and refuses --write-into-project when a reviewed results, data or code file is newer than the
oldest check output (the checks would then describe an older file). Documents are not in that test: they are being rebuilt; the versions judged are recorded by hash (environment variable DOC_SNAP_ROOT: the folder with
the snapshots snap/ (10:21-10:24 build), snap2/ (11:30 slides build and README), snap3/ (plan and plain-words page)).
"""
import hashlib
import time
from collections import Counter

from dcommon import *

WRITE = "--write-into-project" in sys.argv
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
stamp = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(Path(p).stat().st_mtime))
pc = lambda x: f"{100 * x:.0f}%"

D1, D2, D2A, D3, D4, D5 = (jread(n) for n in ("d1_numbers.json", "d2_headline_and_ranges.json", "d2a_first_pass_bootstrap.json", "d3_stage4.json", "d4_rain_clock.json", "d5_key_figure.json"))
FINDINGS = json.load(open(HERE / "findings.json"))
PA, PB = D2["part_A_scores"], D2["part_B_ranges"]

# ---- nothing may have differed ---------------------------------------------------------------------------------------------------------------------------------------------------------
assert D1["n_pass"] == D1["n_claims"] == 69, "d1: a claim did not pass"
assert PA["all_equal"] and PB["all_within_tolerance"] and PB["all_point_estimates_equal"], "d2: something differed"
assert D3["all_equal"] and D3["reproduction"]["max_abs_difference_to_the_stored_single_seed_forecasts_mm_per_day"] == 0.0, "d3: something differed"
assert D4["all_equal"] and D4["rain_class_of_every_onset_equal_to_the_csv"] and D4["missed_and_perfect_flags_equal_to_the_csv"], "d4: something differed"
for k in ("A_values_read_again", "B_cross_checks_read_again", "C_computed_recomputed", "D_rescored_from_stored_forecasts"):
    assert D5[k]["n_equal"] == D5[k]["n"], f"d5: {k} differed"
E = D5["E_drawn_values"]
assert E["labels_equal_to_the_formatted_source"] == E["labels_read"] and all(x["ok"] for x in E["lettering"]), "d5: a printed number differed"

# ---- files reviewed ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
outputs = sorted(OUT.glob("d*.json"))
t_outputs = min(p.stat().st_mtime for p in outputs if p.name in ("d1_numbers.json", "d2_headline_and_ranges.json", "d3_stage4.json", "d4_rain_clock.json", "d5_key_figure.json"))
reviewed = {}


DEPENDED = {f"results/{n}.json" for n in ("lstm_all", "rain_timing", "rain_to_midnight", "aorc_rain_to_midnight", "stage4_realtime", "more_basins", "forecast_input", "uncertainty", "key_figure", "flood_analysis", "same_misses", "onset_rain_clock",
                                              "model_comparison", "lstm_lobo", "baselines", "data_audit", "fair_test", "review_rain_to_midnight", "review_aorc_rain_to_midnight", "review_more_basins", "review_forecast_input")} \
    | {"results/flood_events.csv", "results/onset_rain_clock.csv", "results/basin_audit.csv", "figures/key_figure.png", "figures/key_figure_plain.png", "figures/key_figure_talk.png", "data/aorc/stage4_issue_days.csv"} \
    | {f"code/{n}" for n in ("flood.py", "camels.py", "features.py", "common.py", "check_stage4_realtime.py", "check_onset_rain_clock.py", "11_key_figure.py", "13_uncertainty.py", "06_flood_analysis.py", "fetch_aorc.py", "16_aorc_rain_to_midnight.py")}


def add(rel):
    p = PROJ / rel
    depends = rel in DEPENDED or rel.startswith(("results/preds/", "results/models/", "data/aorc/aorc_", "data/hourly/era5_"))
    reviewed[rel] = dict(sha256=sha(p), bytes=p.stat().st_size, modified=stamp(p), the_checks_depend_on_it=depends)


for p in sorted(RES.glob("*.json")):
    add(f"results/{p.name}")
for n in ("flood_events.csv", "onset_rain_clock.csv", "basin_audit.csv", "baselines.csv"):
    if (RES / n).exists():
        add(f"results/{n}")
for p in sorted((RES / "preds").glob("*.npz")):
    add(f"results/preds/{p.name}")
for s in (0, 1, 2):
    add(f"results/models/lstm_wqa_s{s}.pt")
for p in sorted((DATA / "aorc").glob("aorc_*.npz")):
    add(f"data/aorc/{p.name}")
add("data/aorc/stage4_issue_days.csv")
for p in sorted((DATA / "hourly").glob("era5_*.json")):
    add(f"data/hourly/{p.name}")
for n in ("key_figure.png", "key_figure_plain.png", "key_figure_talk.png"):
    add(f"figures/{n}")
for n in ("flood.py", "camels.py", "features.py", "common.py", "check_stage4_realtime.py", "check_onset_rain_clock.py", "fetch_aorc.py", "06_flood_analysis.py", "11_key_figure.py", "13_uncertainty.py", "16_aorc_rain_to_midnight.py",
          "report_sections/s0_summary.py", "report_sections/s4d_midnight.py", "report_sections/s8_limits.py", "guide_sections/g5_next.py"):
    if (PROJ / "code" / n).exists():
        add(f"code/{n}")
for n in ("README.md", "ROADMAP.md", "NEXT_STEPS_FOR_AGENTS.md"):
    add(n)
stale = [k for k, v in reviewed.items() if v["the_checks_depend_on_it"] and PROJ.joinpath(k).stat().st_mtime > t_outputs]
if WRITE and stale:
    sys.exit(f"refusing to write into the project: reviewed files newer than the oldest check output ({time.strftime('%H:%M:%S', time.localtime(t_outputs))}): {stale[:8]}")

DOCS = {"Flood_Report.pdf": "snap", "Flood_Report.html": "snap", "short_report.md": "snap", "README.md": "snap2", "Flood_Slides.pdf": "snap2", "Flood_Slides.pptx": "snap2", "Flood_Slides.html": "snap2", "slides_notes.md": "snap2",
        "Professor_QA.pdf": "snap", "Professor_QA.md": "snap", "Professor_QA.html": "snap", "Project_Guide.pdf": "snap", "Project_Guide.html": "snap", "Project_Overview.pdf": "snap", "Project_Overview.html": "snap",
        "Presentation_Plan.pdf": "snap3", "Presentation_Plan.md": "snap3", "Project_in_Plain_Words.pdf": "snap3"}
docs_judged = {}
root = os.environ.get("DOC_SNAP_ROOT")
if root:
    for name, folder in DOCS.items():
        p = Path(root) / folder / name
        if p.exists():
            cur = PROJ / name
            docs_judged[name] = dict(sha256_of_the_version_judged=sha(p), bytes=p.stat().st_size, snapshot_taken_from_the_build_of=stamp(p) if False else None,
                                     the_project_file_now=("equal" if cur.exists() and sha(cur) == sha(p) else "changed (rebuilt after the critique)" if cur.exists() else "absent"), project_file_modified=stamp(cur) if cur.exists() else None)
for v in docs_judged.values():
    v.pop("snapshot_taken_from_the_build_of")

# ---- the numbers, from the outputs ---------------------------------------------------------------------------------------------------------------------------------------------------
V = D3["variants"]
mine = PA["mine"]
headline = {}
for s in ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqa", "lstm_wqf", "lstm_wqaF_sub", "lstm_wqaF_clim", "lstm_wqaP", "lstm63_wq", "base_persistence", "base_linreg_wq", "base_rf_wq"):
    headline[s] = {l: dict(nse_median=round(mine[s][l]["nse_median"], 4), onsets_warned=round(mine[s][l]["onset"]["pod"], 4), onsets_false=round(mine[s][l]["onset"]["far"], 4) if mine[s][l]["onset"]["far"] == mine[s][l]["onset"]["far"] else None,
                       onsets=mine[s][l]["onset"]["hits"] + mine[s][l]["onset"]["misses"]) for l in ("l1", "l2", "l3")}
ranges = [dict(what=r["what"], mine=r["mine"], stored=r["builder"], point_mine=r.get("point_mine"), point_stored=r.get("point_builder")) for r in PB["rows"]]
_iv = [r for r in PB["rows"] if isinstance(r["mine"], list)]
MAXD = dict(
    shares_points=max(abs(a - b) for r in _iv if not r["what"].startswith("NSE") for a, b in zip(r["mine"], r["builder"])) * 100,
    nse=max(abs(a - b) for r in _iv if r["what"].startswith("NSE") for a, b in zip(r["mine"], r["builder"])),
    verdict_shares_points=max(abs(r["mine"][k] - r["builder"][k]) for r in PB["rows"] if isinstance(r["mine"], dict) for k in r["mine"]) * 100)
rain_all, rain_miss = D4["mine"]["all"], D4["mine"]["missed_by_lstm_wq"]
numbers = {
    "documents_69_claims": dict(claims=D1["n_claims"], passing=D1["n_pass"], by_document={k: sum(c["document"] == k for c in D1["claims"]) for k in sorted({c["document"] for c in D1["claims"]})},
                                documents={"R": "Flood_Report.pdf", "S": "Flood_Slides.pdf", "Q": "Professor_QA.pdf", "G": "Project_Guide.pdf", "O": "Project_Overview.pdf"}, claims_list=[dict(claim=c["claim"], document=c["document"], pages=c["pages"], source=c["source"]) for c in D1["claims"]]),
    "headline_scores_from_stored_forecasts": dict(compared_with_flood_evaluate=PA["compared_with_flood_evaluate"], all_equal=PA["all_equal"], sets=sorted(mine), scores=headline),
    "ranges_own_4000_draws": dict(draws=PB["draws"], largest_difference_to_the_stored_intervals=MAXD, rows=ranges, onset_verdict_shares_in_my_draws=PB["onset_verdict_shares_in_the_draws"], first_pass_bootstrap_seed1=D2A),
    "stage4_realtime_recomputed": dict(
        reproduction_of_the_stored_forecasts=D3["reproduction"], stage4_hours=D3["hours"], k_aorc_over_stage4=D3["K_aorc_over_stage4"], mean_issue_day_rain_mm=D3["mean_rain_mm_of_the_issue_days"],
        onsets_warned={k: v["onsets_warned"] for k, v in V.items()}, single_seeds={k: v["single_seeds"] for k, v in V.items()}, aorc=D3["reproduction"]["onsets_warned_aorc_mine"], rain_table_alone=D3["reproduction"]["onsets_warned_rain_table_alone_stored_lstm_wq"],
        share_of_the_aorc_effect_surviving=D3["share_of_the_aorc_effect_surviving"], interval_of_that_share_95_own_2000_draws=D3["interval_of_the_share_surviving_95"], onsets_lost_and_gained_against_aorc=D3["onsets_lost_and_gained_stage4_against_aorc"],
        issue_days_of_all_12_basins=D3["issue_days_of_all_12_basins"], numbers_compared_with_results_stage4_realtime_json=D3["n_compared"], all_equal=D3["all_equal"], seconds=D3["seconds_total"],
        onsets_without_any_replaced_hour=D3["onsets_without_any_replaced_hour"], fit_on_the_onset_days=D3["fit_on_the_onset_days"], agreement_of_the_two_products={k: v for k, v in D3["agreement_of_the_two_products"].items() if k != "per_basin"},
        controls_on_the_rain_of_the_issue_day=D3["controls_against_the_builders"], controls_my_draws={k: dict(onsets_warned=v["onsets_warned"], mean=v["mean"]) for k, v in D3["controls"].items()},
        second_set_of_networks_wqa_rep=dict(reproduction=D3["wqa_rep"]["reproduction"], onsets_warned={k: v["onsets_warned"] for k, v in D3["wqa_rep"]["variants"].items()}, single_seeds={k: v["single_seeds"] for k, v in D3["wqa_rep"]["variants"].items()},
                                            share_of_the_aorc_effect_surviving=D3["wqa_rep"]["share_of_the_aorc_effect_surviving"], issue_days_of_all_12_basins=D3["wqa_rep"]["issue_days_of_all_12_basins"])),
    "rain_clock_recomputed": dict(
        windows_hours=D4["quoted_shares"]["hours_of_the_windows"], mean_mm=D4["quoted_shares"]["mean_mm"], shares_of_the_mean_rain={k: v for k, v in D4["quoted_shares"].items() if k.startswith("share")},
        all_611=rain_all, missed_by_lstm_wq_497=rain_miss, missed_but_warned_with_the_perfect_forecast=D4["onsets_missed_but_warned_by_the_perfect_forecast_model"], rain_classes={k: dict(n=v["n"], median_share_known_by_table=v["median_share_known_by_table"], median_share_known_by_midnight=v["median_share_known_by_midnight"]) for k, v in D4["mine"]["by_rain_class"].items()},
        rain_timing_classes_against_flood_analysis=D4["rain_timing_classes_against_flood_analysis"], fields_compared_with_results_onset_rain_clock_json=D4["compared"], all_equal=D4["all_equal"], per_onset_rows_largest_difference_mm=D4["per_onset_rows_max_abs_difference_mm"],
        two_sum_routes_largest_difference_mm=D4["two_routes_largest_difference_mm"]),
    "key_figure_checked": dict(
        sources=D5["F_counts_and_hashes"]["sources"], with_file_and_key=D5["F_counts_and_hashes"]["with_file_and_key"], computed=D5["F_counts_and_hashes"]["computed"], flagged_in_figure=D5["F_counts_and_hashes"]["flagged_in_figure"],
        values_read_again=(D5["A_values_read_again"]["n_equal"], D5["A_values_read_again"]["n"]), cross_checks_read_again=(D5["B_cross_checks_read_again"]["n_equal"], D5["B_cross_checks_read_again"]["n"]),
        computed_recomputed=(D5["C_computed_recomputed"]["n_equal"], D5["C_computed_recomputed"]["n"]), scores_rederived_from_forecasts=(D5["D_rescored_from_stored_forecasts"]["n_equal"], D5["D_rescored_from_stored_forecasts"]["n"]),
        sets_scored=D5["D_rescored_from_stored_forecasts"]["sets_scored"], printed_labels=(E["labels_equal_to_the_formatted_source"], E["labels_read"]), labels_per_rendering=E["labels_per_rendering"], lettering_and_caption_numbers=(sum(x["ok"] for x in E["lettering"]), len(E["lettering"])),
        ocr=E.get("ocr_cross_check"), my_comparisons_in_all=D5["F_counts_and_hashes"]["my_comparisons"], builder_checks_block=D5["F_counts_and_hashes"]["builder_says"], sha256_of_the_checked_files=D5["F_counts_and_hashes"]["sha256"],
        modified=D5["F_counts_and_hashes"]["files_modified"], printed_labels_list=[dict(printed=x["printed"], source=x["source"], value=x["value"], renderings=x["renderings"]) for x in E["labels"]]),
}
n_checks_pass = None
checks = [
    dict(id="c1", what="69 numbers of the documents (page 1, sections 4.6-4.8, section 8, cost and fair test; slides 7, 10-13; Q&A H1, H2, H6, H11-H13; guide p.32; overview) against the results files they come from",
         how="d1_numbers.py: the fragment as the document must print it is built from the results file (read as JSON, the key is in numbers.documents_69_claims.claims_list) and looked for in the text of each PDF page (PDFKit text) or Markdown file; 15 claims are fixed sentences compared with the truth rebuilt from the file",
         result="pass", numbers=dict(claims=69, passing=69, by_document=numbers["documents_69_claims"]["by_document"], note="the versions judged are those of the hashes in files_reviewed.documents_judged (the 10:21-10:24 build, the 11:30 slides)")),
    dict(id="c2", what="the headline scores of 14 prediction sets at 1, 2 and 3 days: median NSE of the 12 basins and the warning tables of all high-flow days and of the onsets (hits, misses, false alarms, correct negatives)",
         how="d2_headline_and_ranges.py part A: the critic's own scoring from results/preds/*.npz (high-flow level = 95th percentile of the training years computed from the flow table; onset = high-flow day forecast while the river was at or below the level lead days before; warned = forecast above the level) against flood.evaluate on the same files",
         result="pass", numbers=dict(numbers_compared=PA["compared_with_flood_evaluate"], equal=PA["compared_with_flood_evaluate"], sets=len(mine), largest_difference="0 to 1e-9", headline={k: headline[k]["l1"] for k in ("lstm_wq", "lstm_wqn", "lstm_wqa", "lstm_wqf", "lstm_w")})),
    dict(id="c3", what="the resampling ranges that the documents quote (section 8 and the Q&A D11: 19% (10% to 29%), 30% (20% to 42%), 64% (55% to 72%), 51% (41% to 61%), 39% (28% to 48%), 37% (27% to 46%), the gap closed 11% to 44%, 3,998 of 4,000, the NSE differences, 83% / 7% / 10%) and the verdict shares in the draws",
         how="d2_headline_and_ranges.py part B: 4,000 draws of the 9 test water years and the 12 basins with replacement (own random numbers, seed 20261008; the builder's seed is 20261004), a whole water year kept together, 2.5th to 97.5th percentile; the point estimates without resampling are compared at 1e-9; tolerance for the intervals 0.0105 (1.05 points)",
         result="pass", numbers=dict(statistics_redrawn=len(ranges), all_intervals_within_tolerance=PB["all_within_tolerance"], all_point_estimates_equal=PB["all_point_estimates_equal"], rows=ranges, first_pass_seed1_same_ranges=D2A)),
    dict(id="c4", what="my own AORC day table and my own forward pass of the three saved lstm_wqa networks reproduce the stored forecasts (the base of the real-time check)",
         how="d3_stage4.py: the hourly AORC basin means (data/aorc/aorc_*.npz, hour-ending UTC) summed by the local wall-clock day (summer time included; Central time for Leaf River MS); input matrix from the scalers stored in each .pt (dyn mean and spread, log1p flow, ten statics); own torch.nn.LSTM(18, 64) and Linear(64, 3) loaded from the saved state; 365 days of reading, blocks of 365 days; shift to the target day, never below zero",
         result="pass", numbers=dict(largest_difference_to_the_stored_single_seed_forecasts_mm_per_day=D3["reproduction"]["max_abs_difference_to_the_stored_single_seed_forecasts_mm_per_day"], largest_difference_to_the_stored_seed_average=D3["reproduction"]["max_abs_difference_to_the_stored_seed_average"],
                                    onsets_warned_aorc=D3["reproduction"]["onsets_warned_aorc_mine"], single_seeds=D3["reproduction"]["onsets_warned_aorc_single_seeds_mine"], onsets=611, onsets_equal_to_results_flood_events_csv=True)),
    dict(id="c5", what="the real-time check on all 611 onsets: results/stage4_realtime.json (294 of 611 warned with Stage IV on the issue day, 275 without its last hour, 255 without two hours, 319 with Stage IV x 1.0634, 296 and 274 with AORC itself minus 1 and 2 hours, 289 with the day before as well, 313 AORC, 114 rain table alone; 90% of the AORC effect survives)",
         how="d3_stage4.py: the Stage IV hours of the stored series (data/aorc/stage4_issue_days.csv: 14,664 hours of the issue days, hour-ending UTC; hours without a value replaced by the AORC hour) assigned to the local day in which they start, summed, put in place of the issue-day value of the AORC column for all 611 onsets at once, the three networks run forward, the three forecasts averaged, an onset counted as warned when the 1-day forecast of its target day exceeds the level (trigger 1.0); one thread, nothing trained",
         result="pass", numbers=dict(onsets_warned={k: v["onsets_warned"] for k, v in V.items()}, single_seeds={k: v["single_seeds"] for k, v in V.items()}, share_surviving=D3["share_of_the_aorc_effect_surviving"], numbers_compared=D3["n_compared"], equal=D3["n_compared"], hours_replaced_by_aorc=D3["hours"]["replaced_by_the_aorc_hour"], k=D3["K_aorc_over_stage4"])),
    dict(id="c6", what="the warnings given on the issue days of all 12 basins (the false-warning side of the real-time check)",
         how="d3_stage4.py: Stage IV (and Stage IV x k) substituted on each of the 368 issue days in every basin, the warnings of the next day counted on the basin-days when the river was at or below the level",
         result="pass", numbers=D3["issue_days_of_all_12_basins"]),
    dict(id="c7", what="the intervals of the share of the AORC effect that survives (85% to 97% drawing onsets, 84% to 97% drawing issue days)",
         how="d3_stage4.py: 2,000 draws of the 611 onsets (and of the 368 issue days, an issue day keeping its onsets together), share = (Stage IV - table) / (AORC - table) of the drawn onsets",
         result="pass", numbers=dict(mine=D3["interval_of_the_share_surviving_95"], stored=[[0.8453518041237114, 0.9714326298701298], [0.8420991432068543, 0.9718368872452593]], tolerance=0.01)),
    dict(id="c7b", what="the second set of three networks of results/stage4_realtime.json (lstm_wqa_rep: AORC in place of the Daymet rain, 6 daily inputs): the same substitutions and the 12-basin tally",
         how="d3_stage4.py: the same code with the networks of results/models/lstm_wqa_rep_s0..2.pt (the input matrix follows each network's own list of columns)", result="pass",
         numbers=dict(reproduction=D3["wqa_rep"]["reproduction"], onsets_warned={k: v["onsets_warned"] for k, v in D3["wqa_rep"]["variants"].items()}, share_surviving=D3["wqa_rep"]["share_of_the_aorc_effect_surviving"]["stage4"], issue_days_of_all_12_basins=D3["wqa_rep"]["issue_days_of_all_12_basins"])),
    dict(id="c7c", what="the rest of the Stage IV file: the onsets whose issue day has no replaced hour (609), the fit of the forecasts on the onset days (mean absolute error 4.64 mm/day with AORC and with Stage IV, 5.83 with the rain table alone), the agreement of the two rain products on the issue days (correlation 0.955, ratio 0.94), and the three controls on the rain of the issue day (another onset's Stage IV rain, an ordinary day's AORC rain, no rain)",
         how="d3_stage4.py: the subset, the fit statistics (mean absolute error, Spearman, sum of forecasts over sum of observations) and the product statistics recomputed from my own tables and forecasts and compared at 1e-9; the controls redrawn with my own random numbers (10 draws each, own seed) and compared as ranges (my mean within 5 onsets of the builder's and the ranges overlap); no rain is deterministic and equal",
         result="pass", numbers=dict(without_replaced_hour=D3["onsets_without_any_replaced_hour"], fit=D3["fit_on_the_onset_days"], agreement={k: v for k, v in D3["agreement_of_the_two_products"].items() if k != "per_basin"}, controls=D3["controls_against_the_builders"])),
    dict(id="c8", what="the rain clock of the 611 onsets: results/onset_rain_clock.json (mean 6.5 / 12.6 / 8.5 / 8.1 mm in the four windows, shares 18 / 35 / 46%, the 497 onsets missed by lstm_wq with 3.8% of their rain known by the table and 50% by midnight, the 308 of them warned with the perfect forecast, the five rain classes)",
         how="d4_rain_clock.py: the hourly ERA5 files (value at T = hour ending T, UTC) read with my own reader; windows in local standard time (UTC-5 Eastern, UTC-6 Central for Leaf River MS) 09:00 (t-1) to 09:00 (t), 09:00 (t) to 24:00 (t), 00:00 (t+1) to 09:00 (t+1), 09:00 (t+1) to 24:00 (t+1) summed per onset by a slice sum (and by a cumulative-sum difference: the two routes differ by 3e-11 mm, which matters only for two onsets that have exactly half of their rain in the table window); onsets, missed and warned flags from my own scoring of results/preds; rain class from the Daymet column",
         result="pass", numbers=dict(fields_compared=D4["compared"], equal=D4["compared"], onset_rows=611, rows_largest_difference_mm=D4["per_onset_rows_max_abs_difference_mm"]["total"], rain_classes_and_flags_equal_to_the_csv=True, missed=D4["onsets_missed_by_lstm_wq"], missed_but_warned_by_perfect=D4["onsets_missed_but_warned_by_the_perfect_forecast_model"], windows_hours=D4["quoted_shares"]["hours_of_the_windows"])),
    dict(id="c9", what="the warned share by rain class (0-25%: 37 of 52, 25-50%: 36 of 69, 50-75%: 32 of 149, 75-100%: 8 of 335, under 1 mm: 1 of 6) of results/flood_analysis.json against my own classes",
         how="d4_rain_clock.py: class = floor(4 x share of the two days' Daymet rain on the target day), under 1 mm on both days apart; warned = 1-day forecast of lstm_wq above the level", result="pass",
         numbers=D4["rain_timing_classes_against_flood_analysis"]),
    dict(id="c10", what="key figure: the 491 values of 'sources' that name a file and a key, read again", how="d5_key_figure.py part A: own path reader (keys with dots, [n] indices, '(number of entries)', '(hits + misses)', the GO sentence read as 0.70), floats at 1e-12", result="pass",
         numbers=dict(equal=D5["A_values_read_again"]["n_equal"], of=D5["A_values_read_again"]["n"], largest_difference=D5["A_values_read_again"]["largest_difference"])),
    dict(id="c11", what="key figure: the 777 cross-check entries ('checked_against') read from their files and compared with the source value", how="d5_key_figure.py part B: tolerance 0.0005 as the builder's; the largest difference found is reported", result="pass",
         numbers=dict(equal=D5["B_cross_checks_read_again"]["n_equal"], of=D5["B_cross_checks_read_again"]["n"], largest_difference=D5["B_cross_checks_read_again"]["largest_difference"], not_machine_readable=D5["B_cross_checks_read_again"]["not_machine_readable"], not_machine_readable_text=D5["B_cross_checks_read_again"]["not_machine_readable_text"])),
    dict(id="c12", what="key figure: the 116 computed sources (differences, ratios of steps, highest minus lowest of the kinds of model, shares of the gap, 294/611 and 275/611) recomputed from the other sources", how="d5_key_figure.py part C", result="pass",
         numbers=dict(equal=D5["C_computed_recomputed"]["n_equal"], of=D5["C_computed_recomputed"]["n"], largest_difference=D5["C_computed_recomputed"]["largest_difference"])),
    dict(id="c13", what="key figure: every source of the form l<lead>.<set>.nse / onset_warned / onset_false (and the per-seed ones where single-seed forecasts are stored) re-derived from the stored forecasts", how="d5_key_figure.py part D: own scoring as in c2 on 49 prediction sets", result="pass",
         numbers=dict(equal=D5["D_rescored_from_stored_forecasts"]["n_equal"], of=D5["D_rescored_from_stored_forecasts"]["n"], of_them_drawn_in_the_figure=D5["D_rescored_from_stored_forecasts"]["n_of_them_drawn_in_the_figure"], sets_scored=D5["D_rescored_from_stored_forecasts"]["sets_scored"],
                      not_rescored=f"{D5['D_rescored_from_stored_forecasts']['n_no_stored_forecast_file']} per-seed values of the Transformer experiment (model_comparison): no single-seed forecasts are stored; they were only read again (c10)")),
    dict(id="c14", what="key figure: the numbers printed in the three pictures (34 bar labels in the full rendering, 30 in the plain one, 26 in the talk one) against their sources, and the lettering and caption numbers",
         how="d5_key_figure.py part E: the labels were read by eye from figures/key_figure.png, key_figure_plain.png and key_figure_talk.png (zoom on the one the OCR misread) and each is matched with the source it must come from and formatted as the figure prints it (percent to the whole number, NSE to two decimals); Apple Vision OCR (ocr.swift) of the three pictures as a second reading; the 'mixed' labels against the stored readings",
         result="pass", numbers=dict(labels_equal_to_the_formatted_source=E["labels_equal_to_the_formatted_source"], labels=E["labels_read"], per_rendering=E["labels_per_rendering"], lettering_ok=sum(x["ok"] for x in E["lettering"]), lettering=len(E["lettering"]), ocr_found=E["ocr_cross_check"]["found_in_the_ocr_text"],
                                    ocr_not_found=E["ocr_cross_check"]["not_found_in_the_ocr_text"], mixed_labels=E["mixed_labels"])),
    dict(id="c15", what="key figure: the counts of the file's own checks block (607 sources, 159 flagged as drawn, 1,039 comparisons at 0.0005, no disagreement)", how="d5_key_figure.py part F: the sources recounted, the flagged ones recounted; the hashes of the file and the three pictures recorded", result="pass",
         numbers=dict(sources=D5["F_counts_and_hashes"]["sources"], flagged=D5["F_counts_and_hashes"]["flagged_in_figure"], flagged_matches_the_builders_count=D5["F_counts_and_hashes"]["flagged_matches_the_builders_count"], builder_says=D5["F_counts_and_hashes"]["builder_says"], my_comparisons=D5["F_counts_and_hashes"]["my_comparisons"])),
    dict(id="c16", what="round 2: the numbers of the final documents (built 13:48-13:50) against the version judged: every number that is new was traced to a results file",
         how="r2_new_numbers.py: the numeric tokens that are more frequent in the final text of the report, the Q&A, the guide, the slides, the overview and the plain-words page than in the judged build; each read in its sentence and compared with the file named below (the token list is in OUT/r2_new_numbers.json); the 20 slides of the PPTX against the PDF (numbers of every slide equal, tables included)",
         result="pass", numbers=dict(traced=[
             dict(number="+0.18 against +0.08 and +0.07 (NSE gain of the LSTM, ridge regression and forest given the AORC rain)", where="report p.1 and conclusion 6, slide 10", source="aorc_rain_to_midnight.json yardstick.new_input_given_to_each_model.*.by_lead.l1.nse_gain = 0.178 / 0.082 / 0.071"),
             dict(number="+0.07/+0.11/+0.12 (ridge), +0.07/+0.19/+0.21 (forest), -0.07/+0.15/+0.20 (LSTM): NSE gain of the forecast at 1/2/3 days", where="report p.17", source="forecast_input.json yardstick.new_input_given_to_each_model: 0.068/0.109/0.124, 0.067/0.186/0.214, -0.074/0.155 (0.1548)/0.201"),
             dict(number="average precision +0.065 (+0.006 to +0.126)", where="report p.16-17", source="review_forecast_input.json check i3 differences['wqaF_sub - wqa']: 0.065, ci95 [0.006, 0.126]"),
             dict(number="2 of 14,664 hours came from AORC; 25% with Stage IV and 30% with AORC of the issue-day warnings false; 95% interval 85% to 97%", where="report p.14", source="stage4_realtime.json fallback, issue_days_of_all_basins, variants.stage4.intervals; equal to d3"),
             dict(number="NOT YET for onsets in 92% (AORC) and 86% (forecast) of the draws 1 day ahead, no draw reaches GO", where="report p.27 (section 8)", source="uncertainty.json verdict_stability lstm_wqa.onset.l1 0.918 and lstm_wqaF_sub.onset.l1 0.856; my own draws 0.915 and 0.859 (d2)"),
             dict(number="the rain of the 63 hours around the 611 flood onsets", where="report p.13", source="the windows 24 + 15 + 9 + 15 h of results/onset_rain_clock.json (regenerated), reproduced by d4"),
             dict(number="98 listed checks over four rounds; 36 wording items; 170 checks (130 + 40); 171 onsets", where="report p.14, Q&A H12 p.22, guide p.33", source="the counts blocks of review_rain_to_midnight.json (27 + 23 + 17), review_aorc_rain_to_midnight.json (31), review_more_basins.json (36), review_forecast_input.json (36); must_fix + should_fix of the same blocks (8 + 9 + 8) + (2 + 9) = 36; the E1b review's 171 onsets"),
             dict(number="a river never seen: 0.13 NSE, 11 of 12 basins, 0.43 to 0.22, 0.02, 0.22 to 0.38", where="report p.1 (moved from conclusion 5)", source="lstm_lobo.json and more_basins.json, as in the judged build (c1)"),
             dict(number="NSE 0.77 to 0.69, 0.38 to 0.53, 0.25 to 0.45; 45% to 48%; 0.59", where="Q&A, guide, slides 11, overview, plain-words page", source="forecast_input.json summary_test.sets.wqaF_sub / wqa; stage4_realtime.json; lstm_all.json"),
             dict(number="the plan: 289 + 92 + 128 + 91 = 600 s; clock checkpoints 2:54, 4:36, 8:29", where="Presentation_Plan.pdf / .md", source="the slide times of the plan's own table, summed")],
             documents_compared={"Flood_Report": "judged 10:21 build against final", "Professor_QA": "same", "Project_Guide": "same", "Flood_Slides": "11:30 build against final", "Project_Overview": "same", "Project_in_Plain_Words": "11:34 text against final"},
             pptx_against_pdf="the numbers of all 20 slides equal (text frames and tables)", speaking_rate_words_per_second="2.17 to 2.39 by whitespace count (the plan says 'at most 2.3'); slide 10 59 s, slide 11 41 s")),
    dict(id="w1", what="document check 1, NUMBERS", how="CRITIQUE.md of 2026-10-08 (the six checks): 69 claims by script (c1), the headline scores with flood.evaluate (c2), counts, rounding where it could tip", result="pass: no wrong number, no wrong key, no stale file",
         numbers=dict(wording_around_numbers="M3, S6, S7, S12")),
    dict(id="w2", what="document check 2, WORDING RULES (the fixed wordings)", how="grep over the HTML, Markdown, PDF text and PPTX notes, then reading", result="findings", numbers=dict(items="M3, M5, M6, S4, S6, S11")),
    dict(id="w3", what="document check 3, HONESTY (who wrote and ran what, who checked what, dates, the team's open checks)", how="reading and grep of all documents, team.json", result="findings", numbers=dict(items="M4, M7, S5, S9")),
    dict(id="w4", what="document check 4, CONSISTENCY ACROSS DOCUMENTS", how="slide numbers against the report, Q&A answers against the sections they cite, guide ladder, overview, key-figure headline", result="findings", numbers=dict(items="M3, S7, S13")),
    dict(id="w5", what="document check 5, LAYOUT", how="about 60 renders (contact sheets of every page, 110-300 dpi crops of the figure pages), python-pptx for the slides", result="findings", numbers=dict(items="M1, S1, S2")),
    dict(id="w6", what="document check 6, INTERNAL LOGIC", how="page 1's findings against the sections they cite, section 8's conclusions against the findings, the open items across documents", result="findings", numbers=dict(items="S3, S6")),
]
n_listed = sum(c["id"].startswith("c") for c in checks)
n_checks_pass = sum(c["id"].startswith("c") and c["result"] == "pass" for c in checks)
n_find = sum(c["id"].startswith("w") for c in checks)
must, should = [f for f in FINDINGS if f["severity"] == "must_fix"], [f for f in FINDINGS if f["severity"] == "should_fix"]

CNT = Counter(f["status_short"] for f in FINDINGS)
IDS_BY = {k: [f["id"] for f in FINDINGS if f["status_short"] == k] for k in ("closed", "partly closed", "open")}
S_CLOSED = [i for i in IDS_BY["closed"] if i.startswith("S")]               # the should-fix items that are closed (IDS_BY['closed'] holds the eight must-fix items as well)
assert CNT["closed"] + CNT["partly closed"] + CNT["open"] == len(FINDINGS) and all(f["status_short"] == "closed" for f in must), "round 2: a must-fix item is not closed"
V1, S4 = V["stage4"], V["stage4_no_last_hour"]
answers = {
    "do_the_documents_quote_a_number_that_does_not_reproduce": f"No. 69 of 69 sampled numbers equal the results files (c1); the headline scores of 14 prediction sets equal flood.evaluate and the critic's own scoring in all {PA['compared_with_flood_evaluate']} numbers (c2). The wording around some numbers is another matter (must_fix, should_fix).",
    "real_time_check_on_all_611_onsets": (f"Recomputed with the critic's own code: with no change the critic's AORC day table and forward pass reproduce the stored forecasts exactly (largest difference 0.0 mm/day) and {D3['reproduction']['onsets_warned_aorc_mine']} onsets warned; with the stored Stage IV hours on the issue day: "
        f"{V1['onsets_warned']} of 611 ({pc(V1['share'])}; single seeds {V1['single_seeds']}), {S4['onsets_warned']} ({pc(S4['share'])}) with the last hour out, {V['stage4_no_last_two_hours']['onsets_warned']} with two hours out, {V['stage4_k']['onsets_warned']} with Stage IV x {D3['K_aorc_over_stage4']:.4f}, {V['aorc_no_last_hour']['onsets_warned']} and {V['aorc_no_last_two_hours']['onsets_warned']} "
        f"with AORC itself minus its last hour and two hours, {V['stage4_and_day_before']['onsets_warned']} with the day before as well; {D3['reproduction']['onsets_warned_rain_table_alone_stored_lstm_wq']} with the rain table alone; {pc(D3['share_of_the_aorc_effect_surviving']['stage4'])} of the AORC effect survives (own intervals {pc(D3['interval_of_the_share_surviving_95']['drawing_onsets'][0])} to {pc(D3['interval_of_the_share_surviving_95']['drawing_onsets'][1])} drawing onsets, "
        f"{pc(D3['interval_of_the_share_surviving_95']['drawing_issue_days'][0])} to {pc(D3['interval_of_the_share_surviving_95']['drawing_issue_days'][1])} drawing issue days); {D3['onsets_lost_and_gained_stage4_against_aorc']['lost']} onsets lost and {D3['onsets_lost_and_gained_stage4_against_aorc']['gained']} gained against AORC. All {D3['n_compared']} numbers compared (the two sets of networks, the 12-basin tallies, the intervals, the onsets without a replaced hour, the fits on the onset days, the agreement of the two products) equal the builder's file; the controls on the issue-day rain, redrawn with my own random numbers, are consistent with the builder's (Stage IV rain of another onset: my mean {D3['controls_against_the_builders']['another_onset_stage4']['mine_mean']:.0f} of 611 against {D3['controls_against_the_builders']['another_onset_stage4']['builder_mean']:.0f}; AORC rain of an ordinary day: {D3['controls_against_the_builders']['ordinary_day_aorc']['mine_mean']:.0f} against {D3['controls_against_the_builders']['ordinary_day_aorc']['builder_mean']:.0f}; no rain: {D3['controls_against_the_builders']['no_rain']['mine_mean']:.0f}, equal). "
        "What this establishes: the stored numbers follow from the stored Stage IV hours and the saved networks, and the count of onsets warned is a coarse test (another onset's rain gives nearly as many as the right rain). What it does not: that the stored hours are what a service would have had at midnight (whether the archive holds first postings or later gauge-corrected versions was not established in any review), that networks trained on Stage IV would behave the same, and the raw archive reduction."),
    "rain_clock": (f"Recomputed from the ERA5 hourly cache with the critic's own code: all {D4['compared']} summary fields of results/onset_rain_clock.json equal (to 1e-9) and the 611 per-onset rows equal (largest difference {D4['per_onset_rows_max_abs_difference_mm']['total']:.1e} mm); the shares of the mean rain are "
        f"{pc(D4['quoted_shares']['share_before_0900_of_the_issue_day_window_table'])} / {pc(D4['quoted_shares']['share_0900_to_midnight_of_the_issue_day'])} / {pc(D4['quoted_shares']['share_on_the_target_day'])}; of the {D4['onsets_missed_by_lstm_wq']} onsets missed by the LSTM with the river gauge the median share of the rain known by the table is {rain_miss['median_share_known_by_table']:.1%} and by midnight {rain_miss['median_share_known_by_midnight']:.0%}. "
        "One wording point (S14, closed in the final files): the four windows cover 63 hours (24 + 15 + 9 + 15); the judged build said '48 hours' (report p.13, the script's docstring, the 'what' of the results file); the final report, the script and the regenerated results file say 63 hours, and the critic's code reproduces all 121 fields and the 611 rows of the regenerated file. The ERA5 cache itself and its hour-ending convention were taken as stored."),
    "resampling_ranges": (f"Redrawn with the critic's own random numbers (4,000 draws of the 9 test water years and 12 basins): all {len(ranges)} statistics redrawn (the ranges that the documents quote, the NSE differences and the verdict shares of the draws) equal the stored ones within {MAXD['shares_points']:.1f} point ({MAXD['nse']:.4f} NSE; verdict shares {MAXD['verdict_shares_points']:.1f} point), and the point estimates equal at 1e-9. One scope point: 'the verdict NOT YET for onsets holds in every draw at every lead time, also with rain up to the evening' is true for the LSTM with the river gauge and with the rain up to the evening "
        "(100% of the draws at 1, 2 and 3 days) but not for the new sets: AORC 1 day NOT YET in 91.8% of the stored draws (91.5% of mine; CAUTION in the rest), the forecast set 1 day in 85.6% (mine 85.9%); no draw reaches GO (S15: partly closed in the final files: the report, p.27, now gives 92% and 86%; Q&A D11, p.13, still has the unscoped sentence)."),
    "key_figure": (f"results/key_figure.json holds: {D5['F_counts_and_hashes']['sources']} sources, all re-read, cross-checked and recomputed ({D5['F_counts_and_hashes']['my_comparisons']} comparisons in all, none differing by more than {max(D5['B_cross_checks_read_again']['largest_difference'], D5['C_computed_recomputed']['largest_difference']):.0e}), {D5['D_rescored_from_stored_forecasts']['n']} scores re-derived from the stored forecasts; "
        f"the {E['labels_read']} printed bar labels of the three pictures equal their sources as printed; 'mixed' appears beside exactly the three NSE values whose stored reading is mixed (rain up to the evening 1 day, the forecast 2 and 3 days). The positions of the marks were not measured on the pixels and the drawing script was not run."),
    "what_may_the_documents_say_about_this_review": "REVIEW_WORDING says that the real-time check on all onsets, the rain-clock diagnostic, the resampling ranges and the key figure of 8 October 2026 were 'verified by the final critic agent only'. That is supported with these words: recomputed with the critic's own code, from the stored Stage IV hourly series and the saved networks (not from the raw Stage IV archive), from the ERA5 hourly cache as stored, "
        "for the ranges the documents quote (not all 1,884 statistics of results/uncertainty.json), and for the key figure's sources and printed labels (not the marks' positions); by one AI agent, not by the team, and the team's own checks stay open.",
    "is_the_set_ready_to_hand_in": ("Not at 12:30, when the documents were judged: the numbers were sound and eight wording items (M1-M8) had to be fixed and the set rebuilt. After the rebuild (round 2, final files of 13:48-13:50): yes, as the project describes itself, an AI-built and AI-reviewed negative result with the team's own checks still open and said to be open in every document. "
                                   f"All eight must-fix items are closed; of the 17 should-fix items {len(S_CLOSED)} are closed, {len(IDS_BY['partly closed'])} partly ({', '.join(IDS_BY['partly closed'])}) and {len(IDS_BY['open'])} open ({', '.join(IDS_BY['open'])}); every open item is wording or layout, none changes a number or a verdict (see round_2)."),
}
verdict = (f"The numbers hold up under independent recomputation, and the four items that the documents attribute to the final critic agent are supported. All 69 sampled numbers of the documents equal the results files (c1); the headline scores of 14 prediction sets equal flood.evaluate in all {PA['compared_with_flood_evaluate']} numbers (c2); "
    f"the {len(ranges)} resampling statistics that I redrew (the ranges the documents quote and verdict shares) are reproduced by 4,000 draws of my own within {MAXD['shares_points']:.1f} point ({MAXD['nse']:.4f} NSE, {MAXD['verdict_shares_points']:.1f} point for the verdict shares) and their point estimates exactly (c3); the real-time check on all 611 onsets is reproduced from the stored Stage IV hours and the saved networks "
    f"(294 / 275 / 255 onsets, 90% of the AORC effect, the 12-basin false-warning tally, the second set of networks, the controls; {D3['n_compared']} numbers equal, c4-c7c), the rain clock from the ERA5 cache (121 fields and 611 rows equal, c8-c9), and the key figure's {D5['F_counts_and_hashes']['my_comparisons']} comparisons all agree (c10-c15). "
    "I found no numerical discrepancy and no failed check. I found two wording points while recomputing: the rain clock spans 63 hours, not 48 (S14), and 'the verdict NOT YET for onsets holds in every draw' is true for the LSTM with the river gauge and with the rain up to the evening but not for the AORC and forecast sets, "
    "where CAUTION appears in 8% and 14% of the draws at 1 day (S15). The document pass of 10:23-12:30 found eight must-fix wording items (M1-M8) and thirteen should-fix items (S1-S13); S14 and S15 were found while recomputing and S16 and S17 in the first look at the rebuilt files. "
    f"Round 2 on the final files (13:48-13:50 builds): all eight must-fix items are closed; of the 17 should-fix items {len(S_CLOSED)} are closed, {len(IDS_BY['partly closed'])} partly ({', '.join(IDS_BY['partly closed'])}) and {len(IDS_BY['open'])} open ({', '.join(IDS_BY['open'])}), all wording or layout, none changing a number or a verdict (round_2). "
    "What this record does not establish: that the stored Stage IV hours are what a service would have had at midnight, anything about the raw Stage IV archive, or that networks trained on Stage IV would behave as those trained on AORC (see not_checked). It is one AI agent's recomputation; the team's own checks are still open.")
not_checked = [
    "No network was trained or retrained: the three saved lstm_wqa networks were only run forward. Whether a new training with the same seeds reproduces them (thread count, operating system) is untested; the saved networks and forecasts are what was checked.",
    "The raw Stage IV archive was not re-read: the check starts from the stored hourly series data/aorc/stage4_issue_days.csv (14,664 hours of the issue days of the 611 onsets; 2 hours replaced by the AORC hour). Its reduction from the archive files, the basin means and hour labels, and whether the archive holds first postings or later corrected versions ('real time') were not examined; the E1b review lists the last point as not established, and it still is.",
    "Networks trained on Stage IV, and false warnings on days that are not issue days of an onset, were not tested (the 12-basin tally covers the 368 issue days only): the real-time check is a check on the test years with networks trained on AORC, not a service.",
    "The ERA5 hourly cache (data/hourly/era5_*.json), its hour-ending convention and its clock (UTC) were used as stored; the rain clock was not repeated with another product or at another time convention. The AORC hourly series (data/aorc/aorc_*.npz) was used as stored (its fetch was checked in the E1b review).",
    "Resampling: only the statistics the documents quote were redrawn (the design 'both': years and basins together), not the other statistics of results/uncertainty.json (1,884 in all), not the designs 'years only' and 'basins only', and not the spread-and-skill tables.",
    "Key figure: 11_key_figure.py was not run or redrawn; the printed labels and lettering were read by eye and by OCR, the positions of the marks (bars, dots, single-seed ranges, the other kinds of model) were not measured on the pixels; the file's 159 'drawn' numbers include positions that are not printed.",
    "Documents: the PPTX was not opened in PowerPoint; the printed colour of the PDFs was not checked on paper; the HTML versions were checked as text. The literature numbers of the cited papers were not re-read. The numbers checks c1-c15 are of the versions judged (hashes in files_reviewed.documents_judged); round 2 compared the final documents with them at the level of words and numbers (grep, a token comparison, c16, and the reading of every changed passage) and looked at contact sheets of every page at 70 dpi and at zooms of report p.1 and p.17 and slides 10-13; it did not re-read every unchanged sentence of the 31-page report, the 25-page Q&A or the 37-page guide.",
    "Everything here is an AI agent's recomputation. It does not replace the team's own check of the experiments, the figures or the documents (team.json is untouched and all check_done flags are false).",
]

FINAL_PAGES = {"Flood_Report.pdf": 31, "Flood_Slides.pdf": 20, "Professor_QA.pdf": 25, "Project_Guide.pdf": 37, "Project_Overview.pdf": 1, "Project_in_Plain_Words.pdf": 1, "Presentation_Plan.pdf": 3}
FINAL_NAMES = ["Flood_Report.pdf", "Flood_Report.html", "short_report.md", "README.md", "PLAN.md", "Flood_Slides.pdf", "Flood_Slides.pptx", "Flood_Slides.html", "slides_notes.md", "Professor_QA.pdf", "Professor_QA.md", "Professor_QA.html",
               "Project_Guide.pdf", "Project_Guide.html", "Project_Overview.pdf", "Project_Overview.html", "Project_in_Plain_Words.pdf", "Presentation_Plan.pdf", "Presentation_Plan.md"]
final_files = {n: dict(sha256=sha(PROJ / n), bytes=(PROJ / n).stat().st_size, modified=stamp(PROJ / n), **({"pages": FINAL_PAGES[n]} if n in FINAL_PAGES else {})) for n in FINAL_NAMES}
round_2 = {
    "when": f"2026-10-08, 13:51 to about {time.strftime('%H:%M')} (local time, UTC+4), on the final files: documents built 13:48-13:50 (the report rebuilt once more at 13:50 with PAGE_LIMIT raised to 31: 31 pages); a provisional look at the files of 13:11-13:21 was made at about 13:40-13:50 and sent to the coordinator",
    "files_judged": final_files,
    "method": ["r2_check.py: the text of every final PDF per page (PDFKit), the Markdown by paragraph, the HTML without tags, the PPTX with its notes, searched for the patterns of M1-M8 and quick patterns for S5, S8, S10-S12, S14, S15; every hit read",
               "r2_ctx.py: the passages read in context",
               "r2_new_numbers.py: the numbers that are new in each final document against the version judged, each traced to a results file (check c16)",
               "contact sheets of every page of the report (31), the slides (20), the Q&A (25) and the guide (37) at 70 dpi; zooms of report p.1 and p.17 (130 and 230 dpi) and of slides 10-13; the overview and the plain-words page at 70 dpi",
               "the PPTX against the PDF (numbers of each of the 20 slides, tables included); the plan's arithmetic; the speaking rate of the notes",
               "d1-d5 and d2a rerun at 13:52-14:00 against the regenerated results files (results/onset_rain_clock.json and .csv and code/check_onset_rain_clock.py changed in the rebuild): d4 reproduces all 121 fields and 611 rows again; d3 all 161 numbers; nothing else changed"],
    "counts": dict(must_fix_closed=sum(f["status_short"] == "closed" for f in must), must_fix=len(must), should_fix_closed=sum(f["status_short"] == "closed" for f in should), should_fix_partly_closed=sum(f["status_short"] == "partly closed" for f in should),
                   should_fix_open=sum(f["status_short"] == "open" for f in should), should_fix=len(should)),
    "closed": IDS_BY["closed"], "partly_closed": IDS_BY["partly closed"], "open": IDS_BY["open"],
    "items": {f["id"]: dict(status=f["status_short"], title=f["title"], evidence=f["round_2"]["evidence"]) for f in FINDINGS},
    "new_in_round_2": [
        "Page 1 of the final report no longer has the paragraph 'What it took, and what is missing' (it stood on page 1 of the judged build and of the 13:40 build, with 'five trainings in place of three' added): the final layout pass put the 'A river never seen in training' sentences there; short_report.md keeps the paragraph. Page 1 still says that the verdict for onsets is NOT YET, that the warnings 'cannot be trusted yet' and that 'the team members' own checks are still to be done (section 9)'; the open items (the ensemble, a network trained on Stage IV, five trainings) are in section 8 (p.29), slide 12, the Q&A and the guide. Not blocking; one line would restore it (see S3).",
        "The report has 31 pages (30 before; Table 8.1 now stays on p.28), the Q&A 25 (23 before), the slides 20, the guide 37, the overview and the plain-words page one page each, the plan three; no blank page, orphan heading or cut-off text was seen on the contact sheets; Figure 4.6's top edge is clean.",
        "No number in the final documents is new without a source (c16); the PPTX and the PDF carry the same numbers on all 20 slides; the plan's times add up to 600 s and its checkpoints equal the sums; the speaking rate of the notes is 2.17 to 2.39 words per second by whitespace count (the plan says 'at most 2.3': within 0.1).",
        "The review sentence now cites results/engine_review.json as well as results/review_*.json in the report, the Q&A, the guide, the short report, the notes and the README (the overview and the slides carry the short form without file names), as asked in the provisional round 2."],
    "verdict": (f"Round 2, on the final files. The set is ready to hand in as the project describes itself: an AI-built and AI-reviewed negative result ('among the models tried, the limit is the information in the inputs, not the network'; for onsets still NOT YET), with the team members' own checks still to be done and said to be open in every document. "
                f"All eight must-fix items are closed (M1 placeholder, M2 stale 'not tested', M3 NSE of the forecast, M4 real-time wording, M5 AORC wording, M6 hindsight beside the 51%, M7 review coverage, M8 Q&A G5) and no new must-fix item was found. Of the 17 should-fix items {len(S_CLOSED)} are closed, {len(IDS_BY['partly closed'])} are partly closed "
                f"({', '.join(IDS_BY['partly closed'])}) and {len(IDS_BY['open'])} are open ({', '.join(IDS_BY['open'])}): S3 and S15 because page 1 of the report lost its 'what is missing' paragraph and because Q&A D11 still has the unscoped sentence 'the verdict NOT YET for onsets holds in every resample at every lead time', S2 because the lettering of the new figures is still 3 to 6 pt in print, "
                "S13 because the README counts the four experiments differently from the report, and S16 because slide 11 no longer says that the shuffled control measures harm or that the E2b numbers are comparable only inside their own split. All of these are wording or layout; none changes a number, a verdict or a claim of the findings, and none should stop the hand-in. "
                "Three one-line fixes would close all but S2: two clauses on slide 11 (S16), one sentence in Q&A D11 (S15) and one sentence in the README 'Who did what' (S13). The numbers of the final documents were compared with the judged build and every new number was traced to a results file (c16); the recomputations (d1-d5) were rerun on the regenerated files and still agree. "
                "This is one AI agent's judgment; it does not replace the team's own checks."),
}
rec = {
    "who": "the final critic of the document set of 8 October 2026: one independent AI agent (Claude), started separately with its own brief (judge the rebuilt documents; report findings only, no project file edited; at about 12:40 the coordinating agent allowed it to write this file and the folder code/review/documents_2026-10-08/ and nothing else); the documents and the code behind them were written and run by other AI agent sessions (Claude Code) directed by the team lead",
    "human_review": "not yet done: the team's own check of the documents, of the real-time check, the rain clock, the ranges and the key figure is still open (team.json untouched by this review)",
    "when": f"2026-10-08: the document pass from about 10:23 to 12:30 (CRITIQUE.md), the recomputations of this record from about 12:42 (load average up to 58 on two cores while other agents rebuilt documents on the same laptop), round 2 on the final files from 13:51 to {time.strftime('%H:%M')} (local time, UTC+4)",
    "what": ("independent recomputation, with the critic's own code, of what the final document set of 8 October 2026 quotes and of the files behind four of its items: (1) 69 numbers of the documents against the results files, page by page; (2) the headline scores of 14 prediction sets from the stored forecasts (against flood.evaluate); "
             "(3) the resampling ranges of results/uncertainty.json that the documents quote (own 4,000 draws); (4) the real-time check of results/stage4_realtime.json (code/check_stage4_realtime.py) on all 611 onsets: own AORC day table, own forward pass of the three saved lstm_wqa networks, the stored Stage IV hours substituted on the issue day; "
             "(5) the rain-clock diagnostic of results/onset_rain_clock.json (code/check_onset_rain_clock.py) from the ERA5 hourly cache; (6) the key figure (results/key_figure.json and figures/key_figure.png, key_figure_plain.png, key_figure_talk.png); and, by reading, the wording rules, honesty, consistency, layout and internal logic of the documents "
             f"(the six checks of the critique: {len(must)} must-fix and {len(should)} should-fix items); and round 2, the final files of 13:48-13:50 against every item. Nothing was trained."),
    "files_reviewed": dict(documents_judged=docs_judged, **reviewed),
    "method": dict(
        own_code_folder="code/review/documents_2026-10-08/ (in the project; README.md there explains how to run it) and the session scratchpad for the outputs",
        run_with="from the project root with ~/AI_Class/.venv/bin/python, one thread (FLOOD_THREADS=1, torch.set_num_threads(1)), PYTHONDONTWRITEBYTECODE=1; run_all.sh runs d5, d4, d1, d2, d2a, d3 in that order; d9_assemble_review.py builds this file",
        nothing_trained=True, nothing_installed=True, network_use="none",
        project_writes="only this file and the folder code/review/documents_2026-10-08/ (no bytecode, no cache); the project's numbered scripts were not run",
        engine_used=["flood.load_data (the daily CAMELS tables: rain, sun, temperatures, humidity, day length, flow)", "flood.basins (the ten static properties of the basins, for the input matrix of d3)", "flood.thresholds and flood.evaluate and flood.load_preds (only to compare my own scoring and levels with the engine's: d2, d2a, d3 assertion)"],
        own_code_for=["the high-flow levels, the onsets and the onset warning tables from results/preds/*.npz (d2, d4, d5)", "the AORC day sums by the local wall-clock day and the Stage IV day sums (d3)", "the input matrices from the scalers stored in the saved networks and the LSTM forward pass with torch.nn.LSTM (d3)",
                      "the rain-clock windows in local standard time from the ERA5 hourly files (d4)", "the bootstrap of the 9 water years x 12 basins, the verdict shares and the NSE differences (d2)", "the path reader and the recomputation of the key figure's sources (d5)", "the text search of the 69 claims in the PDF pages (d1)"],
        tools_for_reading=["pdf2txt.swift (PDFKit text per page)", "ocr.swift (Apple Vision text recognition of the three key-figure pictures)", "python-pptx and lxml for the slides", "PIL contact sheets and crops of rendered pages (tools/pdf2png)"],
        thresholds_of_agreement="scores and counts: equal (1e-9 for floats; 1e-12 for the key figure's values); resampling intervals: within 0.0105 (about one point); interval of the surviving share: within 0.01"),
    "checks": checks,
    "answers": answers,
    "numbers": numbers,
    "discrepancies": [
        f"No numerical discrepancy: every number of results/stage4_realtime.json, results/onset_rain_clock.json, results/key_figure.json and the headline scores of the documents that I recomputed equals my value (0.0 against the engine for the scores; at most 6e-17 for the key figure's sources; the rain-clock rows to 1.4e-14 mm; my resampling intervals within {MAXD['shares_points']:.2f} point ({MAXD['nse']:.4f} NSE; verdict shares {MAXD['verdict_shares_points']:.1f} point) of the stored ones, my point estimates equal; the Stage IV counts exactly, the controls within their random range).",
        "Wording against numbers, rain clock: the windows cover 63 hours (24 + 15 + 9 + 15), the report (p.13), the script's docstring and the results file's 'what' say '48 hours' (S14). The numbers are right for the windows.",
        "Wording against numbers, ranges: 'the verdict NOT YET for onsets holds in every draw at every lead time' (report p.27, Q&A D11 p.11) is true for the sets named in results/uncertainty.json (the LSTM with the river gauge, with the rain up to the evening) and not for the AORC set (NOT YET in 91.8% of the stored draws, 91.5% of mine, at 1 day) or the forecast set (85.6%, mine 85.9%; 99.05% and 99.65% at 2 and 3 days); no draw reaches GO (S15).",
        "Robustness, not a discrepancy: four of the rain clock's 121 summary fields sit on a boundary and moved with a summation order that differs by 3e-11 mm (my first run, with differences of cumulative sums, against the slice sums that the builder's file equals): 'the share of onsets with half the rain known by the table' (15.9% against 16.0% of all onsets, 6.8% against 7.0% of the 497 missed, 75.0% against 76.9% of the 0-25% class; two onsets, 02298608 on 2014-08-09 and 02349900 on 2009-03-29, have exactly half of their rain in the table window and the test is >= 0.5 on a floating-point ratio) and the median share known by the table of the missed onsets (3.7% against 3.8%: the median is exactly 3.75%). The headline of the clock (the shares of the mean rain, the medians by midnight) is not affected.",
    ],
    "must_fix": must,
    "should_fix": should,
    "observations": [
        f"Stage IV against AORC on the issue days of the 611 onsets: mean rain {D3['mean_rain_mm_of_the_issue_days']['aorc']:.2f} mm (AORC) against {D3['mean_rain_mm_of_the_issue_days']['stage4']:.2f} mm (Stage IV), k = {D3['K_aorc_over_stage4']:.4f}, correlation {D3['agreement_of_the_two_products']['correlation']:.3f}; Stage IV x k warns of {V['stage4_k']['onsets_warned']} onsets, {V['stage4_k']['onsets_warned'] - D3['reproduction']['onsets_warned_aorc_mine']} more than AORC: the loss of {D3['reproduction']['onsets_warned_aorc_mine'] - V['stage4']['onsets_warned']} onsets with raw Stage IV is recovered by restoring AORC's level (the radar product reads 6% less rain on these days), at the price of more false warnings (12-basin tally: {D3['issue_days_of_all_12_basins']['stage4_times_k']['false_warnings']} against {D3['issue_days_of_all_12_basins']['stage4']['false_warnings']}).",
        f"Dropping the last hour(s) of the issue day costs about the same with either product: AORC {D3['reproduction']['onsets_warned_aorc_mine']} to {V['aorc_no_last_hour']['onsets_warned']} to {V['aorc_no_last_two_hours']['onsets_warned']} onsets, Stage IV {V['stage4']['onsets_warned']} to {V['stage4_no_last_hour']['onsets_warned']} to {V['stage4_no_last_two_hours']['onsets_warned']}; the one-hour delay of a real-time product (imitated by dropping the last hour) costs 17 to 19 onsets, about 3 points, whichever product is used.",
        f"On the issue days of all 12 basins Stage IV gives fewer false warnings than AORC ({D3['issue_days_of_all_12_basins']['stage4']['false_warnings']} against {D3['issue_days_of_all_12_basins']['aorc']['false_warnings']}; {pc(D3['issue_days_of_all_12_basins']['stage4']['far'])} against {pc(D3['issue_days_of_all_12_basins']['aorc']['far'])} of the warnings) and fewer hits ({D3['issue_days_of_all_12_basins']['stage4']['hits']} against {D3['issue_days_of_all_12_basins']['aorc']['hits']}): partly a moved operating point, not only a loss; Stage IV x k gives {D3['issue_days_of_all_12_basins']['stage4_times_k']['hits']} hits and {D3['issue_days_of_all_12_basins']['stage4_times_k']['false_warnings']} false warnings.",
        "The check on the stored forecasts is bit for bit: my own AORC day table (from the hourly series, local wall-clock days, summer time included) and my own LSTM forward pass give the stored single-seed and seed-average forecasts with a largest difference of 0.0 mm/day, which also confirms that the stored networks were trained on that very table and that the scalers in the .pt files are the ones used.",
        f"Verdict shares in my draws for the onsets (at the high-flow level, 4,000 draws): AORC 1 day NOT YET {PB['onset_verdict_shares_in_the_draws']['lstm_wqa.l1']['NOT YET']:.1%} (CAUTION {PB['onset_verdict_shares_in_the_draws']['lstm_wqa.l1']['CAUTION']:.1%}); the forecast set 1 / 2 / 3 days NOT YET {PB['onset_verdict_shares_in_the_draws']['lstm_wqaF_sub.l1']['NOT YET']:.1%} / {PB['onset_verdict_shares_in_the_draws']['lstm_wqaF_sub.l2']['NOT YET']:.1%} / {PB['onset_verdict_shares_in_the_draws']['lstm_wqaF_sub.l3']['NOT YET']:.1%}; GO in none.",
        "The key figure's three renderings print the same numbers where they print the same rows (the plain one leaves out two rows, the talk one has its own omissions: 34, 30 and 26 printed bar labels); the asterisk of the Transformer is explained on report p.1 and slide 13.",
    ],
    "round_2": round_2,
    "counts": dict(checks_listed=n_listed, checks_passed=n_checks_pass, findings_not_pass_fail=n_find, failed=0, must_fix=len(must), should_fix=len(should), numerical_discrepancies=0,
                   numbers_compared_in_all=D1["n_claims"] + PA["compared_with_flood_evaluate"] + len(ranges) + D3["n_compared"] + D4["compared"] + 611 + D5["F_counts_and_hashes"]["my_comparisons"]),
    "verdict": verdict,
    "scripts": dict(folder="code/review/documents_2026-10-08/ (in the project; README.md there explains how to run them)",
                    run_with="from the project root with ~/AI_Class/.venv/bin/python, one thread, nothing trained, nothing written into the project (outputs go to $REVIEW_OUT); run_all.sh, then d9_assemble_review.py [--write-into-project]",
                    seconds_of_the_final_runs=dict(d1_numbers=None, d2_headline_and_ranges=D2["seconds"], d3_stage4=D3["seconds_total"], d4_rain_clock=D4["seconds"], d5_key_figure=D5["seconds"], note="laptop, one thread, load average 20-58 from other agents, so longer than on an idle machine"),
                    script_hashes={p.name: sha(p)[:16] for p in sorted(HERE.glob("*")) if p.suffix in (".py", ".swift", ".sh", ".json", ".md") and p.name != "d9_assemble_review.py"} | {"d9_assemble_review.py": "(this script)"}),
    "not_checked": not_checked,
    "dates": dict(critic_document_pass="2026-10-08 10:23 to about 12:30", recomputations="2026-10-08 from about 12:42 (rerun of d1-d5 at 13:52-14:00 on the regenerated results files)", round_2=f"2026-10-08 13:51 to about {time.strftime('%H:%M')} on the final files (documents built 13:48-13:50)", documents_judged="the build of 10:21-10:24 (report, Q&A, guide, overview, short report), the slides of 11:30, README.md of 11:31, the plan and plain-words page of 11:32 (hashes in files_reviewed.documents_judged)",
                  builders_rebuild="13:11-13:21 (first rebuild), 13:48-13:50 (final: the report once more at 13:50 with PAGE_LIMIT 31)", this_file=time.strftime("%Y-%m-%d %H:%M:%S")),
    "status_of_the_document_items": {f["id"]: f["status_short"] + " (round 2, final files of 13:48-13:50)" for f in FINDINGS},
}
rec["scripts"]["seconds_of_the_final_runs"].pop("d1_numbers")
out_path = OUT / "review_documents_2026-10-08.json"
with open(out_path, "w") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o))
print("written", out_path, out_path.stat().st_size, "bytes;", f"{n_listed} checks listed, {n_find} findings, {len(must)} must-fix, {len(should)} should-fix")
if WRITE:
    dst = RES / "review_documents_2026-10-08.json"
    dst.write_text(out_path.read_text())
    print("written into the project:", dst)
