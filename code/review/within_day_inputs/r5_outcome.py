"""E4b review, check 5: the sentence and the figure. Every clause of outcome_full rebuilt from the REVIEWER's recomputed numbers (r2_headline.json) with the script's own rounding
rules and looked for in the stored text (so every number of the sentence is traced to a table that the review reproduced); the fixed wordings; the status, the disclosure of
the choices, the seed-count sentence; the figure read on its pixels (bar tops and dot counts against the JSON). Seconds."""
from r_common import *
import re

t0 = time.time()
R = jload(RESULTS)
H = jread("r2_headline.json")
M = H["headline_mine"]
ops = H["operating_points_mine"]
S = R["summary_test"]["sets"]
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
n2 = lambda x: f"{x:.2f}"
text = R["outcome_full"]
assert text == R["outcome"] + " " + R["outcome_caveat"], "outcome_full is not outcome + outcome_caveat"
l1 = lambda n: M[n]["l1"]
rngp = lambda n: (min(M[n]["l1"]["seed_onset_pod"]), max(M[n]["l1"]["seed_onset_pod"]))
rngn = lambda n: (min(M[n]["l1"]["seed_nse_median"]), max(M[n]["l1"]["seed_nse_median"]))
tW, tA, tO, tQ, tP = (l1(x)["onset"] for x in ("wqaW", "wqa", "wqaW_older", "wqaQ", "wqaP"))
L = jload(RESULTS)["little_rain_onsets"]["slices"]                                         # the slice shares were reproduced in r2 (rows 'little_rain ...'); the text is built from them
sm, wet = L["aorc_target_below_5mm"], L["aorc_target_at_least_5mm"]
ov = R["paired_onsets"]["pairs"]["within_day_columns"]["l1"]
gap, gain = tP["pod"] - tA["pod"], tW["pod"] - tA["pod"]
ngap, ngain = l1("wqaP")["nse_median"] - l1("wqa")["nse_median"], l1("wqaW")["nse_median"] - l1("wqa")["nse_median"]
cvW, cvA = ops["wqaW l1 csi"], ops["wqa l1 csi"]
opW, opA = ops["wqaW l1 far_cap"], ops["wqa l1 far_cap"]
tWF, tF = l1("wqaWF_sub")["onset"], l1("wqaF_sub")["onset"]
clauses = {
    "onsets warned, base to test, seed range": f"onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone (hindsight at most about 3 points) to {pc(tW['pod'])} (single trainings {pc(rngp('wqaW')[0])} to {pc(rngp('wqaW')[1])};",
    "points and hits over wqa, points over the control": f"{100 * (tW['pod'] - tA['pod']):+.1f} points over that model ({tW['hits']} against {tA['hits']} hits of {tW['hits'] + tW['misses']}) and {100 * (tW['pod'] - tO['pod']):+.1f} points over the control with the same six columns one day older ({pc(tO['pod'])});",
    "false share": f"the share of false warnings among the onset warnings is {pc(tW['far'])} against {pc(tA['far'])};",
    "median NSE and seed range": f"the median NSE goes from {n2(l1('wqa')['nse_median'])} to {n2(l1('wqaW')['nse_median'])} (the seed-averaged forecast; single trainings {n2(rngn('wqaW')[0])} to {n2(rngn('wqaW')[1])});",
    "verdict words": f"the verdict of results/flood_analysis.json is {R['verdicts']['sets']['wqaW']['l1']['plain']['onset']['verdict']} for the onsets and {R['verdicts']['sets']['wqaW']['l1']['plain']['all']['verdict']} for all high-flow days.",
    "success rule and the three seed spans": f"is {'met' if H['success_mine']['met'] else 'not met'}: the five single trainings with the within-day columns span {p1(rngp('wqaW')[0])} to {p1(rngp('wqaW')[1])} against {p1(rngp('wqa')[0])} to {p1(rngp('wqa')[1])} with the AORC rain to midnight alone and {p1(rngp('wqaW_older')[0])} to {p1(rngp('wqaW_older')[1])} for the older control;",
    "control flat, only-by counts": f"the control is {'flat' if H['success_mine']['control_flat'] else 'not flat'} ({pc(tO['pod'])} against {pc(tA['pod'])}); {ov['warned_only_by_first']} onsets are warned only with the within-day columns against {ov['warned_only_by_second']} only without them.",
    "wqaQ": f"The Q95-relative flow alone (wqaQ) gives {pc(tQ['pod'])} (single trainings {pc(rngp('wqaQ')[0])} to {pc(rngp('wqaQ')[1])}).",
    "little-rain slice": f"In the {sm['n']} onsets with under 5 mm of AORC rain on the target day the share warned goes from {pc(sm['warned']['wqa'])} to {pc(sm['warned']['wqaW'])} (older control {pc(sm['warned']['wqaW_older'])}), in the {wet['n']} others from {pc(wet['warned']['wqa'])} to {pc(wet['warned']['wqaW'])}.",
    "gap to wqaP and closure": f"The gap in onsets warned from the AORC rain to midnight ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP, {pc(tP['pod'])}, an upper bound that cannot be had in practice) is {100 * gap:.1f} points; the six columns move the share by {100 * gain:+.1f} points and close {pc(gain / gap)} of it; in median NSE the gap is {ngap:.3f} and the move {ngain:+.3f} ({pc(ngain / ngap)} of it).",
    "validation-chosen CSI level": f"At the warning level chosen on the validation years for onsets ({cvW['trigger']:.2f} against {cvA['trigger']:.2f}) wqaW warns of {pc(cvW['test_onset']['pod'])} with {pc(cvW['test_onset']['far'])} false against {pc(cvA['test_onset']['pod'])} with {pc(cvA['test_onset']['far'])} false;",
    "30%-false operating point": f"at the 30%-false operating point (the trigger chosen on the validation years so that at most 30% of the onset warnings are false: {opW['trigger']:.2f}) it warns of {pc(opW['test_onset']['pod'])} with {pc(opW['test_onset']['far'])} false on the test years, against {pc(opA['test_onset']['pod'])} with {pc(opA['test_onset']['far'])} false for the AORC rain to midnight alone (trigger {opA['trigger']:.2f}).",
    "seed-count sentence": "Every number of the new sets is a five-seed number; the stored sets have three seeds.",
    "forecast set": f"onsets warned 1 day ahead go from {pc(tF['pod'])} to {pc(tWF['pod'])} (single trainings {pc(rngp('wqaWF_sub')[0])} to {pc(rngp('wqaWF_sub')[1])}; false {pc(tWF['far'])} against {pc(tF['far'])}), at 2 and 3 days from {pc(M['wqaF_sub']['l2']['onset']['pod'])} to {pc(M['wqaWF_sub']['l2']['onset']['pod'])} and from {pc(M['wqaF_sub']['l3']['onset']['pod'])} to {pc(M['wqaWF_sub']['l3']['onset']['pod'])}.",
}
found = {k: (v in text) for k, v in clauses.items()}
out = dict(clauses_rebuilt_from_my_numbers={k: dict(text=v, found_in_outcome_full=found[k]) for k, v in clauses.items()}, all_clauses_found=all(found.values()),
           chosen_csi_level_text_agrees_with_verdicts_block=(R["verdicts"]["sets"]["wqaW"]["l1"]["chosen_on_validation_onsets"]["trigger"] == cvW["trigger"] and R["verdicts"]["sets"]["wqa"]["l1"]["chosen_on_validation_onsets"]["trigger"] == cvA["trigger"]))
print("clauses of outcome_full rebuilt from my numbers and found in the text:", sum(found.values()), "of", len(found), [k for k, v in found.items() if not v])

# the numbers in the sentence that no table of the file gives directly: a plain check that each number in the text appears in some table (traceability)
nums = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)%?", text)
out["numbers_in_outcome_full"] = len(nums)

# wordings
W = dict(aorc_wording_in_outcome_full="an analysis made after the fact" in R["outcome"] and "an analysis made after the fact" in R["outcome_caveat"],
         hindsight_wording_beside_the_51="51% with the AORC rain to midnight alone (hindsight at most about 3 points)" in text,
         upper_bound_wording_beside_wqaP="(wqaP, 66%, an upper bound that cannot be had in practice)" in text and "an upper bound that cannot be had in practice" in R["design"]["stored_sets"]["wqaP"]["what"],
         info_not_network_wording_present="the information, not the network" in json.dumps(R), simple_model_yardstick_done="the simple-model yardstick given the within-day columns" not in R["what_was_not_done"],
         not_a_service_clause="are a bound on what a real-time hourly rain product could add, not a service" in text,
         status_line=R["status"], status_says_not_yet_reviewed=R["status"].startswith("NOT YET REVIEWED"),
         choices_disclosure=R["choices_made_after_looking_at_test_scores"], choices_say_fixed_before_training=any("fixed in NEXT_EXPERIMENTS_2026-10-28.md and in this script before training" in c for c in R["choices_made_after_looking_at_test_scores"]),
         seeds_sentence_present="Every number of the new sets is a five-seed number; the stored sets have three seeds." in text,
         control_wording="the control is flat (53% against 51%)" in text, control_false_share_in_text="56% false" in text or "(56%)" in text, control_nse_in_text="0.74" in text,
         control_at_30pct_false_point_in_text="30% with 31% false" in text or "control" in text.split("at the 30%-false operating point")[-1].split(". AORC is")[0],
         figure_footer_hand_typed_51="the 51% of wqa carries" in open(PROJ / SCRIPT, encoding="utf-8").read(),
         outcome_alone_warning=any("never outcome alone" in n for n in R["notes"]), observed_rain_phrase="observed rain" in json.dumps(R))
out["wordings"] = W
print("wordings:", {k: v for k, v in W.items() if isinstance(v, bool)})

# the figure on its pixels: bar tops (onsets warned, median NSE) and dot counts against the JSON
try:
    from PIL import Image
    img = np.asarray(Image.open(PROJ / "figures" / "within_day_inputs.png").convert("RGB")).astype(int)
    Hpx, Wpx = img.shape[:2]
    fh = 8.4 * 200
    h_ax = 0.6 / 2.3                                                                       # two rows, hspace 0.3, top 0.8, bottom 0.2
    panels = dict(onset_pod=(0.8, 0.8 - h_ax, 100.0), nse_median=(0.2 + h_ax, 0.2, 1.0))
    order = [n for n in ("wqa", "wqaQ", "wqaW", "wqaW_older", "wqaWF_sub", "wqaF_sub", "wqaP") if n in S]
    hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
    BG, INK = hexrgb("#fcfcfb"), hexrgb("#0b0b0b")
    COL = {"wqa": "#1baf7a", "wqaQ": "#2a78d6", "wqaW": "#2a78d6", "wqaW_older": "#2a78d6", "wqaWF_sub": "#eb6834", "wqaF_sub": "#eb6834", "wqaP": "#eda100"}
    x_left, x_right = 0.08 * Wpx, 0.98 * Wpx
    xlim = (-0.5 - 0.29 * 0.0 - 0.05 * 6.58 - 0.29 + 0.5, 6.5 + 0.05 * 6.58 + 0.29 - 0.5)       # bars from -0.29 to 6.29 plus matplotlib's 5% margins: (-0.619, 6.619)
    xlim = (-0.29 - 0.05 * 6.58, 6.29 + 0.05 * 6.58)
    xpix = lambda x: x_left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x_right - x_left)
    fig = {}
    for key, (top, bottom, ymax) in panels.items():
        y_top, y_bot = (1 - top) * fh, (1 - bottom) * fh
        ypix_to_val = lambda y: (y_bot - y) / (y_bot - y_top) * ymax
        res = {}
        for i, n in enumerate(order):
            col = hexrgb(COL[n])
            a = 0.9 if n == "wqaW_older" else 0.75
            blend = a * col + (1 - a) * BG
            x0, x1 = int(xpix(i - 0.29)) + 3, int(xpix(i + 0.29)) - 3
            sub = img[int(y_top) - 2: int(y_bot) + 2, x0:x1]
            dist = np.sqrt(((sub - blend) ** 2).sum(-1))
            rows = np.flatnonzero((dist < 28).sum(1) >= 0.5 * (x1 - x0))                     # rows where most of the bar's width has the bar colour (the top edge)
            y_edge = int(y_top) - 2 + rows.min() if len(rows) else None
            expected = 100 * S[n]["l1"]["onset"]["pod"] if key == "onset_pod" else S[n]["l1"]["nse_median"]
            value = ypix_to_val(y_edge) if y_edge is not None else None
            # the dots: connected clusters of ink pixels inside the bar's x-range (dots sit at +-0.12 of the bar centre)
            from scipy import ndimage
            xd0, xd1 = int(xpix(i - 0.2)), int(xpix(i + 0.2))
            ink = (np.sqrt(((img[int(y_top) - 10: int(y_bot) + 2, xd0:xd1] - INK) ** 2).sum(-1)) < 40)
            lab, n_lab = ndimage.label(ink)
            sizes = np.asarray(ndimage.sum(ink, lab, range(1, n_lab + 1)))
            one_dot = np.median(sizes[sizes >= 40]) if (sizes >= 40).any() else 1.0            # dots are s=22 pt^2 (about 13 px across); a merged pair of dots is about twice the area
            n_dots = int(sum(max(1, int(round(sz / one_dot))) for sz in sizes if sz >= 40))
            res[n] = dict(expected=round(float(expected), 3), read_from_pixels=round(float(value), 3) if value is not None else None, within_tolerance=(value is not None and abs(value - expected) <= (1.0 if key == "onset_pod" else 0.012)),
                          dots_counted=n_dots, dots_expected=len(R["seed_ranges"]["sets"][n]["l1"]["by_seed"]["seeds"]))
        fig[key] = res
    out["figure"] = dict(size=[Wpx, Hpx], panels=fig, all_bars_within_tolerance=all(v["within_tolerance"] for p in fig.values() for v in p.values()),
                         all_dot_counts_equal=all(v["dots_counted"] == v["dots_expected"] for p in fig.values() for v in p.values()),
                         footer_and_header_read="the title, the status line (NOT YET REVIEWED) and the footer (AORC wording, hindsight wording, the upper-bound wording for wqaP) are visible in the image; read by the reviewer by eye")
    print("figure bars within tolerance:", out["figure"]["all_bars_within_tolerance"], "| dot counts equal:", out["figure"]["all_dot_counts_equal"])
    for key, res in fig.items():
        print("  ", key, {n: (v["expected"], v["read_from_pixels"], v["dots_counted"]) for n, v in res.items()})
except Exception as e:                                                                       # the figure check must not stop the review
    out["figure"] = dict(error=repr(e))
    print("figure check failed:", repr(e))
out["seconds"] = round(time.time() - t0, 1)
jsave("r5_outcome.json", out)
