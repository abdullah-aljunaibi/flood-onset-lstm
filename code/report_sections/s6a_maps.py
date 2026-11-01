"""Section 6.1 - the basins from above: the elevation map with the three landscapes, the two rasters and the method, our GIS
statistics per basin and their check against CAMELS.

Reads results/basin_gis.json (written by 07_basin_maps.py); only if it exists, results/lstm_all_scores.csv (the per-basin score
table of 04_lstm_all.py) for the paragraph that sets river behaviour and forecast accuracy against the landscape; and, only if it
exists, results/engine_review.json for the sentence on who wrote and who checked the script (the state of the team members' own checks is
read through code/team.py from team.json, which the team fills in).
Nothing is computed here beyond counting, ranges, medians of four basins, differences and formatting.

Layout: both map figures of 07_basin_maps.py are drawn for a width of 170 mm (smallest lettering 7.5 pt), so a map is shown at about that
width or left out, never shrunk. The elevation map (figures/basin_maps.png, the output the plan names, Figure 6.1) is placed at
MAP_SHARE % of the text width as a float (see LAYOUT in s0_summary.py); the three landscapes are described in one paragraph, in the
order of its rows, and Table 6.1 is sized to fit under the map on the same page. The land-cover map is off by default (SHOW_COVER):
its numbers (forest, farmland and wetland share of every basin) are columns of Table 6.1.
"""
import re

import team
from flood import LAND_SHORT, STATIC
from report_common import INPUT_NAME, LEAD_WORD, cite, count, fig, n2, pct, signed, table
from report_sections.s0_summary import LAYOUT

MAP_SHARE = 80                      # % of the text width (184 mm) of a map figure; drawn for 170 mm (92%). Up to 81 the table still fits under the map
FIG_MAP, FIG_COVER, TAB_GIS = "Figure 6.1", "Figure 6.2", "Table 6.1"       # section 6.2 numbers its figures after these (see s6b_satellite.py)
SHOW_COVER = False                  # True: also show the land-cover map (figures/basin_landcover.png) at MAP_SHARE, about 0.7 page
SHOW_SKILL = True                   # False: leave out the paragraph that sets the scores of lstm_all_scores.csv against the landscape (about 0.09 page)
SHOW_TABLE = True                   # False: leave out the per-basin table; the text on elevation and slope then runs over the full width (about 0.17 page)
ELEV_GAP_M = 25.0                   # a mean elevation further than this from CAMELS is named in the text (the limit used in the notes of 07_basin_maps.py)
CLOSER_POINTS = 2.0                 # a tested explanation counts as "closer to CAMELS" when it cuts the mean gap by more than this (as in 07_basin_maps.py)
CHECKED = {"log_area": "area", "elev_mean": "mean elevation", "slope_mean": "mean slope", "frac_forest": "forest share"}     # model inputs we can recompute
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
TIGHT = "<style>table.gis{font-size:8.3pt} table.gis th,table.gis td{padding:1pt 3pt;line-height:1.26} table.gis th{white-space:nowrap}</style>"     # closer rows, one-line header: the table must fit under the map


def _word(n):
    return NUMBER.get(n, str(n))


def _and(parts):
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _rng(vals, fmt=count):
    vals = list(vals)
    return f"{fmt(min(vals))} to {fmt(max(vals))}"


def _slope(v):
    return n2(v, 0 if v >= 10 else 1)


def _name(r):
    return r["short"].split(",")[0]                              # 'South Toe, NC' -> 'South Toe'


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def _nb(s):
    return s.replace(" ", "&nbsp;")


def _skill(R, T, basins, alike=()):
    """How slowly the rivers change and how well the LSTM forecasts, by landscape - only when the per-basin score table of the
    all-basin LSTM (results/lstm_all_scores.csv, from a full run) exists and is complete; otherwise an empty string.
    alike: gauge ids of the basins outside the southern group whose terrain is like that of the southern basins.
    One paragraph: the per-basin scores themselves are in the per-basin table of section 4.2."""
    S = T.get("lstm_all_scores")
    if not SHOW_SKILL or S is None or not {"gauge_id", "model", "lead", "nse"} <= set(S.columns) or R.get("lstm_all", {}).get("quick_mode"):
        return ""
    lead = min(LEAD_WORD)
    MTN, SOUTH = list(LAND_SHORT)[0], list(LAND_SHORT)[-1]
    short = lambda k: LAND_SHORT[k].lower()
    cols = []                                                    # (name in the text, {landscape: NSE of its basins}, {landscape: their median})
    pers = None                                                  # NSE of 'tomorrow = today' by gauge id
    for model, title in (("base_persistence", QUOTED), ("lstm_w", f"with {INPUT_NAME['w']}"), ("lstm_wq", f"with {INPUT_NAME['wq']}")):
        s = S[(S.model == model) & (S.lead == lead)].set_index("gauge_id").nse
        pers = s if pers is None else pers
        if set(s.index) != set(basins) or s.isna().any():
            return ""
        by_land = {k: [float(s[g]) for g, r in basins.items() if r["landscape"] == k] for k in LAND_SHORT}
        cols.append((title, by_land, {k: _median(v) for k, v in by_land.items()}))
    tops = [max(med, key=med.get) for _, _, med in cols]
    apart = [min(by_land[t]) > max(v for k in LAND_SHORT if k != t for v in by_land[k]) for (_, by_land, _), t in zip(cols, tops)]
    pm = cols[0][2]                                              # medians of 'tomorrow = today'
    p_top, p_low = tops[0], min(pm, key=pm.get)
    per_land = sorted({len(v) for v in cols[0][1].values()})
    each = f"{_word(per_land[0])} basins" if len(per_land) == 1 else "a few basins"
    rest = _and(f"{n2(pm[k])} in the {short(k)}" for k in LAND_SHORT if k != p_top)
    also = " and ".join(t for t, same in zip(("with past weather", "with the river gauge as well"), (t == p_top for t in tops[1:])) if same)
    also = f"the LSTM of section 4 also scores highest there{'' if all(t == p_top for t in tops[1:]) else f' {also}'}" if also else "the LSTM of section 4 does not"
    if p_top == SOUTH:
        tops_own = [g for g in alike if float(pers[g]) >= max(cols[0][1][basins[g]["landscape"]])]       # scores highest in its own group
        twin = (f", and {_and(n2(float(pers[g])) for g in tops_own)} in {_and(_name(basins[g]) for g in tops_own)}, the flat, wet northern basin{'s' if len(tops_own) > 1 else ''}") if tops_own and len(tops_own) == len(alike) else ""
        found = (f"The slowly changing rivers tend to be the flat, wet ones: {QUOTED}, which needs no model, scores a median NSE of {n2(pm[p_top])} {LEAD_WORD[lead]} in the test "
                 f"years in the {short(p_top)}, against {rest}{twin}. That is what one expects of flat ground with wetlands, which holds water back; {also}.")
    else:
        found = (f"{QUOTED_CAP}, which needs no model and scores high where a river changes slowly, has its highest median NSE {LEAD_WORD[lead]} in the test years in "
                 f"the {short(p_top)} ({n2(pm[p_top])}, against {rest}); {also}.")
    doubts = ([f"with {QUOTED} the {short(p_low)}, not the mountains, score lowest"] if p_top == SOUTH and p_low != MTN else []) + \
             ([f"single basins of different landscapes overlap{'' if not any(apart) else f' in {len(apart) - sum(apart)} of the {_word(len(apart))} comparisons'} "
               f"(Table 4.2)"] if not all(apart) else [])
    answer = "Only as a tendency." if not all(apart) else "Yes, in these basins." if len(set(tops)) == 1 else "Not in one direction."
    return f"""
<p><b>Does the landscape show in the rivers and the forecasts?</b> {answer} {found} {f'But {", ".join(doubts)}, and' if doubts else 'But'} {each} per landscape,
which also differ in rainfall and size, cannot show the cause.</p>"""


def html(R, T):
    G = R["basin_gis"]
    B, C, SRC, M = G["basins"], G["comparison_with_camels"], G["sources"], G["methods"]
    n = len(B)
    order = sorted(B, key=lambda g: (list(LAND_SHORT).index(B[g]["landscape"]), B[g]["camels"]["area_gages2_km2"]))      # as in the maps
    land = {k: [B[g] for g in order if B[g]["landscape"] == k] for k in LAND_SHORT}
    MTN, NORTH, SOUTH = list(LAND_SHORT)
    grids = list(next(iter(B.values()))["slope_mean_by_grid_m_per_km"])                     # '30 m', '90 m', '310 m', '920 m'
    fine, coarse = grids[0], grids[-1]
    year = SRC["land_cover"]["year"]
    share = lambda r, k: r["land_cover_share"][k]
    forest = lambda r: share(r, "forest")
    farm = lambda r: share(r, "cropland and pasture")
    wet = lambda r: share(r, "wetland")
    relief = lambda r: r["relief_m"]
    slope = lambda r: r["slope_mean_m_per_km"]
    d_elev = lambda r: r["difference"]["elev_mean_m"]                                      # our mean elevation minus the CAMELS value, m
    lower1 = lambda s: s[0].lower() + s[1:]
    basin_of = lambda c: c["largest_difference_basin"].split(",")[0]

    # ---- who did it: stated from results/engine_review.json (who wrote the code, who reviewed it, whether the team's own check is done) ----
    ER = R.get("engine_review", {})
    who = str(ER.get("who", ""))
    by_agent = "Claude Code" in who and "the team lead" in who
    reviewed = "AI reviewer" in who and any("script" in x and "reviewer" in x for x in ER.get("notes", []))
    pending = str(ER.get("human_review", "")).lower().startswith("not yet")
    checks = team.checks_state()                                                           # the members' own checks, as entered by the team in team.json
    if checks["n_done"]:                                                                   # a member has entered a finished check: the words every document uses
        own = f"; {team.checks_sentence()}{'' if 'section 9' in team.checks_sentence() else ' (section 9)'}."
    else:
        own = "; the team has not yet." if pending and reviewed else "."
    made = ("In the plan this part belongs to the remote-sensing and GIS pair"
            + (("; so far Claude Code, the team&rsquo;s coding assistant, wrote and ran its script under the team lead&rsquo;s direction"
                + (", and an independent AI reviewer agent checked it" if reviewed else "") + own) if by_agent else "."))

    # ---- the question, and (placed after the map) the rasters and the method -----------------------------------------
    ask = f"""
<p>What do the {n} basins look like from above, and can the basin properties read by the LSTM of section 4 be trusted? {made}</p>"""
    how = f"""
<p><b>The two rasters</b> (maps stored as a grid of cells). <i>Elevation:</i> {SRC["elevation"]["product"]}; {SRC["elevation"]["resolution"].split(", ", 1)[-1].split(";")[0]}
cells. The provider describes a composite of SRTM, the Shuttle Radar Topography Mission {cite("farr2007")}, and US Geological Survey elevation; the tiles do
not name the source of a cell, so it is not called pure SRTM here. <i>Land cover:</i> the {year} layer of the Annual NLCD (National Land Cover Database) series,
{lower1(SRC["land_cover"]["resolution"]).split(",")[0]}; the same classes as the single-year NLCD {year} release {cite("homer2015")}, but not that release.</p>
<p><b>The method.</b> The script lays each basin outline, which comes with CAMELS-US, over both rasters; a cell belongs to a basin when its centre lies inside. Elevation
cells are weighted by their true area; land-cover shares are counts of cells in {_word(len(SRC["land_cover"]["groups"]))} groups merged from the map&rsquo;s
{len(SRC["land_cover"]["classes"])} classes. Nothing is fitted or chosen, so the test years are not touched.</p>"""

    # ---- three landscapes: one short description per row of the elevation map ---------------------------------------
    odd_n = [r for r in land[NORTH] if r["dominant_land_cover"] != "forest"]                 # a northern basin that looks like the southern ones
    odd_s = [r for r in land[SOUTH] if r["dominant_land_cover"] == "forest"]                 # a southern basin that looks like the northern ones
    alike, odd_n_text, odd_s_text = [], "", ""
    if odd_n:
        like_south = all(wet(r) >= min(map(wet, land[SOUTH])) and relief(r) <= max(map(relief, land[SOUTH])) for r in odd_n)
        alike = [g for g in order if B[g] in odd_n] if like_south else []
        what = "as flat and as wet as the southern basins" if like_south else "not mostly forest"
        odd_n_text = (f" {'Exception' if len(odd_n) == 1 else 'Exceptions'}: {_and(_name(r) for r in odd_n)}, {what} "
                      f"({_and(f'{count(relief(r))} m of relief, {pct(wet(r))} wetland' for r in odd_n)}).")
    if odd_s:
        odd_s_text = f" {'Exception' if len(odd_s) == 1 else 'Exceptions'}: {_and(_name(r) for r in odd_s)}, mostly forest ({_and(pct(forest(r)) for r in odd_s)})."
    skill = _skill(R, T, B, alike)
    by_slope = sorted(LAND_SHORT, key=lambda k: _median(map(slope, land[k])))                # landscapes from the flattest to the steepest
    by_forest = max(LAND_SHORT, key=lambda k: _median(map(forest, land[k])))
    mtn_word = _and((["steep"] if by_slope[-1] == MTN else []) + (["wooded"] if by_forest == MTN else [])) or "high"
    south_word = "the flattest" if by_slope[0] == SOUTH else "flat as well"
    notes = {
        MTN: (f"<b>{LAND_SHORT[MTN]}:</b> southern Appalachians, {mtn_word}. Mean slope {_rng(map(slope, land[MTN]), _slope)} m per km on the {fine} grid, "
              f"{_rng(map(forest, land[MTN]), pct)} forest, {_rng(map(relief, land[MTN]))} m between the lowest and the highest point."),
        NORTH: (f"<b>{LAND_SHORT[NORTH]}:</b> rolling Piedmont hills and coastal plain. {_rng(map(slope, land[NORTH]), _slope)} m per km, "
                f"{_rng(map(forest, land[NORTH]), pct)} forest, {_rng(map(farm, land[NORTH]), pct)} farmland.{odd_n_text}"),
        SOUTH: (f"<b>{LAND_SHORT[SOUTH]}:</b> {south_word}. {_rng(map(slope, land[SOUTH]), _slope)} m per km, {_rng(map(relief, land[SOUTH]))} m of relief; "
                f"{_rng(map(farm, land[SOUTH]), pct)} farmland and {_rng(map(wet, land[SOUTH]), pct)} wetland (mountain basins: at most "
                f"{pct(max(map(wet, land[MTN])), 1)}).{odd_s_text}"),
    }
    # the three landscapes as running text (it can fill the foot of a page; a block under the map could not)
    landscapes = f"<p><b>The three landscapes</b> (the rows of {FIG_MAP}, from the top). " + " ".join(notes[k] for k in LAND_SHORT) + "</p>"
    map_caption = (f"<b>{FIG_MAP}. Elevation inside each basin outline.</b> Each basin is coloured from its <i>own</i> lowest point (light) to its own highest "
                   f"(dark) and the relief shading is strengthened in the flat basins: neither can be compared between two maps.")
    maps = '<div class="ff">' + fig("basin_maps.png", map_caption, MAP_SHARE) + "</div>" + landscapes       # a full-width float (see LAYOUT)
    if SHOW_COVER:
        maps += fig("basin_landcover.png", f"<b>{FIG_COVER}. Land cover in {year}.</b> The line above each map names the two largest groups and their share of the basin. One row per "
                                           f"landscape, basins in order of area; the white dot is the river gauge.", MAP_SHARE)
    expect = "" if skill else ("<p>Why it matters here: steep ground is expected to pass rain to the river faster than flat ground with wetlands, which holds water back. "
                               "That is the usual expectation in hydrology; the maps alone cannot test it.</p>")

    # ---- our statistics against CAMELS: table on the left, the two findings it shows on the right ------------------
    sg = C["slope_mean_by_grid"]
    rows = []
    for r in (B[g] for g in order):
        rows.append([_nb(r["short"]), _nb(f"{count(r['elev_mean_m'])} ({signed(d_elev(r), 0)})"), _slope(r["slope_mean_by_grid_m_per_km"][fine]),
                     _slope(r["camels"]["slope_mean_m_per_km"]), _nb(f"{n2(100 * forest(r), 0)} ({signed(r['difference']['forest_share_points'], 0)})"),
                     n2(100 * farm(r), 0), n2(100 * wet(r), 0)])
    rows.append(["<b>Median&nbsp;diff.</b>", f"<b>{signed(C['elev_mean']['median_difference'], 1)}&nbsp;m</b>", f"<b>{signed(sg[fine]['median_difference'], 0)}%</b>",
                 "", f"<b>{signed(C['forest_share']['median_difference'], 0)}</b>", "", ""])
    gis_table = table(["Basin", "Elevation, m (diff.)", f"Slope, {_nb(fine)}", "Slope, CAMELS", "Forest, % (diff.)", "Farmland, %", "Wetland, %"], rows,
                      cls="num gis", widths=["20%", "18%", "12%", "14%", "16%", "10%", "10%"])
    gis_caption = (f'<p class="cap"><b>{TAB_GIS}. The recomputed statistics against CAMELS,</b> basins in the order of the maps. Slope in m per km; land cover in '
                   f"{year}. In brackets and in the last row: recomputed minus CAMELS (forest in percentage points).</p>")

    checked = [w for k, w in CHECKED.items() if k in STATIC]
    intro = f"""<p><b>Do the recomputed numbers agree with CAMELS?</b> The LSTM of section 4 reads {len(STATIC)} fixed properties of each basin from the CAMELS attribute table
{cite("addor2017")}, which states neither the grid nor the land-cover year behind them. {_word(len(checked)).capitalize()} of them can be recomputed from the two rasters:
{_and(checked)}.</p>"""

    # elevation
    e, eo = C["elev_mean"], C["elev_mean_vs_outline_file"]
    off = [B[g] for g in order if abs(d_elev(B[g])) > ELEV_GAP_M]
    shift = max(abs(v - r["elev_mean_m"]) for r in B.values() for v in r["elev_mean_by_grid_m"].values())
    if off:
        direction = "higher than" if all(d_elev(r) > 0 for r in off) else "lower than" if all(d_elev(r) < 0 for r in off) else "away from"
        where = (f"{_word(len(off))} of the {_word(len(land[MTN]))} mountain basins" if all(r["landscape"] == MTN for r in off) else f"{len(off)} of {n} basins")
        elev = (f"<p><b>Mean elevation: the recomputed value is {direction} CAMELS in {where}, and the cause was not found.</b> The difference is "
                f"{_and(f'{signed(d_elev(r), 0)} m in {_name(r)}' for r in sorted(off, key=d_elev))}; the median over all {n} basins is only {signed(e['median_difference'], 1)} m. "
                f"Grid size does not explain it: averaged to cells of up to {coarse}, the mean moves by at most {n2(shift, 1)} m. A second mean elevation, stored in the outline "
                f"file, is within {n2(abs(eo['largest_difference']), 1)} m of the recomputed one in every basin.</p>")
    else:
        elev = (f"<p><b>Mean elevation agrees.</b> The median difference is {signed(e['median_difference'], 1)} m and no basin is more than {n2(ELEV_GAP_M, 0)} m away "
                f"(largest {signed(e['largest_difference'], 0)} m, {basin_of(e)}).</p>")

    # slope
    falls = all(sg[x]["median_difference"] > sg[y]["median_difference"] for x, y in zip(grids, grids[1:]))
    best = min(grids, key=lambda g: sg[g]["mean_absolute_difference"])
    steps = _and(f"{signed(sg[g]['median_difference'], 0)}% ({g})" for g in grids[1:])
    slope_txt = (f"<p><b>Slope means little without its grid size{': a classic scale effect in GIS' if falls else ''}.</b> On the {fine} grid the recomputed mean slope is "
                 f"{n2(1 + sg[fine]['median_difference'] / 100, 1)} times the CAMELS value in the median ({signed(sg[fine]['median_difference'], 0)}%). On coarser cells the "
                 f"difference {'shrinks' if falls else 'changes'} to {steps}" + (": a coarse cell averages ridges and valleys away. " if falls else ". ")
                 + f"That the {best} grid comes closest suggests, but does not prove, that CAMELS used a grid about that coarse.</p>")

    # area and forest
    a = C["area_vs_gages2"]
    f, fw, fk = C["forest_share"], C["forest_plus_woody_wetland_share"], C["forest_share_1km_majority"]
    lower = f["mean_difference"] < 0
    n_side = sum((r["difference"]["forest_share_points"] < 0) == lower for r in B.values())
    n_closer = sum(abs(r["forest_share_1km_majority"] - r["camels"]["frac_forest"]) < abs(forest(r) - r["camels"]["frac_forest"]) for r in B.values())
    gain = lambda c: f["mean_absolute_difference"] - c["mean_absolute_difference"]                                    # points by which a variant is nearer to CAMELS
    block = re.search(r"\((\d+) m\)", M["land_cover"])
    block = f"{block.group(1)} m" if block else "about 1 km"
    swamp = f"counting forested swamps as forest {'leaves' if abs(gain(fw)) <= CLOSER_POINTS else 'gives'} {n2(fw['mean_absolute_difference'], 1)} points"
    definition = f"Definition is {'part of the reason' if gain(fw) > CLOSER_POINTS else 'not the reason'}: {swamp}."
    coarse_map = f"on a {block} grid where each block takes its most common group the gap {'falls to' if gain(fk) > CLOSER_POINTS else 'is'} {n2(fk['mean_absolute_difference'], 1)} points"
    if gain(fk) > CLOSER_POINTS:
        resolution = f"Resolution is part of it: {coarse_map}, {'and it is closer in all' if n_closer == n else 'but it is closer in only'} {n_closer} of {n} basins."
    else:
        resolution = f"Resolution is not the reason either: {coarse_map}."
    rest = (f"<p><b>Area {'agrees' if abs(a['largest_difference']) < 5 else 'differs in places'}:</b> the largest difference is {signed(a['largest_difference'], 1)}% ({basin_of(a)}); "
            f"CAMELS uses another outline. <b>Forest share: the recomputed share is {'lower' if lower else 'higher'} in {n_side} of {n} basins,</b> on average "
            f"{n2(f['mean_absolute_difference'], 1)} percentage points from CAMELS (largest {signed(f['largest_difference'], 1)}, {basin_of(f)}). {definition} {resolution}</p>")

    # the findings as running text, then the table: text can fill the page beside which the floated map could not be placed
    check = f"{elev}{slope_txt}"
    gis_block = f'<div class="blk">{gis_table}{gis_caption}</div>' if SHOW_TABLE else ""
    ranks = [C[k]["rank_correlation"] for k in ("area_vs_gages2", "elev_mean", "slope_mean", "forest_share")]
    same_class = "do not change" if G["landscape_classes_unchanged_with_our_elevation"] else "would change"
    holds = "That order is the same" if min(ranks) > 0.9995 else "That order is nearly the same" if min(ranks) >= 0.9 else "That order holds only in part"
    means = f"""
<div class="note blk"><b>What this means for the model.</b> It reads the CAMELS values, each put on one scale over the {n} basins, so it mainly sees how the basins compare with one
another. {holds}: the rank correlation between the recomputed values and CAMELS is {_rng(ranks, n2)} for the {_word(len(ranks))} properties (1 means the same
order), and the {_word(len(LAND_SHORT))} landscape groups {same_class} with the recomputed elevations. The model was not retrained with these values, so the results do
not say whether they would change a forecast.</div>"""

    return f"""{LAYOUT}{TIGHT}<h2>6 Remote sensing and GIS</h2>
<h3>6.1 The basins from above</h3>
{ask}{maps}{expect}{skill}{how}{intro}
{check}
{rest}
{gis_block}{means}"""
