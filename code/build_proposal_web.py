"""The PROPOSAL talk as ONE self-contained HTML file -> proposal/Flood_Proposal_Presentation.html (also copied to
proposal/web/index.html with nothing else beside it, for the upload to a website).

This is the presentation shown to the class and the professor on 2026-10-15 (the user's words: "this html will be the presentation
that we will show the class and the professor"), and later put online. It keeps the six PARTS and the timing of the
six-slide deck (code/build_proposal_slides.py: 600 s, 150 s per member) but spreads them over more screens, each with one
visual idea, and adds one interactive screen: a day-by-day REPLAY of the stored forecasts of the preliminary runs, as the model
would run in operation (issue a forecast every evening, warn when it crosses the flood level, then see what the river did).

Nothing is computed here that a results file does not hold, except the replay, which only re-reads the stored forecasts
(results/preds/*.npz, written by the experiment scripts) and the rain columns of the caches, and re-applies the engine's own
warning rule in the browser (code/flood.py: a day is high when obs > thr; a warning when the forecast > thr; an ONSET is a high
day forecast while the river was still at or below thr lead days before). The pooled counts the page shows for the whole test
period are read from results/aorc_rain_to_midnight.json, results/forecast_input.json and results/stage4_realtime.json and
asserted against the headline numbers of the proposal, so a changed results file fails the build instead of printing a stale number.

Everything is inline (no CDN, no web font: the page must work offline in the classroom and in China): CSS, JS, the data of
the replay as base64 Uint16 arrays (flow and rain in hundredths of a mm per day). Screen 3 draws its own map as an inline SVG from
proposal_web/assets/map/basin_map.json (EPSG:5070 outlines, land, gauges; code/proposal_web/make_map_asset.py writes it) and shows the
elevation panel of a clicked basin, cropped here from figures/basin_maps.png into 12 small palette PNGs; the cover carries a USGS
photograph (proposal_web/assets/cover_photo.jpg, credit read from cover_photo.json, printed on screen 5). The long deck keeps the
basin map and the data-flow figure as base64 PNG. Text is copied from proposal/AI_and_ALLMs_Proposal_Flood.md and the deck (build_proposal_slides.py: TITLE, SUB,
AIM, GAP, RQ, DISCLOSURE, NOTES), so the talk, the deck and the proposal say the same thing.

The look lives in the template's CSS tokens (light values in :root, dark values in :root[data-theme="dark"]; key D or the button in the bar
toggles them): blue = rain, navy = the river, orange = a warning; tonal panels, one shadowed hero per screen. Motion (falling rain, branches that draw in, count-ups, the warning pulse) is
switched off by prefers-reduced-motion. In full screen the bar at the foot hides after 3 s without mouse movement.

Run from the project root:  ~/AI_Class/.venv/bin/python code/build_proposal_web.py
"""
import base64
import html
import io
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_proposal_slides as DK                      # the deck: text constants, NOTES, the speakers and the seconds
import schedule
import team
from common import FIG, RES, ROOT

OUT = ROOT / "proposal"
WEB = OUT / "web"
NAME = "Flood_Proposal_Presentation"
TEMPLATE_LONG = Path(__file__).resolve().parent / "proposal_web" / "deck.html"       # the 12-screen version (python code/build_proposal_web.py --long)
TEMPLATE = Path(__file__).resolve().parent / "proposal_web" / "deck5.html"           # the five-screen version shown in class (default)
PRED = RES / "preds"
TEST = ("2005-10-01", "2014-09-30")
FROM = "2005-09-28"                      # three days before the test years: the issue-day flow of the first targets (the engine reads obs.shift(lead))
SETS = {"wq": "lstm_wq", "wqa": "lstm_wqa", "wqaF": "lstm_wqaF_sub", "wqf": "lstm_wqf"}       # the ladder of information shown in the replay
HEADLINE = dict(nse_lstm=0.59, nse_simple=0.56, warned_wq=19, warned_wqn=30, warned_wqa=51, warned_s4=48, warned_wqf=64,
                warned_F=(45, 39, 37), far_wq=59, far_wqa=50, csi_wqa=0.34, onsets=611)


# The five screens (600 s; 150 s per member; the team lead screens 1 and 2 = 60 + 90 s) and their notes, at most 2.3 words per second.
PARTS5 = [("A", 60), ("A", 90), ("A", 30), ("R2", 150), ("G1", 150), ("G2", 150)]   # 630 s: the team lead 180 (the six-step screen added 2026-10-15 at the user's request)
NOTES5 = [
    ("[F for full screen] Good morning. Let me start with one night in November 2006. [point to the chart] Little River, North Carolina. That evening a "
     "forecast model with the usual rain table looked at the river and said: nothing coming. The same model, shown the rain that had already fallen that day, said: "
     "warning. By morning the river had gone from almost dry to forty-two millimetres of runoff. Same model, same river; the difference was what it could see. "
     "Most flood studies ask which model is best. We are asking something nobody has measured on these rivers: what does the clock of the data cost a flood warning? "
     "[point to the key] Follow three colours through the talk: blue is rain, navy is the river, orange is a warning.",
     "Here is why that night was no accident."),
    ("[press 1] This is a day in the life of a forecast. It is issued at midnight. But the rain table that everyone trains on closes its day at nine in the morning, so "
     "the orange hours, rain already on the ground, are invisible to the model: a third of the rain around a flood onset. Trained that way, the model warns nineteen "
     "percent of onsets a day ahead, and almost none at two or three days. Not a model problem: a clock problem, found by measuring every product's clock against hourly "
     "rain. [press 2] Give the model rain to the evening: thirty. [press 3] To midnight: fifty-one, forty-eight with the real-time radar product a service can read. "
     "[press 4] Add a weather forecast: two- and three-day warnings appear for the first time, thirty-nine and thirty-seven. One model, four levels of information, one "
     "answer: the information sets the limit. [click RQ1] Our three questions: does a deep network beat simple models; [click RQ2] which information closes the gap; "
     "[click RQ3] and can the warnings be trusted, judged by a rule we fixed before the experiments: seventy percent warned, at most thirty percent false.",
     "Member 2 will show you where this happens, and with what data."),
    ("Six steps. Public data, eleven free products. One table per basin, twelve basins over thirty-four years. A fair split: train, tune, test once. "
     "One deep network against three simple models, each fed extra information in turn. A forecast every evening for one to three days. A score by rules fixed in advance.",
     "Member 2 will show where this happens, and with what data."),
    ("[point to the funnel] Twelve basins chosen by rule from the 671 of CAMELS-US: 78 candidates in the south-east, 63 with complete records, four per landscape, "
     "and no model score used. [hover Leaf River on the map] Each basin on the map gives its area and elevation range; they span 113 to 1,928 square kilometres, "
     "[click South Toe] from the mountains, here South Toe with its relief, [click Myakka] to the flat Florida lowlands. The map is drawn in the Albers projection with "
     "a scale bar and grid north. [hover CAMELS-US on the mind map] Every input is a public product: CAMELS gives the daily flow and Daymet weather, obtained from NCAR. "
     "[hover AORC] AORC gives hourly rain on a one-kilometre grid, an analysis made after the fact, read from Amazon open data and averaged inside each outline. "
     "[hover GEFSv12] The GEFS reforecast is an archived weather forecast, so a forecast can be tested on past years without hindsight. [hover Stage IV] Stage IV is "
     "real-time radar rain. [hover IMERG] IMERG and MERRA-2 give satellite rain and soil wetness. The grey ones, elevation tiles, land cover and the storm tracks, serve "
     "the maps and the checks. Each product is summed over the local day into one basin-by-day table. One point for this course: no optical or SAR scene is processed, "
     "so capture date and sun angle do not apply.",
     "Member 3 will now show how the forecast runs and how it is judged."),
    ("[hover the five method tiles] One pipeline: basin means, features, one LSTM for all twelve basins against three simple baselines, a split by target day with "
     "training to 2000, validation to 2005 and test from 2005 to 2014, and scoring with criteria fixed in advance. This is how it would run, with the rain table and the "
     "gauge first. [press O] Every evening the model reads the rain of the day and the river and issues a forecast for the next three days, the orange dots. Here the "
     "forecast stays under the flood level: no warning. [press S] The next day tells: a red cross when the river crossed the level and we had no warning, a green tick "
     "when we had. [press O, then S] Again. [click '+ rain to midnight', then O] Now the same river with the rain of the issue day to midnight: the forecast rises and a "
     "warning goes out, [press S] and the next day tells again. [point to the tally] The tally gives the verdict by the criteria. [point to the whole-period line] Over "
     "nine years and twelve basins: nineteen percent of onsets warned with the rain table, fifty-one with rain to midnight, half of the warnings false, verdict not yet. "
     "The rain to midnight is an analysis made after the fact; with the real-time radar product in its place, forty-eight. A replay, not a service.",
     "Member 4 will close with the targets, the plan and our questions."),
    ("[point to the tiles] Our baseline is the best simple model with the same inputs. First results: forty-eight percent of onsets warned with real-time inputs, "
     "and thirty-five percent false warnings with the forecast beside the rain. The targets are the GO level, not yet reached: seventy percent warned, thirty percent "
     "false, and five trainings per model; a network trained on the radar product itself is a stretch goal. [hover the 19% bar, then 51%, then 64%] The ladder: "
     "nineteen with the rain table, fifty-one with rain to midnight, sixty-four with a perfect forecast that cannot be had; every bar carries its caveat. "
     "[click Week 1, 2, 3] Three weeks to the fifth of November: this week your feedback and our own checks; next week the remaining trainings and the GIS layers; the "
     "last week the report and the hand-in. [point to the disclosure] AI assistance is disclosed: Claude Code wrote and ran the code under the team lead's direction, "
     "independent AI reviewers checked it, and our own checks are open. [point to the questions] Three questions for you: is a US public dataset acceptable; is a "
     "rigorous negative result acceptable; and how should the use of AI tools be stated? Thank you.",
     ""),
]
WPS = 2.3


def datasets():
    """The eleven products for the data mind map, from results/data_flow.json: name, role, what, how obtained, how computed, grid, step, period (strings cut short)."""
    def cut(t, n=190):
        t = re.sub(r"\s+", " ", str(t)).strip()
        return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"
    def url(u):
        u = re.sub(r"\?.*$", "", str(u)).strip()
        return cut(u, 90)
    out = []
    for p in json.loads((RES / "data_flow.json").read_text(encoding="utf-8"))["datasets"]:
        role = p["role"]
        rc = "every" if "every model" in role else "extra" if "extra input" in role or "alternative" in role else "check"
        ob = p.get("obtained") or {}
        obtained = ob.get("url_pattern", "") if isinstance(ob, dict) else str(ob)
        how = (ob.get("how") or ob.get("method") or "") if isinstance(ob, dict) else ""
        out.append(dict(short=p["short"], role=cut(role, 110), role_class=rc, one_line=cut(p.get("one_line", "")), obtained=cut((url(obtained) + " " + how).strip(), 150),
                        turned=cut(p.get("turned_into_input", "")), grid=cut(p.get("spatial_grid", ""), 70), step=cut(p.get("time_step", ""), 50), period=cut(p.get("period_used", ""), 60)))
    assert len(out) == 11, len(out)
    return out


def b64_u16(a, scale=100.0):
    """A float array (mm/day) -> base64 of little-endian uint16 in hundredths, NaN -> 65535."""
    x = np.asarray(a, float)
    v = np.where(np.isnan(x), 65535, np.clip(np.round(x * scale), 0, 65534)).astype("<u2")
    return base64.b64encode(v.tobytes()).decode("ascii")


def replay_data():
    """The data of the replay: per basin, the test years day by day: obs, rain table (Daymet), rain to midnight (AORC), the GEFS
    forecast rain of the next three days, and the forecasts of the four sets at the three leads, by TARGET day as the engine stores them."""
    FA = json.loads((RES / "flood_analysis.json").read_text())
    thr = FA["thresholds_mm_day"]
    DA = json.loads((RES / "data_audit.json").read_text())
    info = {s["gauge_id"]: s for s in DA["selected"]}
    short = {g: FA["basins"][g]["short"] for g in thr}
    Z = {k: np.load(PRED / f"{v}.npz", allow_pickle=True) for k, v in SETS.items()}
    basins = list(Z["wq"]["basins"])
    dates = pd.DatetimeIndex(Z["wq"]["dates"])
    assert all(list(z["basins"]) == basins and (z["dates"] == Z["wq"]["dates"]).all() for z in Z.values()), "the prediction sets do not share one index"
    sel = (dates >= FROM) & (dates <= TEST[1])
    days = dates[sel]
    assert len(days) == 3290 and str(days[3].date()) == TEST[0], len(days)
    import pickle
    DM = pickle.load(open(ROOT / "data/derived/camels12_daymet.pkl", "rb"))
    out = dict(start=str(days[0].date()), test_start=3, n=len(days), basins=[], sets=list(SETS), set_names=SETS)
    for i, g in enumerate(basins):
        d = DM[g]
        assert d.index.equals(dates), g
        obs = Z["wq"]["obs"][i][sel]
        assert np.allclose(obs, d.q_mm.values[sel]), f"{g}: the stored obs differ from the daily table"
        gf = pd.read_csv(ROOT / f"data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
        # AORC rain of local day t = aorc_1 of the issue day t-1 (the table is by issue day: aorc_k = AORC rain of the local day t+k)
        aorc_day = gf["aorc_1"].shift(1, freq="D").reindex(days)
        fc = {k: gf[f"fc_bc_{k}"].reindex(days) for k in (1, 2, 3)}       # the forecast rain of day t+k, issued on day t (bias corrected, the input of E2a)
        b = dict(id=g, name=short[g], landscape=info[g]["landscape"], area=round(float(info[g]["area_gages2"])), thr=round(float(thr[g]), 3),
                 lat=info[g]["gauge_lat"], lon=info[g]["gauge_lon"],
                 obs=b64_u16(obs), rain=b64_u16(d.prcp.values[sel]), aorc=b64_u16(aorc_day.values),
                 fc=[b64_u16(fc[k].values) for k in (1, 2, 3)],
                 pred={k: [b64_u16(z["pred"][i][sel][:, l]) for l in range(3)] for k, z in Z.items()})
        out["basins"].append(b)
    return out


def pooled():
    """The pooled warning tables of the whole test period, read from the results files (the numbers of the proposal), 1 day ahead."""
    A = json.loads((RES / "aorc_rain_to_midnight.json").read_text())["summary_test"]["sets"]
    Fc = json.loads((RES / "forecast_input.json").read_text())["summary_test"]["sets"]
    S4 = json.loads((RES / "stage4_realtime.json").read_text())
    P = {}
    for k, src in (("wq", A["wq"]), ("wqn", A["wqn"]), ("wqa", A["wqa"]), ("wqf", A["wqf"]), ("wqaF", Fc["wqaF_sub"])):
        P[k] = {f"l{l}": dict(hits=src[f"l{l}"]["onset"]["hits"], misses=src[f"l{l}"]["onset"]["misses"], false=src[f"l{l}"]["onset"]["false_alarms"],
                              pod=src[f"l{l}"]["onset"]["pod"], far=src[f"l{l}"]["onset"]["far"], csi=src[f"l{l}"]["onset"]["csi"], nse=src[f"l{l}"]["nse_median"])
                for l in (1, 2, 3)}
    pc = lambda x: int(round(100 * x))
    assert pc(A["wqn"]["l1"]["onset"]["pod"]) == HEADLINE["warned_wqn"]
    assert pc(P["wq"]["l1"]["pod"]) == HEADLINE["warned_wq"] and pc(P["wqa"]["l1"]["pod"]) == HEADLINE["warned_wqa"] and pc(P["wqf"]["l1"]["pod"]) == HEADLINE["warned_wqf"]
    assert tuple(pc(P["wqaF"][f"l{l}"]["pod"]) for l in (1, 2, 3)) == HEADLINE["warned_F"], [P["wqaF"][f"l{l}"]["pod"] for l in (1, 2, 3)]
    assert pc(P["wq"]["l1"]["far"]) == HEADLINE["far_wq"] and pc(P["wqa"]["l1"]["far"]) == HEADLINE["far_wqa"] and round(P["wqa"]["l1"]["csi"], 2) == HEADLINE["csi_wqa"]
    assert P["wq"]["l1"]["hits"] + P["wq"]["l1"]["misses"] == HEADLINE["onsets"]
    assert round(P["wq"]["l1"]["nse"], 2) == HEADLINE["nse_lstm"]
    m = re.search(r"warn of (\d+) \((\d+)%\) against (\d+) with AORC", S4["outcome"])
    assert m and int(m.group(2)) == HEADLINE["warned_s4"] and int(m.group(3)) == P["wqa"]["l1"]["hits"], S4["outcome"][:200]
    P["stage4"] = dict(hits=int(m.group(1)), pod=int(m.group(1)) / HEADLINE["onsets"])
    return P


ASSETS = Path(__file__).resolve().parent / "proposal_web" / "assets"
ICONS = {"i-rain": "cloud-rain", "i-wave": "waves", "i-bell": "bell", "i-db": "database", "i-grid": "table-2", "i-brain": "brain", "i-split": "split", "i-shield": "shield-check"}


def font_b64():
    """Inter (variable, latin subset; SIL Open Font License 1.1, assets/LICENSE-Inter.txt) as base64 woff2, so the page looks the same on every computer."""
    f = ASSETS / "inter-latin-wght-normal.woff2"
    b = f.read_bytes()
    assert b[:4] == b"wOF2", f
    return base64.b64encode(b).decode("ascii")


def icon_symbols():
    """The Lucide icons (ISC licence, assets/LICENSE-Lucide.txt) as <symbol> elements with the ids the template uses."""
    out = ["<!-- Icons: Lucide (https://lucide.dev), ISC licence; see code/proposal_web/assets/LICENSE-Lucide.txt -->", '<svg style="display:none" aria-hidden="true">']
    for sid, name in ICONS.items():
        t = (ASSETS / "icons" / f"{name}.svg").read_text(encoding="utf-8")
        inner = re.search(r"<svg[^>]*>(.*)</svg>", t, re.S).group(1).strip()
        inner = re.sub(r"\s+", " ", inner)
        out.append(f'  <symbol id="{sid}" viewBox="0 0 24 24">{inner}</symbol>')
    out.append("</svg>")
    return "\n".join(out)


def png_b64(path, crop=None, max_w=None):
    im = Image.open(path)
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    if max_w and im.size[0] > max_w:
        im = im.convert("RGB").resize((max_w, int(im.size[1] * max_w / im.size[0])), Image.LANCZOS)
    b = io.BytesIO()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return base64.b64encode(b.getvalue()).decode("ascii"), im.size


MAPDIR = ASSETS / "map"
MAP_MARGIN = dict(west=40, east=60, north=14, south=40)       # px of the map (1 px = 1.5 km) beyond the outermost rings: 60 km west (Leaf River) and south (Myakka, Charlie Creek) as asked,
                                                              # 90 km east (room for the labels beside the Hyco Creek and NE Cape Fear rings), 21 km north


def map_asset():
    """Everything screen 3 needs from basin_map.json (the numbers are copied as they are; nothing is recomputed except the crop of the frame):
    the viewBox cropped to the 12 rings plus a margin (the frame of the asset holds far more sea than the deck wants), the land path, the
    12 basins (name, landscape, geodesic area, elevation range and mean of OUR GIS statistics, outline, gauge, ring, label hint), the
    scale bar, the grid convergence, and the 12 elevation panels cropped from figures/basin_maps.png as palette PNGs (base64)."""
    J = json.loads((MAPDIR / "basin_map.json").read_text(encoding="utf-8"))
    assert J["viewBox"] == [0, 0, 1000, 940] and len(J["basins"]) == 12
    G = json.loads((RES / "basin_gis.json").read_text(encoding="utf-8"))["basins"]        # the numbers on the map are our GIS statistics: a changed results file fails the build instead of printing stale numbers
    for b in J["basins"]:
        g = G[b["id"]]
        assert (b["name"] == g["short"] and abs(b["area_km2"] - g["area_km2"]) < .06 and b["elev_min"] == g["elev_min_m"] and b["elev_max"] == g["elev_max_m"]
                and abs(b["elev_mean_gis"] - g["elev_mean_m"]) < .06), f"basin_map.json is out of date for {b['id']} {b['name']}: rerun code/proposal_web/make_map_asset.py"
    rings = [b["ring"] for b in J["basins"]]
    m = MAP_MARGIN
    x0 = int(np.floor(min(cx - r for cx, cy, r in rings) - m["west"]))
    x1 = int(np.ceil(max(cx + r for cx, cy, r in rings) + m["east"]))
    y0 = int(np.floor(min(cy - r for cx, cy, r in rings) - m["north"]))
    y1 = int(np.ceil(max(cy + r for cx, cy, r in rings) + m["south"]))
    assert x0 >= 0 and y0 >= 0 and x1 <= 1000 and y1 <= 940, (x0, y0, x1, y1)
    short = {l["key"]: l for l in J["landscapes"]}
    fix = lambda t: t.replace("32.5N", "32.5\u00b0 N")
    basins = [dict(id=b["id"], name=b["name"], key=b["landscape_key"], landscape=fix(b["landscape"]), area=b["area_km2"], emin=b["elev_min"], emax=b["elev_max"],
                   emean=b["elev_mean_gis"], path=b["path"], gauge=b["gauge"], ring=b["ring"], label=b["label"]) for b in J["basins"]]
    # the panels: 330 x 330 px crops of the 4 x 3 grid of the figure (the box list comes with the asset), 256-colour PNG (28 kB instead of 62 kB each, same look)
    pg = J["panel_grid"]
    im = Image.open(FIG / "basin_maps.png").convert("RGB")
    assert list(im.size) == pg["size"], (im.size, pg["size"])
    panels = {}
    for gid, box in pg["boxes"].items():
        c = im.crop(tuple(box))
        assert c.size == (330, 330), (gid, c.size)
        b = io.BytesIO()
        c.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(b, "PNG", optimize=True)
        panels[gid] = base64.b64encode(b.getvalue()).decode("ascii")
    assert set(panels) == {b["id"] for b in basins}
    sb = J["scale_bar"]
    mj = dict(viewBox=[x0, y0, x1 - x0, y1 - y0], land=J["land"], basins=basins,
              landscapes={k: dict(name=fix(v["name"]), short=v["short"]) for k, v in short.items()},
              scale=dict(km=sb["km"], px=sb["px"], err=round(100 * max(abs(v - 1) for v in sb["parallel_scale_over_the_frame"]), 1)), conv=J["grid_north"]["convergence_deg"])
    return mj, panels


def cover_photo():
    """The cover photograph (USGS, public domain) as base64 JPEG and its credit, both read from assets/, never typed here.
    credit_line is 'Photo: Stewart Tomlinson, USGS, public domain'; screen 5 prints it as 'Cover photograph: Stewart Tomlinson, USGS, public domain.'"""
    meta = json.loads((ASSETS / "cover_photo.json").read_text(encoding="utf-8"))
    raw = (ASSETS / "cover_photo.jpg").read_bytes()
    assert raw[:3] == b"\xff\xd8\xff", "cover_photo.jpg is not a JPEG"
    assert list(Image.open(io.BytesIO(raw)).size) == meta["asset_px"], "cover_photo.jpg and cover_photo.json disagree on the size"
    credit = "Cover photograph: " + re.sub(r"^\s*Photo:\s*", "", meta["credit_line"]).strip().rstrip(".") + "."
    detail = f"{meta['title']} ({meta['date_taken']}). Source: {meta['source_url']}. Licence: {meta['licence']}."
    return base64.b64encode(raw).decode("ascii"), dict(credit=credit, detail=detail, source_url=meta["source_url"], licence=meta["licence"], title=meta["title"])


def colour_literals(tpl):
    """Colour literals (hex, rgb(a), hsl(a)) of the five-screen template that sit OUTSIDE its two token blocks (:root and :root[data-theme="dark"]).
    Every colour of the deck goes through a CSS token (var(--rain) ...), in the stylesheet and in the strings of the chart scripts, so that the dark
    theme is a toggle of the token values and nothing else. The check reads the raw template, before the data, map and font are injected."""
    css = tpl[tpl.index("<style>"):tpl.index("</style>")]
    dark = re.search(r':root\[data-theme="dark"\]\{[^}]*\}', css)
    assert dark, "the template has no dark token block"
    rest = css[dark.end():] + tpl[tpl.index("</style>"):]
    return re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)", rest)


def main(long=False):
    names = [team.label(r) for r in ("A", "R2", "G1", "G2")]
    if long:
        S = DK.deck()
        parts = [dict(n=i + 1, who=team.label(s["who"]), sec=s["sec"]) for i, s in enumerate(S)]
        notes = [dict(say=a, hand=b) for a, b in DK.NOTES]
    else:
        parts = [dict(n=i + 1, who=team.label(k), sec=sec) for i, (k, sec) in enumerate(PARTS5)]
        notes = [dict(say=a, hand=b) for a, b in NOTES5]
        for (k, sec), (say, hand) in zip(PARTS5, NOTES5):
            n = len((say + " " + hand).split())
            assert n <= WPS * sec, f"notes of a {sec} s screen are too long: {n} words"
        per = {}
        for k, sec in PARTS5:
            per[k] = per.get(k, 0) + sec
        assert all(150 <= v <= 180 for v in per.values()), per
    assert sum(p["sec"] for p in parts) in (DK.TOTAL, 630)
    data = replay_data()
    P = pooled()
    m = re.search(r'^INSTRUCTOR\s*=\s*"([^"]+)"', (ROOT / "code" / "build_proposal.py").read_text(encoding="utf-8"), re.M)
    instructor = m.group(1) if m else ""                        # the proposal's cover line (build_proposal.py, owned by the other session)
    from report_common import REFERENCES as _R
    FALLBACK = {"kratzert2019": "https://hess.copernicus.org/articles/23/5089/2019/", "nearing2024": "https://doi.org/10.1038/s41586-024-07145-1",
                "newman2015": "https://hess.copernicus.org/articles/19/209/2015/", "guan2022": "https://doi.org/10.1175/MWR-D-21-0245.1"}
    SHORT = {"kratzert2019": ("Kratzert et al., 2019", "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets", "HESS 23"),
             "nearing2024": ("Nearing et al., 2024", "Global prediction of extreme floods in ungauged watersheds", "Nature 627"),
             "newman2015": ("Newman et al., 2015", "Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA (CAMELS)", "HESS 19"),
             "guan2022": ("Guan et al., 2022", "GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications", "Mon. Weather Rev. 150")}
    refs = []
    for k in ("kratzert2019", "nearing2024", "newman2015", "guan2022"):
        full = _R[k][1] if k in _R else ""
        m = re.search(r"https?://\S+", full.replace("&ndash;", "-"))
        refs.append(dict(who=SHORT[k][0], title=SHORT[k][1], where=SHORT[k][2], url=(m.group(0).rstrip(".") if m else FALLBACK[k])))
    cfg = dict(title=DK.TITLE, instructor=instructor, refs=refs, sub=DK.SUB, aim=DK.AIM, gap=DK.GAP, rq=DK.RQ, disclosure=DK.DISCLOSURE, names=names, parts=parts, notes=notes,
               pooled=P, headline=HEADLINE, test=TEST, datasets=datasets(),
               schedule=dict(deadline=schedule.d(schedule.DEADLINE, True), proposal_day=schedule.d(schedule.PROPOSAL_DAY, True), stretch=schedule.STRETCH,
                             weeks=[dict(n=w["n"], dates=f"{schedule.d(w['from_'])} – {schedule.d(w['to'])}", title=w["title"], what=w["what"], closes=w["closes"]) for w in schedule.WEEKS],
                             milestones=[[schedule.d(m[0]), m[1]] for m in schedule.MILESTONES]))
    t = (TEMPLATE_LONG if long else TEMPLATE).read_text(encoding="utf-8")
    if long:
        map_b64, map_size = png_b64(FIG / "basin_maps.png", max_w=1200)
        flow_b64, flow_size = png_b64(FIG / "data_flow_talk.png", crop=DK.DATA_CROP, max_w=1800)
        extra = {"__MAP__": map_b64, "__FLOW__": flow_b64}
        what = f"map {map_size}, flow {flow_size}"
    else:
        lit = colour_literals(t)
        assert not lit, f"colour literals outside the CSS tokens in deck5.html (use var(--token), add a token if needed): {sorted(set(lit))[:10]}"
        mj, panels = map_asset()
        photo_b64, photo = cover_photo()
        cfg["photo"] = photo
        areas = [b["area"] for b in mj["basins"]]
        assert f"{min(areas):,.0f} to {max(areas):,.0f} km²" in t, "the area range printed on screen 3 no longer matches the 12 outlines"
        extra = {"__MAPJSON__": json.dumps(mj, separators=(",", ":")), "__PANELS__": json.dumps(panels, separators=(",", ":")), "__PHOTO__": photo_b64}
        what = f"map viewBox {mj['viewBox']}, {len(panels)} panels of {sum(map(len, panels.values())) / 1e3:.0f} kB, photo {len(photo_b64) / 1e3:.0f} kB"
    for key, val in (("__CFG__", json.dumps(cfg)), ("__DATA__", json.dumps(data)), ("__FONT__", font_b64()), ("__ICONS__", icon_symbols()), *extra.items()):
        assert key in t or key in ("__FONT__", "__ICONS__"), key      # the long deck keeps its own font and icons
        t = t.replace(key, val)
    assert not re.search(r"__(CFG|DATA|MAP|MAPJSON|PANELS|PHOTO|FLOW)__", t), "a placeholder was left unfilled"
    out = OUT / (f"{NAME}_long.html" if long else f"{NAME}.html")
    out.write_text(t, encoding="utf-8")
    if not long:
        WEB.mkdir(exist_ok=True)
        shutil.copyfile(out, WEB / "index.html")
        md = ["# The web presentation: five screens, who speaks, what to say", "",
              "`proposal/Flood_Proposal_Presentation.html` (same file as `proposal/web/index.html`). 630 s; the team lead screens 1, 2 and 3 (60 + 90 + 30 s), the others 150 s each. "
              "Keys: arrows or space; N notes; F full screen (the bar at the foot hides after 3 s without mouse movement and returns on movement); T timer; "
              "D dark theme (also the half-moon button next to N; the choice is remembered in the browser; light is the default); ? help. Screen 2: keys 1-4 move the issue time, click a branch. "
              "Screen 3: hover a basin on the map for its area and elevation range, click it for its elevation relief; hover a data product. "
              "Screen 4: P play, S step, O next onset, hover the chart. Screen 5: hover a bar, click a week.", "",
              "| Screen | Presenter | Seconds | Starts at |", "|---|---|---|---|"]
        t0 = 0
        for i, p_ in enumerate(parts, 1):
            md.append(f"| {i} | {p_['who']} | {p_['sec']} | {t0 // 60}:{t0 % 60:02d} |"); t0 += p_["sec"]
        for i, (p_, n) in enumerate(zip(parts, notes), 1):
            md += ["", f"## Screen {i} · {p_['who']} · {p_['sec']} s ({len(n['say'].split())} words)", "", n["say"]] + ([f"**Hand-over:** {n['hand']}"] if n["hand"] else [])
        md += ["", f"{photo['credit']} {photo['detail']}"]
        (OUT / f"{NAME}_notes.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {WEB / 'index.html'}; replay: {len(data['basins'])} basins x {data['n']} days x {len(SETS)} sets; "
          f"{what}")


if __name__ == "__main__":
    main(long="--long" in sys.argv)
