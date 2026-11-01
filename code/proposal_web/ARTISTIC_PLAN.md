# Artistic plan for the five-screen web presentation (written 2026-10-14)

Self-contained brief: a fresh session (any model) can execute it from this file alone.

## What exists
- The deck: `proposal/Flood_Proposal_Presentation.html` (copy at `proposal/web/index.html`), built by
  `~/AI_Class/.venv/bin/python code/build_proposal_web.py` (run from the project root `~/AI_Class/Project-Flood`) from the template
  `code/proposal_web/deck5.html`. The builder injects `__CFG__` (text, parts, notes, pooled numbers, schedule, datasets), `__DATA__`
  (the replay data), `__MAP__` (figures/basin_maps.png), `__FONT__` (Inter, `code/proposal_web/assets/`), `__ICONS__` (Lucide SVGs in
  `code/proposal_web/assets/icons/`, map in `ICONS` of the builder). `--long` builds the old 12-screen version from `deck.html` (leave it alone).
- Screenshots: `~/AI_Class/.venv/bin/python code/proposal_web/shots5.py proposal/Flood_Proposal_Presentation.html <outdir>` (Playwright +
  the installed Chrome; exercises the slider, the mind maps, the replay, the ladder). Look at the PNGs after every build.
- Five screens, fixed: 1 cover (real-onset chart), 2 issue-time clock + research-question tree, 3 map + data mind map, 4 the replay
  (DO NOT change its layout: the user asked to keep it), 5 target tiles + ladder + timeline + questions.

## Rules that must survive every change
- Five screens, one file, nothing loaded from the internet (no CDN, no web font links); size under 4 MB.
- Every number comes from `CFG`/`DATA` (results files) as now; never type a result into the template; keep "preliminary", the AORC
  caveat ("an analysis made after the fact"), targets as targets, the replay note.
- `prefers-reduced-motion`: all animation off.
- After each phase: build, run shots5.py, look at every screen, fix overflow or collisions, then go on. No console errors.
- Do not touch `code/build_proposal.py`, `code/build_proposal_slides.py`, `code/build_presentation_plan_proposal.py`, `code/schedule.py`
  (another session owns them).

## Phase A, foundation (do first)
1. Tokens: `--rain` (blue), `--river` (navy), `--warn` (orange), `--land-m`, `--land-n`, `--land-s` (three landscape colours, not orange),
   plus surfaces and inks, each with a light value in `:root` and a dark value in `:root[data-theme="dark"]`.
   Charts (cover chart, clock SVG, lead-time bars, replay `draw()`, ladder) use `style="fill:var(--rain)"` etc. instead of hex.
2. One spacing grid (8 px) and one corner radius (12 px). Outlined cards become tonal panels (`--panel`: pale blue on off-white); only the
   hero of each screen keeps a shadow: screen 2 the big number, 3 the map, 4 the chart, 5 the ladder.
3. Cover key, one line: "blue = rain · navy = the river · orange = a warning".

## Phase B, motion
1. Cover: rain lines fall (CSS keyframes, staggered delays), the river paths drift (slow `stroke-dashoffset` or translate loop); the
   hydrograph keeps drawing itself.
2. Screens 2 and 3: branch paths draw in sequence (`stroke-dasharray` animation with per-branch delay on enter), leaves fade after.
3. Screen 3 funnel numbers count up on enter (same count-up as the big number of screen 2).
4. Replay: when a forecast dot is above the flood level and the previous `today` had none, add a halo circle with a scale-and-fade
   animation and flash the WARNING badge once (CSS class toggled in `draw()`).
5. A faint river curve (8 % opacity) at the foot of screens 2 to 5.

## Phase C, the custom map (screen 3)
1. Data: the 12 outlines from the CAMELS shapefile via `fetch_aorc.outlines(ids)` (see `code/fetch_aorc.py`; ids from
   `AO.ids_of_the_study()`), simplified with shapely (tolerance about 0.01°), plus the land polygon from cartopy's Natural Earth files in
   `~/.local/share/cartopy` (50 m land or coastline; no country borders are installed, none needed). Project with pyproj to EPSG:5070
   (Conus Albers), fit to a viewBox, write compact SVG paths; the builder injects them as `__MAPSVG__` (keep the PNG for the panels).
2. Draw: pale land, hairline coast, outlines filled by landscape (`--land-*`), gauges as white dots; scale bar, north arrow, caption with
   CRS (geographic lat/lon source, drawn in EPSG:5070), legend and sources (CAMELS-US outlines; Natural Earth coastline).
3. Hover a basin: name, area, elevation range (from `DATA.basins` and results/data_audit.json). Click: show that basin's elevation panel,
   cropped from figures/basin_maps.png (a 4 x 3 grid of panels; compute the crop by row and column) beside the map.

## Phase D, cinematic full screen
- In `document.fullscreenElement`, hide `#chrome` after 3 s without mouse movement, show it on movement; `fit()` then uses the full height.

## Phase E, dark theme
- Dark token values from the validated palette (blue #3987e5, orange #d95926, aqua #199e70; surfaces about #0b1627 / #13223a; inks
  #e8eef7 / #c3c2b7). Key `D` toggles `data-theme`, remembered in localStorage (wrapped in try/catch). Default light.
- Run the validator: `node /private/tmp/claude-501/bundled-skills/2.1.287/940a06c38f622340317d9658ceedfe33/dataviz/scripts/validate_palette.js "<hex,hex>" --mode dark --surface "#0b1627"` (path may differ in a new session: the dataviz skill's `scripts/validate_palette.js`).

## Phase F, the photograph
- Prefer a public-domain USGS photograph of a flooded small south-eastern river (USGS Multimedia Gallery); otherwise Wikimedia Commons
  with a CC licence verified on the file page. Resize to 1600 px, JPEG quality 70 (about 150 KB), embed as base64 under the cover at
  opacity about 0.18 beneath a navy gradient; credit line on screen 5 (photographer, source, licence).

## Phase G, verification
- shots5.py in light and dark; the validator for both themes; file under 4 MB; no console errors; rebuild `Flood_Proposal_Presentation_notes.md`
  (the builder writes it); copy to `proposal/web/index.html` (the builder does); report with the screenshot paths and anything unsure.
