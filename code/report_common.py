"""Shared pieces of the report, slides and Q&A builders: loads every results file once, formats numbers, inlines figures.

Nothing here computes a result. The builders only read results/*.json, results/*.csv and figures/*.png, so the documents can be
rebuilt in seconds after any rerun, and every number in the text is taken from those files, never typed by hand.
"""
import base64
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from common import AQUA, AXIS, BLUE, FIG, GRID, INK, INK2, MUTED, ORANGE, RED, RES, ROOT, SURFACE, YELLOW

FONT = "'Segoe UI','Helvetica Neue',system-ui,sans-serif"
LEAD_WORD = {1: "1 day ahead", 2: "2 days ahead", 3: "3 days ahead"}
MODEL_NAME = {"base_persistence": "Tomorrow = today", "base_climatology": "Calendar-day average", "base_linreg": "Ridge regression",
              "base_rf": "Random forest", "lstm": "LSTM", "lstm1": "LSTM (one basin)", "cmp_lstm": "LSTM", "cmp_transformer": "Transformer"}
INPUT_NAME = {"w": "past weather", "wq": "past weather + river gauge", "wf": "past weather + perfect forecast",
              "wqf": "past weather + river gauge + perfect forecast", "q": "river gauge", "calendar": "calendar only", "none": "calendar only"}
TAG_COL = {"GO": AQUA, "CAUTION": YELLOW, "NOT YET": RED}
TAG_TINT = {"GO": "#d8f1e7", "CAUTION": "#fbebc2", "NOT YET": "#fadbdb"}      # a light fill of the same colour, so that a grid of verdicts reads at a glance


# The only sources the documents may cite (each one checked; add nothing from memory). key -> (short form, full reference)
REFERENCES = {
    "newman2015": ("Newman et al., 2015", "Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., Viger, R., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T., Duan, Q. (2015). Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA. Hydrology and Earth System Sciences, 19, 209&ndash;223."),
    "addor2017": ("Addor et al., 2017", "Addor, N., Newman, A. J., Mizukami, N., Clark, M. P. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. Hydrology and Earth System Sciences, 21, 5293&ndash;5313."),
    "thornton2021": ("Thornton et al., 2021", "Thornton, P. E., Shrestha, R., Thornton, M., Kao, S.-C., Wei, Y., Wilson, B. E. (2021). Gridded daily weather data for North America with comprehensive uncertainty quantification. Scientific Data, 8, 190."),
    "nash1970": ("Nash and Sutcliffe, 1970", "Nash, J. E., Sutcliffe, J. V. (1970). River flow forecasting through conceptual models, part I: a discussion of principles. Journal of Hydrology, 10, 282&ndash;290."),
    "gupta2009": ("Gupta et al., 2009", "Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: implications for improving hydrological modelling. Journal of Hydrology, 377, 80&ndash;91."),
    "hochreiter1997": ("Hochreiter and Schmidhuber, 1997", "Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735&ndash;1780."),
    "vaswani2017": ("Vaswani et al., 2017", "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems 30."),
    "kratzert2018": ("Kratzert et al., 2018", "Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M. (2018). Rainfall&ndash;runoff modelling using Long Short-Term Memory (LSTM) networks. Hydrology and Earth System Sciences, 22, 6005&ndash;6022."),
    "kratzert2019": ("Kratzert et al., 2019a", "Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences, 23, 5089&ndash;5110."),
    "kratzert2019pub": ("Kratzert et al., 2019b", "Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G. S. (2019). Toward improved predictions in ungauged basins: exploiting the power of machine learning. Water Resources Research, 55, 11344&ndash;11354."),
    "kratzert2021": ("Kratzert et al., 2021", "Kratzert, F., Klotz, D., Hochreiter, S., Nearing, G. S. (2021). A note on leveraging synergy in multiple meteorological data sets with deep learning for rainfall&ndash;runoff modeling. Hydrology and Earth System Sciences, 25, 2685&ndash;2703."),
    "feng2020": ("Feng et al., 2020", "Feng, D., Fang, K., Shen, C. (2020). Enhancing streamflow forecast and extracting insights using long-short term memory networks with data integration at continental scales. Water Resources Research, 56, e2019WR026793."),
    "nearing2024": ("Nearing et al., 2024", "Nearing, G., Cohen, D., Dube, V., Gauch, M., Gilon, O., Harrigan, S., Hassidim, A., Klotz, D., Kratzert, F., Metzger, A., Nevo, S., Pappenberger, F., Prudhomme, C., Shalev, G., Shenzis, S., Tekalign, T. Y., Weitzner, D., Matias, Y. (2024). Global prediction of extreme floods in ungauged watersheds. Nature, 627, 559&ndash;563."),
    "gelaro2017": ("Gelaro et al., 2017", "Gelaro, R., and 30 co-authors (2017). The Modern-Era Retrospective Analysis for Research and Applications, version 2 (MERRA-2). Journal of Climate, 30, 5419&ndash;5454."),
    "power": ("NASA POWER", "NASA Langley Research Center, POWER project: daily point data service, https://power.larc.nasa.gov (GPM IMERG rainfall and MERRA-2 variables; retrieved 3 October 2026)."),
    "farr2007": ("Farr et al., 2007", "Farr, T. G., and 17 co-authors (2007). The Shuttle Radar Topography Mission. Reviews of Geophysics, 45, RG2004."),
    "homer2015": ("Homer et al., 2015", "Homer, C., Dewitz, J., Yang, L., Jin, S., Danielson, P., Xian, G., Coulston, J., Herold, N., Wickham, J., Megown, K. (2015). Completion of the 2011 National Land Cover Database for the conterminous United States. Photogrammetric Engineering and Remote Sensing, 81(5), 345&ndash;354."),
    "landsea2013": ("Landsea and Franklin, 2013", "Landsea, C. W., Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new database format. Monthly Weather Review, 141, 3576&ndash;3592."),
    "bevis1992": ("Bevis et al., 1992", "Bevis, M., Businger, S., Herring, T. A., Rocken, C., Anthes, R. A., Ware, R. H. (1992). GPS meteorology: remote sensing of atmospheric water vapor using the Global Positioning System. Journal of Geophysical Research, 97(D14), 15787&ndash;15801."),
    # added 8 October 2026 for sections 4.6 to 4.8; each entry holds only what the checked records hold (the authors of Beck et al. as surnames, because
    # results/review_rain_to_midnight.json, check f2, records no initials; the data sources as results/data_access.json names them, read on 7 October 2026)
    "beck2019": ("Beck et al., 2019", "Beck, Pan, Roy, Weedon, Pappenberger, van Dijk, Huffman, Adler, Wood (2019). Daily evaluation of 26 precipitation datasets using Stage-IV gauge-radar data for the CONUS. Hydrology and Earth System Sciences, 23, 207&ndash;224. https://hess.copernicus.org/articles/23/207/2019/ (page read on 7 October 2026)."),
    "kratzert2024": ("Kratzert et al., 2024", "Kratzert, F., Gauch, M., Klotz, D., Nearing, G. (2024). HESS Opinions: Never train a Long Short-Term Memory (LSTM) network on a single basin. Hydrology and Earth System Sciences, 28, 4187&ndash;4201. https://doi.org/10.5194/hess-28-4187-2024"),
    "guan2022": ("Guan et al., 2022", "Guan, H., Zhu, Y., Sinsky, E., Fu, B., Li, W., Zhou, X., Xue, X., Hou, D., Peng, J., Nageswararao, M. M., Tallapragada, V., Hamill, T. M., Whitaker, J. S., Bates, G., Pegion, P., Frederick, S., Rosencrans, M., Kumar, A. (2022). GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications. Monthly Weather Review, 150, 647&ndash;665. https://doi.org/10.1175/MWR-D-21-0245.1"),
    "aorc": ("NOAA AORC v1.1", "NOAA Analysis of Record for Calibration (AORC), version 1.1: hourly rain on a 1 km grid, 1979&ndash;2025; AWS Open Data, https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/ (read 7 October 2026)."),
    "gefsv12": ("NOAA GEFSv12 reforecast", "NOAA Global Ensemble Forecast System version 12 reforecast, 2000&ndash;2019, 0.25 degree; AWS Open Data, https://noaa-gefs-retrospective.s3.amazonaws.com/ (read 7 October 2026)."),
    "stage4": ("NCEP Stage IV", "NCEP Stage IV hourly radar and gauge precipitation analysis; Iowa Mesonet archive, https://mesonet.agron.iastate.edu/archive/ (read 7 October 2026)."),
}
CITED = []                                                  # keys in the order they are first cited; the source list prints these


def cite(key):
    """'(Newman et al., 2015)' - and remembers the key for the source list. Only keys of REFERENCES are allowed."""
    if key not in CITED:
        CITED.append(key)
    return f"({REFERENCES[key][0]})"


def reference_list(keys=None):
    """The full references as an HTML list: the cited ones (in alphabetical order) or the given keys."""
    keys = sorted(keys or CITED, key=lambda k: REFERENCES[k][1])
    return '<ul class="refs">' + "".join(f"<li>{REFERENCES[k][1]}</li>" for k in keys) + "</ul>"


def load_results():
    """(R, T): every results/*.json as a dict keyed by file name without extension, and every results/*.csv as a table."""
    R = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RES.glob("*.json"))}
    T = {p.stem: pd.read_csv(p, dtype={"gauge_id": str}) for p in sorted(RES.glob("*.csv"))}
    return R, T


def model_name(pred_set):
    """Plain name of a prediction set, e.g. base_rf_wq -> 'Random forest (past weather + river gauge)'."""
    for stem in sorted(MODEL_NAME, key=len, reverse=True):
        if pred_set == stem:
            return MODEL_NAME[stem]
        if pred_set.startswith(stem + "_"):
            inp = pred_set[len(stem) + 1:].split("_s")[0]
            return f"{MODEL_NAME[stem]} ({INPUT_NAME.get(inp, inp)})"
    return pred_set


# ---- numbers ---------------------------------------------------------------------------------------------------------
def _bad(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def n2(v, d=2):
    """0.5987 -> '0.60'; missing -> a dash."""
    return "&ndash;" if _bad(v) else f"{v:.{d}f}".replace("-", "&minus;")


def signed(v, d=2):
    """0.05 -> '+0.05', -0.05 -> '-0.05' with a real minus sign."""
    return "&ndash;" if _bad(v) else ("+" if v >= 0 else "&minus;") + f"{abs(v):.{d}f}"


def pct(v, d=0):
    """0.61 -> '61%'."""
    return "&ndash;" if _bad(v) else f"{100 * v:.{d}f}%"


def count(v):
    """1562 -> '1,562'."""
    return "&ndash;" if _bad(v) else f"{int(round(v)):,}"


# ---- html pieces -----------------------------------------------------------------------------------------------------
def img(name):
    return f'<img src="data:image/png;base64,{base64.b64encode((FIG / name).read_bytes()).decode()}" alt="{name}">'


def fig(name, caption, width=None):
    """A figure from figures/ with its caption; width in % of the text width (default: full)."""
    style = f' style="width:{width}%;margin-left:auto;margin-right:auto"' if width else ""
    return f"<figure{style}>{img(name)}<figcaption>{caption}</figcaption></figure>"


def table(header, rows, cls="num", caption=None, widths=None):
    """header: list of column titles; rows: list of lists of cell strings (already formatted)."""
    cols = "".join(f'<col style="width:{w}">' for w in widths) if widths else ""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<p class="cap">{caption}</p>' if caption else ""
    return f'<table class="{cls}"><colgroup>{cols}</colgroup><tr>{head}</tr>{body}</table>{cap}'


def kpis(items):
    """A row of big numbers: items = [(value string, label string), ...]."""
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items) + "</div>"


def tag(verdict):
    """GO / CAUTION / NOT YET as a coloured label: the outline in the verdict's colour, the fill a light tint of it."""
    return f'<span class="tag" style="border-color:{TAG_COL.get(verdict, AXIS)};background:{TAG_TINT.get(verdict, "transparent")}">{verdict}</span>'


CSS = f"""
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--muted:{MUTED};--grid:{GRID};--axis:{AXIS};--blue:{BLUE};--orange:{ORANGE};--aqua:{AQUA};--yellow:{YELLOW};--red:{RED}}}
@page{{size:A4;margin:12mm 13mm}}
body{{background:var(--surface);color:var(--ink);font:9.6pt/1.42 {FONT};margin:0}}
h1{{font-size:18pt;margin:0 0 1pt}} h2{{font-size:13pt;margin:15pt 0 4pt;break-after:avoid}} h3{{font-size:10.4pt;margin:10pt 0 3pt;break-after:avoid}}
p{{margin:0 0 5pt}} ul,ol{{margin:0 0 5pt;padding-left:14pt}} li{{margin-bottom:2pt}}
.sub{{color:var(--ink2);margin:0 0 7pt}} code{{font-size:8.3pt;color:var(--ink2)}}
table{{border-collapse:collapse;width:100%;font-size:8.7pt;margin:3pt 0 2pt}} th,td{{text-align:left;vertical-align:top;padding:3pt 6pt;border-top:1px solid var(--grid)}}
th{{color:var(--muted);font-weight:600}} .num td{{font-variant-numeric:tabular-nums}} .rec th{{width:96pt;color:var(--ink2)}}
tr{{break-inside:avoid}}
.verdict{{border:2px solid var(--blue);border-radius:9px;padding:7pt 11pt;margin:6pt 0 8pt;font-size:10.3pt}}
.note{{border-left:3px solid var(--orange);padding:4pt 9pt;margin:7pt 0;font-size:8.9pt;background:#fdf6f2}}
.ask{{border-left:3px solid var(--blue);padding:4pt 9pt;margin:7pt 0;font-size:8.9pt;background:#f3f7fd}}
figure{{margin:6pt 0;break-inside:avoid}} figure img{{width:100%}} figcaption,.cap{{font-size:8.1pt;color:var(--ink2);margin-top:2pt}}
.pb{{break-before:page}} .blk{{break-inside:avoid}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:10pt;align-items:start}} .two figure{{margin:0}}
.kpis{{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:8pt;margin:6pt 0}}
.kpi{{border:1px solid var(--grid);border-radius:8px;padding:5pt 8pt}} .kpi .v{{font-size:15pt;font-weight:600}} .kpi .l{{font-size:8.1pt;color:var(--ink2)}}
.refs{{font-size:8.2pt;color:var(--ink2);padding-left:12pt}} .refs li{{margin-bottom:2.5pt}}
.tag{{display:inline-block;border:1.6px solid var(--axis);border-radius:20px;padding:0 7pt;font-size:8.2pt;font-weight:600;white-space:nowrap}}
"""


def page(title, body, css_extra=""):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}{css_extra}</style></head><body>{body}</body></html>'


def find_browser():
    """A Chromium browser for printing: Chrome or Edge on macOS and Windows, or one on the PATH (Linux); None when there is none."""
    paths = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")
    found = next((p for p in paths if Path(p).exists()), None)
    return found or next((w for w in map(shutil.which, ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge")) if w), None)


def print_pdf(html_path, pdf_path, quiet=False):
    """Print an HTML file to PDF with a headless Chromium browser (Chrome on the Mac, Edge on the Windows PC).
    quiet: say nothing when no browser is found (for a test print)."""
    browser = find_browser()
    if not browser:
        if not quiet:
            print("no Chromium browser found - HTML only:", html_path)
        return False
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", Path(html_path).as_uri()],
                   check=True, timeout=180, capture_output=True)
    return True


def pdf_pages(pdf_path):
    """The number of pages of a PDF that print_pdf wrote, counted from its page objects (no PDF library is installed); None if it cannot be read."""
    try:
        return len(re.findall(rb"/Type\s*/Page(?![s\w])", Path(pdf_path).read_bytes())) or None
    except OSError:
        return None
