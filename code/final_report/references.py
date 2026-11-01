"""Verified IEEE references for the Project-Flood final report (hand-in item 3, the Word report).

REFERENCES is an ordered dict  key -> {"ieee", "doi", "verified", "supports"}:
  ieee      the full IEEE entry without the "[n]", as plain text (the journal name is set in italics: see ITALIC and runs())
  doi       the bare DOI ("" when the work has none)
  verified  how the entry was checked: the literature audit of 28 Oct 2026, plus the live re-read made when this file was written
  supports  one line: the claim in this project that the reference supports, with the results file and key where there is one
LABELS gives every key a short label ("Newman et al., 2015") and a venue; they feed REFERENCES_KEYS.md and the Citation column of table T4.

Use (no third-party imports; run Python with -X utf8 on this Windows PC):
    from references import REFERENCES, cite_order, numbering, marker, format_ieee, runs
    first_cited = ["newman2015", "addor2017", "newman2015", "kratzert2018"]   # keys in the order the text cites them (repeats are fine)
    num = numbering(first_cited)               # {"newman2015": 1, "addor2017": 2, "kratzert2018": 3}
    marker(num, "newman2015", "addor2017")     # "[1], [2]"  (three or more consecutive numbers collapse to a range: "[1]–[3]")
    cite_order(first_cited)                    # ["[1] A. J. Newman et al., ...", "[2] ...", "[3] ..."]
IEEE numbers follow the order of FIRST citation in the finished text, so the builder collects the keys while it assembles the
document and numbers them last. Only the keys that were cited reach the reference list; an unknown key raises KeyError.
Command line:  python -X utf8 references.py  (self-check and count),  --list  (every entry, numbered in file order),  --keys  (the key table).

Sources and rules
 - The entries are copied from the literature audit (scratchpad audits/agent1.md, 28 Oct 2026): author initials, titles, venues,
   volumes, issues, pages, years and DOIs are Crossref's. Six authors or fewer are listed in full; seven or more: first author + et al.
 - The data-product entries (camels_ts, camels_attr, nlcd_annual) come from the DataCite record of the DOI; aorc, gefsv12 and
   stage4 have no DOI and carry the registry or archive link, with the titles the project recorded in results/data_flow.json.
 - Keys equal the keys of code/report_common.py REFERENCES wherever the same work is there (REPORT_COMMON_KEYS); the other keys are
   new, in the same style (author + year, or the product name).
 - "Verified" means the bibliographic data match Crossref or DataCite. It does NOT mean that a team member has read the paper:
   results/published_scores.json (checked_by) and results/sources_checked.json (notes) record that no member has read these papers yet.
   Never write that the team reviewed them without saying who read them.
 - Left out on purpose (the audit marked them non-existent, unverified or not needed):
     "Hamill et al. 2022 BAMS" (does not exist as described: Hamill et al. 2022 is a Mon. Weather Rev. paper on the GEFSv12
     reanalysis, and the GEFSv12 reforecast dataset paper is Guan et al. 2022 = guan2022);
     homer2015 (DOI 10.14358/PERS.81.5.345 returns 404 and the paper is not found in Crossref; it is also the older single-year
     'NLCD 2011', not the Annual NLCD the project read, which is nlcd_annual); bevis1992 (not in the audit's verified list);
     gelaro2017, power and the IMERG paper (MERRA-2 and IMERG through NASA POWER are not in the keep-list of this file);
     Lin and Mitchell 2005 (no DOI; the Stage IV product description) and the other optional papers of the audit (O4 excepted, below).
 - frame2022 (audit O4) is here although the keep-list did not name it: row 8 of table T4 (audit section 5) cites it.
 - No Daymet data DOI is listed: the Daymet version inside CAMELS is not recorded in the project (results/data_flow.json, unverified,
   camels_us.camels_daymet), and ORNL DAAC has separate DOIs for its versions. Cite newman2015 and addor2017 for the forcing and
   thornton2021 for the Daymet V4 time-of-observation finding.
"""
from collections import OrderedDict
import difflib

AUDIT_DATE = "28 Oct 2026"

REFERENCES = OrderedDict()   # key -> {"ieee", "doi", "verified", "supports"}
LABELS = OrderedDict()       # key -> {"short", "venue"}
ITALIC = {}                  # key -> substrings of the ieee text that IEEE sets in italics (journal or proceedings name)

# keys that code/report_common.py REFERENCES already uses for the same work (writers may keep citing with these)
REPORT_COMMON_KEYS = frozenset([
    "newman2015", "addor2017", "thornton2021", "nash1970", "gupta2009", "hochreiter1997", "vaswani2017", "kratzert2018",
    "kratzert2019", "kratzert2019pub", "kratzert2021", "feng2020", "nearing2024", "farr2007", "landsea2013", "beck2019",
    "kratzert2024", "guan2022", "aorc", "gefsv12", "stage4",
])


def _add(key, *, short, venue, ieee, doi, verified, supports, italic=()):
    if key in REFERENCES:
        raise ValueError("duplicate reference key: " + key)
    REFERENCES[key] = {"ieee": ieee, "doi": doi, "verified": verified, "supports": supports}
    LABELS[key] = {"short": short, "venue": venue}
    if italic:
        ITALIC[key] = tuple(italic)


# ---------------------------------------------------------------------------------------------------------------------------
# The references. Order: the 16 core references of the audit (its suggested order of first citation), the optional ones kept,
# then the data products.
#
# UNVERIFIED (kept out of the entries on purpose):
#   - xia2012: the article number "D03109" that is often quoted; Crossref registers "2011JD016048" and no pages.
#   - farr2007: the article number "RG2004" of the old project entry; Crossref registers "2005RG000183" and no pages.
#   - vaswani2017: the page numbers "5998-6008"; they are not on the NeurIPS page.
#   - frame2022: the DOI of its 2023 corrigendum (mentioned in results/published_context.json) was not looked up.
#   - nlcd_annual: the collection number (1.0, 1.1 or 1.2) of the 2011 layer read on 3 Oct 2026.
#   - the Daymet version inside CAMELS (see the module docstring).
# ---------------------------------------------------------------------------------------------------------------------------

_add("newman2015",  # audit [1]
     short="Newman et al., 2015",
     venue="Hydrol. Earth Syst. Sci. 19(1)",
     doi="10.5194/hess-19-209-2015",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'A. J. Newman et al., "Development of a large-sample watershed-scale hydrometeorological data set for the '
         "contiguous USA: data set characteristics and assessment of regional variability in hydrologic model "
         'performance," Hydrol. Earth Syst. Sci., vol. 19, no. 1, pp. 209–223, 2015, doi: 10.5194/hess-19-209-2015.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 12 authors, so IEEE "
         "lists the first author with et al. The title is given with its subtitle exactly as Crossref registers it; "
         "older project texts (report_common.py, the proposal, the web deck) shorten it and spell 'dataset'."
     ),
     supports=(
         "The data: CAMELS-US daily flow and basin-mean Daymet weather (results/data_flow.json datasets[0].sources "
         "and documentation_read[0]); section 4.1 data inventory; cite it together with addor2017, as the CAMELS "
         "readme asks."
     ))

_add("addor2017",  # audit [2]
     short="Addor et al., 2017",
     venue="Hydrol. Earth Syst. Sci. 21(10)",
     doi="10.5194/hess-21-5293-2017",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'N. Addor, A. J. Newman, N. Mizukami, and M. P. Clark, "The CAMELS data set: catchment attributes and '
         'meteorology for large-sample studies," Hydrol. Earth Syst. Sci., vol. 21, no. 10, pp. 5293–5313, 2017, '
         "doi: 10.5194/hess-21-5293-2017."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 4 authors, all listed."
     ),
     supports=(
         "The static basin properties used as model inputs (results/data_flow.json all_information.static_properties "
         "and datasets[0]); section 4.1; cite it together with newman2015."
     ))

_add("thornton2021",  # audit [3]
     short="Thornton et al., 2021",
     venue="Sci. Data 8(1)",
     doi="10.1038/s41597-021-00973-0",
     italic=("Sci. Data",),
     ieee=(
         'P. E. Thornton, R. Shrestha, M. Thornton, S.-C. Kao, Y. Wei, and B. E. Wilson, "Gridded daily weather data '
         'for North America with comprehensive uncertainty quantification," Sci. Data, vol. 8, no. 1, Art. no. 190, '
         "2021, doi: 10.1038/s41597-021-00973-0."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, article number, year and author initials match; 6 authors, all "
         "listed. Art. no. 190 is the article number Crossref registers."
     ),
     supports=(
         "Daymet V4 shifts the rain of stations that observe before noon back one day, a timing bias not examined in "
         "earlier versions (results/sources_checked.json sources[0]); the CAMELS files predate V4, so cite it for "
         "the clock argument, not as the CAMELS product; sections 2.3 and 4.2."
     ))

_add("kratzert2021",  # audit [4]
     short="Kratzert et al., 2021",
     venue="Hydrol. Earth Syst. Sci. 25(5)",
     doi="10.5194/hess-25-2685-2021",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'F. Kratzert, D. Klotz, S. Hochreiter, and G. S. Nearing, "A note on leveraging synergy in multiple '
         'meteorological data sets with deep learning for rainfall–runoff modeling," Hydrol. Earth Syst. Sci., vol. '
         "25, no. 5, pp. 2685–2703, 2021, doi: 10.5194/hess-25-2685-2021."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 4 authors, all listed."
     ),
     supports=(
         "A one-day lag between CAMELS rain products (Daymet and Maurer) and the gain from combining forcing "
         "products (results/sources_checked.json sources[1]; results/published_context.json sources.kratzert2021); "
         "supports the clock diagnosis and the NLDAS experiment (script 10)."
     ))

_add("hochreiter1997",  # audit [5]
     short="Hochreiter and Schmidhuber, 1997",
     venue="Neural Comput. 9(8)",
     doi="10.1162/neco.1997.9.8.1735",
     italic=("Neural Comput.",),
     ieee=(
         'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, no. 8, pp. 1735–1780, '
         "1997, doi: 10.1162/neco.1997.9.8.1735."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 2 authors, all listed. "
         "The proposal reference [8] omits 'no. 8'; this entry has it."
     ),
     supports=(
         "The model class: one small LSTM shared by all rivers (section 5.2)."
     ))

_add("kratzert2018",  # audit [6]
     short="Kratzert et al., 2018",
     venue="Hydrol. Earth Syst. Sci. 22(11)",
     doi="10.5194/hess-22-6005-2018",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'F. Kratzert, D. Klotz, C. Brenner, K. Schulz, and M. Herrnegger, "Rainfall–runoff modelling using Long '
         'Short-Term Memory (LSTM) networks," Hydrol. Earth Syst. Sci., vol. 22, no. 11, pp. 6005–6022, 2018, doi: '
         "10.5194/hess-22-6005-2018."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 5 authors, all listed."
     ),
     supports=(
         "First LSTM rainfall-runoff benchmark on CAMELS with Daymet forcing, including the South Atlantic-Gulf "
         "region that holds the study basins (results/published_scores.json sources.kratzert2018.regions); section "
         "2.2 and table T4."
     ))

_add("kratzert2019",  # audit [7]
     short="Kratzert et al., 2019a",
     venue="Hydrol. Earth Syst. Sci. 23(12)",
     doi="10.5194/hess-23-5089-2019",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'F. Kratzert, D. Klotz, G. Shalev, G. Klambauer, S. Hochreiter, and G. Nearing, "Towards learning '
         "universal, regional, and local hydrological behaviors via machine learning applied to large-sample "
         'datasets," Hydrol. Earth Syst. Sci., vol. 23, no. 12, pp. 5089–5110, 2019, doi: 10.5194/hess-23-5089-2019.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 6 authors, all listed."
     ),
     supports=(
         "The design choice of one LSTM shared by all basins with static attributes, and the NSE* loss; published "
         "scores are set beside the upper-bound rows of this project only (results/published_scores.json "
         "sources.kratzert2019 and rows_of_this_project_to_set_beside_them); sections 2.2, 5.2, 8.1."
     ))

_add("kratzert2019pub",  # audit [8]
     short="Kratzert et al., 2019b",
     venue="Water Resour. Res. 55(12)",
     doi="10.1029/2019WR026065",
     italic=("Water Resour. Res.",),
     ieee=(
         'F. Kratzert, D. Klotz, M. Herrnegger, A. K. Sampson, S. Hochreiter, and G. S. Nearing, "Toward improved '
         'predictions in ungauged basins: Exploiting the power of machine learning," Water Resour. Res., vol. 55, '
         "no. 12, pp. 11344–11354, 2019, doi: 10.1029/2019WR026065."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 6 authors, all listed. "
         "The entry in report_common.py lacked the issue number; this one has it."
     ),
     supports=(
         "Prediction in basins that supplied no training data, set beside the rivers-never-seen test of this "
         "project, which is NOT comparable in level (results/published_scores.json "
         "rows_of_this_project_to_set_beside_them.basin_never_seen; results/lstm_lobo.json; results/more_basins.json "
         "outcome_full); sections 5.4 and 8.1."
     ))

_add("feng2020",  # audit [9]
     short="Feng et al., 2020",
     venue="Water Resour. Res. 56(9)",
     doi="10.1029/2019WR026793",
     italic=("Water Resour. Res.",),
     ieee=(
         'D. Feng, K. Fang, and C. Shen, "Enhancing streamflow forecast and extracting insights using long-short '
         'term memory networks with data integration at continental scales," Water Resour. Res., vol. 56, no. 9, '
         "Art. no. e2019WR026793, 2020, doi: 10.1029/2019WR026793."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, article number, year and author initials match; 3 authors, all "
         "listed. Art. no. e2019WR026793 is the article number Crossref registers."
     ),
     supports=(
         "The river gauge as an input (data integration of lagged flow); the published value is set beside the "
         "perfect-weather upper bound of this project, because that forecast still reads the weather of the target "
         "day (results/published_scores.json sources.feng2020 and how_to_word_it); sections 2.2 and 8.1."
     ))

_add("kratzert2024",  # audit [10]
     short="Kratzert et al., 2024",
     venue="Hydrol. Earth Syst. Sci. 28(17)",
     doi="10.5194/hess-28-4187-2024",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'F. Kratzert, M. Gauch, D. Klotz, and G. Nearing, "HESS Opinions: Never train a Long Short-Term Memory '
         '(LSTM) network on a single basin," Hydrol. Earth Syst. Sci., vol. 28, no. 17, pp. 4187–4201, 2024, doi: '
         "10.5194/hess-28-4187-2024."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 4 authors, all listed."
     ),
     supports=(
         "Why one network is trained on many rivers rather than one per river (scripts 03 and 04; "
         "results/more_basins.json outcome_full); sections 2.2, 5.3, 8.1."
     ))

_add("nearing2024",  # audit [11]
     short="Nearing et al., 2024",
     venue="Nature 627(8004)",
     doi="10.1038/s41586-024-07145-1",
     italic=("Nature",),
     ieee=(
         'G. Nearing et al., "Global prediction of extreme floods in ungauged watersheds," Nature, vol. 627, no. '
         "8004, pp. 559–563, 2024, doi: 10.1038/s41586-024-07145-1."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 18 authors, so IEEE "
         "lists the first author with et al. Same DOI as on the web deck."
     ),
     supports=(
         "State of the art in AI flood warning, and the frame for the warning task of this project (onset warned, "
         "false warning, the GO rule); sections 2.1 and 2.2."
     ))

_add("nash1970",  # audit [12]
     short="Nash and Sutcliffe, 1970",
     venue="J. Hydrol. 10(3)",
     doi="10.1016/0022-1694(70)90255-6",
     italic=("J. Hydrol.",),
     ieee=(
         'J. E. Nash and J. V. Sutcliffe, "River flow forecasting through conceptual models part I — A discussion of '
         'principles," J. Hydrol., vol. 10, no. 3, pp. 282–290, 1970, doi: 10.1016/0022-1694(70)90255-6.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 2 authors, all listed."
     ),
     supports=(
         "The NSE metric (section 6.3; every nse_median key in results/*.json)."
     ))

_add("gupta2009",  # audit [13]
     short="Gupta et al., 2009",
     venue="J. Hydrol. 377(1-2)",
     doi="10.1016/j.jhydrol.2009.08.003",
     italic=("J. Hydrol.",),
     ieee=(
         'H. V. Gupta, H. Kling, K. K. Yilmaz, and G. F. Martinez, "Decomposition of the mean squared error and NSE '
         'performance criteria: Implications for improving hydrological modelling," J. Hydrol., vol. 377, no. 1–2, '
         "pp. 80–91, 2009, doi: 10.1016/j.jhydrol.2009.08.003."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 4 authors, all listed. "
         "The issue is '1-2' as Crossref registers it."
     ),
     supports=(
         "The KGE reported beside NSE (code/04_lstm_all.py; results/lstm_all.json "
         "sets.*.test.summary.l1.kge_median); section 6.3."
     ))

_add("fall2023",  # audit [14]
     short="Fall et al., 2023",
     venue="JAWRA J. Amer. Water Resour. Assoc. 59(6)",
     doi="10.1111/1752-1688.13143",
     italic=("JAWRA J. Amer. Water Resour. Assoc.",),
     ieee=(
         'G. Fall et al., "The Office of Water Prediction\'s Analysis of Record for Calibration, version 1.1: '
         'Dataset description and precipitation evaluation," JAWRA J. Amer. Water Resour. Assoc., vol. 59, no. 6, '
         "pp. 1246–1272, 2023, doi: 10.1111/1752-1688.13143."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 9 authors, so IEEE "
         "lists the first author with et al."
     ),
     supports=(
         "The AORC v1.1 hourly rain of E1b, which the report calls 'an analysis made after the fact' "
         "(results/aorc_rain_to_midnight.json outcome_full); sections 4.1 and 6.2; the data access record is aorc."
     ))

_add("nelson2016",  # audit [15]
     short="Nelson et al., 2016",
     venue="Weather Forecast. 31(2)",
     doi="10.1175/WAF-D-14-00112.1",
     italic=("Weather Forecast.",),
     ieee=(
         'B. R. Nelson, O. P. Prat, D.-J. Seo, and E. Habib, "Assessment and implications of NCEP Stage IV '
         'quantitative precipitation estimates for product intercomparisons," Weather Forecast., vol. 31, no. 2, pp. '
         "371–394, 2016, doi: 10.1175/WAF-D-14-00112.1."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 4 authors, all listed."
     ),
     supports=(
         "The real-time Stage IV check, 'a check on the test years, not a service' (results/stage4_realtime.json "
         "models.wqa.variants.stage4.share_of_the_onsets); sections 6.2 and 7.3; the data access record is stage4."
     ))

_add("guan2022",  # audit [16]
     short="Guan et al., 2022",
     venue="Mon. Weather Rev. 150(3)",
     doi="10.1175/MWR-D-21-0245.1",
     italic=("Mon. Weather Rev.",),
     ieee=(
         'H. Guan et al., "GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological '
         'applications," Mon. Weather Rev., vol. 150, no. 3, pp. 647–665, 2022, doi: 10.1175/MWR-D-21-0245.1.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 18 authors, so IEEE "
         "lists the first author with et al. Same DOI as in report_common.py and on the web deck."
     ),
     supports=(
         "E2: the archived GEFSv12 control-member rain forecast as an input (results/forecast_input.json "
         "e2_success.by_lead and outcome_full); sections 4.1 and 6.2; the data access record is gefsv12."
     ))

_add("hersbach2020",  # audit O1
     short="Hersbach et al., 2020",
     venue="Q. J. R. Meteorol. Soc. 146(730)",
     doi="10.1002/qj.3803",
     italic=("Q. J. R. Meteorol. Soc.",),
     ieee=(
         'H. Hersbach et al., "The ERA5 global reanalysis," Q. J. R. Meteorol. Soc., vol. 146, no. 730, pp. '
         "1999–2049, 2020, doi: 10.1002/qj.3803."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 43 authors, so IEEE "
         "lists the first author with et al. The ERA5 data DOI 10.24381/cds.adbb2d47 resolves (doi.org 302 to the "
         "Climate Data Store; DataCite title 'ERA5 hourly data on single levels from 1940 to present')."
     ),
     supports=(
         "E1: hourly ERA5 rain, read through the Open-Meteo archive API (results/data_flow.json datasets[4]), as an "
         "input with rain to local midnight (results/rain_to_midnight.json outcome_full)."
     ))

_add("xia2012",  # audit O2
     short="Xia et al., 2012",
     venue="J. Geophys. Res. Atmos. 117(D3)",
     doi="10.1029/2011JD016048",
     italic=("J. Geophys. Res. Atmos.",),
     ieee=(
         'Y. Xia et al., "Continental-scale water and energy flux analysis and validation for the North American '
         "Land Data Assimilation System project phase 2 (NLDAS-2): 1. Intercomparison and application of model "
         'products," J. Geophys. Res. Atmos., vol. 117, no. D3, 2012, Art. no. D03109, doi: 10.1029/2011JD016048.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, year and author initials match; 17 authors, so IEEE lists the "
         "first author with et al. Crossref registers the electronic article number (2011JD016048) and no pages; the "
         "printed AGU citation number D03109 is added after the references audit of 30 Oct 2026, which asked for it so "
         "that the list treats article numbers alike."
     ),
     supports=(
         "NLDAS-2 rain, the CAMELS forcing product whose day ends in the afternoon, about 17:30 local standard time "
         "(measured median 17:40: results/day_windows.json day_ends.nldas); the NLDAS experiment, script 10."
     ))

_add("vaswani2017",  # audit O3
     short="Vaswani et al., 2017",
     venue="Adv. Neural Inf. Process. Syst. 30 (NIPS 2017)",
     doi="",
     italic=("Advances in Neural Information Processing Systems 30 (NIPS 2017)",),
     ieee=(
         'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems 30 '
         "(NIPS 2017), 2017. [Online]. Available: "
         "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
     ),
     verified=(
         "No Crossref record exists for the NeurIPS paper. Title, eight authors and venue were read on the NeurIPS "
         "proceedings page (HTTP 200 in the literature audit on 28 Oct 2026, and again, with the title on the page, "
         "when this file was written). The Crossref hits 10.65215/... (2025, posted-content) belong to a third-party "
         "preprint mirror and are not used. The arXiv DOI 10.48550/arXiv.1706.03762 resolves but is a different "
         "object (the preprint), so the entry carries no DOI. The page numbers are not on the page and are left out."
     ),
     supports=(
         "Only if the Transformer comparison stays (script 09; results/model_comparison.json): the Transformer model "
         "class. That comparison used its own two-seed LSTM with shorter sequences (results/model_comparison.json "
         "seeds and settings.lstm; critic C9), so its LSTM score differs from the headline LSTM."
     ))

_add("beck2019",  # audit O7
     short="Beck et al., 2019",
     venue="Hydrol. Earth Syst. Sci. 23(1)",
     doi="10.5194/hess-23-207-2019",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'H. E. Beck et al., "Daily evaluation of 26 precipitation datasets using Stage-IV gauge-radar data for the '
         'CONUS," Hydrol. Earth Syst. Sci., vol. 23, no. 1, pp. 207–224, 2019, doi: 10.5194/hess-23-207-2019.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 9 authors, so IEEE "
         "lists the first author with et al. The initials come from Crossref (report_common.py held surnames only)."
     ),
     supports=(
         "Its abstract names gauge reporting times as a factor in how well precipitation products match Stage IV, "
         "which supports the clock argument (section 2.3)."
     ))

_add("farr2007",  # audit O21
     short="Farr et al., 2007",
     venue="Rev. Geophys. 45(2)",
     doi="10.1029/2005RG000183",
     italic=("Rev. Geophys.",),
     ieee=(
         'T. G. Farr et al., "The Shuttle Radar Topography Mission," Rev. Geophys., vol. 45, no. 2, 2007, Art. no. '
         "RG2004, doi: 10.1029/2005RG000183."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, year and author initials match; 18 authors, so IEEE lists the "
         "first author with et al. Crossref registers the electronic article number (2005RG000183) and no pages; the "
         "printed AGU citation number RG2004 is added after the references audit of 30 Oct 2026."
     ),
     supports=(
         "SRTM elevation behind the basin maps and the check of the CAMELS elevation (results/basin_gis.json); no "
         "model reads it."
     ))

_add("landsea2013",  # audit O22
     short="Landsea and Franklin, 2013",
     venue="Mon. Weather Rev. 141(10)",
     doi="10.1175/MWR-D-12-00254.1",
     italic=("Mon. Weather Rev.",),
     ieee=(
         'C. W. Landsea and J. L. Franklin, "Atlantic hurricane database uncertainty and presentation of a new '
         'database format," Mon. Weather Rev., vol. 141, no. 10, pp. 3576–3592, 2013, doi: 10.1175/MWR-D-12-00254.1.'
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 2 authors, all listed."
     ),
     supports=(
         "HURDAT2 storm matching in the flood-event analysis (results/flood_analysis.json events.hurdat; "
         "code/06_flood_analysis.py)."
     ))

_add("frame2022",  # audit O4
     short="Frame et al., 2022",
     venue="Hydrol. Earth Syst. Sci. 26(13)",
     doi="10.5194/hess-26-3377-2022",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=(
         'J. M. Frame et al., "Deep learning rainfall–runoff predictions of extreme events," Hydrol. Earth Syst. '
         "Sci., vol. 26, no. 13, pp. 3377–3392, 2022, doi: 10.5194/hess-26-3377-2022."
     ),
     verified=(
         "Crossref record of the DOI read by the literature audit on 28 Oct 2026 and read again live when this file "
         "was written: title, journal, volume, issue, pages, year and author initials match; 9 authors, so IEEE "
         "lists the first author with et al. A 2023 corrigendum is mentioned in results/published_context.json; its "
         "DOI was not looked up."
     ),
     supports=(
         "Table T4, row 8 (audit section 5): LSTM skill on extreme events, in a simulation setting with no "
         "warning-timing score; discussion in section 2.2."
     ))

# data products (audit section 3; metadata from DataCite or from the project record)
_add("camels_ts",  # audit section 3
     short="Newman, 2014 (CAMELS data)",
     venue="NSF NCAR dataset (DataCite)",
     doi="10.5065/D6MW2F4D",
     ieee=(
         'A. Newman, "A large-sample watershed-scale hydrometeorological dataset for the contiguous USA," NSF '
         "National Center for Atmospheric Research, 2014, doi: 10.5065/D6MW2F4D."
     ),
     verified=(
         "The DOI is the one the NCAR CAMELS page names for the time series (results/data_flow.json "
         "documentation_read[3]). DataCite record read on 28 Oct 2026: title, creator (the record lists one, Newman, "
         "Andrew), publisher and year as registered. doi.org resolves (302 to https://zenodo.org/records/15529996, "
         "literature audit). The entry names no version; the files of the project are time series v1.2 "
         "(results/data_flow.json datasets[0].version)."
     ),
     supports=(
         "CAMELS-US time series (flow, basin-mean forcing, outlines): the files every model is trained and scored on "
         "(results/data_flow.json datasets[0]); cite it with newman2015 and addor2017."
     ))

_add("camels_attr",  # audit section 3
     short="Addor et al., 2017 (CAMELS attributes)",
     venue="NSF NCAR dataset (DataCite)",
     doi="10.5065/D6G73C3Q",
     ieee=(
         'N. Addor, A. J. Newman, N. Mizukami, and M. P. Clark, "Catchment attributes for large-sample studies," NSF '
         "National Center for Atmospheric Research, 2017, doi: 10.5065/D6G73C3Q."
     ),
     verified=(
         "The DOI is the one the NCAR CAMELS page names for the attributes (results/data_flow.json "
         "documentation_read[3]). DataCite record read on 28 Oct 2026: title, four creators, publisher and year as "
         "registered. The DOI resolves and lands on the time-series record 10.5065/D6MW2F4D (DataCite relation "
         "IsOriginalFormOf; 302 in the literature audit)."
     ),
     supports=(
         "CAMELS-US attribute tables v2.0, the static basin properties (results/data_flow.json datasets[0].version); "
         "cite it with addor2017."
     ))

_add("nlcd_annual",  # audit section 3
     short="USGS, 2024 (Annual NLCD)",
     venue="USGS data release (DataCite)",
     doi="10.5066/P94UXNTS",
     ieee=(
         'U.S. Geological Survey, "Annual National Land Cover Database (NLCD) Collection 1 Products," version used '
         "not recorded, 2024. [Dataset]. doi: 10.5066/P94UXNTS."
     ),
     verified=(
         "DataCite record read on 28 Oct 2026: creator United States Geological Survey (written U.S. Geological "
         "Survey in the entry, as the publisher field has it), year 2024; doi.org resolves (302 to ScienceBase item "
         "655ceb8ad34ee4b6e05cc51a, literature audit, and again on 28 Oct 2026). The title in the record now ends "
         "with '(ver. 1.2, June 2026)'; the collection number of the 2011 layer read on 3 Oct 2026 is not recorded "
         "in the project (results/data_flow.json, the list of facts it could not establish, dataset nlcd; the "
         "metadata the project read is Collection 1.0, documentation_read[14]), so the entry names no version. It "
         "replaces homer2015, which is the older single-year NLCD 2011 and not the product used."
     ),
     supports=(
         "Annual NLCD land cover, 2011 layer: the map and the check of the CAMELS forest fraction; no model reads it "
         "(results/basin_gis.json sources.land_cover; results/data_flow.json datasets[8])."
     ))

_add("aorc",  # audit section 3
     short="NOAA AORC v1.1",
     venue="AWS Open Data registry (NOAA OWP)",
     doi="",
     ieee=(
         'NOAA National Weather Service, Office of Water Prediction, "NOAA Analysis of Record for Calibration (AORC) '
         'Dataset," version 1.1, Registry of Open Data on AWS. [Online]. Available: '
         "https://registry.opendata.aws/noaa-nws-aorc/ (accessed Oct. 8, 2026)."
     ),
     verified=(
         "The registry title is the one recorded in results/data_flow.json documentation_read[10] (page read on 8 "
         "Oct 2026); the version is datasets[1].version. Both links answered HTTP 200 in the literature audit on 28 "
         "Oct 2026 and again when this file was written. No DOI is recorded for the dataset. The data were read from "
         "the AWS bucket on 7 Oct 2026 (results/data_flow.json all_information.download_dates.by_product.aorc). The "
         "journal reference is fall2023."
     ),
     supports=(
         "Data access record of the AORC v1.1 hourly rain (E1b, and the perfect-rain bound); the journal reference "
         "is fall2023."
     ))

_add("gefsv12",  # audit section 3
     short="NOAA GEFSv12 reforecast",
     venue="AWS Open Data registry (NOAA NCEP)",
     doi="",
     ieee=(
         'NOAA National Centers for Environmental Prediction, "NOAA Global Ensemble Forecast System (GEFS) '
         'Re-forecast," GEFS version 12, Registry of Open Data on AWS. [Online]. Available: '
         "https://registry.opendata.aws/noaa-gefs-reforecast/ (accessed Oct. 8, 2026)."
     ),
     verified=(
         "The registry title is the one recorded in results/data_flow.json documentation_read[11] (page read on 8 "
         "Oct 2026); the version and the control member c00 are datasets[2].version. The link answered HTTP 200 in "
         "the literature audit on 28 Oct 2026 and again when this file was written. No DOI is recorded for the "
         "dataset. The data were read on 7 Oct 2026 (results/data_flow.json "
         "all_information.download_dates.by_product.gefs). The journal reference is guan2022."
     ),
     supports=(
         "Data access record of the GEFSv12 reforecast, control member only (E2); the journal reference is guan2022."
     ))

_add("stage4",  # audit section 3
     short="NCEP Stage IV",
     venue="Iowa Environmental Mesonet archive (NCEP)",
     doi="",
     ieee=(
         'NOAA National Centers for Environmental Prediction, "NCEP Stage IV hourly radar and gauge precipitation '
         'analysis," Iowa Environmental Mesonet archive. [Online]. Available: '
         "https://mesonet.agron.iastate.edu/archive/ (accessed Oct. 7–8, 2026)."
     ),
     verified=(
         "The link answered HTTP 200 in the literature audit on 28 Oct 2026 and again when this file was written. "
         "The product name is the one report_common.py holds; the hours were fetched on 7-8 Oct 2026 "
         "(results/data_flow.json all_information.download_dates.by_product.stage4). The original product "
         "description, Lin and Mitchell (2005), has no DOI and is not listed. Whether the Iowa copy holds first "
         "postings or later quality-controlled versions is not established (results/data_flow.json, the list of "
         "facts it could not establish, dataset stage4). The journal reference is nelson2016."
     ),
     supports=(
         "Data access record of NCEP Stage IV hourly rain (the real-time check of the saved networks); the journal "
         "reference is nelson2016."
     ))


# ---------------------------------------------------------------------------------------------------------------------------
# Added 30 Oct 2026 by the report fixer after the three audits (audit_report_references_2026-10-30.json, missing citations;
# audit_report_quality_2026-10-30.json should-fix 21-23; audit_report_compliance_2026-10-30.json P13). Every DOI below was read
# live from Crossref or DataCite on 30 Oct 2026 (title, venue, volume, issue, pages, year, authors); the links without a DOI
# answered HTTP 200 the same day. As for every entry: verified bibliographic data, not a team member's reading.
# ---------------------------------------------------------------------------------------------------------------------------
_add("gauch2021",
     short="Gauch et al., 2021",
     venue="Hydrol. Earth Syst. Sci. 25(4)",
     doi="10.5194/hess-25-2045-2021",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=('M. Gauch, F. Kratzert, D. Klotz, G. Nearing, J. Lin, and S. Hochreiter, "Rainfall–runoff prediction at multiple '
           'timescales with a single Long Short-Term Memory network," Hydrol. Earth Syst. Sci., vol. 25, no. 4, pp. 2045–2062, '
           "2021, doi: 10.5194/hess-25-2045-2021."),
     verified="Crossref record read live on 30 Oct 2026: six authors, title, journal, volume 25, issue 4, pages 2045-2062, year 2021 match; the abstract was read from the same record.",
     supports="Table T4: one LSTM predicting daily and hourly flow, processing recent inputs at a finer timescale (abstract); the closest published design to the within-day inputs of E4b and E4e.")

_add("hunt2022",
     short="Hunt et al., 2022",
     venue="Hydrol. Earth Syst. Sci. 26(21)",
     doi="10.5194/hess-26-5449-2022",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=('K. M. R. Hunt, G. R. Matthews, F. Pappenberger, and C. Prudhomme, "Using a long short-term memory (LSTM) neural '
           'network to boost river streamflow forecasts over the western United States," Hydrol. Earth Syst. Sci., vol. 26, '
           "no. 21, pp. 5449–5472, 2022, doi: 10.5194/hess-26-5449-2022."),
     verified="Crossref record read live on 30 Oct 2026: four authors, title, journal, volume 26, issue 21, pages 5449-5472, year 2022 match; the abstract was read from the same record.",
     supports="Table T4: an LSTM fed weather-forecast variables beat raw and bias-corrected GloFAS forecasts at western US gauges (abstract); a forecast setting from another research group.")

_add("alfieri2013",
     short="Alfieri et al., 2013",
     venue="Hydrol. Earth Syst. Sci. 17(3)",
     doi="10.5194/hess-17-1161-2013",
     italic=("Hydrol. Earth Syst. Sci.",),
     ieee=('L. Alfieri et al., "GloFAS – global ensemble streamflow forecasting and flood early warning," Hydrol. Earth Syst. '
           "Sci., vol. 17, no. 3, pp. 1161–1175, 2013, doi: 10.5194/hess-17-1161-2013."),
     verified="Crossref record read live on 30 Oct 2026: seven authors (first author + et al.), title, journal, volume 17, issue 3, pages 1161-1175, year 2013 match; the abstract was read from the same record.",
     supports="Section 2.2: operational early warning compares ensemble streamflow forecasts with climatological thresholds (abstract).")

_add("maurer2002",
     short="Maurer et al., 2002",
     venue="J. Climate 15(22)",
     doi="10.1175/1520-0442(2002)015<3237:ALTHBD>2.0.CO;2",
     italic=("J. Climate",),
     ieee=('E. P. Maurer, A. W. Wood, J. C. Adam, D. P. Lettenmaier, and B. Nijssen, "A long-term hydrologically based dataset '
           'of land surface fluxes and states for the conterminous United States," J. Climate, vol. 15, no. 22, pp. 3237–3251, '
           "2002, doi: 10.1175/1520-0442(2002)015<3237:ALTHBD>2.0.CO;2."),
     verified="Crossref record read live on 30 Oct 2026: five authors, title, journal, volume 15, issue 22, pages 3237-3251, year 2002 match.",
     supports="The Maurer forcing of CAMELS (Table T5) and the Daymet-Maurer lag reported in kratzert2021.")

_add("gelaro2017",
     short="Gelaro et al., 2017",
     venue="J. Climate 30(14)",
     doi="10.1175/JCLI-D-16-0758.1",
     italic=("J. Climate",),
     ieee=('R. Gelaro et al., "The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2)," '
           "J. Climate, vol. 30, no. 14, pp. 5419–5454, 2017, doi: 10.1175/JCLI-D-16-0758.1."),
     verified="Crossref record read live on 30 Oct 2026: 31 authors (first author + et al.), title, journal, volume 30, issue 14, pages 5419-5454, year 2017 match.",
     supports="MERRA-2 soil wetness and precipitable water read through NASA POWER (satellite-input experiment, Table T5).")

_add("imerg",
     short="Huffman et al., 2023 (IMERG)",
     venue="NASA GES DISC dataset (DataCite)",
     doi="10.5067/GPM/IMERGDF/DAY/07",
     ieee=('G. J. Huffman et al., "GPM IMERG Final Precipitation L3 1 day 0.1 degree x 0.1 degree V07," NASA Goddard Earth '
           "Sciences Data and Information Services Center, 2023. [Dataset]. doi: 10.5067/GPM/IMERGDF/DAY/07."),
     verified="DataCite record read live on 30 Oct 2026: creators Huffman, Stocker, Bolvin, Nelkin, Tan and others, publisher NASA GES DISC, 2023, title match. The IMERG release that NASA POWER served to the project is not recorded; this is the current daily record.",
     supports="IMERG satellite rain read through NASA POWER (satellite-input experiment, day-offset check; Table T5).")

_add("power",
     short="NASA POWER",
     venue="NASA Langley Research Center (web service)",
     doi="",
     ieee=('NASA Langley Research Center, "Prediction Of Worldwide Energy Resources (POWER) project data access." [Online]. '
           "Available: https://power.larc.nasa.gov/ (accessed Oct. 30, 2026)."),
     verified="Link answered HTTP 200 on 30 Oct 2026; the service through which MERRA-2 and IMERG values were read (results/data_flow.json datasets: MERRA-2 (POWER), IMERG (POWER)).",
     supports="The access service of the MERRA-2 and IMERG values (Table T5).")

_add("openmeteo",
     short="Zippenfenig, 2024 (Open-Meteo)",
     venue="Zenodo software record (DataCite)",
     doi="10.5281/zenodo.7970649",
     ieee='P. Zippenfenig, "Open-Meteo.com Weather API," Zenodo, 2024. [Software]. doi: 10.5281/zenodo.7970649.',
     verified="DataCite record read live on 30 Oct 2026: creator Zippenfenig, Patrick; publisher Zenodo; 2024; licence CC BY 4.0. The historical-weather page answered HTTP 200 the same day.",
     supports="The service through which hourly ERA5 rain was read (E1, Table T5); its API data licence is CC BY 4.0 (open-meteo.com/en/licence, read 30 Oct 2026).")

_add("usgs_nwis",
     short="USGS Water Data",
     venue="USGS data release (DataCite)",
     doi="10.5066/F7P55KJN",
     ieee='U.S. Geological Survey, "USGS Water Data for the Nation," U.S. Geological Survey. [Dataset]. doi: 10.5066/F7P55KJN.',
     verified="DataCite record read live on 30 Oct 2026: creator and publisher United States Geological Survey, title match, rights CC0 1.0.",
     supports="The 15-minute flow record reduced to hours for E4e and the clock check of the daily flow (Table T5).")

_add("breiman2001",
     short="Breiman, 2001",
     venue="Mach. Learn. 45(1)",
     doi="10.1023/A:1010933404324",
     italic=("Mach. Learn.",),
     ieee='L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.',
     verified="Crossref record read live on 30 Oct 2026: author, title, journal, volume 45, issue 1, pages 5-32, year 2001 match.",
     supports="The random-forest baseline (Section 5.2).")

_add("hoerl1970",
     short="Hoerl and Kennard, 1970",
     venue="Technometrics 12(1)",
     doi="10.1080/00401706.1970.10488634",
     italic=("Technometrics",),
     ieee=('A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," Technometrics, '
           "vol. 12, no. 1, pp. 55–67, 1970, doi: 10.1080/00401706.1970.10488634."),
     verified="Crossref record read live on 30 Oct 2026: two authors, title, journal, volume 12, issue 1, pages 55-67, year 1970 match.",
     supports="The ridge-regression baseline (Section 5.2).")

_add("kingma2015",
     short="Kingma and Ba, 2015",
     venue="ICLR 2015 (arXiv:1412.6980)",
     doi="",
     italic=("Proc. 3rd Int. Conf. Learn. Represent. (ICLR)",),
     ieee=('D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. 3rd Int. Conf. Learn. Represent. '
           "(ICLR), San Diego, CA, USA, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980"),
     verified="arXiv record 1412.6980 read on 30 Oct 2026 (title; comment 'Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015'); link HTTP 200.",
     supports="The Adam optimiser of the LSTM trainer (Section 5.2).")

_add("paszke2019",
     short="Paszke et al., 2019",
     venue="NeurIPS 2019 (arXiv:1912.01703)",
     doi="",
     italic=("Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)",),
     ieee=('A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Proc. Adv. Neural '
           "Inf. Process. Syst. (NeurIPS), vol. 32, 2019. [Online]. Available: https://arxiv.org/abs/1912.01703"),
     verified="arXiv record 1912.01703 read on 30 Oct 2026 (title; comment 'NeurIPS 2019'); link HTTP 200.",
     supports="The deep-learning library of the LSTM and Transformer (Section 4.3; torch pinned in requirements.txt).")

_add("pedregosa2011",
     short="Pedregosa et al., 2011",
     venue="J. Mach. Learn. Res. 12",
     doi="",
     italic=("J. Mach. Learn. Res.",),
     ieee=('F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, '
           "2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html"),
     verified="JMLR page answered HTTP 200 on 30 Oct 2026 (volume 12, 2011).",
     supports="The library of the ridge and random-forest baselines (Section 4.3).")

_add("wilcoxon1945",
     short="Wilcoxon, 1945",
     venue="Biometrics Bull. 1(6)",
     doi="10.2307/3001968",
     italic=("Biometrics Bull.",),
     ieee='F. Wilcoxon, "Individual comparisons by ranking methods," Biometrics Bull., vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.',
     verified="Crossref record read live on 30 Oct 2026: author, title, journal, volume 1, issue 6, first page 80, year 1945 match (Crossref gives the first page only).",
     supports="The Wilcoxon signed-rank test of the seed rule (Section 6.3).")

_add("mcnemar1947",
     short="McNemar, 1947",
     venue="Psychometrika 12(2)",
     doi="10.1007/BF02295996",
     italic=("Psychometrika",),
     ieee=('Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," '
           "Psychometrika, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996."),
     verified="Crossref record read live on 30 Oct 2026: author, title, journal, volume 12, issue 2, pages 153-157, year 1947 match.",
     supports="The exact McNemar test on paired onsets (E3, Section 6.3).")



# ---------------------------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------------------------
def _need(key):
    if key not in REFERENCES:
        near = difflib.get_close_matches(str(key), list(REFERENCES), n=3, cutoff=0.5)
        hint = ("; did you mean " + ", ".join(near) + "?") if near else ""
        raise KeyError("unknown reference key %r%s" % (key, hint))


def format_ieee(key, n):
    """One reference as '[n] <entry>' (plain text; runs() gives the same text with the journal name marked for italics)."""
    _need(key)
    return "[%s] %s" % (n, REFERENCES[key]["ieee"])


def numbering(keys_in_order_of_first_appearance):
    """OrderedDict key -> n (1, 2, 3 ...) in the order the keys first appear; a repeated key keeps its first number."""
    out = OrderedDict()
    for key in keys_in_order_of_first_appearance:
        _need(key)
        if key not in out:
            out[key] = len(out) + 1
    return out


def cite_order(keys_in_order_of_first_appearance):
    """The numbered reference list: ['[1] <entry>', '[2] <entry>', ...] in the order the keys first appear (repeats are ignored).
    Pass every key that the finished text cites, in the order of its first citation; keys that are not cited stay out."""
    return [format_ieee(key, n) for key, n in numbering(keys_in_order_of_first_appearance).items()]


def marker(order, *keys):
    """The in-text citation for one or more keys, e.g. '[3]', '[1], [4]', or '[2]-[5]' (en dash) for three or more consecutive
    numbers. order: the dict from numbering() or the list of keys in order of first appearance."""
    nums = order if isinstance(order, dict) else numbering(order)
    ns = sorted(set(nums[key] for key in keys))
    parts = []
    i = 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append("[%d]–[%d]" % (ns[i], ns[j]))
        else:
            parts.extend("[%d]" % ns[m] for m in range(i, j + 1))
        i = j + 1
    return ", ".join(parts)


def runs(key, n=None):
    """The entry as [(text, italic), ...] for python-docx: '[n] ' first when n is given, the journal or proceedings name italic."""
    _need(key)
    entry = REFERENCES[key]["ieee"]
    spans = sorted((entry.index(s), entry.index(s) + len(s)) for s in ITALIC.get(key, ()))
    out = [("[%s] " % n, False)] if n is not None else []
    pos = 0
    for a, b in spans:
        if a > pos:
            out.append((entry[pos:a], False))
        out.append((entry[a:b], True))
        pos = b
    if pos < len(entry):
        out.append((entry[pos:], False))
    return out


def check():
    """Self-test of the data. Returns a list of problem strings; an empty list means the module is consistent."""
    bad = []
    if list(LABELS) != list(REFERENCES):
        bad.append("LABELS and REFERENCES differ in keys or order")
    seen_doi, seen_short = {}, {}
    for key, e in REFERENCES.items():
        if set(e) != {"ieee", "doi", "verified", "supports"}:
            bad.append("%s: wrong fields %s" % (key, sorted(e)))
            continue
        ieee, doi = e["ieee"], e["doi"]
        for f in ("ieee", "verified", "supports"):
            if not isinstance(e[f], str) or not e[f].strip():
                bad.append("%s: empty %s" % (key, f))
            elif e[f] != e[f].strip() or "  " in e[f]:
                bad.append("%s: stray space in %s" % (key, f))
            elif "unverified" in e[f].lower():
                bad.append("%s: the word 'unverified' is in %s (it belongs in code comments only)" % (key, f))
        if "*" in ieee:
            bad.append("%s: asterisk in ieee (the italic text is kept in ITALIC)" % key)
        if ieee.count('"') != 2:
            bad.append("%s: the title must sit in exactly one pair of double quotes" % key)
        if doi:
            if not doi.startswith("10.") or " " in doi:
                bad.append("%s: odd DOI %r" % (key, doi))
            if not ieee.endswith("doi: %s." % doi):
                bad.append("%s: ieee does not end with 'doi: <doi>.'" % key)
            if doi.lower() in seen_doi:
                bad.append("%s: DOI also used by %s" % (key, seen_doi[doi.lower()]))
            seen_doi[doi.lower()] = key
        elif "Available: http" not in ieee:
            bad.append("%s: no DOI and no 'Available: http' link" % key)
        for s in ITALIC.get(key, ()):
            if ieee.count(s) != 1:
                bad.append("%s: italic text %r occurs %d times in the entry" % (key, s, ieee.count(s)))
        head = ieee.split(', "')[0]
        if not head.endswith(" et al."):
            n_auth = len(head.replace(", and ", ", ").replace(" and ", ", ").split(", "))
            if n_auth > 6:
                bad.append("%s: %d authors listed in full (IEEE: first author + et al. from seven)" % (key, n_auth))
        short = LABELS.get(key, {}).get("short")
        if not short or not LABELS[key].get("venue"):
            bad.append("%s: missing short label or venue" % key)
        elif short in seen_short:
            bad.append("%s: short label also used by %s" % (key, seen_short[short]))
        else:
            seen_short[short] = key
    for key in ITALIC:
        if key not in REFERENCES:
            bad.append("ITALIC names unknown key %s" % key)
    for key in sorted(REPORT_COMMON_KEYS):
        if key not in REFERENCES:
            bad.append("REPORT_COMMON_KEYS names %s, which is not in REFERENCES" % key)
    return bad


def keys_markdown():
    """The key table of REFERENCES_KEYS.md: key, short label, venue, the claim it supports."""
    rows = ["| key | label | venue | supports (claim in this project) |", "|---|---|---|---|"]
    for key, e in REFERENCES.items():
        lab = LABELS[key]
        rows.append("| `%s` | %s | %s | %s |" % (key, lab["short"], lab["venue"], e["supports"].replace("|", "/")))
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        for line in cite_order(list(REFERENCES)):
            print(line)
    elif "--keys" in sys.argv:
        print(keys_markdown())
    else:
        problems = check()
        n_doi = sum(1 for e in REFERENCES.values() if e["doi"])
        print("references.py: %d references (%d with a DOI, %d without)" % (len(REFERENCES), n_doi, len(REFERENCES) - n_doi))
        print("self-check:", "ok" if not problems else "%d problem(s)" % len(problems))
        for p in problems:
            print("  -", p)
        sys.exit(1 if problems else 0)
