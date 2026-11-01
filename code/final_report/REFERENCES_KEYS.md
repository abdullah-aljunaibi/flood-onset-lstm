# REFERENCES_KEYS: citation keys for the Project-Flood final report

Written 28 Oct 2026 for the section writers. The entries and the functions are in `code/final_report/references.py` (29 entries, 25 with a DOI). This file is the quick list; if the two ever differ, `references.py` wins. `python -X utf8 references.py --keys` prints the key table, `--list` the whole list, and with no option it runs the self-check.

## How to cite

- Cite by key (`newman2015`), never by typing a reference. An unknown key raises KeyError and suggests the nearest keys.
- IEEE numbers follow the order of first citation in the finished text. The builder collects the keys while it assembles the document and numbers them last: `cite_order(keys)` returns `["[1] ...", "[2] ...", ...]`; `format_ieee(key, n)` gives one line; `numbering(keys)` maps key to number; `marker(numbering, *keys)` gives the in-text `[n]` (`[1], [4]`; three or more consecutive numbers become `[2]–[5]`).
- The journal or proceedings name is italic in IEEE: `runs(key, n)` returns `(text, italic)` pieces for python-docx. `REFERENCES[key]["ieee"]` is plain text without the `[n]`.
- Template rule: every in-text citation must match a bibliography entry, and every entry must be cited. Pass `cite_order` only the keys the text really cites.
- Keys of `code/report_common.py` REFERENCES keep their name where the same work exists (the "old" keys below); the "new" keys follow the same style.

## What "verified" means, and what it does not

- The 22 journal and proceedings DOIs were read again against the live Crossref record on 28 Oct 2026, after the literature audit of the same day: title, journal, volume, issue, pages or article number, year, author initials and author count agree with the entry. The 3 data DOIs were read in DataCite. The links of `aorc`, `gefsv12`, `stage4` and `vaswani2017` answered HTTP 200.
- That is a bibliographic check. No team member has read these papers: `results/published_scores.json` (`checked_by`) says "No team member has read these papers yet", and `results/sources_checked.json` (`notes`) asks the team to read the passages itself. Write "read by an AI literature agent", or have a member read the papers of table T4 before submission. Never write that the team reviewed them.
- Published scores are quoted only from `results/published_scores.json` (and `results/published_context.json`), set beside the upper-bound rows of this project only, and worded "of the same size" or "of the same order as", never "reproduced", "matched", "matches" or "we expect" (`how_to_word_it` in both files).

## Keys (29)

| key | label | venue | supports (claim in this project) |
|---|---|---|---|
| `newman2015` | Newman et al., 2015 | Hydrol. Earth Syst. Sci. 19(1) | The data: CAMELS-US daily flow and basin-mean Daymet weather (results/data_flow.json datasets[0].sources and documentation_read[0]); section 4.1 data inventory; cite it together with addor2017, as the CAMELS readme asks. |
| `addor2017` | Addor et al., 2017 | Hydrol. Earth Syst. Sci. 21(10) | The static basin properties used as model inputs (results/data_flow.json all_information.static_properties and datasets[0]); section 4.1; cite it together with newman2015. |
| `thornton2021` | Thornton et al., 2021 | Sci. Data 8(1) | Daymet V4 shifts the rain of stations that observe before noon back one day, a timing bias not examined in earlier versions (results/sources_checked.json sources[0]); the CAMELS files predate V4, so cite it for the clock argument, not as the CAMELS product; sections 2.3 and 4.2. |
| `kratzert2021` | Kratzert et al., 2021 | Hydrol. Earth Syst. Sci. 25(5) | A one-day lag between CAMELS rain products (Daymet and Maurer) and the gain from combining forcing products (results/sources_checked.json sources[1]; results/published_context.json sources.kratzert2021); supports the clock diagnosis and the NLDAS experiment (script 10). |
| `hochreiter1997` | Hochreiter and Schmidhuber, 1997 | Neural Comput. 9(8) | The model class: one small LSTM shared by all rivers (section 5.2). |
| `kratzert2018` | Kratzert et al., 2018 | Hydrol. Earth Syst. Sci. 22(11) | First LSTM rainfall-runoff benchmark on CAMELS with Daymet forcing, including the South Atlantic-Gulf region that holds the study basins (results/published_scores.json sources.kratzert2018.regions); section 2.2 and table T4. |
| `kratzert2019` | Kratzert et al., 2019a | Hydrol. Earth Syst. Sci. 23(12) | The design choice of one LSTM shared by all basins with static attributes, and the NSE* loss; published scores are set beside the upper-bound rows of this project only (results/published_scores.json sources.kratzert2019 and rows_of_this_project_to_set_beside_them); sections 2.2, 5.2, 8.1. |
| `kratzert2019pub` | Kratzert et al., 2019b | Water Resour. Res. 55(12) | Prediction in basins that supplied no training data, set beside the rivers-never-seen test of this project, which is NOT comparable in level (results/published_scores.json rows_of_this_project_to_set_beside_them.basin_never_seen; results/lstm_lobo.json; results/more_basins.json outcome_full); sections 5.4 and 8.1. |
| `feng2020` | Feng et al., 2020 | Water Resour. Res. 56(9) | The river gauge as an input (data integration of lagged flow); the published value is set beside the perfect-weather upper bound of this project, because that forecast still reads the weather of the target day (results/published_scores.json sources.feng2020 and how_to_word_it); sections 2.2 and 8.1. |
| `kratzert2024` | Kratzert et al., 2024 | Hydrol. Earth Syst. Sci. 28(17) | Why one network is trained on many rivers rather than one per river (scripts 03 and 04; results/more_basins.json outcome_full); sections 2.2, 5.3, 8.1. |
| `nearing2024` | Nearing et al., 2024 | Nature 627(8004) | State of the art in AI flood warning, and the frame for the warning task of this project (onset warned, false warning, the GO rule); sections 2.1 and 2.2. |
| `nash1970` | Nash and Sutcliffe, 1970 | J. Hydrol. 10(3) | The NSE metric (section 6.3; every nse_median key in results/*.json). |
| `gupta2009` | Gupta et al., 2009 | J. Hydrol. 377(1-2) | The KGE reported beside NSE (code/04_lstm_all.py; results/lstm_all.json sets.*.test.summary.l1.kge_median); section 6.3. |
| `fall2023` | Fall et al., 2023 | JAWRA J. Amer. Water Resour. Assoc. 59(6) | The AORC v1.1 hourly rain of E1b, which the report calls 'an analysis made after the fact' (results/aorc_rain_to_midnight.json outcome_full); sections 4.1 and 6.2; the data access record is aorc. |
| `nelson2016` | Nelson et al., 2016 | Weather Forecast. 31(2) | The real-time Stage IV check, 'a check on the test years, not a service' (results/stage4_realtime.json models.wqa.variants.stage4.share_of_the_onsets); sections 6.2 and 7.3; the data access record is stage4. |
| `guan2022` | Guan et al., 2022 | Mon. Weather Rev. 150(3) | E2: the archived GEFSv12 control-member rain forecast as an input (results/forecast_input.json e2_success.by_lead and outcome_full); sections 4.1 and 6.2; the data access record is gefsv12. |
| `hersbach2020` | Hersbach et al., 2020 | Q. J. R. Meteorol. Soc. 146(730) | E1: hourly ERA5 rain, read through the Open-Meteo archive API (results/data_flow.json datasets[4]), as an input with rain to local midnight (results/rain_to_midnight.json outcome_full). |
| `xia2012` | Xia et al., 2012 | J. Geophys. Res. Atmos. 117(D3) | NLDAS-2 rain, the CAMELS forcing product whose day ends in the afternoon, about 17:30 local standard time (measured median 17:40: results/day_windows.json day_ends.nldas); the NLDAS experiment, script 10. |
| `vaswani2017` | Vaswani et al., 2017 | Adv. Neural Inf. Process. Syst. 30 (NIPS 2017) | Only if the Transformer comparison stays (script 09; results/model_comparison.json): the Transformer model class. That comparison used its own two-seed LSTM with shorter sequences (results/model_comparison.json seeds and settings.lstm; critic C9), so its LSTM score differs from the headline LSTM. |
| `beck2019` | Beck et al., 2019 | Hydrol. Earth Syst. Sci. 23(1) | Its abstract names gauge reporting times as a factor in how well precipitation products match Stage IV, which supports the clock argument (section 2.3). |
| `farr2007` | Farr et al., 2007 | Rev. Geophys. 45(2) | SRTM elevation behind the basin maps and the check of the CAMELS elevation (results/basin_gis.json); no model reads it. |
| `landsea2013` | Landsea and Franklin, 2013 | Mon. Weather Rev. 141(10) | HURDAT2 storm matching in the flood-event analysis (results/flood_analysis.json events.hurdat; code/06_flood_analysis.py). |
| `frame2022` | Frame et al., 2022 | Hydrol. Earth Syst. Sci. 26(13) | Table T4, row 8 (audit section 5): LSTM skill on extreme events, in a simulation setting with no warning-timing score; discussion in section 2.2. |
| `camels_ts` | Newman, 2014 (CAMELS data) | NSF NCAR dataset (DataCite) | CAMELS-US time series (flow, basin-mean forcing, outlines): the files every model is trained and scored on (results/data_flow.json datasets[0]); cite it with newman2015 and addor2017. |
| `camels_attr` | Addor et al., 2017 (CAMELS attributes) | NSF NCAR dataset (DataCite) | CAMELS-US attribute tables v2.0, the static basin properties (results/data_flow.json datasets[0].version); cite it with addor2017. |
| `nlcd_annual` | USGS, 2024 (Annual NLCD) | USGS data release (DataCite) | Annual NLCD land cover, 2011 layer: the map and the check of the CAMELS forest fraction; no model reads it (results/basin_gis.json sources.land_cover; results/data_flow.json datasets[8]). |
| `aorc` | NOAA AORC v1.1 | AWS Open Data registry (NOAA OWP) | Data access record of the AORC v1.1 hourly rain (E1b, and the perfect-rain bound); the journal reference is fall2023. |
| `gefsv12` | NOAA GEFSv12 reforecast | AWS Open Data registry (NOAA NCEP) | Data access record of the GEFSv12 reforecast, control member only (E2); the journal reference is guan2022. |
| `stage4` | NCEP Stage IV | Iowa Environmental Mesonet archive (NCEP) | Data access record of NCEP Stage IV hourly rain (the real-time check of the saved networks); the journal reference is nelson2016. |

Old keys (same work and same key as in `report_common.py`, 21): `newman2015`, `addor2017`, `thornton2021`, `kratzert2021`, `hochreiter1997`, `kratzert2018`, `kratzert2019`, `kratzert2019pub`, `feng2020`, `kratzert2024`, `nearing2024`, `nash1970`, `gupta2009`, `guan2022`, `vaswani2017`, `beck2019`, `farr2007`, `landsea2013`, `aorc`, `gefsv12`, `stage4`.
New keys (8): `fall2023`, `nelson2016`, `hersbach2020`, `xia2012`, `frame2022`, `camels_ts`, `camels_attr`, `nlcd_annual`.
Not carried over from `report_common.py` (4): `homer2015`, `bevis1992`, `gelaro2017`, `power`; see "Left out on purpose".

## Table T4 (literature comparison)

The 8 rows of audit section 5 are `kratzert2018`, `kratzert2019`, `kratzert2019pub`, `feng2020`, `kratzert2021`, `kratzert2024`, `nearing2024`, `frame2022`. The Citation column can take the label of each key (for example "Kratzert et al., 2019a") with its IEEE number in front, as the audit does.

## Notes on single entries

- `newman2015`: the title is given with its subtitle exactly as Crossref has it ("data set"); the proposal and the web deck use a shortened title.
- `thornton2021`: the Daymet V4 paper. The CAMELS files predate V4, so it supports the clock argument, not the CAMELS forcing. No Daymet data DOI is listed: the Daymet version inside CAMELS is not recorded, and ORNL DAAC has a separate DOI for each version. Cite `newman2015` and `addor2017` for the forcing.
- `camels_ts`, `camels_attr`: two DOIs; the second lands on the first. DataCite lists one creator for the time-series DOI. Cite them with `newman2015` and `addor2017`, as the CAMELS readme asks.
- `nlcd_annual`: no version in the entry (the collection number of the 2011 layer is not recorded). It replaces `homer2015`.
- `aorc`, `gefsv12`, `stage4`: no DOI; the entry carries the registry or archive link and the access date. Pair each with its journal paper (`fall2023`, `guan2022`, `nelson2016`).
- `fall2023`, `nelson2016`: the fixed wordings of HANDOFF.md section 7 apply to the sentence (AORC "an analysis made after the fact", with "hindsight at most about 3 points" beside every 51%; Stage IV "a check on the test years, not a service").
- `vaswani2017`: no DOI and no page numbers, on purpose. Cite it only if the Transformer comparison stays, and say that the comparison used its own two-seed LSTM with shorter sequences.
- `xia2012`, `farr2007`: Crossref registers an article number and no pages; the entries carry neither, and the DOI identifies the paper.
- `frame2022`: not in the keep-list of the order; added because row 8 of table T4 cites it. Drop it if T4 loses that row.

## Left out on purpose

- "Hamill et al. 2022 BAMS": does not exist as described. Hamill et al. 2022 is a Mon. Weather Rev. paper on the GEFSv12 reanalysis; the GEFSv12 reforecast paper is Guan et al. 2022 (`guan2022`).
- `homer2015`: its DOI 10.14358/PERS.81.5.345 returns 404 and Crossref does not find the paper; it is also the older single-year NLCD 2011, not the Annual NLCD the project read (`nlcd_annual`).
- `bevis1992`: not in the audit's verified list.
- `gelaro2017` (MERRA-2), `power` (NASA POWER) and the IMERG paper (audit O9, O8 are verified in audit section 2): outside the keep-list. Add them if the satellite-inputs row stays in table T5.
- Lin and Mitchell 2005 (the Stage IV description): no DOI; the PDF answers HTTP 200 (audit O13).
- The other optional papers of audit section 2 (O5, O6, O10 to O12, O14 to O20): verified, not needed by the plan.

## Facts that are not recorded, so they are not written anywhere

Page numbers of Vaswani et al.; the article number D03109 of Xia et al.; "RG2004" of Farr et al.; the DOI of the 2023 corrigendum of Frame et al.; the NLCD collection number of the 2011 layer; the Daymet version inside CAMELS.
