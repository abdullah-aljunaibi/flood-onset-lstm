# Data: where every input comes from

All inputs are public. The repository ships none of the raw downloads. The reduced copies the scripts read are in
`flood-onset-lstm_data_caches.zip` on the Releases page; CAMELS-US is downloaded from its own page. Each source keeps
its own licence and citation. Please cite the original data, not this repository, when you use them.

## 1. Download first (needed for every script)

| Data | Where | Put it here |
|---|---|---|
| CAMELS-US v1.2 time series (Daymet, NLDAS and Maurer forcing; observed daily flow), basin outlines, catchment attributes v2.0 | https://ral.ucar.edu/solutions/products/camels (DOIs 10.5065/D6MW2F4D and 10.5065/D6G73C3Q) | `data/camels_us/`: unzip `basin_timeseries_v1p2_metForcing_obsFlow.zip` (gives `basin_dataset_public_v1p2/`) and `basin_set_full_res.zip` (gives `basin_set_full_res/`), and put the `camels_*.txt` attribute files beside them |
| The reduced caches below | Releases: `flood-onset-lstm_data_caches.zip` | unzip at the repository root (it creates `data/...`) |
| The stored forecasts | Releases: `flood-onset-lstm_preds.zip` | unzip at the repository root (it creates `results/preds/`) |

`code/01_data_audit.py` picks the 12 study rivers from CAMELS by a fixed rule. The first run of `code/flood.py`
parses the CAMELS files once into `data/derived/`.

## 2. The reduced caches (in the data-caches zip) and how to make them again

| Folder | What it holds | Source | Script that makes it |
|---|---|---|---|
| `data/aorc/` | hourly basin-mean AORC v1.1 rain (1 km grid), 1980-2014, 12 basins | NOAA/NWS AORC on AWS Open Data, https://registry.opendata.aws/noaa-nws-aorc/ | `code/fetch_aorc.py` (about 4.5 GB read; needs the `zstd` tool) |
| `data/hourly/` | hourly ERA5 rain at each basin (and MERRA-2) | ERA5 (Copernicus) via the Open-Meteo archive API, https://open-meteo.com/en/docs/historical-weather-api; NASA MERRA-2 | `code/check_day_windows.py`, `code/fetch_satellite.py` |
| `data/gefs/` | GEFSv12 reforecast control-member rain, reduced to the basins | NOAA GEFSv12 reforecast on AWS Open Data, https://registry.opendata.aws/noaa-gefs-reforecast/ | `code/fetch_gefs.py`, `code/gefs_to_basins.py` (byte ranges of the GRIB files; a GRIB decoder) |
| `data/hourly_flow/` | USGS 15-minute discharge reduced to hours, 12 gauges | USGS Water Services / the USGS water-data API | `code/fetch_usgs_iv.py` (optional key: `USGS_API_KEY`) |
| `data/aorc/stage4_issue_days.csv` and related | Stage IV hourly radar + gauge rain on the issue days of the test-year onsets | NOAA/NCEP Stage IV, Iowa Environmental Mesonet archive, https://mesonet.agron.iastate.edu/ | `code/check_stage4_realtime.py` (set `STAGE4_NO_FETCH=1` to run offline) |
| `data/satellite/` | IMERG and MERRA-2 daily series per basin (rain, soil wetness, water vapour) | NASA POWER, https://power.larc.nasa.gov/ | `code/fetch_satellite.py` |
| `data/dem/` | elevation tiles over the basins | SRTM via the AWS terrain tiles (https://github.com/tilezen/joerd) | `code/07_basin_maps.py` |
| `data/landcover/` | Annual NLCD land-cover grids over the basins | USGS / MRLC land-cover web service | `code/07_basin_maps.py` |
| `data/hurdat/` | Atlantic storm tracks | NOAA NHC HURDAT2, https://www.nhc.noaa.gov/data/hurdat/ | `code/check_2004_floods.py`, `code/06_flood_analysis.py` |
| `data/derived/` | daily tables of the basins, parsed once from CAMELS | CAMELS-US | `code/flood.py` (made on first use) |
| `data/basemap/` | US state outlines (drawing only) | public US state boundaries | used by `code/build_overview.py` |

`results/data_flow.json` (`code/19_data_flow.py`) records, for every dataset, where it came from, its time span, what
was read and how it enters the model; `results/aorc_data.json` holds the checksums of the AORC download. NOAA announced
on 2026-04-02 that the AORC Zarr files are being regenerated, so a new download may differ slightly from these caches.

## 3. Citations

- Newman, A. J. et al. (2015). Development of a large-sample watershed-scale hydrometeorological data set for the
  contiguous USA. *HESS* 19, 209-223.
- Addor, N. et al. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies.
  *HESS* 21, 5293-5313.
- Thornton, P. E. et al. Daymet: daily surface weather data on a 1-km grid for North America (ORNL DAAC).
- NOAA/NWS, Analysis of Record for Calibration (AORC) v1.1.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS* 146, 1999-2049. Contains modified Copernicus Climate
  Change Service information; Open-Meteo data are CC BY 4.0.
- Hamill, T. M. et al. (2022). The reanalysis for the Global Ensemble Forecast System, version 12. *MWR* 150, 59-79.
- U.S. Geological Survey, National Water Information System.
- Landsea, C. W. and Franklin, J. L. (2013). Atlantic hurricane database uncertainty and presentation of a new
  database format. *MWR* 141, 3576-3592.
