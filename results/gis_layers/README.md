# GIS layers of the 12 study basins (deliverable D3)

Written by `code/27_gis_layers.py` on 2026-11-01 14:28. 28 files (9.4 MB) plus this README. Rebuild, never hand-edit: `python -X utf8 code/27_gis_layers.py` from the project root (a Python with rasterio, shapely, pyproj, geopandas). The numbers below are the script's own; the full record with sha256 hashes is `results/gis_layers.json`.

Claude Code wrote and ran this script under the team lead's direction; **the team's own check of these layers is not done.**

An independent AI reviewer agent (Claude) recomputed these layers on 2026-10-30 (results/review_gis_layers.json); the team's own check of these layers is not done.

## What the layers are

- **Outlines** (`basins_12.gpkg` layer `outlines`, `basins_12.geojson`): the 12 CAMELS-US basin polygons taken unchanged from `data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp` (hru_id = gauge id), with 2 invalid geometries repaired by buffer(0) as in script 07 (Hyco Creek, NC, Charlie Creek, FL; areas unchanged). Attributes: `gauge_id`, `name` (CAMELS gauge name), `short` (the project's short name), `landscape` (the class of 01_data_audit.py), `area_km2` (planar area in EPSG:5070, equal-area), `area_geodesic_km2` (GRS80 ellipsoid), `camels_area_gages2_km2` and `camels_area_geospa_fabric_km2` (the CAMELS table), `outline_file_area_km2` (the AREA field of the shapefile), `elev_mean_m` / `elev_min_m` / `elev_max_m` (from the exported elevation layer, cells with centre inside the outline, mean weighted by cos(latitude)), `camels_elev_mean_m`, `outline_file_elev_mean_m`, `dominant_nlcd_class` and `dominant_nlcd_share` (the single NLCD class with the largest share), `dominant_land_cover` and `dominant_group_share` (the seven groups of results/basin_gis.json), `forest_share` (NLCD 41+42+43), `camels_frac_forest`, `camels_dom_land_cover` (MODIS IGBP, another map), the differences to CAMELS (`area_diff_gages2_pct`, `elev_diff_camels_m`, `forest_diff_camels_points`), cell counts and the file names of the two rasters.
- **Gauges** (`basins_12.gpkg` layer `gauges`, `gauges_12.geojson`): the 12 gauge points at the CAMELS `gauge_lat`/`gauge_lon` of `data/camels_us/camels_topo.txt`, as the table gives them (not rounded: results/data_audit.json holds the same values rounded to 0.001 degree, which would put the points 9 to 60 m away), with the elevation cell at the gauge, whether the point lies inside its outline (12 of 12 do) and its geodesic distance to the outline (at most 0.63 km). The elevation cell and the distance equal results/basin_gis.json (part of the check below).
- **Elevation** (`elevation_<gauge_id>.tif`, 12 files): the 1-arc-second cells of the SRTM-format tiles, on the tiles' own grid (cell centres on whole arc-seconds, north up), cut to the outline buffered by 15 arc-seconds (about 460 m north-south, 370-410 m east-west) so that the ridge lines around the basin stay visible; everything outside the buffer is nodata (-32768, also the tiles' void code). Whole metres above the EGM96 geoid, int16, LZW. The tiles are WGS 84 and the outlines NAD83; as in script 07 no datum shift was applied (about 1 m, far below the 30 m cell). Voids inside the outlines: 0 cells.
- **Land cover** (`landcover_<gauge_id>.tif`, 12 files): the Annual NLCD 2011 class codes of `data/landcover/annual_nlcd_2011_<gauge_id>.tif`, on the native 30 m grid in EPSG:5070, cut to the outline itself (cells whose centre lies inside; no buffer); outside = nodata 250. The NLCD colour table is kept, so QGIS shows the usual colours. Legend: `landcover_classes.csv` (15 classes, 15 of them present in the 12 basins).

## Files

| file | kind | layer / content | CRS | resolution | units | nodata | size | sha256 |
|---|---|---|---|---|---|---|---|---|
| `basins_12.gpkg` | GeoPackage (vector) | outlines (12 polygons), gauges (12 points) | EPSG:4269 geographic NAD83 (the source shapefile's CRS) | vector; the outline vertices follow 30 m raster-cell edges (results/basin_gis.json notes) | degrees; area_km2 in km2 (planar, EPSG:5070); elevations in m; shares 0-1 | none | 1.27 MB | `502f5f77ce3bc26f13f74e708e9ec7475318a89420a8a8ad9a92932aece9a8a0` |
| `basins_12.geojson` | GeoJSON (vector) | outlines (12 polygons) | EPSG:4326 WGS 84 (from EPSG:4269 by pyproj: NAD83 to WGS 84 (1) (with axis order normalized for visualization) = a null transformation, coordinates unchanged; 49 grid-shift operations are not installed) | vector, coordinates to 7 decimals (about 1 cm) | as basins_12.gpkg | none | 2.05 MB | `7262b0bf31c1ad29659918d7f207461760e13748004fb6702d5db1a5591767bd` |
| `gauges_12.geojson` | GeoJSON (vector) | gauges (12 points) | EPSG:4326 WGS 84 (from EPSG:4269 by pyproj: NAD83 to WGS 84 (1) (with axis order normalized for visualization) = a null transformation, coordinates unchanged; 49 grid-shift operations are not installed) | points, coordinates to 7 decimals | degrees; elev_at_gauge_m in m | none | 0.01 MB | `ceb31726605ac8fd911c22f04e681da23fbfd7d9daac5512e1405c7ab9fdb334` |
| `landcover_classes.csv` | CSV legend | - | - | - | class code, name, group, RGB colour (the colour table of the source files) | - | 0.00 MB | `9e4685c8c110ebf5b60a2051ae89fb0383ae833b076990cd6f38c7f38ed604e9` |
| `elevation_02077200.tif` | GeoTIFF int16, LZW, tiled | elevation, Hyco Creek, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (24.94 m east-west x 30.82 m north-south here); 509 x 775 cells | metres (whole metres) | -32768 | 0.13 MB | `0e8694001623ea44b824cc9e6256785598211bc1570ba07eeed07146ebfe6ccd` |
| `elevation_02108000.tif` | GeoTIFF int16, LZW, tiled | elevation, NE Cape Fear, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.35 m east-west x 30.82 m north-south here); 2365 x 1557 cells | metres (whole metres) | -32768 | 0.58 MB | `1457393063b892b039bcd7119e0259cc936449f3b2c722b5895436e5af331c76` |
| `elevation_02128000.tif` | GeoTIFF int16, LZW, tiled | elevation, Little River, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.19 m east-west x 30.82 m north-south here); 604 x 1141 cells | metres (whole metres) | -32768 | 0.23 MB | `0b0f30c11001234e38ac2775c8d5acf6a7f15e40eb505f058a0e16cd8d108d46` |
| `elevation_02137727.tif` | GeoTIFF int16, LZW, tiled | elevation, Catawba, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.16 m east-west x 30.82 m north-south here); 908 x 747 cells | metres (whole metres) | -32768 | 0.42 MB | `6bd06100672536f8a9faf2161e58dc18bc1482382458f95306501c5dfccbdf7b` |
| `elevation_02221525.tif` | GeoTIFF int16, LZW, tiled | elevation, Murder Creek, GA | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.85 m east-west x 30.81 m north-south here); 1110 x 1114 cells | metres (whole metres) | -32768 | 0.34 MB | `170cdce363806a6128799cd8bcabb9aabde5f8831cdc58a24ea9b0c30e500101` |
| `elevation_02296500.tif` | GeoTIFF int16, LZW, tiled | elevation, Charlie Creek, FL | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (27.46 m east-west x 30.78 m north-south here); 1273 x 1601 cells | metres (whole metres) | -32768 | 0.23 MB | `c78ef0cdbd3313a60269b14278cec2839054a7785bcf575fb8272e9e448d4ca2` |
| `elevation_02298608.tif` | GeoTIFF int16, LZW, tiled | elevation, Myakka, FL | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (27.47 m east-west x 30.78 m north-south here); 961 x 814 cells | metres (whole metres) | -32768 | 0.07 MB | `3f25e2acdf025210af971e51558b45bf087baa71b29f7e40158673565d2b0bb8` |
| `elevation_02349900.tif` | GeoTIFF int16, LZW, tiled | elevation, Turkey Creek, GA | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (26.18 m east-west x 30.8 m north-south here); 639 x 514 cells | metres (whole metres) | -32768 | 0.07 MB | `441253e080fb2d21426f68151f38df7c16b11f4b7710c4f8a99d4e6c639e7ed8` |
| `elevation_02472000.tif` | GeoTIFF int16, LZW, tiled | elevation, Leaf River, MS | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (26.24 m east-west x 30.8 m north-south here); 1806 x 2454 cells | metres (whole metres) | -32768 | 1.19 MB | `8bab7af377f6d14f79e9c6d5625dfdd8ce42176c0f9d0afb8bf64b047a1236bf` |
| `elevation_03439000.tif` | GeoTIFF int16, LZW, tiled | elevation, French Broad, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.29 m east-west x 30.82 m north-south here); 647 x 778 cells | metres (whole metres) | -32768 | 0.25 MB | `841f6fce850f9f7999ef15220aeaa595fcd7129fd0a17a6f8e20b26c958c0d03` |
| `elevation_03463300.tif` | GeoTIFF int16, LZW, tiled | elevation, South Toe, NC | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (25.12 m east-west x 30.82 m north-south here); 512 x 564 cells | metres (whole metres) | -32768 | 0.15 MB | `3b58df7e61f252c92e418fe4f97d95938b58f706ad5855fef16df41fc80beef1` |
| `elevation_03473000.tif` | GeoTIFF int16, LZW, tiled | elevation, SF Holston, VA | EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid | 1 arc-second (24.84 m east-west x 30.83 m north-south here); 2397 x 1293 cells | metres (whole metres) | -32768 | 0.91 MB | `929c302a8f0823e58ab2c3fafc153a326a745dacda516c1ba473bf671235ad6d` |
| `landcover_02077200.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Hyco Creek, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 378 x 772 cells | NLCD class code (landcover_classes.csv) | 250 | 0.05 MB | `593a497c7497b7800bf9292b7f57eb5461987518cf29be0852eb9b34ae2ca1e4` |
| `landcover_02108000.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, NE Cape Fear, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 2140 x 1552 cells | NLCD class code (landcover_classes.csv) | 250 | 0.33 MB | `26d5a18c90a8bc6add5e7955e885915023be0a19a2e9e99aa50e641a1ea94d8f` |
| `landcover_02128000.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Little River, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 521 x 1138 cells | NLCD class code (landcover_classes.csv) | 250 | 0.08 MB | `216c3d1854a06bd43588797ace620fe1b7ffab35f8880defff4785886633d37e` |
| `landcover_02137727.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Catawba, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 736 x 746 cells | NLCD class code (landcover_classes.csv) | 250 | 0.06 MB | `93973588e747291b97cef3b4a57e29449a42a1c3c01206f97395c02121e3138b` |
| `landcover_02221525.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Murder Creek, GA | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 985 x 1039 cells | NLCD class code (landcover_classes.csv) | 250 | 0.13 MB | `679e2130051c6237aff34a75f42e5f614bc4c79264883e402834a129f9dbf44c` |
| `landcover_02296500.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Charlie Creek, FL | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 1160 x 1562 cells | NLCD class code (landcover_classes.csv) | 250 | 0.14 MB | `e454ee81b7e2d87e393415a08a6cf518dd67b4c05b111b0c0722c590ec2af73b` |
| `landcover_02298608.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Myakka, FL | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 856 x 803 cells | NLCD class code (landcover_classes.csv) | 250 | 0.06 MB | `ae315ce658f03cb5c198492c25ca532206998b552ebb1f45d2d685d6163cecb9` |
| `landcover_02349900.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Turkey Creek, GA | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 532 x 528 cells | NLCD class code (landcover_classes.csv) | 250 | 0.03 MB | `e94a1d566042877bc244b270ee5d63ca0b7d52140483c1d7df8566a4f1f400a2` |
| `landcover_02472000.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, Leaf River, MS | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 1463 x 2480 cells | NLCD class code (landcover_classes.csv) | 250 | 0.43 MB | `5fa5c4461848ce76ae41da0fea8ae345fd95990d447faf7f275f4b1ecc8a2ece` |
| `landcover_03439000.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, French Broad, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 535 x 776 cells | NLCD class code (landcover_classes.csv) | 250 | 0.04 MB | `1b94a1413263ab18478de8788d6f4f180106b760ff7e190e38decb0f4b804c2c` |
| `landcover_03463300.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, South Toe, NC | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 368 x 569 cells | NLCD class code (landcover_classes.csv) | 250 | 0.03 MB | `ccf559b2cdb54bcfc94e89a3c9cec107addf1ec7014e0f557134bc8564ec9696` |
| `landcover_03473000.tif` | GeoTIFF uint8 with colour table, LZW, tiled | land cover 2011, SF Holston, VA | EPSG:5070 NAD83 / Conus Albers (the native grid) | 30 m; 1783 x 1542 cells | NLCD class code (landcover_classes.csv) | 250 | 0.14 MB | `aba6af094120b42ac44156a3bcda9813268112cf286f807467dc8ad5b35ae0d4` |

## Sources and licences

The licence column repeats what the project files record. **UNVERIFIED** means that neither `results/data_flow.json` nor `results/basin_gis.json` holds a licence for the product; the exported files are derived from sources whose terms are not recorded here, and no licence of their own is asserted here. The access facts beside it are from `results/data_flow.json`.

| source | used for | what the project files record | licence |
|---|---|---|---|
| CAMELS basin outlines (basin_set_full_res, HCDN_nhru_final_671.shp) (producer: USGS (geospatial fabric), distributed with CAMELS) | outlines, gauges, attributes | version: CAMELS v1.2 (11 Mar 2016); four basins were fixed by hand in v1.1 (readme_FIRST.txt); obtained: CAMELS download: basin_set_full_res.zip; downloaded 2026-10-02 (file times, laptop clock UTC+04); login: see CAMELS-US | UNVERIFIED (no licence recorded in results/data_flow.json or results/basin_gis.json; data_flow.json records: login needed = no account recorded; the page asks the downloader to accept its terms of use); data/camels_us/readme.txt asks that publications using the CAMELS attributes cite Addor et al. 2017 and publications using the CAMELS time series cite Newman et al. 2015; results/data_flow.json names Newman et al. 2015 as the source of the outlines (no request to cite is recorded for them) |
| 1-arc-second elevation tiles in SRTM format (Mapzen/Tilezen terrain tiles) (producer: Mapzen / Tilezen (Terrain Tiles on AWS Open Data), from NASA SRTM, USGS 3DEP and other sources) | `elevation_<gauge_id>.tif` | version: SRTM was flown in February 2000; the tile set was built in 2016-2017 (results/basin_gis.json); the provider's version 1 was built in 2016Q2 and version 1.1 in 2017Q3. The SRTM release inside is not stated in the tiles: the tile builder's example configuration names SRTMGL1.003 (NASA SRTM version 3, void-filled, folder 2000.02.11), the configuration of the production build was not read (SRTM release: unverified); obtained: https://s3.amazonaws.com/elevation-tiles-prod/skadi/N35/N35W083.hgt.gz (the pattern: `skadi/<N or S><lat>/<N or S><lat><E or W><lon>.hgt.gz`, named by the south-west corner); downloaded 2026-10-03 (file times, laptop clock UTC+04); login: no (AWS Open Data) | UNVERIFIED (no licence recorded in results/data_flow.json or results/basin_gis.json; data_flow.json records: login needed = no (AWS Open Data)) |
| Annual NLCD land cover, 2011 layer (USGS / MRLC) (producer: US Geological Survey (EROS) with the Multi-Resolution Land Characteristics consortium) | `landcover_<gauge_id>.tif` | version: the 2011 layer of the yearly series, read from the USGS GeoServer layer Land-Cover-Native_conus_year_data; the collection number is not recorded in the project (the metadata read here is Collection 1.0, 1985-2023). It is not the older single-year 'NLCD 2011' release (collection: unverified); obtained: https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs (WCS 1.0.0 GetCoverage, one request per basin outline box, time=2011-01-01); downloaded 2026-10-03 (file times, laptop clock UTC+04); login: no | UNVERIFIED (no licence recorded in results/data_flow.json or results/basin_gis.json; data_flow.json records: login needed = no) |

## Check: the exported layers against results/basin_gis.json and the CAMELS attribute table

The clipping rule is the one of script 07 (a cell counts when its centre lies inside the outline), so the statistics read back from the exported rasters should equal results/basin_gis.json to numerical precision: tolerance 0.001 km2 on area, 0.01 m on mean elevation, 1e-06 on the forest share, the same cell counts and the same dominant group, and for the gauge point the same elevation cell and the same distance to the outline (tolerance 1e-06 km). **12 of 12 basins agree.** Against the CAMELS table (another source, other outlines and grids): area median +0.26% (largest +3.8%, Hyco Creek, NC); mean elevation median +1.6 m (largest +163 m, South Toe, NC); forest share median -12.7 points (largest -20.7, Little River, NC). Differences are exported minus reference.

### Area (km2)

| gauge_id | basin | exported (EPSG:5070) | basin_gis.json | diff | CAMELS area_gages2 | diff % |
|---|---|---|---|---|---|---|
| 02077200 | Hyco Creek, NC | 126.345 | 126.345 | -0.000000 | 121.69 | +3.83 |
| 02108000 | NE Cape Fear, NC | 1566.835 | 1566.835 | -0.000000 | 1569.48 | -0.17 |
| 02128000 | Little River, NC | 271.999 | 271.999 | -0.000000 | 273.48 | -0.54 |
| 02137727 | Catawba, NC | 329.307 | 329.307 | -0.000000 | 329.54 | -0.07 |
| 02221525 | Murder Creek, GA | 493.290 | 493.290 | +0.000000 | 490.80 | +0.51 |
| 02296500 | Charlie Creek, FL | 892.254 | 892.254 | +0.000000 | 886.40 | +0.66 |
| 02298608 | Myakka, FL | 313.855 | 313.855 | -0.000000 | 312.34 | +0.49 |
| 02349900 | Turkey Creek, GA | 124.291 | 124.291 | +0.000000 | 122.87 | +1.16 |
| 02472000 | Leaf River, MS | 1927.706 | 1927.706 | +0.000000 | 1927.13 | +0.03 |
| 03439000 | French Broad, NC | 178.186 | 178.186 | +0.000000 | 178.67 | -0.27 |
| 03463300 | South Toe, NC | 113.160 | 113.160 | -0.000000 | 112.16 | +0.89 |
| 03473000 | SF Holston, VA | 784.365 | 784.365 | -0.000000 | 784.81 | -0.06 |

### Mean elevation (m)

| gauge_id | basin | exported layer | basin_gis.json | diff | CAMELS elev_mean | diff | note |
|---|---|---|---|---|---|---|---|
| 02077200 | Hyco Creek, NC | 186.41 | 186.41 | +0.0000 | 194.00 | -7.6 |  |
| 02108000 | NE Cape Fear, NC | 34.13 | 34.13 | -0.0000 | 31.61 | +2.5 |  |
| 02128000 | Little River, NC | 208.14 | 208.14 | +0.0000 | 195.23 | +12.9 |  |
| 02137727 | Catawba, NC | 649.06 | 649.06 | +0.0000 | 602.18 | +46.9 | more than 25 m from CAMELS (see below) |
| 02221525 | Murder Creek, GA | 176.74 | 176.74 | +0.0000 | 176.01 | +0.7 |  |
| 02296500 | Charlie Creek, FL | 26.53 | 26.53 | -0.0000 | 31.23 | -4.7 |  |
| 02298608 | Myakka, FL | 21.84 | 21.84 | -0.0000 | 22.79 | -1.0 |  |
| 02349900 | Turkey Creek, GA | 119.16 | 119.16 | -0.0000 | 116.19 | +3.0 |  |
| 02472000 | Leaf River, MS | 118.42 | 118.42 | +0.0000 | 123.24 | -4.8 |  |
| 03439000 | French Broad, NC | 963.28 | 963.28 | -0.0000 | 863.18 | +100.1 | more than 25 m from CAMELS (see below) |
| 03463300 | South Toe, NC | 1188.54 | 1188.54 | -0.0000 | 1025.66 | +162.9 | more than 25 m from CAMELS (see below) |
| 03473000 | SF Holston, VA | 899.79 | 899.79 | -0.0000 | 916.29 | -16.5 |  |

**Known difference, carried over from results/basin_gis.json:** 3 mountain basins differ from the CAMELS elev_mean by more than 25 m (Catawba, NC +47 m, French Broad, NC +100 m, South Toe, NC +163 m). The note of script 07 reads: "Elevation: our mean differs from CAMELS elev_mean by +1.6 m in the median, but by more than 25 m in: Catawba, NC +47 m, French Broad, NC +100 m, South Toe, NC +163 m. Averaging our grid to coarser cells (up to 920 m) moves our mean by at most 2.5 m, so grid size does not explain this. The elev_mean field stored in the outline file is close to ours (mean absolute difference 8.4 m, largest -24.7 m, South Toe, NC). We cannot tell from the CAMELS documentation why its table is lower in these basins; the ranking of the basins is nearly the same (rank correlation 0.99) and the landscape classes of 01_data_audit.py (mountain = mean elevation above 600 m) are unchanged with our elevations."

### Dominant land cover and forest share

| gauge_id | basin | dominant group (exported) | basin_gis.json | same | dominant NLCD class (exported) | forest % exported | forest % basin_gis | CAMELS frac_forest % | diff points | CAMELS dom_land_cover (MODIS) |
|---|---|---|---|---|---|---|---|---|---|---|
| 02077200 | Hyco Creek, NC | forest | forest | yes | 41 deciduous forest | 62.6 | 62.6 | 71.1 | -8.5 | cropland/natural vegetation mosaic |
| 02108000 | NE Cape Fear, NC | cropland and pasture | cropland and pasture | yes | 82 cultivated crops | 21.9 | 21.9 | 42.3 | -20.5 | cropland/natural vegetation mosaic |
| 02128000 | Little River, NC | forest | forest | yes | 41 deciduous forest | 65.2 | 65.2 | 85.9 | -20.7 | cropland/natural vegetation mosaic |
| 02137727 | Catawba, NC | forest | forest | yes | 41 deciduous forest | 84.4 | 84.4 | 98.6 | -14.2 | Mixed Forests |
| 02221525 | Murder Creek, GA | forest | forest | yes | 42 evergreen forest | 63.2 | 63.2 | 82.5 | -19.4 | Mixed Forests |
| 02296500 | Charlie Creek, FL | cropland and pasture | cropland and pasture | yes | 81 pasture/hay | 2.4 | 2.4 | 16.8 | -14.3 | cropland/natural vegetation mosaic |
| 02298608 | Myakka, FL | cropland and pasture | cropland and pasture | yes | 81 pasture/hay | 4.5 | 4.5 | 7.6 | -3.0 | Woody Savannas |
| 02349900 | Turkey Creek, GA | cropland and pasture | cropland and pasture | yes | 82 cultivated crops | 16.4 | 16.4 | 9.9 | +6.5 | Croplands |
| 02472000 | Leaf River, MS | forest | forest | yes | 42 evergreen forest | 52.4 | 52.4 | 67.0 | -14.6 | Mixed Forests |
| 03439000 | French Broad, NC | forest | forest | yes | 41 deciduous forest | 89.4 | 89.4 | 99.5 | -10.0 | Mixed Forests |
| 03463300 | South Toe, NC | forest | forest | yes | 43 mixed forest | 94.3 | 94.3 | 100.0 | -5.7 | Deciduous Broadleaf Forest |
| 03473000 | SF Holston, VA | forest | forest | yes | 41 deciduous forest | 75.1 | 75.1 | 86.3 | -11.2 | Deciduous Broadleaf Forest |

The forest share at 30 m is below the CAMELS frac_forest in most basins; results/basin_gis.json (note 'Forest:') tests two reasons (woody wetlands, grid size) and finds neither a full explanation. CAMELS dom_land_cover comes from a MODIS map with other classes (IGBP), so it is shown for orientation only.

## Verification (read back after writing)

- PASS gpkg_outlines_count_12: 12 features
- PASS gpkg_outlines_crs_4269: EPSG:4269
- PASS gpkg_outlines_ids: gauge ids
- PASS gpkg_outlines_geometry_unchanged: planar area in degrees identical
- PASS gpkg_gauges_count_12: 12 features, EPSG:4269
- PASS gpkg_gauges_inside_outlines: every gauge point lies inside its outline
- PASS gpkg_layers: ['outlines', 'gauges']
- PASS geojson_count_12_crs_4326: 12 features, EPSG:4326
- PASS geojson_matches_gpkg: area_km2 attribute identical
- PASS gauges_geojson_count_12_crs_4326: 12 features, EPSG:4326
- PASS twelve_rasters_each: 12 elevation, 12 land cover
- PASS all_basins_agree_with_basin_gis: all 12
- PASS total_size_under_150_mb: 9.4 MB
- PASS readme_tables_cells_per_row: 5 tables, 67 rows, each with its header's number of cells
- PASS the 12 elevation GeoTIFFs: EPSG:4326, int16, nodata -32768, LZW, 1-arc-second grid, bounds contain the outline, mean elevation and cell count read back equal results/basin_gis.json
- PASS the 12 land-cover GeoTIFFs: EPSG:5070, uint8, nodata 250, LZW, 30 m, colour table, bounds contain the outline, cells outside the outline are nodata, class shares and cell counts read back equal results/basin_gis.json

## How to open

QGIS: drag `basins_12.gpkg` in (both layers), then the GeoTIFFs; the land-cover files carry their colour table. The three CRS differ (EPSG:4269, 4326, 5070): let QGIS reproject on the fly. Python: `geopandas.read_file('basins_12.gpkg', layer='outlines')`, `rasterio.open('elevation_02077200.tif')`. The GeoJSON twins are for web maps (Leaflet, geojson.io).

## Notes

- Order of the basins: results/data_audit.json `selected` (the order of every other results file). Landscapes: Northern lowlands and Piedmont (4), Mountains (above 600 m) (4), Southern lowlands (south of 32.5N) (4).
- The elevation grid is the tiles' own: cell centres on whole arc-seconds, origin at half a cell; 8,930,167 cells inside the 12 outlines, 9,780,302 written including the buffer. Land cover: 7,912,882 cells inside the outlines, 0 without a valid class.
- The GeoPackage records its write time, so its sha256 changes with every run even when nothing else does; the GeoTIFFs and GeoJSON files are deterministic.
- The GeoJSON twins were reprojected with pyproj's choice for EPSG:4269 to EPSG:4326: "NAD83 to WGS 84 (1) (with axis order normalized for visualization) = a null transformation, coordinates unchanged; 49 grid-shift operations are not installed".
- Nothing here is fitted, tuned or selected; the train/validation/test split plays no role. No model reads these layers (the models read the CAMELS attribute values).
