# Data Sources And Redistribution Scope

Only the processed `d_u` raster is redistributed in this package. All other
environmental, inventory, and geospatial context products should be obtained
from the original providers. This matches the current data-sharing plan in the
submission.

| Layer | Source | Link |
| --- | --- | --- |
| `d_u` vertical deformation rate | This study, Sentinel-1 TS-InSAR, 2015-2021 analysis window | Packaged at `data/du_2015_2021/du_2015_2021_mm_yr.tif` |
| Raw Sentinel-1 SAR | Copernicus Sentinel-1 | https://dataspace.copernicus.eu/ |
| SRTM DEM used in InSAR processing | Shuttle Radar Topography Mission | https://doi.org/10.1029/2005RG000183 |
| Copernicus DEM / terrain derivatives | Copernicus GLO-30 DEM | https://doi.org/10.5270/ESA-c5d3d65 |
| NDVI | TPDC vegetation index data of Qinghai Tibet Plateau | https://doi.org/10.11888/Ecolo.tpdc.270449 |
| GPP | 30-m GPP dataset for China | https://doi.org/10.1038/s41597-024-03893-x |
| MAGT | TPDC MAGT and permafrost thermal stability dataset | https://doi.org/10.11888/Geogra.tpdc.270672 |
| MAAT and precipitation | TPDC ANUSPLIN gridded meteorology | https://doi.org/10.11888/Meteoro.tpdc.270239 |
| Soil products | TPDC digital soil mapping products for the Qinghai-Tibet Plateau | https://doi.org/10.11888/Terre.tpdc.272482 |
| VWC35 / ground-ice context | Zou et al. 2024 ground-ice product | https://doi.org/10.1002/ppp.2226 |
| Permafrost distribution | Zou et al. 2017 TTOP permafrost map | https://doi.org/10.5194/tc-11-2527-2017 |
| Thermokarst lake inventory | Wei et al. 2021 Sentinel-based thermokarst lake inventory | https://doi.org/10.1029/2021EA001950 |
| Retrogressive thaw slump inventory | Xia et al. 2022 QTEC RTS inventory | https://doi.org/10.5194/essd-14-3875-2022 |
| Large-lake inventory | TPDC lake dataset v3.1 / Zhang et al. 2019 | https://doi.org/10.1016/j.scib.2019.07.018 |

Derived spatial-contrast predictors, distance fields, and model-sample tables
are not redistributed because they encode third-party environmental layers or
inventory geometries. The current manuscript tables and figure-result summaries
are included for inspection under `tables/`.
