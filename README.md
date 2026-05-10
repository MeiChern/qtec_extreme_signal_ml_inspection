# QTEC Extreme Deformation Reviewer Inspection Package

This folder is a reviewer-facing inspection package for the PNAS Nexus initial
submission:

`Ground deformation extremes align with permafrost-transition and thermokarst-context signals on the Tibetan Plateau.`

The layout is organized for inspection rather than to mirror the local working
tree.

## Included Contents

- `data/du_2015_2021/`: processed vertical deformation rate (`d_u`) GeoTIFF and metadata.
- `figures/lowres_watermarked/`: NHRS-watermarked low-resolution Figures 1-5 and S1-S5; each JPEG is below 2 MB.
- `tables/manuscript_tables/`: staged Table S1-S8 result files used by the manuscript.
- `tables/figure_source_tables/`: key figure-support outputs and calibration summaries.
- `code/figure_drivers/`: renamed source snapshots for submitted figure scripts.
- `code/analysis_support/`: renamed helper and diagnostic scripts.
- `code/original_project_helpers/`: renamed legacy project helpers retained for provenance.
- `DATA_SOURCES.md`: third-party data links and redistribution scope.
- `REPRODUCIBILITY_LIMITS.md`: scope notes for what can and cannot be rerun from this reduced package.
- `MANIFEST.tsv` and `checksums.sha256`: inventory and checksums.
- `LICENSE`: dual-license terms for source code and redistributed research materials.

The manuscript source folder is intentionally not included in this package.
This package contains the inspection data, tables, low-resolution figure assets,
and source-code snapshots needed to evaluate the submitted results without
redistributing the manuscript text.

## Data Scope

Only the processed `d_u` raster is redistributed as data. The package does not
redistribute raw Sentinel-1 SLCs, full InSAR time series, environmental raster
stacks, environmental sample tables, thermokarst inventories, permafrost maps,
or model cache files. Third-party products should be obtained from the providers
listed in `DATA_SOURCES.md` and Table S1.

The deformation-gradient magnitude (`|grad d_u|`) used in the paper is derived
from `d_u`; it is not redistributed here as a separate data raster because the
requested data-sharing scope is `d_u` only.

## Code Scope

The scripts in `code/` are renamed inspection copies. They preserve the original
source content with a short provenance header, but many still contain original
module imports and expect the original project data layout. They are included to
make the analysis logic inspectable, not to promise one-command reruns from this
reduced package.

For package integrity verification, run:

```bash
shasum -a 256 -c checksums.sha256
```

## License

Source code in this repository is licensed under the MIT License.

Unless otherwise noted, data files, tables, figures, documentation, and other
research materials are licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

The NHRS Lab logo, institutional marks, and any third-party materials are not
covered by these licenses unless explicitly stated. Reuse of excluded materials
requires permission from the relevant rights holder.

## Notes For Reviewers

- Negative `d_u` values indicate subsidence.
- The packaged `d_u` GeoTIFF uses the project CRS read from the QTEC railway
  shapefile projection and the cell-center grid convention used by the submitted
  figure scripts.
- The figure JPEGs are low-resolution inspection assets, not final production
  figure files for journal upload.
