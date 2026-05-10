# Reproducibility Limits

This package is complete for reviewer inspection of the submitted result tables,
low-resolution figure assets, renamed code snapshots, and the redistributed
processed `d_u` raster.

The manuscript source folder is intentionally not included. The package retains
manuscript-derived result tables and figure-support summaries for inspection.

The package is not a raw-data reprocessing archive. It does not include:

- raw Sentinel-1 SLC or interferogram products;
- full MintPy/ISCE2 working directories;
- environmental covariate rasters or harmonized environmental sample tables;
- thermokarst lake, retrogressive thaw slump, permafrost-map, or railway
  shapefile products from third-party providers;
- trained model cache files or full wall-to-wall susceptibility caches;
- deformation-gradient rasters other than what can be rederived from `d_u`.

Full reruns of the susceptibility models and map products require obtaining
third-party inputs listed in `DATA_SOURCES.md`, harmonizing them to the analysis
grid, and restoring the original project-style module names/layout expected by
the code snapshots.
