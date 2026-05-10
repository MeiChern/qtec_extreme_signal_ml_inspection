#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/FigureS1_environmental_predictor_maps.py
# Renamed package path: code/figure_drivers/figureS1_environmental_predictor_maps.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import json

from submission_build_common import (
    ENV_RASTER_DIR,
    FONT,
    TABLE_DIR,
    add_scalebar_lonlat,
    clip_artist_to_hull,
    clip_gdf_to_hull,
    coverage_hull_lonlat_from_path,
    decimate_raster,
    ensure_style,
    grid_meta,
    load_env_raster,
    load_geo_layers,
    masked_raster,
    robust_limits,
    save_figure,
    text_halo,
    transformers,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


FIG_STEM = "FigureS1_environmental_predictor_maps"
MAP_XLIM = (88.18, 97.86)
MAP_YLIM = (27.8, 38.5)
RAIL_DASHES = (3.5, 2.0)
COLORBAR_BOUNDS = [0.53, 0.06, 0.36, 0.038]
DISPLAY_TRANSFORMS: dict[str, tuple[float, float]] = {
    # TPDC ANUSPLIN meteorology stores MAAT and precipitation as tenths.
    # Display and colorbar limits must therefore be computed after raw / 10.
    "temperature_mean": (0.1, 0.0),
    "precipitation_mean": (0.1, 0.0),
}

# Ordered to match the requested Table 01-based supplementary layout.
VAR_SPECS = (
    {"var": "magt", "title": "MAGT (°C)"},
    {"var": "temperature_mean", "title": "MAAT (°C)"},
    {"var": "precipitation_mean", "title": "Precipitation (mm yr$^{-1}$)"},
    {"var": "dirpr", "title": "Direct radiation (W m$^{-2}$)"},
    {"var": "difpr", "title": "Diffuse radiation (W m$^{-2}$)"},
    {"var": "bulk_density", "title": "Bulk density (kg m$^{-3}$)"},
    {"var": "cf", "title": "Coarse fragments (vol.%)"},
    {"var": "soc", "title": "Soil organic carbon (kg m$^{-2}$)"},
    {"var": "soil_thickness", "title": "Soil thickness (m)"},
    {"var": "vwc35", "title": "VWC35 (%)"},
    {"var": "dem", "title": "Elevation (m a.s.l.)"},
    {"var": "slope", "title": "Slope (°)"},
    {"var": "twi", "title": "TWI (-)"},
    {"var": "ndvi", "title": "NDVI (-)"},
    {"var": "gpp_mean", "title": "GPP (gC m$^{-2}$ yr$^{-1}$)"},
)
COLORMAPS = {
    "magt": "coolwarm",
    "precipitation_mean": "YlGnBu",
    "temperature_mean": "coolwarm",
    "bulk_density": "cividis",
    "cf": "viridis",
    "soc": "YlOrBr",
    "soil_thickness": "copper",
    "vwc35": "PuBu",
    "dem": "terrain",
    "difpr": "magma",
    "dirpr": "inferno",
    "slope": "cividis",
    "twi": "viridis",
    "gpp_mean": "YlGn",
    "ndvi": "Greens",
}
PANEL_LABELS = tuple("ABCDEFGHIJKLMNO")


def colorbar_formatter(scale: float):
    if scale >= 1000.0:
        return FuncFormatter(lambda x, pos: f"{x / 1000.0:g}k")
    if scale >= 100.0:
        return FuncFormatter(lambda x, pos: f"{x:.0f}")
    if scale >= 10.0:
        return FuncFormatter(lambda x, pos: f"{x:.1f}".rstrip("0").rstrip("."))
    return FuncFormatter(lambda x, pos: f"{x:.2f}".rstrip("0").rstrip("."))


def transform_variable_for_plot(var: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    if var == "slope":
        return np.degrees(out)
    scale_offset = DISPLAY_TRANSFORMS.get(var)
    if scale_offset is None:
        return out
    scale, offset = scale_offset
    return out * float(scale) + float(offset)


def _fmt_lon(x, _pos):
    if not np.isfinite(x):
        return ""
    hemi = "E" if x >= 0 else "W"
    value = abs(float(x))
    text = f"{int(round(value))}" if np.isclose(value, round(value), atol=1e-6) else f"{value:.1f}".rstrip("0").rstrip(".")
    return rf"${text}^\circ$ {hemi}"


def _fmt_lat(y, _pos):
    if not np.isfinite(y):
        return ""
    hemi = "N" if y >= 0 else "S"
    value = abs(float(y))
    text = f"{int(round(value))}" if np.isclose(value, round(value), atol=1e-6) else f"{value:.1f}".rstrip("0").rstrip(".")
    return rf"${text}^\circ$ {hemi}"


def build_lonlat_mesh(*, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    meta = grid_meta()
    res = float(meta["res"])
    gx0, gy1 = int(meta["gx0"]), int(meta["gy1"])
    nrows, ncols = int(meta["nrows"]), int(meta["ncols"])
    row_idx = np.arange(0, nrows, stride, dtype=np.int64)
    col_idx = np.arange(0, ncols, stride, dtype=np.int64)
    easting = (gx0 + col_idx).astype(float) * res
    northing = (gy1 - row_idx).astype(float) * res
    e_grid, n_grid = np.meshgrid(easting, northing)
    to_lonlat = transformers()[0]
    lon_grid, lat_grid = to_lonlat.transform(e_grid, n_grid)
    return (
        np.flipud(np.asarray(lon_grid, dtype=float)),
        np.flipud(np.asarray(lat_grid, dtype=float)),
    )


def get_lonlat_mesh(stride: int, cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if stride not in cache:
        cache[stride] = build_lonlat_mesh(stride=stride)
    return cache[stride]


def plot_dashed_railway(ax, railway_gdf) -> None:
    if railway_gdf.empty:
        return
    railway_gdf.plot(ax=ax, color="white", linewidth=1.5, alpha=1.0, zorder=4.0)
    railway_gdf.plot(
        ax=ax,
        color="black",
        linewidth=0.75,
        linestyle=(0, RAIL_DASHES),
        alpha=1.0,
        zorder=4.1,
    )


def style_map_axis(ax, *, show_xlabel: bool, show_ylabel: bool) -> None:
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_xticks([89.0, 93.0, 97.0])
    ax.set_yticks([28.0, 33.0, 38.0])
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude" if show_xlabel else "")
    ax.set_ylabel("Latitude" if show_ylabel else "")
    ax.tick_params(
        length=2.4,
        pad=1.2,
        labelbottom=show_xlabel,
        labelleft=show_ylabel,
    )
    ax.set_aspect("auto")


def main() -> None:
    ensure_style()
    layers = load_geo_layers()
    railway_ll = layers["railway_ll"]
    lonlat_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    fig, axes = plt.subplots(3, 5, figsize=(13.2, 8.7), constrained_layout=False)
    plt.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.085, wspace=0.09, hspace=0.18)
    display_limits: dict[str, dict[str, float | str]] = {}

    for idx, (ax, spec, label) in enumerate(zip(axes.flat, VAR_SPECS, PANEL_LABELS)):
        var = spec["var"]
        raw_arr = np.asarray(load_env_raster(var))
        arr = transform_variable_for_plot(var, raw_arr)
        arr_ds, stride = decimate_raster(arr, target_max=1100)
        plot_arr = masked_raster(np.flipud(arr_ds), zero_is_nodata=True)
        lo, hi = robust_limits(plot_arr.filled(np.nan), 2, 98)
        display_limits[var] = {
            "title": spec["title"],
            "scale": DISPLAY_TRANSFORMS.get(var, (1.0, 0.0))[0],
            "offset": DISPLAY_TRANSFORMS.get(var, (1.0, 0.0))[1],
            "display_p02": lo,
            "display_p98": hi,
        }
        lon_plot, lat_plot = get_lonlat_mesh(stride, lonlat_cache)
        hull = coverage_hull_lonlat_from_path(str(ENV_RASTER_DIR / f"{var}_mean_f32.memmap"))
        plot_cmap = plt.get_cmap(COLORMAPS[var]).copy()
        plot_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        im = ax.pcolormesh(
            lon_plot,
            lat_plot,
            plot_arr,
            cmap=plot_cmap,
            vmin=lo,
            vmax=hi,
            shading="auto",
            rasterized=True,
            zorder=0,
        )
        clip_artist_to_hull(ax, im, hull)
        plot_dashed_railway(ax, clip_gdf_to_hull(railway_ll, hull))

        style_map_axis(ax, show_xlabel=idx >= 10, show_ylabel=(idx % 5) == 0)
        ax.set_title(spec["title"], fontsize=FONT["axis"] - 0.2, pad=3.5)
        if idx == 0:
            add_scalebar_lonlat(ax, length_km=100, lon=88.85, lat=28.18, linewidth=1.2, color="0.1")

        panel_text = ax.annotate(
            label,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-3.5, 3.5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=FONT["panel"] - 0.8,
            fontweight="bold",
            color="0.05",
            clip_on=False,
        )
        panel_text.set_path_effects(text_halo(2.0))

        cax = ax.inset_axes(COLORBAR_BOUNDS)
        cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()
        tick_scale = max(abs(t) for t in cb.get_ticks()) if len(cb.get_ticks()) else 1.0
        cb.ax.xaxis.set_major_formatter(colorbar_formatter(tick_scale))
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor("0.45")
        cb.ax.tick_params(labelsize=FONT["annotation"] - 0.2, length=1.8, pad=0.7)

    save_figure(fig, FIG_STEM)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / f"{FIG_STEM}_display_limits.json").write_text(
        json.dumps(display_limits, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
