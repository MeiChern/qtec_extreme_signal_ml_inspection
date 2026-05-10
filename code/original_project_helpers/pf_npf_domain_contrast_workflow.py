#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure_reorganize_pf_vs_npf_in_du_and_gradient.py
# Renamed package path: code/original_project_helpers/pf_npf_domain_contrast_workflow.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from pyproj import CRS, Transformer

import Figure_reorganize_railway_buffer_analysis as railbuf
import Figure_reorganized_railway_extreme_deformation_inspection as railinspect

SEED = 42
CHUNKSIZE = 400_000
FIG_BASENAME = "Figure_reorganize_pf_vs_npf_in_du_and_gradient"
SAMPLE_PER_DOMAIN_DEFAULT = 180_000
CURVE_Q_DEFAULT = 22
HIST_BINS_DU = 54
HIST_BINS_GRAD = 48
TAIL_COLOR_BLEND = 0.45
HIST_CORE_BLEND = 0.62
HIST_CORE_ALPHA = 0.22
HIST_TAIL_ALPHA = 0.38
HIST_LINESTYLE = (0, (4.0, 2.0))
REL_CORE_BLEND = 0.60
REL_CORE_FILL_ALPHA = 0.10
REL_TAIL_FILL_ALPHA = 0.18

PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
DOMAIN_LABELS = {"pf": "Permafrost", "npf": "Non-Permafrost"}
DEFAULT_BUFFER_WIDTH_KM = railinspect.DEFAULT_BUFFER_WIDTH_KM
MAP_XLIM = railinspect.MAIN_XLIM
MAP_LAT_LIMITS = (27.8, 38.5)
MAP_LAT_TICKS = [28, 30, 32, 34, 36, 38]
SCALE_BAR_KM = 100.0
TARGET_MAP_PIXELS = 900
DENSITY_GRID_NX = 420
DENSITY_GRID_NY = 320
DENSITY_BANDWIDTH_KM = 22.0
BACKGROUND_MASK_COLOR = "#E3E3E3"
MAP_SPOT_SIZE = 3.0
MAP_PF_SPOT_COLOR = railbuf.blend_with_white(PF_COLOR, 0.34)
MAP_NPF_SPOT_COLOR = railbuf.blend_with_white(NPF_COLOR, 0.34)
MAP_SPOT_ALPHA = 0.72
METEORO_SITE_COLOR = "#3A3A3A"
METEORO_SITE_SIZE = 36.0
METEORO_INSET_SITE_SIZE = 62.0
GOLMUD_LABEL_X = 96.2
GOLMUD_LABEL_Y = 36.0
XIAOZAOHUO_LABEL_X = 93.5
XIAOZAOHUO_LABEL_Y_OFFSET = 0.0
SHANNAN_LABEL_X = railinspect.SHANNAN_LABEL_X
SHANNAN_LABEL_Y_OFFSET = -0.55
TUOTUOHE_INSET_HEIGHT = 0.23
TUOTUOHE_INSET_TOP_LAT = 33.0
TUOTUOHE_INSET_Y0 = (
    (TUOTUOHE_INSET_TOP_LAT - MAP_LAT_LIMITS[0]) / (MAP_LAT_LIMITS[1] - MAP_LAT_LIMITS[0]) - TUOTUOHE_INSET_HEIGHT
)
MAP_ZOOM_SPECS = (
    {
        "site_label": "Wudaoliang",
        "bounds": (0.02, 0.75, 0.25, 0.23),
        "xpad": 0.42,
        "ypad": 0.24,
    },
    {
        "site_label": "Tuotuohe",
        "bounds": (0.63, TUOTUOHE_INSET_Y0, 0.25, TUOTUOHE_INSET_HEIGHT),
        "xpad": 0.34,
        "ypad": 0.22,
    },
)
MAP_EXTENT_CACHE_NAMES = (
    "Figure_reorganize_du_du_gradient_ml_features_lonlat_extent.json",
    "figure4_lonlat_extent.json",
)

def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def log_step(message: str) -> None:
    print(f"[{FIG_BASENAME}] {message}")


def ensure_proj_lib() -> None:
    proj_dir = Path(sys.executable).resolve().parents[1] / "share" / "proj"
    if "PROJ_LIB" not in os.environ and proj_dir.exists():
        os.environ["PROJ_LIB"] = str(proj_dir)


def style_open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def apply_bold_nonlegend(ax) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        str(label).strip("()").lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="0.05",
        clip_on=False,
    )


def soften_color(color: str, blend: float = TAIL_COLOR_BLEND) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - blend) * rgb + blend)


def add_domain_box(ax, *, loc: str = "lower right") -> None:
    handles = [
        Line2D([0], [0], color=PF_COLOR, linewidth=3.0, solid_capstyle="round"),
        Line2D([0], [0], color=NPF_COLOR, linewidth=3.0, solid_capstyle="round"),
    ]
    legend = ax.legend(
        handles,
        [DOMAIN_LABELS["pf"], DOMAIN_LABELS["npf"]],
        loc=loc,
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        handlelength=1.8,
        handletextpad=0.6,
        borderpad=0.45,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.70")
    legend.get_frame().set_linewidth(0.8)


def add_relationship_box(ax, *, loc: str = "lower right") -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=PF_COLOR,
            linewidth=3.0,
            linestyle=HIST_LINESTYLE,
            solid_capstyle="round",
        ),
        Line2D(
            [0],
            [0],
            color=NPF_COLOR,
            linewidth=3.0,
            linestyle=HIST_LINESTYLE,
            solid_capstyle="round",
        ),
        Line2D(
            [0],
            [0],
            color=PF_COLOR,
            linewidth=3.0,
            linestyle="-",
            solid_capstyle="round",
        ),
        Line2D(
            [0],
            [0],
            color=NPF_COLOR,
            linewidth=3.0,
            linestyle="-",
            solid_capstyle="round",
        ),
    ]
    labels = [
        r"PF core: $|d_u| \leq P95$",
        r"NPF core: $|d_u| \leq P95$",
        r"PF extreme: $|d_u| > P95$",
        r"NPF extreme: $|d_u| > P95$",
    ]
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        handlelength=1.9,
        handletextpad=0.7,
        borderpad=0.50,
        labelspacing=0.42,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.70")
    legend.get_frame().set_linewidth(0.8)


def format_degree_hemisphere(value: float, *, positive: str, negative: str) -> str:
    if not np.isfinite(value):
        return ""
    hemi = positive if value >= 0 else negative
    abs_value = abs(float(value))
    if np.isclose(abs_value, round(abs_value), atol=1e-6):
        coord_text = f"{int(round(abs_value))}"
    else:
        coord_text = f"{abs_value:.1f}".rstrip("0").rstrip(".")
    return rf"${coord_text}^\circ$ {hemi}"


def robust_clip(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def configure_spatial_map_axis(ax, *, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    yticks = [tick for tick in MAP_LAT_TICKS if float(ylim[0]) <= tick <= float(ylim[1])]
    if yticks:
        ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: format_degree_hemisphere(x, positive="E", negative="W")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: format_degree_hemisphere(y, positive="N", negative="S")))
    ax.set_aspect("auto")
    ax.set_anchor("SW")
    ax.grid(False)
    style_open_axes(ax)
    apply_bold_nonlegend(ax)


def zonal_degree_length_km(lat_deg: float) -> float:
    lat_rad = np.deg2rad(float(lat_deg))
    return (
        111.41288 * np.cos(lat_rad)
        - 0.09350 * np.cos(3.0 * lat_rad)
        + 0.00012 * np.cos(5.0 * lat_rad)
    )


def choose_scale_bar_anchor(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    lon0 = float(xlim[0]) + 0.07 * (float(xlim[1]) - float(xlim[0]))
    lat0 = float(ylim[0]) + 0.46 * (float(ylim[1]) - float(ylim[0]))
    return lon0, lat0


def add_scale_bar_lonlat(
    ax,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    length_km: float = SCALE_BAR_KM,
    color: str = "k",
) -> None:
    lon0, lat0 = choose_scale_bar_anchor(xlim, ylim)
    km_per_deg_lon = zonal_degree_length_km(lat0)
    if not np.isfinite(km_per_deg_lon) or km_per_deg_lon <= 0.0:
        return

    dlon = float(length_km) / km_per_deg_lon
    lon1 = lon0 + dlon
    cap_half_height = 0.14
    text_pad = 0.18

    ax.plot([lon0, lon1], [lat0, lat0], color=color, linewidth=2.2, solid_capstyle="butt", zorder=6.2)
    ax.plot([lon0, lon0], [lat0 - cap_half_height, lat0 + cap_half_height], color=color, linewidth=1.4, zorder=6.2)
    ax.plot([lon1, lon1], [lat0 - cap_half_height, lat0 + cap_half_height], color=color, linewidth=1.4, zorder=6.2)
    ax.text(
        0.5 * (lon0 + lon1),
        lat0 + text_pad,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(1.0, 1.0, 1.0, 0.78), edgecolor="none"),
        zorder=6.4,
    )


def choose_stride(nrows: int, ncols: int, target_max: int = TARGET_MAP_PIXELS) -> int:
    return max(1, int(max(nrows, ncols) / target_max))


def build_lonlat_mesh_from_grid(
    *,
    res: float,
    gx0: int,
    gy1: int,
    nrows: int,
    ncols: int,
    stride: int,
    source_crs,
) -> tuple[np.ndarray, np.ndarray]:
    row_idx = np.arange(0, nrows, stride, dtype=np.int64)
    col_idx = np.arange(0, ncols, stride, dtype=np.int64)
    easting = (gx0 + col_idx).astype(float) * res
    northing = (gy1 - row_idx).astype(float) * res
    e_grid, n_grid = np.meshgrid(easting, northing)

    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    lon_grid, lat_grid = transformer.transform(e_grid, n_grid)
    return np.flipud(np.asarray(lon_grid, dtype=float)), np.flipud(np.asarray(lat_grid, dtype=float))


def resolve_lonlat_mesh_extent(lon_plot: np.ndarray, lat_plot: np.ndarray) -> list[float]:
    finite = np.isfinite(lon_plot) & np.isfinite(lat_plot)
    if not finite.any():
        raise RuntimeError("Projected-to-lonlat grid transform did not produce any finite cells for plotting.")
    return [
        float(np.nanmin(lon_plot[finite])),
        float(np.nanmax(lon_plot[finite])),
        float(np.nanmin(lat_plot[finite])),
        float(np.nanmax(lat_plot[finite])),
    ]


def resolve_cached_lonlat_extent(cache_dir: Path) -> tuple[list[float], Path | None]:
    for name in MAP_EXTENT_CACHE_NAMES:
        path = cache_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            extent = payload.get("extent_lonlat")
            if isinstance(extent, list) and len(extent) == 4:
                return [float(v) for v in extent], path
        except Exception:
            continue
    return [float(MAP_XLIM[0]), float(MAP_XLIM[1]), float(MAP_LAT_LIMITS[0]), float(MAP_LAT_LIMITS[1])], None


def resolve_source_crs(crs_info_path: Path, railway_shp: Path):
    if crs_info_path.exists():
        payload = crs_info_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"CRS:\s*(.+?)<br/>\[INFO\] Transform:", payload, flags=re.DOTALL)
        if match:
            return CRS.from_wkt(match.group(1).strip())

    prj_path = railway_shp.with_suffix(".prj")
    if prj_path.exists():
        return CRS.from_wkt(prj_path.read_text(encoding="utf-8", errors="ignore"))
    raise FileNotFoundError(f"Could not resolve a projected CRS from {crs_info_path} or {prj_path}.")


def resolve_dem_raster_path(base_dir: Path, cache_dir: Path) -> Path:
    candidates = [
        cache_dir / "env_review" / "rasters" / "dem_mean_f32.memmap",
        cache_dir / "Figure_reorganize_extreme_deformation_susceptibity_transition_rasters" / "raw_rasters" / "dem_mean_f32.memmap",
        base_dir / "outputs" / "gradient_driver_analysis" / "env_spatial_gradient_analysis" / "rasters" / "dem__raw_f32.memmap",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not resolve a DEM raster memmap for map background.")


def build_elevation_mask(elevation: np.ndarray) -> np.ndarray:
    elev = np.asarray(elevation, dtype=float)
    finite = np.isfinite(elev)
    if not finite.any():
        return np.full_like(elev, np.nan, dtype=float)
    return np.where(finite, 1.0, np.nan)


def gaussian_kernel1d(sigma_bins: float, *, truncate: float = 3.0) -> np.ndarray:
    if not np.isfinite(sigma_bins) or sigma_bins <= 0.05:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(truncate * sigma_bins)))
    coords = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (coords / float(sigma_bins)) ** 2)
    kernel /= np.sum(kernel)
    return kernel


def smooth_grid_gaussian(arr: np.ndarray, *, sigma_y: float, sigma_x: float) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    kernel_y = gaussian_kernel1d(sigma_y)
    kernel_x = gaussian_kernel1d(sigma_x)
    if kernel_y.size > 1:
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel_y, mode="same"), 0, out)
    if kernel_x.size > 1:
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel_x, mode="same"), 1, out)
    return out


def build_spatial_kde(
    sub_df: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bandwidth_km: float = DENSITY_BANDWIDTH_KM,
    nx: int = DENSITY_GRID_NX,
    ny: int = DENSITY_GRID_NY,
) -> np.ndarray:
    lon = pd.to_numeric(sub_df.get("longitude"), errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(sub_df.get("latitude"), errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= float(xlim[0]))
        & (lon <= float(xlim[1]))
        & (lat >= float(ylim[0]))
        & (lat <= float(ylim[1]))
    )
    if not np.any(mask):
        return np.full((int(ny), int(nx)), np.nan, dtype=float)

    hist, _, _ = np.histogram2d(
        lat[mask],
        lon[mask],
        bins=[int(ny), int(nx)],
        range=[[float(ylim[0]), float(ylim[1])], [float(xlim[0]), float(xlim[1])]],
    )

    mean_lat = 0.5 * (float(ylim[0]) + float(ylim[1]))
    km_per_deg_lon = max(1e-6, zonal_degree_length_km(mean_lat))
    km_per_deg_lat = 111.132
    dx_km = max(1e-6, (float(xlim[1]) - float(xlim[0])) * km_per_deg_lon / float(nx))
    dy_km = max(1e-6, (float(ylim[1]) - float(ylim[0])) * km_per_deg_lat / float(ny))
    sigma_x = float(bandwidth_km) / dx_km
    sigma_y = float(bandwidth_km) / dy_km
    density = smooth_grid_gaussian(hist, sigma_y=sigma_y, sigma_x=sigma_x)
    if np.nanmax(density) > 0.0:
        density = density / float(np.nanmax(density))
    density = np.asarray(density, dtype=float)
    density[density <= 0.0] = np.nan
    return density


def build_density_cmap(base_color: str, name: str) -> LinearSegmentedColormap:
    rgb = np.asarray(to_rgb(base_color), dtype=float)
    light = 1.0 - 0.76 * (1.0 - rgb)
    return LinearSegmentedColormap.from_list(name, ["#ffffff", tuple(light), tuple(rgb)])


def render_density_overlay(
    ax,
    density: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    cmap,
    zorder: float = 2.4,
):
    vals = np.asarray(density, dtype=float)
    finite = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite.size == 0:
        return

    lo, hi = np.percentile(finite, [22.0, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0

    normed = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    rgba = cmap_obj(normed)
    alpha = np.power(normed, 0.82) * 0.92
    alpha[~np.isfinite(vals)] = 0.0
    alpha[normed <= 0.0] = 0.0
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        interpolation="bilinear",
        zorder=zorder,
        aspect="auto",
    )


def add_meteoro_site_marker(
    ax,
    site_row: pd.Series,
    *,
    show_label: bool,
    marker_size: float,
) -> None:
    ax.scatter(
        [float(site_row["longitude"])],
        [float(site_row["latitude"])],
        s=marker_size,
        marker="^",
        color=METEORO_SITE_COLOR,
        edgecolors=METEORO_SITE_COLOR,
        linewidths=0.0,
        zorder=5.4,
    )
    if not show_label:
        return
    text = ax.annotate(
        str(site_row["site_label"]),
        xy=(float(site_row["longitude"]), float(site_row["latitude"])),
        xytext=(4.0, 2.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=METEORO_SITE_COLOR,
        zorder=6.2,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def add_reference_meteoro_sites(ax, site_df: pd.DataFrame) -> None:
    if site_df.empty:
        return
    xmid = float(np.mean(ax.get_xlim()))
    for site in site_df.itertuples(index=False):
        site_row = pd.Series({"site_label": site.site_label, "longitude": site.longitude, "latitude": site.latitude})
        add_meteoro_site_marker(ax, site_row, show_label=False, marker_size=METEORO_SITE_SIZE)
        label_key = str(site.site_label).strip().lower()
        if float(site.longitude) >= xmid:
            dx = -railinspect.METEORO_LABEL_OFFSET_PT
            ha = "right"
        else:
            dx = railinspect.METEORO_LABEL_OFFSET_PT
            ha = "left"
        if label_key == "golmud":
            text = ax.annotate(
                str(site.site_label),
                xy=(float(site.longitude), float(site.latitude)),
                xytext=(GOLMUD_LABEL_X, GOLMUD_LABEL_Y),
                textcoords="data",
                ha="right",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=METEORO_SITE_COLOR,
                zorder=6.2,
            )
        elif label_key == "xiaozaohuo":
            text = ax.annotate(
                str(site.site_label),
                xy=(float(site.longitude), float(site.latitude)),
                xytext=(XIAOZAOHUO_LABEL_X, float(site.latitude) + XIAOZAOHUO_LABEL_Y_OFFSET),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=METEORO_SITE_COLOR,
                zorder=6.2,
            )
        elif label_key == "shannan":
            text = ax.annotate(
                str(site.site_label),
                xy=(float(site.longitude), float(site.latitude)),
                xytext=(SHANNAN_LABEL_X, float(site.latitude) + SHANNAN_LABEL_Y_OFFSET),
                textcoords="data",
                ha="right",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=METEORO_SITE_COLOR,
                zorder=6.2,
            )
        else:
            text = ax.annotate(
                str(site.site_label),
                xy=(float(site.longitude), float(site.latitude)),
                xytext=(dx, 2.0),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=METEORO_SITE_COLOR,
                zorder=6.2,
            )
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def overlay_railway(ax, railway_geom, *, crs) -> None:
    lines = railbuf.flatten_lines(railway_geom)
    if not lines:
        return
    railway_series = railbuf.gpd.GeoSeries(lines, crs=crs).to_crs(railinspect.AXIS_CRS)
    for geom in railway_series:
        if geom is None or geom.is_empty:
            continue
        x, y = geom.xy
        ax.plot(x, y, color="white", linewidth=2.8, zorder=3.6)
        ax.plot(x, y, color="black", linewidth=1.5, linestyle=(0, (5, 3)), zorder=3.8)


def plot_extreme_map_content(
    ax,
    *,
    railway_geom,
    crs,
    background_mask: np.ndarray | None,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    sub_df: pd.DataFrame,
) -> None:
    if background_mask is not None:
        ax.pcolormesh(
            lon_plot,
            lat_plot,
            np.ma.masked_invalid(background_mask),
            cmap=ListedColormap([BACKGROUND_MASK_COLOR]),
            shading="auto",
            rasterized=True,
            zorder=0.0,
        )

    pf_df = sub_df.loc[sub_df["domain"].eq("Permafrost")].copy()
    npf_df = sub_df.loc[sub_df["domain"].eq("Non-Permafrost")].copy()
    if not pf_df.empty:
        ax.scatter(
            pf_df["longitude"],
            pf_df["latitude"],
            s=MAP_SPOT_SIZE,
            color=MAP_PF_SPOT_COLOR,
            edgecolors="none",
            linewidths=0.0,
            alpha=MAP_SPOT_ALPHA,
            rasterized=len(pf_df) > 3000,
            zorder=2.3,
        )
    if not npf_df.empty:
        ax.scatter(
            npf_df["longitude"],
            npf_df["latitude"],
            s=MAP_SPOT_SIZE,
            color=MAP_NPF_SPOT_COLOR,
            edgecolors="none",
            linewidths=0.0,
            alpha=MAP_SPOT_ALPHA,
            rasterized=len(npf_df) > 3000,
            zorder=2.6,
        )
    overlay_railway(ax, railway_geom, crs=crs)


def add_spatial_zoom_insets(
    ax,
    *,
    railway_context: dict[str, object],
    background_mask: np.ndarray | None,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    sub_df: pd.DataFrame,
    meteoro_sites: pd.DataFrame,
) -> None:
    for spec in MAP_ZOOM_SPECS:
        match = meteoro_sites.loc[meteoro_sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site_row = match.iloc[0]
        inset_ax = ax.inset_axes(spec["bounds"])
        plot_extreme_map_content(
            inset_ax,
            railway_geom=railway_context["network_geom"],
            crs=railway_context["crs"],
            background_mask=background_mask,
            lon_plot=lon_plot,
            lat_plot=lat_plot,
            sub_df=sub_df,
        )
        inset_ax.set_xlim(float(site_row["longitude"]) - float(spec["xpad"]), float(site_row["longitude"]) + float(spec["xpad"]))
        inset_ax.set_ylim(float(site_row["latitude"]) - float(spec["ypad"]), float(site_row["latitude"]) + float(spec["ypad"]))
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.grid(False)
        inset_ax.set_facecolor("none")
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
        add_meteoro_site_marker(
            inset_ax,
            site_row,
            show_label=True,
            marker_size=METEORO_INSET_SITE_SIZE,
        )


def plot_spatial_extreme_panel(
    ax,
    *,
    sub_df: pd.DataFrame,
    railway_context: dict[str, object],
    background_mask: np.ndarray | None,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    meteoro_sites: pd.DataFrame,
    map_xlim: tuple[float, float],
    map_ylim: tuple[float, float],
    title: str,
    legend_labels: tuple[str, str],
) -> None:
    plot_extreme_map_content(
        ax,
        railway_geom=railway_context["network_geom"],
        crs=railway_context["crs"],
        background_mask=background_mask,
        lon_plot=lon_plot,
        lat_plot=lat_plot,
        sub_df=sub_df,
    )
    configure_spatial_map_axis(ax, title=title, xlim=map_xlim, ylim=map_ylim)
    add_reference_meteoro_sites(ax, meteoro_sites)
    add_scale_bar_lonlat(ax, xlim=map_xlim, ylim=map_ylim)
    add_spatial_zoom_insets(
        ax,
        railway_context=railway_context,
        background_mask=background_mask,
        lon_plot=lon_plot,
        lat_plot=lat_plot,
        sub_df=sub_df,
        meteoro_sites=meteoro_sites,
    )

    handles = [
        Line2D([0], [0], color="black", linewidth=1.5, linestyle=(0, (5, 3)), label="Railway"),
        Line2D([0], [0], marker="o", linestyle="None", color="none", markerfacecolor=MAP_PF_SPOT_COLOR, markeredgecolor="none", alpha=MAP_SPOT_ALPHA, markersize=4.4, label=legend_labels[0]),
        Line2D([0], [0], marker="o", linestyle="None", color="none", markerfacecolor=MAP_NPF_SPOT_COLOR, markeredgecolor="none", alpha=MAP_SPOT_ALPHA, markersize=4.4, label=legend_labels[1]),
        Line2D([0], [0], marker="^", linestyle="None", color="none", markerfacecolor=METEORO_SITE_COLOR, markeredgecolor="none", markersize=6.0, label="Meteorological site"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        handletextpad=0.65,
        borderpad=0.40,
        labelspacing=0.35,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.78")
    legend.get_frame().set_linewidth(0.8)
    railbuf.apply_bold_legend(legend)


def resolve_map_background(
    base_dir: Path,
    *,
    cache_dir: Path,
    railway_shp: Path,
    crs_info_path: Path,
) -> dict[str, object]:
    dem_path = resolve_dem_raster_path(base_dir, cache_dir)
    meta_path = base_dir / "outputs" / "grad_rasters" / "grid_meta.npz"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing required grid metadata for hillshade: {meta_path}")

    meta = np.load(meta_path)
    res = float(meta["res"])
    gx0 = int(meta["gx0"])
    gy1 = int(meta["gy1"])
    nrows = int(meta["nrows"])
    ncols = int(meta["ncols"])
    stride = choose_stride(nrows, ncols, target_max=TARGET_MAP_PIXELS)
    map_source_crs = resolve_source_crs(crs_info_path, railway_shp)
    dem_mm = np.memmap(dem_path, dtype="float32", mode="r", shape=(nrows, ncols))
    dem_ds = np.asarray(dem_mm[::stride, ::stride], dtype=float)
    background_mask = np.flipud(build_elevation_mask(dem_ds))
    lon_plot, lat_plot = build_lonlat_mesh_from_grid(
        res=res,
        gx0=gx0,
        gy1=gy1,
        nrows=nrows,
        ncols=ncols,
        stride=stride,
        source_crs=map_source_crs,
    )
    extent_lonlat = resolve_lonlat_mesh_extent(lon_plot, lat_plot)
    return {
        "background_mask": background_mask,
        "lon_plot": lon_plot,
        "lat_plot": lat_plot,
        "extent_lonlat": extent_lonlat,
        "map_source_crs": map_source_crs.to_wkt(),
        "background_dem_path": dem_path,
        "background_stride": int(stride),
        "grid_meta_path": meta_path,
    }


def resolve_spatial_context(
    csv_path: Path,
    *,
    cache_dir: Path,
    railway_shp: Path,
    meteoro_shp: Path,
    crs_info_path: Path,
    buffer_width_km: float,
    chunksize: int,
) -> dict[str, object]:
    base_dir = csv_path.parent
    ensure_proj_lib()
    railway_context = railbuf.load_railway_context(railway_shp)
    meteoro_sites = railinspect.load_meteoro_sites(meteoro_shp)
    extent_source = None
    background_context = resolve_map_background(
        base_dir,
        cache_dir=cache_dir,
        railway_shp=railway_shp,
        crs_info_path=crs_info_path,
    )
    extent_lonlat = background_context["extent_lonlat"]
    map_xlim = (float(extent_lonlat[0]), float(extent_lonlat[1]))
    map_ylim = (float(MAP_LAT_LIMITS[0]), float(MAP_LAT_LIMITS[1]))

    suffix = str(buffer_width_km).replace(".", "p")
    extreme_map_cache = cache_dir / f"{FIG_BASENAME}_extreme_map_points_kde_hillshade_{suffix}km.csv.gz"
    extreme_map_df = railinspect.resolve_extreme_map_points(
        csv_path,
        cache_path=extreme_map_cache,
        railway_context=railway_context,
        buffer_width_km=float(buffer_width_km),
        chunksize=chunksize,
        xlim=map_xlim,
        ylim=map_ylim,
    )
    return {
        "railway_context": railway_context,
        "meteoro_sites": meteoro_sites,
        "map_xlim": map_xlim,
        "map_ylim": map_ylim,
        "extreme_map_df": extreme_map_df,
        "extent_lonlat": extent_lonlat,
        "extent_source": extent_source,
        "extreme_map_cache": extreme_map_cache,
        **background_context,
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Perma_Distr_map", "d_u", "grad_mag", "grad_mag_km"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "grad_mag_km" not in out.columns and "grad_mag" in out.columns:
        out["grad_mag_km"] = out["grad_mag"] * 1000.0
    elif "grad_mag" in out.columns and "grad_mag_km" in out.columns:
        fallback = out["grad_mag"] * 1000.0
        out["grad_mag_km"] = out["grad_mag_km"].where(np.isfinite(out["grad_mag_km"]), fallback)

    out["domain"] = np.where(
        out["Perma_Distr_map"] == 1,
        "pf",
        np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
    )
    if "d_u" in out.columns:
        out["abs_du"] = np.abs(out["d_u"])
    return out


def resolve_usecols(csv_path: Path) -> list[str]:
    available_cols = pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist()
    required = {"Perma_Distr_map", "d_u"}
    missing = sorted(required.difference(available_cols))
    if missing:
        raise RuntimeError(f"Missing required CSV column(s): {', '.join(missing)}")

    usecols = ["Perma_Distr_map", "d_u"]
    if "grad_mag_km" in available_cols:
        usecols.append("grad_mag_km")
    if "grad_mag" in available_cols:
        usecols.append("grad_mag")
    if "grad_mag_km" not in usecols and "grad_mag" not in usecols:
        raise RuntimeError("The pixel CSV does not contain grad_mag_km or grad_mag.")
    return usecols


def build_negative_du_sample(
    csv_path: Path,
    *,
    usecols: list[str],
    sample_per_domain: int,
    chunksize: int,
    seed: int,
) -> pd.DataFrame:
    log_step("building balanced PF/NPF sample for d_u <= 0")
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, pd.DataFrame | None] = {"pf": None, "npf": None}

    for idx, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        if idx == 1 or idx % 10 == 0:
            log_step(f"processed {idx} chunk(s)")

        chunk = engineer_features(chunk)
        for domain in ("pf", "npf"):
            sub = chunk.loc[
                chunk["domain"].eq(domain)
                & np.isfinite(chunk["d_u"])
                & np.isfinite(chunk["grad_mag_km"])
                & (chunk["d_u"] <= 0.0),
                ["Perma_Distr_map", "d_u", "abs_du", "grad_mag_km", "domain"],
            ].copy()
            if sub.empty:
                continue
            sub["_sample_key"] = rng.random(len(sub))
            keep = sub if reservoirs[domain] is None else pd.concat([reservoirs[domain], sub], ignore_index=True)
            if len(keep) > sample_per_domain:
                keep = keep.nsmallest(sample_per_domain, "_sample_key").reset_index(drop=True)
            reservoirs[domain] = keep

    parts: list[pd.DataFrame] = []
    for domain in ("pf", "npf"):
        sub = reservoirs[domain]
        if sub is None or sub.empty:
            raise RuntimeError(f"Sampling returned no rows for domain={domain}.")
        if len(sub) < sample_per_domain:
            log_step(f"requested {sample_per_domain} rows for {domain}, retained {len(sub)} available rows")
        parts.append(sub.drop(columns="_sample_key", errors="ignore"))

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def resolve_sample(
    csv_path: Path,
    *,
    cache_path: Path,
    sample_per_domain: int,
    chunksize: int,
) -> pd.DataFrame:
    required_cols = {"Perma_Distr_map", "d_u", "grad_mag_km", "domain", "abs_du"}
    if cache_path.exists():
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if required_cols.issubset(cache_cols):
            log_step(f"loading cached sample: {cache_path}")
            return engineer_features(pd.read_csv(cache_path))

    df = build_negative_du_sample(
        csv_path=csv_path,
        usecols=resolve_usecols(csv_path),
        sample_per_domain=sample_per_domain,
        chunksize=chunksize,
        seed=SEED,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, compression="gzip")
    log_step(f"saved sample cache: {cache_path}")
    return df


def clip_percentile(vals: pd.Series | np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> pd.Series:
    arr = pd.Series(pd.to_numeric(vals, errors="coerce"))
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return arr
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    return arr[(arr >= lo) & (arr <= hi)]


def make_binned_curve_with_uncertainty(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    q: int,
    uncertainty_scale: float = 0.1,
    bin_edges: np.ndarray | None = None,
) -> pd.DataFrame:
    tmp = df[[xcol, ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < q:
        return pd.DataFrame(columns=["x", "y", "ylo", "yhi", "n"])

    if bin_edges is None:
        tmp["bin"] = pd.qcut(tmp[xcol], q=q, duplicates="drop")
        out = (
            tmp.groupby("bin", observed=True)
            .agg(
                x=(xcol, "median"),
                y=(ycol, "mean"),
                ystd=(ycol, "std"),
                n=(ycol, "size"),
            )
            .reset_index(drop=True)
        )
    else:
        tmp["bin"] = pd.cut(tmp[xcol], bins=bin_edges, include_lowest=True, right=True)
        out = (
            tmp.groupby("bin", observed=True)
            .agg(
                y=(ycol, "mean"),
                ystd=(ycol, "std"),
                n=(ycol, "size"),
            )
            .reset_index()
        )
        if out.empty:
            return pd.DataFrame(columns=["x", "y", "ylo", "yhi", "n"])
        out["x"] = np.asarray(
            [float(iv.right) if pd.notna(iv) else np.nan for iv in out["bin"]],
            dtype=float,
        )
        out = out.drop(columns="bin")

    out["x"] = pd.to_numeric(out["x"], errors="coerce").astype(float)
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype(float)
    out["ystd"] = out["ystd"].fillna(0.0)
    band = uncertainty_scale * out["ystd"]
    out["ylo"] = out["y"] - band
    out["yhi"] = out["y"] + band
    out["ylo"] = pd.to_numeric(out["ylo"], errors="coerce").astype(float)
    out["yhi"] = pd.to_numeric(out["yhi"], errors="coerce").astype(float)
    return out[["x", "y", "ylo", "yhi", "n"]]


def upper_percentile(vals: pd.Series | np.ndarray, percentile: float = 95.0) -> float | None:
    arr = pd.Series(pd.to_numeric(vals, errors="coerce"))
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float(np.percentile(arr, percentile))


def split_xy_at_threshold(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    split_x: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if x.size == 0 or y.size == 0 or split_x is None or not np.isfinite(split_x):
        return x, y, np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if split_x <= x[0]:
        return np.array([], dtype=float), np.array([], dtype=float), x, y
    if split_x >= x[-1]:
        return x, y, np.array([], dtype=float), np.array([], dtype=float)

    y_split = float(np.interp(split_x, x, y))
    core_mask = x < split_x
    tail_mask = x > split_x
    core_x = np.concatenate([x[core_mask], [split_x]])
    core_y = np.concatenate([y[core_mask], [y_split]])
    tail_x = np.concatenate([[split_x], x[tail_mask]])
    tail_y = np.concatenate([[y_split], y[tail_mask]])
    return core_x, core_y, tail_x, tail_y


def split_curve_at_threshold(curve: pd.DataFrame, *, split_x: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = curve.sort_values("x").reset_index(drop=True)
    if ordered.empty or split_x is None or not np.isfinite(split_x):
        return ordered, ordered.iloc[0:0].copy()

    x = ordered["x"].to_numpy(dtype=float)
    if split_x <= x[0]:
        return ordered.iloc[0:0].copy(), ordered
    if split_x >= x[-1]:
        return ordered, ordered.iloc[0:0].copy()

    boundary = {"x": float(split_x)}
    for col in ("y", "ylo", "yhi"):
        boundary[col] = float(np.interp(split_x, x, ordered[col].to_numpy(dtype=float)))
    core = pd.concat([ordered.loc[ordered["x"] < split_x], pd.DataFrame([boundary])], ignore_index=True)
    tail = pd.concat([pd.DataFrame([boundary]), ordered.loc[ordered["x"] > split_x]], ignore_index=True)
    return core, tail


def annotate_percentile_position(
    ax,
    *,
    xpos: float | None,
    color: str,
    y_frac: float,
) -> None:
    if xpos is None:
        return

    xmin, xmax = ax.get_xlim()
    if xpos < xmin or xpos > xmax:
        return
    x_span = xmax - xmin
    if not np.isfinite(x_span) or x_span <= 0.0:
        return

    ax.axvline(
        xpos,
        color=color,
        linewidth=1.1,
        linestyle=(0, (4.0, 2.0)),
        alpha=0.95,
        zorder=4,
    )

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    x_text = xpos + 0.012 * x_span
    ha = "left"
    if x_text > xmax - 0.01 * x_span:
        x_text = xpos - 0.012 * x_span
        ha = "right"
    value_label = f"{xpos:.1f}" if abs(xpos) >= 1.0 else f"{xpos:.2f}"
    ax.text(
        x_text,
        y_frac,
        f"P95 = {value_label}",
        transform=trans,
        ha=ha,
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
        rotation=90,
        bbox=dict(boxstyle="round,pad=0.16", facecolor=(1.0, 1.0, 1.0, 0.7), edgecolor="none"),
        zorder=5,
    )


def annotate_upper_percentile(
    ax,
    vals: pd.Series | np.ndarray,
    *,
    color: str,
    y_frac: float,
) -> None:
    annotate_percentile_position(
        ax,
        xpos=upper_percentile(vals, percentile=95.0),
        color=color,
        y_frac=y_frac,
    )


def plot_hist_panel(
    ax,
    *,
    vals_pf: pd.Series,
    vals_npf: pd.Series,
    bins: np.ndarray,
    xlim: tuple[float, float],
    xlabel: str,
    title: str,
) -> None:
    if len(bins) < 2:
        style_open_axes(ax)
        return

    h_pf, edges = np.histogram(vals_pf, bins=bins, density=True)
    h_npf, _ = np.histogram(vals_npf, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    p95_pf = upper_percentile(vals_pf, percentile=95.0)
    p95_npf = upper_percentile(vals_npf, percentile=95.0)

    def draw_histogram_domain(hist_vals: np.ndarray, *, color: str, p95: float | None, zorder: float) -> None:
        core_color = soften_color(color, blend=HIST_CORE_BLEND)
        tail_color = color
        if p95 is None:
            ax.bar(
                centers,
                hist_vals,
                width=widths,
                color=core_color,
                alpha=HIST_CORE_ALPHA,
                edgecolor="none",
                align="center",
                zorder=zorder,
            )
            ax.plot(
                centers,
                hist_vals,
                color=color,
                linewidth=1.8,
                linestyle=HIST_LINESTYLE,
                zorder=zorder + 1.0,
            )
        else:
            core_mask = centers < p95
            tail_mask = ~core_mask
            if core_mask.any():
                ax.bar(
                    centers[core_mask],
                    hist_vals[core_mask],
                    width=widths[core_mask],
                    color=core_color,
                    alpha=HIST_CORE_ALPHA,
                    edgecolor="none",
                    align="center",
                    zorder=zorder,
                )
            if tail_mask.any():
                ax.bar(
                    centers[tail_mask],
                    hist_vals[tail_mask],
                    width=widths[tail_mask],
                    color=tail_color,
                    alpha=HIST_TAIL_ALPHA,
                    edgecolor="none",
                    align="center",
                    zorder=zorder + 0.05,
                )
            core_x, core_y, tail_x, tail_y = split_xy_at_threshold(centers, hist_vals, split_x=p95)
            if core_x.size:
                ax.plot(
                    core_x,
                    core_y,
                    color=color,
                    linewidth=1.8,
                    linestyle=HIST_LINESTYLE,
                    zorder=zorder + 1.0,
                )
            if tail_x.size:
                ax.plot(
                    tail_x,
                    tail_y,
                    color=tail_color,
                    linewidth=1.9,
                    linestyle="-",
                    zorder=zorder + 1.1,
                )

    draw_histogram_domain(h_pf, color=PF_COLOR, p95=p95_pf, zorder=2.0)
    draw_histogram_domain(h_npf, color=NPF_COLOR, p95=p95_npf, zorder=2.1)

    ymax = max(
        float(np.nanmax(h_pf)) if h_pf.size else 0.0,
        float(np.nanmax(h_npf)) if h_npf.size else 0.0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.08 * ymax if ymax > 0.0 else 1.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    style_open_axes(ax)
    apply_bold_nonlegend(ax)


def build_hist_bins(vals: pd.Series, n_bins: int, *, force_hi: float | None = None, force_lo: float | None = None) -> np.ndarray:
    vals = clip_percentile(vals, 0.5, 99.5)
    if vals.empty:
        return np.array([])
    lo = float(vals.min()) if force_lo is None else float(force_lo)
    hi = float(vals.max()) if force_hi is None else float(force_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])
    return np.linspace(lo, hi, int(n_bins) + 1)


def plot_relationship_panel(ax, sample_df: pd.DataFrame, *, q: int) -> None:
    curve_inputs = sample_df[["abs_du", "grad_mag_km", "domain"]].replace([np.inf, -np.inf], np.nan).dropna()
    if curve_inputs.empty:
        style_open_axes(ax)
        return

    y_all = clip_percentile(curve_inputs["grad_mag_km"], 0.5, 99.5)
    x_hi = 20.0
    y_hi = float(y_all.max()) if not y_all.empty else float(curve_inputs["grad_mag_km"].max())
    bin_edges = np.linspace(0.0, x_hi, max(6, int(q)) + 1)
    p95_positions: dict[str, float] = {}

    for domain, color in (("pf", PF_COLOR), ("npf", NPF_COLOR)):
        sub_all = curve_inputs.loc[curve_inputs["domain"] == domain, ["abs_du", "grad_mag_km"]].copy()
        if sub_all.empty:
            continue
        tail_start = upper_percentile(sub_all["abs_du"], percentile=95.0)
        if tail_start is None:
            continue
        p95_positions[domain] = tail_start

        sub = sub_all.loc[
            (sub_all["abs_du"] >= 0.0)
            & (sub_all["abs_du"] <= x_hi)
            & (sub_all["grad_mag_km"] <= y_hi)
        ].copy()
        if sub.empty:
            continue
        curve = make_binned_curve_with_uncertainty(
            sub,
            xcol="abs_du",
            ycol="grad_mag_km",
            q=q,
            uncertainty_scale=0.1,
            bin_edges=bin_edges,
        )
        if curve.empty:
            continue
        core_color = soften_color(color, blend=REL_CORE_BLEND)
        tail_color = color
        core_curve, tail_curve = split_curve_at_threshold(
            curve.loc[:, ["x", "y", "ylo", "yhi"]].copy(),
            split_x=tail_start,
        )

        if not core_curve.empty:
            ax.fill_between(
                core_curve["x"],
                core_curve["ylo"],
                core_curve["yhi"],
                color=core_color,
                alpha=REL_CORE_FILL_ALPHA,
                zorder=1,
            )
            ax.plot(
                core_curve["x"],
                core_curve["y"],
                color=color,
                linewidth=1.9,
                linestyle=HIST_LINESTYLE,
                zorder=2,
            )
        if not tail_curve.empty:
            ax.fill_between(
                tail_curve["x"],
                tail_curve["ylo"],
                tail_curve["yhi"],
                color=tail_color,
                alpha=REL_TAIL_FILL_ALPHA,
                zorder=3,
            )
            ax.plot(
                tail_curve["x"],
                tail_curve["y"],
                color=tail_color,
                linewidth=2.25,
                linestyle="-",
                zorder=4,
            )

    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 19.0)
    ax.set_title(r"$|d_u|$ vs $|\nabla d_u|$", fontweight="bold")
    ax.set_xlabel(r"$|d_u|$ for $d_u \leq 0$ (mm/yr)", fontweight="bold")
    ax.set_ylabel(r"mean $|\nabla d_u|$ (mm/yr/km)", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}" if abs(x) >= 1 else f"{x:.1f}"))
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15])
    style_open_axes(ax)
    apply_bold_nonlegend(ax)
    for domain, color in (("pf", PF_COLOR), ("npf", NPF_COLOR)):
        if domain in p95_positions:
            annotate_percentile_position(
                ax,
                xpos=p95_positions[domain],
                color=color,
                y_frac=0.30,
            )


def build_figure(
    sample_df: pd.DataFrame,
    *,
    spatial_context: dict[str, object],
    buffer_width_km: float,
    fig_dir: Path,
    curve_q: int,
) -> tuple[Path, Path]:
    neg_df = sample_df.loc[sample_df["d_u"] <= 0.0].copy()
    if neg_df.empty:
        raise RuntimeError("The sampled dataset contains no rows with d_u <= 0.")

    du_pf = clip_percentile(neg_df.loc[neg_df["domain"] == "pf", "abs_du"], 0.5, 99.5)
    du_npf = clip_percentile(neg_df.loc[neg_df["domain"] == "npf", "abs_du"], 0.5, 99.5)
    grad_pf = clip_percentile(neg_df.loc[neg_df["domain"] == "pf", "grad_mag_km"], 0.5, 99.5)
    grad_npf = clip_percentile(neg_df.loc[neg_df["domain"] == "npf", "grad_mag_km"], 0.5, 99.5)

    du_bins = build_hist_bins(pd.concat([du_pf, du_npf], ignore_index=True), HIST_BINS_DU, force_lo=0.0)
    grad_bins = build_hist_bins(pd.concat([grad_pf, grad_npf], ignore_index=True), HIST_BINS_GRAD, force_lo=0.0)
    if len(du_bins) < 2:
        raise RuntimeError("Could not build d_u histogram bins from the sampled negative-d_u data.")
    if len(grad_bins) < 2:
        raise RuntimeError("Could not build gradient histogram bins from the sampled negative-d_u data.")

    extreme_map_df = spatial_context["extreme_map_df"]
    du_extreme = extreme_map_df.loc[extreme_map_df["is_extreme_du"]].copy()
    grad_extreme = extreme_map_df.loc[extreme_map_df["is_extreme_grad"]].copy()

    fig = plt.figure(figsize=(14.6, 10.4))
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.0, 1.38],
        left=0.050,
        right=0.992,
        bottom=0.055,
        top=0.965,
        hspace=0.24,
        wspace=0.30,
    )

    ax_du = fig.add_subplot(gs[0, 0:2])
    ax_grad = fig.add_subplot(gs[0, 2:4])
    ax_rel = fig.add_subplot(gs[0, 4:6])
    ax_du_map = fig.add_subplot(gs[1, 0:3])
    ax_grad_map = fig.add_subplot(gs[1, 3:6])

    plot_hist_panel(
        ax_du,
        vals_pf=du_pf,
        vals_npf=du_npf,
        bins=du_bins,
        xlim=(0.0, 20.0),
        xlabel=r"$|d_u|$ for $d_u \leq 0$ (mm/yr)",
        title=r"$|d_u|$: Permafrost vs Non-Permafrost",
    )
    annotate_upper_percentile(
        ax_du,
        neg_df.loc[neg_df["domain"] == "pf", "abs_du"],
        color=PF_COLOR,
        y_frac=0.30,
    )
    annotate_upper_percentile(
        ax_du,
        neg_df.loc[neg_df["domain"] == "npf", "abs_du"],
        color=NPF_COLOR,
        y_frac=0.30,
    )
    ax_du.set_xticks([0, 5, 10, 15, 20])
    ax_du.set_yticks([0.00, 0.10, 0.20, 0.30])
    ax_du.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax_du.set_ylim(0.0, max(0.32, ax_du.get_ylim()[1]))
    add_domain_box(ax_du, loc="upper right")
    add_panel_label(ax_du, "A")

    plot_hist_panel(
        ax_grad,
        vals_pf=grad_pf,
        vals_npf=grad_npf,
        bins=grad_bins,
        xlim=(0.0, 40.0),
        xlabel=r"$|\nabla d_u|$ where $d_u \leq 0$ (mm/yr/km)",
        title=r"$|\nabla d_u|$: Permafrost vs Non-Permafrost",
    )
    annotate_upper_percentile(
        ax_grad,
        neg_df.loc[neg_df["domain"] == "pf", "grad_mag_km"],
        color=PF_COLOR,
        y_frac=0.30,
    )
    annotate_upper_percentile(
        ax_grad,
        neg_df.loc[neg_df["domain"] == "npf", "grad_mag_km"],
        color=NPF_COLOR,
        y_frac=0.30,
    )
    ax_grad.set_xticks([0, 10, 20, 30, 40])
    ax_grad.set_yticks([0.00, 0.10, 0.20])
    ax_grad.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
    ax_grad.set_ylim(0.0, max(0.22, ax_grad.get_ylim()[1]))
    add_domain_box(ax_grad, loc="upper right")
    add_panel_label(ax_grad, "B")

    plot_relationship_panel(ax_rel, neg_df, q=curve_q)
    add_panel_label(ax_rel, "C")

    add_relationship_box(ax_rel)

    plot_spatial_extreme_panel(
        ax_du_map,
        sub_df=du_extreme,
        railway_context=spatial_context["railway_context"],
        background_mask=spatial_context["background_mask"],
        lon_plot=spatial_context["lon_plot"],
        lat_plot=spatial_context["lat_plot"],
        meteoro_sites=spatial_context["meteoro_sites"],
        map_xlim=spatial_context["map_xlim"],
        map_ylim=spatial_context["map_ylim"],
        title=r"Spatial Distribution of Extreme $d_u$",
        legend_labels=(
            rf"Permafrost: $d_u<{railbuf.PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr",
            rf"Non-permafrost: $d_u<{railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr",
        ),
    )
    add_panel_label(ax_du_map, "D")

    plot_spatial_extreme_panel(
        ax_grad_map,
        sub_df=grad_extreme,
        railway_context=spatial_context["railway_context"],
        background_mask=spatial_context["background_mask"],
        lon_plot=spatial_context["lon_plot"],
        lat_plot=spatial_context["lat_plot"],
        meteoro_sites=spatial_context["meteoro_sites"],
        map_xlim=spatial_context["map_xlim"],
        map_ylim=spatial_context["map_ylim"],
        title=r"Spatial Distribution of Extreme $|\nabla d_u|$",
        legend_labels=(
            rf"Permafrost: $|\nabla d_u|>{railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km",
            rf"Non-permafrost: $|\nabla d_u|>{railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km",
        ),
    )
    add_panel_label(ax_grad_map, "E")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PF vs NPF d_u and d_u-gradient summaries plus railway-context spatial maps of extreme deformation."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-per-domain", type=int, default=SAMPLE_PER_DOMAIN_DEFAULT)
    parser.add_argument("--curve-q", type=int, default=CURVE_Q_DEFAULT)
    parser.add_argument("--buffer-width-km", type=float, default=DEFAULT_BUFFER_WIDTH_KM)
    parser.add_argument("--railway-shp", type=Path, default=None)
    parser.add_argument("--meteoro-shp", type=Path, default=None)
    parser.add_argument("--crs-info", type=Path, default=None)
    args = parser.parse_args()

    configure_style()

    base_dir = args.base_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    )
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    for path in (fig_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    railway_shp = (
        args.railway_shp.resolve()
        if args.railway_shp is not None
        else (base_dir / "human_features" / "qtec_railway_clip.shp")
    )
    meteoro_shp = (
        args.meteoro_shp.resolve()
        if args.meteoro_shp is not None
        else (base_dir / "human_features" / "qtec_meteoro_station_sites.shp")
    )
    crs_info_path = (
        args.crs_info.resolve()
        if args.crs_info is not None
        else (base_dir / "crs_info.txt")
    )
    required = [csv_path, railway_shp, meteoro_shp]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(str(p) for p in missing))

    buffer_width_km = float(args.buffer_width_km)
    if not np.isfinite(buffer_width_km) or buffer_width_km <= 0.0:
        raise RuntimeError("buffer_width_km must be a positive finite number.")

    cache_path = cache_dir / f"{FIG_BASENAME}_sample_n{int(args.sample_per_domain)}.csv.gz"
    sample_df = resolve_sample(
        csv_path,
        cache_path=cache_path,
        sample_per_domain=int(args.sample_per_domain),
        chunksize=int(args.chunksize),
    )
    spatial_context = resolve_spatial_context(
        csv_path,
        cache_dir=cache_dir,
        railway_shp=railway_shp,
        meteoro_shp=meteoro_shp,
        crs_info_path=crs_info_path,
        buffer_width_km=buffer_width_km,
        chunksize=int(args.chunksize),
    )

    fig_png, fig_pdf = build_figure(
        sample_df,
        spatial_context=spatial_context,
        buffer_width_km=buffer_width_km,
        fig_dir=fig_dir,
        curve_q=max(6, int(args.curve_q)),
    )

    meta_path = cache_dir / f"{FIG_BASENAME}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "figure_png": str(fig_png),
                "figure_pdf": str(fig_pdf),
                "sample_cache": str(cache_path),
                "sample_per_domain": int(args.sample_per_domain),
                "curve_q": int(args.curve_q),
                "buffer_width_km": buffer_width_km,
                "extreme_map_cache": str(spatial_context["extreme_map_cache"]),
                "background_dem_path": str(spatial_context["background_dem_path"]),
                "background_stride": int(spatial_context["background_stride"]),
                "grid_meta_path": str(spatial_context["grid_meta_path"]),
                "map_source_crs": spatial_context["map_source_crs"],
                "lonlat_extent_source": str(spatial_context["extent_source"]) if spatial_context["extent_source"] is not None else None,
                "n_pf": int((sample_df["domain"] == "pf").sum()),
                "n_npf": int((sample_df["domain"] == "npf").sum()),
                "n_extreme_du_map_points": int(spatial_context["extreme_map_df"]["is_extreme_du"].sum()),
                "n_extreme_grad_map_points": int(spatial_context["extreme_map_df"]["is_extreme_grad"].sum()),
            },
            indent=2,
        )
    )

    log_step(f"saved figure PNG: {fig_png}")
    log_step(f"saved figure PDF: {fig_pdf}")
    log_step(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
