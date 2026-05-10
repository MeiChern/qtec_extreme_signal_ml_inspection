#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/figure4_regional_deformation_context.py
# Renamed package path: code/original_project_helpers/regional_deformation_context_helpers.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

"""
Figure 4 (revised layout for the short paper)

Layout:
Top row
A. Composite d_u panel: longitudinal profile + map + latitudinal profile
B. Composite spatial-gradient panel: longitudinal profile + map + latitudinal profile

Bottom row
C. Combined basin-wise boxplots of d_u and spatial gradient magnitude
D. Mirrored Permafrost / Non-Permafrost negative-d_u histogram
E. Negative |d_u| vs spatial gradient magnitude line plot for Permafrost and Non-Permafrost

Expected project layout:
  .
  ├── df_all_data_with_wright_du_plus_grad.csv
  └── outputs/
      ├── grad_rasters/
      │   ├── grid_meta.npz
      │   ├── du_f32.memmap
      │   └── gradmag_f32.memmap
      └── gradient_driver_analysis/
          └── tables/pixel_model_sample.parquet         # optional legacy sample cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SEED = 42
CHUNKSIZE = 400_000
HOTSPOT_Q_DEFAULT = 0.95
DESIRED_SAMPLE_PER_DOMAIN = {"pf": 180_000, "npf": 180_000}
FIGURE_NAME = "Figure4_regional_deformation_context_v2"
TARGET_MAX_PIXELS = 900

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

# -----------------------------------------------------------------------------
# Styling / coordinate helpers
# -----------------------------------------------------------------------------
PF_COLOR = "#9FD3E6"   # icy blue
NPF_COLOR = "#B8DDB1"  # shallow green
DOMAIN_LABELS = {"pf": "Permafrost", "npf": "Non-Permafrost"}
DU_PROFILE_COLOR = "#1E78AC"
GMAG_PROFILE_COLOR = "#82550D"

MAP_LABEL_X = "Longitude (°)"
MAP_LABEL_Y = "Latitude (°)"


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


def meridional_degree_length_m(lat_deg: float | np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=float))
    return (
        111132.92
        - 559.82 * np.cos(2.0 * lat_rad)
        + 1.175 * np.cos(4.0 * lat_rad)
        - 0.0023 * np.cos(6.0 * lat_rad)
    )


def geographic_aspect_ratio(lat_min: float, lat_max: float) -> float | str:
    mean_lat = 0.5 * (float(lat_min) + float(lat_max))
    cos_lat = float(np.cos(np.deg2rad(mean_lat)))
    if not np.isfinite(cos_lat) or abs(cos_lat) < 1e-6:
        return "auto"
    return 1.0 / cos_lat


def build_meridional_lat_ticks(
    lat_min: float,
    lat_max: float,
    max_ticks: int = 4,
) -> tuple[np.ndarray, list[str]]:
    if not np.isfinite(lat_min) or not np.isfinite(lat_max) or lat_max <= lat_min:
        return np.array([], dtype=float), []

    lat_grid = np.linspace(lat_min, lat_max, 1024)
    seg_m = 0.5 * (
        meridional_degree_length_m(lat_grid[1:]) + meridional_degree_length_m(lat_grid[:-1])
    ) * np.diff(lat_grid)
    cum_km = np.concatenate([[0.0], np.cumsum(seg_m) / 1000.0])

    dist_targets = np.linspace(0.0, float(cum_km[-1]), max(2, int(max_ticks)))
    tick_lats = np.interp(dist_targets, cum_km, lat_grid)

    labels = []
    for lat, dist in zip(tick_lats, dist_targets):
        hemi = "N" if lat >= 0 else "S"
        labels.append(f"{abs(lat):.1f}°{hemi}\n{dist:.0f} km")
    return tick_lats, labels


def format_lonlat_axes(
    ax,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    extent_lonlat: Optional[list[float]] = None,
    use_meridional_lat_ticks: bool = False,
    fill_plot_region: bool = True,
) -> None:
    ax.set_xlabel(MAP_LABEL_X if show_xlabel else "")
    ax.set_ylabel(MAP_LABEL_Y if show_ylabel else "")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    if use_meridional_lat_ticks and extent_lonlat is not None:
        _, _, lat_min, lat_max = extent_lonlat
        yticks, ylabels = build_meridional_lat_ticks(lat_min, lat_max)
        if len(yticks):
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
        if show_ylabel:
            ax.set_ylabel("Latitude (° / km)")
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    if extent_lonlat is not None:
        if fill_plot_region:
            # Critical for matching A with B and D with E in visible plotting width
            ax.set_aspect("auto")
            ax.set_anchor("SW")
        else:
            ax.set_aspect(
                geographic_aspect_ratio(extent_lonlat[2], extent_lonlat[3]),
                adjustable="box",
            )

    style_open_axes(ax)


def add_map_inset_colorbar(ax, mappable, label: str, rect: list[float]) -> None:
    cax = ax.inset_axes(rect, transform=ax.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)

    cb = plt.colorbar(mappable, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.set_label(label, fontsize=8, fontweight="bold")


def resolve_lonlat_extent(
    csv_path: Path,
    cache_json: Path,
    *,
    min_e: float,
    max_n: float,
    nrows: int,
    ncols: int,
    res: float,
    chunksize: int = CHUNKSIZE,
    max_points: int = 80000,
) -> list[float]:
    """
    Approximate raster lon/lat extent from CSV point coordinates using an affine fit:
      lon ~ a0 + a1*easting + a2*northing
      lat ~ b0 + b1*easting + b2*northing
    """
    if cache_json.exists():
        try:
            payload = json.loads(cache_json.read_text())
            return payload["extent_lonlat"]
        except Exception:
            pass

    pts = []
    usecols = ["easting", "northing", "longitude", "latitude"]
    n_kept = 0

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(e) & np.isfinite(n) & np.isfinite(lon) & np.isfinite(lat)
        if not ok.any():
            continue

        idx = np.flatnonzero(ok)
        step = max(1, len(idx) // 6000)
        idx = idx[::step]
        keep = np.column_stack([e[idx], n[idx], lon[idx], lat[idx]])
        pts.append(keep)
        n_kept += len(keep)
        if n_kept >= max_points:
            break

    if not pts:
        raise RuntimeError("Could not infer longitude/latitude extent from CSV.")

    pts = np.vstack(pts)[:max_points]
    e = pts[:, 0]
    n = pts[:, 1]
    lon = pts[:, 2]
    lat = pts[:, 3]

    A = np.column_stack([np.ones_like(e), e, n])
    coef_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    coef_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)

    def pred_xy(e0: float, n0: float) -> tuple[float, float]:
        row = np.array([1.0, e0, n0], dtype=float)
        return float(row @ coef_lon), float(row @ coef_lat)

    left = min_e
    right = min_e + ncols * res
    top = max_n
    bottom = max_n - nrows * res

    corners = [
        pred_xy(left, top),
        pred_xy(right, top),
        pred_xy(left, bottom),
        pred_xy(right, bottom),
    ]
    lon_vals = [c[0] for c in corners]
    lat_vals = [c[1] for c in corners]

    extent_lonlat = [min(lon_vals), max(lon_vals), min(lat_vals), max(lat_vals)]
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps({"extent_lonlat": extent_lonlat}, indent=2))
    return extent_lonlat


def make_binned_curve_with_uncertainty(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    q: int = 20,
    uncertainty_scale: float = 0.2,
) -> pd.DataFrame:
    tmp = df[[xcol, ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < q:
        return pd.DataFrame(columns=["x", "y", "ylo", "yhi", "n"])

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
    out["ystd"] = out["ystd"].fillna(0.0)
    out["ylo"] = out["y"] - uncertainty_scale * out["ystd"]
    out["yhi"] = out["y"] + uncertainty_scale * out["ystd"]
    return out[["x", "y", "ylo", "yhi", "n"]]


def draw_horizontal_boxplot(ax, data: list[np.ndarray], labels: list[str], colors: list[tuple]) -> None:
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.4),
    )
    for i, color in enumerate(colors):
        bp["boxes"][i].set_facecolor((1.0, 1.0, 1.0, 0.85))
        bp["boxes"][i].set_edgecolor(color)
        bp["medians"][i].set_color(color)
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)

    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    style_open_axes(ax)


def summarize_spatial_band_profile(
    arr: np.ndarray,
    *,
    band_axis: int,
    coord_min: float,
    coord_max: float,
    max_bins: int = 64,
) -> pd.DataFrame:
    vals = np.asarray(arr, dtype=float)
    if vals.ndim != 2:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    n_bands = int(vals.shape[band_axis])
    if n_bands == 0:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    n_bins = max(2, min(int(max_bins), n_bands))
    coords = np.linspace(coord_min, coord_max, n_bands)
    rows = []

    for idx in np.array_split(np.arange(n_bands), n_bins):
        if idx.size == 0:
            continue
        subset = vals[idx, :] if band_axis == 0 else vals[:, idx]
        finite = subset[np.isfinite(subset)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "coord": float(np.nanmean(coords[idx])),
                "mean": float(np.nanmean(finite)),
                "std": float(np.nanstd(finite)),
                "n": int(finite.size),
            }
        )

    return pd.DataFrame(rows, columns=["coord", "mean", "std", "n"])


def make_profile_coord_formatter(coord_kind: str) -> FuncFormatter:
    if coord_kind == "lat":
        return FuncFormatter(lambda x, _: f"{abs(x):.0f}°{'N' if x >= 0 else 'S'}")
    return FuncFormatter(lambda x, _: f"{abs(x):.0f}°{'E' if x >= 0 else 'W'}")


def style_single_profile_axis(ax, *, color: str, grid_axis: str | None = None) -> None:
    style_open_axes(ax)
    ax.spines["left"].set_color(color)
    ax.spines["bottom"].set_color(color)
    ax.tick_params(axis="both", colors=color)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.15, linewidth=0.5)


def use_right_y_axis(ax, *, color: str, labelpad: float = 26.0) -> None:
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(color)
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        colors=color,
    )
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_horizontalalignment("left")
    ax.yaxis.label.set_verticalalignment("center")
    ax.yaxis.labelpad = labelpad


def profile_spans_zero(profile: pd.DataFrame) -> bool:
    if profile.empty:
        return False
    half_std = 0.5 * profile["std"].to_numpy(dtype=float)
    lo = np.nanmin(profile["mean"].to_numpy(dtype=float) - half_std)
    hi = np.nanmax(profile["mean"].to_numpy(dtype=float) + half_std)
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi)


def plot_top_profile(
    ax,
    profile: pd.DataFrame,
    *,
    color: str,
    ylabel: str,
    title: Optional[str] = None,
    coord_limits: Optional[tuple[float, float]] = None,
    pad_right: float = 0.0,
):
    if not profile.empty:
        half_std = 0.5 * profile["std"].to_numpy(dtype=float)
        coord = profile["coord"].to_numpy(dtype=float)
        mean = profile["mean"].to_numpy(dtype=float)
        ax.fill_between(
            coord,
            mean - half_std,
            mean + half_std,
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        ax.plot(
            coord,
            mean,
            color=color,
            linewidth=1.8,
        )
    if title is not None:
        ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, color=color, fontweight="bold")
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.margins(x=0.0)
    if coord_limits is not None:
        lo, hi = coord_limits
        ax.set_xlim(lo, hi + float(pad_right))
    style_single_profile_axis(ax, color=color, grid_axis="y")


def plot_side_profile(
    ax,
    profile: pd.DataFrame,
    *,
    color: str,
    xlabel: str,
    coord_limits: Optional[tuple[float, float]] = None,
):
    if not profile.empty:
        half_std = 0.5 * profile["std"].to_numpy(dtype=float)
        coord = profile["coord"].to_numpy(dtype=float)
        mean = profile["mean"].to_numpy(dtype=float)
        ax.fill_betweenx(
            coord,
            mean - half_std,
            mean + half_std,
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        ax.plot(
            mean,
            coord,
            color=color,
            linewidth=1.8,
        )
    ax.set_xlabel(xlabel, color=color, fontweight="bold")
    ax.tick_params(axis="y", labelleft=False, left=False)
    ax.margins(y=0.0)
    if coord_limits is not None:
        ax.set_ylim(*coord_limits)
    style_single_profile_axis(ax, color=color, grid_axis="x")


def style_boxplot_group(bp, colors: list[tuple], hatch: str) -> None:
    for i, color in enumerate(colors):
        bp["boxes"][i].set_facecolor((1.0, 1.0, 1.0, 0.92))
        bp["boxes"][i].set_edgecolor(color)
        bp["boxes"][i].set_hatch(hatch)
        bp["boxes"][i].set_linewidth(1.4)
        bp["medians"][i].set_color(color)
        bp["medians"][i].set_linewidth(1.5)
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)


def draw_combined_basin_boxplot(
    ax,
    du_data: list[np.ndarray],
    grad_data: list[np.ndarray],
    labels: list[str],
    colors: list[tuple],
):
    if not labels:
        style_open_axes(ax)
        return None

    base_pos = np.arange(len(labels), 0, -1, dtype=float)
    du_pos = base_pos + 0.18
    grad_pos = base_pos - 0.18

    bp_du = ax.boxplot(
        du_data,
        vert=False,
        positions=du_pos,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.4),
    )
    ax_grad = ax.twiny()
    bp_grad = ax_grad.boxplot(
        grad_data,
        vert=False,
        positions=grad_pos,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.4),
    )

    style_boxplot_group(bp_du, colors, hatch="...")
    style_boxplot_group(bp_grad, colors, hatch="////")

    ax.set_yticks(base_pos)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.35, len(labels) + 0.65)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.set_xlabel(r"$d_u$ (mm/yr)", color="k", fontweight="bold")
    ax.tick_params(axis="x", colors="k")
    ax.spines["bottom"].set_color("k")
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)

    ax_grad.set_xlabel(r"$|\nabla d_u|$ (mm/yr/km)", color="k", fontweight="bold")
    ax_grad.tick_params(axis="x", colors="k")
    ax_grad.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    ax_grad.spines["top"].set_color("k")
    ax_grad.spines["bottom"].set_visible(False)
    ax_grad.spines["left"].set_visible(False)
    ax_grad.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=(1.0, 1.0, 1.0, 0.92), edgecolor="0.35", hatch="...", label=r"$d_u$"),
        Patch(facecolor=(1.0, 1.0, 1.0, 0.92), edgecolor="0.35", hatch="////", label=r"$|\nabla d_u|$"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    return ax_grad


def make_composite_triptych(fig, parent_spec, *, map_width: float = 4.8, side_width: float = 2.05):
    gs_local = parent_spec.subgridspec(
        2,
        2,
        height_ratios=[1.0, 5.0],
        width_ratios=[map_width, side_width],
        hspace=0.0,
        wspace=0.0,
    )

    ax_map = fig.add_subplot(gs_local[1, 0])
    ax_top = fig.add_subplot(gs_local[0, 0], sharex=ax_map)
    ax_side = fig.add_subplot(gs_local[1, 1], sharey=ax_map)
    return ax_top, ax_map, ax_side


def use_map_spines_for_marginals(ax_top, ax_map, ax_side) -> None:
    # A/D use the map's top spine as the shared boundary
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )

    ax_map.spines["top"].set_visible(True)
    ax_map.tick_params(
        axis="x",
        which="both",
        top=False,
        labeltop=False,
    )

    # C/F use the map's right spine as the shared boundary
    ax_side.spines["left"].set_visible(False)
    ax_side.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    ax_map.spines["right"].set_visible(True)
    ax_map.tick_params(
        axis="y",
        which="both",
        right=False,
        labelright=False,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def robust_clip(arr: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def centered_clip(
    arr: np.ndarray,
    center: float = 0.0,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> tuple[float, float]:
    lo, hi = robust_clip(arr, p_lo=p_lo, p_hi=p_hi)
    span = max(abs(lo - center), abs(hi - center))
    return center - span, center + span


def choose_stride(nrows: int, ncols: int, target_max: int = TARGET_MAX_PIXELS) -> int:
    return max(1, int(max(nrows, ncols) / target_max))


def open_memmap(path: Path, dtype: str, mode: str, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def en_to_rc(
    easting: np.ndarray,
    northing: np.ndarray,
    *,
    res: float,
    gx0: int,
    gy1: int,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.rint(np.asarray(easting, dtype=np.float64) / res).astype(np.int64)
    gy = np.rint(np.asarray(northing, dtype=np.float64) / res).astype(np.int64)
    col = (gx - gx0).astype(np.int32)
    row = (gy1 - gy).astype(np.int32)
    return row, col


def get_extent(min_e: float, max_n: float, nrows: int, ncols: int, res: float) -> list[float]:
    left = min_e
    right = min_e + ncols * res
    bottom = max_n - nrows * res
    top = max_n
    return [left, right, bottom, top]


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", alpha=0.12),
    )


def engineer_features_min(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Perma_Distr_map" in out.columns:
        out["Perma_Distr_map"] = pd.to_numeric(out["Perma_Distr_map"], errors="coerce")
    if "d_u" in out.columns:
        out["d_u"] = pd.to_numeric(out["d_u"], errors="coerce")
        out["abs_du"] = out["d_u"].abs()
    if "grad_mag" in out.columns:
        out["grad_mag"] = pd.to_numeric(out["grad_mag"], errors="coerce")
        out["grad_mag_km"] = out["grad_mag"] * 1000.0
        out["log_grad_mag"] = np.log1p(out["grad_mag"].clip(lower=0))
    if "Main-Basin" in out.columns:
        out["Main-Basin"] = out["Main-Basin"].astype("string").fillna("missing")
    else:
        out["Main-Basin"] = pd.Series("missing", index=out.index, dtype="string")
    out["domain"] = np.where(
        out["Perma_Distr_map"] == 1,
        "pf",
        np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
    )
    return out


def clip_series(vals: pd.Series, p_lo: float = 0.5, p_hi: float = 99.5) -> pd.Series:
    arr = pd.to_numeric(vals, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return arr
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    return arr[(arr >= lo) & (arr <= hi)]


def make_binned_curve(df: pd.DataFrame, xcol: str, ycol: str, q: int = 20) -> pd.DataFrame:
    tmp = df[[xcol, ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.dropna()
    if len(tmp) < q:
        return pd.DataFrame(columns=["x", "y", "n"])
    tmp["bin"] = pd.qcut(tmp[xcol], q=q, duplicates="drop")
    out = (
        tmp.groupby("bin", observed=True)
        .agg(x=(xcol, "median"), y=(ycol, "mean"), n=(ycol, "size"))
        .reset_index(drop=True)
    )
    return out


def maybe_read_legacy_sample(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        return engineer_features_min(df)
    except Exception as exc:
        print(f"Legacy sample exists but could not be read from {path}: {exc}")
        return None


def summarize_domain_counts(csv_path: Path, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = ["Perma_Distr_map", "d_u", "grad_mag"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = engineer_features_min(chunk)
        valid_any = chunk["d_u"].notna() | chunk["grad_mag"].notna()
        tmp = (
            chunk.loc[valid_any, "domain"]
            .value_counts(dropna=False)
            .rename_axis("domain")
            .reset_index(name="n_rows")
        )
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby("domain", as_index=False)["n_rows"].sum()
    return out.sort_values("domain").reset_index(drop=True)


def build_stratified_sample(
    csv_path: Path,
    usecols: list[str],
    desired_per_domain: dict[str, int],
    chunksize: int = CHUNKSIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    counts = summarize_domain_counts(csv_path, chunksize=chunksize)
    count_map = dict(zip(counts["domain"], counts["n_rows"]))
    prob_map = {
        k: min(1.0, 1.15 * desired_per_domain[k] / max(count_map.get(k, 1), 1))
        for k in desired_per_domain
    }
    print("Sampling probabilities:", prob_map)

    rng_local = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = engineer_features_min(chunk)
        valid_any = chunk["d_u"].notna() | chunk["grad_mag"].notna()
        chunk = chunk.loc[valid_any].copy()
        if chunk.empty:
            continue

        masks = []
        for domain_name, prob in prob_map.items():
            submask = chunk["domain"].eq(domain_name).to_numpy()
            if not submask.any():
                continue
            draw = rng_local.random(submask.sum()) < prob
            tmpmask = np.zeros(len(chunk), dtype=bool)
            tmpmask[np.flatnonzero(submask)] = draw
            masks.append(tmpmask)

        if masks:
            take = np.logical_or.reduce(masks)
            sampled = chunk.loc[take].copy()
            if not sampled.empty:
                parts.append(sampled)

    if not parts:
        raise RuntimeError("Sampling returned no rows. Check the input CSV and columns.")

    df = pd.concat(parts, ignore_index=True)
    out_parts = []
    for domain_name, quota in desired_per_domain.items():
        sub = df.loc[df["domain"] == domain_name].copy()
        if len(sub) > quota:
            sub = sub.sample(quota, random_state=seed)
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def resolve_sample(
    csv_path: Path,
    cache_csv_gz: Path,
    legacy_sample_path: Path,
    sample_per_domain: int,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    if cache_csv_gz.exists():
        print(f"Reading cached figure sample: {cache_csv_gz}")
        df = pd.read_csv(cache_csv_gz)
        if "Main-Basin" in df.columns:
            return engineer_features_min(df)
        print("Cached figure sample is missing Main-Basin; rebuilding cache.")

    legacy_df = maybe_read_legacy_sample(legacy_sample_path)
    if legacy_df is not None:
        needed = [
            c
            for c in ["Perma_Distr_map", "Main-Basin", "d_u", "grad_mag", "domain", "abs_du", "log_grad_mag"]
            if c in legacy_df.columns
        ]
        legacy_df = legacy_df[needed].copy()
        if "Main-Basin" not in legacy_df.columns:
            legacy_df["Main-Basin"] = "missing"
        cache_csv_gz.parent.mkdir(parents=True, exist_ok=True)
        legacy_df.to_csv(cache_csv_gz, index=False, compression="gzip")
        print(f"Copied legacy sample into figure cache: {cache_csv_gz}")
        return engineer_features_min(legacy_df)

    print("Building a new stratified figure sample from the full CSV ...")
    usecols = ["Perma_Distr_map", "Main-Basin", "d_u", "grad_mag"]
    sample_df = build_stratified_sample(
        csv_path=csv_path,
        usecols=usecols,
        desired_per_domain={"pf": sample_per_domain, "npf": sample_per_domain},
        chunksize=chunksize,
        seed=SEED,
    )
    cache_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(cache_csv_gz, index=False, compression="gzip")
    print(f"Saved new figure sample: {cache_csv_gz}")
    return sample_df


def resolve_hotspot_threshold(
    csv_path: Path,
    q: float,
    legacy_summary_path: Path,
    cache_json: Path,
    chunksize: int = CHUNKSIZE,
) -> float:
    if legacy_summary_path.exists():
        try:
            summary = pd.read_csv(legacy_summary_path)
            thr = float(summary.loc[0, "strong_threshold_grad_mag"])
            print(f"Using legacy hotspot threshold from {legacy_summary_path}: {thr:.6f}")
            return thr
        except Exception as exc:
            print(f"Could not read legacy hotspot summary at {legacy_summary_path}: {exc}")

    if cache_json.exists():
        payload = json.loads(cache_json.read_text())
        thr = float(payload["strong_threshold_grad_mag"])
        print(f"Using cached hotspot threshold from {cache_json}: {thr:.6f}")
        return thr

    print(f"Computing exact permafrost grad_mag quantile from CSV (q={q:.3f}) ...")
    parts: list[np.ndarray] = []
    usecols = ["Perma_Distr_map", "grad_mag"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy() == 1
        vals = pd.to_numeric(chunk["grad_mag"], errors="coerce").to_numpy(dtype=np.float32)
        mask = pf & np.isfinite(vals)
        if mask.any():
            parts.append(vals[mask])

    if not parts:
        raise RuntimeError("No valid permafrost grad_mag values found. Cannot compute hotspot threshold.")

    vals = np.concatenate(parts)
    thr = float(np.nanquantile(vals, q))
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps({"q": q, "strong_threshold_grad_mag": thr}, indent=2))
    print(f"Computed hotspot threshold: {thr:.6f}")
    return thr


def resolve_permafrost_raster(
    csv_path: Path,
    out_path: Path,
    legacy_path: Path,
    *,
    nrows: int,
    ncols: int,
    res: float,
    gx0: int,
    gy1: int,
    chunksize: int = CHUNKSIZE,
) -> Path:
    if legacy_path.exists():
        print(f"Using legacy permafrost raster: {legacy_path}")
        return legacy_path
    if out_path.exists():
        print(f"Using cached permafrost raster: {out_path}")
        return out_path

    print("Building permafrost raster cache from CSV ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(out_path, dtype="uint8", mode="w+", shape=(nrows, ncols))
    mm[:] = np.uint8(255)

    usecols = ["easting", "northing", "Perma_Distr_map"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy()
        row, col = en_to_rc(
            chunk["easting"].to_numpy(),
            chunk["northing"].to_numpy(),
            res=res,
            gx0=gx0,
            gy1=gy1,
        )
        ok = (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols) & np.isfinite(pf)
        if ok.any():
            mm[row[ok], col[ok]] = pf[ok].astype(np.uint8)

    mm.flush()
    print(f"Wrote permafrost raster cache: {out_path}")
    return out_path


def select_top_basins(sample_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    pf = sample_df.loc[sample_df["domain"] == "pf"].copy()
    if pf.empty or "Main-Basin" not in pf.columns:
        return []

    vc = pf["Main-Basin"].fillna("missing").astype(str)
    vc = vc.loc[vc != "missing"].value_counts()
    basins = vc.head(top_n).index.tolist()

    if not basins:
        basins = (
            pf["Main-Basin"]
            .fillna("missing")
            .astype(str)
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
    return basins


def resolve_main_basin_raster(
    csv_path: Path,
    out_path: Path,
    code_json_path: Path,
    selected_basins: list[str],
    *,
    nrows: int,
    ncols: int,
    res: float,
    gx0: int,
    gy1: int,
    chunksize: int = CHUNKSIZE,
) -> tuple[Path, dict]:
    selected_basins = [str(b) for b in selected_basins]
    other_label = "Other"
    nodata = -1

    expected_meta = {
        "selected_basins": selected_basins,
        "other_label": other_label,
        "nodata": nodata,
    }

    if out_path.exists() and code_json_path.exists():
        try:
            meta = json.loads(code_json_path.read_text())
            if meta.get("selected_basins") == selected_basins:
                print(f"Using cached main-basin raster: {out_path}")
                return out_path, meta
        except Exception:
            pass

    print("Building main-basin raster cache from CSV ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = open_memmap(out_path, dtype="int16", mode="w+", shape=(nrows, ncols))
    mm[:] = np.int16(nodata)

    code_map = {b: i for i, b in enumerate(selected_basins)}
    other_code = len(selected_basins)

    usecols = ["easting", "northing", "Perma_Distr_map", "Main-Basin"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy()
        basin = chunk["Main-Basin"].astype("string").fillna("missing").astype(str).to_numpy()

        row, col = en_to_rc(
            chunk["easting"].to_numpy(),
            chunk["northing"].to_numpy(),
            res=res,
            gx0=gx0,
            gy1=gy1,
        )
        ok = (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols) & (pf == 1)
        if not ok.any():
            continue

        rr = row[ok]
        cc = col[ok]
        bb = basin[ok]
        codes = np.array([code_map.get(x, other_code) for x in bb], dtype=np.int16)
        mm[rr, cc] = codes

    mm.flush()
    meta = {
        "selected_basins": selected_basins,
        "other_label": other_label,
        "other_code": other_code,
        "nodata": nodata,
    }
    code_json_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote main-basin raster cache: {out_path}")
    return out_path, meta


def mirrored_density_hist(ax, vals_pf: pd.Series, vals_npf: pd.Series) -> None:
    vals_pf = clip_series(vals_pf, 0.5, 99.5)
    vals_npf = clip_series(vals_npf, 0.5, 99.5)
    all_vals = pd.concat([vals_pf, vals_npf], ignore_index=True)
    if all_vals.empty:
        return

    lo, hi = robust_clip(all_vals.to_numpy(dtype=float), 0.5, 99.5)
    bins = np.linspace(lo, hi, 40)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = np.diff(bins)

    h_pf, _ = np.histogram(vals_pf, bins=bins, density=True)
    h_npf, _ = np.histogram(vals_npf, bins=bins, density=True)

    lo = lo if lo < 0.0 else -1e-6
    hi = hi if hi > 0.0 else 1e-6
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    cmap = plt.get_cmap("coolwarm")
    colors = cmap(norm(centers))

    ax.bar(centers, h_pf, width=width * 0.98, color=colors, alpha=0.85, edgecolor="none", align="center")
    ax.bar(centers, -h_npf, width=width * 0.98, color=colors, alpha=0.45, edgecolor="none", align="center")

    ax.set_xlabel(r"$d_u$ (mm/yr)")
    ax.set_ylabel("density (PF ↑ / NPF ↓)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(y):.02f}"))

    # inset colorbar to explain d_u shading
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = inset_axes(ax, width="45%", height="4%", loc="lower center", borderpad=1.1)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"$d_u$ (mm/yr)", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=7, length=2, pad=1)

    handles = [
        Patch(facecolor="0.4", alpha=0.85, label="PF"),
        Patch(facecolor="0.4", alpha=0.45, label="NPF"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 4: revised regional deformation context")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-per-domain", type=int, default=DESIRED_SAMPLE_PER_DOMAIN["pf"])
    parser.add_argument("--hotspot-q", type=float, default=HOTSPOT_Q_DEFAULT)
    parser.add_argument("--target-max-pixels", type=int, default=TARGET_MAX_PIXELS)
    parser.add_argument("--top-basins", type=int, default=5)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    workdir = args.out_dir.resolve() if args.out_dir is not None else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    figdir = workdir / "figures"
    cachedir = workdir / "cache"
    figdir.mkdir(parents=True, exist_ok=True)
    cachedir.mkdir(parents=True, exist_ok=True)

    pixel_csv = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    grad_raster_dir = base_dir / "outputs" / "grad_rasters"
    legacy_workdir = base_dir / "outputs" / "gradient_driver_analysis"
    legacy_tabledir = legacy_workdir / "tables"

    meta_path = grad_raster_dir / "grid_meta.npz"
    du_path = grad_raster_dir / "du_f32.memmap"
    gmag_path = grad_raster_dir / "gradmag_f32.memmap"
    legacy_sample_path = legacy_tabledir / "pixel_model_sample.parquet"

    figure_sample_path = cachedir / "figure4_pixel_sample.csv.gz"

    summary_out_path = figdir / f"{FIGURE_NAME}_summary.csv"
    png_out_path = figdir / f"{FIGURE_NAME}.png"
    pdf_out_path = figdir / f"{FIGURE_NAME}.pdf"

    required = [pixel_csv, meta_path, du_path, gmag_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n  - " + "\n  - ".join(str(p) for p in missing)
        )

    meta = np.load(meta_path)
    res = float(meta["res"])
    nrows = int(meta["nrows"])
    ncols = int(meta["ncols"])
    gx0 = int(meta["gx0"])
    gy1 = int(meta["gy1"])
    min_e = float(meta["min_e"])
    max_n = float(meta["max_n"])
    
    lonlat_extent_cache_path = cachedir / "figure4_lonlat_extent.json"

    extent_lonlat = resolve_lonlat_extent(
        csv_path=pixel_csv,
        cache_json=lonlat_extent_cache_path,
        min_e=min_e,
        max_n=max_n,
        nrows=nrows,
        ncols=ncols,
        res=res,
        chunksize=args.chunksize,
    )
    print(f"Output folder: {workdir}")
    print(f"Grid: {nrows} x {ncols} @ {res} m")

    sample_df = resolve_sample(
        csv_path=pixel_csv,
        cache_csv_gz=figure_sample_path,
        legacy_sample_path=legacy_sample_path,
        sample_per_domain=args.sample_per_domain,
        chunksize=args.chunksize,
    )

    selected_basins = select_top_basins(sample_df, top_n=args.top_basins)

    stride = choose_stride(nrows, ncols, target_max=args.target_max_pixels)

    du_mm = open_memmap(du_path, dtype="float32", mode="r", shape=(nrows, ncols))
    gmag_mm = open_memmap(gmag_path, dtype="float32", mode="r", shape=(nrows, ncols))

    du_ds = np.array(du_mm[::stride, ::stride], copy=False)
    gmag_ds = np.array(gmag_mm[::stride, ::stride], copy=False)
    gmag_km_ds = gmag_ds * 1000.0
    du_plot = np.flipud(du_ds)
    gmag_km_plot = np.flipud(gmag_km_ds)

    lat_du_profile = summarize_spatial_band_profile(
        du_plot,
        band_axis=0,
        coord_min=extent_lonlat[2],
        coord_max=extent_lonlat[3],
        max_bins=60,
    )
    lat_grad_profile = summarize_spatial_band_profile(
        gmag_km_plot,
        band_axis=0,
        coord_min=extent_lonlat[2],
        coord_max=extent_lonlat[3],
        max_bins=60,
    )
    lon_du_profile = summarize_spatial_band_profile(
        du_plot,
        band_axis=1,
        coord_min=extent_lonlat[0],
        coord_max=extent_lonlat[1],
        max_bins=60,
    )
    lon_grad_profile = summarize_spatial_band_profile(
        gmag_km_plot,
        band_axis=1,
        coord_min=extent_lonlat[0],
        coord_max=extent_lonlat[1],
        max_bins=60,
    )

    pf_sample = sample_df.loc[sample_df["domain"] == "pf"].copy()
    basin_order = [b for b in selected_basins if b in pf_sample["Main-Basin"].astype(str).unique().tolist()]
    if not basin_order:
        basin_order = (
            pf_sample["Main-Basin"]
            .astype(str)
            .value_counts()
            .head(args.top_basins)
            .index.tolist()
        )

    n_named = max(len(basin_order), 1)
    basin_colors = list(plt.get_cmap("tab10").colors[:n_named])
    basin_color_by_name = {
        str(basin): basin_colors[min(i, len(basin_colors) - 1)]
        for i, basin in enumerate(basin_order)
    }
    if "Inner Plateau" in basin_color_by_name and "Qaidam" in basin_color_by_name:
        basin_color_by_name["Inner Plateau"], basin_color_by_name["Qaidam"] = (
            basin_color_by_name["Qaidam"],
            basin_color_by_name["Inner Plateau"],
        )

    basin_du_data, basin_grad_data, basin_labels, basin_box_colors = [], [], [], []
    for i, basin in enumerate(basin_order):
        du_vals = clip_series(
            pf_sample.loc[pf_sample["Main-Basin"].astype(str) == str(basin), "d_u"],
            0.5,
            99.5,
        )
        grad_vals = clip_series(
            pf_sample.loc[pf_sample["Main-Basin"].astype(str) == str(basin), "grad_mag_km"],
            0.5,
            99.5,
        )
        if len(du_vals) == 0 or len(grad_vals) == 0:
            continue
        basin_du_data.append(du_vals.to_numpy())
        basin_grad_data.append(grad_vals.to_numpy())
        basin_labels.append(str(basin))
        basin_box_colors.append(
            basin_color_by_name.get(str(basin), basin_colors[min(i, len(basin_colors) - 1)])
        )

    fig = plt.figure(figsize=(14.4, 11.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2, 0.95],
        left=0.06,
        right=0.98,
        top=0.935,
        bottom=0.08,
        hspace=0.26,
    )

    gs_top = gs[0, 0].subgridspec(1, 2, wspace=0.12)
    gs_bottom = gs[1, 0].subgridspec(
        1,
        3,
        width_ratios=[1.8, 1.4, 1.2],
        wspace=0.26,
    )

    # Top-row triptychs
    axA_top, axA_map, axA_side = make_composite_triptych(
        fig,
        gs_top[0, 0],
        map_width=4.8,
        side_width=2.05,
    )

    axB_top, axB_map, axB_side = make_composite_triptych(
        fig,
        gs_top[0, 1],
        map_width=4.8,
        side_width=2.05,
    )

    # Bottom row
    axC = fig.add_subplot(gs_bottom[0, 0])
    axD = fig.add_subplot(gs_bottom[0, 1])
    axE = fig.add_subplot(gs_bottom[0, 2])

    # -------------------------------------------------------------------------
    # A. d_u composite: longitudinal profile + map + latitudinal profile
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # A/B/C. d_u composite: top profile + map + side profile
    vminA, vmaxA = centered_clip(du_ds, center=0.0, p_lo=2, p_hi=98)
    if vminA >= 0.0:
        vminA = -1e-6
    if vmaxA <= 0.0:
        vmaxA = 1e-6

    normA = TwoSlopeNorm(vmin=vminA, vcenter=0.0, vmax=vmaxA)

    imA = axA_map.imshow(
        du_plot,
        extent=extent_lonlat,
        origin="lower",
        cmap="coolwarm",
        norm=normA,
        aspect="auto",
    )

    format_lonlat_axes(
        axA_map,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=True,
        fill_plot_region=True,
    )

    add_map_inset_colorbar(axA_map, imA, "mm/yr", [0.74, 0.10, 0.07, 0.34])

    plot_top_profile(
        axA_top,
        lon_du_profile,
        color=DU_PROFILE_COLOR,
        ylabel="Mean\n(mm/yr)",
        title=r"Vertical deformation rate $d_u$",
        coord_limits=(extent_lonlat[0], extent_lonlat[1]),
    )
    use_right_y_axis(axA_top, color=DU_PROFILE_COLOR)
    axA_top.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)

    plot_side_profile(
        axA_side,
        lat_du_profile,
        color=DU_PROFILE_COLOR,
        xlabel="Mean\n(mm/yr)",
        coord_limits=(extent_lonlat[2], extent_lonlat[3]),
    )
    axA_side.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)

    use_map_spines_for_marginals(axA_top, axA_map, axA_side)

    add_panel_label(axA_top, "A")
    add_panel_label(axA_map, "B")
    add_panel_label(axA_side, "C")


    # -------------------------------------------------------------------------
    # D/E/F. gradient composite: top profile + map + side profile   
    # -------------------------------------------------------------------------
    vminB, vmaxB = robust_clip(gmag_km_ds, p_lo=2, p_hi=98)

    imB = axB_map.imshow(
        gmag_km_plot,
        extent=extent_lonlat,
        origin="lower",
        cmap="viridis",
        vmin=vminB,
        vmax=vmaxB,
        aspect="auto",
    )

    format_lonlat_axes(
        axB_map,
        show_ylabel=True,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=True,
        fill_plot_region=True,
    )

    add_map_inset_colorbar(axB_map, imB, "mm/yr/km", [0.74, 0.10, 0.07, 0.34])

    plot_top_profile(
        axB_top,
        lon_grad_profile,
        color=GMAG_PROFILE_COLOR,
        ylabel="Mean\n(mm/yr/km)",
        title=r"Spatial gradient magnitude $|\nabla d_u|$",
        coord_limits=(extent_lonlat[0], extent_lonlat[1]),
    )
    use_right_y_axis(axB_top, color=GMAG_PROFILE_COLOR)
    axB_top.set_ylim(-2.0, 15.0)
    axB_top.set_yticks([0.0, 10.0])

    plot_side_profile(
        axB_side,
        lat_grad_profile,
        color=GMAG_PROFILE_COLOR,
        xlabel="Mean\n(mm/yr/km)",
        coord_limits=(extent_lonlat[2], extent_lonlat[3]),
    )

    use_map_spines_for_marginals(axB_top, axB_map, axB_side)

    # E owns the left y-axis ticks/labels
    axB_map.tick_params(
        axis="y",
        which="both",
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )

    add_panel_label(axB_top, "D")
    add_panel_label(axB_map, "E")
    add_panel_label(axB_side, "F")

    # -------------------------------------------------------------------------
    # G. Combined basin-wise boxplots
    # -------------------------------------------------------------------------
    axC_top = draw_combined_basin_boxplot(
        axC,
        basin_du_data,
        basin_grad_data,
        basin_labels,
        basin_box_colors,
    )
    axC.set_title("Basin-wide permafrost distributions", fontweight="bold")
    axC.set_ylabel("Main basin", fontweight="bold")
    for tick, color in zip(axC.get_yticklabels(), basin_box_colors):
        tick.set_color(color)
    add_panel_label(axC, "G")

    # -------------------------------------------------------------------------
    # H. Mirrored negative-d_u histogram for Permafrost / Non-Permafrost
    # -------------------------------------------------------------------------
    vals_pf = clip_series(sample_df.loc[(sample_df["domain"] == "pf") & (sample_df["d_u"] < 0), "d_u"], 0.5, 99.5)
    vals_npf = clip_series(sample_df.loc[(sample_df["domain"] == "npf") & (sample_df["d_u"] < 0), "d_u"], 0.5, 99.5)
    all_vals = pd.concat([vals_pf, vals_npf], ignore_index=True)
    if not all_vals.empty:
        lo, hi = robust_clip(all_vals.to_numpy(dtype=float), 0.5, 99.5)
        lo = min(lo, -1e-6)
        hi = min(hi, 0.0)
        bins = np.linspace(lo, hi, 72)
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = np.diff(bins)

        h_pf, _ = np.histogram(vals_pf, bins=bins, density=True)
        h_npf, _ = np.histogram(vals_npf, bins=bins, density=True)
        y_max = max(float(np.nanmax(h_pf)) if len(h_pf) else 0.0, float(np.nanmax(h_npf)) if len(h_npf) else 0.0)
        y_max = 1.08 * y_max if y_max > 0 else 1.0

        normF = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=1e-6)
        cmapF = plt.get_cmap("coolwarm")
        bin_colors = cmapF(normF(centers))

        axD.axhspan(0.0, y_max, facecolor=PF_COLOR, alpha=0.28, zorder=0)
        axD.axhspan(-y_max, 0.0, facecolor=NPF_COLOR, alpha=0.28, zorder=0)

        axD.bar(centers, h_pf, width=width * 0.98, color=bin_colors, alpha=0.9, edgecolor="none", align="center", zorder=2)
        axD.bar(centers, -h_npf, width=width * 0.98, color=bin_colors, alpha=0.9, edgecolor="none", align="center", zorder=2)
        axD.set_xlim(lo, 0.0)
        axD.set_ylim(-y_max, y_max)

    axD.set_title(r"$d_u$: Permafrost vs Non-Permafrost", fontweight="bold")
    axD.set_xlabel(r"$d_u$ for $d_u < 0$ (mm/yr)", fontweight="bold")
    axD.set_ylabel("Density", fontweight="bold")
    axD.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(y):.02f}"))
    axD.axhline(0.0, color="k", linewidth=1.2, zorder=3)
    style_open_axes(axD)
    add_panel_label(axD, "H")

    axD.text(
        0.02, 0.90, DOMAIN_LABELS["pf"],
        transform=axD.transAxes,
        fontsize=10,
        fontweight="bold",
        color="0.25",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=PF_COLOR, alpha=0.8, edgecolor="none"),
    )
    axD.text(
        0.02, 0.08, DOMAIN_LABELS["npf"],
        transform=axD.transAxes,
        fontsize=10,
        fontweight="bold",
        color="0.25",
        bbox=dict(boxstyle="round,pad=0.18", facecolor=NPF_COLOR, alpha=0.8, edgecolor="none"),
    )

    # -------------------------------------------------------------------------
    # I. Negative |d_u| vs spatial gradient magnitude
    # -------------------------------------------------------------------------
    for dom, color in [("pf", PF_COLOR), ("npf", NPF_COLOR)]:
        sub = sample_df.loc[
            (sample_df["domain"] == dom) & (sample_df["d_u"] < 0),
            ["d_u", "grad_mag_km"],
        ].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) == 0:
            continue
        sub["neg_du"] = -sub["d_u"]
        x_hi = np.nanpercentile(sub["neg_du"], 99.5)
        y_hi = np.nanpercentile(sub["grad_mag_km"], 99.5)
        sub = sub.loc[(sub["neg_du"] <= x_hi) & (sub["grad_mag_km"] <= y_hi)].copy()
        curve = make_binned_curve_with_uncertainty(sub, xcol="neg_du", ycol="grad_mag_km", q=20)
        if curve.empty:
            continue
        axE.fill_between(curve["x"], curve["ylo"], curve["yhi"], color=color, alpha=0.22)
        axE.plot(
            curve["x"],
            curve["y"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=color,
            label=DOMAIN_LABELS[dom],
        )

    axE.set_title(r"$|d_u|$ vs $|\nabla d_u|$", fontweight="bold")
    axE.set_xlabel(r"$|d_u|$ for $d_u < 0$ (mm/yr)", fontweight="bold")
    axE.set_ylabel(r"mean $|\nabla d_u|$ (mm/yr/km)", fontweight="bold")
    style_open_axes(axE)
    handles, labels = axE.get_legend_handles_labels()
    if handles:
        axE.legend(handles=handles, labels=labels, title="Domain", frameon=False)
    add_panel_label(axE, "I")

    # -------------------------------------------------------------------------
    # Bold everything except legends
    # -------------------------------------------------------------------------
    axes_to_bold = [axA_top, axA_map, axA_side, axB_top, axB_map, axB_side, axC, axD, axE]
    if axC_top is not None:
        axes_to_bold.append(axC_top)
    for ax in axes_to_bold:
        apply_bold_nonlegend(ax)

    plt.savefig(png_out_path, bbox_inches="tight")
    plt.savefig(pdf_out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure PNG: {png_out_path}")
    print(f"Saved figure PDF: {pdf_out_path}")

    summary = pd.DataFrame(
        {
            "metric": [
                "sample_n_pf",
                "sample_n_npf",
                "negative_du_n_pf",
                "negative_du_n_npf",
                "du_map_stride",
                "grid_res_m",
                "grid_nrows",
                "grid_ncols",
                "selected_basins",
            ],
            "value": [
                int((sample_df["domain"] == "pf").sum()),
                int((sample_df["domain"] == "npf").sum()),
                int(((sample_df["domain"] == "pf") & (sample_df["d_u"] < 0)).sum()),
                int(((sample_df["domain"] == "npf") & (sample_df["d_u"] < 0)).sum()),
                stride,
                res,
                nrows,
                ncols,
                "; ".join(selected_basins),
            ],
        }
    )
    summary.to_csv(summary_out_path, index=False)
    print(f"Saved figure summary: {summary_out_path}")


if __name__ == "__main__":
    main()
