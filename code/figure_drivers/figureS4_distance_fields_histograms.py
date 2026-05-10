#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/FigureS4_tlrts_distance_distribution.py
# Renamed package path: code/figure_drivers/figureS4_distance_fields_histograms.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from submission_build_common import (
    DU_RASTER_PATH,
    FONT,
    SOURCE_CACHE_DIR,
    add_panel_label,
    blend_with_white,
    clip_gdf_to_hull,
    coverage_hull_lonlat_from_path,
    ensure_style,
    load_geo_layers,
    plot_gdf_with_halo,
    read_joblib_df,
    save_figure,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

import Figure_reorganize_railway_buffer_analysis as railbuf


FIG_STEM = "FigureS4_tlrts_distance_distribution"
MAP_EXTENT = [88.18, 97.86, 27.8, 38.5]

PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"

BOUNDARY_THRESHOLD_KM = 5.0
BOUNDARY_VMAX_KM = 20.0
BOUNDARY_TICKS = np.array([-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0], dtype=float)

ABRUPT_THRESHOLD_KM = 2.0
ABRUPT_VMAX_KM = 10.0
ABRUPT_TICKS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

RAILWAY_BUFFER_WIDTH_KM = 1.0
RAILWAY_BUFFER_HALF_WIDTH_M = 500.0
PROFILE_BIN_LENGTH_KM = 75.0
PROFILE_STD_SCALE = 0.25
PROFILE_YTICKS = np.array([0.0, 250.0, 500.0, 750.0, 1000.0], dtype=float)


def load_sample() -> pd.DataFrame:
    return read_joblib_df(
        SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_du_tlrts_distance_sample.joblib.gz"
    )


def _fmt_lon(x, _pos):
    if not np.isfinite(x):
        return ""
    hemi = "E" if x >= 0 else "W"
    v = abs(float(x))
    t = f"{int(round(v))}" if np.isclose(v, round(v), atol=1e-6) else f"{v:.1f}".rstrip("0").rstrip(".")
    return rf"${t}^\circ$ {hemi}"


def _fmt_lat(y, _pos):
    if not np.isfinite(y):
        return ""
    hemi = "N" if y >= 0 else "S"
    v = abs(float(y))
    t = f"{int(round(v))}" if np.isclose(v, round(v), atol=1e-6) else f"{v:.1f}".rstrip("0").rstrip(".")
    return rf"${t}^\circ$ {hemi}"


def format_abs_ticklabels(ticks: np.ndarray) -> list[str]:
    labels: list[str] = []
    for tick in np.asarray(ticks, dtype=float):
        if np.isclose(tick, 0.0):
            labels.append("0")
        else:
            labels.append(f"{abs(tick):.0f}")
    return labels


def prepare_boundary_points(df: pd.DataFrame) -> pd.DataFrame:
    work = df.loc[df["domain"].isin(["pf", "npf"])].copy()
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["zou_boundary_distance_km"] = pd.to_numeric(work["zou_boundary_distance_km"], errors="coerce")
    work = work.loc[
        np.isfinite(work["longitude"])
        & np.isfinite(work["latitude"])
        & np.isfinite(work["zou_boundary_distance_km"])
    ].copy()
    work["signed_boundary_distance_km"] = work["zou_boundary_distance_km"]
    work.loc[work["domain"].eq("npf"), "signed_boundary_distance_km"] *= -1.0
    return work


def prepare_abrupt_points(df: pd.DataFrame) -> pd.DataFrame:
    work = df.loc[df["domain"].eq("pf")].copy()
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    work["tlrts_distance_km"] = pd.to_numeric(work["tlrts_distance_km"], errors="coerce")
    work = work.loc[
        np.isfinite(work["longitude"])
        & np.isfinite(work["latitude"])
        & np.isfinite(work["tlrts_distance_km"])
    ].copy()
    return work


def build_railway_context(layers: dict[str, object]) -> dict[str, object]:
    geom = layers["railway_proj"].geometry
    network_geom = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    return {
        "crs": layers["railway_proj"].crs,
        "network_geom": network_geom,
        "main_line": railbuf.longest_line(network_geom),
    }


def build_corridor_profile_sample(df: pd.DataFrame, railway_context: dict[str, object], *, value_col: str) -> pd.DataFrame:
    work = df[["longitude", "latitude", value_col]].copy()
    points = railbuf.gpd.GeoSeries(
        railbuf.gpd.points_from_xy(work["longitude"], work["latitude"]),
        crs="EPSG:4326",
    ).to_crs(railway_context["crs"])
    work["distance_to_rail_m"] = points.distance(railway_context["network_geom"]).to_numpy(dtype=float)
    work["along_km"] = np.fromiter(
        (float(railway_context["main_line"].project(geom)) / 1000.0 for geom in points),
        dtype=float,
        count=len(points),
    )
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[
        np.isfinite(work["distance_to_rail_m"])
        & (work["distance_to_rail_m"] <= RAILWAY_BUFFER_HALF_WIDTH_M)
        & np.isfinite(work["along_km"])
        & np.isfinite(work[value_col])
    ].copy()
    return work.sort_values("along_km").reset_index(drop=True)


def build_profile_table(sample_df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    return railbuf.summarize_metric_profile(
        sample_df,
        value_col=value_col,
        bin_length_km=PROFILE_BIN_LENGTH_KM,
        std_scale=PROFILE_STD_SCALE,
    )


def padded_limits(values: np.ndarray, *, pad_frac: float, floor: float | None = None, ceil: float | None = None) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(arr.min())
        hi = float(arr.max())
        if np.isclose(lo, hi):
            hi = lo + 1.0
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    if floor is not None:
        lo = min(lo, float(floor))
    if ceil is not None:
        hi = max(hi, float(ceil))
    return lo, hi


def style_map_axes(ax) -> None:
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("auto")
    ax.tick_params(length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_map_context(ax, railway) -> None:
    plot_gdf_with_halo(
        ax,
        railway,
        color="0.08",
        linewidth=1.0,
        halo_width=3.2,
        alpha=0.95,
        zorder=4,
    )
    style_map_axes(ax)


def add_inset_colorbar(ax, mappable, *, ticks: np.ndarray, ticklabels: list[str] | None, label: str) -> None:
    cax = ax.inset_axes([0.52, 0.12, 0.42, 0.035])
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks.tolist())
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=FONT["annotation"], length=2, pad=1)
    cb.set_label(label, fontsize=FONT["annotation"])


def plot_side_profile(
    ax,
    profile_df: pd.DataFrame,
    *,
    line_color: str,
    fill_color: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    xticks: np.ndarray,
    threshold_lines: list[tuple[float, str]],
) -> None:
    if not profile_df.empty:
        ax.fill_betweenx(
            profile_df["center_km"],
            profile_df["ylo"],
            profile_df["yhi"],
            color=fill_color,
            alpha=0.68,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            profile_df["mean"],
            profile_df["center_km"],
            color=line_color,
            linewidth=1.6,
            zorder=2,
        )

    for xpos, color in threshold_lines:
        ax.axvline(xpos, color=color, linestyle="--", linewidth=1.0, zorder=0)

    ax.set_title(title, fontsize=FONT["axis"], pad=6)
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_linespacing(1.0)
    ax.set_ylabel("Distance along railway from Lhasa (km)")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1100.0)
    ax.set_xticks(xticks.tolist())
    ax.set_yticks(PROFILE_YTICKS.tolist())
    ax.grid(axis="both", color="0.92", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def plot_distance_histogram(
    ax,
    values: np.ndarray,
    *,
    bins: np.ndarray,
    cmap,
    norm,
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: np.ndarray,
    threshold_lines: list[tuple[float, str | None, str]],
    absolute_xticks: bool = False,
) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    inside = finite[(finite >= bins[0]) & (finite <= bins[-1])]
    counts, edges = np.histogram(inside, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    colors = cmap(np.clip(norm(centers), 0.0, 1.0))

    ax.bar(
        centers,
        counts,
        width=widths,
        color=colors,
        edgecolor="0.35",
        linewidth=0.35,
        align="center",
        alpha=0.92,
    )
    legend_lines = []
    for xpos, label, color in threshold_lines:
        line = ax.axvline(xpos, color=color, linestyle="--", linewidth=1.2, label=label)
        if label is not None:
            legend_lines.append(line)

    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", x=0.08, pad=6)
    ax.set_xticks(xticks.tolist())
    if absolute_xticks:
        ax.set_xticklabels(format_abs_ticklabels(xticks))
        ax.text(0.17, 0.93, "NPF", transform=ax.transAxes, color=NPF_COLOR, fontsize=FONT["annotation"], ha="center", va="top")
        ax.text(0.83, 0.93, "PF", transform=ax.transAxes, color=PF_COLOR, fontsize=FONT["annotation"], ha="center", va="top")
    ax.grid(axis="y", color="0.92", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_lines:
        ax.legend(handles=legend_lines, frameon=False, loc="upper right", fontsize=FONT["annotation"])

    outside = int(np.sum((finite < bins[0]) | (finite > bins[-1])))
    if outside > 0 and finite.size > 0:
        pct = 100.0 * outside / finite.size
        ax.text(
            0.98,
            0.86,
            f"{pct:.1f}% out of range",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT["annotation"],
        )


def main() -> None:
    ensure_style()
    df = load_sample()
    boundary = prepare_boundary_points(df)
    abrupt = prepare_abrupt_points(df)

    layers = load_geo_layers()
    coverage = coverage_hull_lonlat_from_path(str(DU_RASTER_PATH))
    railway = clip_gdf_to_hull(layers["railway_ll"], coverage)
    railway_context = build_railway_context(layers)

    boundary_profile_sample = build_corridor_profile_sample(
        boundary,
        railway_context,
        value_col="signed_boundary_distance_km",
    )
    abrupt_profile_sample = build_corridor_profile_sample(
        abrupt,
        railway_context,
        value_col="tlrts_distance_km",
    )
    boundary_profile = build_profile_table(
        boundary_profile_sample,
        value_col="signed_boundary_distance_km",
    )
    abrupt_profile = build_profile_table(
        abrupt_profile_sample,
        value_col="tlrts_distance_km",
    )
    boundary_profile_xlim = padded_limits(
        boundary_profile[["ylo", "yhi"]].to_numpy(dtype=float).ravel(),
        pad_frac=0.06,
        floor=-BOUNDARY_THRESHOLD_KM,
        ceil=BOUNDARY_THRESHOLD_KM,
    )
    abrupt_profile_xlim = padded_limits(
        abrupt_profile[["ylo", "yhi"]].to_numpy(dtype=float).ravel(),
        pad_frac=0.08,
        floor=0.0,
        ceil=ABRUPT_THRESHOLD_KM,
    )

    fig = plt.figure(figsize=(12.6, 8.8))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.34, 1.0], hspace=0.30, wspace=0.20)
    top_left = outer[0, 0].subgridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)
    bottom_left = outer[1, 0].subgridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)

    boundary_ax = fig.add_subplot(top_left[0, 0])
    boundary_profile_ax = fig.add_subplot(top_left[0, 1])
    boundary_hist_ax = fig.add_subplot(outer[0, 1])
    abrupt_ax = fig.add_subplot(bottom_left[0, 0])
    abrupt_profile_ax = fig.add_subplot(bottom_left[0, 1])
    abrupt_hist_ax = fig.add_subplot(outer[1, 1])
    plt.subplots_adjust(left=0.070, right=0.985, top=0.94, bottom=0.08)

    boundary_cmap = plt.get_cmap("coolwarm_r")
    boundary_norm = TwoSlopeNorm(vmin=-BOUNDARY_VMAX_KM, vcenter=0.0, vmax=BOUNDARY_VMAX_KM)
    boundary_scatter = boundary_ax.scatter(
        boundary["longitude"],
        boundary["latitude"],
        c=boundary["signed_boundary_distance_km"],
        s=4.0,
        cmap=boundary_cmap,
        norm=boundary_norm,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    add_map_context(boundary_ax, railway)
    boundary_ax.set_title("PF/NPF boundary distance field", loc="left", x=0.08, pad=6)
    add_inset_colorbar(
        boundary_ax,
        boundary_scatter,
        ticks=BOUNDARY_TICKS,
        ticklabels=format_abs_ticklabels(BOUNDARY_TICKS),
        label=r"Boundary distance (km; NPF < 0 < PF)",
    )
    plot_side_profile(
        boundary_profile_ax,
        boundary_profile,
        line_color="0.18",
        fill_color=blend_with_white("#44546A", 0.38),
        title="1 km railway buffer profile",
        xlabel=r"Mean $d_B$ (km)",
        xlim=boundary_profile_xlim,
        xticks=np.array([-50.0, -20.0, 0.0, 20.0], dtype=float),
        threshold_lines=[
            (-BOUNDARY_THRESHOLD_KM, "0.10"),
            (0.0, "0.35"),
            (BOUNDARY_THRESHOLD_KM, "0.10"),
        ],
    )
    boundary_profile_ax.axvspan(
        boundary_profile_xlim[0], 0.0,
        color=NPF_COLOR, alpha=0.08, zorder=0,
    )
    boundary_profile_ax.axvspan(
        0.0, boundary_profile_xlim[1],
        color=PF_COLOR, alpha=0.08, zorder=0,
    )
    plot_distance_histogram(
        boundary_hist_ax,
        boundary["signed_boundary_distance_km"].to_numpy(dtype=float),
        bins=np.linspace(-BOUNDARY_VMAX_KM, BOUNDARY_VMAX_KM, 41),
        cmap=boundary_cmap,
        norm=boundary_norm,
        title="PF/NPF boundary-distance histogram",
        xlabel="Boundary distance (km; left = NPF, right = PF)",
        ylabel="Sample count",
        xticks=BOUNDARY_TICKS,
        threshold_lines=[
            (-BOUNDARY_THRESHOLD_KM, r"$|d_B| = 5$ km threshold", "0.10"),
            (BOUNDARY_THRESHOLD_KM, None, "0.10"),
        ],
        absolute_xticks=True,
    )
    boundary_hist_ax.axvspan(
        -BOUNDARY_VMAX_KM, 0.0,
        color=NPF_COLOR, alpha=0.06, zorder=0,
    )
    boundary_hist_ax.axvspan(
        0.0, BOUNDARY_VMAX_KM,
        color=PF_COLOR, alpha=0.06, zorder=0,
    )

    abrupt_cmap = plt.get_cmap("Reds_r")
    abrupt_norm = Normalize(vmin=0.0, vmax=ABRUPT_VMAX_KM)
    abrupt_scatter = abrupt_ax.scatter(
        abrupt["longitude"],
        abrupt["latitude"],
        c=abrupt["tlrts_distance_km"],
        s=4.0,
        cmap=abrupt_cmap,
        norm=abrupt_norm,
        alpha=0.74,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    add_map_context(abrupt_ax, railway)
    abrupt_ax.set_title("PF distance to nearest lake/slump", loc="left", x=0.08, pad=6)
    add_inset_colorbar(
        abrupt_ax,
        abrupt_scatter,
        ticks=ABRUPT_TICKS,
        ticklabels=None,
        label=r"Distance to nearest thermokarst lake or thaw slump (km)",
    )
    plot_side_profile(
        abrupt_profile_ax,
        abrupt_profile,
        line_color="0.08",
        fill_color=blend_with_white(PF_COLOR, 0.58),
        title="1 km railway buffer profile",
        xlabel=r"Mean $d_A$ (km)",
        xlim=abrupt_profile_xlim,
        xticks=np.array([0.0, 2.0, 6.0, 10.0], dtype=float),
        threshold_lines=[(ABRUPT_THRESHOLD_KM, "#7A0C0C")],
    )
    abrupt_profile_ax.axvspan(
        abrupt_profile_xlim[0], abrupt_profile_xlim[1],
        color=PF_COLOR, alpha=0.08, zorder=0,
    )
    plot_distance_histogram(
        abrupt_hist_ax,
        abrupt["tlrts_distance_km"].to_numpy(dtype=float),
        bins=np.linspace(0.0, ABRUPT_VMAX_KM, 31),
        cmap=abrupt_cmap,
        norm=abrupt_norm,
        title="Lake/slump-distance histogram",
        xlabel="Distance to nearest thermokarst lake or thaw slump (km)",
        ylabel="PF sample count",
        xticks=ABRUPT_TICKS,
        threshold_lines=[(ABRUPT_THRESHOLD_KM, r"$d_A = 2$ km threshold", "#7A0C0C")],
    )
    abrupt_hist_ax.axvspan(
        0.0, ABRUPT_VMAX_KM,
        color=PF_COLOR, alpha=0.06, zorder=0,
    )

    add_panel_label(boundary_ax, "A", x=-0.08, y=1.02)
    add_panel_label(boundary_hist_ax, "B", x=-0.08, y=1.02)
    add_panel_label(abrupt_ax, "C", x=-0.08, y=1.02)
    add_panel_label(abrupt_hist_ax, "D", x=-0.08, y=1.02)

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
