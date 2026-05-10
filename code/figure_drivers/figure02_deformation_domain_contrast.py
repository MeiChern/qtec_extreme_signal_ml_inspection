#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure02_regional_deformation_domain_contrast.py
# Renamed package path: code/figure_drivers/figure02_deformation_domain_contrast.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from submission_build_common import (
    DU_RASTER_PATH,
    FONT,
    GRAD_RASTER_PATH,
    ROOT_DIR,
    SOURCE_CACHE_DIR,
    SOURCE_TABLE_DIR,
    add_panel_label,
    add_scalebar_lonlat,
    blend_with_white,
    clip_gdf_to_hull,
    coverage_hull_lonlat_from_path,
    decimate_raster,
    ensure_style,
    grid_meta,
    load_du_raster,
    load_geo_layers,
    load_grad_raster,
    lonlat_extent,
    masked_raster,
    project_crs,
    robust_limits,
    save_figure,
    symmetric_limits,
    text_halo,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


FIG_STEM = "Figure02_regional_deformation_domain_contrast"
PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
DU_COLOR = "#1E5BAA"
GRAD_COLOR = "#C1272D"

# Histogram style (following Figure_reorganize_pf_vs_npf_in_du_and_gradient)
HIST_BINS_DU = 54
HIST_BINS_GRAD = 48
HIST_CORE_BLEND = 0.62
HIST_CORE_ALPHA = 0.22
HIST_TAIL_ALPHA = 0.38
HIST_LINESTYLE = (0, (4.0, 2.0))
COMBINED_LINEWIDTH = 0.9
P95_TOKEN = r"P_{\mathrm{95}}"

# Relationship panel
REL_CORE_BLEND = 0.60
REL_CORE_FILL_ALPHA = 0.10
REL_TAIL_FILL_ALPHA = 0.18
CURVE_Q = 22

# Map overlay
METEORO_COLOR = "#3A3A3A"
METEORO_SIZE = 26
RAIL_DASHES = (5, 3)
RAILWAY_LEGEND_LON = 92.5
RAILWAY_LEGEND_LAT = 28.0
COLORBAR_RECT = [0.55, 0.155, 0.20, 0.032]

RAILWAY_SHP = ROOT_DIR / "human_features" / "qtec_railway_clip.shp"
METEORO_SHP = ROOT_DIR / "human_features" / "qtec_meteoro_station_sites.shp"

STATION_LABEL_SPECS = {
    "Tuotuohe": (-6.0, 4.0, "right"),
    "Wudaoliang": (6.0, 4.0, "left"),
    "Xiaozaohuo": (6.0, 4.0, "left"),
    "Golmud": (6.0, -9.0, "left"),
    "Banga": (-6.0, 4.0, "right"),
    "Nagqu": (6.0, 4.0, "left"),
    "Lhasa": (-6.0, -10.0, "right"),
    "Shannan": (6.0, -10.0, "left"),
}

MAP_ZOOM_SPECS = (
    {
        "site_label": "Wudaoliang",
        "bounds": (0.02, 0.75, 0.25, 0.23),
        "xpad": 0.42,
        "ypad": 0.24,
    },
    {
        "site_label": "Tuotuohe",
        "bounds": (0.52, 0.26, 0.25, 0.23),
        "xpad": 0.34,
        "ypad": 0.22,
    },
)


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def load_samples() -> pd.DataFrame:
    path = SOURCE_CACHE_DIR / "Figure_reorganize_pf_vs_npf_in_du_and_gradient_sample_n180000.csv.gz"
    df = pd.read_csv(path)
    if "abs_du" not in df.columns and "d_u" in df.columns:
        df["abs_du"] = np.abs(pd.to_numeric(df["d_u"], errors="coerce"))
    return df


def load_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SOURCE_TABLE_DIR / "Figure_reorganized_railway_extreme_deformation_inspection_profiles.csv")
    subset = df.loc[df["buffer_width_km"].round(1).eq(1.0)].copy()
    return (
        subset.loc[subset["metric"].eq("d_u")].copy(),
        subset.loc[subset["metric"].eq("grad_mag_km")].copy(),
    )


def load_meteoro_sites() -> gpd.GeoDataFrame:
    path = ROOT_DIR / "human_features" / "qtec_meteoro_station_sites.shp"
    gdf = gpd.read_file(path)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    for col in ("english_st", "english_station_name", "station_na", "station_name", "NAME"):
        if col not in gdf.columns:
            continue
        labels = gdf[col].astype(str).str.strip()
        labels = labels.mask(labels.eq("")).mask(labels.str.lower().isin({"nan", "none"}))
        if labels.notna().any():
            gdf["site_label"] = labels
            break
    return gdf.loc[gdf["site_label"].notna()].copy()


def load_profile_sites() -> pd.DataFrame:
    try:
        import Figure_reorganized_railway_extreme_deformation_inspection as railinspect
        import Figure_reorganize_railway_buffer_analysis as railbuf
        rc = railbuf.load_railway_context(RAILWAY_SHP)
        return railinspect.load_profile_sites(METEORO_SHP, railway_context=rc)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# lon/lat mesh for pcolormesh
# ---------------------------------------------------------------------------

def build_lonlat_mesh(*, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    meta = grid_meta()
    res = float(meta["res"])
    gx0, gy1 = int(meta["gx0"]), int(meta["gy1"])
    nrows, ncols = int(meta["nrows"]), int(meta["ncols"])
    row_idx = np.arange(0, nrows, stride, dtype=np.int64)
    col_idx = np.arange(0, ncols, stride, dtype=np.int64)
    easting = (gx0 + col_idx).astype(float) * res
    northing = (gy1 - row_idx).astype(float) * res
    e_grid, n_grid = np.meshgrid(easting, northing)
    transformer = Transformer.from_crs(project_crs(), "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(e_grid, n_grid)
    return (
        np.flipud(np.asarray(lon_grid, dtype=float)),
        np.flipud(np.asarray(lat_grid, dtype=float)),
    )


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

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


def soften_color(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - blend) * rgb + blend)


def upper_percentile(vals, percentile: float = 95.0):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.percentile(arr, percentile))


def clip_percentile(vals, p_lo: float = 0.5, p_hi: float = 99.5):
    arr = pd.Series(pd.to_numeric(vals, errors="coerce"))
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return arr
    lo, hi = np.percentile(arr, [p_lo, p_hi])
    return arr[(arr >= lo) & (arr <= hi)]


# ---------------------------------------------------------------------------
# map panels (A, B) — styled after Figure_reorganize_du_du_gradient_ml_features
# ---------------------------------------------------------------------------

def style_map_axis(ax) -> None:
    ax.set_xlim(88.18, 97.86)
    ax.set_ylim(27.8, 38.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("auto")
    ax.tick_params(length=3)


def color_metric_axis(ax, axis: str, color: str) -> None:
    if axis == "x":
        ax.xaxis.label.set_color(color)
        ax.tick_params(axis="x", colors=color)
        ax.spines["bottom"].set_edgecolor(color)
        labels = ax.get_xticklabels()
    elif axis == "y":
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="y", colors=color)
        ax.spines["left"].set_edgecolor(color)
        labels = ax.get_yticklabels()
    else:
        raise ValueError(f"Unsupported axis: {axis!r}")

    for label in labels:
        label.set_color(color)


def color_profile_axis(ax, color: str) -> None:
    ax.xaxis.label.set_color(color)
    ax.tick_params(axis="x", colors=color)
    for label in ax.get_xticklabels():
        label.set_color(color)
    if ax.spines["bottom"].get_visible():
        ax.spines["bottom"].set_edgecolor(color)

    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="y", colors="black")
    for label in ax.get_yticklabels():
        label.set_color("black")
    if ax.spines["left"].get_visible():
        ax.spines["left"].set_edgecolor("black")


def plot_dashed_railway(ax, railway_gdf) -> None:
    if railway_gdf.empty:
        return
    railway_gdf.plot(ax=ax, color="white", linewidth=2.8, alpha=1.0, zorder=4)
    railway_gdf.plot(
        ax=ax, color="black", linewidth=1.5,
        linestyle=(0, RAIL_DASHES), alpha=1.0, zorder=5,
    )


def add_railway_legend(ax) -> None:
    x0, x1 = float(RAILWAY_LEGEND_LON), float(RAILWAY_LEGEND_LON) + 1.45
    y0 = float(RAILWAY_LEGEND_LAT)
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.8, zorder=8)
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5, linestyle=(0, RAIL_DASHES), zorder=9)
    ax.text(
        x1 + 0.12, y0, "Railway",
        ha="left", va="center", fontsize=FONT["annotation"],
        color="0.1",
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(1.0, 1.0, 1.0, 0.78), edgecolor="none"),
        zorder=10,
    )


def add_inset_colorbar(ax, mappable, *, label, ticks=None) -> None:
    cax = ax.inset_axes(COLORBAR_RECT, transform=ax.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=FONT["annotation"], length=2, pad=1)
    cb.set_label(label, fontsize=FONT["annotation"], labelpad=2)


def draw_meteoro_sites(ax, sites: gpd.GeoDataFrame) -> None:
    from shapely.geometry import box

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    panel_geom = box(xlim[0], ylim[0], xlim[1], ylim[1])
    visible = sites.loc[sites.geometry.intersects(panel_geom)].copy()
    if visible.empty:
        return
    ax.scatter(
        visible.geometry.x.to_numpy(dtype=float),
        visible.geometry.y.to_numpy(dtype=float),
        s=METEORO_SIZE, marker="s",
        color=METEORO_COLOR, edgecolors=METEORO_COLOR,
        linewidths=0.0, zorder=5.3,
    )
    for site in visible.itertuples(index=False):
        label = str(site.site_label).strip()
        dx, dy, ha = STATION_LABEL_SPECS.get(label, (5.0, 3.0, "left"))
        text = ax.annotate(
            label,
            xy=(float(site.geometry.x), float(site.geometry.y)),
            xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="bottom",
            fontsize=FONT["annotation"] - 0.3,
            color=METEORO_COLOR, zorder=6.2,
        )
        text.set_path_effects(text_halo(2.0))


def add_zoom_insets(ax, *, plot_arr, lon_plot, lat_plot, cmap, norm, sites, railway_ll, coverage) -> None:
    for spec in MAP_ZOOM_SPECS:
        match = sites.loc[sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site = match.iloc[0]
        lon, lat = float(site.geometry.x), float(site.geometry.y)
        inset = ax.inset_axes(spec["bounds"])
        inset.pcolormesh(
            lon_plot, lat_plot, plot_arr,
            cmap=cmap, norm=norm,
            shading="auto", rasterized=True, zorder=0,
        )
        rail_clip = clip_gdf_to_hull(railway_ll, coverage)
        if not rail_clip.empty:
            rail_clip.plot(ax=inset, color="white", linewidth=2.8, alpha=1.0, zorder=4)
            rail_clip.plot(
                ax=inset, color="black", linewidth=1.5,
                linestyle=(0, RAIL_DASHES), alpha=1.0, zorder=5,
            )
        inset.set_xlim(lon - spec["xpad"], lon + spec["xpad"])
        inset.set_ylim(lat - spec["ypad"], lat + spec["ypad"])
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.scatter(
            [lon], [lat], s=METEORO_SIZE * 2, marker="s",
            color=METEORO_COLOR, linewidths=0.0, zorder=5.3,
        )
        text = inset.annotate(
            str(site.site_label),
            xy=(lon, lat), xytext=(4.0, 2.0), textcoords="offset points",
            ha="left", va="bottom",
            fontsize=FONT["annotation"], color=METEORO_COLOR, zorder=6.2,
        )
        text.set_path_effects(text_halo(2.0))


def plot_map(
    ax,
    *,
    raster: np.ndarray,
    raster_path,
    cmap,
    norm,
    title: str,
    cbar_label: str,
    title_color: str | None = None,
    sites: gpd.GeoDataFrame,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    cbar_ticks=None,
) -> None:
    arr, stride = decimate_raster(raster, target_max=1400)
    plot_arr = np.flipud(arr)
    plot_arr = np.ma.masked_invalid(np.where(np.isclose(plot_arr, 0.0), np.nan, plot_arr.astype(float)))
    plot_cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    plot_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    coverage = coverage_hull_lonlat_from_path(str(raster_path))

    im = ax.pcolormesh(
        lon_plot, lat_plot, plot_arr,
        cmap=plot_cmap, norm=norm,
        shading="auto", rasterized=True, zorder=0,
    )

    layers = load_geo_layers()
    railway = clip_gdf_to_hull(layers["railway_ll"], coverage)
    plot_dashed_railway(ax, railway)
    draw_meteoro_sites(ax, sites)

    style_map_axis(ax)
    ax.set_title(title, pad=6, color=title_color)

    add_scalebar_lonlat(ax, length_km=100, lon=89.0, lat=34.0)
    add_railway_legend(ax)
    add_inset_colorbar(ax, im, label=cbar_label, ticks=cbar_ticks)

    add_zoom_insets(
        ax, plot_arr=plot_arr, lon_plot=lon_plot, lat_plot=lat_plot,
        cmap=plot_cmap, norm=norm,
        sites=sites, railway_ll=layers["railway_ll"], coverage=coverage,
    )


# ---------------------------------------------------------------------------
# profile panel
# ---------------------------------------------------------------------------

def annotate_profile_sites(ax, profile_sites: pd.DataFrame) -> None:
    if profile_sites.empty:
        return
    ylim = ax.get_ylim()
    ylo, yhi = min(ylim), max(ylim)
    for site in profile_sites.itertuples(index=False):
        km = float(site.along_km)
        if km < ylo or km > yhi:
            continue
        ax.axhline(y=km, color="0.75", linewidth=0.3, linestyle=":", alpha=0.5, zorder=0.5)
        ax.text(
            0.97, km, str(site.site_label),
            transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="right", va="center",
            fontsize=FONT["annotation"] - 1.0,
            color="0.45",
            path_effects=text_halo(1.5),
            clip_on=True,
        )


def plot_profile(
    ax,
    prof: pd.DataFrame,
    *,
    color: str,
    title: str,
    xlabel: str,
    show_ylabel: bool,
    profile_sites: pd.DataFrame | None = None,
) -> None:
    ax.fill_betweenx(
        prof["center_km"], prof["ylo"], prof["yhi"],
        color=blend_with_white(color, 0.72), alpha=0.7, linewidth=0,
    )
    ax.plot(prof["mean"], prof["center_km"], color=color, linewidth=1.6)
    ax.set_title(title, fontsize=FONT["axis"], pad=6)
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_linespacing(1.0)
    if show_ylabel:
        ax.set_ylabel("Distance along railway from Lhasa (km)")
    else:
        ax.set_ylabel("")
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    if profile_sites is not None:
        annotate_profile_sites(ax, profile_sites)


# ---------------------------------------------------------------------------
# histogram helpers (C, D — styled after Figure_reorganize_pf_vs_npf_in_du_and_gradient)
# ---------------------------------------------------------------------------

def split_xy_at_threshold(x_vals, y_vals, *, split_x):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if x.size == 0 or split_x is None or not np.isfinite(split_x):
        return x, y, np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if split_x <= x[0]:
        return np.array([], dtype=float), np.array([], dtype=float), x, y
    if split_x >= x[-1]:
        return x, y, np.array([], dtype=float), np.array([], dtype=float)
    y_split = float(np.interp(split_x, x, y))
    core_x = np.concatenate([x[x < split_x], [split_x]])
    core_y = np.concatenate([y[x < split_x], [y_split]])
    tail_x = np.concatenate([[split_x], x[x > split_x]])
    tail_y = np.concatenate([[y_split], y[x > split_x]])
    return core_x, core_y, tail_x, tail_y


def build_hist_bins(vals, n_bins, *, force_lo=None, force_hi=None):
    clipped = clip_percentile(vals, 0.5, 99.5)
    if clipped.empty:
        return np.array([])
    lo = float(clipped.min()) if force_lo is None else float(force_lo)
    hi = float(clipped.max()) if force_hi is None else float(force_hi)
    if hi <= lo:
        return np.array([])
    return np.linspace(lo, hi, int(n_bins) + 1)


def annotate_p95(ax, *, xpos, color, y_data=None, y_frac=None, linewidth=1.1):
    if xpos is None:
        return
    xmin, xmax = ax.get_xlim()
    if xpos < xmin or xpos > xmax:
        return
    ax.axvline(xpos, color=color, linewidth=linewidth, linestyle=(0, (4.0, 2.0)), alpha=0.95, zorder=4)
    x_span = xmax - xmin
    x_text = xpos + 0.012 * x_span
    ha = "left"
    if x_text > xmax - 0.01 * x_span:
        x_text = xpos - 0.012 * x_span
        ha = "right"
    value_label = f"{xpos:.1f}" if abs(xpos) >= 1.0 else f"{xpos:.2f}"
    if y_data is not None:
        ax.text(
            x_text, y_data,
            rf"${P95_TOKEN}$ = {value_label}",
            ha=ha, va="bottom",
            fontsize=FONT["annotation"], color=color, rotation=90,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=(1.0, 1.0, 1.0, 0.7), edgecolor="none"),
            zorder=5,
        )
    else:
        if y_frac is None:
            y_frac = 0.15
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            x_text, y_frac,
            rf"${P95_TOKEN}$ = {value_label}",
            transform=trans, ha=ha, va="bottom",
            fontsize=FONT["annotation"], color=color, rotation=90,
            bbox=dict(boxstyle="round,pad=0.16", facecolor=(1.0, 1.0, 1.0, 0.7), edgecolor="none"),
            zorder=5,
        )


def cde_legend_items():
    handles = [
        Line2D([0], [0], color=PF_COLOR, linewidth=2.5, linestyle=HIST_LINESTYLE),
        Line2D([0], [0], color=NPF_COLOR, linewidth=2.5, linestyle=HIST_LINESTYLE),
        Line2D([0], [0], color=PF_COLOR, linewidth=2.5, linestyle="-"),
        Line2D([0], [0], color=NPF_COLOR, linewidth=2.5, linestyle="-"),
        Line2D([0], [0], color="0.15", linewidth=COMBINED_LINEWIDTH, linestyle="-"),
    ]
    labels = [
        rf"PF core ($\leq {P95_TOKEN}$)",
        rf"NPF core ($\leq {P95_TOKEN}$)",
        rf"PF extreme ($> {P95_TOKEN}$)",
        rf"NPF extreme ($> {P95_TOKEN}$)",
        "All (combined)",
    ]
    return handles, labels


def add_cde_shared_legend(fig) -> None:
    handles, labels = cde_legend_items()
    legend = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(handles),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        handlelength=3.0,
        handletextpad=0.55,
        columnspacing=1.25,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("0.70")
    legend.get_frame().set_linewidth(0.8)


def _draw_domain_hist(ax, hist_vals, centers, widths, *, color, p95, zorder):
    core_color = soften_color(color, blend=HIST_CORE_BLEND)
    if p95 is None:
        ax.bar(
            centers, hist_vals, width=widths,
            color=core_color, alpha=HIST_CORE_ALPHA,
            edgecolor="none", align="center", zorder=zorder,
        )
        ax.plot(
            centers, hist_vals, color=color,
            linewidth=1.8, linestyle=HIST_LINESTYLE, zorder=zorder + 1.0,
        )
    else:
        core_mask = centers < p95
        tail_mask = ~core_mask
        if core_mask.any():
            ax.bar(
                centers[core_mask], hist_vals[core_mask], width=widths[core_mask],
                color=core_color, alpha=HIST_CORE_ALPHA,
                edgecolor="none", align="center", zorder=zorder,
            )
        if tail_mask.any():
            ax.bar(
                centers[tail_mask], hist_vals[tail_mask], width=widths[tail_mask],
                color=color, alpha=HIST_TAIL_ALPHA,
                edgecolor="none", align="center", zorder=zorder + 0.05,
            )
        core_x, core_y, tail_x, tail_y = split_xy_at_threshold(centers, hist_vals, split_x=p95)
        if core_x.size:
            ax.plot(core_x, core_y, color=color, linewidth=1.8, linestyle=HIST_LINESTYLE, zorder=zorder + 1.0)
        if tail_x.size:
            ax.plot(tail_x, tail_y, color=color, linewidth=1.9, linestyle="-", zorder=zorder + 1.1)


def plot_hist_panel(ax, values: pd.DataFrame, *, column: str, xlabel: str, title: str) -> None:
    pf_vals = clip_percentile(values.loc[values["domain"].eq("pf"), column], 0.5, 99.5)
    npf_vals = clip_percentile(values.loc[values["domain"].eq("npf"), column], 0.5, 99.5)
    all_vals = clip_percentile(values[column], 0.5, 99.5)
    if pf_vals.empty or npf_vals.empty:
        return

    n_bins = HIST_BINS_DU if "du" in column else HIST_BINS_GRAD
    bins = build_hist_bins(
        pd.concat([pf_vals, npf_vals], ignore_index=True), n_bins, force_lo=0.0,
    )
    if len(bins) < 2:
        return

    h_pf, edges = np.histogram(pf_vals, bins=bins, density=True)
    h_npf, _ = np.histogram(npf_vals, bins=bins, density=True)
    h_all, _ = np.histogram(all_vals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    p95_pf = upper_percentile(pf_vals, 95.0)
    p95_npf = upper_percentile(npf_vals, 95.0)
    p95_all = upper_percentile(all_vals, 95.0)

    _draw_domain_hist(ax, h_pf, centers, widths, color=PF_COLOR, p95=p95_pf, zorder=2.0)
    _draw_domain_hist(ax, h_npf, centers, widths, color=NPF_COLOR, p95=p95_npf, zorder=2.1)

    ax.plot(centers, h_all, color="0.15", linewidth=COMBINED_LINEWIDTH, zorder=3.0)

    ymax = max(
        float(np.nanmax(h_pf)) if h_pf.size else 0.0,
        float(np.nanmax(h_npf)) if h_npf.size else 0.0,
        float(np.nanmax(h_all)) if h_all.size else 0.0,
    )
    ax.set_xlim(0.0, float(bins[-1]))
    ax.set_ylim(0.0, 1.08 * ymax if ymax > 0.0 else 1.0)

    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="0.92", linewidth=0.5)

    annotate_p95(ax, xpos=p95_pf, color=PF_COLOR, y_data=0.1, linewidth=1.4)
    annotate_p95(ax, xpos=p95_npf, color=NPF_COLOR, y_data=0.1, linewidth=1.4)
    annotate_p95(ax, xpos=p95_all, color="0.15", y_data=0.1, linewidth=0.8)


# ---------------------------------------------------------------------------
# relationship panel (E — styled after Figure_reorganize_pf_vs_npf_in_du_and_gradient)
# ---------------------------------------------------------------------------

def make_binned_curve(df, *, xcol, ycol, q, bin_edges=None) -> pd.DataFrame:
    tmp = df[[xcol, ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < q:
        return pd.DataFrame(columns=["x", "y", "ylo", "yhi"])
    if bin_edges is None:
        tmp["bin"] = pd.qcut(tmp[xcol], q=q, duplicates="drop")
        out = tmp.groupby("bin", observed=True).agg(
            x=(xcol, "median"), y=(ycol, "mean"), ystd=(ycol, "std"),
        ).reset_index(drop=True)
    else:
        tmp["bin"] = pd.cut(tmp[xcol], bins=bin_edges, include_lowest=True, right=True)
        out = tmp.groupby("bin", observed=True).agg(
            y=(ycol, "mean"), ystd=(ycol, "std"),
        ).reset_index()
        if out.empty:
            return pd.DataFrame(columns=["x", "y", "ylo", "yhi"])
        out["x"] = np.asarray(
            [float(iv.right) if pd.notna(iv) else np.nan for iv in out["bin"]],
            dtype=float,
        )
        out = out.drop(columns="bin")
    out["ystd"] = out["ystd"].fillna(0.0)
    band = 0.1 * out["ystd"]
    out["ylo"] = out["y"] - band
    out["yhi"] = out["y"] + band
    return out[["x", "y", "ylo", "yhi"]]


def split_curve_at_threshold(curve, *, split_x):
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
    core = pd.concat(
        [ordered.loc[ordered["x"] < split_x], pd.DataFrame([boundary])],
        ignore_index=True,
    )
    tail = pd.concat(
        [pd.DataFrame([boundary]), ordered.loc[ordered["x"] > split_x]],
        ignore_index=True,
    )
    return core, tail


def plot_relationship(ax, df: pd.DataFrame) -> None:
    curve_inputs = df[["abs_du", "grad_mag_km", "domain"]].replace([np.inf, -np.inf], np.nan).dropna()
    if curve_inputs.empty:
        return

    y_all = clip_percentile(curve_inputs["grad_mag_km"], 0.5, 99.5)
    x_hi = 20.0
    y_hi = float(y_all.max()) if not y_all.empty else 20.0
    bin_edges = np.linspace(0.0, x_hi, max(6, CURVE_Q) + 1)
    p95_positions: dict[str, float] = {}

    for domain, color in [("pf", PF_COLOR), ("npf", NPF_COLOR)]:
        sub_all = curve_inputs.loc[curve_inputs["domain"] == domain, ["abs_du", "grad_mag_km"]].copy()
        if sub_all.empty:
            continue
        tail_start = upper_percentile(sub_all["abs_du"], 95.0)
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
        curve = make_binned_curve(sub, xcol="abs_du", ycol="grad_mag_km", q=CURVE_Q, bin_edges=bin_edges)
        if curve.empty:
            continue
        core_color = soften_color(color, blend=REL_CORE_BLEND)
        core_curve, tail_curve = split_curve_at_threshold(curve, split_x=tail_start)
        if not core_curve.empty:
            ax.fill_between(
                core_curve["x"], core_curve["ylo"], core_curve["yhi"],
                color=core_color, alpha=REL_CORE_FILL_ALPHA, zorder=1,
            )
            ax.plot(
                core_curve["x"], core_curve["y"],
                color=color, linewidth=1.9, linestyle=HIST_LINESTYLE, zorder=2,
            )
        if not tail_curve.empty:
            ax.fill_between(
                tail_curve["x"], tail_curve["ylo"], tail_curve["yhi"],
                color=color, alpha=REL_TAIL_FILL_ALPHA, zorder=3,
            )
            ax.plot(
                tail_curve["x"], tail_curve["y"],
                color=color, linewidth=2.25, linestyle="-", zorder=4,
            )

    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 19.0)
    ax.set_title(r"$|d_u|$ vs $|\nabla d_u|$", pad=6)
    ax.set_xlabel(r"$|d_u|$ for $d_u \leq 0$ (mm yr$^{-1}$)")
    ax.set_ylabel(r"Mean $|\nabla d_u|$ (mm yr$^{-1}$ km$^{-1}$)")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15])
    ax.grid(color="0.92", linewidth=0.5)

    for domain, color in [("pf", PF_COLOR), ("npf", NPF_COLOR)]:
        if domain in p95_positions:
            annotate_p95(ax, xpos=p95_positions[domain], color=color, y_frac=0.38, linewidth=1.4)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_style()
    du = load_du_raster()
    grad = load_grad_raster()
    du_profile, grad_profile = load_profiles()
    sample = load_samples()
    sites = load_meteoro_sites()
    profile_sites = load_profile_sites()

    _, stride_du = decimate_raster(np.asarray(du), target_max=1400)
    lon_plot, lat_plot = build_lonlat_mesh(stride=stride_du)

    fig = plt.figure(figsize=(11.8, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 0.9], hspace=0.28, bottom=0.145)
    top = gs[0].subgridspec(1, 2, wspace=0.22)
    top_a = top[0].subgridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)
    top_b = top[1].subgridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)
    bottom = gs[1].subgridspec(1, 3, wspace=0.26)

    ax_a = fig.add_subplot(top_a[0, 0])
    ax_ap = fig.add_subplot(top_a[0, 1])
    ax_b = fig.add_subplot(top_b[0, 0])
    ax_bp = fig.add_subplot(top_b[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])
    ax_e = fig.add_subplot(bottom[0, 2])

    lo, hi = symmetric_limits(np.asarray(du), p=98.5)
    plot_map(
        ax_a,
        raster=np.asarray(du),
        raster_path=DU_RASTER_PATH,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
        title=r"Vertical deformation rate $d_u$",
        cbar_label=r"$d_u$ (mm yr$^{-1}$)",
        title_color=DU_COLOR,
        sites=sites,
        lon_plot=lon_plot, lat_plot=lat_plot,
    )
    plot_profile(
        ax_ap, du_profile, color=DU_COLOR,
        title="1 km railway buffer profile",
        xlabel="Mean $d_u$\n(mm/yr)",
        show_ylabel=True,
        profile_sites=profile_sites,
    )
    color_profile_axis(ax_ap, DU_COLOR)

    grad_km = np.asarray(grad, dtype=float) * 1000.0  # mm/yr/m -> mm/yr/km
    plot_map(
        ax_b,
        raster=grad_km,
        raster_path=GRAD_RASTER_PATH,
        cmap="Reds",
        norm=plt.Normalize(vmin=0.0, vmax=32.0),
        title=r"Spatial gradient magnitude $|\nabla d_u|$",
        cbar_label=r"$|\nabla d_u|$ (mm yr$^{-1}$ km$^{-1}$)",
        title_color=GRAD_COLOR,
        sites=sites,
        lon_plot=lon_plot, lat_plot=lat_plot,
        cbar_ticks=[0, 10, 20, 30],
    )
    plot_profile(
        ax_bp, grad_profile, color=GRAD_COLOR,
        title="1 km railway buffer profile",
        xlabel=r"Mean $|\nabla d_u|$" + "\n(mm/yr/km)",
        show_ylabel=True,
        profile_sites=profile_sites,
    )
    color_profile_axis(ax_bp, GRAD_COLOR)

    plot_hist_panel(
        ax_c, sample,
        column="abs_du",
        xlabel=r"$|d_u|$ for $d_u<0$ (mm yr$^{-1}$)",
        title="PF vs NPF subsidence distributions",
    )
    plot_hist_panel(
        ax_d, sample,
        column="grad_mag_km",
        xlabel=r"$|\nabla d_u|$ (mm yr$^{-1}$ km$^{-1}$)",
        title="PF vs NPF gradient distributions",
    )
    plot_relationship(ax_e, sample)
    color_metric_axis(ax_c, "x", DU_COLOR)
    color_metric_axis(ax_d, "x", GRAD_COLOR)
    color_metric_axis(ax_e, "x", DU_COLOR)
    color_metric_axis(ax_e, "y", GRAD_COLOR)

    for ax, label in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (ax_e, "E")]:
        add_panel_label(ax, label, x=-0.1, y=1.02)

    add_cde_shared_legend(fig)
    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
