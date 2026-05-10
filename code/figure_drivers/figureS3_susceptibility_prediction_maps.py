#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/FigureS3_susceptibility_prediction_maps.py
# Renamed package path: code/figure_drivers/figureS3_susceptibility_prediction_maps.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from submission_build_common import (
    FONT,
    ROOT_DIR,
    SOURCE_CACHE_DIR,
    add_panel_label,
    add_scalebar_lonlat,
    blend_with_white,
    clip_gdf_to_hull,
    coverage_hull_lonlat_from_path,
    decimate_raster,
    en_to_rc,
    ensure_style,
    grid_meta,
    load_geo_layers,
    load_json,
    open_memmap,
    project_crs,
    save_figure,
    text_halo,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pyproj import Transformer


FIG_STEM = "FigureS3_susceptibility_prediction_maps"
META_PATH = SOURCE_CACHE_DIR / "Figure_reorganize_extreme_deformation_susceptibity_meta.json"
PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
DU_COLOR = "#1E5BAA"
GRAD_COLOR = "#C1272D"
METEORO_COLOR = "#3A3A3A"
METEORO_SIZE = 26
RAIL_DASHES = (5, 3)
RAILWAY_LEGEND_LON = 92.5
RAILWAY_LEGEND_LAT = 28.0
COLORBAR_RECT = [0.55, 0.155, 0.20, 0.032]
EXTENT_LONLAT = [88.18, 97.86, 27.8, 38.5]
METEORO_SHP = ROOT_DIR / "human_features" / "qtec_meteoro_station_sites.shp"
PROFILE_SAMPLE_PATH = SOURCE_CACHE_DIR / "Figure_reorganize_railway_buffer_analysis_corridor_sample_1p0km.csv.gz"
PROFILE_BIN_KM = 10.0
PROFILE_STD_SCALE = 0.1

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


def predictor_path(target: str, domain: str):
    meta = load_json(META_PATH)
    return meta["targets"][target]["predictor_probability_rasters"][domain]


def shared_limits(target: str) -> tuple[float, float]:
    vals = []
    for domain in ["pf", "npf"]:
        arr = np.asarray(open_memmap(Path(predictor_path(target, domain)))[::24, ::24], dtype=float)
        arr = arr[np.isfinite(arr) & ~np.isclose(arr, 0.0)]
        vals.append(arr)
    joined = np.concatenate(vals)
    lo, hi = np.percentile(joined, [5, 95])
    return float(lo), float(hi)


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


@lru_cache(maxsize=4)
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
    transformer = Transformer.from_crs(project_crs(), "EPSG:4326", always_xy=True)
    lon_grid, lat_grid = transformer.transform(e_grid, n_grid)
    return (
        np.flipud(np.asarray(lon_grid, dtype=float)),
        np.flipud(np.asarray(lat_grid, dtype=float)),
    )


@lru_cache(maxsize=1)
def load_meteoro_sites() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(METEORO_SHP)
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


@lru_cache(maxsize=1)
def load_corridor_sample() -> pd.DataFrame:
    return pd.read_csv(PROFILE_SAMPLE_PATH)


@lru_cache(maxsize=1)
def load_profile_sites() -> pd.DataFrame:
    try:
        import Figure_reorganized_railway_extreme_deformation_inspection as railinspect
        import Figure_reorganize_railway_buffer_analysis as railbuf

        railway_context = railbuf.load_railway_context(ROOT_DIR / "human_features" / "qtec_railway_clip.shp")
        return railinspect.load_profile_sites(METEORO_SHP, railway_context=railway_context)
    except Exception:
        return pd.DataFrame(columns=["site_label", "along_km"])


def style_map_axis(ax) -> None:
    ax.set_xlim(EXTENT_LONLAT[0], EXTENT_LONLAT[1])
    ax.set_ylim(EXTENT_LONLAT[2], EXTENT_LONLAT[3])
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("auto")
    ax.tick_params(length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
        ax=ax,
        color="black",
        linewidth=1.5,
        linestyle=(0, RAIL_DASHES),
        alpha=1.0,
        zorder=5,
    )


def add_railway_legend(ax) -> None:
    x0, x1 = float(RAILWAY_LEGEND_LON), float(RAILWAY_LEGEND_LON) + 1.45
    y0 = float(RAILWAY_LEGEND_LAT)
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.8, zorder=8)
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5, linestyle=(0, RAIL_DASHES), zorder=9)
    ax.text(
        x1 + 0.12,
        y0,
        "Railway",
        ha="left",
        va="center",
        fontsize=FONT["annotation"],
        color="0.1",
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(1.0, 1.0, 1.0, 0.78), edgecolor="none"),
        zorder=10,
    )


def add_inset_colorbar(ax, mappable, *, label: str, ticks=None) -> None:
    cax = ax.inset_axes(COLORBAR_RECT, transform=ax.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    if ticks is not None:
        cb.set_ticks(ticks)
    else:
        cb.locator = MaxNLocator(nbins=3)
        cb.update_ticks()
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(labelsize=FONT["annotation"], length=2, pad=1)
    cb.set_label(label, fontsize=FONT["annotation"], labelpad=1)


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
        s=METEORO_SIZE,
        marker="s",
        color=METEORO_COLOR,
        edgecolors=METEORO_COLOR,
        linewidths=0.0,
        zorder=5.3,
    )
    for site in visible.itertuples(index=False):
        label = str(site.site_label).strip()
        dx, dy, ha = STATION_LABEL_SPECS.get(label, (5.0, 3.0, "left"))
        text = ax.annotate(
            label,
            xy=(float(site.geometry.x), float(site.geometry.y)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=FONT["annotation"] - 0.3,
            color=METEORO_COLOR,
            zorder=6.2,
        )
        text.set_path_effects(text_halo(2.0))


def add_zoom_insets(ax, *, plot_arr, lon_plot, lat_plot, cmap, vmin, vmax, sites, railway_gdf) -> None:
    for spec in MAP_ZOOM_SPECS:
        match = sites.loc[sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site = match.iloc[0]
        lon, lat = float(site.geometry.x), float(site.geometry.y)
        inset = ax.inset_axes(spec["bounds"])
        inset.pcolormesh(
            lon_plot,
            lat_plot,
            plot_arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
            zorder=0,
        )
        if not railway_gdf.empty:
            railway_gdf.plot(ax=inset, color="white", linewidth=2.8, alpha=1.0, zorder=4)
            railway_gdf.plot(
                ax=inset,
                color="black",
                linewidth=1.5,
                linestyle=(0, RAIL_DASHES),
                alpha=1.0,
                zorder=5,
            )
        inset.set_xlim(lon - spec["xpad"], lon + spec["xpad"])
        inset.set_ylim(lat - spec["ypad"], lat + spec["ypad"])
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.scatter(
            [lon],
            [lat],
            s=METEORO_SIZE * 2,
            marker="s",
            color=METEORO_COLOR,
            linewidths=0.0,
            zorder=5.3,
        )
        text = inset.annotate(
            str(site.site_label),
            xy=(lon, lat),
            xytext=(4.0, 2.0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=FONT["annotation"],
            color=METEORO_COLOR,
            zorder=6.2,
        )
        text.set_path_effects(text_halo(2.0))


def attach_raster_values_to_sample(sample_df: pd.DataFrame, *, path: str, out_col: str) -> pd.DataFrame:
    out = sample_df.copy()
    mm = open_memmap(Path(path))
    easting = pd.to_numeric(out["easting"], errors="coerce").to_numpy(dtype=float)
    northing = pd.to_numeric(out["northing"], errors="coerce").to_numpy(dtype=float)
    row, col = en_to_rc(easting, northing)
    nrows, ncols = mm.shape
    ok = (
        np.isfinite(easting)
        & np.isfinite(northing)
        & (row >= 0)
        & (row < nrows)
        & (col >= 0)
        & (col < ncols)
    )
    values = np.full(len(out), np.nan, dtype=float)
    if np.any(ok):
        values[ok] = np.asarray(mm[row[ok], col[ok]], dtype=float)
    values[np.isclose(values, 0.0)] = np.nan
    out[out_col] = values
    return out


def summarize_metric_profile(
    sample_df: pd.DataFrame,
    *,
    value_col: str,
    bin_length_km: float = PROFILE_BIN_KM,
    std_scale: float = PROFILE_STD_SCALE,
) -> pd.DataFrame:
    tmp = sample_df[["along_km", value_col]].copy()
    tmp["along_km"] = pd.to_numeric(tmp["along_km"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return pd.DataFrame(columns=["bin_left_km", "bin_right_km", "center_km", "mean", "std", "ylo", "yhi", "n"])

    max_dist = float(tmp["along_km"].max())
    if not np.isfinite(max_dist) or max_dist <= 0.0:
        max_dist = float(bin_length_km)
    edges = np.arange(0.0, max_dist + bin_length_km, bin_length_km)
    if len(edges) < 2:
        edges = np.array([0.0, float(bin_length_km)], dtype=float)
    if edges[-1] <= max_dist:
        edges = np.append(edges, edges[-1] + bin_length_km)

    tmp["bin"] = pd.cut(tmp["along_km"], edges, include_lowest=True, right=True)
    out = (
        tmp.groupby("bin", observed=True)
        .agg(mean=(value_col, "mean"), std=(value_col, "std"), n=(value_col, "size"))
        .reset_index()
    )
    if out.empty:
        return pd.DataFrame(columns=["bin_left_km", "bin_right_km", "center_km", "mean", "std", "ylo", "yhi", "n"])

    out["std"] = pd.to_numeric(out["std"], errors="coerce").fillna(0.0)
    out["bin_left_km"] = np.asarray([float(iv.left) for iv in out["bin"]], dtype=float)
    out["bin_right_km"] = np.asarray([float(iv.right) for iv in out["bin"]], dtype=float)
    out["center_km"] = 0.5 * (out["bin_left_km"] + out["bin_right_km"])
    band = float(std_scale) * out["std"]
    out["ylo"] = out["mean"] - band
    out["yhi"] = out["mean"] + band
    out = out.drop(columns="bin")
    return out[["bin_left_km", "bin_right_km", "center_km", "mean", "std", "ylo", "yhi", "n"]]


@lru_cache(maxsize=8)
def build_panel_profile(path: str) -> pd.DataFrame:
    sample_df = attach_raster_values_to_sample(load_corridor_sample(), path=path, out_col="susceptibility")
    return summarize_metric_profile(sample_df, value_col="susceptibility")


def profile_xlim(*profiles: pd.DataFrame) -> tuple[float, float]:
    hi = 0.0
    for prof in profiles:
        if prof.empty:
            continue
        vals = pd.to_numeric(prof["yhi"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            hi = max(hi, float(np.nanmax(vals)))
    hi = max(hi, 0.05)
    return (0.0, 1.08 * hi)


def profile_ylim(*profiles: pd.DataFrame) -> tuple[float, float]:
    hi = 0.0
    for prof in profiles:
        if prof.empty:
            continue
        vals = pd.to_numeric(prof["bin_right_km"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            hi = max(hi, float(np.nanmax(vals)))
    hi = max(hi, 1.0)
    return (0.0, hi)


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
            0.97,
            km,
            str(site.site_label),
            transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="right",
            va="center",
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
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    profile_sites: pd.DataFrame | None = None,
) -> None:
    if prof.empty:
        ax.set_title(title, fontsize=FONT["axis"], pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Distance along railway from Lhasa (km)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return
    ax.fill_betweenx(
        prof["center_km"],
        prof["ylo"],
        prof["yhi"],
        color=blend_with_white(color, 0.72),
        alpha=0.7,
        linewidth=0,
    )
    ax.plot(prof["mean"], prof["center_km"], color=color, linewidth=1.6)
    ax.set_title(title, fontsize=FONT["axis"], pad=6)
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_linespacing(1.0)
    ax.set_ylabel("Distance along railway from Lhasa (km)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if profile_sites is not None:
        annotate_profile_sites(ax, profile_sites)


def plot_panel(
    ax,
    *,
    path: str,
    cmap: str,
    title: str,
    title_color: str,
    vmin: float,
    vmax: float,
    sites: gpd.GeoDataFrame,
    railway_ll: gpd.GeoDataFrame,
) -> None:
    arr = np.asarray(open_memmap(Path(path)), dtype=float)
    arr_ds, stride = decimate_raster(arr, target_max=1400)
    plot_arr = np.flipud(arr_ds)
    plot_arr = np.ma.masked_invalid(np.where(np.isclose(plot_arr, 0.0), np.nan, plot_arr))
    lon_plot, lat_plot = build_lonlat_mesh(stride=stride)
    plot_cmap = plt.get_cmap(cmap).copy()
    plot_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    coverage = coverage_hull_lonlat_from_path(str(path))
    railway = clip_gdf_to_hull(railway_ll, coverage)

    ax.set_facecolor("white")
    im = ax.pcolormesh(
        lon_plot,
        lat_plot,
        plot_arr,
        cmap=plot_cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
        zorder=0,
    )
    plot_dashed_railway(ax, railway)
    style_map_axis(ax)
    draw_meteoro_sites(ax, sites)
    ax.set_title(title, pad=6, color=title_color)
    add_scalebar_lonlat(ax, length_km=100, lon=89.0, lat=34.0)
    add_railway_legend(ax)
    add_inset_colorbar(ax, im, label="Susceptibility")
    add_zoom_insets(
        ax,
        plot_arr=plot_arr,
        lon_plot=lon_plot,
        lat_plot=lat_plot,
        cmap=plot_cmap,
        vmin=vmin,
        vmax=vmax,
        sites=sites,
        railway_gdf=railway,
    )


def main() -> None:
    ensure_style()
    profile_sites = load_profile_sites()
    du_lo, du_hi = shared_limits("d_u")
    grad_lo, grad_hi = shared_limits("grad_mag_km")
    layers = load_geo_layers()
    sites = load_meteoro_sites()
    panel_specs = [
        ("A", "d_u", "pf", "Blues", r"PF model · extreme $d_u$"),
        ("B", "d_u", "npf", "Blues", r"NPF model · extreme $d_u$"),
        ("C", "grad_mag_km", "pf", "Reds", r"PF model · extreme $|\nabla d_u|$"),
        ("D", "grad_mag_km", "npf", "Reds", r"NPF model · extreme $|\nabla d_u|$"),
    ]
    profiles = {
        (target, domain): build_panel_profile(predictor_path(target, domain))
        for _, target, domain, _, _ in panel_specs
    }
    du_xlim = profile_xlim(profiles[("d_u", "pf")], profiles[("d_u", "npf")])
    grad_xlim = profile_xlim(profiles[("grad_mag_km", "pf")], profiles[("grad_mag_km", "npf")])
    du_ylim = profile_ylim(profiles[("d_u", "pf")], profiles[("d_u", "npf")])
    grad_ylim = profile_ylim(profiles[("grad_mag_km", "pf")], profiles[("grad_mag_km", "npf")])

    fig = plt.figure(figsize=(14.3, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.045, right=0.985, top=0.94, bottom=0.08, wspace=0.20, hspace=0.22)

    for idx, (label, target, domain, cmap, title) in enumerate(panel_specs):
        row, col = divmod(idx, 2)
        sub = gs[row, col].subgridspec(1, 2, width_ratios=[4.2, 1.1], wspace=0.12)
        ax_map = fig.add_subplot(sub[0, 0])
        ax_prof = fig.add_subplot(sub[0, 1])
        plot_panel(
            ax_map,
            path=predictor_path(target, domain),
            cmap=cmap,
            title=title,
            title_color=DU_COLOR if target == "d_u" else GRAD_COLOR,
            vmin=du_lo if target == "d_u" else grad_lo,
            vmax=du_hi if target == "d_u" else grad_hi,
            sites=sites,
            railway_ll=layers["railway_ll"],
        )
        plot_profile(
            ax_prof,
            profiles[(target, domain)],
            color=DU_COLOR if target == "d_u" else GRAD_COLOR,
            title="1 km railway buffer profile",
            xlabel="mean\nsusceptibility",
            xlim=du_xlim if target == "d_u" else grad_xlim,
            ylim=du_ylim if target == "d_u" else grad_ylim,
            profile_sites=profile_sites,
        )
        color_profile_axis(ax_prof, DU_COLOR if target == "d_u" else GRAD_COLOR)
        add_panel_label(ax_map, label, x=-0.10, y=1.02)

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
