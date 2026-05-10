#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure_reorganize_du_du_gradient_ml_features.py
# Renamed package path: code/original_project_helpers/deformation_environment_feature_workflow.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.ticker import FuncFormatter
from pyproj import CRS, Transformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import shapefile
except Exception as exc:
    raise RuntimeError("pyshp is required for the railway overlay.") from exc

import Figure_reorganized_railway_extreme_deformation_inspection as railinspect
import figure4_regional_deformation_context as fig4
import figure5_2_du_attribution_by_basin as fig5

SEED = 42
CHUNKSIZE = 400_000
FIG_BASENAME = "Figure_reorganize_du_du_gradient_ml_features"
DEFAULT_SAMPLE_TOTAL = 220_000
TARGET_LABELS = {
    "d_u": r"$d_u$",
    "grad_mag_km": r"$|\nabla d_u|$",
}
TARGET_UNITS = {
    "d_u": "mm/yr",
    "grad_mag_km": "mm/yr/km",
}
GRADIENT_THEME_COLOR = "#B33A3A"
ML_BLOCK_WIDTH_RATIOS = (3.2, 2.0)
BOTTOM_ROW_WIDTH_RATIOS = (1.5, 1.0, 1.0)
GRAD_SCATTER_LIMITS = (0.0, 31.0)
IMPORTANCE_BAR_BLEND = 0.84
MAP_LAT_LIMITS = (27.8, 38.5)
MAP_LAT_TICKS = [28, 30, 32, 34, 36, 38]
MAP_COLORBAR_HEIGHT = 0.032
MAP_COLORBAR_TOP_LAT = 29.8
MAP_COLORBAR_Y0 = (
    (MAP_COLORBAR_TOP_LAT - MAP_LAT_LIMITS[0]) / (MAP_LAT_LIMITS[1] - MAP_LAT_LIMITS[0]) - MAP_COLORBAR_HEIGHT
)
SCALE_BAR_LON = 89.0
SCALE_BAR_LAT = 37.0
SCALE_BAR_KM = 250.0
DU_SCATTER_LABEL_TICKS = (-15.0, -10.0, -5.0, 0.0, 5.0)
RAILWAY_LEGEND_LON = 94.0
RAILWAY_LEGEND_LAT = 28.5
TOP_ROW_LABELS = ("a", "b")
BOTTOM_ROW_LABELS = ("c", "d", "e")
PANEL_LABEL_X = -0.08
PANEL_LABEL_Y = 1.03
MAP_COLORBAR_RECT = [0.67, MAP_COLORBAR_Y0, 0.20, MAP_COLORBAR_HEIGHT]
METEORO_SITE_COLOR = "#3A3A3A"
METEORO_SITE_SIZE = 36.0
METEORO_INSET_SITE_SIZE = 62.0
A4_WIDTH_IN = 8.67
DEFAULT_FIG_ASPECT = 9.09/8.67
DEFAULT_FIGSIZE = (A4_WIDTH_IN, A4_WIDTH_IN * DEFAULT_FIG_ASPECT)
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


def engineer_ml_sample(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Perma_Distr_map" in out.columns:
        out["Perma_Distr_map"] = pd.to_numeric(out["Perma_Distr_map"], errors="coerce")
    if "Main-Basin" in out.columns:
        out["Main-Basin"] = out["Main-Basin"].astype("string").fillna("missing")
    else:
        out["Main-Basin"] = pd.Series("missing", index=out.index, dtype="string")
    if "d_u" in out.columns:
        out["d_u"] = pd.to_numeric(out["d_u"], errors="coerce")
    if "grad_mag" in out.columns:
        out["grad_mag"] = pd.to_numeric(out["grad_mag"], errors="coerce")
    if "grad_mag_km" in out.columns:
        out["grad_mag_km"] = pd.to_numeric(out["grad_mag_km"], errors="coerce")
    elif "grad_mag" in out.columns:
        out["grad_mag_km"] = out["grad_mag"] * 1000.0
    for col in fig5.FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["domain"] = np.where(
        out["Perma_Distr_map"] == 1,
        "pf",
        np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
    )
    return out


def resolve_ml_sample(
    csv_path: Path,
    cache_dir: Path,
    sample_total: int,
    chunksize: int = CHUNKSIZE,
) -> tuple[pd.DataFrame, Path]:
    required_cols = {"Perma_Distr_map", "Main-Basin", "d_u", "grad_mag", "grad_mag_km", *fig5.FEATURES}
    dedicated_cache = cache_dir / f"{FIG_BASENAME}_pf_sample.csv.gz"
    cache_candidates = [
        dedicated_cache,
        cache_dir / "figure7_2_context_attr_pf_sample.csv.gz",
    ]

    for cache_path in cache_candidates:
        if not cache_path.exists():
            continue
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if not required_cols.issubset(cache_cols):
            continue
        df = engineer_ml_sample(pd.read_csv(cache_path))
        if len(df) > sample_total:
            df = df.sample(sample_total, random_state=SEED).reset_index(drop=True)
        return df, cache_path

    usecols = list(
        dict.fromkeys(
            ["Perma_Distr_map", "Main-Basin", "d_u", "grad_mag", "grad_mag_km"] + fig5.FEATURES
        )
    )
    df = fig5.build_pf_sample(
        csv_path=csv_path,
        usecols=usecols,
        sample_total=sample_total,
        chunksize=chunksize,
        seed=SEED,
    )
    df = engineer_ml_sample(df)
    dedicated_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dedicated_cache, index=False, compression="gzip")
    return df, dedicated_cache


def fit_target_model(df: pd.DataFrame, target: str, model_name: str) -> dict:
    cols = fig5.FEATURES + [target]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    keep_mask = np.isfinite(sub[target].to_numpy(dtype=float))
    sub = sub.loc[keep_mask].copy()
    if len(sub) < 300:
        raise RuntimeError(f"Not enough rows to fit model: {model_name} [{target}]")

    X = sub[fig5.FEATURES].copy()
    y = sub[target].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = ExtraTreesRegressor(
        n_estimators=800,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=0.8,
    )
    model.fit(X_train_imp, y_train)
    pred = model.predict(X_test_imp)

    return {
        "name": model_name,
        "target": target,
        "model": model,
        "imputer": imputer,
        "X_train_df": X_train.reset_index(drop=True),
        "X_test_df": X_test.reset_index(drop=True),
        "y_test": y_test,
        "pred_test": pred,
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "feature_importance": pd.Series(model.feature_importances_, index=fig5.FEATURES).sort_values(
            ascending=False
        ),
    }


def build_density_cmap(base_color: str, name: str) -> LinearSegmentedColormap:
    rgb = np.asarray(to_rgb(base_color), dtype=float)
    light = 1.0 - 0.68 * (1.0 - rgb)
    return LinearSegmentedColormap.from_list(name, ["#ffffff", tuple(light), tuple(rgb)])


def gradient_scatter_limits(result: dict) -> tuple[float, float]:
    vals = np.concatenate(
        [
            np.asarray(result["y_test"], dtype=float),
            np.asarray(result["pred_test"], dtype=float),
        ]
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    lo = max(0.0, float(lo))
    hi = float(hi)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    hi *= 1.04
    return (lo, hi)


def plot_obs_pred_target(
    ax,
    result: dict,
    *,
    target_label: str,
    units: str,
    title: str,
    scatter_limits: tuple[float, float],
    base_color: str,
    equal_aspect: bool = True,
) -> None:
    obs = np.asarray(result["y_test"], dtype=float)
    pred = np.asarray(result["pred_test"], dtype=float)
    lo, hi = scatter_limits
    x_sc, y_sc, density_sc = fig5.estimate_relative_density(obs, pred, lo=lo, hi=hi)
    density_vmax = float(np.nanmax(density_sc)) if density_sc.size else 1.0

    ax.scatter(
        x_sc,
        y_sc,
        c=density_sc,
        cmap=build_density_cmap(base_color, f"density_{result['target']}"),
        vmin=0.0,
        vmax=density_vmax if density_vmax > 0 else 1.0,
        s=9,
        alpha=0.9,
        linewidths=0,
        rasterized=len(x_sc) > 3000,
    )
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.35", linewidth=1.0)

    fit_mask = (
        np.isfinite(obs)
        & np.isfinite(pred)
        & (obs >= lo)
        & (obs <= hi)
        & (pred >= lo)
        & (pred <= hi)
    )
    if fit_mask.sum() >= 10:
        a, b = np.polyfit(obs[fit_mask], pred[fit_mask], 1)
        xx = np.linspace(lo, hi, 200)
        ax.plot(xx, a * xx + b, linestyle="--", color="#f02c21", linewidth=1.2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    ax.set_xlabel(f"Observed {target_label} ({units})", fontweight="bold")
    ax.set_ylabel(f"Predicted {target_label} ({units})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=4)
    ax.text(
        0.97,
        0.04,
        f"$R^2$={result['r2']:.2f}\nMAE={result['mae']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.18", alpha=0.12),
    )
    fig5.style_open_axes(ax)


def add_plain_panel_label(ax, label: str) -> None:
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


def plot_ml_block(
    fig,
    parent_spec,
    *,
    result: dict,
    title: str,
    panel_label: str,
    base_color: str,
    scatter_limits: tuple[float, float],
    axis_label_ticks: tuple[float, ...] | None = None,
):
    block = parent_spec.subgridspec(1, 2, width_ratios=ML_BLOCK_WIDTH_RATIOS, wspace=0.16)
    ax_sc = fig.add_subplot(block[0, 0])
    ax_imp = fig.add_subplot(block[0, 1])

    plot_obs_pred_target(
        ax_sc,
        result,
        target_label=TARGET_LABELS[result["target"]],
        units=TARGET_UNITS[result["target"]],
        title=title,
        scatter_limits=scatter_limits,
        base_color=base_color,
    )
    fig5.plot_top_importance(
        ax_imp,
        result,
        top_n=10,
        face_color=fig5.lighten_color(base_color, blend=IMPORTANCE_BAR_BLEND),
    )
    apply_importance_axis_format(ax_imp)
    apply_scatter_axis_label_ticks(ax_sc, axis_label_ticks)
    add_plain_panel_label(ax_sc, panel_label)
    return ax_sc, ax_imp


def apply_importance_axis_format(ax) -> None:
    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _pos: (
                "0"
                if np.isclose(x, 0.0, atol=0.003)
                else "0.05"
                if np.isclose(x, 0.05, atol=0.003)
                else "0.10"
                if np.isclose(x, 0.10, atol=0.003)
                else ""
            )
        )
    )


def apply_scatter_axis_label_ticks(ax, axis_label_ticks: tuple[float, ...] | None = None) -> None:
    if axis_label_ticks is None:
        return
    tick_formatter = FuncFormatter(
        lambda x, _pos: (
            f"{int(round(x))}"
            if any(np.isclose(x, ref, atol=0.2) for ref in axis_label_ticks)
            else ""
        )
    )
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_formatter(tick_formatter)


def plot_importance_panel(
    ax,
    *,
    result: dict,
    panel_label: str,
    base_color: str,
) -> None:
    fig5.plot_top_importance(
        ax,
        result,
        top_n=10,
        face_color=fig5.lighten_color(base_color, blend=IMPORTANCE_BAR_BLEND),
    )
    apply_importance_axis_format(ax)
    add_plain_panel_label(ax, panel_label)


def save_importance_table(result: dict, table_dir: Path) -> Path:
    out_path = table_dir / f"{FIG_BASENAME}_{result['target']}_importance.csv"
    out = result["feature_importance"].rename("importance").reset_index()
    out.columns = ["feature", "importance"]
    out.to_csv(out_path, index=False)
    return out_path


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


def summarize_spatial_band_profile_with_coords(
    arr: np.ndarray,
    coord_grid: np.ndarray,
    *,
    band_axis: int,
    max_bins: int = 64,
) -> pd.DataFrame:
    vals = np.asarray(arr, dtype=float)
    coords = np.asarray(coord_grid, dtype=float)
    if vals.ndim != 2 or coords.shape != vals.shape:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    n_bands = int(vals.shape[band_axis])
    if n_bands == 0:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    coord_1d = np.nanmean(coords, axis=0 if band_axis == 1 else 1)
    n_bins = max(2, min(int(max_bins), n_bands))
    rows = []

    for idx in np.array_split(np.arange(n_bands), n_bins):
        if idx.size == 0:
            continue
        subset = vals[idx, :] if band_axis == 0 else vals[:, idx]
        finite = subset[np.isfinite(subset)]
        coord_vals = coord_1d[idx]
        coord_vals = coord_vals[np.isfinite(coord_vals)]
        if finite.size == 0 or coord_vals.size == 0:
            continue
        rows.append(
            {
                "coord": float(np.nanmean(coord_vals)),
                "mean": float(np.nanmean(finite)),
                "std": float(np.nanstd(finite)),
                "n": int(finite.size),
            }
        )

    return pd.DataFrame(rows, columns=["coord", "mean", "std", "n"])


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


def resolve_source_crs(crs_info_path: Path, railway_shp: Path) -> CRS:
    if crs_info_path.exists():
        payload = crs_info_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"CRS:\s*(.+?)<br/>\[INFO\] Transform:", payload, flags=re.DOTALL)
        if match:
            return CRS.from_wkt(match.group(1).strip())

    prj_path = railway_shp.with_suffix(".prj")
    if prj_path.exists():
        return CRS.from_wkt(prj_path.read_text(encoding="utf-8", errors="ignore"))
    raise FileNotFoundError(f"Could not resolve a projected CRS from {crs_info_path} or {prj_path}.")


def format_degree_hemisphere(value: float, *, positive: str, negative: str, bold: bool = False) -> str:
    if not np.isfinite(value):
        return ""
    hemi = positive if value >= 0 else negative
    abs_value = abs(float(value))
    if np.isclose(abs_value, round(abs_value), atol=1e-6):
        coord_text = f"{int(round(abs_value))}"
    else:
        coord_text = f"{abs_value:.1f}".rstrip("0").rstrip(".")
    if bold:
        return rf"$\mathbf{{{coord_text}}}^{{\circ}}\mathbf{{{hemi}}}$"
    return rf"${coord_text}^\circ$ {hemi}"


def apply_requested_map_lat_axis(ax) -> None:
    ax.set_ylim(*MAP_LAT_LIMITS)
    ax.set_yticks(MAP_LAT_TICKS)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: format_degree_hemisphere(x, positive="E", negative="W", bold=True))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _pos: format_degree_hemisphere(y, positive="N", negative="S", bold=True))
    )
    ax.set_ylabel("Latitude")


def zonal_degree_length_km(lat_deg: float) -> float:
    lat_rad = np.deg2rad(float(lat_deg))
    return (
        111.41288 * np.cos(lat_rad)
        - 0.09350 * np.cos(3.0 * lat_rad)
        + 0.00012 * np.cos(5.0 * lat_rad)
    )


def add_scale_bar_lonlat(
    ax,
    *,
    lon0: float = SCALE_BAR_LON,
    lat0: float = SCALE_BAR_LAT,
    length_km: float = SCALE_BAR_KM,
    color: str = "k",
) -> None:
    km_per_deg_lon = zonal_degree_length_km(lat0)
    if not np.isfinite(km_per_deg_lon) or km_per_deg_lon <= 0:
        return

    dlon = float(length_km) / km_per_deg_lon
    lon1 = lon0 + dlon
    cap_half_height = 0.14
    text_pad = 0.18

    ax.plot([lon0, lon1], [lat0, lat0], color=color, linewidth=2.2, solid_capstyle="butt", zorder=6)
    ax.plot([lon0, lon0], [lat0 - cap_half_height, lat0 + cap_half_height], color=color, linewidth=1.4, zorder=6)
    ax.plot([lon1, lon1], [lat0 - cap_half_height, lat0 + cap_half_height], color=color, linewidth=1.4, zorder=6)
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
        zorder=7,
    )


def center_multiline_label(text_obj) -> None:
    text_obj.set_multialignment("center")
    text_obj.set_linespacing(1.0)


def center_multiline_right_ylabel(ax) -> None:
    center_multiline_label(ax.yaxis.label)


def center_multiline_xlabel(ax) -> None:
    center_multiline_label(ax.xaxis.label)


def choose_scale_bar_anchor(extent_lonlat: list[float]) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = map(float, extent_lonlat)
    lon0 = xmin + 0.07 * (xmax - xmin)
    lat0 = ymin + 0.46 * (ymax - ymin)
    return lon0, lat0


def add_horizontal_inset_colorbar(ax, mappable, *, label: str, rect: list[float] = MAP_COLORBAR_RECT) -> None:
    cax = ax.inset_axes(rect, transform=ax.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)

    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.set_label(label, fontsize=8, fontweight="bold", labelpad=1)


def load_railway_segments(railway_shp: Path) -> list[np.ndarray]:
    proj_dir = Path(sys.executable).resolve().parents[1] / "share" / "proj"
    if "PROJ_LIB" not in os.environ and proj_dir.exists():
        os.environ["PROJ_LIB"] = str(proj_dir)

    prj_path = railway_shp.with_suffix(".prj")
    if not prj_path.exists():
        raise FileNotFoundError(f"Missing railway projection file: {prj_path}")

    source_crs = CRS.from_wkt(prj_path.read_text())
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)

    segments: list[np.ndarray] = []
    with shapefile.Reader(str(railway_shp)) as shp:
        for geom in shp.shapes():
            points = np.asarray(geom.points, dtype=float)
            if len(points) == 0:
                continue
            parts = list(geom.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                seg_xy = points[start:end]
                if len(seg_xy) < 2:
                    continue
                lon, lat = transformer.transform(seg_xy[:, 0], seg_xy[:, 1])
                segments.append(np.column_stack([lon, lat]))
    return segments


def overlay_railway(ax, railway_segments: list[np.ndarray]) -> None:
    for seg in railway_segments:
        ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=2.8, zorder=4)
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=1.5, linestyle=(0, (5, 3)), zorder=5)


def add_railway_legend(ax, lon: float = RAILWAY_LEGEND_LON, lat: float = RAILWAY_LEGEND_LAT) -> None:
    x0 = float(lon)
    x1 = x0 + 1.45
    y0 = float(lat)
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.8, zorder=8)
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5, linestyle=(0, (5, 3)), zorder=9)
    ax.text(
        x1 + 0.12,
        y0,
        "Railway",
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="0.1",
        bbox=dict(boxstyle="round,pad=0.12", facecolor=(1.0, 1.0, 1.0, 0.78), edgecolor="none"),
        zorder=10,
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
        zorder=5.3,
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


def add_meteoro_sites(ax, site_df: pd.DataFrame) -> None:
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


def add_map_zoom_insets(
    ax,
    *,
    plot_arr: np.ndarray,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    cmap,
    pcolor_kwargs: dict[str, object],
) -> None:
    masked = np.ma.masked_invalid(plot_arr)
    for spec in MAP_ZOOM_SPECS:
        match = meteoro_sites.loc[meteoro_sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site_row = match.iloc[0]
        inset_ax = ax.inset_axes(spec["bounds"])
        inset_ax.pcolormesh(
            lon_plot,
            lat_plot,
            masked,
            cmap=cmap,
            shading="auto",
            rasterized=True,
            **pcolor_kwargs,
        )
        overlay_railway(inset_ax, railway_segments)
        inset_ax.set_xlim(float(site_row["longitude"]) - float(spec["xpad"]), float(site_row["longitude"]) + float(spec["xpad"]))
        inset_ax.set_ylim(float(site_row["latitude"]) - float(spec["ypad"]), float(site_row["latitude"]) + float(spec["ypad"]))
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.grid(False)
        inset_ax.set_facecolor("white")
        for spine in inset_ax.spines.values():
            spine.set_visible(False)
        add_meteoro_site_marker(inset_ax, site_row, show_label=True, marker_size=METEORO_INSET_SITE_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize d_u and d_u-gradient context plus ML-feature panels")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-total", type=int, default=DEFAULT_SAMPLE_TOTAL)
    parser.add_argument("--target-max-pixels", type=int, default=fig4.TARGET_MAX_PIXELS)
    parser.add_argument("--railway-shp", type=Path, default=None)
    parser.add_argument("--meteoro-shp", type=Path, default=None)
    parser.add_argument("--crs-info", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (
        base_dir / "outputs" / "deformation_rate_gradient_lake_paper"
    )
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    pixel_csv = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    grad_raster_dir = base_dir / "outputs" / "grad_rasters"
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
    meta_path = grad_raster_dir / "grid_meta.npz"
    du_path = grad_raster_dir / "du_f32.memmap"
    gmag_path = grad_raster_dir / "gradmag_f32.memmap"

    required = [pixel_csv, meta_path, du_path, gmag_path, railway_shp, meteoro_shp]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(str(p) for p in missing))

    meta = np.load(meta_path)
    res = float(meta["res"])
    nrows = int(meta["nrows"])
    ncols = int(meta["ncols"])
    gx0 = int(meta["gx0"])
    gy1 = int(meta["gy1"])
    min_e = float(meta["min_e"])
    max_n = float(meta["max_n"])
    map_source_crs = resolve_source_crs(crs_info_path, railway_shp)

    stride = fig4.choose_stride(nrows, ncols, target_max=args.target_max_pixels)
    du_mm = fig4.open_memmap(du_path, dtype="float32", mode="r", shape=(nrows, ncols))
    gmag_mm = fig4.open_memmap(gmag_path, dtype="float32", mode="r", shape=(nrows, ncols))

    du_ds = np.array(du_mm[::stride, ::stride], copy=False)
    gmag_km_ds = np.array(gmag_mm[::stride, ::stride], copy=False) * 1000.0
    du_plot = np.flipud(du_ds)
    gmag_km_plot = np.flipud(gmag_km_ds)
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

    lat_du_profile = summarize_spatial_band_profile_with_coords(
        du_plot,
        lat_plot,
        band_axis=0,
        max_bins=60,
    )
    lat_grad_profile = summarize_spatial_band_profile_with_coords(
        gmag_km_plot,
        lat_plot,
        band_axis=0,
        max_bins=60,
    )
    lon_du_profile = summarize_spatial_band_profile_with_coords(
        du_plot,
        lon_plot,
        band_axis=1,
        max_bins=60,
    )
    lon_grad_profile = summarize_spatial_band_profile_with_coords(
        gmag_km_plot,
        lon_plot,
        band_axis=1,
        max_bins=60,
    )

    sample_df, sample_source = resolve_ml_sample(
        csv_path=pixel_csv,
        cache_dir=cache_dir,
        sample_total=args.sample_total,
        chunksize=args.chunksize,
    )
    railway_segments = load_railway_segments(railway_shp)
    meteoro_sites = railinspect.load_meteoro_sites(meteoro_shp)

    result_du = fit_target_model(sample_df, target="d_u", model_name="ALL-Basins")
    result_grad = fit_target_model(sample_df, target="grad_mag_km", model_name="ALL-Basins")

    fig = plt.figure(figsize=DEFAULT_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.0, 1.15],
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.075,
        hspace=0.24,
    )
    gs_top = gs[0, 0].subgridspec(1, 2, wspace=0.12)
    gs_bottom = gs[1, 0].subgridspec(1, 3, width_ratios=BOTTOM_ROW_WIDTH_RATIOS, wspace=0.26)

    axA_top, axA_map, axA_side = fig4.make_composite_triptych(
        fig,
        gs_top[0, 0],
        map_width=4.8,
        side_width=2.05,
    )
    axB_top, axB_map, axB_side = fig4.make_composite_triptych(
        fig,
        gs_top[0, 1],
        map_width=4.8,
        side_width=2.05,
    )

    vminA, vmaxA = fig4.centered_clip(du_ds, center=0.0, p_lo=2, p_hi=98)
    if vminA >= 0.0:
        vminA = -1e-6
    if vmaxA <= 0.0:
        vmaxA = 1e-6
    normA = TwoSlopeNorm(vmin=vminA, vcenter=0.0, vmax=vmaxA)

    imA = axA_map.pcolormesh(
        lon_plot,
        lat_plot,
        np.ma.masked_invalid(du_plot),
        cmap="coolwarm",
        norm=normA,
        shading="auto",
        rasterized=True,
    )
    fig4.format_lonlat_axes(
        axA_map,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=False,
        fill_plot_region=True,
    )
    axA_map.set_xlabel("Longitude")
    axA_map.set_xlim(extent_lonlat[0], extent_lonlat[1])
    apply_requested_map_lat_axis(axA_map)
    overlay_railway(axA_map, railway_segments)
    add_meteoro_sites(axA_map, meteoro_sites)
    scale_bar_lon, scale_bar_lat = choose_scale_bar_anchor(extent_lonlat)
    add_scale_bar_lonlat(axA_map, lon0=scale_bar_lon, lat0=scale_bar_lat, length_km=100.0)
    add_railway_legend(axA_map)
    add_horizontal_inset_colorbar(axA_map, imA, label="mm/yr")
    add_map_zoom_insets(
        axA_map,
        plot_arr=du_plot,
        lon_plot=lon_plot,
        lat_plot=lat_plot,
        railway_segments=railway_segments,
        meteoro_sites=meteoro_sites,
        cmap="coolwarm",
        pcolor_kwargs={"norm": normA},
    )

    fig4.plot_top_profile(
        axA_top,
        lon_du_profile,
        color=fig4.DU_PROFILE_COLOR,
        ylabel="Mean $d_u$\n(mm/yr)",
        title=r"Vertical deformation rate $d_u$",
        coord_limits=(extent_lonlat[0], extent_lonlat[1]),
    )
    fig4.use_right_y_axis(axA_top, color=fig4.DU_PROFILE_COLOR)
    center_multiline_right_ylabel(axA_top)
    axA_top.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)

    fig4.plot_side_profile(
        axA_side,
        lat_du_profile,
        color=fig4.DU_PROFILE_COLOR,
        xlabel="Mean $d_u$\n(mm/yr)",
        coord_limits=(extent_lonlat[2], extent_lonlat[3]),
    )
    center_multiline_xlabel(axA_side)
    axA_side.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)

    fig4.use_map_spines_for_marginals(axA_top, axA_map, axA_side)
    add_plain_panel_label(axA_top, TOP_ROW_LABELS[0])

    vminB, vmaxB = fig4.robust_clip(gmag_km_ds, p_lo=2, p_hi=98)
    imB = axB_map.pcolormesh(
        lon_plot,
        lat_plot,
        np.ma.masked_invalid(gmag_km_plot),
        cmap="Reds",
        vmin=vminB,
        vmax=vmaxB,
        shading="auto",
        rasterized=True,
    )
    fig4.format_lonlat_axes(
        axB_map,
        show_ylabel=True,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=False,
        fill_plot_region=True,
    )
    axB_map.set_xlabel("Longitude")
    axB_map.set_xlim(extent_lonlat[0], extent_lonlat[1])
    apply_requested_map_lat_axis(axB_map)
    overlay_railway(axB_map, railway_segments)
    add_meteoro_sites(axB_map, meteoro_sites)
    add_scale_bar_lonlat(axB_map, lon0=scale_bar_lon, lat0=scale_bar_lat, length_km=100.0)
    add_railway_legend(axB_map)
    add_horizontal_inset_colorbar(axB_map, imB, label="mm/yr/km")
    add_map_zoom_insets(
        axB_map,
        plot_arr=gmag_km_plot,
        lon_plot=lon_plot,
        lat_plot=lat_plot,
        railway_segments=railway_segments,
        meteoro_sites=meteoro_sites,
        cmap="Reds",
        pcolor_kwargs={"vmin": vminB, "vmax": vmaxB},
    )

    fig4.plot_top_profile(
        axB_top,
        lon_grad_profile,
        color=GRADIENT_THEME_COLOR,
        ylabel=r"Mean $|\nabla d_u|$" + "\n(mm/yr/km)",
        title=r"Spatial gradient magnitude $|\nabla d_u|$",
        coord_limits=(extent_lonlat[0], extent_lonlat[1]),
    )
    fig4.use_right_y_axis(axB_top, color=GRADIENT_THEME_COLOR)
    center_multiline_right_ylabel(axB_top)
    axB_top.set_ylim(-2.0, 15.0)
    axB_top.set_yticks([0.0, 10.0])

    fig4.plot_side_profile(
        axB_side,
        lat_grad_profile,
        color=GRADIENT_THEME_COLOR,
        xlabel=r"Mean $|\nabla d_u|$" + "\n(mm/yr/km)",
        coord_limits=(extent_lonlat[2], extent_lonlat[3]),
    )
    center_multiline_xlabel(axB_side)

    fig4.use_map_spines_for_marginals(axB_top, axB_map, axB_side)
    axB_map.tick_params(
        axis="y",
        which="both",
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    add_plain_panel_label(axB_top, TOP_ROW_LABELS[1])

    axC_sc = fig.add_subplot(gs_bottom[0, 0])
    plot_obs_pred_target(
        axC_sc,
        result_du,
        target_label=TARGET_LABELS[result_du["target"]],
        units=TARGET_UNITS[result_du["target"]],
        title=r"ALL-Basins $d_u$" + "\nPredicted vs observed",
        scatter_limits=fig5.SCATTER_LIMITS,
        base_color=fig4.DU_PROFILE_COLOR,
        equal_aspect=True,
    )
    apply_scatter_axis_label_ticks(axC_sc, DU_SCATTER_LABEL_TICKS)
    add_plain_panel_label(axC_sc, BOTTOM_ROW_LABELS[0])

    axD_imp = fig.add_subplot(gs_bottom[0, 1])
    plot_importance_panel(
        axD_imp,
        result=result_du,
        panel_label=BOTTOM_ROW_LABELS[1],
        base_color=fig4.DU_PROFILE_COLOR,
    )

    axE_imp = fig.add_subplot(gs_bottom[0, 2])
    plot_importance_panel(
        axE_imp,
        result=result_grad,
        panel_label=BOTTOM_ROW_LABELS[2],
        base_color=GRADIENT_THEME_COLOR,
    )

    for ax in [axA_top, axA_map, axA_side, axB_top, axB_map, axB_side, axC_sc, axD_imp, axE_imp]:
        fig4.apply_bold_nonlegend(ax)

    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    du_importance_path = save_importance_table(result_du, table_dir)
    grad_importance_path = save_importance_table(result_grad, table_dir)
    meta_out = cache_dir / f"{FIG_BASENAME}_meta.json"
    meta_out.write_text(
        json.dumps(
            {
                "sample_source": str(sample_source),
                "sample_n": int(len(sample_df)),
                "grid": {
                    "res_m": res,
                    "nrows": nrows,
                    "ncols": ncols,
                    "stride": stride,
                },
                "crs_info_path": str(crs_info_path),
                "meteoro_shp": str(meteoro_shp),
                "map_source_crs": map_source_crs.to_wkt(),
                "targets": {
                    "d_u": {
                        "r2": float(result_du["r2"]),
                        "mae": float(result_du["mae"]),
                        "importance_table": str(du_importance_path),
                    },
                    "grad_mag_km": {
                        "r2": float(result_grad["r2"]),
                        "mae": float(result_grad["mae"]),
                        "importance_table": str(grad_importance_path),
                    },
                },
                "figure_png": str(out_png),
                "figure_pdf": str(out_pdf),
            },
            indent=2,
        )
    )

    print(f"Saved figure PNG: {out_png}")
    print(f"Saved figure PDF: {out_pdf}")
    print(f"Saved meta JSON: {meta_out}")


if __name__ == "__main__":
    main()
