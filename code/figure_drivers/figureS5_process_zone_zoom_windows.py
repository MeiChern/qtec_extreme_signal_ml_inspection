#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/FigureS5_process_zone_zoom_windows.py
# Renamed package path: code/figure_drivers/figureS5_process_zone_zoom_windows.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from submission_build_common import (
    FONT,
    ROOT_DIR,
    TABLE_DIR,
    crop_raster,
    ensure_style,
    load_du_raster,
    load_env_raster,
    load_geo_layers,
    load_grad_raster,
    project_crs,
    robust_limits,
    save_figure,
    symmetric_limits,
    zou_boundary_projected,
)

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds as window_from_bounds


FIG_STEM = "FigureS5_process_zone_zoom_windows"
WINDOW_KM = 18.0
CELL_SIZE_KM = 12.0
MIN_CENTER_SPACING_KM = 35.0
WAYBACK_CRS = "EPSG:3857"
WAYBACK_DIR = Path("/mnt/d/OneDriveBackUps/OneDrive/Desktop/Wayback_Imagery")
WAYBACK_MIN_COVER_FRAC = 0.75
WAYBACK_TARGET_WIDTH = 760
GRAD_DISPLAY_SCALE = 1000.0
RAIL_DASHES = (8, 5)
# Match Figure01_study_area_overview.py feature colors.
TL_COLOR = "#4E7A84"
RTS_COLOR = "#D84C6F"
BOUNDARY_COLOR = "0.20"
RAIL_COLOR = "0.08"
SCALEBAR_LENGTH_BY_WINDOW_KEY = {
    "transition": 4.0,
    "near_tl": 2.0,
    "near_rts": 1.0,
    "interior": 1.0,
}
HEADER_FONTSIZE = FONT["axis"] + 2.0
ROW_FONTSIZE = FONT["axis"] + 1.6
ANNOTATION_FONTSIZE = FONT["axis"] + 0.4
COLORBAR_FONTSIZE = FONT["axis"] + 0.1
LEGEND_FONTSIZE = FONT["axis"] + 0.8


@dataclass(frozen=True)
class WindowSpec:
    key: str
    label: str


@dataclass(frozen=True)
class WaybackTile:
    path: Path
    bounds: tuple[float, float, float, float]
    area_m2: float


@dataclass(frozen=True)
class ManualWindow:
    key: str
    label: str
    center_easting_m: float
    center_northing_m: float
    window_size_km: float
    wayback_tile: str
    source_bounds_lonlat: tuple[float, float, float, float] | None = None


WINDOW_SPECS = [
    WindowSpec("transition", "PF-NPF transition"),
    WindowSpec("near_tl", "Thermokarst-lake\nproximal"),
    WindowSpec("near_rts", "Thaw slumps"),
    WindowSpec("interior", "Interior non-prox."),
]
WINDOW_SPEC_BY_KEY = {spec.key: spec for spec in WINDOW_SPECS}

MANUAL_WINDOWS = [
    ManualWindow(
        key="transition",
        label="PF-NPF transition",
        center_easting_m=-1657660.2294761506,
        center_northing_m=3582465.4954141835,
        window_size_km=31.9,
        wayback_tile="20221102_TsonagLake_Northeast_Suspicious_Region.tpkx",
        source_bounds_lonlat=(91.8755700, 32.1899722, 92.1697587, 32.4398678),
    ),
    ManualWindow(
        key="near_tl",
        label="Thermokarst-lake\nproximal",
        center_easting_m=-1510884.5362193268,
        center_northing_m=3866920.051055893,
        window_size_km=12.0,
        wayback_tile="20221102_Fenghuoshan_North_Suspicious_Region.tpkx",
        source_bounds_lonlat=(93.0071149, 34.9637935, 93.1163194, 35.0865216),
    ),
    ManualWindow(
        key="near_rts",
        label="Thaw slumps",
        center_easting_m=-1541486.0,
        center_northing_m=3878194.0,
        window_size_km=7.0,
        wayback_tile="Wayback_Wudaoliang_West_RTS_20221102.tpkx",
    ),
    ManualWindow(
        key="interior",
        label="Interior non-prox.",
        center_easting_m=-1538836.6792455767,
        center_northing_m=3836927.441429951,
        window_size_km=5.0,
        wayback_tile="20221102_Fenghuoshan_North_Suspicious_Region.tpkx",
        source_bounds_lonlat=(92.7895518, 34.6955887, 92.8347173, 34.7468273),
    ),
]


def load_zone_sample() -> pd.DataFrame:
    path = ROOT_DIR / "outputs" / "deformation_rate_gradient_lake_paper" / "tables" / "_check_abrupt_thaw_degradation_hazard_v2_sample_with_distance.csv.gz"
    return pd.read_csv(path)


def pf_hotspot_mask(df: pd.DataFrame) -> np.ndarray:
    du = pd.to_numeric(df["d_u"], errors="coerce").to_numpy(dtype=float)
    grad = pd.to_numeric(df["grad_mag_km"], errors="coerce").to_numpy(dtype=float)
    return (du <= -16.5) | (grad >= 30.5)


def candidate_subset(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "transition":
        return df.loc[df["analysis_zone"].eq("pf_frontal_iso")].copy()
    if key == "near_tl":
        tl = pd.to_numeric(df["tl_distance_km"], errors="coerce")
        rts = pd.to_numeric(df["rts_distance_km"], errors="coerce")
        return df.loc[df["analysis_zone"].isin(["pf_abrupt_iso", "pf_overlap"]) & (tl <= rts)].copy()
    if key == "near_rts":
        tl = pd.to_numeric(df["tl_distance_km"], errors="coerce")
        rts = pd.to_numeric(df["rts_distance_km"], errors="coerce")
        return df.loc[df["analysis_zone"].isin(["pf_abrupt_iso", "pf_overlap"]) & (rts < tl)].copy()
    if key == "interior":
        return df.loc[df["analysis_zone"].eq("pf_background")].copy()
    raise ValueError(key)


def score_candidates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    sub = candidate_subset(df, key)
    if sub.empty:
        return sub

    boundary = pd.to_numeric(sub["zou_boundary_distance_km"], errors="coerce").fillna(999.0)
    tl = pd.to_numeric(sub["tl_distance_km"], errors="coerce").fillna(999.0)
    rts = pd.to_numeric(sub["rts_distance_km"], errors="coerce").fillna(999.0)
    tlrts = pd.to_numeric(sub["tlrts_distance_km"], errors="coerce").fillna(999.0)
    du = (-pd.to_numeric(sub["d_u"], errors="coerce")).clip(lower=0.0).fillna(0.0)
    grad = pd.to_numeric(sub["grad_mag_km"], errors="coerce").clip(lower=0.0).fillna(0.0)
    hotspot = pf_hotspot_mask(sub).astype(float)

    score = 2.5 * hotspot + du / 20.0 + grad / 30.0
    if key == "transition":
        score += np.clip(5.0 - boundary, 0.0, 5.0) / 5.0
    elif key == "near_tl":
        score += np.clip(2.0 - tl, 0.0, 2.0) / 2.0
    elif key == "near_rts":
        score += np.clip(2.0 - rts, 0.0, 2.0) / 2.0
    elif key == "interior":
        score += boundary.clip(lower=5.0, upper=20.0) / 20.0
        score += tlrts.clip(lower=2.0, upper=10.0) / 20.0

    scale_m = CELL_SIZE_KM * 1000.0
    sub["hotspot"] = hotspot
    sub["score"] = score
    sub["cell_x"] = np.floor(sub["easting"] / scale_m).astype(int)
    sub["cell_y"] = np.floor(sub["northing"] / scale_m).astype(int)

    grouped = (
        sub.groupby(["cell_x", "cell_y"])
        .agg(
            center_easting=("easting", "median"),
            center_northing=("northing", "median"),
            n_points=("easting", "size"),
            n_hotspots=("hotspot", "sum"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
        )
        .reset_index(drop=True)
        .sort_values(["n_hotspots", "mean_score", "n_points", "max_score"], ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def far_enough(center_e: float, center_n: float, chosen: list[tuple[float, float]]) -> bool:
    if not chosen:
        return True
    dist_km = np.sqrt((np.asarray([center_e]) - np.asarray([e for e, _ in chosen])) ** 2 + (np.asarray([center_n]) - np.asarray([n for _, n in chosen])) ** 2) / 1000.0
    return bool(np.all(dist_km >= MIN_CENTER_SPACING_KM))


def project_window_bounds(center_e: float, center_n: float, window_km: float) -> tuple[float, float, float, float]:
    half_m = float(window_km) * 500.0
    return (
        float(center_e - half_m),
        float(center_n - half_m),
        float(center_e + half_m),
        float(center_n + half_m),
    )


@lru_cache(maxsize=1)
def wayback_transformer() -> Transformer:
    return Transformer.from_crs(project_crs(), WAYBACK_CRS, always_xy=True)


@lru_cache(maxsize=1)
def wayback_tiles() -> tuple[WaybackTile, ...]:
    if not WAYBACK_DIR.exists():
        return tuple()
    tiles: list[WaybackTile] = []
    for path in sorted(WAYBACK_DIR.glob("*.tpkx")):
        try:
            with rasterio.open(path) as src:
                bounds = (float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
        except Exception:
            continue
        width = max(0.0, bounds[2] - bounds[0])
        height = max(0.0, bounds[3] - bounds[1])
        tiles.append(WaybackTile(path=path, bounds=bounds, area_m2=width * height))
    return tuple(tiles)


def wayback_window_bounds(center_e: float, center_n: float, window_km: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = project_window_bounds(center_e, center_n, window_km)
    xs, ys = wayback_transformer().transform(
        [xmin, xmin, xmax, xmax],
        [ymin, ymax, ymin, ymax],
    )
    return (float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys)))


def bbox_overlap_fraction(
    window_bounds: tuple[float, float, float, float],
    tile_bounds: tuple[float, float, float, float],
) -> float:
    left = max(window_bounds[0], tile_bounds[0])
    bottom = max(window_bounds[1], tile_bounds[1])
    right = min(window_bounds[2], tile_bounds[2])
    top = min(window_bounds[3], tile_bounds[3])
    if right <= left or top <= bottom:
        return 0.0
    window_area = max(0.0, window_bounds[2] - window_bounds[0]) * max(0.0, window_bounds[3] - window_bounds[1])
    if window_area <= 0.0:
        return 0.0
    return float(((right - left) * (top - bottom)) / window_area)


def best_wayback_match(center_e: float, center_n: float, *, window_km: float) -> tuple[str, float]:
    tiles = wayback_tiles()
    if not tiles:
        return "", 0.0
    window_bounds = wayback_window_bounds(center_e, center_n, window_km)
    center_x, center_y = wayback_transformer().transform(float(center_e), float(center_n))
    best_tile = None
    best_key = None
    for tile in tiles:
        cover_frac = bbox_overlap_fraction(window_bounds, tile.bounds)
        if cover_frac <= 0.0:
            continue
        contains_center = tile.bounds[0] <= center_x <= tile.bounds[2] and tile.bounds[1] <= center_y <= tile.bounds[3]
        key = (cover_frac, int(contains_center), -tile.area_m2)
        if best_key is None or key > best_key:
            best_key = key
            best_tile = tile
    if best_tile is None or best_key is None:
        return "", 0.0
    return best_tile.path.name, float(best_key[0])


def wayback_cover_fraction(center_e: float, center_n: float, *, window_km: float, tile_name: str) -> float:
    if not tile_name:
        return 0.0
    tile = next((tile for tile in wayback_tiles() if tile.path.name == tile_name), None)
    if tile is None:
        return 0.0
    return bbox_overlap_fraction(wayback_window_bounds(center_e, center_n, window_km), tile.bounds)


def score_window_points(df: pd.DataFrame, key: str, *, center_e: float, center_n: float, window_km: float) -> tuple[int, int, float]:
    sub = candidate_subset(df, key)
    if sub.empty:
        return 0, 0, float("nan")
    half_m = float(window_km) * 500.0
    in_window = sub.loc[
        (pd.to_numeric(sub["easting"], errors="coerce") >= center_e - half_m)
        & (pd.to_numeric(sub["easting"], errors="coerce") <= center_e + half_m)
        & (pd.to_numeric(sub["northing"], errors="coerce") >= center_n - half_m)
        & (pd.to_numeric(sub["northing"], errors="coerce") <= center_n + half_m)
    ].copy()
    if in_window.empty:
        return 0, 0, float("nan")

    hotspot = pf_hotspot_mask(in_window)
    du = (-pd.to_numeric(in_window["d_u"], errors="coerce")).clip(lower=0.0).fillna(0.0)
    grad = pd.to_numeric(in_window["grad_mag_km"], errors="coerce").clip(lower=0.0).fillna(0.0)
    score = 2.5 * hotspot.astype(float) + du / 20.0 + grad / 30.0
    return int(len(in_window)), int(np.sum(hotspot)), float(np.mean(score))


def manual_window_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in MANUAL_WINDOWS:
        n_points, n_hotspots, mean_score = score_window_points(
            df,
            spec.key,
            center_e=spec.center_easting_m,
            center_n=spec.center_northing_m,
            window_km=spec.window_size_km,
        )
        rows.append(
            {
                "window_key": spec.key,
                "window_label": spec.label,
                "center_easting_m": spec.center_easting_m,
                "center_northing_m": spec.center_northing_m,
                "window_size_km": spec.window_size_km,
                "n_points": n_points,
                "n_hotspots": n_hotspots,
                "mean_score": mean_score,
                "wayback_tile": spec.wayback_tile,
                "wayback_cover_frac": wayback_cover_fraction(
                    spec.center_easting_m,
                    spec.center_northing_m,
                    window_km=spec.window_size_km,
                    tile_name=spec.wayback_tile,
                ),
                "source_bounds_lonlat": "" if spec.source_bounds_lonlat is None else ",".join(f"{v:.7f}" for v in spec.source_bounds_lonlat),
            }
        )
    return pd.DataFrame(rows)


def choose_windows(df: pd.DataFrame) -> pd.DataFrame:
    return manual_window_table(df)


def choose_windows_auto(df: pd.DataFrame) -> pd.DataFrame:
    chosen_rows: list[dict[str, object]] = []
    chosen_centers: list[tuple[float, float]] = []
    for spec in WINDOW_SPECS:
        candidates = score_candidates(df, spec.key).copy()
        if candidates.empty:
            continue
        matches = [
            best_wayback_match(
                float(row["center_easting"]),
                float(row["center_northing"]),
                window_km=WINDOW_KM,
            )
            for _, row in candidates.iterrows()
        ]
        candidates["wayback_tile"] = [name for name, _ in matches]
        candidates["wayback_cover_frac"] = [frac for _, frac in matches]
        preferred = candidates.loc[candidates["wayback_cover_frac"] >= WAYBACK_MIN_COVER_FRAC].copy()
        search_order = preferred if not preferred.empty else candidates
        pick = None
        for _, row in search_order.iterrows():
            if far_enough(float(row["center_easting"]), float(row["center_northing"]), chosen_centers):
                pick = row
                break
        if pick is None:
            for _, row in candidates.iterrows():
                if far_enough(float(row["center_easting"]), float(row["center_northing"]), chosen_centers):
                    pick = row
                    break
        if pick is None:
            pick = search_order.iloc[0]
        center_e = float(pick["center_easting"])
        center_n = float(pick["center_northing"])
        chosen_centers.append((center_e, center_n))
        chosen_rows.append(
            {
                "window_key": spec.key,
                "window_label": spec.label,
                "center_easting_m": center_e,
                "center_northing_m": center_n,
                "window_size_km": WINDOW_KM,
                "n_points": int(pick["n_points"]),
                "n_hotspots": int(round(float(pick["n_hotspots"]))),
                "mean_score": float(pick["mean_score"]),
                "wayback_tile": str(pick["wayback_tile"]),
                "wayback_cover_frac": float(pick["wayback_cover_frac"]),
            }
        )
    return pd.DataFrame(chosen_rows)


def global_limits() -> dict[str, object]:
    du = np.asarray(load_du_raster()[::18, ::18])
    grad = np.asarray(load_grad_raster()[::18, ::18], dtype=float) * GRAD_DISPLAY_SCALE
    ndvi = np.asarray(load_env_raster("ndvi")[::18, ::18])
    twi = np.asarray(load_env_raster("twi")[::18, ::18])
    magt = np.asarray(load_env_raster("magt")[::18, ::18])
    du_vmin, du_vmax = symmetric_limits(du, 98.0)
    _, grad_vmax = robust_limits(grad, 2.0, 98.0)
    grad_vmin = 0.0
    ndvi_vmin, ndvi_vmax = robust_limits(ndvi, 2.0, 98.0)
    twi_vmin, twi_vmax = robust_limits(twi, 2.0, 98.0)
    magt_vmin, magt_vmax = robust_limits(magt, 2.0, 98.0)
    magt_norm = None
    if magt_vmin < 0.0 < magt_vmax:
        magt_norm = TwoSlopeNorm(vmin=magt_vmin, vcenter=0.0, vmax=magt_vmax)
    return {
        "du_norm": TwoSlopeNorm(vmin=du_vmin, vcenter=0.0, vmax=du_vmax),
        "grad_limits": (grad_vmin, grad_vmax),
        "ndvi_limits": (ndvi_vmin, ndvi_vmax),
        "twi_limits": (twi_vmin, twi_vmax),
        "magt_limits": (magt_vmin, magt_vmax),
        "magt_norm": magt_norm,
    }


def iter_line_arrays(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield np.asarray(geometry.coords)
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_line_arrays(part)


def plot_overlays(
    ax,
    *,
    center_e: float,
    center_n: float,
    window_km: float,
    show_boundary: bool = True,
    boundary_alpha: float = 0.95,
) -> None:
    layers = load_geo_layers()
    half_m = window_km * 500.0
    xmin = center_e - half_m
    xmax = center_e + half_m
    ymin = center_n - half_m
    ymax = center_n + half_m

    railway = layers["railway_proj"].cx[xmin:xmax, ymin:ymax]
    for geom in railway.geometry:
        for coords in iter_line_arrays(geom):
            x = (coords[:, 0] - center_e) / 1000.0
            y = (coords[:, 1] - center_n) / 1000.0
            ax.plot(x, y, color="white", linewidth=2.8, alpha=1.0, solid_capstyle="round", zorder=3)
            ax.plot(x, y, color=RAIL_COLOR, linewidth=1.25, linestyle=(0, RAIL_DASHES), dash_capstyle="butt", zorder=4)

    if show_boundary:
        lon_b, lat_b = zou_boundary_projected()
        mask = (lon_b >= xmin) & (lon_b <= xmax) & (lat_b >= ymin) & (lat_b <= ymax)
        if np.any(mask):
            ax.scatter((lon_b[mask] - center_e) / 1000.0, (lat_b[mask] - center_n) / 1000.0, s=2.0, color=BOUNDARY_COLOR, alpha=boundary_alpha, zorder=5)

    lakes = layers["lakes_proj"].cx[xmin:xmax, ymin:ymax].geometry.centroid
    if len(lakes):
        ax.scatter(
            (lakes.x.to_numpy(dtype=float) - center_e) / 1000.0,
            (lakes.y.to_numpy(dtype=float) - center_n) / 1000.0,
            s=18,
            color=TL_COLOR,
            alpha=0.70,
            linewidths=0.18,
            edgecolors="white",
            zorder=6,
        )

    rts = layers["rts_proj"].cx[xmin:xmax, ymin:ymax].geometry.centroid
    if len(rts):
        ax.scatter(
            (rts.x.to_numpy(dtype=float) - center_e) / 1000.0,
            (rts.y.to_numpy(dtype=float) - center_n) / 1000.0,
            s=28,
            color=RTS_COLOR,
            marker="^",
            alpha=0.80,
            linewidths=0.20,
            edgecolors="white",
            zorder=7,
        )


def add_boundary_annotation(ax, *, center_e: float, center_n: float, window_km: float) -> None:
    half_m = float(window_km) * 500.0
    xmin = center_e - half_m
    xmax = center_e + half_m
    ymin = center_n - half_m
    ymax = center_n + half_m
    lon_b, lat_b = zou_boundary_projected()
    mask = (lon_b >= xmin) & (lon_b <= xmax) & (lat_b >= ymin) & (lat_b <= ymax)
    if not np.any(mask):
        return

    x = (lon_b[mask] - center_e) / 1000.0
    y = (lat_b[mask] - center_n) / 1000.0
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    x = x[finite]
    y = y[finite]
    mid_x = float(np.nanmedian(x))
    mid_y = float(np.nanmedian(y))
    idx = int(np.argmin((x - mid_x) ** 2 + (y - mid_y) ** 2))
    target_x = float(x[idx])
    target_y = float(y[idx])
    half = float(window_km) * 0.5
    text_x = float(np.clip(target_x + window_km * 0.18, -half + window_km * 0.08, half - window_km * 0.08))
    text_y = float(np.clip(target_y + window_km * 0.13, -half + window_km * 0.08, half - window_km * 0.08))
    ax.annotate(
        "PF-NPF\nboundary",
        xy=(target_x, target_y),
        xytext=(text_x, text_y),
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONTSIZE,
        color=BOUNDARY_COLOR,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.8,
            "color": BOUNDARY_COLOR,
            "alpha": 0.75,
            "shrinkA": 1.0,
            "shrinkB": 1.0,
        },
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        zorder=12,
    )


def scalebar_length_km(window_key: str, window_km: float) -> float:
    return float(SCALEBAR_LENGTH_BY_WINDOW_KEY.get(window_key, max(1.0, round(float(window_km) * 0.125))))


def add_local_scalebar(ax, *, window_key: str, window_km: float) -> None:
    length_km = scalebar_length_km(window_key, float(window_km))
    half = float(window_km) * 0.5
    x0 = -half + float(window_km) * 0.08
    y0 = -half + float(window_km) * 0.08
    x1 = x0 + length_km
    tick = float(window_km) * 0.018
    effects = [pe.Stroke(linewidth=4.2, foreground="0.08"), pe.Normal()]
    for xs, ys in (
        ([x0, x1], [y0, y0]),
        ([x0, x0], [y0 - tick, y0 + tick]),
        ([x1, x1], [y0 - tick, y0 + tick]),
    ):
        line = ax.plot(xs, ys, color="white", linewidth=2.0, solid_capstyle="butt", zorder=13)[0]
        line.set_path_effects(effects)
    ax.text(
        (x0 + x1) * 0.5,
        y0 + tick * 2.1,
        f"{format_endpoint(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        color="white",
        path_effects=[pe.withStroke(linewidth=2.8, foreground="0.08")],
        zorder=13,
    )


def local_ticks(window_km: float) -> list[float]:
    half = float(window_km) * 0.5
    if half >= 12.0:
        step = 10.0
    elif half >= 8.0:
        step = 5.0
    elif half >= 4.0:
        step = 2.0
    else:
        step = 1.0
    return [tick for tick in (-step, 0.0, step) if -half <= tick <= half]


def style_window_axes(ax, *, window_km: float, show_xticklabels: bool, show_yticklabels: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_edgecolor("0.25")


def plot_window(
    ax,
    arr: np.ndarray,
    *,
    center_e: float,
    center_n: float,
    window_km: float,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    value_scale: float = 1.0,
):
    crop, extent, _ = crop_raster(arr, center_e=center_e, center_n=center_n, window_km=window_km)
    if not np.isclose(float(value_scale), 1.0):
        crop = np.asarray(crop, dtype=float) * float(value_scale)
    crop = np.ma.masked_invalid(crop)
    ax.set_facecolor("0.93")
    im = ax.imshow(crop, extent=extent, origin="upper", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return im


def plot_wayback_window(
    ax,
    *,
    window_key: str,
    center_e: float,
    center_n: float,
    window_km: float,
    tile_name: str,
    show_boundary: bool,
) -> None:
    tile_path = WAYBACK_DIR / tile_name if tile_name else None
    if tile_path is None or not tile_path.exists():
        ax.set_facecolor("0.93")
        half = float(window_km) * 0.5
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.text(0.5, 0.5, "No Wayback\ncoverage", transform=ax.transAxes, ha="center", va="center", fontsize=FONT["annotation"], color="0.35")
        return

    left, bottom, right, top = wayback_window_bounds(center_e, center_n, window_km)
    with rasterio.open(tile_path) as src:
        read_window = window_from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        height = max(2, int(round(WAYBACK_TARGET_WIDTH * (top - bottom) / max(right - left, 1.0))))
        image = src.read(
            indexes=[1, 2, 3],
            window=read_window,
            out_shape=(3, height, WAYBACK_TARGET_WIDTH),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=255,
        )

    half = float(window_km) * 0.5
    extent = [-half, half, -half, half]
    ax.imshow(np.moveaxis(image, 0, -1), extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    plot_overlays(ax, center_e=center_e, center_n=center_n, window_km=window_km, show_boundary=show_boundary, boundary_alpha=0.5)
    if show_boundary:
        add_boundary_annotation(ax, center_e=center_e, center_n=center_n, window_km=window_km)
    add_local_scalebar(ax, window_key=window_key, window_km=window_km)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def format_endpoint(value: float) -> str:
    value = float(value)
    if np.isclose(value, round(value), atol=0.05):
        return str(int(round(value)))
    abs_value = abs(value)
    if abs_value >= 10.0:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def add_row_annotation(ax, label: str) -> None:
    fontsize = ROW_FONTSIZE - 1.2 if len(label) > 9 else ROW_FONTSIZE
    ax.text(
        -0.10,
        0.50,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold",
        color="0.08",
        clip_on=False,
    )


def add_column_header(ax, title: str) -> None:
    ax.text(
        0.5,
        1.05,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=HEADER_FONTSIZE,
        fontweight="bold",
        color="0.08",
        clip_on=False,
    )


def add_bottom_center_colorbar(fig: plt.Figure, ax, mappable, *, label: str, tick_values: tuple[float, float]) -> None:
    bbox = ax.get_position()
    cb_width = bbox.width * 0.50
    cb_height = 0.012
    cb_x0 = bbox.x0 + (bbox.width - cb_width) * 0.5
    cb_y0 = bbox.y0 - 0.045
    cax = fig.add_axes([cb_x0, cb_y0, cb_width, cb_height])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_ticks([float(tick_values[0]), float(tick_values[1])])
    cb.ax.set_xticklabels([format_endpoint(tick_values[0]), format_endpoint(tick_values[1])], fontsize=COLORBAR_FONTSIZE)
    cb.set_label(label, fontsize=COLORBAR_FONTSIZE, labelpad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2, width=0.6, pad=1.5)


def add_wayback_timestamp(fig: plt.Figure, ax, text: str) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + bbox.width * 0.5,
        bbox.y0 - 0.039,
        text,
        ha="center",
        va="center",
        fontsize=COLORBAR_FONTSIZE,
        color="0.08",
    )


def main() -> None:
    ensure_style()
    zone_df = load_zone_sample()
    windows = choose_windows(zone_df)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_csv(TABLE_DIR / f"{FIG_STEM}_windows.csv", index=False)

    limits = global_limits()
    du = load_du_raster()
    grad = load_grad_raster()
    ndvi = load_env_raster("ndvi")
    twi = load_env_raster("twi")
    magt = load_env_raster("magt")
    du_cmap = plt.get_cmap("RdBu_r").copy()
    du_cmap.set_bad("0.88")
    grad_cmap = plt.get_cmap("Reds").copy()
    grad_cmap.set_bad("0.92")
    ndvi_cmap = plt.get_cmap("Greens").copy()
    ndvi_cmap.set_bad("0.92")
    twi_cmap = plt.get_cmap("BrBG").copy()
    twi_cmap.set_bad("0.92")
    magt_cmap = plt.get_cmap("RdYlBu_r").copy()
    magt_cmap.set_bad("0.92")

    column_specs = [
        {
            "key": "wayback",
            "title": "Wayback Imagery",
            "kind": "wayback",
        },
        {
            "key": "du",
            "title": r"$\mathbf{d}_{\mathbf{u}}$",
            "kind": "raster",
            "arr": du,
            "cmap": du_cmap,
            "norm": limits["du_norm"],
            "vmin": None,
            "vmax": None,
            "scale": 1.0,
            "colorbar_label": r"$d_u$ (mm yr$^{-1}$)",
            "tick_values": (float(limits["du_norm"].vmin), float(limits["du_norm"].vmax)),
        },
        {
            "key": "grad",
            "title": r"$|\nabla \mathbf{d}_{\mathbf{u}}|$",
            "kind": "raster",
            "arr": grad,
            "cmap": grad_cmap,
            "norm": None,
            "vmin": limits["grad_limits"][0],
            "vmax": limits["grad_limits"][1],
            "scale": GRAD_DISPLAY_SCALE,
            "colorbar_label": r"$|\nabla d_u|$ (mm yr$^{-1}$ km$^{-1}$)",
            "tick_values": limits["grad_limits"],
        },
        {
            "key": "ndvi",
            "title": "NDVI",
            "kind": "raster",
            "arr": ndvi,
            "cmap": ndvi_cmap,
            "norm": None,
            "vmin": limits["ndvi_limits"][0],
            "vmax": limits["ndvi_limits"][1],
            "scale": 1.0,
            "colorbar_label": "NDVI",
            "tick_values": limits["ndvi_limits"],
        },
        {
            "key": "twi",
            "title": "TWI",
            "kind": "raster",
            "arr": twi,
            "cmap": twi_cmap,
            "norm": None,
            "vmin": limits["twi_limits"][0],
            "vmax": limits["twi_limits"][1],
            "scale": 1.0,
            "colorbar_label": "TWI",
            "tick_values": limits["twi_limits"],
        },
        {
            "key": "magt",
            "title": "MAGT",
            "kind": "raster",
            "arr": magt,
            "cmap": magt_cmap,
            "norm": limits["magt_norm"],
            "vmin": None if limits["magt_norm"] is not None else limits["magt_limits"][0],
            "vmax": None if limits["magt_norm"] is not None else limits["magt_limits"][1],
            "scale": 1.0,
            "colorbar_label": "MAGT (°C)",
            "tick_values": limits["magt_limits"],
        },
    ]

    fig, axes = plt.subplots(len(windows), len(column_specs), figsize=(13.8, 9.8), constrained_layout=False)
    plt.subplots_adjust(left=0.11, right=0.995, top=0.855, bottom=0.17, wspace=0.025, hspace=0.045)
    axes = np.atleast_2d(axes)

    column_images: dict[str, object] = {}

    for row_idx, row in windows.reset_index(drop=True).iterrows():
        center_e = float(row["center_easting_m"])
        center_n = float(row["center_northing_m"])
        window_km = float(row["window_size_km"])
        for col_idx, col_spec in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            if col_spec["kind"] == "raster":
                column_images[col_spec["key"]] = plot_window(
                    ax,
                    col_spec["arr"],
                    center_e=center_e,
                    center_n=center_n,
                    window_km=window_km,
                    cmap=col_spec["cmap"],
                    norm=col_spec["norm"],
                    vmin=col_spec["vmin"],
                    vmax=col_spec["vmax"],
                    value_scale=float(col_spec.get("scale", 1.0)),
                )
            else:
                plot_wayback_window(
                    ax,
                    window_key=str(row["window_key"]),
                    center_e=center_e,
                    center_n=center_n,
                    window_km=window_km,
                    tile_name=str(row.get("wayback_tile", "")),
                    show_boundary=str(row["window_key"]) == "transition",
                )

            style_window_axes(
                ax,
                window_km=window_km,
                show_xticklabels=row_idx == len(windows) - 1,
                show_yticklabels=col_idx == 0,
            )
            if row_idx == 0:
                add_column_header(ax, str(col_spec["title"]))
            if col_idx == 0:
                add_row_annotation(ax, str(row["window_label"]))

    legend_handles = [
        Line2D([0], [0], color=RAIL_COLOR, linewidth=1.4, linestyle=(0, RAIL_DASHES), label="Qinghai-Tibet railway"),
        Line2D([0], [0], color=BOUNDARY_COLOR, linewidth=1.0, label="PF-NPF boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TL_COLOR, markeredgecolor="white", markersize=6, alpha=0.75, label="Thermokarst lakes"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=RTS_COLOR, markeredgecolor="white", markersize=6, alpha=0.80, label="Thaw slumps"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945), fontsize=LEGEND_FONTSIZE)

    fig.canvas.draw()
    add_wayback_timestamp(fig, axes[-1, 0], "(20221102)")
    for col_idx, col_spec in enumerate(column_specs):
        if col_spec.get("kind") != "raster":
            continue
        add_bottom_center_colorbar(
            fig,
            axes[-1, col_idx],
            column_images[col_spec["key"]],
            label=str(col_spec["colorbar_label"]),
            tick_values=tuple(float(v) for v in col_spec["tick_values"]),
        )

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
