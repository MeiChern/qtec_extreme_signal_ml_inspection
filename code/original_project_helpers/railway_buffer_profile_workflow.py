#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure_reorganize_railway_buffer_analysis.py
# Renamed package path: code/original_project_helpers/railway_buffer_profile_workflow.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import linemerge

from submission_figure_style import add_panel_label as add_submission_panel_label
from submission_figure_style import apply_style as apply_submission_style

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("geopandas is required for the railway buffer analysis.") from exc


SEED = 42
CHUNKSIZE = 250_000
FIG_BASENAME = "Figure_reorganize_railway_buffer_analysis"
BUFFER_WIDTHS_KM_DEFAULT = (1.0, 2.0, 3.0)
PROFILE_BIN_KM_DEFAULT = 10.0
PROFILE_STD_SCALE = 0.1
BAR_INTERVAL_KM = (400.0, 1000.0)

PERMAFROST_COLOR = "#4C7EA3"
NON_PERMAFROST_COLOR = "#9A6A49"
DU_BASE_COLOR = "#1E5BAA"
GRAD_BASE_COLOR = "#C1272D"
OTHER_COLOR = "#D9D9D9"
PROFILE_WIDTH_ORDER_DESC = (3.0, 2.0, 1.0)
PROFILE_ROW_SIDES = ("left", "right", "left")
PANEL_LABELS = ("A", "B")
EXTREME_REFERENCE_LABEL = "Within Regional Reference Range"
DOMAIN_LABELS = ("Permafrost", "Non-Permafrost")
DOMAIN_DISPLAY = {
    "Permafrost": "Permafrost",
    "Non-Permafrost": "Non-permafrost",
}
PERMAFROST_DONUT_BG = "#EAF3F8"
NON_PERMAFROST_DONUT_BG = "#EFE6DD"

PERMAFROST_EXTREME_DU_THRESHOLD = -16.5
NON_PERMAFROST_EXTREME_DU_THRESHOLD = -7.4
PERMAFROST_EXTREME_GRAD_THRESHOLD = 30.5
NON_PERMAFROST_EXTREME_GRAD_THRESHOLD = 17.0


def configure_style() -> None:
    apply_submission_style()


def log_step(message: str) -> None:
    print(f"[{FIG_BASENAME}] {message}")


def style_open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def add_panel_label(ax, label: str) -> None:
    add_submission_panel_label(ax, label, x=-0.12, y=1.04)


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - blend) * rgb + blend)


def format_width_label(width: float) -> str:
    rounded = round(float(width))
    if np.isclose(float(width), rounded):
        return str(int(rounded))
    return f"{float(width):.1f}".rstrip("0").rstrip(".")


def make_buffer_color_map(base_color: str, width_order_desc: tuple[float, ...]) -> dict[float, tuple[float, float, float]]:
    blends = np.linspace(0.68, 0.22, len(width_order_desc))
    return {
        float(width): blend_with_white(base_color, float(blend))
        for width, blend in zip(width_order_desc, blends)
    }


def make_ring_color_map(
    base_color: str,
    width_order_desc: tuple[float, ...],
    *,
    extra_blend: float = 0.14,
) -> dict[float, tuple[float, float, float]]:
    profile_colors = make_buffer_color_map(base_color, width_order_desc)
    return {
        float(width): blend_with_white(profile_colors[float(width)], extra_blend)
        for width in width_order_desc
    }


def make_background_cmap(base_color: str, *, deep_blend: float, light_blend: float) -> LinearSegmentedColormap:
    deep = blend_with_white(base_color, deep_blend)
    light = blend_with_white(base_color, light_blend)
    return LinearSegmentedColormap.from_list("panel_bg", [deep, light])


def apply_bold_ticklabels(ax) -> None:
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")


def apply_bold_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight("bold")


def apply_bar_tick_labels(ax, labels: list[str] | None) -> None:
    if labels is None or len(labels) == 0:
        return
    ticks = np.asarray(ax.get_yticks(), dtype=float)
    if ticks.size != len(labels):
        ymin, ymax = ax.get_ylim()
        ticks = np.linspace(float(ymin), float(ymax), len(labels))
        ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    apply_bold_ticklabels(ax)


def add_fig_ring_box(
    fig,
    *,
    x: float,
    y: float,
    width_order_desc: tuple[float, ...],
) -> None:
    fig.text(
        x,
        y,
        (
            f"outer ring: {width_order_desc[0]:.0f} km\n"
            f"middle ring: {width_order_desc[1]:.0f} km\n"
            f"inner ring: {width_order_desc[2]:.0f} km"
        ),
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
        fontweight="bold",
        color="0.28",
        bbox=dict(boxstyle="round,pad=0.26", facecolor=(1.0, 1.0, 1.0, 0.90), edgecolor="0.75", linewidth=0.8),
    )


def flatten_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[LineString] = []
        for sub in geom.geoms:
            out.extend(flatten_lines(sub))
        return out
    return []


def longest_line(geom) -> LineString:
    if geom is None or geom.is_empty:
        raise RuntimeError("Railway geometry is empty.")
    merged = linemerge(geom)
    lines = flatten_lines(merged)
    if not lines:
        lines = flatten_lines(geom)
    if not lines:
        raise RuntimeError("Could not derive a line geometry from the railway shapefile.")
    return max(lines, key=lambda g: float(g.length))


def load_railway_context(railway_shp: Path) -> dict[str, object]:
    railway = gpd.read_file(railway_shp)
    railway = railway.loc[railway.geometry.notna() & ~railway.geometry.is_empty].copy()
    if railway.empty:
        raise RuntimeError(f"No valid geometries found in railway shapefile: {railway_shp}")
    if railway.crs is None:
        raise RuntimeError(f"Railway shapefile is missing CRS information: {railway_shp}")
    if not railway.crs.is_projected:
        raise RuntimeError("Railway CRS must be projected in meters for buffer analysis.")

    network_geom = railway.geometry.unary_union
    main_line = longest_line(network_geom)
    return {
        "crs": railway.crs,
        "network_geom": network_geom,
        "main_line": main_line,
        "main_line_length_km": float(main_line.length) / 1000.0,
    }


def resolve_usecols(csv_path: Path) -> list[str]:
    available = pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist()
    required = {"easting", "northing", "d_u", "Perma_Distr_map"}
    missing = sorted(required.difference(available))
    if missing:
        raise RuntimeError(f"Missing required CSV column(s): {', '.join(missing)}")

    usecols = ["easting", "northing", "d_u", "Perma_Distr_map"]
    for col in ("longitude", "latitude"):
        if col in available:
            usecols.append(col)
    if "grad_mag_km" in available:
        usecols.append("grad_mag_km")
    if "grad_mag" in available:
        usecols.append("grad_mag")
    if "grad_mag_km" not in usecols and "grad_mag" not in usecols:
        raise RuntimeError("The pixel CSV does not contain grad_mag_km or grad_mag.")
    return usecols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["longitude", "latitude", "easting", "northing", "d_u", "grad_mag", "grad_mag_km", "Perma_Distr_map"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "grad_mag" in out.columns:
        grad_mag_km = out["grad_mag"] * 1000.0
        if "grad_mag_km" in out.columns:
            out["grad_mag_km"] = out["grad_mag_km"].where(np.isfinite(out["grad_mag_km"]), grad_mag_km)
        else:
            out["grad_mag_km"] = grad_mag_km

    out["domain"] = np.where(
        out["Perma_Distr_map"] == 1,
        "Permafrost",
        np.where(out["Perma_Distr_map"] == 0, "Non-Permafrost", "Other"),
    )
    return out


def build_multi_buffer_samples(
    csv_path: Path,
    *,
    railway_context: dict[str, object],
    buffer_widths_km: tuple[float, ...],
    chunksize: int,
) -> dict[float, pd.DataFrame]:
    log_step("building corridor samples for all requested buffer widths")
    if any((not np.isfinite(width) or width <= 0.0) for width in buffer_widths_km):
        raise RuntimeError("All buffer widths must be positive finite numbers.")

    network_geom = railway_context["network_geom"]
    main_line = railway_context["main_line"]
    crs = railway_context["crs"]
    half_widths_m = {float(width): 500.0 * float(width) for width in buffer_widths_km}
    max_half_width_m = max(half_widths_m.values())
    usecols = resolve_usecols(csv_path)
    parts = {float(width): [] for width in buffer_widths_km}

    for idx, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False), start=1):
        if idx == 1 or idx % 10 == 0:
            log_step(f"processed {idx} chunk(s)")

        chunk = engineer_features(chunk)
        mask = (
            np.isfinite(chunk["easting"])
            & np.isfinite(chunk["northing"])
            & np.isfinite(chunk["d_u"])
            & np.isfinite(chunk["grad_mag_km"])
            & (chunk["d_u"] <= 0.0)
            & chunk["domain"].isin(["Permafrost", "Non-Permafrost"])
        )
        if not mask.any():
            continue

        sub = chunk.loc[mask].copy()
        points = gpd.GeoSeries(gpd.points_from_xy(sub["easting"], sub["northing"]), crs=crs)
        dist_to_rail = points.distance(network_geom).to_numpy(dtype=float)
        keep_any = np.isfinite(dist_to_rail) & (dist_to_rail <= max_half_width_m)
        if not keep_any.any():
            continue

        sub = sub.loc[keep_any].copy()
        kept_points = points.loc[keep_any]
        sub["distance_to_rail_m"] = dist_to_rail[keep_any]
        sub["along_km"] = np.fromiter(
            (float(main_line.project(geom)) / 1000.0 for geom in kept_points),
            dtype=float,
            count=len(kept_points),
        )
        keep_cols = ["easting", "northing"]
        for col in ("longitude", "latitude"):
            if col in sub.columns:
                keep_cols.append(col)
        keep_cols.extend(
            [
                "d_u",
                "grad_mag_km",
                "Perma_Distr_map",
                "domain",
                "distance_to_rail_m",
                "along_km",
            ]
        )
        sub = sub[keep_cols].reset_index(drop=True)

        for width in buffer_widths_km:
            take = sub["distance_to_rail_m"] <= half_widths_m[float(width)]
            if take.any():
                parts[float(width)].append(sub.loc[take].reset_index(drop=True))

    out: dict[float, pd.DataFrame] = {}
    for width in buffer_widths_km:
        width_key = float(width)
        if not parts[width_key]:
            raise RuntimeError(f"No pixels fell inside the requested railway buffer width={width_key:.1f} km.")
        df = pd.concat(parts[width_key], ignore_index=True)
        out[width_key] = df.sort_values("along_km").reset_index(drop=True)
    return out


def resolve_corridor_samples(
    csv_path: Path,
    *,
    cache_paths: dict[float, Path],
    railway_context: dict[str, object],
    buffer_widths_km: tuple[float, ...],
    chunksize: int,
) -> dict[float, pd.DataFrame]:
    required = {
        "easting",
        "northing",
        "d_u",
        "grad_mag_km",
        "Perma_Distr_map",
        "domain",
        "distance_to_rail_m",
        "along_km",
    }
    out: dict[float, pd.DataFrame] = {}
    all_cached = True

    for width in buffer_widths_km:
        cache_path = cache_paths[float(width)]
        if not cache_path.exists():
            all_cached = False
            break
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if not required.issubset(cache_cols):
            all_cached = False
            break

    if all_cached:
        for width in buffer_widths_km:
            cache_path = cache_paths[float(width)]
            log_step(f"loading cached corridor sample: {cache_path}")
            out[float(width)] = engineer_features(pd.read_csv(cache_path))
        return out

    samples = build_multi_buffer_samples(
        csv_path,
        railway_context=railway_context,
        buffer_widths_km=buffer_widths_km,
        chunksize=chunksize,
    )
    for width in buffer_widths_km:
        cache_path = cache_paths[float(width)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        samples[float(width)].to_csv(cache_path, index=False, compression="gzip")
        log_step(f"saved corridor sample cache: {cache_path}")
    return samples


def summarize_metric_profile(
    sample_df: pd.DataFrame,
    *,
    value_col: str,
    bin_length_km: float,
    std_scale: float,
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
        .agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            n=(value_col, "size"),
        )
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


def build_profile_table(
    sample_df: pd.DataFrame,
    *,
    buffer_width_km: float,
    bin_length_km: float,
    std_scale: float,
) -> pd.DataFrame:
    pieces = []
    for metric in ["d_u", "grad_mag_km"]:
        prof = summarize_metric_profile(sample_df, value_col=metric, bin_length_km=bin_length_km, std_scale=std_scale)
        if prof.empty:
            continue
        prof.insert(0, "metric", metric)
        prof.insert(0, "buffer_width_km", float(buffer_width_km))
        pieces.append(prof)
    if not pieces:
        raise RuntimeError(f"Could not build railway profiles for buffer width={buffer_width_km:.1f} km.")
    return pd.concat(pieces, ignore_index=True)


def build_summary_table(sample_df: pd.DataFrame, *, buffer_width_km: float) -> pd.DataFrame:
    rows = []
    for metric in ["d_u", "grad_mag_km", "distance_to_rail_m", "along_km"]:
        vals = pd.to_numeric(sample_df.get(metric), errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        rows.append(
            {
                "buffer_width_km": float(buffer_width_km),
                "metric": metric,
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return pd.DataFrame(rows)


def collect_interval_groups_by_domain(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    metric: str,
    interval_km: tuple[float, float],
) -> dict[str, dict[float, np.ndarray]]:
    lo_km, hi_km = float(interval_km[0]), float(interval_km[1])
    out: dict[str, dict[float, np.ndarray]] = {domain: {} for domain in DOMAIN_LABELS}
    for width in width_order_desc:
        sample_df = samples_by_width[float(width)]
        window_df = sample_df.loc[sample_df["along_km"].between(lo_km, hi_km, inclusive="both")].copy()
        for domain in DOMAIN_LABELS:
            vals = pd.to_numeric(
                window_df.loc[window_df["domain"].eq(domain), metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            out[domain][float(width)] = vals
    return out


def build_donut_stats(samples_by_width: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for width, sample_df in sorted(samples_by_width.items(), reverse=True):
        permafrost_mask = sample_df["domain"].eq("Permafrost")
        non_permafrost_mask = sample_df["domain"].eq("Non-Permafrost")

        permafrost_df = sample_df.loc[permafrost_mask].copy()
        non_permafrost_df = sample_df.loc[non_permafrost_mask].copy()
        extreme_pf_du = permafrost_df["d_u"] < PERMAFROST_EXTREME_DU_THRESHOLD
        extreme_npf_du = non_permafrost_df["d_u"] < NON_PERMAFROST_EXTREME_DU_THRESHOLD
        extreme_pf_grad = permafrost_df["grad_mag_km"] > PERMAFROST_EXTREME_GRAD_THRESHOLD
        extreme_npf_grad = non_permafrost_df["grad_mag_km"] > NON_PERMAFROST_EXTREME_GRAD_THRESHOLD

        rows.extend(
            [
                {
                    "buffer_width_km": float(width),
                    "group": "sample_domain",
                    "label": "Permafrost",
                    "count": int(permafrost_mask.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "sample_domain",
                    "label": "Non-Permafrost",
                    "count": int(non_permafrost_mask.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "permafrost_extreme_du",
                    "label": "Extreme",
                    "count": int(extreme_pf_du.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "permafrost_extreme_du",
                    "label": EXTREME_REFERENCE_LABEL,
                    "count": int((~extreme_pf_du).sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "non_permafrost_extreme_du",
                    "label": "Extreme",
                    "count": int(extreme_npf_du.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "non_permafrost_extreme_du",
                    "label": EXTREME_REFERENCE_LABEL,
                    "count": int((~extreme_npf_du).sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "permafrost_extreme_grad",
                    "label": "Extreme",
                    "count": int(extreme_pf_grad.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "permafrost_extreme_grad",
                    "label": EXTREME_REFERENCE_LABEL,
                    "count": int((~extreme_pf_grad).sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "non_permafrost_extreme_grad",
                    "label": "Extreme",
                    "count": int(extreme_npf_grad.sum()),
                },
                {
                    "buffer_width_km": float(width),
                    "group": "non_permafrost_extreme_grad",
                    "label": EXTREME_REFERENCE_LABEL,
                    "count": int((~extreme_npf_grad).sum()),
                },
            ]
        )
    return pd.DataFrame(rows)


def get_stat_pair(stats_df: pd.DataFrame, *, group: str, label_order: tuple[str, str], width: float) -> list[int]:
    sub = stats_df.loc[
        (stats_df["group"] == group) & np.isclose(stats_df["buffer_width_km"], float(width)),
        ["label", "count"],
    ]
    mapping = dict(zip(sub["label"], sub["count"]))
    return [int(mapping.get(label, 0)) for label in label_order]


def _safe_ring_values(values: list[int]) -> list[float]:
    total = float(sum(values))
    if total > 0.0:
        return [float(v) for v in values]
    return [1.0 for _ in values]


def plot_multi_ring_donut(
    ax,
    *,
    ring_values_by_width: dict[float, list[int]],
    width_order_desc: tuple[float, ...],
    ring_colors_by_width: dict[float, list[str]],
    title: str | None,
    center_text: str,
    legend_labels: list[str] | None = None,
    legend_ncol: int = 1,
    center_text_y: float = 0.10,
    ring_text_by_width: dict[float, str] | None = None,
    ring_text_angle_deg: float = 235.0,
    pct_label_indices: tuple[int, ...] | None = None,
    highlight_first_slice: bool = False,
    first_slice_explode: float = 0.05,
    first_slice_edgecolor: str = "0.15",
    first_slice_linewidth: float = 1.5,
) -> list[dict[str, object]]:
    radii = [1.00, 0.78, 0.56]
    ring_width = 0.18
    ring_records: list[dict[str, object]] = []

    for radius, width in zip(radii, width_order_desc):
        raw_vals = ring_values_by_width[float(width)]
        vals = _safe_ring_values(raw_vals)
        explode = None
        if highlight_first_slice and len(vals) > 0:
            explode = [float(first_slice_explode)] + [0.0] * (len(vals) - 1)

        wedges, _, autotexts = ax.pie(
            vals,
            radius=radius,
            startangle=90,
            counterclock=False,
            colors=ring_colors_by_width[float(width)],
            explode=explode,
            wedgeprops=dict(width=ring_width, edgecolor="white", linewidth=1.0),
            autopct=(lambda pct, total=sum(raw_vals): f"{pct:.2f}%" if total > 0 else ""),
            pctdistance=(radius - 0.5 * ring_width) / radius,
            textprops=dict(color="black", fontweight="bold", fontsize=7.0),
        )

        if highlight_first_slice and len(wedges) > 0:
            wedges[0].set_edgecolor(first_slice_edgecolor)
            wedges[0].set_linewidth(first_slice_linewidth)
            wedges[0].set_zorder(4)

        for idx, text in enumerate(autotexts):
            if pct_label_indices is not None and idx not in pct_label_indices:
                text.set_text("")
            text.set_color("black")
            text.set_fontweight("bold")

        if ring_text_by_width is not None and float(width) in ring_text_by_width:
            angle_rad = np.deg2rad(ring_text_angle_deg)
            label_radius = radius - 0.5 * ring_width
            x = label_radius * np.cos(angle_rad)
            y = label_radius * np.sin(angle_rad)
            rotation = ring_text_angle_deg - 90.0
            if rotation > 90.0:
                rotation -= 180.0
            if rotation < -90.0:
                rotation += 180.0
            ax.text(
                x,
                y,
                ring_text_by_width[float(width)],
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=7.0,
                fontweight="bold",
                color="black",
                zorder=4,
            )

        ring_records.append(
            {
                "width": float(width),
                "radius": float(radius),
                "ring_width": float(ring_width),
                "raw_vals": [int(v) for v in raw_vals],
                "colors": list(ring_colors_by_width[float(width)]),
                "wedges": wedges,
            }
        )

    if center_text:
        ax.text(
            0.0,
            center_text_y,
            center_text,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
        )
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=5)
    ax.set_aspect("equal")

    if legend_labels:
        legend = ax.legend(
            [Patch(facecolor=color, edgecolor="none") for color in ring_colors_by_width[float(width_order_desc[-1])]],
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=legend_ncol,
            frameon=False,
            handlelength=1.0,
            columnspacing=1.2,
        )
        apply_bold_legend(legend)

    return ring_records


def add_extreme_zoom_callout(
    ax,
    *,
    ring_records: list[dict[str, object]],
    width_order_desc: tuple[float, ...],
    background_color: str,
    title: str = "Zoomed extreme %",
    inset_bounds: tuple[float, float, float, float] = (0.325, 0.13, 0.35, 0.26),
) -> None:
    if not ring_records:
        return

    record_by_width = {float(record["width"]): record for record in ring_records}
    pct_values: list[float] = []
    bar_colors: list[tuple[float, float, float] | str] = []
    labels: list[str] = []

    for width in width_order_desc:
        record = record_by_width[float(width)]
        raw_vals = [float(v) for v in record["raw_vals"]]
        total = float(sum(raw_vals))
        pct = 100.0 * raw_vals[0] / total if total > 0.0 else 0.0
        pct_values.append(pct)
        bar_colors.append(record["colors"][0])
        labels.append(f"{format_width_label(width)} km")

    pct_arr = np.asarray(pct_values, dtype=float)
    y = np.arange(len(width_order_desc), dtype=float)
    x_max = max(6.0, float(np.nanmax(pct_arr)) * 1.45 + 0.35)

    inset = ax.inset_axes(inset_bounds)
    inset.set_facecolor(blend_with_white(background_color, 0.10))
    inset.patch.set_alpha(0.96)

    inset.barh(
        y,
        pct_arr,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.54,
        zorder=2,
    )
    inset.set_xlim(0.0, x_max)
    inset.set_ylim(-0.55, len(width_order_desc) - 0.45)
    inset.set_yticks(y)
    inset.set_yticklabels(labels)
    inset.invert_yaxis()
    inset.set_xlabel("%", fontweight="bold", labelpad=1)
    inset.set_title(title, fontsize=7.0, fontweight="bold", pad=2)
    inset.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    inset.tick_params(axis="x", labelsize=6.5)
    inset.tick_params(axis="y", labelsize=6.5, length=0)
    for lab in inset.get_xticklabels() + inset.get_yticklabels():
        lab.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.65")

    for yi, pct in zip(y, pct_arr):
        text_x = min(x_max - 0.10, pct + 0.16)
        ha = "left"
        if pct > 0.72 * x_max:
            text_x = pct - 0.18
            ha = "right"
        inset.text(
            text_x,
            yi,
            f"{pct:.2f}%",
            va="center",
            ha=ha,
            fontsize=6.5,
            fontweight="bold",
            color="black",
            zorder=3,
        )


def style_context_donut_axis(
    ax,
    *,
    background_color: str,
    domain_label: str,
    threshold_label: str,
    column_title: str | None = None,
) -> None:
    ax.set_facecolor(background_color)
    ax.patch.set_facecolor(background_color)
    ax.patch.set_alpha(1.0)
    ax.patch.set_visible(True)
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor=background_color,
            edgecolor="none",
            zorder=-20,
        )
    )
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.18, 1.18)
    ax.text(
        0.04,
        0.50,
        domain_label,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="0.15",
    )
    ax.text(
        0.96,
        0.50,
        threshold_label,
        transform=ax.transAxes,
        rotation=270,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="0.15",
    )
    if column_title is not None:
        ax.set_title(column_title, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("0.84")


def configure_profile_axis(ax, *, side: str, show_x_labels: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    if side == "right":
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(left=False, labelleft=False, right=True, labelright=True)
    else:
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        ax.tick_params(left=True, labelleft=True, right=False, labelright=False)

    if show_x_labels:
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def plot_profile_strip(
    ax,
    *,
    profile_df: pd.DataFrame,
    color: tuple[float, float, float],
    base_color: str,
    metric_kind: str,
    side: str,
    width_km: float,
    unit_label: str,
    show_x_labels: bool,
    title: str | None = None,
    add_zero_line: bool = False,
) -> None:
    if profile_df.empty:
        style_open_axes(ax)
        return

    bg_cmap = (
        make_background_cmap(base_color, deep_blend=0.74, light_blend=0.93)
        if metric_kind == "du"
        else make_background_cmap(base_color, deep_blend=0.70, light_blend=0.93)
    )
    bg = np.linspace(0.0, 1.0, 512)[:, None]
    if metric_kind != "du":
        bg = np.flipud(bg)
    ax.imshow(
        bg,
        extent=(0.0, 1.0, 0.0, 1.0),
        transform=ax.transAxes,
        origin="lower",
        aspect="auto",
        cmap=bg_cmap,
        alpha=0.52,
        zorder=0,
        interpolation="bicubic",
    )

    ax.fill_between(profile_df["center_km"], profile_df["ylo"], profile_df["yhi"], color=color, alpha=0.18, zorder=1)
    ax.plot(profile_df["center_km"], profile_df["mean"], color=color, linewidth=2.2, zorder=2)
    if add_zero_line:
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    for xref in BAR_INTERVAL_KM:
        ax.axvline(float(xref), color="0.30", linewidth=0.95, linestyle=(0, (4, 2)), zorder=0)

    if title is not None:
        ax.set_title(title, fontweight="bold", pad=4)
    ax.set_ylabel(unit_label)
    ax.yaxis.label.set_fontweight("bold")
    if show_x_labels:
        ax.set_xlabel("Along-rail distance (km)")
        ax.xaxis.label.set_fontweight("bold")
    ax.grid(axis="both", color="0.88", linewidth=0.6)
    configure_profile_axis(ax, side=side, show_x_labels=show_x_labels)

    text_xy = (0.50, 0.50)
    text_ha = "center"
    if metric_kind == "du":
        text_xy = (0.04, 0.50) if side == "left" else (0.96, 0.50)
        text_ha = "left" if side == "left" else "right"
    elif np.isclose(width_km, 3.0):
        text_xy = (0.04, 0.50)
        text_ha = "left"
    else:
        text_xy = (0.04, 0.50) if side == "left" else (0.96, 0.50)
        text_ha = "left" if side == "left" else "right"

    ax.text(
        text_xy[0],
        text_xy[1],
        f"{width_km:.0f} km buffer",
        transform=ax.transAxes,
        ha=text_ha,
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.16", facecolor=(1.0, 1.0, 1.0, 0.76), edgecolor="none"),
    )
    apply_bold_ticklabels(ax)


def plot_interval_bar_panel(
    ax,
    *,
    samples_by_width: dict[float, pd.DataFrame],
    width_order_desc: tuple[float, ...],
    metric: str,
    colors_by_width: dict[float, tuple[float, float, float]],
    unit_label: str,
    interval_km: tuple[float, float],
    title: str | None = None,
    y_limits: tuple[float, float] | None = None,
    y_tick_labels: list[str] | None = None,
    legend_loc: str = "upper left",
) -> pd.DataFrame:
    domain_groups = collect_interval_groups_by_domain(
        samples_by_width,
        width_order_desc=width_order_desc,
        metric=metric,
        interval_km=interval_km,
    )
    rows = []
    for domain in DOMAIN_LABELS:
        for width in width_order_desc:
            vals = domain_groups[domain][float(width)]
            mean = float(np.nanmean(vals)) if len(vals) else np.nan
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            err = 0.1 * std
            rows.append(
                {
                    "domain": DOMAIN_DISPLAY[domain],
                    "buffer_width_km": float(width),
                    "n": int(len(vals)),
                    "mean": mean,
                    "std": std,
                    "err": err,
                }
            )
    stats_df = pd.DataFrame(rows)

    x = np.arange(len(width_order_desc), dtype=float)
    bar_width = 0.32
    pf_means = []
    pf_errs = []
    npf_means = []
    npf_errs = []
    for width in width_order_desc:
        width_mask = np.isclose(stats_df["buffer_width_km"], float(width))
        pf_row = stats_df.loc[width_mask & stats_df["domain"].eq(DOMAIN_DISPLAY["Permafrost"])].iloc[0]
        npf_row = stats_df.loc[width_mask & stats_df["domain"].eq(DOMAIN_DISPLAY["Non-Permafrost"])].iloc[0]
        pf_means.append(float(pf_row["mean"]))
        pf_errs.append(float(pf_row["err"]))
        npf_means.append(float(npf_row["mean"]))
        npf_errs.append(float(npf_row["err"]))

    pf_means_arr = np.asarray(pf_means, dtype=float)
    pf_errs_arr = np.asarray(pf_errs, dtype=float)
    npf_means_arr = np.asarray(npf_means, dtype=float)
    npf_errs_arr = np.asarray(npf_errs, dtype=float)

    for idx, width in enumerate(width_order_desc):
        color = colors_by_width[float(width)]
        ax.bar(
            x[idx] - 0.5 * bar_width,
            pf_means_arr[idx],
            yerr=pf_errs_arr[idx],
            width=bar_width,
            color=color,
            edgecolor="none",
            ecolor="0.20",
            capsize=2.8,
            linewidth=0.0,
            zorder=2,
        )
        ax.bar(
            x[idx] + 0.5 * bar_width,
            npf_means_arr[idx],
            yerr=npf_errs_arr[idx],
            width=bar_width,
            color=blend_with_white(color, 0.42),
            edgecolor=color,
            ecolor="0.20",
            capsize=2.8,
            linewidth=0.9,
            hatch="///",
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([format_width_label(width) for width in width_order_desc])
    ax.set_xlabel("Buffer (km)")
    ax.xaxis.label.set_fontweight("bold")
    ax.set_ylabel(unit_label)
    ax.yaxis.label.set_fontweight("bold")
    if title is not None:
        ax.set_title(title, fontweight="bold", pad=4)
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(left=False, labelleft=False, right=True, labelright=True, top=False)
    apply_bold_ticklabels(ax)
    legend = ax.legend(
        handles=[
            Patch(facecolor="0.55", edgecolor="none", label="Permafrost"),
            Patch(facecolor="white", edgecolor="0.35", hatch="///", label="Non-permafrost"),
        ],
        loc=legend_loc,
        frameon=False,
        handlelength=1.4,
    )
    apply_bold_legend(legend)

    combined_means = np.concatenate([pf_means_arr, npf_means_arr])
    combined_errs = np.concatenate([pf_errs_arr, npf_errs_arr])
    valid = np.isfinite(combined_means)
    if valid.any():
        finite_means = combined_means[valid]
        finite_errs = combined_errs[valid]
        y_min = float(np.nanmin(finite_means - finite_errs))
        y_max = float(np.nanmax(finite_means + finite_errs))
        if np.isfinite(y_min) and np.isfinite(y_max):
            pad = max(0.06 * (y_max - y_min if y_max > y_min else 1.0), 0.04)
            if y_limits is None:
                if y_min >= 0:
                    ax.set_ylim(0.0, y_max + 1.8 * pad)
                else:
                    ax.set_ylim(y_min - 0.8 * pad, y_max + 1.8 * pad)
            else:
                ax.set_ylim(*y_limits)

    apply_bar_tick_labels(ax, y_tick_labels)

    return stats_df


def build_interval_stats_table(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    interval_km: tuple[float, float],
) -> pd.DataFrame:
    rows = []
    for metric in ("d_u", "grad_mag_km"):
        domain_groups = collect_interval_groups_by_domain(
            samples_by_width,
            width_order_desc=width_order_desc,
            metric=metric,
            interval_km=interval_km,
        )
        for domain in DOMAIN_LABELS:
            for width in width_order_desc:
                vals = domain_groups[domain][float(width)]
                std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
                rows.append(
                    {
                        "metric": metric,
                        "domain": DOMAIN_DISPLAY[domain],
                        "buffer_width_km": float(width),
                        "interval_lo_km": float(interval_km[0]),
                        "interval_hi_km": float(interval_km[1]),
                        "n": int(len(vals)),
                        "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "std": std,
                        "err_0p1std": 0.1 * std,
                    }
                )
    return pd.DataFrame(rows)


def build_figure(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    bin_length_km: float,
    std_scale: float,
    fig_dir: Path,
) -> tuple[Path, Path]:
    profiles_by_width = {
        float(width): build_profile_table(
            sample_df,
            buffer_width_km=float(width),
            bin_length_km=bin_length_km,
            std_scale=std_scale,
        )
        for width, sample_df in samples_by_width.items()
    }

    du_colors = make_buffer_color_map(DU_BASE_COLOR, width_order_desc)
    grad_colors = make_buffer_color_map(GRAD_BASE_COLOR, width_order_desc)

    fig = plt.figure(figsize=(11.2, 12.2))
    gs = fig.add_gridspec(
        2,
        1,
        left=0.050,
        right=0.985,
        bottom=0.055,
        top=0.95,
        hspace=0.28,
    )
    gs_du_block = gs[0, 0].subgridspec(1, 2, width_ratios=[4.0, 1.0], wspace=0.12)
    gs_grad_block = gs[1, 0].subgridspec(1, 2, width_ratios=[4.0, 1.0], wspace=0.12)
    gs_du = gs_du_block[0, 0].subgridspec(3, 1, hspace=0.06)
    gs_grad = gs_grad_block[0, 0].subgridspec(3, 1, hspace=0.06)

    du_axes = [fig.add_subplot(gs_du[i, 0]) for i in range(3)]
    grad_axes = [fig.add_subplot(gs_grad[0, 0], sharex=du_axes[0])]
    grad_axes.extend(fig.add_subplot(gs_grad[i, 0], sharex=grad_axes[0]) for i in range(1, 3))
    ax_b_bar = fig.add_subplot(gs_du_block[0, 1])
    ax_d_bar = fig.add_subplot(gs_grad_block[0, 1])

    for row_idx, width in enumerate(width_order_desc):
        du_profile = profiles_by_width[float(width)].loc[profiles_by_width[float(width)]["metric"] == "d_u"].copy()
        plot_profile_strip(
            du_axes[row_idx],
            profile_df=du_profile,
            color=du_colors[float(width)],
            base_color=DU_BASE_COLOR,
            metric_kind="du",
            side=PROFILE_ROW_SIDES[row_idx],
            width_km=float(width),
            unit_label="mm/yr",
            show_x_labels=(row_idx == len(width_order_desc) - 1),
            title=rf"Along-rail $d_u$ profile" + "\n" + rf"($d_u \leq 0$, {bin_length_km:.0f} km bins)" if row_idx == 0 else None,
            add_zero_line=True,
        )

    plot_interval_bar_panel(
        ax_b_bar,
        samples_by_width=samples_by_width,
        width_order_desc=width_order_desc,
        metric="d_u",
        colors_by_width=du_colors,
        unit_label="mm/yr",
        interval_km=BAR_INTERVAL_KM,
        title="400-1000km Mean",
        y_tick_labels=["-7.4", "-7.0", "-6.6", "-6.2", "-5.8"],
        legend_loc="lower left",
    )

    for row_idx, width in enumerate(width_order_desc):
        grad_profile = profiles_by_width[float(width)].loc[profiles_by_width[float(width)]["metric"] == "grad_mag_km"].copy()
        plot_profile_strip(
            grad_axes[row_idx],
            profile_df=grad_profile,
            color=grad_colors[float(width)],
            base_color=GRAD_BASE_COLOR,
            metric_kind="grad",
            side=PROFILE_ROW_SIDES[row_idx],
            width_km=float(width),
            unit_label="mm/yr/km",
            show_x_labels=(row_idx == len(width_order_desc) - 1),
            title=rf"Along-rail $|\nabla d_u|$ profile" + "\n" + rf"($d_u \leq 0$, {bin_length_km:.0f} km bins)" if row_idx == 0 else None,
            add_zero_line=False,
        )

    plot_interval_bar_panel(
        ax_d_bar,
        samples_by_width=samples_by_width,
        width_order_desc=width_order_desc,
        metric="grad_mag_km",
        colors_by_width=grad_colors,
        unit_label="mm/yr/km",
        interval_km=BAR_INTERVAL_KM,
        title="400-1000km Mean",
        y_limits=(13.0, 17.0),
        y_tick_labels=["13", "14", "15", "16", "17"],
        legend_loc="upper left",
    )

    add_panel_label(du_axes[0], PANEL_LABELS[0])
    add_panel_label(grad_axes[0], PANEL_LABELS[1])

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze railway-buffer negative-d_u samples across 1/2/3 km corridors with along-rail profiles and PF/NPF interval bars."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--buffer-widths-km", type=float, nargs="+", default=list(BUFFER_WIDTHS_KM_DEFAULT))
    parser.add_argument("--profile-bin-km", type=float, default=PROFILE_BIN_KM_DEFAULT)
    parser.add_argument("--profile-std-scale", type=float, default=PROFILE_STD_SCALE)
    parser.add_argument("--railway-shp", type=Path, default=None)
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
    table_dir = out_dir / "tables"
    for path in (fig_dir, cache_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    railway_shp = (
        args.railway_shp.resolve()
        if args.railway_shp is not None
        else (base_dir / "human_features" / "qtec_railway_clip.shp")
    )
    required = [csv_path, railway_shp]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(str(p) for p in missing))

    raw_widths = tuple(float(width) for width in args.buffer_widths_km)
    unique_widths = sorted({width for width in raw_widths if np.isfinite(width) and width > 0.0})
    if len(unique_widths) != 3:
        raise RuntimeError("buffer_widths_km must contain exactly three unique positive widths.")
    width_order_desc = tuple(sorted(unique_widths, reverse=True))
    if width_order_desc != PROFILE_WIDTH_ORDER_DESC:
        log_step(f"using custom buffer widths: {tuple(sorted(unique_widths))}")

    profile_bin_km = float(args.profile_bin_km)
    profile_std_scale = float(args.profile_std_scale)
    if not np.isfinite(profile_bin_km) or profile_bin_km <= 0.0:
        raise RuntimeError("profile_bin_km must be a positive finite number.")
    if not np.isfinite(profile_std_scale) or profile_std_scale < 0.0:
        raise RuntimeError("profile_std_scale must be a finite non-negative number.")

    railway_context = load_railway_context(railway_shp)
    cache_paths = {
        float(width): cache_dir / f"{FIG_BASENAME}_corridor_sample_{str(width).replace('.', 'p')}km.csv.gz"
        for width in unique_widths
    }
    samples_by_width = resolve_corridor_samples(
        csv_path,
        cache_paths=cache_paths,
        railway_context=railway_context,
        buffer_widths_km=tuple(unique_widths),
        chunksize=int(args.chunksize),
    )

    profile_tables = []
    summary_tables = []
    for width in unique_widths:
        width_key = float(width)
        profile_tables.append(
            build_profile_table(
                samples_by_width[width_key],
                buffer_width_km=width_key,
                bin_length_km=profile_bin_km,
                std_scale=profile_std_scale,
            )
        )
        summary_tables.append(build_summary_table(samples_by_width[width_key], buffer_width_km=width_key))

    combined_profiles = pd.concat(profile_tables, ignore_index=True)
    combined_summary = pd.concat(summary_tables, ignore_index=True)
    donut_stats = build_donut_stats(samples_by_width)
    interval_stats = build_interval_stats_table(
        samples_by_width,
        width_order_desc=width_order_desc,
        interval_km=BAR_INTERVAL_KM,
    )

    profiles_path = table_dir / f"{FIG_BASENAME}_profiles_1_2_3km.csv"
    summary_path = table_dir / f"{FIG_BASENAME}_summary_1_2_3km.csv"
    donut_stats_path = table_dir / f"{FIG_BASENAME}_donut_stats_1_2_3km.csv"
    interval_stats_path = table_dir / f"{FIG_BASENAME}_interval_stats_400_1000km.csv"
    combined_profiles.to_csv(profiles_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    donut_stats.to_csv(donut_stats_path, index=False)
    interval_stats.to_csv(interval_stats_path, index=False)

    fig_png, fig_pdf = build_figure(
        samples_by_width,
        width_order_desc=width_order_desc,
        bin_length_km=profile_bin_km,
        std_scale=profile_std_scale,
        fig_dir=fig_dir,
    )

    meta_path = cache_dir / f"{FIG_BASENAME}_meta_1_2_3km.json"
    meta_path.write_text(
        json.dumps(
            {
                "figure_png": str(fig_png),
                "figure_pdf": str(fig_pdf),
                "sample_caches": {f"{width:.1f}km": str(cache_paths[float(width)]) for width in unique_widths},
                "profiles_csv": str(profiles_path),
                "summary_csv": str(summary_path),
                "donut_stats_csv": str(donut_stats_path),
                "interval_stats_csv": str(interval_stats_path),
                "buffer_widths_km": list(sorted(unique_widths)),
                "profile_bin_km": profile_bin_km,
                "profile_std_scale": profile_std_scale,
                "bar_interval_km": list(BAR_INTERVAL_KM),
                "railway_main_line_length_km": float(railway_context["main_line_length_km"]),
                "n_corridor_pixels": {
                    f"{width:.1f}km": int(len(samples_by_width[float(width)]))
                    for width in unique_widths
                },
                "sampling_filter": "d_u <= 0",
                "permafrost_extreme_du_threshold_mm_per_yr": PERMAFROST_EXTREME_DU_THRESHOLD,
                "non_permafrost_extreme_du_threshold_mm_per_yr": NON_PERMAFROST_EXTREME_DU_THRESHOLD,
                "permafrost_extreme_grad_threshold_mm_per_yr_per_km": PERMAFROST_EXTREME_GRAD_THRESHOLD,
                "non_permafrost_extreme_grad_threshold_mm_per_yr_per_km": NON_PERMAFROST_EXTREME_GRAD_THRESHOLD,
            },
            indent=2,
        )
    )

    log_step(f"saved figure PNG: {fig_png}")
    log_step(f"saved figure PDF: {fig_pdf}")
    log_step(f"saved profile table: {profiles_path}")
    log_step(f"saved summary table: {summary_path}")
    log_step(f"saved donut stats table: {donut_stats_path}")
    log_step(f"saved interval stats table: {interval_stats_path}")
    log_step(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
