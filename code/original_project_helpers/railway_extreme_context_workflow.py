#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure_reorganized_railway_extreme_deformation_inspection.py
# Renamed package path: code/original_project_helpers/railway_extreme_context_workflow.py
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
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

import Figure_reorganize_railway_buffer_analysis as railbuf
import figure4_regional_deformation_context as fig4


FIG_BASENAME = "Figure_reorganized_railway_extreme_deformation_inspection"
SUSCEPTIBILITY_FIG_BASENAME = "Figure_reorganize_extreme_deformation_susceptibity"
DEFAULT_BUFFER_WIDTH_KM = 3.0
AXIS_CRS = "EPSG:4326"
MAX_BACKGROUND_POINTS = 250_000
MAIN_XLIM = (88.5, 97.5)
MAIN_YMIN = 28.3
DONUT_BUFFER_WIDTHS_KM = tuple(sorted(railbuf.BUFFER_WIDTHS_KM_DEFAULT, reverse=True))
PANEL_LABELS = ("A", "B", "C", "D")
SUMMARY_PANEL_LABELS = ("A", "B", "C", "D", "E", "F")
PROFILE_PANEL_LABELS = ("A", "B")
ZOOM_SPECS = (
    {
        "site_label": "Tuotuohe",
        "bounds": (0.04, 0.66, 0.28, 0.28),
        "arrow_anchor": (0.29, 0.66),
        "xpad": 0.34,
        "ypad": 0.22,
    },
    {
        "site_label": "Wudaoliang",
        "bounds": (0.60, 0.16, 0.35, 0.31),
        "arrow_anchor": (0.60, 0.48),
        "xpad": 0.42,
        "ypad": 0.24,
    },
)

PF_POINT_COLOR = "#4599C6"
NPF_POINT_COLOR = "#A76F4D"
CORRIDOR_FACE = "#EFEFEF"
CORRIDOR_EDGE = "#CFCFCF"
RAILWAY_COLOR = "#222222"
PERMAFROST_BG_COLOR = "#BBD7E4"
METEORO_COLOR = "#3F3F3F"
METEORO_LABEL_OFFSET_PT = 6.0
METEORO_MAIN_MARKER_SIZE = 70.0
METEORO_INSET_MARKER_SIZE = 110.0
PF_INSIDE_POINT_SIZE = 3.0
NPF_INSIDE_POINT_SIZE = 3.5
PF_OUTSIDE_POINT_SIZE = 0.1
NPF_OUTSIDE_POINT_SIZE = 0.1
PF_OUTSIDE_POINT_COLOR = railbuf.blend_with_white(PF_POINT_COLOR, 0.28)
NPF_OUTSIDE_POINT_COLOR = railbuf.blend_with_white(NPF_POINT_COLOR, 0.28)
OUTSIDE_NOTE_COLOR = railbuf.blend_with_white("0.35", 0.50)
SHANNAN_LABEL_X = 91.7
SHANNAN_LABEL_Y_OFFSET = -0.3

TARGET_INFO = {
    "d_u": {
        "label": r"$\mathbf{d}_{\mathbf{u}}$",
        "unit": "mm/yr",
        "base_color": railbuf.DU_BASE_COLOR,
        "susceptibility_col": "susceptibility_du",
        "profile_metric": "susceptibility_du",
        "profile_unit": "Mean prob.",
        "threshold_pf": railbuf.PERMAFROST_EXTREME_DU_THRESHOLD,
        "threshold_npf": railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD,
    },
    "grad_mag_km": {
        "label": r"$|\nabla \mathbf{d}_{\mathbf{u}}|$",
        "unit": "mm/yr/km",
        "base_color": railbuf.GRAD_BASE_COLOR,
        "susceptibility_col": "susceptibility_grad",
        "profile_metric": "susceptibility_grad",
        "profile_unit": "Mean prob.",
        "threshold_pf": railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD,
        "threshold_npf": railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD,
    },
}

SUMMARY_YLIMS_BY_PANEL = {
    "A": (-8.1, -4.2),
    "B": (2.8, 6.2),
    "C": (3.4, 9.1),
    "D": (12.0, 17.5),
    "F": (2.2, 9.1),
}
SUSCEPTIBILITY_PROFILE_YTICKS = [0.2, 0.4, 0.6, 0.8]
PROFILE_SITE_LABEL_Y = 0.95
PROFILE_SITE_LABEL_FONT_SIZE = 8.6
PROFILE_SITE_LINE_COLOR = railbuf.blend_with_white("0.20", 0.55)
PROFILE_SITE_LABELS = {"Lhasa", "Nagqu", "Tuotuohe", "Wudaoliang", "Golmud"}


def log_step(message: str) -> None:
    print(f"[{FIG_BASENAME}] {message}")


def degree_formatter(value: float, _pos: int) -> str:
    return f"{value:.1f}"


def plot_geometry(
    ax,
    geom,
    *,
    crs,
    target_crs: str | None = None,
    facecolor: str | None = None,
    edgecolor: str = "0.5",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: float | None = None,
) -> None:
    if geom is None or geom.is_empty:
        return
    series = railbuf.gpd.GeoSeries([geom], crs=crs)
    if target_crs is not None:
        series = series.to_crs(target_crs)
    draw_zorder = zorder if zorder is not None else (0 if facecolor is not None else 1)
    if facecolor is not None:
        series.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=draw_zorder)
    else:
        series.plot(ax=ax, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=draw_zorder)


def configure_map_axis(ax, *, title: str) -> None:
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlabel(r"Longitude ($^\circ$E)")
    ax.set_ylabel(r"Latitude ($^\circ$N)")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_formatter))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.88", linewidth=0.6)
    railbuf.style_open_axes(ax)
    railbuf.apply_bold_ticklabels(ax)


def build_extreme_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    out = sample_df.copy()
    out["is_extreme_du"] = False
    out["is_extreme_grad"] = False

    pf_mask = out["domain"].eq("Permafrost")
    npf_mask = out["domain"].eq("Non-Permafrost")

    out.loc[pf_mask, "is_extreme_du"] = out.loc[pf_mask, "d_u"] < railbuf.PERMAFROST_EXTREME_DU_THRESHOLD
    out.loc[npf_mask, "is_extreme_du"] = out.loc[npf_mask, "d_u"] < railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD
    out.loc[pf_mask, "is_extreme_grad"] = out.loc[pf_mask, "grad_mag_km"] > railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD
    out.loc[npf_mask, "is_extreme_grad"] = out.loc[npf_mask, "grad_mag_km"] > railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD
    return out


def save_dual_format_figure(fig, *, fig_dir: Path, stem: str) -> tuple[Path, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{stem}.png"
    out_pdf = fig_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_susceptibility_meta_path(base_dir: Path, out_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()

    candidates = [
        out_dir / "cache" / f"{SUSCEPTIBILITY_FIG_BASENAME}_meta.json",
        base_dir / "outputs" / "deformation_rate_gradient_lake_paper" / "cache" / f"{SUSCEPTIBILITY_FIG_BASENAME}_meta.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def susceptibility_sidecar_path(prob_path: Path) -> Path:
    return prob_path.with_suffix(prob_path.suffix + ".json")


def load_susceptibility_sources(meta_path: Path) -> dict[str, dict[str, object]]:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing susceptibility metadata: {meta_path}. Run {SUSCEPTIBILITY_FIG_BASENAME}.py first or pass --susceptibility-meta."
        )

    meta = load_json(meta_path)
    targets = meta.get("targets")
    if not isinstance(targets, dict):
        raise RuntimeError(f"Unexpected susceptibility metadata format in {meta_path}.")

    out: dict[str, dict[str, object]] = {}
    for target in ("d_u", "grad_mag_km"):
        target_meta = targets.get(target)
        if not isinstance(target_meta, dict):
            raise RuntimeError(f"Susceptibility metadata is missing target={target}.")
        prob_path = Path(str(target_meta["combined_probability_raster"])).resolve()
        sidecar_path = susceptibility_sidecar_path(prob_path)
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing susceptibility raster for {target}: {prob_path}")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing susceptibility sidecar for {target}: {sidecar_path}")
        sidecar = load_json(sidecar_path)
        cache_signature = sidecar.get("cache_signature")
        if not isinstance(cache_signature, dict) or not isinstance(cache_signature.get("grid"), dict):
            raise RuntimeError(f"Susceptibility sidecar is missing grid metadata: {sidecar_path}")
        out[target] = {
            "prob_path": prob_path,
            "grid": dict(cache_signature["grid"]),
            "sidecar_path": sidecar_path,
        }
    return out


def attach_probability_column(
    df: pd.DataFrame,
    *,
    prob_path: Path,
    grid: dict[str, object],
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    mm = fig4.open_memmap(prob_path, dtype="float32", mode="r", shape=(nrows, ncols))

    easting = pd.to_numeric(out["easting"], errors="coerce").to_numpy(dtype=float)
    northing = pd.to_numeric(out["northing"], errors="coerce").to_numpy(dtype=float)
    row, col = fig4.en_to_rc(
        easting,
        northing,
        res=float(grid["res"]),
        gx0=int(grid["gx0"]),
        gy1=int(grid["gy1"]),
    )
    ok = (
        np.isfinite(easting)
        & np.isfinite(northing)
        & (row >= 0)
        & (row < nrows)
        & (col >= 0)
        & (col < ncols)
    )

    values = np.full(len(out), np.nan, dtype=np.float32)
    if np.any(ok):
        values[ok] = mm[row[ok], col[ok]]
    out[out_col] = values
    return out


def attach_susceptibility_columns(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    susceptibility_sources: dict[str, dict[str, object]],
) -> dict[float, pd.DataFrame]:
    out: dict[float, pd.DataFrame] = {}
    for width, sample_df in samples_by_width.items():
        enriched = sample_df.copy()
        for target, info in susceptibility_sources.items():
            enriched = attach_probability_column(
                enriched,
                prob_path=Path(info["prob_path"]),
                grid=dict(info["grid"]),
                out_col=str(TARGET_INFO[target]["susceptibility_col"]),
            )
        out[float(width)] = enriched
    return out


def build_metric_profile_table(
    sample_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    buffer_width_km: float,
    bin_length_km: float,
    std_scale: float,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for metric in metrics:
        prof = railbuf.summarize_metric_profile(
            sample_df,
            value_col=metric,
            bin_length_km=bin_length_km,
            std_scale=std_scale,
        )
        if prof.empty:
            continue
        prof.insert(0, "metric", metric)
        prof.insert(0, "buffer_width_km", float(buffer_width_km))
        pieces.append(prof)
    if not pieces:
        raise RuntimeError(f"Could not build railway profiles for buffer width={buffer_width_km:.1f} km.")
    return pd.concat(pieces, ignore_index=True)


def build_interval_mean_stats(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    metric: str,
    interval_km: tuple[float, float],
) -> pd.DataFrame:
    domain_groups = railbuf.collect_interval_groups_by_domain(
        samples_by_width,
        width_order_desc=width_order_desc,
        metric=metric,
        interval_km=interval_km,
    )
    rows: list[dict[str, object]] = []
    for domain in ("Permafrost", "Non-Permafrost"):
        for width in width_order_desc:
            vals = np.asarray(domain_groups[domain][float(width)], dtype=float)
            vals = vals[np.isfinite(vals)]
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(
                {
                    "stat_kind": "interval_mean",
                    "domain": domain,
                    "buffer_width_km": float(width),
                    "n": int(len(vals)),
                    "value": float(np.nanmean(vals)) if len(vals) else np.nan,
                    "std": std,
                    "err": 0.1 * std,
                }
            )
    return pd.DataFrame(rows)


def build_extreme_portion_stats(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    target: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for width in width_order_desc:
        sample_df = samples_by_width[float(width)]
        for domain in ("Permafrost", "Non-Permafrost"):
            domain_df = sample_df.loc[sample_df["domain"].eq(domain)].copy()
            values = pd.to_numeric(domain_df[target], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if target == "d_u":
                threshold = TARGET_INFO[target]["threshold_pf"] if domain == "Permafrost" else TARGET_INFO[target]["threshold_npf"]
                extreme = valid & (values < float(threshold))
            else:
                threshold = TARGET_INFO[target]["threshold_pf"] if domain == "Permafrost" else TARGET_INFO[target]["threshold_npf"]
                extreme = valid & (values > float(threshold))
            n = int(np.count_nonzero(valid))
            pct = 100.0 * float(np.count_nonzero(extreme)) / n if n > 0 else np.nan
            rows.append(
                {
                    "stat_kind": "extreme_portion_pct",
                    "domain": domain,
                    "buffer_width_km": float(width),
                    "n": n,
                    "value": pct,
                    "std": np.nan,
                    "err": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_susceptibility_stats(
    samples_by_width: dict[float, pd.DataFrame],
    *,
    width_order_desc: tuple[float, ...],
    target: str,
) -> pd.DataFrame:
    col = str(TARGET_INFO[target]["susceptibility_col"])
    rows: list[dict[str, object]] = []
    for width in width_order_desc:
        sample_df = samples_by_width[float(width)]
        for domain in ("Permafrost", "Non-Permafrost"):
            vals = pd.to_numeric(sample_df.loc[sample_df["domain"].eq(domain), col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(
                {
                    "stat_kind": "mean_susceptibility_pct",
                    "domain": domain,
                    "buffer_width_km": float(width),
                    "n": int(len(vals)),
                    "value": 100.0 * float(np.nanmean(vals)) if len(vals) else np.nan,
                    "std": 100.0 * std,
                    "err": 10.0 * std,
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_buffer_bars(
    ax,
    *,
    stats_df: pd.DataFrame,
    width_order_desc: tuple[float, ...],
    colors_by_width: dict[float, tuple[float, float, float]],
    ylabel: str,
    title: str,
    y_tick_formatter=None,
) -> None:
    x = np.arange(len(width_order_desc), dtype=float)
    bar_width = 0.32

    pf_vals = []
    pf_errs = []
    npf_vals = []
    npf_errs = []
    for width in width_order_desc:
        sub = stats_df.loc[np.isclose(stats_df["buffer_width_km"], float(width))]
        pf_row = sub.loc[sub["domain"].eq("Permafrost")].iloc[0]
        npf_row = sub.loc[sub["domain"].eq("Non-Permafrost")].iloc[0]
        pf_vals.append(float(pf_row["value"]))
        pf_errs.append(float(pf_row["err"]) if np.isfinite(pf_row["err"]) else 0.0)
        npf_vals.append(float(npf_row["value"]))
        npf_errs.append(float(npf_row["err"]) if np.isfinite(npf_row["err"]) else 0.0)

    pf_vals_arr = np.asarray(pf_vals, dtype=float)
    pf_errs_arr = np.asarray(pf_errs, dtype=float)
    npf_vals_arr = np.asarray(npf_vals, dtype=float)
    npf_errs_arr = np.asarray(npf_errs, dtype=float)

    for idx, width in enumerate(width_order_desc):
        color = colors_by_width[float(width)]
        ax.bar(
            x[idx] - 0.5 * bar_width,
            pf_vals_arr[idx],
            yerr=pf_errs_arr[idx] if pf_errs_arr[idx] > 0.0 else None,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=0.9,
            ecolor="0.20",
            capsize=2.8,
            zorder=2,
        )
        ax.bar(
            x[idx] + 0.5 * bar_width,
            npf_vals_arr[idx],
            yerr=npf_errs_arr[idx] if npf_errs_arr[idx] > 0.0 else None,
            width=bar_width,
            color=railbuf.blend_with_white(color, 0.65),
            edgecolor=color,
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            hatch="///",
            ecolor="0.20",
            capsize=2.8,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([railbuf.format_width_label(width) for width in width_order_desc])
    ax.set_xlabel("Buffer (km)")
    ax.xaxis.label.set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_fontweight("bold")
    ax.set_title(title, fontweight="bold", pad=4)
    ax.grid(False)
    railbuf.style_open_axes(ax)
    railbuf.apply_bold_ticklabels(ax)

    combined = np.concatenate(
        [
            pf_vals_arr[np.isfinite(pf_vals_arr)],
            npf_vals_arr[np.isfinite(npf_vals_arr)],
        ]
    )
    if combined.size:
        y_min = float(np.nanmin(combined))
        y_max = float(np.nanmax(combined))
        span = y_max - y_min
        pad = max(0.06 * (span if span > 0.0 else max(abs(y_max), 1.0)), 0.04)
        if y_min >= 0.0:
            ax.set_ylim(0.0, y_max + 1.6 * pad)
        else:
            ax.set_ylim(y_min - 0.8 * pad, y_max + 1.6 * pad)

    if y_tick_formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(y_tick_formatter))


def add_vertical_bar_legend(fig) -> None:
    legend_ax = fig.add_axes([0.010, 0.26, 0.060, 0.48])
    legend_ax.set_axis_off()
    legend_ax.add_patch(
        Rectangle(
            (0.10, 0.63),
            0.26,
            0.12,
            facecolor=(1.0, 1.0, 1.0, 0.92),
            edgecolor="0.45",
            linewidth=1.2,
            hatch="///",
        )
    )
    legend_ax.add_patch(
        Rectangle(
            (0.10, 0.28),
            0.26,
            0.12,
            facecolor="0.55",
            edgecolor="0.55",
            linewidth=1.0,
        )
    )
    legend_ax.text(
        0.50,
        0.69,
        "Non-permafrost",
        rotation=270,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="0.08",
    )
    legend_ax.text(
        0.50,
        0.34,
        "Permafrost",
        rotation=270,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="0.08",
    )


def build_summary_bar_figure(
    *,
    samples_by_width: dict[float, pd.DataFrame],
    width_order_desc: tuple[float, ...],
    fig_dir: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    du_colors = railbuf.make_buffer_color_map(railbuf.DU_BASE_COLOR, width_order_desc)
    grad_colors = railbuf.make_buffer_color_map(railbuf.GRAD_BASE_COLOR, width_order_desc)

    stats_by_target = {
        "d_u": {
            "interval": build_interval_mean_stats(
                samples_by_width,
                width_order_desc=width_order_desc,
                metric="d_u",
                interval_km=railbuf.BAR_INTERVAL_KM,
            ),
            "extreme": build_extreme_portion_stats(samples_by_width, width_order_desc=width_order_desc, target="d_u"),
            "susceptibility": build_susceptibility_stats(samples_by_width, width_order_desc=width_order_desc, target="d_u"),
            "colors": du_colors,
        },
        "grad_mag_km": {
            "interval": build_interval_mean_stats(
                samples_by_width,
                width_order_desc=width_order_desc,
                metric="grad_mag_km",
                interval_km=railbuf.BAR_INTERVAL_KM,
            ),
            "extreme": build_extreme_portion_stats(samples_by_width, width_order_desc=width_order_desc, target="grad_mag_km"),
            "susceptibility": build_susceptibility_stats(
                samples_by_width,
                width_order_desc=width_order_desc,
                target="grad_mag_km",
            ),
            "colors": grad_colors,
        },
    }

    fig = plt.figure(figsize=(13.4, 7.9))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.09,
        wspace=0.22,
        hspace=0.34,
    )

    panel_idx = 0
    all_stats: list[pd.DataFrame] = []
    row_specs = [
        ("d_u", r"Extreme $d_u$"),
        ("grad_mag_km", r"Extreme $|\nabla d_u|$"),
    ]
    col_titles = (
        "400-1000 km mean",
        "Extreme portion in buffer",
        "Mean susceptibility in buffer",
    )

    for row, (target, _) in enumerate(row_specs):
        info = TARGET_INFO[target]
        label = str(info["label"])
        colors_by_width = stats_by_target[target]["colors"]
        interval_stats = stats_by_target[target]["interval"].copy()
        interval_stats["target"] = target
        interval_stats["panel"] = "interval_mean"
        extreme_stats = stats_by_target[target]["extreme"].copy()
        extreme_stats["target"] = target
        extreme_stats["panel"] = "extreme_portion_pct"
        susceptibility_stats = stats_by_target[target]["susceptibility"].copy()
        susceptibility_stats["target"] = target
        susceptibility_stats["panel"] = "mean_susceptibility_pct"
        all_stats.extend([interval_stats, extreme_stats, susceptibility_stats])

        ax_interval = fig.add_subplot(gs[row, 0])
        plot_grouped_buffer_bars(
            ax_interval,
            stats_df=interval_stats,
            width_order_desc=width_order_desc,
            colors_by_width=colors_by_width,
            ylabel=str(info["unit"]),
            title=rf"{col_titles[0]} {label}",
        )
        panel_label = SUMMARY_PANEL_LABELS[panel_idx]
        if panel_label in SUMMARY_YLIMS_BY_PANEL:
            ax_interval.set_ylim(*SUMMARY_YLIMS_BY_PANEL[panel_label])
        add_subplot_label(ax_interval, panel_label)
        panel_idx += 1

        ax_extreme = fig.add_subplot(gs[row, 1])
        plot_grouped_buffer_bars(
            ax_extreme,
            stats_df=extreme_stats,
            width_order_desc=width_order_desc,
            colors_by_width=colors_by_width,
            ylabel="Portion (%)",
            title=rf"{col_titles[1]} {label}",
            y_tick_formatter=lambda y, _pos: f"{y:.0f}%",
        )
        panel_label = SUMMARY_PANEL_LABELS[panel_idx]
        if panel_label in SUMMARY_YLIMS_BY_PANEL:
            ax_extreme.set_ylim(*SUMMARY_YLIMS_BY_PANEL[panel_label])
        if panel_label == "B":
            ax_extreme.set_yticks([3.0, 4.0, 5.0, 6.0])
            ax_extreme.set_yticklabels(["3%", "4%", "5%", "6%"])
            railbuf.apply_bold_ticklabels(ax_extreme)
        add_subplot_label(ax_extreme, panel_label)
        panel_idx += 1

        ax_susc = fig.add_subplot(gs[row, 2])
        susc_title = rf"Mean extreme {label} susceptibility in buffer"
        plot_grouped_buffer_bars(
            ax_susc,
            stats_df=susceptibility_stats,
            width_order_desc=width_order_desc,
            colors_by_width=colors_by_width,
            ylabel="Mean prob. (%)",
            title=susc_title,
            y_tick_formatter=lambda y, _pos: f"{y:.0f}%",
        )
        panel_label = SUMMARY_PANEL_LABELS[panel_idx]
        if panel_label in SUMMARY_YLIMS_BY_PANEL:
            ax_susc.set_ylim(*SUMMARY_YLIMS_BY_PANEL[panel_label])
        add_subplot_label(ax_susc, panel_label)
        panel_idx += 1

    add_vertical_bar_legend(fig)

    stats_out = pd.concat(all_stats, ignore_index=True)
    return (*save_dual_format_figure(fig, fig_dir=fig_dir, stem=f"{FIG_BASENAME}_buffer_bars"), stats_out)


def build_profile_figure(
    *,
    samples_by_width: dict[float, pd.DataFrame],
    width_order_desc: tuple[float, ...],
    bin_length_km: float,
    std_scale: float,
    fig_dir: Path,
    stem: str,
    metric_specs: tuple[tuple[str, str, str, str], ...],
    profile_sites: pd.DataFrame | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    profile_tables = {
        float(width): build_metric_profile_table(
            sample_df,
            metrics=tuple(spec[0] for spec in metric_specs),
            buffer_width_km=float(width),
            bin_length_km=bin_length_km,
            std_scale=std_scale,
        )
        for width, sample_df in samples_by_width.items()
    }

    fig = plt.figure(figsize=(10.6, 12.0))
    gs = fig.add_gridspec(
        len(metric_specs),
        1,
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.955,
        hspace=0.26,
    )

    shared_x_ax = None
    for block_idx, (metric, title, base_color, unit_label) in enumerate(metric_specs):
        gs_block = gs[block_idx, 0].subgridspec(len(width_order_desc), 1, hspace=0.06)
        axes = []
        for row_idx, width in enumerate(width_order_desc):
            sharex = shared_x_ax if shared_x_ax is not None else None
            ax = fig.add_subplot(gs_block[row_idx, 0], sharex=sharex)
            if shared_x_ax is None:
                shared_x_ax = ax
            axes.append(ax)
            profile_df = profile_tables[float(width)].loc[profile_tables[float(width)]["metric"].eq(metric)].copy()
            colors_by_width = railbuf.make_buffer_color_map(base_color, width_order_desc)
            railbuf.plot_profile_strip(
                ax,
                profile_df=profile_df,
                color=colors_by_width[float(width)],
                base_color=base_color,
                metric_kind="du" if metric in {"d_u", "susceptibility_du"} else "grad",
                side="left",
                width_km=float(width),
                unit_label=unit_label,
                show_x_labels=(row_idx == len(width_order_desc) - 1),
                title=title if row_idx == 0 else None,
                add_zero_line=(metric == "d_u"),
            )
            if metric.startswith("susceptibility_"):
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks(SUSCEPTIBILITY_PROFILE_YTICKS)
                ax.set_yticklabels([f"{tick:.1f}" for tick in SUSCEPTIBILITY_PROFILE_YTICKS])
                railbuf.apply_bold_ticklabels(ax)
            if profile_sites is not None and not profile_sites.empty:
                add_profile_site_annotations(ax, profile_sites)
        add_subplot_label(axes[0], PROFILE_PANEL_LABELS[block_idx])

    profile_out = pd.concat(profile_tables.values(), ignore_index=True)
    return (*save_dual_format_figure(fig, fig_dir=fig_dir, stem=stem), profile_out)


def resolve_site_label(station_gdf) -> pd.Series:
    for col in ("english_st", "english_station_name", "station_na", "station_name", "NAME"):
        if col not in station_gdf.columns:
            continue
        labels = station_gdf[col].astype(str).str.strip()
        labels = labels.mask(labels.eq("")).mask(labels.str.lower().isin({"nan", "none"}))
        if labels.notna().any():
            return labels
    if "station_id" in station_gdf.columns:
        return station_gdf["station_id"].map(lambda value: f"Site {value}")
    raise RuntimeError("Could not resolve a site-label field from the meteoro station shapefile.")


def load_meteoro_sites(meteoro_shp: Path) -> pd.DataFrame:
    station_gdf = railbuf.gpd.read_file(meteoro_shp)
    station_gdf = station_gdf.loc[station_gdf.geometry.notna() & ~station_gdf.geometry.is_empty].copy()
    if station_gdf.empty:
        raise RuntimeError(f"No valid meteoro station geometries found in {meteoro_shp}.")

    if {"longitude", "latitude"}.issubset(station_gdf.columns):
        longitude = pd.to_numeric(station_gdf["longitude"], errors="coerce")
        latitude = pd.to_numeric(station_gdf["latitude"], errors="coerce")
    else:
        if station_gdf.crs is None:
            raise RuntimeError(f"Meteoro station shapefile is missing CRS information: {meteoro_shp}")
        station_ll = station_gdf.to_crs(AXIS_CRS)
        longitude = station_ll.geometry.x.astype(float)
        latitude = station_ll.geometry.y.astype(float)

    site_df = pd.DataFrame(
        {
            "site_label": resolve_site_label(station_gdf),
            "longitude": longitude,
            "latitude": latitude,
        }
    )
    site_df = site_df.loc[
        site_df["site_label"].notna() & np.isfinite(site_df["longitude"]) & np.isfinite(site_df["latitude"])
    ].copy()
    if site_df.empty:
        raise RuntimeError(f"No plottable meteoro sites were found in {meteoro_shp}.")
    return site_df


def load_profile_sites(meteoro_shp: Path, *, railway_context: dict[str, object]) -> pd.DataFrame:
    station_gdf = railbuf.gpd.read_file(meteoro_shp)
    station_gdf = station_gdf.loc[station_gdf.geometry.notna() & ~station_gdf.geometry.is_empty].copy()
    if station_gdf.empty:
        raise RuntimeError(f"No valid meteoro station geometries found in {meteoro_shp}.")

    labels = resolve_site_label(station_gdf)
    if station_gdf.crs is None:
        raise RuntimeError(f"Meteoro station shapefile is missing CRS information: {meteoro_shp}")
    if station_gdf.crs != railway_context["crs"]:
        station_gdf = station_gdf.to_crs(railway_context["crs"])

    rows: list[dict[str, object]] = []
    main_line = railway_context["main_line"]
    for label, geom in zip(labels, station_gdf.geometry, strict=False):
        label_str = str(label)
        if label_str not in PROFILE_SITE_LABELS:
            continue
        along_km = float(main_line.project(geom)) / 1000.0
        if np.isfinite(along_km):
            rows.append({"site_label": label_str, "along_km": along_km})

    if not rows:
        raise RuntimeError(f"Could not project meteoro site positions onto the railway for {meteoro_shp}.")
    return pd.DataFrame(rows).sort_values("along_km").reset_index(drop=True)


def add_profile_site_annotations(ax, profile_sites: pd.DataFrame) -> None:
    if profile_sites.empty:
        return

    xmin, xmax = map(float, ax.get_xlim())
    visible = profile_sites.loc[
        profile_sites["along_km"].between(xmin, xmax, inclusive="both")
    ].copy()
    if visible.empty:
        return

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for site in visible.itertuples(index=False):
        x = float(site.along_km)
        if x <= xmin + 0.04 * (xmax - xmin):
            ha = "left"
            x_offset = 3.0
        elif x >= xmax - 0.04 * (xmax - xmin):
            ha = "right"
            x_offset = -3.0
        else:
            ha = "center"
            x_offset = 0.0

        ax.axvline(
            x,
            color=PROFILE_SITE_LINE_COLOR,
            linewidth=0.75,
            linestyle=(0, (1.5, 2.5)),
            zorder=0,
            alpha=0.95,
        )
        txt = ax.annotate(
            str(site.site_label),
            xy=(x, PROFILE_SITE_LABEL_Y),
            xycoords=trans,
            xytext=(x_offset, 0.0),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=PROFILE_SITE_LABEL_FONT_SIZE,
            fontweight="bold",
            color="0.24",
            clip_on=False,
            zorder=4,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def compute_main_ylim(corridor_geom, *, crs, meteoro_sites: pd.DataFrame) -> tuple[float, float]:
    bounds = railbuf.gpd.GeoSeries([corridor_geom], crs=crs).to_crs(AXIS_CRS).total_bounds
    ymin = float(bounds[1])
    ymax = float(bounds[3])
    if not meteoro_sites.empty:
        ymin = min(ymin, float(meteoro_sites["latitude"].min()))
        ymax = max(ymax, float(meteoro_sites["latitude"].max()))
    ypad = max(0.15, 0.04 * (ymax - ymin))
    return MAIN_YMIN, ymax + ypad


def resolve_permafrost_background(
    csv_path: Path,
    *,
    cache_path: Path,
    chunksize: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> pd.DataFrame:
    required_cols = {"longitude", "latitude"}
    if cache_path.exists():
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if required_cols.issubset(cache_cols):
            return pd.read_csv(cache_path)

    parts: list[pd.DataFrame] = []
    xmin = float(xlim[0]) - 0.6
    xmax = float(xlim[1]) + 0.6
    ymin = float(ylim[0]) - 0.6
    ymax = float(ylim[1]) + 0.6
    usecols = ["longitude", "latitude", "Perma_Distr_map"]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        lon = pd.to_numeric(chunk["longitude"], errors="coerce")
        lat = pd.to_numeric(chunk["latitude"], errors="coerce")
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy(dtype=float) == 1.0
        mask = pf & np.isfinite(lon) & np.isfinite(lat) & (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
        if not np.any(mask):
            continue
        parts.append(pd.DataFrame({"longitude": lon.loc[mask].to_numpy(), "latitude": lat.loc[mask].to_numpy()}))

    if not parts:
        raise RuntimeError("No permafrost background pixels were found for the requested map extent.")

    bg_df = pd.concat(parts, ignore_index=True)
    if len(bg_df) > MAX_BACKGROUND_POINTS:
        bg_df = bg_df.sample(MAX_BACKGROUND_POINTS, random_state=42)
    bg_df = bg_df.reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    bg_df.to_csv(cache_path, index=False, compression="gzip")
    return bg_df


def resolve_extreme_map_points(
    csv_path: Path,
    *,
    cache_path: Path,
    railway_context: dict[str, object],
    buffer_width_km: float,
    chunksize: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> pd.DataFrame:
    required_cols = {
        "longitude",
        "latitude",
        "domain",
        "is_extreme_du",
        "is_extreme_grad",
        "inside_corridor",
    }
    if cache_path.exists():
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if required_cols.issubset(cache_cols):
            cached = pd.read_csv(cache_path)
            for col in ("is_extreme_du", "is_extreme_grad", "inside_corridor"):
                cached[col] = (
                    cached[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"1", "true", "t", "yes", "y"})
                )
            return cached

    usecols = railbuf.resolve_usecols(csv_path)
    if "longitude" not in usecols or "latitude" not in usecols:
        raise RuntimeError("The input pixel CSV must contain longitude and latitude columns for REDI map plotting.")

    network_geom = railway_context["network_geom"]
    crs = railway_context["crs"]
    corridor_half_width_m = 500.0 * float(buffer_width_km)
    xmin, xmax = map(float, xlim)
    ymin, ymax = map(float, ylim)
    parts: list[pd.DataFrame] = []

    for idx, chunk in enumerate(pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False), start=1):
        if idx == 1 or idx % 10 == 0:
            log_step(f"processed {idx} chunk(s) for full-extent extreme-point map cache")

        chunk = railbuf.engineer_features(chunk)
        mask = (
            np.isfinite(chunk["easting"])
            & np.isfinite(chunk["northing"])
            & np.isfinite(chunk["longitude"])
            & np.isfinite(chunk["latitude"])
            & np.isfinite(chunk["d_u"])
            & np.isfinite(chunk["grad_mag_km"])
            & chunk["domain"].isin(["Permafrost", "Non-Permafrost"])
            & (chunk["d_u"] <= 0.0)
            & chunk["longitude"].between(xmin, xmax, inclusive="both")
            & chunk["latitude"].between(ymin, ymax, inclusive="both")
        )
        if not mask.any():
            continue

        sub = build_extreme_table(chunk.loc[mask].copy())
        sub = sub.loc[sub["is_extreme_du"] | sub["is_extreme_grad"]].copy()
        if sub.empty:
            continue

        points = railbuf.gpd.GeoSeries(railbuf.gpd.points_from_xy(sub["easting"], sub["northing"]), crs=crs)
        dist_to_rail = points.distance(network_geom).to_numpy(dtype=float)
        sub["inside_corridor"] = np.isfinite(dist_to_rail) & (dist_to_rail <= corridor_half_width_m)
        parts.append(
            sub.loc[
                :,
                ["longitude", "latitude", "domain", "is_extreme_du", "is_extreme_grad", "inside_corridor"],
            ].reset_index(drop=True)
        )

    if not parts:
        raise RuntimeError("No plottable extreme map points were found for the requested REDI extent.")

    out = pd.concat(parts, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, compression="gzip")
    return out


def add_subplot_label(ax, label: str) -> None:
    ax.text(
        -0.07,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="black",
        clip_on=False,
        zorder=8,
    )


def plot_panel_content(
    ax,
    *,
    corridor_geom,
    crs,
    railway_geom,
    permafrost_bg: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> None:
    plot_geometry(
        ax,
        corridor_geom,
        crs=crs,
        target_crs=AXIS_CRS,
        facecolor=CORRIDOR_FACE,
        edgecolor=CORRIDOR_EDGE,
        linewidth=0.8,
        alpha=0.85,
        zorder=0.2,
    )
    ax.scatter(
        permafrost_bg["longitude"],
        permafrost_bg["latitude"],
        s=5.0,
        marker="s",
        color=PERMAFROST_BG_COLOR,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.62,
        rasterized=True,
        zorder=1.0,
    )
    plot_geometry(
        ax,
        railway_geom,
        crs=crs,
        target_crs=AXIS_CRS,
        facecolor=None,
        edgecolor=RAILWAY_COLOR,
        linewidth=1.4,
        alpha=1.0,
        zorder=3.0,
    )

    inside_mask = sub_df["inside_corridor"].astype(bool) if "inside_corridor" in sub_df.columns else pd.Series(True, index=sub_df.index)
    pf_df = sub_df.loc[sub_df["domain"].eq("Permafrost")]
    npf_df = sub_df.loc[sub_df["domain"].eq("Non-Permafrost")]
    pf_outside = pf_df.loc[~inside_mask.loc[pf_df.index]]
    npf_outside = npf_df.loc[~inside_mask.loc[npf_df.index]]
    pf_inside = pf_df.loc[inside_mask.loc[pf_df.index]]
    npf_inside = npf_df.loc[inside_mask.loc[npf_df.index]]

    ax.scatter(
        pf_outside["longitude"],
        pf_outside["latitude"],
        s=PF_OUTSIDE_POINT_SIZE,
        color=PF_OUTSIDE_POINT_COLOR,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.78,
        zorder=2.0,
    )
    ax.scatter(
        npf_outside["longitude"],
        npf_outside["latitude"],
        s=NPF_OUTSIDE_POINT_SIZE,
        color=NPF_OUTSIDE_POINT_COLOR,
        marker="D",
        edgecolors="none",
        linewidths=0.0,
        alpha=0.78,
        zorder=2.0,
    )
    ax.scatter(
        pf_inside["longitude"],
        pf_inside["latitude"],
        s=PF_INSIDE_POINT_SIZE,
        color=PF_POINT_COLOR,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.95,
        zorder=4.0,
    )
    ax.scatter(
        npf_inside["longitude"],
        npf_inside["latitude"],
        s=NPF_INSIDE_POINT_SIZE,
        color=NPF_POINT_COLOR,
        marker="D",
        edgecolors="none",
        linewidths=0.0,
        alpha=0.95,
        zorder=4.0,
    )


def add_site_marker(ax, site_row: pd.Series, *, show_label: bool, marker_size: float = METEORO_MAIN_MARKER_SIZE) -> None:
    ax.scatter(
        [float(site_row["longitude"])],
        [float(site_row["latitude"])],
        s=marker_size,
        marker="^",
        color=METEORO_COLOR,
        edgecolors=METEORO_COLOR,
        linewidths=0.5,
        zorder=5.0,
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
        color=METEORO_COLOR,
        zorder=6.0,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def add_meteoro_sites(ax, site_df: pd.DataFrame) -> None:
    if site_df.empty:
        return
    xmid = float(np.mean(ax.get_xlim()))
    for site in site_df.itertuples(index=False):
        site_row = pd.Series({"site_label": site.site_label, "longitude": site.longitude, "latitude": site.latitude})
        add_site_marker(ax, site_row, show_label=False)
        if float(site.longitude) >= xmid:
            dx = -METEORO_LABEL_OFFSET_PT
            ha = "right"
        else:
            dx = METEORO_LABEL_OFFSET_PT
            ha = "left"
        if str(site.site_label).strip().lower() == "shannan":
            text = ax.annotate(
                str(site.site_label),
                xy=(float(site.longitude), float(site.latitude)),
                xytext=(SHANNAN_LABEL_X, float(site.latitude) + SHANNAN_LABEL_Y_OFFSET),
                textcoords="data",
                ha="right",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=METEORO_COLOR,
                zorder=6.0,
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
                color=METEORO_COLOR,
                zorder=6.0,
            )
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def axes_fraction_to_data(ax, x_frac: float, y_frac: float) -> tuple[float, float]:
    display_xy = ax.transAxes.transform((x_frac, y_frac))
    data_xy = ax.transData.inverted().transform(display_xy)
    return float(data_xy[0]), float(data_xy[1])


def add_zoom_insets(
    ax,
    *,
    corridor_geom,
    crs,
    railway_geom,
    permafrost_bg: pd.DataFrame,
    sub_df: pd.DataFrame,
    meteoro_sites: pd.DataFrame,
) -> None:
    for spec in ZOOM_SPECS:
        match = meteoro_sites.loc[meteoro_sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site_row = match.iloc[0]
        inset_ax = ax.inset_axes(spec["bounds"])
        plot_panel_content(
            inset_ax,
            corridor_geom=corridor_geom,
            crs=crs,
            railway_geom=railway_geom,
            permafrost_bg=permafrost_bg,
            sub_df=sub_df,
        )
        inset_ax.set_xlim(float(site_row["longitude"]) - float(spec["xpad"]), float(site_row["longitude"]) + float(spec["xpad"]))
        inset_ax.set_ylim(float(site_row["latitude"]) - float(spec["ypad"]), float(site_row["latitude"]) + float(spec["ypad"]))
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.grid(False)
        inset_ax.set_facecolor("white")
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
            spine.set_color("0.18")
        add_site_marker(inset_ax, site_row, show_label=True, marker_size=METEORO_INSET_MARKER_SIZE)

        arrow_end = axes_fraction_to_data(ax, float(spec["arrow_anchor"][0]), float(spec["arrow_anchor"][1]))
        arrow = FancyArrowPatch(
            (float(site_row["longitude"]), float(site_row["latitude"])),
            arrow_end,
            arrowstyle="->",
            mutation_scale=18.0,
            linewidth=1.4,
            color="0.15",
            shrinkA=6.0,
            shrinkB=6.0,
            zorder=3.6,
        )
        ax.add_patch(arrow)


def add_donut_panels(
    *,
    fig,
    top_spec,
    donut_samples_by_width: dict[float, pd.DataFrame],
) -> tuple[object, object]:
    width_order_desc = tuple(sorted((float(width) for width in donut_samples_by_width), reverse=True))
    if len(width_order_desc) != 3:
        raise RuntimeError("REDI donut panels require exactly three buffer widths.")

    donut_stats = railbuf.build_donut_stats(donut_samples_by_width)
    domain_pf_ring_colors = railbuf.make_ring_color_map(railbuf.PERMAFROST_COLOR, width_order_desc, extra_blend=0.20)
    domain_npf_ring_colors = railbuf.make_ring_color_map(railbuf.NON_PERMAFROST_COLOR, width_order_desc, extra_blend=0.20)
    du_ring_colors = railbuf.make_ring_color_map(railbuf.DU_BASE_COLOR, width_order_desc, extra_blend=0.16)
    grad_ring_colors = railbuf.make_ring_color_map(railbuf.GRAD_BASE_COLOR, width_order_desc, extra_blend=0.16)

    sample_domain_rings = {
        float(width): railbuf.get_stat_pair(
            donut_stats,
            group="sample_domain",
            label_order=("Permafrost", "Non-Permafrost"),
            width=width,
        )
        for width in width_order_desc
    }
    permafrost_du_rings = {
        float(width): railbuf.get_stat_pair(
            donut_stats,
            group="permafrost_extreme_du",
            label_order=("Extreme", railbuf.EXTREME_REFERENCE_LABEL),
            width=width,
        )
        for width in width_order_desc
    }
    non_permafrost_du_rings = {
        float(width): railbuf.get_stat_pair(
            donut_stats,
            group="non_permafrost_extreme_du",
            label_order=("Extreme", railbuf.EXTREME_REFERENCE_LABEL),
            width=width,
        )
        for width in width_order_desc
    }
    permafrost_grad_rings = {
        float(width): railbuf.get_stat_pair(
            donut_stats,
            group="permafrost_extreme_grad",
            label_order=("Extreme", railbuf.EXTREME_REFERENCE_LABEL),
            width=width,
        )
        for width in width_order_desc
    }
    non_permafrost_grad_rings = {
        float(width): railbuf.get_stat_pair(
            donut_stats,
            group="non_permafrost_extreme_grad",
            label_order=("Extreme", railbuf.EXTREME_REFERENCE_LABEL),
            width=width,
        )
        for width in width_order_desc
    }

    gs_top = top_spec.subgridspec(1, 2, width_ratios=[1.02, 1.68], wspace=0.08)
    gs_other = gs_top[0, 1].subgridspec(2, 2, wspace=0.10, hspace=0.16)

    ax_share = fig.add_subplot(gs_top[0, 0])
    ax_du_pf = fig.add_subplot(gs_other[0, 0])
    ax_grad_pf = fig.add_subplot(gs_other[0, 1])
    ax_du_npf = fig.add_subplot(gs_other[1, 0])
    ax_grad_npf = fig.add_subplot(gs_other[1, 1])

    railbuf.plot_multi_ring_donut(
        ax_share,
        ring_values_by_width=sample_domain_rings,
        width_order_desc=width_order_desc,
        ring_colors_by_width={
            float(width): [domain_pf_ring_colors[float(width)], domain_npf_ring_colors[float(width)]]
            for width in width_order_desc
        },
        title="Permafrost vs Non-permafrost share",
        center_text="",
        legend_labels=None,
        ring_text_by_width={float(width): f"{railbuf.format_width_label(width)}km buffer" for width in width_order_desc},
        ring_text_angle_deg=138.0,
    )
    share_legend = ax_share.legend(
        handles=[
            Patch(facecolor=railbuf.PERMAFROST_COLOR, edgecolor="none", label="Permafrost"),
            Patch(facecolor=railbuf.NON_PERMAFROST_COLOR, edgecolor="none", label="Non-permafrost"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, -0.10),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    railbuf.apply_bold_legend(share_legend)
    add_subplot_label(ax_share, PANEL_LABELS[0])

    du_pf_records = railbuf.plot_multi_ring_donut(
        ax_du_pf,
        ring_values_by_width=permafrost_du_rings,
        width_order_desc=width_order_desc,
        ring_colors_by_width={
            float(width): [du_ring_colors[float(width)], railbuf.blend_with_white(railbuf.OTHER_COLOR, 0.08)]
            for width in width_order_desc
        },
        title=None,
        center_text="",
        legend_labels=None,
        pct_label_indices=(),
        highlight_first_slice=True,
    )
    railbuf.style_context_donut_axis(
        ax_du_pf,
        background_color=railbuf.PERMAFROST_DONUT_BG,
        domain_label="Permafrost",
        threshold_label=rf"$d_u<{railbuf.PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr",
        column_title="Extreme $d_u$ share",
    )
    railbuf.add_extreme_zoom_callout(
        ax_du_pf,
        ring_records=du_pf_records,
        width_order_desc=width_order_desc,
        background_color=railbuf.PERMAFROST_DONUT_BG,
        title="Zoomed extreme %",
    )

    du_npf_records = railbuf.plot_multi_ring_donut(
        ax_du_npf,
        ring_values_by_width=non_permafrost_du_rings,
        width_order_desc=width_order_desc,
        ring_colors_by_width={
            float(width): [du_ring_colors[float(width)], railbuf.blend_with_white(railbuf.OTHER_COLOR, 0.08)]
            for width in width_order_desc
        },
        title=None,
        center_text="",
        legend_labels=None,
        pct_label_indices=(),
        highlight_first_slice=True,
    )
    railbuf.style_context_donut_axis(
        ax_du_npf,
        background_color=railbuf.NON_PERMAFROST_DONUT_BG,
        domain_label="Non-permafrost",
        threshold_label=rf"$d_u<{railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr",
    )
    railbuf.add_extreme_zoom_callout(
        ax_du_npf,
        ring_records=du_npf_records,
        width_order_desc=width_order_desc,
        background_color=railbuf.NON_PERMAFROST_DONUT_BG,
        title="Zoomed extreme %",
    )

    grad_pf_records = railbuf.plot_multi_ring_donut(
        ax_grad_pf,
        ring_values_by_width=permafrost_grad_rings,
        width_order_desc=width_order_desc,
        ring_colors_by_width={
            float(width): [grad_ring_colors[float(width)], railbuf.blend_with_white(railbuf.OTHER_COLOR, 0.08)]
            for width in width_order_desc
        },
        title=None,
        center_text="",
        legend_labels=None,
        pct_label_indices=(),
        highlight_first_slice=True,
    )
    railbuf.style_context_donut_axis(
        ax_grad_pf,
        background_color=railbuf.PERMAFROST_DONUT_BG,
        domain_label="Permafrost",
        threshold_label=rf"$|\nabla d_u|>{railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km",
        column_title="Extreme $|\\nabla d_u|$ share",
    )
    railbuf.add_extreme_zoom_callout(
        ax_grad_pf,
        ring_records=grad_pf_records,
        width_order_desc=width_order_desc,
        background_color=railbuf.PERMAFROST_DONUT_BG,
        title="Zoomed extreme %",
    )

    grad_npf_records = railbuf.plot_multi_ring_donut(
        ax_grad_npf,
        ring_values_by_width=non_permafrost_grad_rings,
        width_order_desc=width_order_desc,
        ring_colors_by_width={
            float(width): [grad_ring_colors[float(width)], railbuf.blend_with_white(railbuf.OTHER_COLOR, 0.08)]
            for width in width_order_desc
        },
        title=None,
        center_text="",
        legend_labels=None,
        pct_label_indices=(),
        highlight_first_slice=True,
    )
    railbuf.style_context_donut_axis(
        ax_grad_npf,
        background_color=railbuf.NON_PERMAFROST_DONUT_BG,
        domain_label="Non-permafrost",
        threshold_label=rf"$|\nabla d_u|>{railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km",
    )
    railbuf.add_extreme_zoom_callout(
        ax_grad_npf,
        ring_records=grad_npf_records,
        width_order_desc=width_order_desc,
        background_color=railbuf.NON_PERMAFROST_DONUT_BG,
        title="Zoomed extreme %",
    )

    add_subplot_label(ax_du_pf, PANEL_LABELS[1])
    return ax_share, ax_du_pf


def build_figure(
    *,
    donut_samples_by_width: dict[float, pd.DataFrame],
    extreme_map_df: pd.DataFrame,
    railway_context: dict[str, object],
    buffer_width_km: float,
    meteoro_sites: pd.DataFrame,
    permafrost_bg: pd.DataFrame,
    fig_dir: Path,
) -> tuple[Path, Path]:
    crs = railway_context["crs"]
    corridor_half_width_m = 500.0 * float(buffer_width_km)
    corridor_geom = railway_context["network_geom"].buffer(corridor_half_width_m)
    du_extreme = extreme_map_df.loc[extreme_map_df["is_extreme_du"]].copy()
    grad_extreme = extreme_map_df.loc[extreme_map_df["is_extreme_grad"]].copy()
    main_ylim = compute_main_ylim(corridor_geom, crs=crs, meteoro_sites=meteoro_sites)

    fig = plt.figure(figsize=(11.6, 13.4))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.96, 1.18],
        left=0.040,
        right=0.992,
        bottom=0.050,
        top=0.955,
        hspace=0.16,
    )
    add_donut_panels(fig=fig, top_spec=gs[0, 0], donut_samples_by_width=donut_samples_by_width)
    gs_bottom = gs[1, 0].subgridspec(1, 2, wspace=0.10)
    axes = [fig.add_subplot(gs_bottom[0, 0]), fig.add_subplot(gs_bottom[0, 1])]

    outside_note_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="none",
        markerfacecolor=OUTSIDE_NOTE_COLOR,
        markeredgecolor="none",
        markersize=4.4,
        label="Smaller/lighter markers: outside corridor",
    )
    panels = [
        (
            axes[0],
            PANEL_LABELS[2],
            du_extreme,
            f"Extreme $d_u$ near railway\n({railbuf.format_width_label(buffer_width_km)} km corridor highlighted)",
            [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=PF_POINT_COLOR, markeredgecolor="none", markeredgewidth=0.0, markersize=6.5, label=rf"Permafrost: $d_u<{railbuf.PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor=NPF_POINT_COLOR, markeredgecolor="none", markeredgewidth=0.0, markersize=6.0, label=rf"Non-permafrost: $d_u<{railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD:.1f}$ mm/yr"),
            ],
        ),
        (
            axes[1],
            PANEL_LABELS[3],
            grad_extreme,
            f"Extreme $|\\nabla d_u|$ near railway\n({railbuf.format_width_label(buffer_width_km)} km corridor highlighted)",
            [
                Line2D([0], [0], marker="o", color="none", markerfacecolor=PF_POINT_COLOR, markeredgecolor="none", markeredgewidth=0.0, markersize=6.5, label=rf"Permafrost: $|\nabla d_u|>{railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km"),
                Line2D([0], [0], marker="D", color="none", markerfacecolor=NPF_POINT_COLOR, markeredgecolor="none", markeredgewidth=0.0, markersize=6.0, label=rf"Non-permafrost: $|\nabla d_u|>{railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD:.1f}$ mm/yr/km"),
            ],
        ),
    ]

    for ax, panel_label, sub_df, title, point_handles in panels:
        plot_panel_content(
            ax,
            corridor_geom=corridor_geom,
            crs=crs,
            railway_geom=railway_context["network_geom"],
            permafrost_bg=permafrost_bg,
            sub_df=sub_df,
        )
        ax.set_xlim(*MAIN_XLIM)
        ax.set_ylim(*main_ylim)
        add_meteoro_sites(ax, meteoro_sites)
        add_zoom_insets(
            ax,
            corridor_geom=corridor_geom,
            crs=crs,
            railway_geom=railway_context["network_geom"],
            permafrost_bg=permafrost_bg,
            sub_df=sub_df,
            meteoro_sites=meteoro_sites,
        )

        handles = [
            Line2D([0], [0], color=RAILWAY_COLOR, linewidth=1.6, label="Railway"),
            *point_handles,
            outside_note_handle,
        ]
        legend = ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white", edgecolor="0.85")
        railbuf.apply_bold_legend(legend)
        configure_map_axis(ax, title=title)
        add_subplot_label(ax, panel_label)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build railway-buffer summaries for d_u, d_u gradient, and cached susceptibility rasters."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--buffer-width-km", type=float, default=DEFAULT_BUFFER_WIDTH_KM)
    parser.add_argument("--chunksize", type=int, default=railbuf.CHUNKSIZE)
    parser.add_argument("--profile-bin-km", type=float, default=railbuf.PROFILE_BIN_KM_DEFAULT)
    parser.add_argument("--profile-std-scale", type=float, default=railbuf.PROFILE_STD_SCALE)
    parser.add_argument("--susceptibility-meta", type=Path, default=None)
    parser.add_argument("--railway-shp", type=Path, default=None)
    parser.add_argument("--meteoro-shp", type=Path, default=None)
    args = parser.parse_args()

    railbuf.configure_style()

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
    meteoro_shp = (
        args.meteoro_shp.resolve()
        if args.meteoro_shp is not None
        else (base_dir / "human_features" / "qtec_meteoro_station_sites.shp")
    )
    required = [csv_path, railway_shp, meteoro_shp]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(str(p) for p in missing))

    buffer_width_km = float(args.buffer_width_km)
    if not np.isfinite(buffer_width_km) or buffer_width_km <= 0.0:
        raise RuntimeError("buffer_width_km must be a positive finite number.")
    profile_bin_km = float(args.profile_bin_km)
    if not np.isfinite(profile_bin_km) or profile_bin_km <= 0.0:
        raise RuntimeError("profile_bin_km must be a positive finite number.")
    profile_std_scale = float(args.profile_std_scale)
    if not np.isfinite(profile_std_scale) or profile_std_scale < 0.0:
        raise RuntimeError("profile_std_scale must be a finite non-negative number.")

    railway_context = railbuf.load_railway_context(railway_shp)
    profile_sites = load_profile_sites(meteoro_shp, railway_context=railway_context)
    susceptibility_meta_path = resolve_susceptibility_meta_path(
        base_dir=base_dir,
        out_dir=out_dir,
        explicit_path=args.susceptibility_meta,
    )
    susceptibility_sources = load_susceptibility_sources(susceptibility_meta_path)
    requested_widths = tuple(sorted({float(buffer_width_km), *(float(width) for width in DONUT_BUFFER_WIDTHS_KM)}))
    cache_paths = {
        float(width): cache_dir / f"{FIG_BASENAME}_corridor_sample_{str(width).replace('.', 'p')}km.csv.gz"
        for width in requested_widths
    }
    samples_by_width = railbuf.resolve_corridor_samples(
        csv_path,
        cache_paths=cache_paths,
        railway_context=railway_context,
        buffer_widths_km=requested_widths,
        chunksize=int(args.chunksize),
    )
    width_order_desc = tuple(sorted((float(width) for width in requested_widths), reverse=True))
    sample_df = samples_by_width[float(buffer_width_km)].copy()
    extreme_df = build_extreme_table(sample_df)
    samples_with_prob_by_width = attach_susceptibility_columns(
        samples_by_width,
        susceptibility_sources=susceptibility_sources,
    )

    table_path = table_dir / f"{FIG_BASENAME}_extreme_points_{str(buffer_width_km).replace('.', 'p')}km.csv"
    extreme_df.to_csv(table_path, index=False)
    bar_png, bar_pdf, bar_stats_df = build_summary_bar_figure(
        samples_by_width=samples_with_prob_by_width,
        width_order_desc=width_order_desc,
        fig_dir=fig_dir,
    )
    profile_png, profile_pdf, profile_df = build_profile_figure(
        samples_by_width=samples_by_width,
        width_order_desc=width_order_desc,
        bin_length_km=profile_bin_km,
        std_scale=profile_std_scale,
        fig_dir=fig_dir,
        stem=f"{FIG_BASENAME}_profiles",
        profile_sites=profile_sites,
        metric_specs=(
            (
                "d_u",
                rf"Along-railway $\mathbf{{d}}_{{\mathbf{{u}}}}$ profile" + "\n" + rf"($\mathbf{{d}}_{{\mathbf{{u}}}} \leq 0$, {profile_bin_km:.0f} km bins)",
                railbuf.DU_BASE_COLOR,
                "mm/yr",
            ),
            (
                "grad_mag_km",
                rf"Along-railway $|\nabla \mathbf{{d}}_{{\mathbf{{u}}}}|$ profile" + "\n" + rf"($\mathbf{{d}}_{{\mathbf{{u}}}} \leq 0$, {profile_bin_km:.0f} km bins)",
                railbuf.GRAD_BASE_COLOR,
                "mm/yr/km",
            ),
        ),
    )
    susceptibility_profile_png, susceptibility_profile_pdf, susceptibility_profile_df = build_profile_figure(
        samples_by_width=samples_with_prob_by_width,
        width_order_desc=width_order_desc,
        bin_length_km=profile_bin_km,
        std_scale=profile_std_scale,
        fig_dir=fig_dir,
        stem=f"{FIG_BASENAME}_susceptibility_profiles",
        profile_sites=profile_sites,
        metric_specs=(
            (
                "susceptibility_du",
                rf"Along-railway extreme $\mathbf{{d}}_{{\mathbf{{u}}}}$ susceptibility" + "\n" + rf"($\mathbf{{d}}_{{\mathbf{{u}}}} \leq 0$, {profile_bin_km:.0f} km bins)",
                railbuf.DU_BASE_COLOR,
                "Mean prob.",
            ),
            (
                "susceptibility_grad",
                rf"Along-railway extreme $|\nabla \mathbf{{d}}_{{\mathbf{{u}}}}|$ susceptibility" + "\n" + rf"($\mathbf{{d}}_{{\mathbf{{u}}}} \leq 0$, {profile_bin_km:.0f} km bins)",
                railbuf.GRAD_BASE_COLOR,
                "Mean prob.",
            ),
        ),
    )

    bar_stats_path = table_dir / f"{FIG_BASENAME}_buffer_bar_stats.csv"
    profiles_path = table_dir / f"{FIG_BASENAME}_profiles.csv"
    susceptibility_profiles_path = table_dir / f"{FIG_BASENAME}_susceptibility_profiles.csv"
    bar_stats_df.to_csv(bar_stats_path, index=False)
    profile_df.to_csv(profiles_path, index=False)
    susceptibility_profile_df.to_csv(susceptibility_profiles_path, index=False)

    meta_path = cache_dir / f"{FIG_BASENAME}_meta_{str(buffer_width_km).replace('.', 'p')}km.json"
    meta_path.write_text(
        json.dumps(
            {
                "figure_outputs": {
                    "buffer_bars": {"png": str(bar_png), "pdf": str(bar_pdf)},
                    "profiles": {"png": str(profile_png), "pdf": str(profile_pdf)},
                    "susceptibility_profiles": {
                        "png": str(susceptibility_profile_png),
                        "pdf": str(susceptibility_profile_pdf),
                    },
                },
                "buffer_width_km": buffer_width_km,
                "meteoro_shp": str(meteoro_shp),
                "sample_cache": str(cache_paths[buffer_width_km]),
                "sample_caches": {f"{width:.1f}km": str(cache_paths[float(width)]) for width in requested_widths},
                "buffer_widths_km": list(width_order_desc),
                "profile_bin_km": profile_bin_km,
                "profile_std_scale": profile_std_scale,
                "susceptibility_meta": str(susceptibility_meta_path),
                "susceptibility_rasters": {
                    target: str(susceptibility_sources[target]["prob_path"])
                    for target in ("d_u", "grad_mag_km")
                },
                "extreme_points_csv": str(table_path),
                "buffer_bar_stats_csv": str(bar_stats_path),
                "profiles_csv": str(profiles_path),
                "susceptibility_profiles_csv": str(susceptibility_profiles_path),
                "n_extreme_du": int(extreme_df["is_extreme_du"].sum()),
                "n_extreme_grad": int(extreme_df["is_extreme_grad"].sum()),
                "permafrost_extreme_du_threshold_mm_per_yr": railbuf.PERMAFROST_EXTREME_DU_THRESHOLD,
                "non_permafrost_extreme_du_threshold_mm_per_yr": railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD,
                "permafrost_extreme_grad_threshold_mm_per_yr_per_km": railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD,
                "non_permafrost_extreme_grad_threshold_mm_per_yr_per_km": railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD,
            },
            indent=2,
        )
    )

    log_step(f"saved buffer-bar PNG: {bar_png}")
    log_step(f"saved buffer-bar PDF: {bar_pdf}")
    log_step(f"saved profile PNG: {profile_png}")
    log_step(f"saved profile PDF: {profile_pdf}")
    log_step(f"saved susceptibility-profile PNG: {susceptibility_profile_png}")
    log_step(f"saved susceptibility-profile PDF: {susceptibility_profile_pdf}")
    log_step(f"saved extreme-point table: {table_path}")
    log_step(f"saved buffer-bar stats: {bar_stats_path}")
    log_step(f"saved profiles table: {profiles_path}")
    log_step(f"saved susceptibility profiles table: {susceptibility_profiles_path}")
    log_step(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
