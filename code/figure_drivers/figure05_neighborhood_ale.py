#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure05_process_zone_ale_stability.py
# Renamed package path: code/figure_drivers/figure05_neighborhood_ale.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from dataclasses import dataclass

from submission_build_common import (
    FONT,
    SOURCE_CACHE_DIR,
    add_panel_label,
    blend_with_white,
    ensure_style,
    read_joblib_df,
    save_figure,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

import Figure_reorganize_extreme_deformation_susceptibity as figsus
import figure6_susceptibility_stacked as fig6
import _revised_check_npf_extreme_thermal_dependence as npfdiag
import process_zone_definitions as process_zones


FIG_STEM = "Figure05_process_zone_ale_stability"
DU_COLOR = "#1E5BAA"
GRAD_COLOR = "#C1272D"
PF_TINT = "#5A8F63"
NPF_TINT = "#9A6A49"
TARGET_COLORS = {"d_u": DU_COLOR, "grad_mag_km": GRAD_COLOR}
PANEL_TINT_BLEND = 0.90
ALE_BAND_ALPHA = 0.055
ZERO_LINE_DASH = (0, (3.0, 3.0))
ZONE_DASH = (0, (5.0, 2.2))
ZONE_DASHDOT = (0, (5.0, 1.8, 1.2, 1.8))
ZONE_DOT = (0, (1.2, 1.8))
ZONE_DASHDOTDOT = (0, (6.0, 1.7, 1.2, 1.7, 1.2, 1.7))
ALE_BINS = 16
ALE_MAX_ROWS = 18000
MIN_MODEL_ROWS = 1000
MIN_RETAINED_FRAC = 0.03
MAX_ZONE_ROWS_TO_FIT = 35000


@dataclass(frozen=True)
class ZoneSpec:
    tag: str
    label: str
    domain: str
    blend: float
    linewidth: float
    linestyle: object


NPF_ZONES = [
    ZoneSpec("npf_all", "All NPF", "npf", 0.00, 2.8, "solid"),
    ZoneSpec("npf_interior", "Interior", "npf", 0.08, 2.2, ZONE_DASH),
    ZoneSpec("npf_transition", "Transition", "npf", 0.16, 2.1, ZONE_DASHDOT),
]
PF_ZONES = [
    ZoneSpec("pf_all", "All PF", "pf", 0.00, 2.8, "solid"),
    ZoneSpec("pf_transition_impacted", "Transition + lake/slump", "pf", 0.08, 2.1, ZONE_DASH),
    ZoneSpec("pf_transition_background", "Transition background", "pf", 0.15, 2.0, ZONE_DASHDOT),
    ZoneSpec("pf_interior_impacted", "Interior + lake/slump", "pf", 0.20, 2.0, ZONE_DOT),
    ZoneSpec("pf_interior_background", "Interior background", "pf", 0.28, 2.0, ZONE_DASHDOTDOT),
]


def load_target_sample(target: str) -> pd.DataFrame:
    name = (
        "_revised_check_pf_extreme_env_dependence_du_tlrts_distance_sample.joblib.gz"
        if target == "d_u"
        else "_revised_check_pf_extreme_env_dependence_grad_tlrts_distance_sample.joblib.gz"
    )
    return read_joblib_df(SOURCE_CACHE_DIR / name)


def build_feature_config() -> tuple[list[str], list[str], list[str], list[str]]:
    return figsus.build_feature_sets(exclude_magt=False)


def zone_mask(df: pd.DataFrame, zone: ZoneSpec, *, target: str) -> np.ndarray:
    dom = df["domain"].astype(str).to_numpy()
    mask = dom == zone.domain
    if zone.tag == "npf_all":
        return mask
    if zone.tag == "pf_all":
        return mask
    process_masks = process_zones.zone_masks(df, target=target)
    if zone.tag in process_masks:
        return process_masks[zone.tag]
    raise ValueError(zone.tag)


def zone_extreme_mask(df: pd.DataFrame, *, target: str, domain: str) -> np.ndarray:
    if target == "d_u":
        threshold = -16.5 if domain == "pf" else -7.4
        return pd.to_numeric(df["d_u"], errors="coerce").to_numpy(dtype=float) <= threshold
    threshold = 30.5 if domain == "pf" else 17.0
    return pd.to_numeric(df["grad_mag_km"], errors="coerce").to_numpy(dtype=float) >= threshold


def downsample_zone_subset(subset: pd.DataFrame, *, target: str, domain: str) -> pd.DataFrame:
    if len(subset) <= MAX_ZONE_ROWS_TO_FIT:
        return subset

    labels = zone_extreme_mask(subset, target=target, domain=domain)
    pos_idx = np.flatnonzero(labels)
    neg_idx = np.flatnonzero(~labels)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        keep = np.linspace(0, len(subset) - 1, num=MAX_ZONE_ROWS_TO_FIT, dtype=int)
        return subset.iloc[np.unique(keep)].reset_index(drop=True)

    rng = np.random.default_rng(figsus.SEED)
    pos_target = int(round(MAX_ZONE_ROWS_TO_FIT * len(pos_idx) / len(subset)))
    pos_target = min(len(pos_idx), max(min(500, len(pos_idx)), pos_target))
    neg_target = min(len(neg_idx), MAX_ZONE_ROWS_TO_FIT - pos_target)
    if neg_target <= 0:
        neg_target = min(len(neg_idx), max(1, MAX_ZONE_ROWS_TO_FIT - pos_target))
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=pos_target, replace=False),
            rng.choice(neg_idx, size=neg_target, replace=False),
        ]
    )
    chosen.sort()
    return subset.iloc[chosen].reset_index(drop=True)


def fit_zone_model(sample_df: pd.DataFrame, *, target: str, zone: ZoneSpec, combined_features: list[str]) -> tuple[dict[str, object] | None, pd.DataFrame]:
    subset = sample_df.loc[zone_mask(sample_df, zone, target=target)].copy().reset_index(drop=True)
    base_n = int(sample_df["domain"].eq(zone.domain).sum())
    if len(subset) < MIN_MODEL_ROWS or (base_n > 0 and len(subset) / base_n < MIN_RETAINED_FRAC):
        return None, subset
    subset = downsample_zone_subset(subset, target=target, domain=zone.domain)
    work = subset.loc[subset["domain"].eq(zone.domain)].copy().reset_index(drop=True)
    work["label"] = figsus.build_domain_label(work, target=target, domain=zone.domain)
    work["block_id"] = fig6.make_spatial_block_id(work, block_size_m=100_000.0)
    work = work.loc[work["label"].notna() & work["block_id"].notna()].reset_index(drop=True)
    if len(work) < MIN_MODEL_ROWS:
        return None, work

    feature_groups = {
        "raw": [feature for feature in combined_features if figsus.feature_origin(feature) == "raw"],
        "contrast": [feature for feature in combined_features if figsus.feature_origin(feature) == "contrast"],
    }
    missing_df = fig6.audit_feature_missingness(work, feature_groups)
    dropped = sorted(
        missing_df.loc[missing_df["missing_frac"] > 0.30, "feature"].drop_duplicates().tolist()
    )
    model_features = [feature for feature in combined_features if feature not in dropped]
    if not model_features:
        return None, work

    X_full = fig6.make_model_frame(work, model_features)
    y_full = work["label"].to_numpy(dtype=int)
    pos_rate = max(float(np.mean(y_full)), 1e-6)
    pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
    full_stack = figsus.build_domain_models(
        pos_weight=pos_weight,
        stack_cv=3,
        n_jobs=1,
    )["ET"]
    full_stack.fit(X_full, y_full)
    return {"full_stack": full_stack, "feature_names": model_features}, work


def ale_curve(result: dict[str, object], subset: pd.DataFrame, feature: str) -> pd.DataFrame:
    X_ref = npfdiag.sample_frame(fig6.make_model_frame(subset, list(result["feature_names"])), max_rows=ALE_MAX_ROWS)
    curve, _ = npfdiag.compute_ale_curve(
        X_ref=X_ref,
        feature=feature,
        predict_fn=lambda X, model=result["full_stack"]: npfdiag.predict_proba_batched(model, X),
        n_bins=ALE_BINS,
    )
    return curve


def fit_zone_results(target: str, zones: list[ZoneSpec]) -> dict[str, dict[str, object]]:
    _raw_features, _contrast_features, combined_features, _transition_base_vars = build_feature_config()
    sample = load_target_sample(target)
    fitted: dict[str, dict[str, object]] = {}
    for zone in zones:
        result, subset = fit_zone_model(sample, target=target, zone=zone, combined_features=combined_features)
        if result is None or subset.empty:
            continue
        fitted[zone.tag] = {"result": result, "subset": subset}
    return fitted


def build_curves(feature: str, zones: list[ZoneSpec], fitted: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zone in zones:
        payload = fitted.get(zone.tag)
        if payload is None:
            continue
        curve = ale_curve(payload["result"], payload["subset"], feature)
        if curve.empty:
            continue
        rows.append({"zone": zone, "curve": curve})
    return rows


def panel_feature_label(feature: str) -> str:
    if feature == "twi":
        return "TWI"
    if feature == "ndvi":
        return "NDVI"
    if feature == "magt":
        return "MAGT (°C)"
    return npfdiag.feature_axis_label(feature)


def zone_color(zone: ZoneSpec, target_color: str) -> tuple[float, float, float]:
    return blend_with_white(target_color, zone.blend)


def panel_tint(domain_color: str) -> tuple[float, float, float]:
    return blend_with_white(domain_color, PANEL_TINT_BLEND)


def ale_tick_label(value: float, _pos: object) -> str:
    if abs(float(value)) < 0.005:
        value = 0.0
    return f"{float(value):.2f}"


def style_ale_axis(ax) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(ale_tick_label))
    ax.tick_params(axis="both", labelsize=FONT["tick"] + 0.5, length=3.2, width=0.7)
    ax.grid(axis="y", color="0.91", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_target_panel(
    ax,
    *,
    title: str,
    feature: str,
    zones: list[ZoneSpec],
    tint: tuple[float, float, float],
    target_color: str,
    fitted: dict[str, dict[str, object]],
    show_ylabel: bool,
    domain_color: str,
    stability_hint: bool,
) -> None:
    ax.set_facecolor(tint)
    ax.axhline(0.0, color="0.45", linewidth=0.7, linestyle=ZERO_LINE_DASH, zorder=1)
    y_bounds: list[float] = []
    curves = build_curves(feature, zones, fitted)
    for item in curves:
        zone = item["zone"]
        curve = item["curve"]
        color = zone_color(zone, target_color)
        ax.fill_between(
            curve["x_plot"],
            curve["ylo"],
            curve["yhi"],
            color=color,
            alpha=ALE_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
    line_items = sorted(curves, key=lambda item: item["zone"].tag.endswith("_all"))
    for item in line_items:
        zone = item["zone"]
        curve = item["curve"]
        color = zone_color(zone, target_color)
        ax.plot(
            curve["x_plot"],
            curve["y"],
            color=color,
            linewidth=zone.linewidth,
            linestyle=zone.linestyle,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=4 if zone.tag.endswith("_all") else 3,
        )
        y_bounds.extend(
            [
                float(np.nanmin(curve["ylo"])),
                float(np.nanmax(curve["yhi"])),
            ]
        )
    if title:
        ax.set_title(title, pad=6, color=domain_color, fontweight="semibold")
    ax.set_xlabel(panel_feature_label(feature))
    ax.set_ylabel("Centered ALE" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=True)
    if y_bounds:
        lo = min(y_bounds)
        hi = max(y_bounds)
        pad = max(0.03, 0.15 * (hi - lo if hi > lo else 0.2))
        ax.set_ylim(lo - pad, hi + pad)
    if stability_hint and len(curves) >= 2:
        # Place the "zone curves converge" hint where the across-zone spread is smallest.
        x_common = curves[0]["curve"]["x_plot"]
        stacked = np.array([np.interp(x_common, item["curve"]["x_plot"], item["curve"]["y"]) for item in curves])
        spread = np.nanstd(stacked, axis=0)
        if spread.size > 0 and np.any(np.isfinite(spread)):
            best = int(np.nanargmin(spread))
            x_best = float(np.asarray(x_common)[best])
            y_best = float(np.nanmedian(stacked[:, best]))
            ax.annotate(
                "zones converge",
                xy=(x_best, y_best),
                xytext=(10, 22),
                textcoords="offset points",
                fontsize=FONT["annotation"] - 0.6,
                color="0.25",
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color="0.55", linewidth=0.5),
                zorder=6,
            )
    style_ale_axis(ax)


def add_row_header(fig, *, y: float, text: str, color: str) -> None:
    fig.text(
        0.018,
        y,
        text,
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT["axis"] + 0.8,
        color=color,
        fontweight="bold",
    )


def color_legend_text(legend, color: str) -> None:
    legend.get_title().set_color(color)
    for text in legend.get_texts():
        text.set_color(color)


def main() -> None:
    ensure_style()
    npf_fits = {target: fit_zone_results(target, NPF_ZONES) for target in ["d_u", "grad_mag_km"]}
    pf_fits = {target: fit_zone_results(target, PF_ZONES) for target in ["d_u", "grad_mag_km"]}

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.4), constrained_layout=False)
    plt.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.24, wspace=0.28, hspace=0.42)

    panel_specs = [
        ("d_u", 0, 0, "NPF · MAGT", "magt", NPF_ZONES, NPF_TINT, npf_fits["d_u"], False),
        ("d_u", 0, 1, "PF · TWI", "twi", PF_ZONES, PF_TINT, pf_fits["d_u"], False),
        ("d_u", 0, 2, "PF · NDVI", "ndvi", PF_ZONES, PF_TINT, pf_fits["d_u"], False),
        ("grad_mag_km", 1, 0, "NPF · MAGT", "magt", NPF_ZONES, NPF_TINT, npf_fits["grad_mag_km"], False),
        ("grad_mag_km", 1, 1, "PF · TWI", "twi", PF_ZONES, PF_TINT, pf_fits["grad_mag_km"], False),
        ("grad_mag_km", 1, 2, "PF · NDVI", "ndvi", PF_ZONES, PF_TINT, pf_fits["grad_mag_km"], False),
    ]
    for target, row, col, title, feature, zones, domain_color, fitted, stability_hint in panel_specs:
        plot_single_target_panel(
            axes[row, col],
            title=title,
            feature=feature,
            zones=zones,
            tint=panel_tint(domain_color),
            target_color=TARGET_COLORS[target],
            fitted=fitted,
            show_ylabel=(col == 0),
            domain_color=domain_color,
            stability_hint=stability_hint,
        )

    for ax, label in zip(axes.flat, ["A", "B", "C", "D", "E", "F"]):
        add_panel_label(ax, label, x=-0.10, y=1.04)

    add_row_header(fig, y=0.750, text=r"Extreme $d_u$ susceptibility", color=DU_COLOR)
    add_row_header(fig, y=0.360, text=r"Extreme $|\nabla d_u|$ susceptibility", color=GRAD_COLOR)

    # Split legend: NPF zone key below panel D, PF zone key below panels E-F.
    neutral = "#404040"
    npf_handles = [
        Line2D([0], [0], color=neutral, linewidth=zone.linewidth,
               linestyle=zone.linestyle, label=zone.label)
        for zone in NPF_ZONES
    ]
    pf_handles = [
        Line2D([0], [0], color=neutral, linewidth=zone.linewidth,
               linestyle=zone.linestyle, label=zone.label)
        for zone in PF_ZONES
    ]
    legend_npf = fig.legend(
        npf_handles,
        [h.get_label() for h in npf_handles],
        title="NPF zones · MAGT panels",
        loc="upper center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.255, 0.160),
        handlelength=3.2,
        fontsize=FONT["axis"] + 0.5,
        title_fontsize=FONT["axis"] + 0.5,
    )
    legend_pf = fig.legend(
        pf_handles,
        [h.get_label() for h in pf_handles],
        title="PF zones · TWI and NDVI panels",
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.665, 0.160),
        handlelength=3.2,
        columnspacing=1.5,
        fontsize=FONT["axis"] + 0.5,
        title_fontsize=FONT["axis"] + 0.5,
    )
    color_legend_text(legend_npf, NPF_TINT)
    color_legend_text(legend_pf, PF_TINT)
    fig.add_artist(legend_npf)
    fig.add_artist(legend_pf)

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
