#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure04_proximity_decomposition.py
# Renamed package path: code/figure_drivers/figure04_distance_decomposition.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from submission_build_common import (
    FONT,
    SOURCE_CACHE_DIR,
    SOURCE_TABLE_DIR,
    TABLE_DIR,
    add_panel_label,
    blend_with_white,
    ensure_style,
    read_joblib_df,
    save_figure,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import process_zone_definitions as zones


FIG_STEM = "Figure04_proximity_decomposition"
PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
DU_COLOR = "#1E5BAA"
GRAD_COLOR = "#C1272D"
BOUNDARY_COLOR = "#4A4A4A"
DU_YLIM = (-12.6, 0.0)
GRAD_YLIM = (2.8, 15.1)
DU_TICKS = [-12.0, -9.0, -6.0, -3.0, 0.0]
GRAD_TICKS = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
SEPARATOR_DASH = (0, (2.0, 2.0))
FIT_DASH = (0, (4.0, 2.2))


def load_profile_samples() -> dict[str, pd.DataFrame]:
    return {
        "du_boundary": read_joblib_df(SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_du_enriched_sample.joblib.gz"),
        "grad_boundary": read_joblib_df(SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_grad_enriched_sample.joblib.gz"),
        "du_tlrts": read_joblib_df(SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_du_tlrts_distance_sample.joblib.gz"),
        "grad_tlrts": read_joblib_df(SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_grad_tlrts_distance_sample.joblib.gz"),
        "zone_sample": pd.read_csv(SOURCE_TABLE_DIR / "_check_abrupt_thaw_degradation_hazard_v2_sample_with_distance.csv.gz"),
    }


def binned_profile(x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_count: int = 20) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return pd.DataFrame(columns=["x", "mean", "std", "se", "n"])
    x_ok = x[ok]
    y_ok = y[ok]
    ids = np.digitize(x_ok, bins) - 1
    rows: list[dict[str, float]] = []
    for idx in range(len(bins) - 1):
        mask = ids == idx
        if int(np.count_nonzero(mask)) < min_count:
            continue
        vals = y_ok[mask]
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "x": 0.5 * (float(bins[idx]) + float(bins[idx + 1])),
                "mean": float(np.mean(vals)),
                "std": std,
                "se": std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def binned_signed_boundary_profile(df: pd.DataFrame, value_col: str, bins: np.ndarray) -> pd.DataFrame:
    d_b = zones.numeric_array(df, zones.BOUNDARY_DISTANCE_COL)
    signed = d_b.copy()
    signed[df["domain"].eq("npf").to_numpy(dtype=bool)] *= -1.0
    values = zones.numeric_array(df, value_col)
    return binned_profile(signed, values, bins)


def binned_boundary_outward_profile(
    df: pd.DataFrame,
    value_col: str,
    *,
    domain: str,
    bins: np.ndarray,
    abrupt_min_km: float | None = None,
) -> pd.DataFrame:
    work = df.loc[df["domain"].eq(domain)].copy()
    d_b = zones.numeric_array(work, zones.BOUNDARY_DISTANCE_COL)
    values = zones.numeric_array(work, value_col)
    mask = np.isfinite(d_b) & (d_b >= 0.0)
    if abrupt_min_km is not None and zones.ABRUPT_DISTANCE_COL in work:
        d_a = zones.numeric_array(work, zones.ABRUPT_DISTANCE_COL)
        mask &= np.isfinite(d_a) & (d_a >= float(abrupt_min_km))
    return binned_profile(d_b[mask], values[mask], bins)


def binned_tlrts_profile(
    df: pd.DataFrame,
    value_col: str,
    bins: np.ndarray,
    *,
    boundary_side: str = "all",
) -> pd.DataFrame:
    work = df.loc[df["domain"].eq("pf")].copy()
    d_a = zones.numeric_array(work, zones.ABRUPT_DISTANCE_COL)
    d_b = zones.numeric_array(work, zones.BOUNDARY_DISTANCE_COL)
    values = zones.numeric_array(work, value_col)
    boundary_km = zones.boundary_transition_km(value_col, "pf")
    mask = np.isfinite(d_a) & (d_a >= 0.0) & (d_a <= float(bins[-1]))
    if boundary_side == "interior":
        mask &= np.isfinite(d_b) & (d_b >= boundary_km)
    elif boundary_side in {"transition", "frontal"}:
        mask &= np.isfinite(d_b) & (d_b < boundary_km)
    return binned_profile(d_a[mask], values[mask], bins)


def hinge_fit(profile: pd.DataFrame, *, kmin: float, kmax: float, label: str) -> dict[str, float | str]:
    work = profile.loc[np.isfinite(profile["x"]) & np.isfinite(profile["mean"])].copy()
    x = work["x"].to_numpy(dtype=float)
    y = work["mean"].to_numpy(dtype=float)
    out: dict[str, float | str] = {
        "fit_label": label,
        "n_bins": int(len(work)),
        "fitted_elbow_km": np.nan,
        "slope_before": np.nan,
        "slope_after": np.nan,
        "bic": np.nan,
    }
    if len(x) < 6:
        return out
    candidates = x[(x > float(kmin)) & (x < float(kmax))]
    if len(candidates) == 0:
        return out
    best: tuple[float, float, np.ndarray] | None = None
    for knot in candidates:
        design = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - knot)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        sse = float(np.sum(residual * residual))
        bic = len(x) * np.log(max(sse / len(x), 1e-12)) + 3.0 * np.log(len(x))
        if best is None or bic < best[0]:
            best = (bic, float(knot), beta)
    if best is None:
        return out
    bic, knot, beta = best
    out.update(
        {
            "fitted_elbow_km": knot,
            "slope_before": float(beta[1]),
            "slope_after": float(beta[1] + beta[2]),
            "bic": bic,
        }
    )
    return out


def build_threshold_fit_table(
    *profiles: pd.DataFrame,
) -> pd.DataFrame:
    _ = profiles
    rows: list[pd.DataFrame] = []
    db_path = TABLE_DIR / "_dB_binary_proximity_check_piecewise_fits.csv"
    da_path = TABLE_DIR / "_dA_binary_proximity_check_piecewise_fits.csv"
    if db_path.exists():
        db = pd.read_csv(db_path)
        db = db.rename(columns={"domain": "profile_group"})
        db["distance_axis"] = "d_B"
        db["threshold_role"] = "domain transition threshold"
        rows.append(db)
    if da_path.exists():
        da = pd.read_csv(da_path)
        da = da.rename(columns={"subset": "profile_group"})
        da["distance_axis"] = "d_A"
        da["threshold_role"] = "lake/slump neighborhood-background threshold"
        rows.append(da)
    if rows:
        return pd.concat(rows, ignore_index=True, sort=False)

    fallback: list[dict[str, object]] = []
    for target in zones.TARGETS:
        for domain in ("npf", "pf"):
            fallback.append(
                {
                    "target": target,
                    "distance_axis": "d_B",
                    "profile_group": domain,
                    "piecewise_break_km": zones.boundary_transition_km(target, domain),
                    "threshold_role": "domain transition threshold",
                    "fit_status": "constant_fallback",
                }
            )
        for group in ("pf_all", "pf_transition", "pf_interior"):
            fallback.append(
                {
                    "target": target,
                    "distance_axis": "d_A",
                    "profile_group": group,
                    "piecewise_break_km": zones.abrupt_impact_km(target, group),
                    "threshold_role": "lake/slump neighborhood-background threshold",
                    "fit_status": "constant_fallback",
                }
            )
    return pd.DataFrame(fallback)


def style_profile_axes(ax, ax2, *, show_left: bool = True, show_right: bool = True) -> None:
    ax.set_ylim(*DU_YLIM)
    ax.set_yticks(DU_TICKS)
    ax2.set_ylim(*GRAD_YLIM)
    ax2.set_yticks(GRAD_TICKS)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.patch.set_visible(False)
    ax.spines["left"].set_color(DU_COLOR)
    ax.spines["left"].set_linewidth(1.0)
    ax2.spines["right"].set_color(GRAD_COLOR)
    ax2.spines["right"].set_linewidth(1.0)
    ax.spines["left"].set_visible(show_left)
    ax2.spines["right"].set_visible(show_right)
    if show_left:
        ax.set_ylabel(r"Mean $d_u$ (mm yr$^{-1}$)", color=DU_COLOR)
        ax.tick_params(axis="y", left=True, labelleft=True, colors=DU_COLOR)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    if show_right:
        ax2.set_ylabel(r"Mean $|\nabla d_u|$ (mm yr$^{-1}$ km$^{-1}$)", color=GRAD_COLOR, labelpad=8)
        ax2.tick_params(axis="y", right=True, labelright=True, colors=GRAD_COLOR)
    else:
        ax2.tick_params(axis="y", right=False, labelright=False)
        ax2.spines["right"].set_visible(False)


def add_vertical_separator(ax, x: float, *, color: str = "0.45", linewidth: float = 0.75, zorder: float = 2.0) -> None:
    ax.axvline(x, color=color, linewidth=linewidth, linestyle=SEPARATOR_DASH, alpha=0.9, zorder=zorder)


def plot_boundary_panel(ax, du_prof: pd.DataFrame, grad_prof: pd.DataFrame):
    xlim = (-20.0, 20.0)
    ax.axvspan(xlim[0], 0.0, color=blend_with_white(NPF_COLOR, 0.84), zorder=0)
    ax.axvspan(0.0, xlim[1], color=blend_with_white(PF_COLOR, 0.84), zorder=0)

    du_band = 0.3 * du_prof["std"].to_numpy(dtype=float)
    ax.fill_between(
        du_prof["x"],
        du_prof["mean"] - du_band,
        du_prof["mean"] + du_band,
        color=blend_with_white(DU_COLOR, 0.52),
        alpha=0.34,
        linewidth=0,
        zorder=1.6,
    )
    du_line, = ax.plot(du_prof["x"], du_prof["mean"], color=DU_COLOR, linewidth=2.0, label=r"$d_u$", zorder=3)
    ax2 = ax.twinx()
    grad_band = 0.3 * grad_prof["std"].to_numpy(dtype=float)
    ax2.fill_between(
        grad_prof["x"],
        grad_prof["mean"] - grad_band,
        grad_prof["mean"] + grad_band,
        color=blend_with_white(GRAD_COLOR, 0.54),
        alpha=0.28,
        linewidth=0,
        zorder=1.5,
    )
    grad_line, = ax2.plot(grad_prof["x"], grad_prof["mean"], color=GRAD_COLOR, linewidth=1.9, label=r"$|\nabla d_u|$", zorder=3)

    boundary_lines = [
        (-zones.boundary_transition_km("d_u", "npf"), DU_COLOR, FIT_DASH, r"$d_u$ NPF"),
        (zones.boundary_transition_km("d_u", "pf"), DU_COLOR, FIT_DASH, r"$d_u$ PF"),
        (-zones.boundary_transition_km("grad_mag_km", "npf"), GRAD_COLOR, SEPARATOR_DASH, r"$|\nabla d_u|$ NPF"),
        (zones.boundary_transition_km("grad_mag_km", "pf"), GRAD_COLOR, SEPARATOR_DASH, r"$|\nabla d_u|$ PF"),
    ]
    for xpos, color, linestyle, _label in boundary_lines:
        ax.axvline(xpos, color=color, linewidth=0.85, linestyle=linestyle, alpha=0.95, zorder=3)
    ax.axvline(0.0, color=BOUNDARY_COLOR, linewidth=0.95, linestyle="--", zorder=3)

    text_zorder = 20
    ax.text(-12.5, -0.55, "NPF", color=NPF_COLOR, fontsize=FONT["annotation"], ha="center", va="center", zorder=text_zorder)
    ax.text(12.5, -0.55, "PF", color=PF_COLOR, fontsize=FONT["annotation"], ha="center", va="center", zorder=text_zorder)
    ax.text(0.0, -11.9, "mapped boundary", color=BOUNDARY_COLOR, fontsize=FONT["annotation"], ha="center", va="bottom", zorder=text_zorder)

    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Signed distance to PF/NPF boundary, $d_B$ (km)" "\n" "Negative values indicate distance into the NPF domain.")
    ax.set_title("Boundary-distance deformation profiles", pad=6)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    ax.legend(
        handles=[
            du_line,
            grad_line,
            Line2D([0], [0], color=DU_COLOR, linewidth=0.9, linestyle=FIT_DASH, label=r"$d_u$ fitted $d_B$ break"),
            Line2D([0], [0], color=GRAD_COLOR, linewidth=0.9, linestyle=SEPARATOR_DASH, label=r"$|\nabla d_u|$ fitted $d_B$ break"),
        ],
        loc="center left",
        frameon=False,
        ncol=1,
        fontsize=FONT["annotation"],
        handlelength=2.0,
    )
    style_profile_axes(ax, ax2, show_left=True, show_right=False)
    return ax2


def plot_tlrts_panel(
    ax,
    du_interior: pd.DataFrame,
    grad_interior: pd.DataFrame,
    du_frontal: pd.DataFrame,
    grad_frontal: pd.DataFrame,
):
    ax.axvspan(0.0, 20.0, color=blend_with_white(PF_COLOR, 0.88), zorder=0)

    for profile in (du_interior, du_frontal):
        band = 0.2 * profile["std"].to_numpy(dtype=float)
        ax.fill_between(
            profile["x"],
            profile["mean"] - band,
            profile["mean"] + band,
            color=blend_with_white(DU_COLOR, 0.52),
            alpha=0.26,
            linewidth=0,
            zorder=1.5,
        )
    du_db = zones.boundary_transition_km("d_u", "pf")
    grad_db = zones.boundary_transition_km("grad_mag_km", "pf")
    du_int, = ax.plot(du_interior["x"], du_interior["mean"], color=DU_COLOR, linewidth=2.0, label=rf"$d_u$, $d_B\geq{du_db:g}$", zorder=3)
    du_front, = ax.plot(du_frontal["x"], du_frontal["mean"], color=DU_COLOR, linewidth=1.6, linestyle=FIT_DASH, label=rf"$d_u$, $d_B<{du_db:g}$", zorder=3)
    ax2 = ax.twinx()
    for profile in (grad_interior, grad_frontal):
        band = 0.2 * profile["std"].to_numpy(dtype=float)
        ax2.fill_between(
            profile["x"],
            profile["mean"] - band,
            profile["mean"] + band,
            color=blend_with_white(GRAD_COLOR, 0.54),
            alpha=0.21,
            linewidth=0,
            zorder=1.4,
        )
    grad_int, = ax2.plot(grad_interior["x"], grad_interior["mean"], color=GRAD_COLOR, linewidth=1.9, label=rf"$|\nabla d_u|$, $d_B\geq{grad_db:g}$", zorder=3)
    grad_front, = ax2.plot(grad_frontal["x"], grad_frontal["mean"], color=GRAD_COLOR, linewidth=1.55, linestyle=FIT_DASH, label=rf"$|\nabla d_u|$, $d_B<{grad_db:g}$", zorder=3)

    threshold_lines = [
        (zones.abrupt_impact_km("d_u", "pf_interior"), DU_COLOR, "solid", r"$d_u$ int."),
        (zones.abrupt_impact_km("d_u", "pf_transition"), DU_COLOR, FIT_DASH, r"$d_u$ trans."),
        (zones.abrupt_impact_km("grad_mag_km", "pf_interior"), GRAD_COLOR, "solid", r"$|\nabla d_u|$ int."),
        (zones.abrupt_impact_km("grad_mag_km", "pf_transition"), GRAD_COLOR, FIT_DASH, r"$|\nabla d_u|$ trans."),
    ]
    for xpos, color, linestyle, label in threshold_lines:
        ax.axvline(xpos, color=color, linewidth=0.85, linestyle=linestyle, alpha=0.9, zorder=3)
        ax.text(
            xpos,
            0.02,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=FONT["annotation"] - 1.2,
            color=color,
            zorder=20,
        )

    ax.set_xlim(20.0, 0.0)
    ax.set_xlabel(r"Distance to nearest thermokarst lake or thaw slump, $d_A$ (km)")
    ax.set_title("Mapped lake/slump distance profiles", pad=6)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    ax.legend(
        handles=[du_int, du_front, grad_int, grad_front],
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=FONT["annotation"] - 0.5,
        handlelength=2.2,
        columnspacing=0.9,
    )
    style_profile_axes(ax, ax2, show_left=False, show_right=True)
    return ax2


def summary_lookup(summary: pd.DataFrame, zone_key: str, field: str) -> float:
    row = summary.loc[summary["zone_key"].eq(zone_key)]
    if row.empty:
        return np.nan
    return float(pd.to_numeric(row[field], errors="coerce").iloc[0])


def text_color_for_face(face: tuple[float, float, float, float]) -> str:
    rgb = np.asarray(face[:3], dtype=float)
    luminance = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    return "white" if luminance < 0.48 else "0.12"


def draw_stat_cell(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    density: float,
    share: float,
    cmap,
    norm,
    label: str = "",
) -> None:
    face = cmap(norm(density if np.isfinite(density) else 0.0))
    rect = Rectangle((x, y), w, h, facecolor=face, edgecolor="white", linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    color = text_color_for_face(face)
    d_text = "NA" if not np.isfinite(density) else f"{density * 100:.1f}%"
    s_text = "NA" if not np.isfinite(share) else f"{share * 100:.1f}%"
    ax.text(x + w / 2.0, y + h * 0.58, d_text, ha="center", va="center", fontsize=FONT["annotation"] + 0.2, color=color, fontweight="semibold", zorder=3)
    ax.text(x + w / 2.0, y + h * 0.34, s_text, ha="center", va="center", fontsize=FONT["annotation"] - 0.35, color=color, zorder=3)
    if label:
        ax.text(x + w / 2.0, y + h * 0.10, label, ha="center", va="bottom", fontsize=FONT["annotation"] - 1.0, color=color, zorder=3)


def plot_decomposition_panel(ax, summary: pd.DataFrame, *, target: str, color: str, title: str) -> None:
    ax.set_axis_off()
    thresholds = zones.threshold_values(target)
    vals = summary[["extreme_density"]].to_numpy(dtype=float).ravel()
    vmax = max(0.01, float(np.nanmax(vals)) * 1.05 if np.isfinite(vals).any() else 0.1)
    cmap = LinearSegmentedColormap.from_list(f"{target}_decomp", [(1, 1, 1), blend_with_white(color, 0.30), color])
    norm = Normalize(vmin=0.0, vmax=vmax)

    cell_w = 1.0
    cell_h = 0.72
    gap = 0.08
    npf_x = 0.25
    pf_x = 2.55
    top_y = 1.05
    bottom_y = 0.25

    ax.text(npf_x + cell_w / 2, 2.05, "NPF", ha="center", va="bottom", color=NPF_COLOR, fontsize=FONT["axis"], fontweight="semibold")
    ax.text(pf_x + cell_w + gap / 2, 2.05, "PF neighborhood table", ha="center", va="bottom", color=PF_COLOR, fontsize=FONT["axis"], fontweight="semibold")
    ax.text(0.0, 2.46, title, ha="left", va="bottom", fontsize=FONT["title"], color=color, fontweight="semibold")
    ax.text(
        1.50,
        2.52,
        r"$E_z$: zone extremes; $N_z$: zone pixels",
        ha="left",
        va="bottom",
        fontsize=FONT["annotation"] - 0.8,
        color="0.38",
    )
    ax.text(
        1.50,
        2.34,
        r"$E_{\mathit{domain}}$: domain extremes",
        ha="left",
        va="bottom",
        fontsize=FONT["annotation"] - 0.8,
        color="0.38",
    )
    ax.text(5.0, 2.52, r"density $=\boldsymbol{E}_z/\boldsymbol{N}_z$", ha="right", va="bottom", fontsize=FONT["annotation"], color="0.35", fontweight="semibold")
    ax.text(5.0, 2.34, r"share $=E_z/E_{\mathit{domain}}$", ha="right", va="bottom", fontsize=FONT["annotation"], color="0.35")
    ax.plot([1.74, 1.74], [0.12, 2.15], color="0.78", linewidth=0.85, zorder=1, clip_on=False)

    npf_specs = [
        ("npf_transition", top_y, rf"$d_B<{thresholds['npf_boundary_transition_km']:g}$"),
        ("npf_interior", bottom_y, rf"$d_B\geq{thresholds['npf_boundary_transition_km']:g}$"),
    ]
    for zone_key, y, label in npf_specs:
        draw_stat_cell(
            ax,
            x=npf_x,
            y=y,
            w=cell_w,
            h=cell_h,
            density=summary_lookup(summary, zone_key, "extreme_density"),
            share=summary_lookup(summary, zone_key, "domain_extreme_share"),
            cmap=cmap,
            norm=norm,
            label=label,
        )

    ax.text(npf_x - 0.08, top_y + cell_h / 2, "Transition", ha="right", va="center", fontsize=FONT["annotation"], color=NPF_COLOR)
    ax.text(npf_x - 0.08, bottom_y + cell_h / 2, "Interior", ha="right", va="center", fontsize=FONT["annotation"], color=NPF_COLOR)

    col_labels = [
        "Lake/slump\nneighborhood",
        "Background",
    ]
    for col, label in enumerate(col_labels):
        ax.text(pf_x + col * (cell_w + gap) + cell_w / 2, bottom_y - 0.13, label, ha="center", va="top", fontsize=FONT["annotation"], color="0.25")
    row_specs = [
        (0, top_y, "Transition", thresholds["pf_boundary_transition_km"], thresholds["pf_transition_abrupt_impact_km"]),
        (1, bottom_y, "Interior", thresholds["pf_boundary_transition_km"], thresholds["pf_interior_abrupt_impact_km"]),
    ]
    for row_idx, y, row_label, boundary_km, abrupt_km in row_specs:
        ax.text(
            pf_x - 0.10,
            y + cell_h / 2,
            f"{row_label}\n" + (rf"$d_B<{boundary_km:g}$" if row_idx == 0 else rf"$d_B\geq{boundary_km:g}$"),
            ha="right",
            va="center",
            fontsize=FONT["annotation"] - 0.4,
            color=PF_COLOR,
            linespacing=0.92,
        )
        for col_idx, zone_key in enumerate(zones.PF_MATRIX_ZONES[row_idx]):
            x = pf_x + col_idx * (cell_w + gap)
            label = rf"$d_A<{abrupt_km:g}$" if col_idx == 0 else rf"$d_A\geq{abrupt_km:g}$"
            draw_stat_cell(
                ax,
                x=x,
                y=y,
                w=cell_w,
                h=cell_h,
                density=summary_lookup(summary, zone_key, "extreme_density"),
                share=summary_lookup(summary, zone_key, "domain_extreme_share"),
                cmap=cmap,
                norm=norm,
                label=label,
            )

    ax.set_xlim(0.0, 5.05)
    ax.set_ylim(-0.02, 2.70)


def build_all_summaries(zone_sample: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            zones.process_zone_summary(zone_sample, target="d_u"),
            zones.process_zone_summary(zone_sample, target="grad_mag_km"),
        ],
        ignore_index=True,
    )


def main() -> None:
    ensure_style()
    samples = load_profile_samples()
    boundary_bins = np.linspace(-20.0, 20.0, 41)
    tlrts_bins = np.linspace(0.0, 20.0, 41)

    du_boundary = binned_signed_boundary_profile(samples["du_boundary"], "d_u", boundary_bins)
    grad_boundary = binned_signed_boundary_profile(samples["grad_boundary"], "grad_mag_km", boundary_bins)

    du_tlrts_interior = binned_tlrts_profile(samples["du_tlrts"], "d_u", tlrts_bins, boundary_side="interior")
    grad_tlrts_interior = binned_tlrts_profile(samples["grad_tlrts"], "grad_mag_km", tlrts_bins, boundary_side="interior")
    du_tlrts_frontal = binned_tlrts_profile(samples["du_tlrts"], "d_u", tlrts_bins, boundary_side="transition")
    grad_tlrts_frontal = binned_tlrts_profile(samples["grad_tlrts"], "grad_mag_km", tlrts_bins, boundary_side="transition")

    threshold_table = build_threshold_fit_table()
    summary_table = build_all_summaries(samples["zone_sample"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table_export = threshold_table.rename(columns=zones.EXPORT_COLUMN_NAMES)
    threshold_table_export.to_csv(TABLE_DIR / f"{FIG_STEM}_threshold_fits.csv", index=False)
    threshold_table_export.to_csv(TABLE_DIR / f"{FIG_STEM}_trend_tests.csv", index=False)
    zones.export_process_zone_summary(summary_table).to_csv(TABLE_DIR / f"{FIG_STEM}_process_zone_summary.csv", index=False)

    fig = plt.figure(figsize=(11.8, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.56, 0.44], hspace=0.32, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_boundary_panel(ax_a, du_boundary, grad_boundary)
    plot_tlrts_panel(ax_b, du_tlrts_interior, grad_tlrts_interior, du_tlrts_frontal, grad_tlrts_frontal)
    plot_decomposition_panel(
        ax_c,
        summary_table.loc[summary_table["target"].eq("d_u")],
        target="d_u",
        color=DU_COLOR,
        title=r"Extreme $d_u$",
    )
    plot_decomposition_panel(
        ax_d,
        summary_table.loc[summary_table["target"].eq("grad_mag_km")],
        target="grad_mag_km",
        color=GRAD_COLOR,
        title=r"Extreme $|\nabla d_u|$",
    )

    for ax, label, xpos in [
        (ax_a, "A", -0.10),
        (ax_b, "B", -0.10),
        (ax_c, "C", -0.06),
        (ax_d, "D", -0.06),
    ]:
        add_panel_label(ax, label, x=xpos, y=1.03)

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
