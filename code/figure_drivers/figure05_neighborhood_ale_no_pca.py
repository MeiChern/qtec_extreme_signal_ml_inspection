#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure05_process_zone_ale_stability_no_pca.py
# Renamed package path: code/figure_drivers/figure05_neighborhood_ale_no_pca.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import numpy as np
import pandas as pd

from submission_build_common import TABLE_DIR

import Figure05_process_zone_ale_stability as base
import susceptibility_no_pca_common as no_pca


FIG_STEM = "Figure05_process_zone_ale_stability_no_pca"


def summarize_curves(
    *,
    target: str,
    domain: str,
    feature: str,
    zones: list[base.ZoneSpec],
    fitted: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in base.build_curves(feature, zones, fitted):
        zone = item["zone"]
        curve = item["curve"]
        if curve.empty:
            continue
        x = pd.to_numeric(curve["x_plot"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(curve["y"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        rows.append(
            {
                "target": target,
                "domain": domain,
                "feature": feature,
                "zone": zone.tag,
                "zone_label": zone.label,
                "n_ale_bins": int(finite.sum()),
                "feature_min": float(np.nanmin(x[finite])),
                "feature_max": float(np.nanmax(x[finite])),
                "ale_start": float(y[finite][0]),
                "ale_end": float(y[finite][-1]),
                "ale_min": float(np.nanmin(y[finite])),
                "ale_max": float(np.nanmax(y[finite])),
                "ale_range": float(np.nanmax(y[finite]) - np.nanmin(y[finite])),
                "model_preprocessing": "median_imputation_no_pca",
            }
        )
    return rows


def write_ale_summary(
    *,
    npf_fits: dict[str, dict[str, dict[str, object]]],
    pf_fits: dict[str, dict[str, dict[str, object]]],
) -> None:
    rows: list[dict[str, object]] = []
    for target in ["d_u", "grad_mag_km"]:
        rows.extend(
            summarize_curves(
                target=target,
                domain="npf",
                feature="magt",
                zones=base.NPF_ZONES,
                fitted=npf_fits[target],
            )
        )
        for feature in ["twi", "ndvi"]:
            rows.extend(
                summarize_curves(
                    target=target,
                    domain="pf",
                    feature=feature,
                    zones=base.PF_ZONES,
                    fitted=pf_fits[target],
                )
            )
    out = pd.DataFrame(rows)
    path = TABLE_DIR / f"{FIG_STEM}_ale_summary.csv"
    out.to_csv(path, index=False)
    print(f"[{FIG_STEM}] Wrote no-PCA ALE summary: {path}", flush=True)


def main() -> None:
    no_pca.install_no_pca_model_builder()
    base.FIG_STEM = FIG_STEM
    base.ensure_style()

    npf_fits = {target: base.fit_zone_results(target, base.NPF_ZONES) for target in ["d_u", "grad_mag_km"]}
    pf_fits = {target: base.fit_zone_results(target, base.PF_ZONES) for target in ["d_u", "grad_mag_km"]}

    fig, axes = base.plt.subplots(2, 3, figsize=(12.0, 8.4), constrained_layout=False)
    base.plt.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.24, wspace=0.28, hspace=0.42)

    panel_specs = [
        ("d_u", 0, 0, "NPF · MAGT", "magt", base.NPF_ZONES, base.NPF_TINT, npf_fits["d_u"], False),
        ("d_u", 0, 1, "PF · TWI", "twi", base.PF_ZONES, base.PF_TINT, pf_fits["d_u"], False),
        ("d_u", 0, 2, "PF · NDVI", "ndvi", base.PF_ZONES, base.PF_TINT, pf_fits["d_u"], False),
        (
            "grad_mag_km",
            1,
            0,
            "NPF · MAGT",
            "magt",
            base.NPF_ZONES,
            base.NPF_TINT,
            npf_fits["grad_mag_km"],
            False,
        ),
        (
            "grad_mag_km",
            1,
            1,
            "PF · TWI",
            "twi",
            base.PF_ZONES,
            base.PF_TINT,
            pf_fits["grad_mag_km"],
            False,
        ),
        (
            "grad_mag_km",
            1,
            2,
            "PF · NDVI",
            "ndvi",
            base.PF_ZONES,
            base.PF_TINT,
            pf_fits["grad_mag_km"],
            False,
        ),
    ]
    for target, row, col, title, feature, zones, domain_color, fitted, stability_hint in panel_specs:
        base.plot_single_target_panel(
            axes[row, col],
            title=title,
            feature=feature,
            zones=zones,
            tint=base.panel_tint(domain_color),
            target_color=base.TARGET_COLORS[target],
            fitted=fitted,
            show_ylabel=(col == 0),
            domain_color=domain_color,
            stability_hint=stability_hint,
        )

    for ax, label in zip(axes.flat, ["A", "B", "C", "D", "E", "F"]):
        base.add_panel_label(ax, label, x=-0.10, y=1.04)

    base.add_row_header(fig, y=0.750, text=r"Extreme $d_u$ susceptibility", color=base.DU_COLOR)
    base.add_row_header(fig, y=0.360, text=r"Extreme $|\nabla d_u|$ susceptibility", color=base.GRAD_COLOR)

    neutral = "#404040"
    npf_handles = [
        base.Line2D([0], [0], color=neutral, linewidth=zone.linewidth, linestyle=zone.linestyle, label=zone.label)
        for zone in base.NPF_ZONES
    ]
    pf_handles = [
        base.Line2D([0], [0], color=neutral, linewidth=zone.linewidth, linestyle=zone.linestyle, label=zone.label)
        for zone in base.PF_ZONES
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
        fontsize=base.FONT["axis"] + 0.5,
        title_fontsize=base.FONT["axis"] + 0.5,
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
        fontsize=base.FONT["axis"] + 0.5,
        title_fontsize=base.FONT["axis"] + 0.5,
    )
    base.color_legend_text(legend_npf, base.NPF_TINT)
    base.color_legend_text(legend_pf, base.PF_TINT)
    fig.add_artist(legend_npf)
    fig.add_artist(legend_pf)

    png, pdf = base.save_figure(fig, FIG_STEM)
    print(f"[{FIG_STEM}] {png}", flush=True)
    print(f"[{FIG_STEM}] {pdf}", flush=True)
    write_ale_summary(npf_fits=npf_fits, pf_fits=pf_fits)


if __name__ == "__main__":
    main()
