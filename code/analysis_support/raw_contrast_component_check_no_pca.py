#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure03_raw_contrast_component_check_no_pca.py
# Renamed package path: code/analysis_support/raw_contrast_component_check_no_pca.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import pandas as pd

from submission_build_common import TABLE_DIR

import Figure03_raw_contrast_component_check as base
import Figure_reorganize_extreme_deformation_susceptibity as figsus
import susceptibility_no_pca_common as no_pca


FIG_STEM = "Figure03_raw_contrast_component_check_no_pca"
FAMILY_ORDER = [
    "thermal",
    "terrain_energy",
    "soil_substrate",
    "vegetation_productivity",
    "hydro_wetness",
]


def family_order_for_target(_target: str) -> list[str]:
    return FAMILY_ORDER


def install_no_pca_et_importance_model() -> None:
    def build_domain_models_no_pca_et(*, pos_weight: float, stack_cv, n_jobs: int) -> dict[str, object]:
        models = no_pca.build_domain_models_no_pca(
            pos_weight=pos_weight,
            stack_cv=stack_cv,
            n_jobs=n_jobs,
        )
        models["Stack"] = models["ET"]
        return models

    figsus.build_domain_models = build_domain_models_no_pca_et


def write_summary_table() -> None:
    frames: list[pd.DataFrame] = []
    for target in ["d_u", "grad_mag_km"]:
        for domain in base.PLOT_DOMAINS:
            frame = base.get_component_importance(target, domain).copy()
            frame["model_preprocessing"] = "median_imputation_no_pca"
            frame["interpretation_model"] = "extratrees"
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["target", "domain", "family", "component"]).reset_index(drop=True)
    path = TABLE_DIR / f"{FIG_STEM}_component_importance_summary.csv"
    out.to_csv(path, index=False)
    base.log_step(f"Wrote no-PCA component summary: {path.name}")


def main() -> None:
    install_no_pca_et_importance_model()
    base.FIG_STEM = FIG_STEM
    base.CACHE_VERSION = 3
    base.family_order_for_target = family_order_for_target
    base.main()
    write_summary_table()


if __name__ == "__main__":
    main()
