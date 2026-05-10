#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/build_selected_figures.py
# Renamed package path: code/original_project_helpers/build_selected_figures_original_layout.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

"""Rebuild the manuscript and supplementary figure set used by the submission."""

from __future__ import annotations

import os
import shutil
import subprocess

from _run_source import PROJECT_ROOT, SCRIPT_DIR, resolve_python

FIGURE_SCRIPTS = (
    "Figure01_study_area_overview.py",
    "Figure02_regional_deformation_domain_contrast.py",
    "Figure03_raw_contrast_component_check_no_pca.py",
    "Figure04_proximity_decomposition.py",
    "Figure05_process_zone_ale_stability_no_pca.py",
    "FigureS1_environmental_predictor_maps.py",
    "FigureS2_methodology_diagram.py",
    "FigureS3_susceptibility_prediction_maps.py",
    "FigureS4_tlrts_distance_distribution.py",
    "FigureS5_process_zone_zoom_windows.py",
)

PROMOTED_OUTPUTS = {
    "Figure03_raw_contrast_component_check_no_pca.py": (
        (
            "pnas_nexus_submission/results/figures/Figure03_raw_contrast_component_check_no_pca.png",
            "pnas_nexus_submission/results/figures/Figure03_domain_specific_susceptibility.png",
        ),
        (
            "pnas_nexus_submission/results/figures/Figure03_raw_contrast_component_check_no_pca.pdf",
            "pnas_nexus_submission/results/figures/Figure03_domain_specific_susceptibility.pdf",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_component_importance_summary.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_component_importance_summary.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_du_npf_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_du_npf_component_importance.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_du_pf_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_du_pf_component_importance.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_du_combined_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_du_combined_component_importance.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_grad_npf_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_grad_npf_component_importance.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_grad_pf_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_grad_pf_component_importance.csv",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure03_raw_contrast_component_check_no_pca_grad_combined_importance.csv",
            "pnas_nexus_submission/results/tables/Figure03_domain_specific_susceptibility_grad_combined_component_importance.csv",
        ),
    ),
    "Figure05_process_zone_ale_stability_no_pca.py": (
        (
            "pnas_nexus_submission/results/figures/Figure05_process_zone_ale_stability_no_pca.png",
            "pnas_nexus_submission/results/figures/Figure05_process_zone_ale_stability.png",
        ),
        (
            "pnas_nexus_submission/results/figures/Figure05_process_zone_ale_stability_no_pca.pdf",
            "pnas_nexus_submission/results/figures/Figure05_process_zone_ale_stability.pdf",
        ),
        (
            "pnas_nexus_submission/results/tables/Figure05_process_zone_ale_stability_no_pca_ale_summary.csv",
            "pnas_nexus_submission/results/tables/Figure05_process_zone_ale_stability_ale_summary.csv",
        ),
    ),
}


def promote_outputs(source_name: str) -> None:
    for source_rel, dest_rel in PROMOTED_OUTPUTS.get(source_name, ()):
        source = PROJECT_ROOT / source_rel
        dest = PROJECT_ROOT / dest_rel
        if not source.exists():
            raise FileNotFoundError(f"Expected generated output not found: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"[build_selected_figures] Promoted {source.name} -> {dest.name}", flush=True)


def run_submission_script(source_name: str) -> None:
    source = SCRIPT_DIR / source_name
    if not source.exists():
        raise FileNotFoundError(f"Bundled submission figure script not found: {source}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{SCRIPT_DIR}:{existing_pythonpath}"
        if existing_pythonpath
        else str(SCRIPT_DIR)
    )

    print(f"[build_selected_figures] Running: {source.name}", flush=True)
    subprocess.run([resolve_python(), str(source)], check=True, cwd=str(PROJECT_ROOT), env=env)
    promote_outputs(source_name)


def main() -> None:
    for source_name in FIGURE_SCRIPTS:
        run_submission_script(source_name)


if __name__ == "__main__":
    main()
