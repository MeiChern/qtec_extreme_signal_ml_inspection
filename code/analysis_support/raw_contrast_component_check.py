#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure03_raw_contrast_component_check.py
# Renamed package path: code/analysis_support/raw_contrast_component_check.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import warnings
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import average_precision_score

from submission_build_common import FONT, TABLE_DIR, add_panel_label, blend_with_white, ensure_style, save_figure

import Figure03_domain_specific_susceptibility as fig3
import Figure_reorganize_extreme_deformation_susceptibity as figsus
import figure6_susceptibility_stacked as fig6


FIG_STEM = "Figure03_raw_contrast_component_check"
CACHE_VERSION = 1
COMPONENT_MODEL_N_JOBS = max(1, int(os.environ.get("FIG3_COMPONENT_N_JOBS", "4")))
COMPONENTS = ("raw", "contrast")
COMPONENT_LABELS = {"raw": "Raw", "contrast": "Spatial contrast"}
PLOT_DOMAINS = ("npf", "pf", "combined")
FAMILY_DISPLAY_LABELS = {
    "thermal": "Thermal\nconditions",
    "terrain_energy": "Terrain\nand energy",
    "soil_substrate": "Soil\nand substrate",
    "vegetation_productivity": "Vegetation\nand productivity",
    "hydro_wetness": "Hydrologic\nwetness",
}
FAMILY_TABLE_LABELS = {
    "thermal": "Thermal conditions",
    "terrain_energy": "Terrain and energy",
    "soil_substrate": "Soil and substrate",
    "vegetation_productivity": "Vegetation and productivity",
    "hydro_wetness": "Hydrologic wetness",
}
COMPONENT_OFFSETS = {
    ("npf", "raw"): -0.36,
    ("npf", "contrast"): -0.22,
    ("pf", "raw"): -0.08,
    ("pf", "contrast"): 0.08,
    ("combined", "raw"): 0.22,
    ("combined", "contrast"): 0.36,
}


def log_step(message: str) -> None:
    print(f"[{FIG_STEM}] {message}", flush=True)


def component_cache_path(target: str, domain: str) -> object:
    stub = figsus.TARGET_META[target]["table_stub"]
    return TABLE_DIR / f"{FIG_STEM}_{stub}_{domain}_importance.csv"


def component_groups_for_features(feature_names: list[str]) -> list[dict[str, object]]:
    available = set(feature_names)
    assigned: set[str] = set()
    groups: list[dict[str, object]] = []

    for family, label, features in fig3.FAMILY_DEFINITIONS:
        present = [feature for feature in features if feature in available]
        if not present:
            continue
        assigned.update(present)
        for component in COMPONENTS:
            component_features = [
                feature
                for feature in present
                if fig3.feature_origin(feature) == component
            ]
            if not component_features:
                continue
            groups.append(
                {
                    "family": family,
                    "family_label": FAMILY_TABLE_LABELS.get(family, label),
                    "component": component,
                    "component_label": COMPONENT_LABELS[component],
                    "features": component_features,
                }
            )

    ignored = set(fig3.COMBINED_EXCLUDED_FEATURES) | set(fig3.GROUND_ICE_CONTEXT_FEATURES)
    unassigned = sorted(available - assigned - ignored)
    if unassigned:
        raise RuntimeError("Unassigned predictors in raw/contrast split: " + ", ".join(unassigned))
    return groups


def valid_component_cache(df: pd.DataFrame, *, target: str, domain: str) -> bool:
    required = {
        "target",
        "domain",
        "family",
        "family_label",
        "component",
        "component_label",
        "importance",
        "importance_std",
        "n_features",
        "features",
        "cache_version",
    }
    if not required.issubset(df.columns):
        return False
    version = pd.to_numeric(df["cache_version"], errors="coerce")
    return bool(
        version.notna().all()
        and (version == CACHE_VERSION).all()
        and df["target"].astype(str).eq(target).all()
        and df["domain"].astype(str).eq(domain).all()
    )


def load_cached_component_importance(target: str, domain: str) -> pd.DataFrame | None:
    path = component_cache_path(target, domain)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not valid_component_cache(df, target=target, domain=domain):
        log_step(f"Ignoring stale raw/contrast cache for {target}/{domain}: {path.name}")
        return None
    log_step(f"Using raw/contrast cache for {target}/{domain}: {path.name}")
    return df


def compute_component_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups: list[dict[str, object]],
    *,
    target: str,
    domain: str,
) -> pd.DataFrame:
    max_rows = min(fig3.GROUPED_IMPORTANCE_MAX_ROWS, len(X_test))
    perm_idx = np.arange(len(X_test))
    rng = np.random.default_rng(figsus.SEED)
    if len(perm_idx) > max_rows:
        perm_idx = np.sort(rng.choice(perm_idx, size=max_rows, replace=False))

    X_base = X_test.iloc[perm_idx].reset_index(drop=True)
    y_base = np.asarray(y_test, dtype=int)[perm_idx]
    baseline = float(average_precision_score(y_base, estimator.predict_proba(X_base)[:, 1]))

    rows: list[dict[str, object]] = []
    for group in groups:
        features = list(group["features"])
        drops: list[float] = []
        for _repeat in range(fig3.GROUPED_IMPORTANCE_REPEATS):
            order = rng.permutation(len(X_base))
            X_perm = X_base.copy()
            X_perm.loc[:, features] = X_base.iloc[order][features].to_numpy(copy=True)
            perm_score = float(average_precision_score(y_base, estimator.predict_proba(X_perm)[:, 1]))
            drops.append(baseline - perm_score)

        arr = np.asarray(drops, dtype=float)
        rows.append(
            {
                "target": target,
                "domain": domain,
                "family": str(group["family"]),
                "family_label": str(group["family_label"]),
                "component": str(group["component"]),
                "component_label": str(group["component_label"]),
                "importance": float(np.mean(arr)),
                "importance_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n_features": int(len(features)),
                "features": ";".join(features),
                "baseline_average_precision": baseline,
                "n_permutation_rows": int(len(X_base)),
                "n_repeats": int(fig3.GROUPED_IMPORTANCE_REPEATS),
                "scoring": "average_precision",
                "cache_version": int(CACHE_VERSION),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "component"]).reset_index(drop=True)


def fit_component_importance(target: str, domain: str) -> pd.DataFrame:
    log_step(f"Computing raw/contrast grouped importance for {target}/{domain}")
    df, available_features, feature_groups = fig3.get_training_frame(target)
    if domain == "combined":
        work = df.copy()
        work["label"], threshold = fig3.universal_label(work, target)
        log_step(f"{target}/{domain}: pooled target-tail threshold={threshold:.6g}")
    else:
        work = df.loc[df["domain"].eq(domain)].copy().reset_index(drop=True)
        work["label"] = figsus.build_domain_label(work, target=target, domain=domain)
        log_step(
            f"{target}/{domain}: domain target-tail threshold="
            f"{figsus.threshold_for_domain(target, domain):.6g}"
        )

    work["block_id"] = fig6.make_spatial_block_id(work, block_size_m=fig3.COMBINED_BLOCK_SIZE_M)
    work = work.loc[work["label"].notna() & work["block_id"].notna()].reset_index(drop=True)
    train_idx, test_idx = figsus.choose_label_holdout_split(
        work,
        label_col="label",
        test_size=fig3.COMBINED_TEST_SIZE,
    )
    train_df = work.iloc[train_idx].reset_index(drop=True)
    test_df = work.iloc[test_idx].reset_index(drop=True)

    missing_df = fig6.audit_feature_missingness(train_df, feature_groups)
    dropped_features = sorted(
        missing_df.loc[
            missing_df["missing_frac"] > fig3.COMBINED_MAX_MISSING_FRAC,
            "feature",
        ]
        .drop_duplicates()
        .tolist()
    )
    model_features = [feature for feature in available_features if feature not in dropped_features]
    groups = component_groups_for_features(model_features)

    X_train = fig6.make_model_frame(train_df, model_features)
    X_test = fig6.make_model_frame(test_df, model_features)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    groups_train = train_df["block_id"].astype(str).to_numpy()

    pos_rate = max(float(np.mean(y_train)), 1e-6)
    pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
    stack = figsus.build_domain_models(
        pos_weight=pos_weight,
        stack_cv=fig6.FixedGroupKFold(groups_train, n_splits=5),
        n_jobs=COMPONENT_MODEL_N_JOBS,
    )["Stack"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack.fit(X_train, y_train)

    out = compute_component_importance(
        stack,
        X_test,
        y_test,
        groups,
        target=target,
        domain=domain,
    )
    path = component_cache_path(target, domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log_step(f"Wrote raw/contrast grouped importance for {target}/{domain}: {path.name}")
    return out


def get_component_importance(target: str, domain: str) -> pd.DataFrame:
    cached = load_cached_component_importance(target, domain)
    if cached is not None and not cached.empty:
        return cached
    return fit_component_importance(target, domain)


def family_order_for_target(target: str) -> list[str]:
    full_frames = {
        domain: fig3.get_grouped_importance(target, domain)
        for domain in PLOT_DOMAINS
    }
    max_by_family: dict[str, float] = {}
    label_by_family: dict[str, str] = {}
    for frame in full_frames.values():
        for row in frame.itertuples(index=False):
            family = str(row.family)
            label_by_family.setdefault(family, str(row.family_label))
            max_by_family[family] = max(max_by_family.get(family, -np.inf), float(row.importance))
    return sorted(max_by_family, key=lambda family: (-max_by_family[family], label_by_family[family]))


def lookup_component_frames(target: str) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for domain in PLOT_DOMAINS:
        frame = get_component_importance(target, domain)
        for row in frame.itertuples(index=False):
            lookup[(str(row.family), str(row.component), domain)] = {
                "importance": float(row.importance),
                "importance_std": float(row.importance_std),
                "family_label": str(row.family_label),
                "n_features": int(row.n_features),
                "features": str(row.features),
            }
    return lookup


def plot_panel(ax, *, target: str, title: str, target_color: str) -> None:
    family_order = family_order_for_target(target)
    lookup = lookup_component_frames(target)
    ax.set_facecolor(blend_with_white(target_color, 0.93))

    values: list[float] = []
    for family in family_order:
        for domain in PLOT_DOMAINS:
            for component in COMPONENTS:
                payload = lookup[(family, component, domain)]
                value = float(payload["importance"])
                spread = float(payload["importance_std"])
                values.extend([value - spread, value + spread])
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 0.01
    span = max(max_value - min_value, 0.01)
    ax.set_xlim(min(min_value - 0.18 * span, -0.04 * span), max_value + 0.18 * span)
    ax.axvline(0.0, color="0.72", linewidth=0.8, zorder=1)

    y_pos = np.arange(len(family_order), dtype=float)
    bar_height = 0.115
    for y, family in zip(y_pos, family_order):
        for domain in PLOT_DOMAINS:
            color = fig3.DOMAIN_COLORS[domain]
            for component in COMPONENTS:
                payload = lookup[(family, component, domain)]
                value = float(payload["importance"])
                spread = float(payload["importance_std"])
                offset = COMPONENT_OFFSETS[(domain, component)]
                if component == "raw":
                    ax.barh(
                        y + offset,
                        value,
                        height=bar_height,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45,
                        alpha=0.96,
                        zorder=3,
                    )
                else:
                    ax.barh(
                        y + offset,
                        value,
                        height=bar_height,
                        color="none",
                        edgecolor=color,
                        linewidth=0.95,
                        linestyle=(0, (3.0, 1.5)),
                        hatch="////",
                        alpha=0.96,
                        zorder=3,
                    )
                if spread > 0:
                    ax.errorbar(
                        value,
                        y + offset,
                        xerr=spread,
                        fmt="none",
                        ecolor=blend_with_white(color, 0.35),
                        elinewidth=0.75,
                        capsize=1.8,
                        capthick=0.75,
                        zorder=5,
                    )

    labels = []
    for family in family_order:
        labels.append(FAMILY_DISPLAY_LABELS.get(family, lookup[(family, "raw", "npf")]["family_label"]))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT["tick"])
    for label in ax.get_yticklabels():
        label.set_linespacing(0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Component grouped importance (AP drop)")
    ax.set_title(title, pad=8, color=target_color, fontweight="semibold")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color="0.92", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(family_order) - 0.55, -0.70)


def main() -> None:
    ensure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.95), constrained_layout=False)
    plt.subplots_adjust(left=0.15, right=0.985, top=0.88, bottom=0.23, wspace=0.35)

    plot_panel(
        axes[0],
        target="d_u",
        title=r"Extreme $d_u$ susceptibility",
        target_color=fig3.DU_COLOR,
    )
    plot_panel(
        axes[1],
        target="grad_mag_km",
        title=r"Extreme $|\nabla d_u|$ susceptibility",
        target_color=fig3.GRAD_COLOR,
    )
    for ax, label in zip(axes, ["A", "B"]):
        add_panel_label(ax, label, x=-0.14, y=1.02)

    legend_handles = [
        Line2D([0], [0], color=fig3.NPF_COLOR, linewidth=6.0, label="NPF domain"),
        Line2D([0], [0], color=fig3.PF_COLOR, linewidth=6.0, label="PF domain"),
        Line2D([0], [0], color=fig3.COMBINED_COLOR, linewidth=2.8, label="Combined baseline"),
        Patch(facecolor="0.72", edgecolor="white", linewidth=0.45, label="Raw variables"),
        Patch(
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.95,
            linestyle=(0, (3.0, 1.5)),
            hatch="////",
            label="Spatial contrast variables",
        ),
        Line2D([0], [0], color="0.40", linewidth=0.8, marker="|", markersize=8.0, label="Mean +/- SD"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.54, 0.068),
        columnspacing=0.95,
        handletextpad=0.6,
    )
    png, pdf = save_figure(fig, FIG_STEM)
    log_step(str(png))
    log_step(str(pdf))


if __name__ == "__main__":
    main()
