#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure03_domain_specific_susceptibility.py
# Renamed package path: code/figure_drivers/figure03_domain_susceptibility.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from pathlib import Path
import warnings

from submission_build_common import (
    FONT,
    ROOT_DIR,
    SOURCE_CACHE_DIR,
    TABLE_DIR,
    add_panel_label,
    blend_with_white,
    ensure_style,
    save_figure,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import average_precision_score

import Figure_reorganize_extreme_deformation_susceptibity as figsus
import figure6_susceptibility_stacked as fig6


FIG_STEM = "Figure03_domain_specific_susceptibility"
PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
COMBINED_COLOR = "#6F6F6F"
DU_COLOR = "#1E5BAA"
GRAD_COLOR = "#C1272D"
COMBINED_EXCLUDED_FEATURES = {"Perma_Distr_map"}
PLOT_EXCLUDED_FEATURES = COMBINED_EXCLUDED_FEATURES | set(
    getattr(figsus, "IMPORTANCE_PLOT_EXCLUDE_FEATURES", set())
)
DISPLAY_LABELS = {"temperature_mean": "MAAT"}
SUFFIX_LABELS = {
    "__lstd": "local std.",
    "__gmag": "gradient",
}
TOP_N = 5
ROW_CAP = 8
DISPLAY_EPS = 5e-4
GROUPED_CACHE_VERSION = 1
GROUPED_IMPORTANCE_REPEATS = fig6.PERMUTATION_REPEATS
GROUPED_IMPORTANCE_MAX_ROWS = fig6.PERMUTATION_MAX_ROWS
COMBINED_NEIGHBORS_TRANS = 21
COMBINED_BLOCK_SIZE_M = fig6.SPATIAL_BLOCK_SIZE_M
COMBINED_TEST_SIZE = 0.25
COMBINED_MAX_MISSING_FRAC = fig6.MAX_FEATURE_MISSING_FRAC
COMBINED_MODEL_N_JOBS = 1
PANEL_ORDER = ["npf", "pf", "combined"]
DOMAIN_PANEL_ORDER = ["npf", "pf"]
DOMAIN_COLORS = {
    "npf": NPF_COLOR,
    "pf": PF_COLOR,
    "combined": COMBINED_COLOR,
}
FAMILY_DEFINITIONS = [
    (
        "thermal",
        "Thermal",
        ["magt", "temperature_mean", "magt__lstd", "magt__gmag"],
    ),
    (
        "hydro_wetness",
        "Hydro-wetness",
        ["precipitation_mean", "twi", "twi__lstd", "twi__gmag"],
    ),
    (
        "vegetation_productivity",
        "Vegetation/productivity",
        ["ndvi", "gpp_mean", "ndvi__lstd", "ndvi__gmag"],
    ),
    (
        "soil_substrate",
        "Soil/substrate",
        [
            "bulk_density",
            "cf",
            "soc",
            "soil_thickness",
            "bulk_density__lstd",
            "bulk_density__gmag",
            "cf__lstd",
            "cf__gmag",
            "soc__lstd",
            "soc__gmag",
            "soil_thickness__lstd",
            "soil_thickness__gmag",
        ],
    ),
    (
        "terrain_energy",
        "Terrain/energy",
        ["dem", "slope", "difpr", "dirpr", "dem__lstd", "dem__gmag", "slope__lstd", "slope__gmag"],
    ),
]
GROUND_ICE_CONTEXT_FEATURES = ["vwc35"]
TRAINING_FRAME_CACHE: dict[str, tuple[pd.DataFrame, list[str], dict[str, list[str]]]] = {}


def feature_name(feature: str) -> str:
    label = DISPLAY_LABELS.get(feature, fig6.FEATURE_LABELS.get(feature, feature))
    for suffix, suffix_label in SUFFIX_LABELS.items():
        if feature.endswith(suffix):
            base_feature = feature[: -len(suffix)]
            base = DISPLAY_LABELS.get(base_feature, fig6.FEATURE_LABELS.get(base_feature, base_feature))
            return f"{base}\n({suffix_label})"
    return label


def feature_origin(feature: str) -> str:
    return "contrast" if any(feature.endswith(suffix) for suffix in SUFFIX_LABELS) else "raw"


def log_step(message: str) -> None:
    print(f"[{FIG_STEM}] {message}", flush=True)


def universal_label(df: pd.DataFrame, target: str) -> tuple[pd.Series, float]:
    if target == "d_u":
        threshold = float(np.percentile(df["d_u"].to_numpy(dtype=float), 5.0))
        label = (pd.to_numeric(df["d_u"], errors="coerce") <= threshold).astype(int)
    else:
        threshold = float(np.percentile(df["grad_mag_km"].to_numpy(dtype=float), 95.0))
        label = (pd.to_numeric(df["grad_mag_km"], errors="coerce") >= threshold).astype(int)
    return label.rename("label"), threshold


def source_csv_path() -> Path:
    path = ROOT_DIR / "df_all_data_with_wright_du_plus_grad.csv"
    if path.exists():
        return path
    fallback = ROOT_DIR / "df_all_data_with_wright_du.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find the source deformation CSV used to build transition rasters.")


def load_combined_training_frame(target: str) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    sample_name = (
        "Figure_reorganize_extreme_deformation_susceptibity_du_negdu_sample_pf90000_npf90000.csv.gz"
        if target == "d_u"
        else "Figure_reorganize_extreme_deformation_susceptibity_grad_negdu_sample_pf90000_npf90000.csv.gz"
    )
    df = pd.read_csv(SOURCE_CACHE_DIR / sample_name)
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    _raw_features, _contrast_features, combined_features, transition_base_vars = figsus.build_feature_sets(
        exclude_magt=False
    )
    transition_window_size = fig6.neighbors_to_window_size(COMBINED_NEIGHBORS_TRANS)
    transition_cache_dir = SOURCE_CACHE_DIR / f"{figsus.FIG_BASENAME}_transition_rasters"
    log_step(f"{target}: attaching transition metrics from {transition_cache_dir.name}")
    grid, _mean_paths, metric_paths = fig6.ensure_transition_rasters(
        base_dir=ROOT_DIR,
        csv_path=source_csv_path(),
        cache_dir=transition_cache_dir,
        vars_for_transition=transition_base_vars,
        chunksize=figsus.CHUNKSIZE,
        window_size=transition_window_size,
    )
    df = fig6.attach_raster_transition_metrics(
        df=df,
        metric_paths=metric_paths,
        grid=grid,
        vars_for_transition=transition_base_vars,
    )

    available_features = [
        feature
        for feature in combined_features
        if feature not in COMBINED_EXCLUDED_FEATURES and feature in df.columns
    ]
    feature_groups = {
        "raw": [feature for feature in available_features if feature_origin(feature) == "raw"],
        "contrast": [feature for feature in available_features if feature_origin(feature) == "contrast"],
    }
    return df, available_features, feature_groups


def grouped_cache_path(target: str, domain: str) -> Path:
    stub = figsus.TARGET_META[target]["table_stub"]
    return TABLE_DIR / f"{FIG_STEM}_{stub}_{domain}_grouped_family_importance.csv"


def valid_grouped_cache(df: pd.DataFrame, *, target: str, domain: str) -> bool:
    required = {
        "target",
        "domain",
        "family",
        "family_label",
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
        and (version == GROUPED_CACHE_VERSION).all()
        and df["target"].astype(str).eq(target).all()
        and df["domain"].astype(str).eq(domain).all()
    )


def finalize_grouped_importance_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["feature"] = out["family"].astype(str)
    out["origin"] = "family"
    out["display"] = [
        f"{label}\n(n={int(n_features)})"
        for label, n_features in zip(out["family_label"], out["n_features"])
    ]
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def load_cached_grouped_importance(target: str, domain: str) -> pd.DataFrame | None:
    path = grouped_cache_path(target, domain)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not valid_grouped_cache(df, target=target, domain=domain):
        log_step(f"Ignoring stale grouped-family importance cache for {target}/{domain}: {path.name}")
        return None
    log_step(f"Using grouped-family importance cache for {target}/{domain}: {path.name}")
    return finalize_grouped_importance_frame(df)


def get_training_frame(target: str) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    if target not in TRAINING_FRAME_CACHE:
        TRAINING_FRAME_CACHE[target] = load_combined_training_frame(target)
    return TRAINING_FRAME_CACHE[target]


def family_groups_for_features(feature_names: list[str]) -> list[dict[str, object]]:
    available = set(feature_names)
    assigned: set[str] = set()
    groups: list[dict[str, object]] = []
    for family, label, features in FAMILY_DEFINITIONS:
        present = [feature for feature in features if feature in available]
        if not present:
            continue
        groups.append(
            {
                "family": family,
                "family_label": label,
                "features": present,
            }
        )
        assigned.update(present)

    ignored = set(COMBINED_EXCLUDED_FEATURES) | set(GROUND_ICE_CONTEXT_FEATURES)
    unassigned = sorted(available - assigned - ignored)
    if unassigned:
        raise RuntimeError(
            "Main grouped-family importance has unassigned predictors: "
            + ", ".join(unassigned)
        )
    return groups


def grouped_permutation_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups: list[dict[str, object]],
    *,
    target: str,
    domain: str,
) -> pd.DataFrame:
    max_rows = min(GROUPED_IMPORTANCE_MAX_ROWS, len(X_test))
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
        for _repeat in range(GROUPED_IMPORTANCE_REPEATS):
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
                "importance": float(np.mean(arr)),
                "importance_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n_features": int(len(features)),
                "features": ";".join(features),
                "baseline_average_precision": baseline,
                "n_permutation_rows": int(len(X_base)),
                "n_repeats": int(GROUPED_IMPORTANCE_REPEATS),
                "scoring": "average_precision",
                "cache_version": int(GROUPED_CACHE_VERSION),
            }
        )

    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def fit_grouped_importance(target: str, domain: str) -> pd.DataFrame:
    log_step(f"Computing grouped-family importance for {target}/{domain}")
    df, available_features, feature_groups = get_training_frame(target)
    if domain == "combined":
        work = df.copy()
        work["label"], threshold = universal_label(work, target)
        log_step(f"{target}/{domain}: pooled target-tail threshold={threshold:.6g}")
    else:
        work = df.loc[df["domain"].eq(domain)].copy().reset_index(drop=True)
        work["label"] = figsus.build_domain_label(work, target=target, domain=domain)
        log_step(
            f"{target}/{domain}: domain target-tail threshold="
            f"{figsus.threshold_for_domain(target, domain):.6g}"
        )

    work["block_id"] = fig6.make_spatial_block_id(work, block_size_m=COMBINED_BLOCK_SIZE_M)
    work = work.loc[work["label"].notna() & work["block_id"].notna()].reset_index(drop=True)
    if len(work) < 1000:
        raise RuntimeError(f"Too few valid rows remain for grouped importance: {target}/{domain}.")

    train_idx, test_idx = figsus.choose_label_holdout_split(
        work,
        label_col="label",
        test_size=COMBINED_TEST_SIZE,
    )
    train_df = work.iloc[train_idx].reset_index(drop=True)
    test_df = work.iloc[test_idx].reset_index(drop=True)
    log_step(f"{target}/{domain}: train/test split = {len(train_df):,}/{len(test_df):,}")

    missing_df = fig6.audit_feature_missingness(train_df, feature_groups)
    missing_df["origin"] = missing_df["feature"].map(feature_origin)
    dropped_features = sorted(
        missing_df.loc[
            missing_df["missing_frac"] > COMBINED_MAX_MISSING_FRAC,
            "feature",
        ]
        .drop_duplicates()
        .tolist()
    )
    model_features = [feature for feature in available_features if feature not in dropped_features]
    if not model_features:
        raise RuntimeError(f"No predictors remain for grouped importance: {target}/{domain}.")
    groups = family_groups_for_features(model_features)
    family_size_msg = ", ".join(f"{g['family_label']}={len(g['features'])}" for g in groups)
    log_step(
        f"{target}/{domain}: retained {len(model_features)} predictors "
        f"({len(dropped_features)} dropped); family sizes: {family_size_msg}"
    )

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
        n_jobs=COMBINED_MODEL_N_JOBS,
    )["Stack"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack.fit(X_train, y_train)

    out = grouped_permutation_importance(
        stack,
        X_test,
        y_test,
        groups,
        target=target,
        domain=domain,
    )
    cache_path = grouped_cache_path(target, domain)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    log_step(f"Wrote grouped-family importance for {target}/{domain}: {cache_path.name}")
    return finalize_grouped_importance_frame(out)


def get_grouped_importance(target: str, domain: str) -> pd.DataFrame:
    cached = load_cached_grouped_importance(target, domain)
    if cached is not None and not cached.empty:
        return cached
    return fit_grouped_importance(target, domain)


def prepare_importance(df: pd.DataFrame, *, target: str, domain: str) -> pd.DataFrame:
    sub = df.copy()
    if "origin" not in sub.columns:
        sub["origin"] = sub["feature"].map(feature_origin)
    if "display" not in sub.columns:
        sub["display"] = sub["feature"].map(feature_name)
    sub = sub.loc[~sub["feature"].astype(str).isin(PLOT_EXCLUDED_FEATURES)].copy()
    sub = sub.sort_values("importance", ascending=False).reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)
    return sub


def build_feature_lookup(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    for row in df.itertuples(index=False):
        lookup[str(row.feature)] = {
            "importance": float(row.importance),
            "importance_std": float(getattr(row, "importance_std", 0.0)),
            "rank": int(row.rank),
            "display": str(row.display),
            "origin": str(row.origin),
        }
    return lookup


def assemble_panel_records(
    *,
    target: str,
    domain_imp: dict[tuple[str, str], pd.DataFrame],
    combined_df: pd.DataFrame,
) -> tuple[list[dict[str, object]], bool]:
    frames = {
        "npf": prepare_importance(domain_imp[("npf", target)], target=target, domain="npf"),
        "pf": prepare_importance(domain_imp[("pf", target)], target=target, domain="pf"),
        "combined": prepare_importance(combined_df, target=target, domain="combined"),
    }
    lookups = {domain: build_feature_lookup(frame) for domain, frame in frames.items()}

    top_features: list[str] = []
    for domain in DOMAIN_PANEL_ORDER:
        top_features.extend(frames[domain].head(TOP_N)["feature"].astype(str).tolist())
    feature_union = list(dict.fromkeys(top_features))
    missing_by_feature = {
        feature: [domain for domain in PANEL_ORDER if feature not in lookups[domain]]
        for feature in feature_union
    }
    incomplete_features = {
        feature: missing_domains
        for feature, missing_domains in missing_by_feature.items()
        if missing_domains
    }
    if incomplete_features:
        summary = "; ".join(
            f"{feature}: {','.join(missing_domains)}"
            for feature, missing_domains in incomplete_features.items()
        )
        log_step(f"{target}: dropping incomplete feature rows before plotting: {summary}")
    feature_union = [
        feature
        for feature in feature_union
        if not missing_by_feature[feature]
    ]
    if not feature_union:
        raise RuntimeError(f"No complete NPF/PF/combined feature rows remain for {target}.")

    max_importance: dict[str, float] = {}
    display_map: dict[str, str] = {}
    origin_map: dict[str, str] = {}
    for feature in feature_union:
        scores = []
        for domain in PANEL_ORDER:
            payload = lookups[domain].get(feature)
            if payload is None:
                continue
            display_map.setdefault(feature, str(payload["display"]))
            origin_map.setdefault(feature, str(payload["origin"]))
            scores.append(float(payload["importance"]))
        max_importance[feature] = max(scores) if scores else 0.0

    selected = sorted(feature_union, key=lambda feat: (-max_importance[feat], display_map[feat]))
    capped = len(selected) > ROW_CAP
    selected = selected[:ROW_CAP]

    records: list[dict[str, object]] = []
    for feature in selected:
        record = {
            "feature": feature,
            "display": display_map[feature],
            "origin": origin_map[feature],
            "domains": {},
        }
        for domain in PANEL_ORDER:
            payload = lookups[domain].get(feature)
            if payload is None:
                raise RuntimeError(f"Internal plotting selection error: {feature} missing for {domain}.")
            rank = int(payload["rank"])
            imp = float(payload["importance"])
            record["domains"][domain] = {
                "importance": imp,
                "importance_std": float(payload.get("importance_std", 0.0)),
                "rank": rank,
                "present": True,
                "top_n": rank <= TOP_N,
                "negative": imp < 0.0,
            }
        records.append(record)

    summary = ", ".join(str(rec["feature"]) for rec in records)
    cap_note = f" (capped to {ROW_CAP} rows)" if capped else ""
    log_step(f"{target}: selected {len(records)} feature rows{cap_note}: {summary}")
    return records, capped


def plot_domain_bar_panel(
    ax,
    *,
    records: list[dict[str, object]],
    title: str,
    target_color: str,
) -> None:
    ax.set_facecolor(blend_with_white(target_color, 0.93))

    importance_values: list[float] = []
    for record in records:
        for domain in PANEL_ORDER:
            payload = record["domains"][domain]
            if payload["present"]:
                value = float(payload["importance"])
                spread = float(payload.get("importance_std", 0.0))
                importance_values.extend([value - spread, value + spread])
    max_value = max(importance_values) if importance_values else 0.01
    min_value = min(importance_values) if importance_values else 0.0
    x_span = max(max_value - min_value, 0.01)
    ax.set_xlim(min(min_value - 0.18 * x_span, -0.04 * x_span), max_value + 0.22 * x_span)
    ax.axvline(0.0, color="0.72", linewidth=0.8, zorder=1)

    y_pos = np.arange(len(records), dtype=float)
    bar_height = 0.28

    for y, record in zip(y_pos, records):
        for offset, domain in [(-bar_height / 1.8, "npf"), (bar_height / 1.8, "pf")]:
            payload = record["domains"][domain]
            value = float(payload["importance"])
            spread = float(payload.get("importance_std", 0.0))
            if abs(value) >= DISPLAY_EPS:
                ax.barh(
                    y + offset,
                    value,
                    height=bar_height,
                    color=DOMAIN_COLORS[domain],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.96,
                    zorder=3,
                )
                if spread > 0:
                    ax.errorbar(
                        value,
                        y + offset,
                        xerr=spread,
                        fmt="none",
                        ecolor=blend_with_white(DOMAIN_COLORS[domain], 0.35),
                        elinewidth=0.8,
                        capsize=2.2,
                        capthick=0.8,
                        zorder=5,
                    )
            else:
                ax.plot(
                    [0.0],
                    [y + offset],
                    marker="o",
                    markersize=3.0,
                    color=DOMAIN_COLORS[domain],
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    zorder=4,
                )

        combined = record["domains"]["combined"]
        combined_value = float(combined["importance"])
        combined_spread = float(combined.get("importance_std", 0.0))
        ax.vlines(
            combined_value,
            y - 0.38,
            y + 0.38,
            color=COMBINED_COLOR,
            linewidth=1.7,
            alpha=0.95,
            zorder=4,
        )
        if combined_spread > 0:
            ax.hlines(
                y,
                combined_value - combined_spread,
                combined_value + combined_spread,
                color=COMBINED_COLOR,
                linewidth=1.0,
                alpha=0.72,
                zorder=4,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(record["display"]) for record in records], fontsize=FONT["tick"])
    for label in ax.get_yticklabels():
        label.set_linespacing(0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Grouped permutation importance (AP drop)")
    ax.set_title(title, pad=8, color=target_color, fontweight="semibold")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color="0.92", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(records) - 0.45, -0.65)


def main() -> None:
    ensure_style()
    domain_imp = {
        (domain, target): get_grouped_importance(target, domain)
        for target in ["d_u", "grad_mag_km"]
        for domain in ["npf", "pf"]
    }
    combined_du = get_grouped_importance("d_u", "combined")
    combined_grad = get_grouped_importance("grad_mag_km", "combined")
    du_records, du_capped = assemble_panel_records(
        target="d_u",
        domain_imp=domain_imp,
        combined_df=combined_du,
    )
    grad_records, grad_capped = assemble_panel_records(
        target="grad_mag_km",
        domain_imp=domain_imp,
        combined_df=combined_grad,
    )
    if du_capped or grad_capped:
        log_step(f"Feature-row cap triggered in at least one panel; caption should mention the {ROW_CAP}-row cap.")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), constrained_layout=False)
    plt.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.20, wspace=0.34)

    plot_domain_bar_panel(
        axes[0],
        records=du_records,
        title=r"Extreme $d_u$ susceptibility",
        target_color=DU_COLOR,
    )
    plot_domain_bar_panel(
        axes[1],
        records=grad_records,
        title=r"Extreme $|\nabla d_u|$ susceptibility",
        target_color=GRAD_COLOR,
    )
    axes[1].set_xticks([0.00, 0.01, 0.03, 0.05])
    axes[1].set_xticklabels(["0.00", "0.01", "0.03", "0.05"])

    for ax, label in zip(axes, ["A", "B"]):
        add_panel_label(ax, label, x=-0.12, y=1.02)

    legend_handles = [
        Line2D([0], [0], color=NPF_COLOR, linewidth=6.0, label="NPF domain"),
        Line2D([0], [0], color=PF_COLOR, linewidth=6.0, label="PF domain"),
        Line2D([0], [0], color=COMBINED_COLOR, linewidth=1.8, label="Combined baseline"),
        Line2D([0], [0], color="0.40", linewidth=0.8, marker="|", markersize=8.0, label="Mean +/- SD"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.535, 0.064),
        columnspacing=1.2,
        handletextpad=0.6,
    )

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
