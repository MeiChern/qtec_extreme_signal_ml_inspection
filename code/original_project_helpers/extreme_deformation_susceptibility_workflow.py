#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure_reorganize_extreme_deformation_susceptibity.py
# Renamed package path: code/original_project_helpers/extreme_deformation_susceptibility_workflow.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import FuncFormatter
from pyproj import CRS, Transformer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import Figure_reorganize_du_du_gradient_ml_features as figml
import Figure_reorganize_pf_vs_npf_in_du_and_gradient as figpfnpf
import Figure_reorganize_railway_buffer_analysis as railbuf
import Figure_reorganized_railway_extreme_deformation_inspection as railinspect
import figure4_regional_deformation_context as fig4
import figure6_susceptibility_stacked as fig6
from submission_figure_style import add_panel_label as add_submission_panel_label
from submission_figure_style import apply_style as apply_submission_style
from submission_figure_style import EXPORT_DPI
from submission_figure_style import FONT

SEED = 42
CHUNKSIZE = fig6.CHUNKSIZE
FIG_BASENAME = "Figure_reorganize_extreme_deformation_susceptibity"
DEFAULT_SAMPLE_PF = 90_000
DEFAULT_SAMPLE_NPF = 90_000
MODEL_ARTIFACT_VERSION = 3
PCA_EXPLAINED_VARIANCE = 0.95
NEGATIVE_DU_ONLY = True
BASE_FONT_SIZE = 11.2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 0.9
PANEL_LABEL_FONT_SIZE = BASE_FONT_SIZE + 1.8
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE - 1.2
LEGEND_FONT_SIZE = BASE_FONT_SIZE - 1.6
DOMAIN_LABEL_FONT_SIZE = BASE_FONT_SIZE - 0.2
FEATURE_LABEL_FONT_SIZE = BASE_FONT_SIZE - 1.2
COLORBAR_TICK_FONT_SIZE = BASE_FONT_SIZE - 2.0
ROC_TICK_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ROC_TICK_LABELS = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
MODEL_LEGEND_FACE = (0.10, 0.10, 0.10, 0.55)
MODEL_LEGEND_EDGE = (0.32, 0.32, 0.32, 0.55)
TOP_PROFILE_YLIM = (-0.03, 0.26)
TOP_PROFILE_TICKS = [0.0, 0.10, 0.20]
SIDE_PROFILE_XLIM = (-0.03, 0.21)
SIDE_PROFILE_TICKS = [0.0, 0.10, 0.20]
IMPORTANCE_PLOT_EXCLUDE_FEATURES = {"dem__gmag"}
DOMAIN_ORDER = ("pf", "npf")
DOMAIN_LABELS = {"pf": "Permafrost", "npf": "Non-permafrost"}
DOMAIN_BG = {
    "pf": railbuf.blend_with_white(figpfnpf.PF_COLOR, 0.82),
    "npf": railbuf.blend_with_white(figpfnpf.NPF_COLOR, 0.82),
}
PROBABILITY_ZOOM_SPECS = (
    {
        "site_label": "Wudaoliang",
        "bounds": (0.02, 0.75, 0.25, 0.23),
        "arrow_side": "right",
        "xpad": 0.42,
        "ypad": 0.24,
    },
    {
        "site_label": "Tuotuohe",
        "bounds": (0.63, 0.20, 0.25, 0.23),
        "arrow_side": "left",
        "xpad": 0.34,
        "ypad": 0.22,
    },
)

RAW_FEATURE_LABELS = {
    **fig6.FEATURE_LABELS,
    "temperature_mean": "MAAT",
}

TARGET_META = {
    "d_u": {
        "label": r"$d_u$",
        "unit": "mm/yr",
        "theme_color": railbuf.DU_BASE_COLOR,
        "panel_labels": (" ", "C", "E"),
        "table_stub": "du",
        "profile_title": r"Predicted extreme $d_u$ susceptibility",
        "roc_title": r"Extreme $d_u$ Presence Predictor ROC",
        "threshold_pf": railbuf.PERMAFROST_EXTREME_DU_THRESHOLD,
        "threshold_npf": railbuf.NON_PERMAFROST_EXTREME_DU_THRESHOLD,
        "prob_cmap": "Blues",
    },
    "grad_mag_km": {
        "label": r"$|\nabla d_u|$",
        "unit": "mm/yr/km",
        "theme_color": railbuf.GRAD_BASE_COLOR,
        "panel_labels": (" ", "D", "F"),
        "table_stub": "grad",
        "profile_title": r"Predicted extreme $|\nabla d_u|$ susceptibility",
        "roc_title": r"Extreme $|\nabla d_u|$ Presence Predictor ROC",
        "threshold_pf": railbuf.PERMAFROST_EXTREME_GRAD_THRESHOLD,
        "threshold_npf": railbuf.NON_PERMAFROST_EXTREME_GRAD_THRESHOLD,
        "prob_cmap": "Reds",
    },
}

apply_submission_style()
plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 1.0,
        "ytick.labelsize": BASE_FONT_SIZE - 1.0,
    }
)


def log_step(message: str) -> None:
    print(f"[{FIG_BASENAME}] {message}")


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - blend) * rgb + blend)


def add_panel_label(ax, label: str, *, x: float = -0.12, y: float | None = None) -> None:
    if y is None:
        y = 1.04
    add_submission_panel_label(ax, label, x=x, y=y)


def feature_display_name(feature: str) -> str:
    if feature == "Perma_Distr_map":
        return "Permafrost flag"
    if feature.endswith("__lstd"):
        base = feature[:-6]
        return f"{RAW_FEATURE_LABELS.get(base, base)} (lstd)"
    if feature.endswith("__gmag"):
        base = feature[:-6]
        return f"{RAW_FEATURE_LABELS.get(base, base)} (gmag)"
    return RAW_FEATURE_LABELS.get(feature, feature)


def select_importance_rows_for_plot(importance_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if importance_df.empty:
        return importance_df.copy()
    keep = ~importance_df["feature"].astype(str).isin(IMPORTANCE_PLOT_EXCLUDE_FEATURES)
    return importance_df.loc[keep].head(top_n).copy()

def choose_label_holdout_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    test_size: float,
    seed: int = SEED,
    max_tries: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    groups = df["block_id"].astype(str).to_numpy()
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise RuntimeError("Need at least two spatial blocks for train/test splitting.")

    target_test_n = float(test_size) * len(df)
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score = np.inf

    for offset in range(max_tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(df, y, groups))
        if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
            continue
        score = abs(len(test_idx) - target_test_n)
        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)
        if score <= 0.01 * len(df):
            break

    if best_split is None:
        raise RuntimeError("Failed to find a valid grouped holdout split with both classes present.")
    return best_split


def build_feature_sets(
    *,
    exclude_magt: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    raw_features = fig6.RAW_FEATURES.copy()
    transition_base_vars = fig6.TRANSITION_BASE_VARS.copy()
    if exclude_magt:
        raw_features = [feature for feature in raw_features if feature != "magt"]
        transition_base_vars = [feature for feature in transition_base_vars if feature != "magt"]

    contrast_features: list[str] = []
    for var in transition_base_vars:
        contrast_features.extend([f"{var}__lstd", f"{var}__gmag"])
    combined_features = raw_features + contrast_features
    return raw_features, contrast_features, combined_features, transition_base_vars


def feature_origin(feature: str) -> str:
    return "contrast" if feature.endswith("__lstd") or feature.endswith("__gmag") else "raw"


def target_value_column(target: str) -> str:
    if target == "d_u":
        return "d_u"
    if target == "grad_mag_km":
        return "grad_mag_km"
    raise ValueError(f"Unsupported target: {target}")


def threshold_for_domain(target: str, domain: str) -> float:
    meta = TARGET_META[target]
    if domain == "pf":
        return float(meta["threshold_pf"])
    if domain == "npf":
        return float(meta["threshold_npf"])
    raise ValueError(f"Unsupported domain: {domain}")


def build_domain_label(df: pd.DataFrame, *, target: str, domain: str) -> pd.Series:
    values = pd.to_numeric(df[target_value_column(target)], errors="coerce").to_numpy(dtype=float)
    out = np.full(len(df), np.nan, dtype=float)
    valid = np.isfinite(values)
    out[valid] = 0.0
    threshold = threshold_for_domain(target, domain)
    if target == "d_u":
        out[valid & (values < threshold)] = 1.0
    elif target == "grad_mag_km":
        out[valid & (values > threshold)] = 1.0
    else:
        raise ValueError(f"Unsupported target: {target}")
    return pd.Series(out, index=df.index, name=f"is_extreme_{target}_{domain}")


def negative_du_mask(df: pd.DataFrame) -> np.ndarray:
    du = pd.to_numeric(df["d_u"], errors="coerce").to_numpy(dtype=float)
    if NEGATIVE_DU_ONLY:
        return np.isfinite(du) & (du < 0.0)
    return np.isfinite(du)


def valid_target_mask(df: pd.DataFrame, *, target: str) -> np.ndarray:
    values = pd.to_numeric(df[target_value_column(target)], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(values) & negative_du_mask(df)


def resolve_target_sample(
    *,
    csv_path: Path,
    cache_path: Path,
    sample_pf: int,
    sample_npf: int,
    target: str,
    chunksize: int,
) -> pd.DataFrame:
    if cache_path.exists():
        log_step(f"Loading cached susceptibility sample for {target}: {cache_path}")
        return fig6.engineer_features(pd.read_csv(cache_path))

    usecols = [
        "Perma_Distr_map",
        "easting",
        "northing",
        "longitude",
        "latitude",
        "grad_mag",
        "d_u",
    ] + fig6.RAW_FEATURES

    rng = np.random.default_rng(SEED)
    reservoirs = {
        "pf": pd.DataFrame(columns=usecols + ["_sample_key"]),
        "npf": pd.DataFrame(columns=usecols + ["_sample_key"]),
    }
    quotas = {"pf": int(sample_pf), "npf": int(sample_npf)}

    log_step(f"Building one-pass stratified sample for {target}")
    for chunk in fig6.iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label=f"sample build {target}",
    ):
        chunk = fig6.engineer_features(chunk)
        chunk = chunk.loc[valid_target_mask(chunk, target=target)].copy()
        if chunk.empty:
            continue

        for domain, quota in quotas.items():
            sub = chunk.loc[chunk["domain"].eq(domain)].copy()
            if sub.empty:
                continue
            sub["_sample_key"] = rng.random(len(sub))
            keep = pd.concat([reservoirs[domain], sub], ignore_index=True)
            if len(keep) > quota:
                keep = keep.nsmallest(quota, "_sample_key").reset_index(drop=True)
            reservoirs[domain] = keep

    out_parts: list[pd.DataFrame] = []
    for domain, quota in quotas.items():
        sub = reservoirs[domain].drop(columns="_sample_key", errors="ignore").copy()
        if len(sub) < quota:
            log_step(f"Requested {quota} rows for {target}/{domain}, retained {len(sub)} available rows")
        if sub.empty:
            raise RuntimeError(f"Sampling returned no rows for target={target}, domain={domain}.")
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False, compression="gzip")
    log_step(f"Saved sampled dataset for {target}: {cache_path}")
    return out


def build_target_cache_signature(
    *,
    target: str,
    domain: str,
    csv_path: Path,
    sample_cache: Path,
    combined_features: list[str],
    transition_base_vars: list[str],
    transition_window_size: int,
    neighbors_trans: int,
    block_size_m: float,
    max_missing_frac: float,
    model_n_jobs: int,
    test_size: float,
) -> dict[str, object]:
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "seed": SEED,
        "target": target,
        "domain": domain,
        "csv": fig6.file_signature(csv_path),
        "sample_cache": fig6.file_signature(sample_cache),
        "combined_features": list(combined_features),
        "transition_base_vars": list(transition_base_vars),
        "transition_window_size": int(transition_window_size),
        "neighbors_trans": int(neighbors_trans),
        "block_size_m": float(block_size_m),
        "max_missing_frac": float(max_missing_frac),
        "model_n_jobs": int(model_n_jobs),
        "test_size": float(test_size),
        "threshold": threshold_for_domain(target, domain),
        "pca_explained_variance": float(PCA_EXPLAINED_VARIANCE),
        "negative_du_only": bool(NEGATIVE_DU_ONLY),
        "base_learners": [name for name in ["RF", "ET", "HGB", "XGB"] if name != "XGB" or fig6.HAVE_XGB],
        "figure6_model_artifact_version": int(fig6.MODEL_ARTIFACT_VERSION),
        "xgboost_available": bool(fig6.HAVE_XGB),
    }


def load_cached_target_artifacts(
    cache_path: Path,
    cache_signature: dict[str, object],
) -> dict[str, object] | None:
    if not cache_path.exists():
        return None

    try:
        payload = joblib.load(cache_path)
    except Exception as exc:
        log_step(f"Warning: failed to load model cache {cache_path}: {exc}")
        return None

    required_keys = {
        "cache_signature",
        "target",
        "domain",
        "y_test",
        "suite_metrics",
        "suite_probs",
        "full_stack",
        "importance_df",
        "missing_df",
        "feature_names",
        "feature_groups",
        "dropped_features",
        "positive_fraction",
        "n_rows",
        "n_train",
        "n_test",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        log_step(f"Warning: model cache at {cache_path} is incomplete; retraining.")
        return None
    if payload["cache_signature"] != cache_signature:
        log_step(f"Model cache mismatch at {cache_path}; retraining.")
        return None
    return payload


def save_target_artifacts(cache_path: Path, payload: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=("gzip", 3))


def raster_sidecar_path(prob_path: Path) -> Path:
    return prob_path.with_suffix(prob_path.suffix + ".json")


def build_raster_cache_signature(
    *,
    kind: str,
    target: str,
    domain: str | None,
    csv_path: Path,
    model_cache_paths: dict[str, Path] | None,
    feature_names: list[str] | dict[str, list[str]],
    transition_base_vars: list[str],
    transition_window_size: int,
    neighbors_trans: int,
    grid: dict,
) -> dict[str, object]:
    if isinstance(feature_names, dict):
        feature_payload = {key: list(value) for key, value in feature_names.items()}
    else:
        feature_payload = list(feature_names)
    model_payload = (
        {key: fig6.file_signature(path) for key, path in model_cache_paths.items()}
        if model_cache_paths is not None
        else None
    )
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "kind": kind,
        "target": target,
        "domain": domain,
        "csv": fig6.file_signature(csv_path),
        "model_caches": model_payload,
        "feature_names": feature_payload,
        "transition_base_vars": list(transition_base_vars),
        "transition_window_size": int(transition_window_size),
        "neighbors_trans": int(neighbors_trans),
        "negative_du_only": bool(NEGATIVE_DU_ONLY),
        "grid": {
            "nrows": int(grid["nrows"]),
            "ncols": int(grid["ncols"]),
            "res": float(grid["res"]),
            "gx0": int(grid["gx0"]),
            "gy1": int(grid["gy1"]),
        },
    }


def cached_raster_is_valid(prob_path: Path, cache_signature: dict[str, object]) -> bool:
    sidecar_path = raster_sidecar_path(prob_path)
    if not prob_path.exists() or not sidecar_path.exists():
        return False
    try:
        payload = json.loads(sidecar_path.read_text())
    except Exception:
        return False
    return payload.get("cache_signature") == cache_signature


def write_raster_sidecar(
    prob_path: Path,
    *,
    cache_signature: dict[str, object],
    n_predictions: int,
    row_min: int | None,
    row_max: int | None,
    col_min: int | None,
    col_max: int | None,
) -> None:
    sidecar_path = raster_sidecar_path(prob_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "cache_signature": cache_signature,
                "n_predictions": int(n_predictions),
                "row_range": [row_min, row_max],
                "col_range": [col_min, col_max],
            },
            indent=2,
        )
    )


def build_combined_domain_prediction_raster(
    *,
    csv_path: Path,
    prob_path: Path,
    domain_models: dict[str, object],
    domain_feature_names: dict[str, list[str]],
    grid: dict,
    metric_paths: dict[str, dict[str, Path]],
    vars_for_transition: list[str],
    raw_features: list[str],
    chunksize: int,
    cache_signature: dict[str, object] | None = None,
) -> Path:
    log_step(f"Building combined AOI susceptibility raster: {prob_path.name}")
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    shape = (nrows, ncols)
    prob_mm = fig6.open_memmap(prob_path, dtype="float32", mode="w+", shape=shape)
    prob_mm[:] = np.nan
    n_predictions = 0
    row_min: int | None = None
    row_max: int | None = None
    col_min: int | None = None
    col_max: int | None = None

    metric_maps = fig6.load_metric_memmaps(metric_paths, vars_for_transition, grid)
    usecols = list(dict.fromkeys(["Perma_Distr_map", "easting", "northing", "d_u"] + raw_features))

    for chunk in fig6.iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label=f"predict {prob_path.stem}",
    ):
        chunk = fig6.engineer_features(chunk)
        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        row, col = fig6.en_to_rc(
            e,
            n,
            res=float(grid["res"]),
            gx0=int(grid["gx0"]),
            gy1=int(grid["gy1"]),
        )
        ok = np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
        ok &= negative_du_mask(chunk)
        if not ok.any():
            continue

        feature_arrays = {
            feature: pd.to_numeric(chunk[feature], errors="coerce").to_numpy(dtype=np.float32)
            for feature in raw_features
        }
        for var in vars_for_transition:
            lstd = np.full(len(chunk), np.nan, dtype=np.float32)
            gmag = np.full(len(chunk), np.nan, dtype=np.float32)
            lstd[ok] = metric_maps[var]["lstd"][row[ok], col[ok]]
            gmag[ok] = metric_maps[var]["gmag"][row[ok], col[ok]]
            feature_arrays[f"{var}__lstd"] = lstd
            feature_arrays[f"{var}__gmag"] = gmag

        for domain in DOMAIN_ORDER:
            domain_mask = ok & chunk["domain"].eq(domain).to_numpy()
            if not np.any(domain_mask):
                continue
            feature_names = domain_feature_names[domain]
            X_domain = pd.DataFrame(
                {feature: feature_arrays[feature] for feature in feature_names},
                columns=feature_names,
                dtype=np.float32,
            )
            domain_idx = np.flatnonzero(domain_mask)
            probs = domain_models[domain].predict_proba(X_domain.iloc[domain_idx])[:, 1].astype(np.float32)
            prob_mm[row[domain_mask], col[domain_mask]] = probs
            n_predictions += int(domain_idx.size)
            row_vals = row[domain_mask]
            col_vals = col[domain_mask]
            row_min = int(row_vals.min()) if row_min is None else min(row_min, int(row_vals.min()))
            row_max = int(row_vals.max()) if row_max is None else max(row_max, int(row_vals.max()))
            col_min = int(col_vals.min()) if col_min is None else min(col_min, int(col_vals.min()))
            col_max = int(col_vals.max()) if col_max is None else max(col_max, int(col_vals.max()))

    prob_mm.flush()
    if cache_signature is not None:
        write_raster_sidecar(
            prob_path,
            cache_signature=cache_signature,
            n_predictions=n_predictions,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
        )
    return prob_path


def build_single_model_prediction_raster(
    *,
    csv_path: Path,
    prob_path: Path,
    model,
    feature_names: list[str],
    grid: dict,
    metric_paths: dict[str, dict[str, Path]],
    vars_for_transition: list[str],
    raw_features: list[str],
    chunksize: int,
    log_label: str,
    cache_signature: dict[str, object] | None = None,
) -> Path:
    log_step(f"Building full-domain susceptibility raster for {log_label}: {prob_path.name}")
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    shape = (nrows, ncols)
    prob_mm = fig6.open_memmap(prob_path, dtype="float32", mode="w+", shape=shape)
    prob_mm[:] = np.nan
    n_predictions = 0
    row_min: int | None = None
    row_max: int | None = None
    col_min: int | None = None
    col_max: int | None = None

    metric_maps = fig6.load_metric_memmaps(metric_paths, vars_for_transition, grid)
    usecols = list(dict.fromkeys(["Perma_Distr_map", "easting", "northing", "d_u"] + raw_features))

    for chunk in fig6.iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label=f"predict {prob_path.stem}",
    ):
        chunk = fig6.engineer_features(chunk)
        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        row, col = fig6.en_to_rc(
            e,
            n,
            res=float(grid["res"]),
            gx0=int(grid["gx0"]),
            gy1=int(grid["gy1"]),
        )
        ok = np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
        ok &= negative_du_mask(chunk)
        if not ok.any():
            continue

        feature_arrays = {
            feature: pd.to_numeric(chunk[feature], errors="coerce").to_numpy(dtype=np.float32)
            for feature in raw_features
        }
        for var in vars_for_transition:
            lstd = np.full(len(chunk), np.nan, dtype=np.float32)
            gmag = np.full(len(chunk), np.nan, dtype=np.float32)
            lstd[ok] = metric_maps[var]["lstd"][row[ok], col[ok]]
            gmag[ok] = metric_maps[var]["gmag"][row[ok], col[ok]]
            feature_arrays[f"{var}__lstd"] = lstd
            feature_arrays[f"{var}__gmag"] = gmag

        X_full = pd.DataFrame(
            {feature: feature_arrays[feature] for feature in feature_names},
            columns=feature_names,
            dtype=np.float32,
        )
        predict_idx = np.flatnonzero(ok)
        probs = model.predict_proba(X_full.iloc[predict_idx])[:, 1].astype(np.float32)
        prob_mm[row[ok], col[ok]] = probs
        n_predictions += int(predict_idx.size)
        row_vals = row[ok]
        col_vals = col[ok]
        row_min = int(row_vals.min()) if row_min is None else min(row_min, int(row_vals.min()))
        row_max = int(row_vals.max()) if row_max is None else max(row_max, int(row_vals.max()))
        col_min = int(col_vals.min()) if col_min is None else min(col_min, int(col_vals.min()))
        col_max = int(col_vals.max()) if col_max is None else max(col_max, int(col_vals.max()))

    prob_mm.flush()
    if cache_signature is not None:
        write_raster_sidecar(
            prob_path,
            cache_signature=cache_signature,
            n_predictions=n_predictions,
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
        )
    return prob_path


def summarize_probability_raster(
    *,
    prob_path: Path,
    grid: dict,
    source_crs,
    nrows: int,
    ncols: int,
    target_max_pixels: int,
) -> dict[str, object]:
    stride = fig4.choose_stride(nrows, ncols, target_max=target_max_pixels)
    prob_mm = fig4.open_memmap(prob_path, dtype="float32", mode="r", shape=(nrows, ncols))
    prob_ds = np.array(prob_mm[::stride, ::stride], copy=False)
    prob_plot = np.flipud(prob_ds)
    lon_plot, lat_plot = build_lonlat_mesh_from_grid(grid, stride=stride, source_crs=source_crs)

    finite = np.isfinite(lon_plot) & np.isfinite(lat_plot)
    if not finite.any():
        raise RuntimeError("Projected-to-lonlat grid transform did not produce any finite cells for plotting.")

    extent_lonlat = [
        float(np.nanmin(lon_plot[finite])),
        float(np.nanmax(lon_plot[finite])),
        float(np.nanmin(lat_plot[finite])),
        float(np.nanmax(lat_plot[finite])),
    ]

    lon_profile = summarize_spatial_band_profile_with_coords(prob_plot, lon_plot, band_axis=1, max_bins=60)
    lat_profile = summarize_spatial_band_profile_with_coords(prob_plot, lat_plot, band_axis=0, max_bins=60)
    return {
        "stride": stride,
        "prob_plot": prob_plot,
        "lon_plot": lon_plot,
        "lat_plot": lat_plot,
        "extent_lonlat": extent_lonlat,
        "lon_profile": lon_profile,
        "lat_profile": lat_profile,
    }


def build_lonlat_mesh_from_grid(
    grid: dict,
    *,
    stride: int,
    source_crs,
) -> tuple[np.ndarray, np.ndarray]:
    res = float(grid["res"])
    gx0 = int(grid["gx0"])
    gy1 = int(grid["gy1"])
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])

    row_idx = np.arange(0, nrows, stride, dtype=np.int64)
    col_idx = np.arange(0, ncols, stride, dtype=np.int64)
    easting = (gx0 + col_idx).astype(float) * res
    northing = (gy1 - row_idx).astype(float) * res
    e_grid, n_grid = np.meshgrid(easting, northing)

    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    lon_grid, lat_grid = transformer.transform(e_grid, n_grid)
    return np.flipud(np.asarray(lon_grid, dtype=float)), np.flipud(np.asarray(lat_grid, dtype=float))


def summarize_spatial_band_profile_with_coords(
    arr: np.ndarray,
    coord_grid: np.ndarray,
    *,
    band_axis: int,
    max_bins: int = 64,
) -> pd.DataFrame:
    vals = np.asarray(arr, dtype=float)
    coords = np.asarray(coord_grid, dtype=float)
    if vals.ndim != 2 or coords.shape != vals.shape:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    n_bands = int(vals.shape[band_axis])
    if n_bands == 0:
        return pd.DataFrame(columns=["coord", "mean", "std", "n"])

    coord_1d = np.nanmean(coords, axis=0 if band_axis == 1 else 1)
    n_bins = max(2, min(int(max_bins), n_bands))
    rows = []

    for idx in np.array_split(np.arange(n_bands), n_bins):
        if idx.size == 0:
            continue
        subset = vals[idx, :] if band_axis == 0 else vals[:, idx]
        finite = subset[np.isfinite(subset)]
        coord_vals = coord_1d[idx]
        coord_vals = coord_vals[np.isfinite(coord_vals)]
        if finite.size == 0 or coord_vals.size == 0:
            continue
        rows.append(
            {
                "coord": float(np.nanmean(coord_vals)),
                "mean": float(np.nanmean(finite)),
                "std": float(np.nanstd(finite)),
                "n": int(finite.size),
            }
        )

    return pd.DataFrame(rows, columns=["coord", "mean", "std", "n"])


def choose_scale_bar_anchor(extent_lonlat: list[float]) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = map(float, extent_lonlat)
    lon0 = xmin + 0.07 * (xmax - xmin)
    lat0 = ymin + 0.46 * (ymax - ymin)
    return lon0, lat0


def add_probability_zoom_insets(
    ax,
    *,
    raster_summary: dict[str, object],
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    cmap,
) -> None:
    for spec in PROBABILITY_ZOOM_SPECS:
        match = meteoro_sites.loc[meteoro_sites["site_label"].astype(str).eq(spec["site_label"])]
        if match.empty:
            continue
        site_row = match.iloc[0]
        inset_ax = ax.inset_axes(spec["bounds"])
        inset_ax.pcolormesh(
            raster_summary["lon_plot"],
            raster_summary["lat_plot"],
            np.ma.masked_invalid(raster_summary["prob_plot"]),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            rasterized=True,
        )
        figml.overlay_railway(inset_ax, railway_segments)
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
        railinspect.add_site_marker(
            inset_ax,
            site_row,
            show_label=True,
            marker_size=railinspect.METEORO_INSET_MARKER_SIZE,
        )

        arrow_end = inset_axes_fraction_to_parent_data(
            parent_ax=ax,
            inset_ax=inset_ax,
            x_frac=1.0 if spec["arrow_side"] == "right" else 0.0,
            y_frac=0.5,
        )
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


def inset_axes_fraction_to_parent_data(
    *,
    parent_ax,
    inset_ax,
    x_frac: float,
    y_frac: float,
) -> tuple[float, float]:
    display_xy = inset_ax.transAxes.transform((x_frac, y_frac))
    data_xy = parent_ax.transData.inverted().transform(display_xy)
    return float(data_xy[0]), float(data_xy[1])


def make_preprocessed_pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full")),
            ("clf", clf),
        ]
    )


def build_domain_models(
    *,
    pos_weight: float,
    stack_cv,
    n_jobs: int,
) -> dict[str, object]:
    models: dict[str, object] = {
        "RF": make_preprocessed_pipeline(
            RandomForestClassifier(
                n_estimators=350,
                random_state=SEED,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )
        ),
        "ET": make_preprocessed_pipeline(
            ExtraTreesClassifier(
                n_estimators=400,
                random_state=SEED,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )
        ),
        "HGB": make_preprocessed_pipeline(
            HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED,
            )
        ),
    }
    if fig6.HAVE_XGB:
        models["XGB"] = make_preprocessed_pipeline(
            fig6.XGBClassifier(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=SEED,
                scale_pos_weight=pos_weight,
            )
        )

    estimators = [(name.lower(), clone(model)) for name, model in models.items()]
    models["Stack"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2500,
            random_state=SEED,
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=stack_cv,
        n_jobs=n_jobs,
    )
    return models


def fit_and_evaluate_domain_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    groups_train: np.ndarray,
    model_n_jobs: int,
) -> tuple[dict[str, object], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    pos_rate = max(float(np.mean(y_train)), 1e-6)
    pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
    suite = build_domain_models(
        pos_weight=pos_weight,
        stack_cv=fig6.FixedGroupKFold(groups_train, n_splits=5),
        n_jobs=model_n_jobs,
    )

    fitted: dict[str, object] = {}
    metrics: dict[str, dict[str, float]] = {}
    probs: dict[str, np.ndarray] = {}

    for name, model in suite.items():
        est = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train, y_train)
        p = est.predict_proba(X_test)[:, 1]
        fitted[name] = est
        probs[name] = p
        metrics[name] = fig6.classification_metrics(y_test, p)

    return fitted, metrics, probs


def fit_domain_model(
    df: pd.DataFrame,
    *,
    target: str,
    domain: str,
    combined_features: list[str],
    block_size_m: float,
    test_size: float,
    max_missing_frac: float,
    model_n_jobs: int,
) -> dict[str, object]:
    work = df.loc[df["domain"].eq(domain)].copy().reset_index(drop=True)
    work["label"] = build_domain_label(work, target=target, domain=domain)
    work["block_id"] = fig6.make_spatial_block_id(work, block_size_m=block_size_m)
    work = work.loc[work["label"].notna() & work["block_id"].notna()].reset_index(drop=True)
    if len(work) < 1000:
        raise RuntimeError(f"Too few valid rows remain for target={target}, domain={domain}.")

    train_idx, test_idx = choose_label_holdout_split(
        work,
        label_col="label",
        test_size=test_size,
    )
    train_df = work.iloc[train_idx].reset_index(drop=True)
    test_df = work.iloc[test_idx].reset_index(drop=True)

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise RuntimeError(f"Single-class grouped split encountered for target={target}, domain={domain}.")

    feature_groups = {
        "raw": [feature for feature in combined_features if feature_origin(feature) == "raw"],
        "contrast": [feature for feature in combined_features if feature_origin(feature) == "contrast"],
    }
    missing_df = fig6.audit_feature_missingness(train_df, feature_groups)
    missing_df["origin"] = missing_df["feature"].map(feature_origin)
    dropped_features = sorted(
        missing_df.loc[missing_df["missing_frac"] > max_missing_frac, "feature"].drop_duplicates().tolist()
    )
    model_features = [feature for feature in combined_features if feature not in dropped_features]
    if not model_features:
        raise RuntimeError(f"No predictors remain for target={target}, domain={domain}.")

    X_train = fig6.make_model_frame(train_df, model_features)
    X_test = fig6.make_model_frame(test_df, model_features)
    groups_train = train_df["block_id"].astype(str).to_numpy()
    suite_fitted, suite_metrics, suite_probs = fit_and_evaluate_domain_suite(
        X_train,
        y_train,
        X_test,
        y_test,
        groups_train=groups_train,
        model_n_jobs=model_n_jobs,
    )
    del X_train
    gc.collect()

    y_full = work["label"].to_numpy(dtype=int)
    groups_full = work["block_id"].astype(str).to_numpy()
    X_full = fig6.make_model_frame(work, model_features)
    pos_rate = max(float(np.mean(y_full)), 1e-6)
    pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
    full_stack = build_domain_models(
        pos_weight=pos_weight,
        stack_cv=fig6.FixedGroupKFold(groups_full, n_splits=5),
        n_jobs=model_n_jobs,
    )["Stack"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_stack.fit(X_full, y_full)
    del X_full
    gc.collect()

    max_perm_rows = min(fig6.PERMUTATION_MAX_ROWS, len(X_test))
    perm_idx = np.arange(len(X_test))
    if len(perm_idx) > max_perm_rows:
        rng = np.random.default_rng(SEED)
        perm_idx = np.sort(rng.choice(perm_idx, size=max_perm_rows, replace=False))

    perm = permutation_importance(
        suite_fitted["Stack"],
        X_test.iloc[perm_idx],
        y_test[perm_idx],
        n_repeats=fig6.PERMUTATION_REPEATS,
        random_state=SEED,
        scoring="average_precision",
        n_jobs=model_n_jobs,
    )
    importance_df = (
        pd.DataFrame(
            {
                "feature": model_features,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .assign(origin=lambda frame: frame["feature"].map(feature_origin))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "target": target,
        "domain": domain,
        "train_df": train_df,
        "test_df": test_df,
        "y_test": y_test,
        "suite_metrics": suite_metrics,
        "suite_probs": suite_probs,
        "full_stack": full_stack,
        "importance_df": importance_df,
        "missing_df": missing_df,
        "feature_names": model_features,
        "feature_groups": {
            "raw": [feature for feature in model_features if feature_origin(feature) == "raw"],
            "contrast": [feature for feature in model_features if feature_origin(feature) == "contrast"],
        },
        "dropped_features": dropped_features,
        "positive_fraction": float(np.mean(y_full)),
        "n_rows": int(len(work)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }

def plot_domain_roc_axis(
    ax,
    *,
    target: str,
    domain: str,
    domain_result: dict[str, object],
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    meta = TARGET_META[target]
    base_color = meta["theme_color"]
    y_true = np.asarray(domain_result["y_test"], dtype=int)
    probs = domain_result["suite_probs"]
    metrics = domain_result["suite_metrics"]

    model_order = [name for name in ["RF", "ET", "HGB", "XGB", "Stack"] if name in probs]
    submodels = [name for name in model_order if name != "Stack"]
    blends = np.linspace(0.80, 0.42, max(len(submodels), 1))

    ax.set_facecolor(DOMAIN_BG[domain])
    ax.grid(False)
    for name, blend in zip(submodels, blends, strict=False):
        fpr, tpr, _ = roc_curve(y_true, probs[name])
        ax.plot(
            fpr,
            tpr,
            color=blend_with_white(base_color, float(blend)),
            linewidth=1.2,
            alpha=0.97,
            label=f"{name}: {metrics[name]['roc_auc']:.2f}",
        )

    if "Stack" in probs:
        fpr, tpr, _ = roc_curve(y_true, probs["Stack"])
        ax.plot(
            fpr,
            tpr,
            color=base_color,
            linewidth=2.8,
            alpha=1.0,
            label=f"Stack: {metrics['Stack']['roc_auc']:.2f}",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(ROC_TICK_VALUES)
    ax.set_xticklabels(ROC_TICK_LABELS)
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate" if show_ylabel else "", fontweight="bold")
    if not show_ylabel:
        ax.set_yticklabels([])
    title = f"{meta['roc_title']}\n{DOMAIN_LABELS[domain]}" if domain == "pf" else DOMAIN_LABELS[domain]
    ax.set_title(title, fontweight="bold", loc="left" if domain == "pf" else "center")
    if show_legend:
        legend = ax.legend(frameon=True, loc="lower right", fontsize=LEGEND_FONT_SIZE)
        legend.get_frame().set_facecolor(MODEL_LEGEND_FACE)
        legend.get_frame().set_edgecolor(MODEL_LEGEND_EDGE)
        for text in legend.get_texts():
            text.set_color("0.96")
    fig4.style_open_axes(ax)


def plot_split_roc_panel(fig, parent_spec, *, result: dict[str, object], panel_label: str) -> list[object]:
    gs = parent_spec.subgridspec(1, 2, wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for idx, domain in enumerate(DOMAIN_ORDER):
        plot_domain_roc_axis(
            axes[idx],
            target=result["target"],
            domain=domain,
            domain_result=result["domains"][domain],
            show_ylabel=idx == 0,
            show_legend=True,
        )
    add_panel_label(axes[0], panel_label)
    return axes


def plot_circular_importance_panel(
    fig,
    parent_spec,
    *,
    result: dict[str, object],
    panel_label: str,
    top_n: int = 5,
) -> object:
    meta = TARGET_META[result["target"]]
    theme = meta["theme_color"]
    raw_face = blend_with_white(theme, 0.56)
    raw_edge = blend_with_white(theme, 0.34)
    contrast_face = blend_with_white(theme, 0.78)
    contrast_edge = blend_with_white(theme, 0.50)

    ax = fig.add_subplot(parent_spec, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0.0, 1.42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    ax.bar(np.pi, 1.42, width=np.pi, bottom=0.0, color=DOMAIN_BG["pf"], edgecolor="none", alpha=0.95, zorder=0)
    ax.bar(0.0, 1.42, width=np.pi, bottom=0.0, color=DOMAIN_BG["npf"], edgecolor="none", alpha=0.95, zorder=0)

    top_by_domain = {
        domain: select_importance_rows_for_plot(result["domains"][domain]["importance_df"], top_n=top_n)
        for domain in DOMAIN_ORDER
    }
    inner_radius = 0.28
    radial_span = 0.82

    angle_specs = {
        "pf": np.deg2rad(np.linspace(115.0, 245.0, max(len(top_by_domain["pf"]), 1))),
        "npf": np.deg2rad(np.linspace(-65.0, 65.0, max(len(top_by_domain["npf"]), 1))),
    }

    for domain in DOMAIN_ORDER:
        top = top_by_domain[domain]
        if top.empty:
            continue
        domain_max_importance = max(float(top["importance"].max()), 1e-6)
        theta_vals = angle_specs[domain][: len(top)]
        width = np.deg2rad(18.0 if len(top) <= 5 else 14.0)

        for theta, row in zip(theta_vals, top.itertuples(index=False), strict=False):
            height = radial_span * max(float(row.importance), 0.0) / domain_max_importance
            is_contrast = str(row.origin) == "contrast"
            ax.bar(
                theta,
                height,
                width=width,
                bottom=inner_radius,
                color=contrast_face if is_contrast else raw_face,
                edgecolor=contrast_edge if is_contrast else raw_edge,
                linewidth=1.1,
                hatch=".." if is_contrast else None,
                zorder=3,
            )
            label_radius = inner_radius + 0.06
            angle_deg = (np.degrees(theta) + 360.0) % 360.0
            rotation = angle_deg
            ha = "left"
            if 90.0 < angle_deg < 270.0:
                rotation += 180.0
                ha = "right"
            ax.text(
                theta,
                label_radius,
                feature_display_name(str(row.feature)),
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=FEATURE_LABEL_FONT_SIZE,
                fontweight="bold",
                color="0.18",
            )
    ax.text(
        -0.10,
        0.50,
        f"Top {top_n} predictors",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color="0.12",
    )
    fill_legend = ax.legend(
        handles=[
            Patch(facecolor=raw_face, edgecolor=raw_edge, linewidth=1.0, label="Raw"),
            Patch(facecolor=contrast_face, edgecolor=contrast_edge, linewidth=1.0, hatch="..", label="Contrast"),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.06, 0.64),
        ncol=1,
    )
    ax.add_artist(fill_legend)
    ax.legend(
        handles=[
            Patch(facecolor=DOMAIN_BG["pf"], edgecolor="none", label=DOMAIN_LABELS["pf"]),
            Patch(facecolor=DOMAIN_BG["npf"], edgecolor="none", label=DOMAIN_LABELS["npf"]),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.06, 0.36),
        ncol=1,
    )
    add_panel_label(ax, panel_label, x=-0.08)
    return ax


def plot_probability_triptych(
    fig,
    parent_spec,
    *,
    result: dict,
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    source_crs,
    nrows: int,
    ncols: int,
    target_max_pixels: int,
    panel_label: str,
) -> tuple[object, object, object]:
    meta = TARGET_META[result["target"]]
    color = meta["theme_color"]
    ax_top, ax_map, ax_side = fig4.make_composite_triptych(fig, parent_spec, map_width=4.4, side_width=1.85)

    raster_summary = summarize_probability_raster(
        prob_path=result["prob_path"],
        grid=result["grid"],
        source_crs=source_crs,
        nrows=nrows,
        ncols=ncols,
        target_max_pixels=target_max_pixels,
    )
    extent_lonlat = raster_summary["extent_lonlat"]

    cmap = plt.get_cmap(meta["prob_cmap"]).copy()
    cmap.set_bad(alpha=0.0)
    # The susceptibility raster lives in projected Albers coordinates. Plotting it
    # against per-cell lon/lat rasters preserves alignment with lon/lat overlays.
    im = ax_map.pcolormesh(
        raster_summary["lon_plot"],
        raster_summary["lat_plot"],
        np.ma.masked_invalid(raster_summary["prob_plot"]),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        rasterized=True,
    )
    fig4.format_lonlat_axes(
        ax_map,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=False,
        fill_plot_region=True,
    )
    ax_map.set_xlim(extent_lonlat[0], extent_lonlat[1])
    figml.apply_requested_map_lat_axis(ax_map)
    figml.overlay_railway(ax_map, railway_segments)
    railinspect.add_meteoro_sites(ax_map, meteoro_sites)
    scale_bar_lon, scale_bar_lat = choose_scale_bar_anchor(extent_lonlat)
    figml.add_scale_bar_lonlat(ax_map, lon0=scale_bar_lon, lat0=scale_bar_lat, length_km=100.0)
    figml.add_railway_legend(ax_map)
    cax = ax_map.inset_axes([0.67, 0.125, 0.20, 0.032], transform=ax_map.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, length=2, pad=1)
    cb.set_ticks([0.05, 0.95])
    cb.set_ticklabels(["0.05", "0.95"])
    add_probability_zoom_insets(
        ax_map,
        raster_summary=raster_summary,
        railway_segments=railway_segments,
        meteoro_sites=meteoro_sites,
        cmap=cmap,
    )

    fig4.plot_top_profile(
        ax_top,
        raster_summary["lon_profile"],
        color=color,
        ylabel="Mean\nprob.",
        title=meta["profile_title"],
        coord_limits=(extent_lonlat[0], extent_lonlat[1]),
    )
    fig4.use_right_y_axis(ax_top, color=color)
    figml.center_multiline_right_ylabel(ax_top)
    ax_top.set_ylim(*TOP_PROFILE_YLIM)
    ax_top.set_yticks(TOP_PROFILE_TICKS)
    ax_top.set_yticklabels(["0", "0.10", "0.20"])

    fig4.plot_side_profile(
        ax_side,
        raster_summary["lat_profile"],
        color=color,
        xlabel="Mean\nprob.",
        coord_limits=(extent_lonlat[2], extent_lonlat[3]),
    )
    ax_side.set_xlim(*SIDE_PROFILE_XLIM)
    ax_side.set_xticks(SIDE_PROFILE_TICKS)
    ax_side.set_xticklabels(["0", "0.10", "0.20"])

    fig4.use_map_spines_for_marginals(ax_top, ax_map, ax_side)
    add_panel_label(ax_top, panel_label, x=-0.06)
    return ax_top, ax_map, ax_side


def style_horizontal_colorbar(
    cb,
    *,
    label: str,
    ticks: list[float] | None = None,
    ticklabels: list[str] | None = None,
) -> None:
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("0.4")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE, length=2, pad=1)
    cb.set_label(label, fontsize=BASE_FONT_SIZE - 2.0, fontweight="bold", labelpad=1)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)


def add_inset_horizontal_colorbar(
    ax,
    mappable,
    *,
    label: str,
    rect: list[float],
    ticks: list[float] | None = None,
    ticklabels: list[str] | None = None,
) -> None:
    cax = ax.inset_axes(rect, transform=ax.transAxes)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.88))
    cax.set_zorder(5)
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    style_horizontal_colorbar(cb, label=label, ticks=ticks, ticklabels=ticklabels)


def plot_case_roc_axis(
    ax,
    *,
    target: str,
    domain: str,
    domain_result: dict[str, object],
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    meta = TARGET_META[target]
    base_color = meta["theme_color"]
    y_true = np.asarray(domain_result["y_test"], dtype=int)
    probs = domain_result["suite_probs"]
    metrics = domain_result["suite_metrics"]

    model_order = [name for name in ["RF", "ET", "HGB", "XGB", "Stack"] if name in probs]
    submodels = [name for name in model_order if name != "Stack"]
    blends = np.linspace(0.80, 0.42, max(len(submodels), 1))

    ax.set_facecolor(DOMAIN_BG[domain])
    ax.grid(False)
    for name, blend in zip(submodels, blends, strict=False):
        fpr, tpr, _ = roc_curve(y_true, probs[name])
        ax.plot(
            fpr,
            tpr,
            color=blend_with_white(base_color, float(blend)),
            linewidth=1.2,
            alpha=0.97,
            label=f"{name} ({metrics[name]['roc_auc']:.2f})",
        )

    if "Stack" in probs:
        fpr, tpr, _ = roc_curve(y_true, probs["Stack"])
        ax.plot(
            fpr,
            tpr,
            color=base_color,
            linewidth=2.8,
            alpha=1.0,
            label=f"Stack ({metrics['Stack']['roc_auc']:.2f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(ROC_TICK_VALUES)
    ax.set_xticklabels(ROC_TICK_LABELS)
    ax.set_xlabel("False positive rate", fontweight="bold", labelpad=2)
    ax.set_ylabel("True positive rate" if show_ylabel else "", fontweight="bold", labelpad=2)
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    title = ax.set_title(
        rf"Extreme {meta['label']}" + f"\n{DOMAIN_LABELS[domain]}",
        fontweight="bold",
        pad=3,
    )
    title.set_linespacing(0.92)
    if show_legend:
        legend = ax.legend(
            title="ROC-AUC",
            frameon=True,
            loc="lower right",
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )
        legend.get_frame().set_facecolor(MODEL_LEGEND_FACE)
        legend.get_frame().set_edgecolor(MODEL_LEGEND_EDGE)
        for text in legend.get_texts():
            text.set_color("0.96")
        legend.get_title().set_color("0.96")
    fig4.style_open_axes(ax)


def plot_importance_bar_axis(
    ax,
    *,
    target: str,
    domain: str,
    domain_result: dict[str, object],
    top_n: int = 5,
    show_legend: bool = True,
    title: str | None = None,
    use_domain_bg: bool = True,
    show_xlabel: bool = True,
) -> None:
    meta = TARGET_META[target]
    theme = meta["theme_color"]
    raw_face = blend_with_white(theme, 0.56)
    raw_edge = blend_with_white(theme, 0.34)
    contrast_face = blend_with_white(theme, 0.78)
    contrast_edge = blend_with_white(theme, 0.50)

    ax.set_facecolor(DOMAIN_BG[domain] if use_domain_bg else "white")
    ax.grid(False)

    top = select_importance_rows_for_plot(domain_result["importance_df"], top_n=top_n).iloc[::-1].copy()
    if top.empty:
        ax.text(
            0.5,
            0.5,
            "No ranked predictors",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=BASE_FONT_SIZE,
            fontweight="bold",
            color="0.25",
        )
        fig4.style_open_axes(ax)
        return

    y_pos = np.arange(len(top), dtype=float)
    label_effect = [pe.withStroke(linewidth=2.0, foreground=(1.0, 1.0, 1.0, 0.85))]
    for y, row in zip(y_pos, top.itertuples(index=False), strict=False):
        importance = max(float(row.importance), 0.0)
        std = float(row.importance_std) if np.isfinite(row.importance_std) else 0.0
        is_contrast = str(row.origin) == "contrast"
        ax.barh(
            y,
            importance,
            xerr=std if std > 0.0 else None,
            height=0.72,
            color=contrast_face if is_contrast else raw_face,
            edgecolor=contrast_edge if is_contrast else raw_edge,
            hatch="///" if is_contrast else None,
            linewidth=1.1,
            error_kw={"elinewidth": 0.9, "ecolor": "0.28", "capsize": 2.0},
            zorder=3,
        )
        x_max_for_text = max(float(top["importance"].max()), 1e-6)
        label_x = max(0.012 * x_max_for_text, min(0.035 * x_max_for_text, 0.18 * importance))
        txt = ax.text(
            label_x,
            y,
            feature_display_name(str(row.feature)),
            ha="left",
            va="center",
            fontsize=FEATURE_LABEL_FONT_SIZE + 0.3,
            fontweight="bold",
            color="0.12",
            zorder=4,
        )
        txt.set_path_effects(label_effect)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, labelleft=False, length=0)
    x_max = float(np.nanmax(top["importance"].to_numpy(dtype=float))) if len(top) else 0.0
    ax.set_xlim(0.0, max(0.02, 1.14 * x_max))
    if show_xlabel:
        ax.set_xlabel("Permutation importance", fontweight="bold", labelpad=2)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(title or f"Top {min(top_n, len(top))} predictors", fontweight="bold", pad=3)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: "0" if np.isclose(x, 0.0) else f"{x:.2f}".rstrip("0").rstrip("."))
    )
    if show_legend:
        legend = ax.legend(
            handles=[
                Patch(facecolor=raw_face, edgecolor=raw_edge, linewidth=1.0, label="Raw"),
                Patch(facecolor=contrast_face, edgecolor=contrast_edge, linewidth=1.0, hatch="///", label="Contrast"),
            ],
            loc="lower right",
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
        )
        legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.88))
        legend.get_frame().set_edgecolor("0.68")
    fig4.style_open_axes(ax)


def add_meteoro_sites_styled(ax, site_df: pd.DataFrame, *, show_labels: bool) -> None:
    if show_labels:
        figml.add_meteoro_sites(ax, site_df)
        return
    for site in site_df.itertuples(index=False):
        site_row = pd.Series({"site_label": site.site_label, "longitude": site.longitude, "latitude": site.latitude})
        figml.add_meteoro_site_marker(ax, site_row, show_label=False, marker_size=figml.METEORO_SITE_SIZE)


def plot_spatial_probability_map_axis(
    ax,
    *,
    plot_arr: np.ndarray,
    lon_plot: np.ndarray,
    lat_plot: np.ndarray,
    extent_lonlat: list[float],
    title: str,
    cmap,
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    show_xlabel: bool,
    show_ylabel: bool,
    show_site_labels: bool,
    show_scale_bar: bool,
    show_railway_legend: bool,
    add_zoom_insets: bool,
    add_colorbar: bool,
    colorbar_label: str,
    colorbar_rect: list[float] = figml.MAP_COLORBAR_RECT,
    colorbar_ticks: list[float] | None = None,
    colorbar_ticklabels: list[str] | None = None,
    panel_label: str | None = None,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    pcolor_kwargs: dict[str, object] = {}
    if norm is not None:
        pcolor_kwargs["norm"] = norm
    else:
        if vmin is not None:
            pcolor_kwargs["vmin"] = float(vmin)
        if vmax is not None:
            pcolor_kwargs["vmax"] = float(vmax)

    im = ax.pcolormesh(
        lon_plot,
        lat_plot,
        np.ma.masked_invalid(plot_arr),
        cmap=cmap,
        shading="auto",
        rasterized=True,
        **pcolor_kwargs,
    )
    fig4.format_lonlat_axes(
        ax,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        extent_lonlat=extent_lonlat,
        use_meridional_lat_ticks=False,
        fill_plot_region=True,
    )
    ax.set_xlim(extent_lonlat[0], extent_lonlat[1])
    figml.apply_requested_map_lat_axis(ax)
    if not show_xlabel:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Longitude")
    if not show_ylabel:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(title, fontweight="bold", pad=5)

    figml.overlay_railway(ax, railway_segments)
    add_meteoro_sites_styled(ax, meteoro_sites, show_labels=show_site_labels)
    if show_scale_bar:
        scale_bar_lon, scale_bar_lat = choose_scale_bar_anchor(extent_lonlat)
        figml.add_scale_bar_lonlat(ax, lon0=scale_bar_lon, lat0=scale_bar_lat, length_km=100.0)
    if show_railway_legend:
        figml.add_railway_legend(ax)
    if add_colorbar:
        add_inset_horizontal_colorbar(
            ax,
            im,
            label=colorbar_label,
            rect=colorbar_rect,
            ticks=colorbar_ticks,
            ticklabels=colorbar_ticklabels,
        )
    if add_zoom_insets:
        figml.add_map_zoom_insets(
            ax,
            plot_arr=np.asarray(plot_arr, dtype=float),
            lon_plot=lon_plot,
            lat_plot=lat_plot,
            railway_segments=railway_segments,
            meteoro_sites=meteoro_sites,
            cmap=cmap,
            pcolor_kwargs=pcolor_kwargs,
        )
    if panel_label is not None:
        figml.add_plain_panel_label(ax, panel_label)
    return im


def save_dual_format_figure(fig, *, fig_dir: Path, stem: str) -> tuple[Path, Path]:
    out_png = fig_dir / f"{stem}.png"
    out_pdf = fig_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=EXPORT_DPI)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def build_model_performance_figure(
    *,
    results: dict[str, dict[str, object]],
    fig_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 5.1),
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.12, wspace=0.16, hspace=0.34)

    row_domains = ("pf", "npf")
    col_targets = ("d_u", "grad_mag_km")
    panel_labels = list("ABCD")

    for row_idx, domain in enumerate(row_domains):
        strip_color = "#E8EFF5" if domain == "pf" else "#F4EEDF"
        strip_y = 0.90 if row_idx == 0 else 0.47
        fig.text(
            0.5,
            strip_y,
            DOMAIN_LABELS[domain],
            ha="center",
            va="center",
            fontsize=FONT["legend"],
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.22", facecolor=strip_color, edgecolor="none"),
        )

        for col_idx, target in enumerate(col_targets):
            ax = axes[row_idx, col_idx]
            domain_result = results[target]["domains"][domain]
            plot_importance_bar_axis(
                ax,
                target=target,
                domain=domain,
                domain_result=domain_result,
                top_n=5,
                show_legend=False,
                title=TARGET_META[target]["profile_title"].replace("Predicted extreme ", ""),
                use_domain_bg=False,
                show_xlabel=row_idx == len(row_domains) - 1,
            )
            auc = float(domain_result["suite_metrics"]["Stack"]["roc_auc"])
            ax.text(
                0.98,
                0.06,
                f"Stack ROC-AUC = {auc:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=FONT["annotation"],
                color="0.26",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=(1.0, 1.0, 1.0, 0.92), edgecolor="none"),
            )
            add_panel_label(ax, panel_labels[row_idx * len(col_targets) + col_idx], x=-0.10, y=1.03)
            fig4.apply_bold_nonlegend(ax)

    for col_idx in range(len(col_targets)):
        x_max = max(float(axes[row_idx, col_idx].get_xlim()[1]) for row_idx in range(len(row_domains)))
        for row_idx in range(len(row_domains)):
            axes[row_idx, col_idx].set_xlim(0.0, x_max)

    legend_handles = [
        Patch(
            facecolor=blend_with_white(TARGET_META["d_u"]["theme_color"], 0.56),
            edgecolor=blend_with_white(TARGET_META["d_u"]["theme_color"], 0.34),
            linewidth=1.0,
            label="Raw predictor",
        ),
        Patch(
            facecolor=blend_with_white(TARGET_META["d_u"]["theme_color"], 0.78),
            edgecolor=blend_with_white(TARGET_META["d_u"]["theme_color"], 0.50),
            linewidth=1.0,
            hatch="///",
            label="Spatial-contrast predictor",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=FONT["legend"],
    )
    legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.96))
    legend.get_frame().set_edgecolor("0.72")

    return save_dual_format_figure(
        fig,
        fig_dir=fig_dir,
        stem=f"{FIG_BASENAME}_model_performance",
    )


def build_predictor_domain_figure(
    *,
    results: dict[str, dict[str, object]],
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    source_crs,
    fig_dir: Path,
    nrows: int,
    ncols: int,
    target_max_pixels: int,
) -> tuple[Path, Path]:
    target_order = ["d_u", "grad_mag_km"]
    a4_landscape = getattr(figml, "A4_LANDSCAPE_FIGSIZE", (11.69, 8.27))
    a4_width = float(a4_landscape[0])
    colorbar_top_lat = 29.5
    colorbar_y0 = (
        (colorbar_top_lat - figml.MAP_LAT_LIMITS[0]) / (figml.MAP_LAT_LIMITS[1] - figml.MAP_LAT_LIMITS[0])
        - figml.MAP_COLORBAR_HEIGHT
    )
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    diff_arrays: list[np.ndarray] = []

    for target in target_order:
        summaries[target] = {}
        for domain in DOMAIN_ORDER:
            raster_summary = summarize_probability_raster(
                prob_path=results[target]["predictor_prob_paths"][domain],
                grid=results[target]["grid"],
                source_crs=source_crs,
                nrows=nrows,
                ncols=ncols,
                target_max_pixels=target_max_pixels,
            )
            summaries[target][domain] = raster_summary
        diff_plot = summaries[target]["pf"]["prob_plot"] - summaries[target]["npf"]["prob_plot"]
        diff_arrays.append(diff_plot[np.isfinite(diff_plot)])

    nonempty_diff_arrays = [vals for vals in diff_arrays if vals.size > 0]
    diff_values = np.concatenate(nonempty_diff_arrays) if nonempty_diff_arrays else np.array([])
    diff_lo, diff_hi = fig4.centered_clip(diff_values, center=0.0, p_lo=2.0, p_hi=98.0)
    if diff_lo >= 0.0:
        diff_lo = -1e-6
    if diff_hi <= 0.0:
        diff_hi = 1e-6
    diff_norm = TwoSlopeNorm(vmin=diff_lo, vcenter=0.0, vmax=diff_hi)

    fig = plt.figure(figsize=(a4_width, a4_width * 0.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.055,
        wspace=0.08,
        hspace=0.12,
    )
    axes = np.empty((2, 3), dtype=object)
    panel_labels = list("ABCDEF")

    column_titles = (
        "Permafrost-trained predictor",
        "Non-permafrost-trained predictor",
        "Permafrost minus non-permafrost",
    )
    row_colorbar_rect = [0.67, colorbar_y0, 0.20, figml.MAP_COLORBAR_HEIGHT]
    row_colorbar_ticks = [0.05, 0.95]
    row_colorbar_ticklabels = ["0.05", "0.95"]

    for row, target in enumerate(target_order):
        meta = TARGET_META[target]
        cmap = plt.get_cmap(meta["prob_cmap"]).copy()
        cmap.set_bad(alpha=0.0)
        pf_summary = summaries[target]["pf"]
        npf_summary = summaries[target]["npf"]
        diff_plot = pf_summary["prob_plot"] - npf_summary["prob_plot"]
        extent_lonlat = pf_summary["extent_lonlat"]

        for col, domain in enumerate(DOMAIN_ORDER):
            ax = fig.add_subplot(gs[row, col])
            axes[row, col] = ax
            summary = summaries[target][domain]
            im = plot_spatial_probability_map_axis(
                ax,
                plot_arr=summary["prob_plot"],
                lon_plot=summary["lon_plot"],
                lat_plot=summary["lat_plot"],
                extent_lonlat=extent_lonlat,
                title=column_titles[col] if row == 0 else "",
                cmap=cmap,
                railway_segments=railway_segments,
                meteoro_sites=meteoro_sites,
                show_xlabel=row == 1,
                show_ylabel=col == 0,
                show_site_labels=True,
                show_scale_bar=row == 0 and col == 0,
                show_railway_legend=row == 0 and col == 0,
                add_zoom_insets=False,
                add_colorbar=col == 1,
                colorbar_label=rf"Extreme {meta['label']} susceptibility",
                colorbar_rect=row_colorbar_rect,
                colorbar_ticks=row_colorbar_ticks,
                colorbar_ticklabels=row_colorbar_ticklabels,
                panel_label=panel_labels[row * 3 + col],
                vmin=0.0,
                vmax=1.0,
            )

        ax_diff = fig.add_subplot(gs[row, 2])
        axes[row, 2] = ax_diff
        plot_spatial_probability_map_axis(
            ax_diff,
            plot_arr=diff_plot,
            lon_plot=pf_summary["lon_plot"],
            lat_plot=pf_summary["lat_plot"],
            extent_lonlat=extent_lonlat,
            title=column_titles[2] if row == 0 else "",
            cmap="coolwarm",
            railway_segments=railway_segments,
            meteoro_sites=meteoro_sites,
            show_xlabel=row == 1,
            show_ylabel=False,
            show_site_labels=True,
            show_scale_bar=False,
            show_railway_legend=False,
            add_zoom_insets=False,
            add_colorbar=True,
            colorbar_label="Susceptibility difference",
            colorbar_rect=row_colorbar_rect,
            colorbar_ticks=[diff_lo, 0.0, diff_hi],
            colorbar_ticklabels=[
                f"{diff_lo:.2f}".rstrip("0").rstrip("."),
                "0",
                f"{diff_hi:.2f}".rstrip("0").rstrip("."),
            ],
            panel_label=panel_labels[row * 3 + 2],
            norm=diff_norm,
        )

        ax_diff.text(
            1.065,
            0.5,
            rf"Extreme {meta['label']}",
            transform=ax_diff.transAxes,
            rotation=270,
            ha="center",
            va="center",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            color="0.10",
            clip_on=False,
        )

    for ax in axes.ravel().tolist():
        fig4.apply_bold_nonlegend(ax)

    return save_dual_format_figure(
        fig,
        fig_dir=fig_dir,
        stem=f"{FIG_BASENAME}_predictor_domain_maps",
    )


def build_combined_domain_figure(
    *,
    results: dict[str, dict[str, object]],
    railway_segments: list[np.ndarray],
    meteoro_sites: pd.DataFrame,
    source_crs,
    fig_dir: Path,
    nrows: int,
    ncols: int,
    target_max_pixels: int,
) -> tuple[Path, Path]:
    target_order = ["d_u", "grad_mag_km"]
    a4_landscape = getattr(figml, "A4_LANDSCAPE_FIGSIZE", (11.69, 8.27))
    a4_width = float(a4_landscape[0])
    summaries: dict[str, dict[str, object]] = {}

    for target in target_order:
        summaries[target] = summarize_probability_raster(
            prob_path=results[target]["prob_path"],
            grid=results[target]["grid"],
            source_crs=source_crs,
            nrows=nrows,
            ncols=ncols,
            target_max_pixels=target_max_pixels,
        )

    fig = plt.figure(figsize=(a4_width, a4_width * 0.6), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.055,
        right=0.985,
        top=0.965,
        bottom=0.06,
        wspace=0.10,
    )
    all_axes: list[object] = []
    panel_labels = list("AB")

    for col, target in enumerate(target_order):
        meta = TARGET_META[target]
        summary = summaries[target]
        cmap = plt.get_cmap(meta["prob_cmap"]).copy()
        cmap.set_bad(alpha=0.0)
        ax = fig.add_subplot(gs[0, col])
        plot_spatial_probability_map_axis(
            ax,
            plot_arr=summary["prob_plot"],
            lon_plot=summary["lon_plot"],
            lat_plot=summary["lat_plot"],
            extent_lonlat=summary["extent_lonlat"],
            title=rf"Extreme {meta['label']} susceptibility",
            cmap=cmap,
            railway_segments=railway_segments,
            meteoro_sites=meteoro_sites,
            show_xlabel=True,
            show_ylabel=col == 0,
            show_site_labels=True,
            show_scale_bar=col == 0,
            show_railway_legend=col == 0,
            add_zoom_insets=True,
            add_colorbar=True,
            colorbar_label="Susceptibility",
            colorbar_ticks=[0.05, 0.95],
            colorbar_ticklabels=["0.05", "0.95"],
            panel_label=panel_labels[col],
            vmin=0.0,
            vmax=1.0,
        )
        all_axes.append(ax)

    for ax in all_axes:
        fig4.apply_bold_nonlegend(ax)

    return save_dual_format_figure(
        fig,
        fig_dir=fig_dir,
        stem=f"{FIG_BASENAME}_combined_domain_maps",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extreme deformation susceptibility for d_u and d_u gradient")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-pf", type=int, default=DEFAULT_SAMPLE_PF)
    parser.add_argument("--sample-npf", type=int, default=DEFAULT_SAMPLE_NPF)
    parser.add_argument("--neighbors-trans", type=int, default=21)
    parser.add_argument("--block-size-km", type=float, default=fig6.SPATIAL_BLOCK_SIZE_M / 1000.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-missing-frac", type=float, default=fig6.MAX_FEATURE_MISSING_FRAC)
    parser.add_argument("--model-n-jobs", type=int, default=fig6.DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--target-max-pixels", type=int, default=fig4.TARGET_MAX_PIXELS)
    parser.add_argument("--exclude-magt", action="store_true")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-raster", action="store_true")
    parser.add_argument("--railway-shp", type=Path, default=None)
    parser.add_argument("--meteoro-shp", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    )
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    table_dir = out_dir / "tables"
    model_cache_dir = (
        args.model_cache_dir.resolve()
        if args.model_cache_dir is not None
        else (cache_dir / "models")
    )
    for path in [fig_dir, cache_dir, table_dir, model_cache_dir]:
        path.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        csv_path = base_dir / "df_all_data_with_wright_du.csv"
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
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(str(path) for path in missing))

    model_n_jobs = max(1, int(args.model_n_jobs))
    block_size_m = float(args.block_size_km) * 1000.0
    raw_features, contrast_features, combined_features, transition_base_vars = build_feature_sets(
        exclude_magt=bool(args.exclude_magt)
    )
    sample_cache_tag = "negdu" if NEGATIVE_DU_ONLY else "all_du"
    sample_cache_by_target = {
        target: cache_dir / (
            f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{sample_cache_tag}_sample_pf{args.sample_pf}_npf{args.sample_npf}.csv.gz"
        )
        for target in TARGET_META
    }
    transition_cache_dir = cache_dir / f"{FIG_BASENAME}_transition_rasters"
    transition_window_size = fig6.neighbors_to_window_size(args.neighbors_trans)
    for target, sample_cache in sample_cache_by_target.items():
        if not sample_cache.exists():
            log_step(f"Creating sample cache for {target}")
            resolve_target_sample(
                csv_path=csv_path,
                cache_path=sample_cache,
                sample_pf=int(args.sample_pf),
                sample_npf=int(args.sample_npf),
                target=target,
                chunksize=int(args.chunksize),
            )

    grid = fig6.build_or_load_grid(base_dir, csv_path)

    railway_segments = figml.load_railway_segments(railway_shp)
    meteoro_sites = railinspect.load_meteoro_sites(meteoro_shp)
    sample_df_by_target: dict[str, pd.DataFrame] = {}
    metric_paths: dict[str, dict[str, Path]] | None = None
    railway_prj = railway_shp.with_suffix(".prj")
    if not railway_prj.exists():
        raise FileNotFoundError(f"Missing railway projection file: {railway_prj}")
    map_source_crs = CRS.from_wkt(railway_prj.read_text())

    def ensure_spatial_support() -> dict[str, dict[str, Path]]:
        nonlocal grid, metric_paths
        if metric_paths is None:
            log_step("Ensuring transition rasters are available")
            grid, _, metric_paths = fig6.ensure_transition_rasters(
                base_dir=base_dir,
                csv_path=csv_path,
                cache_dir=transition_cache_dir,
                vars_for_transition=transition_base_vars,
                chunksize=int(args.chunksize),
                window_size=transition_window_size,
            )
        return metric_paths

    def ensure_training_sample(target: str) -> pd.DataFrame:
        if target not in sample_df_by_target:
            log_step(f"Resolving stratified PF/NPF sample for {target}")
            sample_df = resolve_target_sample(
                csv_path=csv_path,
                cache_path=sample_cache_by_target[target],
                sample_pf=int(args.sample_pf),
                sample_npf=int(args.sample_npf),
                target=target,
                chunksize=int(args.chunksize),
            )
            log_step(f"Attaching transition metrics to sampled points for {target}")
            sample_df = fig6.attach_raster_transition_metrics(
                df=sample_df,
                metric_paths=ensure_spatial_support(),
                grid=grid,
                vars_for_transition=transition_base_vars,
            )
            sample_df_by_target[target] = sample_df
        return sample_df_by_target[target]

    results: dict[str, dict[str, object]] = {}
    perf_rows: list[dict[str, object]] = []

    for target in ["d_u", "grad_mag_km"]:
        result: dict[str, object] = {"target": target, "domains": {}}
        target_all_cached = True
        sample_cache = sample_cache_by_target[target]

        for domain in DOMAIN_ORDER:
            model_cache_path = (
                model_cache_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_artifacts.joblib.gz"
            )
            cache_signature = build_target_cache_signature(
                target=target,
                domain=domain,
                csv_path=csv_path,
                sample_cache=sample_cache,
                combined_features=combined_features,
                transition_base_vars=transition_base_vars,
                transition_window_size=transition_window_size,
                neighbors_trans=int(args.neighbors_trans),
                block_size_m=block_size_m,
                max_missing_frac=float(args.max_missing_frac),
                model_n_jobs=model_n_jobs,
                test_size=float(args.test_size),
            )
            cached = None if args.force_retrain else load_cached_target_artifacts(model_cache_path, cache_signature)

            if cached is not None:
                log_step(f"Loaded cached model artifacts for {target}/{domain}: {model_cache_path}")
                domain_result = dict(cached)
            else:
                target_all_cached = False
                domain_result = fit_domain_model(
                    ensure_training_sample(target),
                    target=target,
                    domain=domain,
                    combined_features=combined_features,
                    block_size_m=block_size_m,
                    test_size=float(args.test_size),
                    max_missing_frac=float(args.max_missing_frac),
                    model_n_jobs=model_n_jobs,
                )
                save_target_artifacts(
                    model_cache_path,
                    {
                        "cache_signature": cache_signature,
                        "target": target,
                        "domain": domain,
                        "y_test": domain_result["y_test"],
                        "suite_metrics": domain_result["suite_metrics"],
                        "suite_probs": domain_result["suite_probs"],
                        "full_stack": domain_result["full_stack"],
                        "importance_df": domain_result["importance_df"],
                        "missing_df": domain_result["missing_df"],
                        "feature_names": domain_result["feature_names"],
                        "feature_groups": domain_result["feature_groups"],
                        "dropped_features": domain_result["dropped_features"],
                        "positive_fraction": domain_result["positive_fraction"],
                        "n_rows": domain_result["n_rows"],
                        "n_train": domain_result["n_train"],
                        "n_test": domain_result["n_test"],
                    },
                )
                log_step(f"Saved model artifacts for {target}/{domain}: {model_cache_path}")

            domain_result["model_cache_path"] = model_cache_path
            result["domains"][domain] = domain_result

            domain_result["missing_df"].to_csv(
                table_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_feature_missingness.csv",
                index=False,
            )
            domain_result["importance_df"].to_csv(
                table_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_permutation_importance.csv",
                index=False,
            )

            for model_name, metrics in domain_result["suite_metrics"].items():
                perf_rows.append(
                    {
                        "target": target,
                        "domain": domain,
                        "model": model_name,
                        "roc_auc": metrics["roc_auc"],
                        "ap": metrics["ap"],
                        "brier": metrics["brier"],
                    }
                )

        combined_raster_signature = build_raster_cache_signature(
            kind="combined_domain",
            target=target,
            domain=None,
            csv_path=csv_path,
            model_cache_paths={
                domain: result["domains"][domain]["model_cache_path"]
                for domain in DOMAIN_ORDER
            },
            feature_names={
                domain: result["domains"][domain]["feature_names"]
                for domain in DOMAIN_ORDER
            },
            transition_base_vars=transition_base_vars,
            transition_window_size=transition_window_size,
            neighbors_trans=int(args.neighbors_trans),
            grid=grid,
        )
        prob_path = cache_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_susceptibility_f32.memmap"
        if (
            cached_raster_is_valid(prob_path, combined_raster_signature)
            and target_all_cached
            and not args.force_raster
        ):
            log_step(f"Using cached susceptibility raster for {target}: {prob_path}")
        else:
            build_combined_domain_prediction_raster(
                csv_path=csv_path,
                prob_path=prob_path,
                domain_models={
                    domain: result["domains"][domain]["full_stack"]
                    for domain in DOMAIN_ORDER
                },
                domain_feature_names={
                    domain: result["domains"][domain]["feature_names"]
                    for domain in DOMAIN_ORDER
                },
                grid=grid,
                metric_paths=ensure_spatial_support(),
                vars_for_transition=transition_base_vars,
                raw_features=raw_features,
                chunksize=int(args.chunksize),
                cache_signature=combined_raster_signature,
            )
        result["prob_path"] = prob_path
        predictor_prob_paths: dict[str, Path] = {}
        for domain in DOMAIN_ORDER:
            predictor_raster_signature = build_raster_cache_signature(
                kind="single_model_full_domain",
                target=target,
                domain=domain,
                csv_path=csv_path,
                model_cache_paths={domain: result["domains"][domain]["model_cache_path"]},
                feature_names=result["domains"][domain]["feature_names"],
                transition_base_vars=transition_base_vars,
                transition_window_size=transition_window_size,
                neighbors_trans=int(args.neighbors_trans),
                grid=grid,
            )
            predictor_prob_path = (
                cache_dir
                / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_predictor_full_domain_f32.memmap"
            )
            if (
                cached_raster_is_valid(predictor_prob_path, predictor_raster_signature)
                and target_all_cached
                and not args.force_raster
            ):
                log_step(f"Using cached full-domain predictor raster for {target}/{domain}: {predictor_prob_path}")
            else:
                build_single_model_prediction_raster(
                    csv_path=csv_path,
                    prob_path=predictor_prob_path,
                    model=result["domains"][domain]["full_stack"],
                    feature_names=result["domains"][domain]["feature_names"],
                    grid=grid,
                    metric_paths=ensure_spatial_support(),
                    vars_for_transition=transition_base_vars,
                    raw_features=raw_features,
                    chunksize=int(args.chunksize),
                    log_label=f"{target}/{domain}",
                    cache_signature=predictor_raster_signature,
                )
            predictor_prob_paths[domain] = predictor_prob_path
        result["predictor_prob_paths"] = predictor_prob_paths
        result["grid"] = dict(grid)
        results[target] = result

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(table_dir / f"{FIG_BASENAME}_model_performance.csv", index=False)

    performance_png, performance_pdf = build_model_performance_figure(
        results=results,
        fig_dir=fig_dir,
    )
    predictor_png, predictor_pdf = build_predictor_domain_figure(
        results=results,
        railway_segments=railway_segments,
        meteoro_sites=meteoro_sites,
        source_crs=map_source_crs,
        fig_dir=fig_dir,
        nrows=int(grid["nrows"]),
        ncols=int(grid["ncols"]),
        target_max_pixels=int(args.target_max_pixels),
    )
    combined_png, combined_pdf = build_combined_domain_figure(
        results=results,
        railway_segments=railway_segments,
        meteoro_sites=meteoro_sites,
        source_crs=map_source_crs,
        fig_dir=fig_dir,
        nrows=int(grid["nrows"]),
        ncols=int(grid["ncols"]),
        target_max_pixels=int(args.target_max_pixels),
    )
    figure_outputs = {
        "model_performance": {"png": performance_png, "pdf": performance_pdf},
        "predictor_domain_maps": {"png": predictor_png, "pdf": predictor_pdf},
        "combined_domain_maps": {"png": combined_png, "pdf": combined_pdf},
    }

    meta_out = cache_dir / f"{FIG_BASENAME}_meta.json"
    meta_out.write_text(
        json.dumps(
            {
                "figure_png": str(performance_png),
                "figure_pdf": str(performance_pdf),
                "figure_outputs": {
                    name: {fmt: str(path) for fmt, path in formats.items()}
                    for name, formats in figure_outputs.items()
                },
                "sample_cache_by_target": {target: str(path) for target, path in sample_cache_by_target.items()},
                "transition_window_size": int(transition_window_size),
                "transition_base_vars": transition_base_vars,
                "grid_crs_info": Path(base_dir / "crs_info.txt").read_text().strip() if (base_dir / "crs_info.txt").exists() else None,
                "meteoro_shp": str(meteoro_shp),
                "raw_features": raw_features,
                "contrast_features": contrast_features,
                "combined_features": combined_features,
                "pca_explained_variance": float(PCA_EXPLAINED_VARIANCE),
                "negative_du_only": bool(NEGATIVE_DU_ONLY),
                "base_learners": [name for name in ["RF", "ET", "HGB", "XGB"] if name != "XGB" or fig6.HAVE_XGB],
                "block_size_km": float(args.block_size_km),
                "test_size": float(args.test_size),
                "max_missing_frac": float(args.max_missing_frac),
                "exclude_magt": bool(args.exclude_magt),
                "sample_pf": int(args.sample_pf),
                "sample_npf": int(args.sample_npf),
                "targets": {
                    target: {
                        "combined_probability_raster": str(results[target]["prob_path"]),
                        "predictor_probability_rasters": {
                            domain: str(results[target]["predictor_prob_paths"][domain])
                            for domain in DOMAIN_ORDER
                        },
                        "domains": {
                            domain: {
                                "positive_fraction": results[target]["domains"][domain]["positive_fraction"],
                                "n_rows": results[target]["domains"][domain]["n_rows"],
                                "n_train": results[target]["domains"][domain]["n_train"],
                                "n_test": results[target]["domains"][domain]["n_test"],
                                "model_cache_path": str(results[target]["domains"][domain]["model_cache_path"]),
                                "feature_names": results[target]["domains"][domain]["feature_names"],
                                "dropped_features": results[target]["domains"][domain]["dropped_features"],
                                "permutation_importance_csv": str(
                                    table_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_permutation_importance.csv"
                                ),
                                "feature_missingness_csv": str(
                                    table_dir / f"{FIG_BASENAME}_{TARGET_META[target]['table_stub']}_{domain}_feature_missingness.csv"
                                ),
                                "threshold": threshold_for_domain(target, domain),
                            }
                            for domain in DOMAIN_ORDER
                        },
                    }
                    for target in ["d_u", "grad_mag_km"]
                },
            },
            indent=2,
        )
    )

    for name, formats in figure_outputs.items():
        log_step(f"Saved {name} PNG: {formats['png']}")
        log_step(f"Saved {name} PDF: {formats['pdf']}")
    log_step(f"Saved performance table: {table_dir / f'{FIG_BASENAME}_model_performance.csv'}")
    log_step(f"Saved meta JSON: {meta_out}")


if __name__ == "__main__":
    main()
