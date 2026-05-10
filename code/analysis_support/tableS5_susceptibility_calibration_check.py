#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure06_susceptibility_calibration.py
# Renamed package path: code/analysis_support/tableS5_susceptibility_calibration_check.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from submission_build_common import (
    CACHE_DIR,
    FONT,
    ROOT_DIR,
    SOURCE_CACHE_DIR,
    TABLE_DIR,
    add_panel_label,
    blend_with_white,
    ensure_style,
    save_figure,
)

import Figure_reorganize_extreme_deformation_susceptibity as figsus
import figure6_susceptibility_stacked as fig6


FIG_STEM = "Figure06_susceptibility_calibration"
SOURCE_META_PATH = SOURCE_CACHE_DIR / f"{figsus.FIG_BASENAME}_meta.json"
SOURCE_TRANSITION_CACHE_DIR = SOURCE_CACHE_DIR / f"{figsus.FIG_BASENAME}_transition_rasters"
DOMAIN_ORDER = ("pf", "npf")
TARGET_ORDER = ("d_u", "grad_mag_km")
TARGET_LABELS = {
    "d_u": r"Extreme $d_u$",
    "grad_mag_km": r"Extreme $|\nabla d_u|$",
}
TARGET_COLORS = {
    "d_u": "#1E5BAA",
    "grad_mag_km": "#C1272D",
}
PF_COLOR = "#5A8F63"
NPF_COLOR = "#9A6A49"
COMBINED_COLOR = "#6F6F6F"
FUSION_COLOR = "#242424"
DOMAIN_COLORS = {"pf": PF_COLOR, "npf": NPF_COLOR}
DOMAIN_LABELS = {"pf": "PF", "npf": "NPF"}
METHOD_ORDER = ("pf_expert", "npf_expert", "dual_expert", "dual_rank", "pooled_baseline")
FIGURE_METHOD_ORDER = ("pf_expert", "npf_expert", "dual_expert", "pooled_baseline")
METHOD_LABELS = {
    "pf_expert": "PF expert",
    "npf_expert": "NPF expert",
    "dual_expert": "Calib. dual",
    "dual_rank": "Rank dual",
    "pooled_baseline": "Pooled baseline",
}
METHOD_COLORS = {
    "pf_expert": PF_COLOR,
    "npf_expert": NPF_COLOR,
    "dual_expert": FUSION_COLOR,
    "dual_rank": "#007C89",
    "pooled_baseline": COMBINED_COLOR,
}
METHOD_STYLES = {
    "pf_expert": "-",
    "npf_expert": "-",
    "dual_expert": "-",
    "dual_rank": "-.",
    "pooled_baseline": "--",
}
MODEL_CACHE_VERSION = 3
ATTACHED_SAMPLE_CACHE_VERSION = 1
SEED = figsus.SEED
DEFAULT_MAX_DOMAIN_ROWS = 60_000
DEFAULT_N_ESTIMATORS = 320
DEFAULT_MODEL_N_JOBS = max(1, min(4, os.cpu_count() or 1))
TRAIN_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
TEST_FRACTION = 0.20
CALIBRATION_BINS = 8
SPATIAL_BLOCK_SIZE_M = fig6.SPATIAL_BLOCK_SIZE_M
MAX_FEATURE_MISSING_FRAC = fig6.MAX_FEATURE_MISSING_FRAC
TRANSITION_NEIGHBORS = 21
EPS = 1e-5


def log_step(message: str) -> None:
    print(f"[{FIG_STEM}] {message}", flush=True)


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_csv_path() -> Path:
    path = ROOT_DIR / "df_all_data_with_wright_du_plus_grad.csv"
    if path.exists():
        return path
    fallback = ROOT_DIR / "df_all_data_with_wright_du.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find the source deformation CSV used for susceptibility training.")


def load_source_meta() -> dict[str, object]:
    if not SOURCE_META_PATH.exists():
        raise FileNotFoundError(f"Missing susceptibility metadata: {SOURCE_META_PATH}")
    return json.loads(SOURCE_META_PATH.read_text(encoding="utf-8"))


def sample_path_for_target(target: str) -> Path:
    meta = load_source_meta()
    sample_path = Path(meta["sample_cache_by_target"][target])
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing cached susceptibility sample for {target}: {sample_path}")
    return sample_path


def attached_sample_cache_path(target: str, max_domain_rows: int) -> Path:
    stub = figsus.TARGET_META[target]["table_stub"]
    return CACHE_DIR / f"{FIG_STEM}_{stub}_attached_max{max_domain_rows}_v{ATTACHED_SAMPLE_CACHE_VERSION}.csv.gz"


def target_prediction_cache_path(target: str, max_domain_rows: int, n_estimators: int) -> Path:
    stub = figsus.TARGET_META[target]["table_stub"]
    return CACHE_DIR / f"{FIG_STEM}_{stub}_predictions_max{max_domain_rows}_trees{n_estimators}_v{MODEL_CACHE_VERSION}.joblib.gz"


def model_signature(
    *,
    target: str,
    sample_path: Path,
    max_domain_rows: int,
    n_estimators: int,
    model_n_jobs: int,
    feature_names: list[str],
) -> dict[str, object]:
    return {
        "version": MODEL_CACHE_VERSION,
        "target": target,
        "seed": SEED,
        "sample_path": file_signature(sample_path),
        "max_domain_rows": int(max_domain_rows),
        "n_estimators": int(n_estimators),
        "model_n_jobs": int(model_n_jobs),
        "feature_names": list(feature_names),
        "train_fraction": float(TRAIN_FRACTION),
        "calibration_fraction": float(CALIBRATION_FRACTION),
        "test_fraction": float(TEST_FRACTION),
        "block_size_m": float(SPATIAL_BLOCK_SIZE_M),
        "transition_neighbors": int(TRANSITION_NEIGHBORS),
        "max_feature_missing_frac": float(MAX_FEATURE_MISSING_FRAC),
    }


def build_expert_model(*, n_estimators: int, model_n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=figsus.PCA_EXPLAINED_VARIANCE, svd_solver="full")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=int(n_estimators),
                    random_state=SEED,
                    n_jobs=int(model_n_jobs),
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def universal_label(df: pd.DataFrame, target: str) -> tuple[pd.Series, float]:
    if target == "d_u":
        values = pd.to_numeric(df["d_u"], errors="coerce").to_numpy(dtype=float)
        threshold = float(np.nanpercentile(values[np.isfinite(values)], 5.0))
        label = (pd.to_numeric(df["d_u"], errors="coerce") <= threshold).astype(int)
    else:
        values = pd.to_numeric(df["grad_mag_km"], errors="coerce").to_numpy(dtype=float)
        threshold = float(np.nanpercentile(values[np.isfinite(values)], 95.0))
        label = (pd.to_numeric(df["grad_mag_km"], errors="coerce") >= threshold).astype(int)
    return label.rename("universal_label"), threshold


def critical_label(df: pd.DataFrame, target: str) -> pd.Series:
    out = np.full(len(df), np.nan, dtype=float)
    domains = df["domain"].astype(str).to_numpy()
    for domain in DOMAIN_ORDER:
        mask = domains == domain
        if not np.any(mask):
            continue
        labels = figsus.build_domain_label(df.loc[mask], target=target, domain=domain)
        out[mask] = labels.to_numpy(dtype=float)
    return pd.Series(out, index=df.index, name="critical_label")


def choose_group_split(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
    max_tries: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    groups = df["block_id"].astype(str).to_numpy()
    domains = df["domain"].astype(str).to_numpy()
    target_test_n = float(test_size) * len(df)
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score = np.inf

    def valid_subset(idx: np.ndarray) -> bool:
        if idx.size == 0 or np.unique(y[idx]).size < 2:
            return False
        for domain in DOMAIN_ORDER:
            dom_mask = domains[idx] == domain
            if dom_mask.sum() < 200:
                return False
            if np.unique(y[idx][dom_mask]).size < 2:
                return False
        return True

    for offset in range(max_tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(df, y, groups))
        if not valid_subset(train_idx) or not valid_subset(test_idx):
            continue
        score = abs(len(test_idx) - target_test_n)
        if score < best_score:
            best_score = score
            best_split = (np.sort(train_idx), np.sort(test_idx))
        if score <= 0.01 * len(df):
            break

    if best_split is None:
        raise RuntimeError("Failed to find a spatial group split with both classes in both domains.")
    return best_split


def make_three_way_split(df: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    holdout_size = CALIBRATION_FRACTION + TEST_FRACTION
    train_idx, holdout_idx = choose_group_split(
        df,
        y,
        test_size=holdout_size,
        seed=SEED,
    )
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    holdout_y = y[holdout_idx]
    cal_rel_idx, test_rel_idx = choose_group_split(
        holdout_df,
        holdout_y,
        test_size=TEST_FRACTION / holdout_size,
        seed=SEED + 1000,
    )
    return {
        "train": train_idx,
        "calibration": holdout_idx[cal_rel_idx],
        "test": holdout_idx[test_rel_idx],
    }


def make_model_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return fig6.make_model_frame(df, feature_names)


def fit_binary_model(
    train_df: pd.DataFrame,
    feature_names: list[str],
    y_train: np.ndarray,
    *,
    n_estimators: int,
    model_n_jobs: int,
    label: str,
) -> Pipeline:
    if np.unique(y_train).size < 2:
        raise RuntimeError(f"{label}: training labels contain a single class.")
    model = build_expert_model(n_estimators=n_estimators, model_n_jobs=model_n_jobs)
    X_train = make_model_frame(train_df, feature_names)
    log_step(f"Fitting {label}: {len(train_df):,} rows, positive fraction={np.mean(y_train):.3f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    del X_train
    gc.collect()
    return model


def predict_model(model: Pipeline, df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    X = make_model_frame(df, feature_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict_proba(X)[:, 1].astype(float)
    del X
    return np.clip(pred, EPS, 1.0 - EPS)


def logit_features(p_pf: np.ndarray, p_npf: np.ndarray) -> np.ndarray:
    p_pf = np.clip(np.asarray(p_pf, dtype=float), EPS, 1.0 - EPS)
    p_npf = np.clip(np.asarray(p_npf, dtype=float), EPS, 1.0 - EPS)
    return np.column_stack(
        [
            np.log(p_pf / (1.0 - p_pf)),
            np.log(p_npf / (1.0 - p_npf)),
        ]
    )


def percentile_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        raise RuntimeError("Cannot rank-normalize against an empty reference distribution.")
    values = np.asarray(values, dtype=float)
    ranks = np.searchsorted(ref, values, side="right").astype(float) / float(ref.size)
    return np.clip(ranks, EPS, 1.0 - EPS)


def rank_dual_feature(
    *,
    p_pf: np.ndarray,
    p_npf: np.ndarray,
    ref_pf: np.ndarray,
    ref_npf: np.ndarray,
) -> np.ndarray:
    pf_rank = percentile_against_reference(p_pf, ref_pf)
    npf_rank = percentile_against_reference(p_npf, ref_npf)
    return np.maximum(pf_rank, npf_rank)


def load_attached_target_sample(
    target: str,
    *,
    max_domain_rows: int,
    force: bool,
) -> tuple[pd.DataFrame, list[str]]:
    raw_features, _contrast_features, combined_features, transition_base_vars = figsus.build_feature_sets(
        exclude_magt=False
    )
    cache_path = attached_sample_cache_path(target, max_domain_rows)
    if cache_path.exists() and not force:
        log_step(f"Loading attached sample cache for {target}: {cache_path.name}")
        df = pd.read_csv(cache_path)
        available_features = [feature for feature in combined_features if feature in df.columns]
        return df, available_features

    sample_path = sample_path_for_target(target)
    log_step(f"Loading source sample for {target}: {sample_path.name}")
    df = pd.read_csv(sample_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = fig6.engineer_features(df)
    df = df.loc[df["domain"].isin(DOMAIN_ORDER)].copy()

    if max_domain_rows > 0:
        parts = []
        for domain in DOMAIN_ORDER:
            sub = df.loc[df["domain"].eq(domain)].copy()
            n = min(int(max_domain_rows), len(sub))
            parts.append(sub.sample(n=n, random_state=SEED).copy())
        df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    transition_window_size = fig6.neighbors_to_window_size(TRANSITION_NEIGHBORS)
    log_step(f"Attaching transition metrics for {target} from {SOURCE_TRANSITION_CACHE_DIR.name}")
    grid, _mean_paths, metric_paths = fig6.ensure_transition_rasters(
        base_dir=ROOT_DIR,
        csv_path=source_csv_path(),
        cache_dir=SOURCE_TRANSITION_CACHE_DIR,
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
    df["critical_label"] = critical_label(df, target)
    df["block_id"] = fig6.make_spatial_block_id(df, block_size_m=SPATIAL_BLOCK_SIZE_M)
    df = df.loc[df["critical_label"].notna() & df["block_id"].notna()].reset_index(drop=True)
    df["critical_label"] = df["critical_label"].astype(int)

    available_features = [feature for feature in combined_features if feature in df.columns]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, compression="gzip")
    log_step(f"Wrote attached sample cache for {target}: {cache_path.name}")
    return df, available_features


def retained_features(train_df: pd.DataFrame, available_features: list[str]) -> tuple[list[str], pd.DataFrame]:
    feature_groups = {
        "raw": [feature for feature in available_features if figsus.feature_origin(feature) == "raw"],
        "contrast": [feature for feature in available_features if figsus.feature_origin(feature) == "contrast"],
    }
    missing_df = fig6.audit_feature_missingness(train_df, feature_groups)
    dropped = set(
        missing_df.loc[
            missing_df["missing_frac"] > MAX_FEATURE_MISSING_FRAC,
            "feature",
        ]
        .astype(str)
        .tolist()
    )
    keep = [feature for feature in available_features if feature not in dropped]
    if not keep:
        raise RuntimeError("No predictors remain after missingness filtering.")
    return keep, missing_df


def fit_target_predictions(
    target: str,
    *,
    max_domain_rows: int,
    n_estimators: int,
    model_n_jobs: int,
    force: bool,
) -> dict[str, object]:
    sample_path = sample_path_for_target(target)
    df, available_features = load_attached_target_sample(
        target,
        max_domain_rows=max_domain_rows,
        force=force,
    )
    signature = model_signature(
        target=target,
        sample_path=sample_path,
        max_domain_rows=max_domain_rows,
        n_estimators=n_estimators,
        model_n_jobs=model_n_jobs,
        feature_names=available_features,
    )
    pred_cache_path = target_prediction_cache_path(target, max_domain_rows, n_estimators)
    if pred_cache_path.exists() and not force:
        try:
            cached = joblib.load(pred_cache_path)
            if cached.get("signature") == signature:
                log_step(f"Using cached held-out predictions for {target}: {pred_cache_path.name}")
                return cached
        except Exception as exc:
            log_step(f"Ignoring unreadable prediction cache for {target}: {exc}")

    y_critical = df["critical_label"].to_numpy(dtype=int)
    split = make_three_way_split(df, y_critical)
    train_df = df.iloc[split["train"]].reset_index(drop=True)
    cal_df = df.iloc[split["calibration"]].reset_index(drop=True)
    test_df = df.iloc[split["test"]].reset_index(drop=True)
    feature_names, missing_df = retained_features(train_df, available_features)
    log_step(
        f"{target}: retained {len(feature_names)} predictors; "
        f"split train/cal/test = {len(train_df):,}/{len(cal_df):,}/{len(test_df):,}"
    )

    models: dict[str, Pipeline] = {}
    for domain in DOMAIN_ORDER:
        domain_train = train_df.loc[train_df["domain"].eq(domain)].reset_index(drop=True)
        y_domain = figsus.build_domain_label(domain_train, target=target, domain=domain).to_numpy(dtype=int)
        models[f"{domain}_expert"] = fit_binary_model(
            domain_train,
            feature_names,
            y_domain,
            n_estimators=n_estimators,
            model_n_jobs=model_n_jobs,
            label=f"{target}/{domain} expert",
        )

    universal_train_label, universal_threshold = universal_label(train_df, target)
    models["pooled_baseline"] = fit_binary_model(
        train_df,
        feature_names,
        universal_train_label.to_numpy(dtype=int),
        n_estimators=n_estimators,
        model_n_jobs=model_n_jobs,
        label=f"{target}/pooled baseline",
    )

    cal_scores = {
        "pf_expert": predict_model(models["pf_expert"], cal_df, feature_names),
        "npf_expert": predict_model(models["npf_expert"], cal_df, feature_names),
        "pooled_baseline": predict_model(models["pooled_baseline"], cal_df, feature_names),
    }
    fusion_model = LogisticRegression(max_iter=2000, random_state=SEED)
    fusion_model.fit(
        logit_features(cal_scores["pf_expert"], cal_scores["npf_expert"]),
        cal_df["critical_label"].to_numpy(dtype=int),
    )
    cal_rank_max = rank_dual_feature(
        p_pf=cal_scores["pf_expert"],
        p_npf=cal_scores["npf_expert"],
        ref_pf=cal_scores["pf_expert"],
        ref_npf=cal_scores["npf_expert"],
    )
    rank_fusion_model = LogisticRegression(max_iter=2000, random_state=SEED)
    rank_fusion_model.fit(
        cal_rank_max.reshape(-1, 1),
        cal_df["critical_label"].to_numpy(dtype=int),
    )

    test_scores = {
        "pf_expert": predict_model(models["pf_expert"], test_df, feature_names),
        "npf_expert": predict_model(models["npf_expert"], test_df, feature_names),
        "pooled_baseline": predict_model(models["pooled_baseline"], test_df, feature_names),
    }
    test_scores["dual_expert"] = np.clip(
        fusion_model.predict_proba(logit_features(test_scores["pf_expert"], test_scores["npf_expert"]))[:, 1],
        EPS,
        1.0 - EPS,
    )
    test_scores["dual_max"] = np.maximum(test_scores["pf_expert"], test_scores["npf_expert"])
    test_scores["dual_rank_raw"] = rank_dual_feature(
        p_pf=test_scores["pf_expert"],
        p_npf=test_scores["npf_expert"],
        ref_pf=cal_scores["pf_expert"],
        ref_npf=cal_scores["npf_expert"],
    )
    test_scores["dual_rank"] = np.clip(
        rank_fusion_model.predict_proba(test_scores["dual_rank_raw"].reshape(-1, 1))[:, 1],
        EPS,
        1.0 - EPS,
    )

    pred_df = pd.DataFrame(
        {
            "target": target,
            "domain": test_df["domain"].astype(str).to_numpy(),
            "critical_label": test_df["critical_label"].to_numpy(dtype=int),
            "d_u": pd.to_numeric(test_df["d_u"], errors="coerce").to_numpy(dtype=float),
            "grad_mag_km": pd.to_numeric(test_df["grad_mag_km"], errors="coerce").to_numpy(dtype=float),
            "easting": pd.to_numeric(test_df["easting"], errors="coerce").to_numpy(dtype=float),
            "northing": pd.to_numeric(test_df["northing"], errors="coerce").to_numpy(dtype=float),
            **test_scores,
        }
    )
    split_summary = pd.DataFrame(
        [
            {
                "target": target,
                "split": split_name,
                "domain": domain,
                "n": int((df.iloc[idx]["domain"].astype(str) == domain).sum()),
                "positive_fraction": float(
                    df.iloc[idx].loc[df.iloc[idx]["domain"].astype(str) == domain, "critical_label"].mean()
                ),
            }
            for split_name, idx in split.items()
            for domain in DOMAIN_ORDER
        ]
    )
    payload = {
        "signature": signature,
        "target": target,
        "predictions": pred_df,
        "split_summary": split_summary,
        "missingness": missing_df,
        "feature_names": feature_names,
        "universal_threshold": float(universal_threshold),
    }
    pred_cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, pred_cache_path, compress=("gzip", 3))
    log_step(f"Wrote held-out prediction cache for {target}: {pred_cache_path.name}")
    return payload


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    if np.unique(y).size < 2:
        return np.nan
    return float(average_precision_score(y, p))


def top_decile_rate(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(y) == 0:
        return np.nan, np.nan
    cutoff = float(np.nanquantile(p, 0.90))
    top = p >= cutoff
    if not np.any(top):
        return np.nan, np.nan
    rate = float(np.mean(y[top]))
    base = float(np.mean(y))
    lift = rate / base if base > 0 else np.nan
    return rate, lift


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_order = METHOD_ORDER + ("dual_max", "dual_rank_raw")
    for target in TARGET_ORDER:
        target_df = predictions.loc[predictions["target"].eq(target)].copy()
        for eval_domain in ("all",) + DOMAIN_ORDER:
            domain_df = (
                target_df
                if eval_domain == "all"
                else target_df.loc[target_df["domain"].eq(eval_domain)]
            )
            y = domain_df["critical_label"].to_numpy(dtype=int)
            if len(y) == 0:
                continue
            for method in method_order:
                p = domain_df[method].to_numpy(dtype=float)
                decile_rate, decile_lift = top_decile_rate(y, p)
                rows.append(
                    {
                        "target": target,
                        "eval_domain": eval_domain,
                        "method": method,
                        "method_label": METHOD_LABELS.get(
                            method,
                            "Dual max" if method == "dual_max" else "Rank dual raw",
                        ),
                        "n": int(len(y)),
                        "positive_fraction": float(np.mean(y)),
                        "mean_prediction": float(np.mean(p)),
                        "prediction_to_observed_ratio": float(np.mean(p) / np.mean(y)) if np.mean(y) > 0 else np.nan,
                        "average_precision": safe_ap(y, p),
                        "roc_auc": safe_auc(y, p),
                        "brier": float(brier_score_loss(y, p)),
                        "top_decile_rate": decile_rate,
                        "top_decile_lift": decile_lift,
                    }
                )
    return pd.DataFrame(rows)


def build_calibration_bins(predictions: pd.DataFrame, *, n_bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in TARGET_ORDER:
        target_df = predictions.loc[predictions["target"].eq(target)].copy()
        for domain in DOMAIN_ORDER:
            domain_df = target_df.loc[target_df["domain"].eq(domain)].copy()
            if domain_df.empty:
                continue
            y = domain_df["critical_label"].to_numpy(dtype=int)
            for method in METHOD_ORDER:
                p = domain_df[method].to_numpy(dtype=float)
                tmp = pd.DataFrame({"y": y, "p": p})
                q = min(int(n_bins), max(2, len(tmp) // 100))
                tmp["bin"] = pd.qcut(
                    tmp["p"].rank(method="first"),
                    q=q,
                    labels=False,
                    duplicates="drop",
                )
                grouped = tmp.groupby("bin", observed=True)
                for bin_id, sub in grouped:
                    rows.append(
                        {
                            "target": target,
                            "eval_domain": domain,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "bin": int(bin_id),
                            "n": int(len(sub)),
                            "mean_prediction": float(sub["p"].mean()),
                            "observed_rate": float(sub["y"].mean()),
                            "positive_n": int(sub["y"].sum()),
                            "prediction_min": float(sub["p"].min()),
                            "prediction_max": float(sub["p"].max()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_skill_heatmap(ax, metrics_df: pd.DataFrame, *, target: str) -> None:
    sub = metrics_df.loc[
        metrics_df["target"].eq(target)
        & metrics_df["eval_domain"].isin(DOMAIN_ORDER)
        & metrics_df["method"].isin(FIGURE_METHOD_ORDER)
    ].copy()
    value = np.full((len(DOMAIN_ORDER), len(FIGURE_METHOD_ORDER)), np.nan, dtype=float)
    lift = np.full_like(value, np.nan)
    for i, domain in enumerate(DOMAIN_ORDER):
        for j, method in enumerate(FIGURE_METHOD_ORDER):
            row = sub.loc[sub["eval_domain"].eq(domain) & sub["method"].eq(method)]
            if row.empty:
                continue
            value[i, j] = float(row.iloc[0]["average_precision"])
            lift[i, j] = float(row.iloc[0]["top_decile_lift"])

    vmax = max(float(np.nanmax(value)) if np.isfinite(value).any() else 0.1, 0.12)
    cmap = LinearSegmentedColormap.from_list(
        f"{target}_skill",
        ["white", blend_with_white(TARGET_COLORS[target], 0.72), TARGET_COLORS[target]],
    )
    im = ax.imshow(value, vmin=0.0, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(FIGURE_METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in FIGURE_METHOD_ORDER], rotation=33, ha="right")
    ax.set_yticks(np.arange(len(DOMAIN_ORDER)))
    ax.set_yticklabels([f"{DOMAIN_LABELS[d]} test" for d in DOMAIN_ORDER])
    for tick, domain in zip(ax.get_yticklabels(), DOMAIN_ORDER):
        tick.set_color(DOMAIN_COLORS[domain])
        tick.set_fontweight("semibold")
    ax.set_title(f"{TARGET_LABELS[target]}\nheld-out skill", color=TARGET_COLORS[target], pad=7)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(FIGURE_METHOD_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAIN_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(DOMAIN_ORDER)):
        for j in range(len(FIGURE_METHOD_ORDER)):
            if not np.isfinite(value[i, j]):
                continue
            text_color = "white" if value[i, j] > 0.58 * vmax else "0.12"
            ax.text(
                j,
                i,
                f"AP {value[i, j]:.2f}\n{lift[i, j]:.1f}x",
                ha="center",
                va="center",
                fontsize=FONT["annotation"],
                color=text_color,
                linespacing=0.9,
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.set_label("AP", fontsize=FONT["annotation"], labelpad=2)
    cbar.ax.tick_params(labelsize=FONT["annotation"], length=2)


def calibration_axis_limit(bins_df: pd.DataFrame, metrics_df: pd.DataFrame, *, target: str, domain: str) -> float:
    sub = bins_df.loc[bins_df["target"].eq(target) & bins_df["eval_domain"].eq(domain)]
    vals = []
    if not sub.empty:
        vals.extend(pd.to_numeric(sub["mean_prediction"], errors="coerce").dropna().tolist())
        vals.extend(pd.to_numeric(sub["observed_rate"], errors="coerce").dropna().tolist())
    base = metrics_df.loc[
        metrics_df["target"].eq(target)
        & metrics_df["eval_domain"].eq(domain)
        & metrics_df["method"].eq("dual_expert"),
        "positive_fraction",
    ]
    if not base.empty:
        vals.append(float(base.iloc[0]))
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    hi = float(np.nanmax(finite)) if finite.size else 0.1
    return min(1.0, max(0.12, hi * 1.16))


def plot_calibration_axis(
    ax,
    bins_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    target: str,
    domain: str,
) -> None:
    ax.set_facecolor(blend_with_white(DOMAIN_COLORS[domain], 0.94))
    sub = bins_df.loc[bins_df["target"].eq(target) & bins_df["eval_domain"].eq(domain)].copy()
    for method in FIGURE_METHOD_ORDER:
        msub = sub.loc[sub["method"].eq(method)].sort_values("mean_prediction")
        if msub.empty:
            continue
        ax.plot(
            msub["mean_prediction"],
            msub["observed_rate"],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            linewidth=1.45 if method != "dual_expert" else 1.9,
            marker="o",
            markersize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.95,
            zorder=3,
        )
    lim = calibration_axis_limit(bins_df, metrics_df, target=target, domain=domain)
    ax.plot([0.0, lim], [0.0, lim], color="0.48", linestyle=":", linewidth=0.9, zorder=1)
    base = metrics_df.loc[
        metrics_df["target"].eq(target)
        & metrics_df["eval_domain"].eq(domain)
        & metrics_df["method"].eq("dual_expert"),
        "positive_fraction",
    ]
    if not base.empty:
        ax.axhline(float(base.iloc[0]), color=DOMAIN_COLORS[domain], linewidth=0.9, alpha=0.55, zorder=1)
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(color="white", linewidth=0.8, zorder=0)
    ax.set_title(f"{DOMAIN_LABELS[domain]} calibration", color=DOMAIN_COLORS[domain], pad=7)
    ax.set_xlabel("Mean predicted susceptibility")
    ax.set_ylabel("Observed extreme rate")


def build_figure(metrics_df: pd.DataFrame, bins_df: pd.DataFrame) -> tuple[Path, Path]:
    ensure_style()
    fig = plt.figure(figsize=(11.3, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.145,
        width_ratios=[1.18, 1.0, 1.0],
        wspace=0.35,
        hspace=0.47,
    )
    labels = iter("ABCDEF")
    for row, target in enumerate(TARGET_ORDER):
        ax_skill = fig.add_subplot(gs[row, 0])
        plot_skill_heatmap(ax_skill, metrics_df, target=target)
        add_panel_label(ax_skill, next(labels), x=-0.20, y=1.05)

        for col, domain in enumerate(DOMAIN_ORDER, start=1):
            ax_cal = fig.add_subplot(gs[row, col])
            plot_calibration_axis(ax_cal, bins_df, metrics_df, target=target, domain=domain)
            add_panel_label(ax_cal, next(labels), x=-0.16, y=1.05)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            marker="o",
            linewidth=1.8 if method == "dual_expert" else 1.45,
            markersize=4.0,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
        if method in FIGURE_METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.54, 0.045),
        columnspacing=1.35,
        handletextpad=0.55,
    )
    return save_figure(fig, FIG_STEM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and fuse PF/NPF susceptibility expert predictions.")
    parser.add_argument("--max-domain-rows", type=int, default=DEFAULT_MAX_DOMAIN_ROWS)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--model-n-jobs", type=int, default=DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    results = [
        fit_target_predictions(
            target,
            max_domain_rows=int(args.max_domain_rows),
            n_estimators=int(args.n_estimators),
            model_n_jobs=max(1, int(args.model_n_jobs)),
            force=bool(args.force),
        )
        for target in TARGET_ORDER
    ]
    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    split_summary = pd.concat([result["split_summary"] for result in results], ignore_index=True)
    metrics_df = build_metrics(predictions)
    bins_df = build_calibration_bins(predictions)

    predictions_path = TABLE_DIR / f"{FIG_STEM}_heldout_predictions.csv.gz"
    metrics_path = TABLE_DIR / f"{FIG_STEM}_metrics.csv"
    bins_path = TABLE_DIR / f"{FIG_STEM}_calibration_bins.csv"
    split_path = TABLE_DIR / f"{FIG_STEM}_split_summary.csv"
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    metrics_df.to_csv(metrics_path, index=False)
    bins_df.to_csv(bins_path, index=False)
    split_summary.to_csv(split_path, index=False)
    fig_png, fig_pdf = build_figure(metrics_df, bins_df)

    log_step(f"Wrote held-out predictions: {predictions_path}")
    log_step(f"Wrote metrics: {metrics_path}")
    log_step(f"Wrote calibration bins: {bins_path}")
    log_step(f"Wrote split summary: {split_path}")
    log_step(f"Saved figure PNG: {fig_png}")
    log_step(f"Saved figure PDF: {fig_pdf}")


if __name__ == "__main__":
    main()
