#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/figure6_susceptibility_stacked.py
# Renamed package path: code/analysis_support/susceptibility_model_workflow.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import pickle
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from figure6_0_transition_metric_review import (
    build_or_load_grid,
    build_raw_raster_cache,
    choose_stride,
    derive_transition_metric_rasters,
    en_to_rc,
    finalize_mean_rasters,
    get_extent,
    open_memmap,
)

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True      
except Exception:
    HAVE_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
CHUNKSIZE = 300_000
FIG_BASENAME = "Figure6_gradient_magnitude_susceptibility_stacked"
WINDOWS_FIG_BASENAME = "Figure6_gradient_magnitude_susceptibility_windows"
MODEL_ARTIFACT_VERSION = 4
MAIN_THRESHOLD_Q = 0.95
THRESHOLD_SWEEP_QS = [0.90, 0.93, 0.95, 0.97]
SPATIAL_BLOCK_SIZE_M = 100_000.0
MAX_FEATURE_MISSING_FRAC = 0.30
PERMUTATION_MAX_ROWS = 15_000
PERMUTATION_REPEATS = 15
CSV_PROGRESS_EVERY = 10
DEFAULT_MODEL_N_JOBS = 1
HOTSPOT_DENSITY_SIGMA_PX = 3.0
DEFAULT_N_ZOOM = 4
DEFAULT_ZOOM_SIZE_KM = 140.0
DEFAULT_MIN_CENTER_SPACING_KM = 180.0
DEFAULT_MIN_WINDOW_POINTS = 140
DEFAULT_MIN_WINDOW_HOTSPOTS = 8
ZOOM_WINDOW_LINEAR_SCALE = 0.25
ZOOM_FORBIDDEN_BOXES = [
    (1_485_000.0, 1_565_000.0, 3_800_000.0, 4_000_000.0),
]

RAW_FEATURES = [
    "magt",
    "precipitation_mean",
    "temperature_mean",
    "bulk_density",
    "cf",
    "soc",
    "soil_thickness",
    "dem",
    "difpr",
    "dirpr",
    "slope",
    "twi",
    "gpp_mean",
    "ndvi",
]

TRANSITION_BASE_VARS = [
    "soil_thickness",
    "cf",
    "soc",
    "bulk_density",
    "dem",
    "slope",
    "twi",
    "ndvi",
    "magt",
]

FEATURE_LABELS = {
    "magt": "MAGT",
    "precipitation_mean": "Precipitation",
    "temperature_mean": "Temperature",
    "bulk_density": "Bulk density",
    "cf": "Coarse fragments",
    "soc": "SOC",
    "soil_thickness": "Soil thickness",
    "dem": "Elevation",
    "difpr": "Diffuse rad.",
    "dirpr": "Direct rad.",
    "slope": "Slope",
    "twi": "TWI",
    "gpp_mean": "GPP",
    "ndvi": "NDVI",
}

PF_COLOR = "#9FD3E6"
NPF_COLOR = "#B8DDB1"
RAW_COLOR = "#4C78A8"
CONTRAST_COLOR = "#F28E2B"
WINDOW_COLORS = ["#F28E2B", "#4C78A8", "#59A14F", "#E15759"]
GRAD_MAG_DISPLAY_SCALE = 1000.0  # stored grad_mag is mm/yr/m; figures report mm/yr/km


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def robust_clip(arr: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def style_open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def style_map_axes(ax) -> None:
    style_open_axes(ax)
    ax.set_xlabel("Longitude (°)", fontweight="bold")
    ax.set_ylabel("Latitude (°)", fontweight="bold")


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        str(label).strip("()").lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="0.05",
        clip_on=False,
    )


def log_step(message: str) -> None:
    print(f"[figure6] {message}")


def iter_csv_chunks(
    csv_path: Path,
    *,
    usecols: list[str],
    chunksize: int,
    label: str,
    progress_every: int = CSV_PROGRESS_EVERY,
):
    for i, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        if i == 1 or i % progress_every == 0:
            log_step(f"{label}: processed {i} chunk(s)")
        yield chunk
    log_step(f"{label}: finished")


def engineer_features(df: pd.DataFrame, *, copy_df: bool = False) -> pd.DataFrame:
    out = df.copy() if copy_df else df
    if "Perma_Distr_map" in out.columns:
        out["Perma_Distr_map"] = pd.to_numeric(out["Perma_Distr_map"], errors="coerce")
    if "d_u" in out.columns:
        out["d_u"] = pd.to_numeric(out["d_u"], errors="coerce")
    if "grad_mag" in out.columns:
        out["grad_mag"] = pd.to_numeric(out["grad_mag"], errors="coerce")
    if "grad_mag_km" in out.columns:
        out["grad_mag_km"] = pd.to_numeric(out["grad_mag_km"], errors="coerce")
    if "grad_mag" in out.columns:
        grad_mag_km = out["grad_mag"] * GRAD_MAG_DISPLAY_SCALE
        if "grad_mag_km" in out.columns:
            out["grad_mag_km"] = out["grad_mag_km"].where(np.isfinite(out["grad_mag_km"]), grad_mag_km)
        else:
            out["grad_mag_km"] = grad_mag_km
        out["log_grad_mag"] = np.log1p(out["grad_mag"].clip(lower=0))
    for c in ["easting", "northing", "longitude", "latitude"] + RAW_FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Perma_Distr_map" in out.columns:
        out["domain"] = np.where(
            out["Perma_Distr_map"] == 1,
            "pf",
            np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
        )
    return out


def build_stratified_sample(
    csv_path: Path,
    usecols: list[str],
    desired_per_domain: dict[str, int],
    chunksize: int = CHUNKSIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    log_step("Building one-pass stratified PF/NPF sample")
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, pd.DataFrame] = {
        dom: pd.DataFrame(columns=usecols + ["_sample_key"])
        for dom in desired_per_domain
    }

    for chunk in iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label="sample build",
    ):
        chunk = engineer_features(chunk)
        chunk = chunk.loc[np.isfinite(chunk["grad_mag"])]
        if chunk.empty:
            continue

        for dom, quota in desired_per_domain.items():
            sub = chunk.loc[chunk["domain"] == dom].copy()
            if sub.empty:
                continue
            sub["_sample_key"] = rng.random(len(sub))
            keep = pd.concat([reservoirs[dom], sub], ignore_index=True)
            if len(keep) > quota:
                keep = keep.nsmallest(quota, "_sample_key").reset_index(drop=True)
            reservoirs[dom] = keep

    out_parts = []
    for dom, quota in desired_per_domain.items():
        sub = reservoirs[dom].drop(columns="_sample_key", errors="ignore").copy()
        if len(sub) < quota:
            log_step(f"Requested {quota} rows for {dom}, retained {len(sub)} available rows")
        if sub.empty:
            raise RuntimeError(f"Sampling returned no rows for domain={dom}.")
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_spatial_block_id(
    df: pd.DataFrame,
    *,
    block_size_m: float = SPATIAL_BLOCK_SIZE_M,
    xcol: str = "easting",
    ycol: str = "northing",
) -> pd.Series:
    e = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    out = np.full(len(df), None, dtype=object)
    ok = np.isfinite(e) & np.isfinite(n)
    if ok.any():
        bx = np.floor(e[ok] / block_size_m).astype(np.int64)
        by = np.floor(n[ok] / block_size_m).astype(np.int64)
        out[ok] = bx.astype(str) + "_" + by.astype(str)
    return pd.Series(out, index=df.index, name="block_id", dtype="object")


def compute_quantile_thresholds(
    values: np.ndarray,
    quantiles: list[float],
) -> dict[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("No valid grad_mag values available for thresholding.")
    return {float(q): float(np.nanquantile(vals, q)) for q in quantiles}


def compute_pf_threshold_from_full_csv(
    csv_path: Path,
    q: float,
    chunksize: int = CHUNKSIZE,
) -> float:
    parts = []
    usecols = ["Perma_Distr_map", "grad_mag"]
    for chunk in iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label=f"legacy threshold q={q:.2f}",
    ):
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy() == 1
        vals = pd.to_numeric(chunk["grad_mag"], errors="coerce").to_numpy(dtype=float)
        mask = pf & np.isfinite(vals)
        if mask.any():
            parts.append(vals[mask])
    if not parts:
        raise RuntimeError("No valid PF grad_mag values found.")
    return compute_quantile_thresholds(np.concatenate(parts), [q])[q]


def resolve_region_thresholds(
    csv_path: Path,
    cache_path: Path,
    quantiles: list[float],
    chunksize: int = CHUNKSIZE,
) -> dict[float, float]:
    quantiles = [float(q) for q in quantiles]
    cache_signature = {
        "csv": file_signature(csv_path),
        "quantiles": quantiles,
    }

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("cache_signature") == cache_signature:
                cached = payload.get("thresholds_by_q", {})
                out = {float(q): float(cached[str(q)]) for q in quantiles}
                if len(out) == len(quantiles):
                    log_step(f"Loaded cached full-region thresholds: {cache_path}")
                    return out
        except Exception as exc:
            log_step(f"Warning: failed to read cached region thresholds at {cache_path}: {exc}")

    log_step("Computing full-region grad_mag thresholds from CSV")
    parts = []
    for chunk in iter_csv_chunks(
        csv_path,
        usecols=["grad_mag"],
        chunksize=chunksize,
        label="full-region threshold",
    ):
        vals = pd.to_numeric(chunk["grad_mag"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            parts.append(vals)
    if not parts:
        raise RuntimeError("No valid regional grad_mag values found.")

    thresholds = compute_quantile_thresholds(np.concatenate(parts), quantiles)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {
            "cache_signature": cache_signature,
            "thresholds_by_q": {str(q): float(v) for q, v in thresholds.items()},
        },
        indent=2,
    ))
    log_step(f"Saved full-region thresholds: {cache_path}")
    return thresholds


def choose_spatial_holdout_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    block_size_m: float,
    seed: int = SEED,
    q_for_validity: float = MAIN_THRESHOLD_Q,
    threshold_grad: float | None = None,
    max_tries: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    work = df.reset_index(drop=True).copy()
    work["block_id"] = make_spatial_block_id(work, block_size_m=block_size_m)
    work = work.loc[work["block_id"].notna()].reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No PF rows have valid coordinates for spatial block splitting.")

    groups = work["block_id"].astype(str).to_numpy()
    grad_mag = work["grad_mag"].to_numpy(dtype=float)
    unique_blocks = np.unique(groups)
    if unique_blocks.size < 2:
        raise RuntimeError("Need at least two spatial blocks for train/test holdout.")

    target_test_n = test_size * len(work)
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score = math.inf

    for offset in range(max_tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        train_idx, test_idx = next(splitter.split(work, groups=groups))
        thr = (
            float(threshold_grad)
            if threshold_grad is not None
            else compute_quantile_thresholds(grad_mag[train_idx], [q_for_validity])[q_for_validity]
        )
        y_train = (grad_mag[train_idx] >= thr).astype(int)
        y_test = (grad_mag[test_idx] >= thr).astype(int)
        if y_train.min() == y_train.max() or y_test.min() == y_test.max():
            continue
        score = abs(len(test_idx) - target_test_n)
        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)
        if score <= 0.01 * len(work):
            break

    if best_split is None:
        raise RuntimeError("Failed to find a valid spatial block holdout split with both classes present.")

    return best_split


class FixedGroupKFold:
    def __init__(self, groups: np.ndarray, n_splits: int = 5):
        groups = np.asarray(groups, dtype=object)
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError("Need at least two groups for grouped CV.")
        self.groups = groups
        self.n_splits = max(2, min(int(n_splits), int(unique_groups.size)))

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.groups):
            raise ValueError("FixedGroupKFold received X with mismatched length.")
        splitter = GroupKFold(n_splits=self.n_splits)
        yield from splitter.split(np.zeros(len(self.groups)), y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def audit_feature_missingness(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for group_name, features in feature_groups.items():
        for feature in features:
            vals = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            finite_n = int(np.isfinite(vals).sum())
            rows.append({
                "feature_set": group_name,
                "feature": feature,
                "finite_n": finite_n,
                "missing_frac": float(1.0 - finite_n / max(len(df), 1)),
            })
    return pd.DataFrame(rows).sort_values(["missing_frac", "feature"], ascending=[False, True]).reset_index(drop=True)


def drop_high_missing_features(
    feature_groups: dict[str, list[str]],
    missing_df: pd.DataFrame,
    *,
    max_missing_frac: float,
) -> tuple[dict[str, list[str]], list[str]]:
    dropped = sorted(
        missing_df.loc[missing_df["missing_frac"] > max_missing_frac, "feature"].drop_duplicates().tolist()
    )
    if not dropped:
        return {k: list(v) for k, v in feature_groups.items()}, []

    kept = {
        name: [feature for feature in features if feature not in dropped]
        for name, features in feature_groups.items()
    }
    return kept, dropped


def classification_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "roc_auc": np.nan,
        "ap": np.nan,
        "brier": float(brier_score_loss(y_true, p)),
    }
    if np.unique(y_true).size >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    return out


def make_model_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return df.loc[:, feature_names].astype(np.float32, copy=True)


def resolve_sample(
    csv_path: Path,
    cache_path: Path,
    sample_pf: int,
    sample_npf: int,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    if cache_path.exists():
        log_step(f"Loading cached susceptibility sample: {cache_path}")
        df = pd.read_csv(cache_path)
        return engineer_features(df)

    usecols = [
        "Perma_Distr_map",
        "easting",
        "northing",
        "longitude",
        "latitude",
        "grad_mag",
        "d_u",
    ] + RAW_FEATURES

    df = build_stratified_sample(
        csv_path=csv_path,
        usecols=usecols,
        desired_per_domain={"pf": sample_pf, "npf": sample_npf},
        chunksize=chunksize,
        seed=SEED,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, compression="gzip")
    log_step(f"Saved sampled PF/NPF dataset: {cache_path}")
    return df


# -----------------------------------------------------------------------------
# Transition metrics on sampled points
# -----------------------------------------------------------------------------
def local_transition_metrics(
    df: pd.DataFrame,
    vars_for_transition: list[str],
    xcol: str = "easting",
    ycol: str = "northing",
    n_neighbors: int = 21,
) -> pd.DataFrame:
    out = df.copy()

    coords = out[[xcol, ycol]].to_numpy(dtype=float)
    ok_xy = np.isfinite(coords).all(axis=1)
    if ok_xy.sum() < n_neighbors:
        raise RuntimeError("Not enough valid coordinates for local transition features.")

    coords_valid = coords[ok_xy]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(coords_valid)
    _, idx_valid = nbrs.kneighbors(coords_valid, return_distance=True)

    full_idx = np.full((len(out), n_neighbors), -1, dtype=int)
    full_idx[ok_xy] = np.flatnonzero(ok_xy)[idx_valid]

    x0 = coords[:, 0][:, None]
    y0 = coords[:, 1][:, None]
    x_nb = coords[full_idx.clip(min=0), 0]
    y_nb = coords[full_idx.clip(min=0), 1]
    dx = x_nb - x0
    dy = y_nb - y0
    valid_geo = full_idx >= 0

    for var in vars_for_transition:
        z = pd.to_numeric(out[var], errors="coerce").to_numpy(dtype=float)
        z0 = z[:, None]
        z_nb = z[full_idx.clip(min=0)]
        valid = valid_geo & np.isfinite(z0) & np.isfinite(z_nb)
        z_nb_masked = np.where(valid, z_nb, np.nan)
        out[f"{var}__lstd"] = np.nanstd(z_nb_masked, axis=1)

        dz = np.where(valid, z_nb - z0, 0.0)
        vx = np.where(valid, dx, 0.0)
        vy = np.where(valid, dy, 0.0)
        sxx = np.sum(vx * vx, axis=1)
        syy = np.sum(vy * vy, axis=1)
        sxy = np.sum(vx * vy, axis=1)
        sxz = np.sum(vx * dz, axis=1)
        syz = np.sum(vy * dz, axis=1)
        det = sxx * syy - sxy * sxy
        enough = valid.sum(axis=1) >= 5
        good = enough & np.isfinite(det) & (np.abs(det) > 1e-12)

        bx = np.full(len(out), np.nan, dtype=float)
        by = np.full(len(out), np.nan, dtype=float)
        bx[good] = (sxz[good] * syy[good] - syz[good] * sxy[good]) / det[good]
        by[good] = (syz[good] * sxx[good] - sxz[good] * sxy[good]) / det[good]
        out[f"{var}__gmag"] = np.sqrt(bx * bx + by * by)

    return out


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_model_cache_signature(
    *,
    csv_path: Path,
    sample_cache: Path,
    raw_features: list[str],
    contrast_features: list[str],
    combined_features: list[str],
    threshold_grad_095: float,
    transition_window_size: int,
    transition_base_vars: list[str],
    neighbors_trans: int,
    block_size_m: float,
    max_missing_frac: float,
    model_n_jobs: int,
    test_size: float,
) -> dict[str, object]:
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "seed": SEED,
        "csv": file_signature(csv_path),
        "sample_cache": file_signature(sample_cache),
        "raw_features": list(raw_features),
        "contrast_features": list(contrast_features),
        "combined_features": list(combined_features),
        "threshold_q": MAIN_THRESHOLD_Q,
        "threshold_grad_095": round(float(threshold_grad_095), 12),
        "threshold_sweep_qs": list(THRESHOLD_SWEEP_QS),
        "threshold_source": "full_region_all_domains",
        "transition_window_size": int(transition_window_size),
        "transition_base_vars": list(transition_base_vars),
        "neighbors_trans": int(neighbors_trans),
        "block_size_m": float(block_size_m),
        "max_missing_frac": float(max_missing_frac),
        "model_n_jobs": int(model_n_jobs),
        "test_size": float(test_size),
        "xgboost_available": bool(HAVE_XGB),
    }


def load_cached_training_artifacts(
    cache_path: Path,
    cache_signature: dict[str, object],
) -> dict[str, object] | None:
    if not cache_path.exists():
        return None

    try:
        try:
            payload = joblib.load(cache_path)
        except OSError:
            with gzip.open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
    except Exception as exc:
        log_step(f"Warning: failed to load model cache {cache_path}: {exc}")
        return None

    required_keys = {
        "cache_signature",
        "suite_fitted",
        "suite_metrics",
        "suite_probs",
        "full_stack",
        "sweep_df",
        "magt_ablation_df",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        log_step(f"Warning: model cache at {cache_path} is incomplete; retraining.")
        return None

    if payload["cache_signature"] != cache_signature:
        log_step(f"Model cache mismatch at {cache_path}; retraining.")
        return None

    return payload


def save_training_artifacts(cache_path: Path, payload: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, cache_path, compress=("gzip", 3))


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
def build_models(pos_weight: float, *, stack_cv=None, n_jobs: int = DEFAULT_MODEL_N_JOBS) -> dict[str, object]:
    models: dict[str, object] = {
        "Logit": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=SEED,
            )),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=350,
                random_state=SEED,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ]),
        "ET": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=400,
                random_state=SEED,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )),
        ]),
    }

    if HAVE_XGB:
        models["XGB"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
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
            )),
        ])
    else:
        models["HGB"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=0.05,
                max_depth=6,
                random_state=SEED,
            )),
        ])

    estimators = [(k.lower(), clone(v)) for k, v in models.items()]
    models["Stack"] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2500,
            random_state=SEED,
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=stack_cv if stack_cv is not None else 5,
        n_jobs=n_jobs,
    )
    return models


def fit_and_evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[object, dict[str, float], np.ndarray]:
    est = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X_train, y_train)
    p = est.predict_proba(X_test)[:, 1]
    return est, classification_metrics(y_test, p), p


def fit_and_evaluate_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    *,
    n_jobs: int = DEFAULT_MODEL_N_JOBS,
) -> tuple[dict[str, object], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    pos_rate = max(float(np.mean(y_train)), 1e-6)
    pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
    stack_cv = FixedGroupKFold(groups_train, n_splits=5)
    models = build_models(pos_weight=pos_weight, stack_cv=stack_cv, n_jobs=n_jobs)

    fitted = {}
    metrics = {}
    probs = {}

    for name, model in models.items():
        est, model_metrics, p = fit_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        fitted[name] = est
        probs[name] = p
        metrics[name] = model_metrics

    return fitted, metrics, probs


# -----------------------------------------------------------------------------
# Full-grid transition metrics and prediction map
# -----------------------------------------------------------------------------
def neighbors_to_window_size(n_neighbors: int) -> int:
    width = max(3, int(np.ceil(np.sqrt(max(n_neighbors, 1)))))
    if width % 2 == 0:
        width += 1
    return width


def ensure_transition_rasters(
    base_dir: Path,
    csv_path: Path,
    cache_dir: Path,
    vars_for_transition: list[str],
    chunksize: int,
    window_size: int,
) -> tuple[dict, dict[str, Path], dict[str, dict[str, Path]]]:
    grid = build_or_load_grid(base_dir, csv_path)

    raw_raster_dir = cache_dir / "raw_rasters"
    metric_raster_dir = cache_dir / f"metric_rasters_w{window_size:02d}"
    raw_raster_dir.mkdir(parents=True, exist_ok=True)
    metric_raster_dir.mkdir(parents=True, exist_ok=True)

    mean_vars = ["longitude", "latitude"] + vars_for_transition
    mean_paths = {var: raw_raster_dir / f"{var}_mean_f32.memmap" for var in mean_vars}
    missing_means = [var for var, path in mean_paths.items() if not path.exists()]
    if missing_means:
        cache_paths = build_raw_raster_cache(
            csv_path=csv_path,
            out_dir=raw_raster_dir,
            vars_to_use=mean_vars,
            grid=grid,
            chunksize=chunksize,
        )
        mean_paths = finalize_mean_rasters(
            cache_paths=cache_paths,
            out_dir=raw_raster_dir,
            vars_to_use=mean_vars,
            grid=grid,
        )

    metric_paths = derive_transition_metric_rasters(
        mean_paths={var: mean_paths[var] for var in vars_for_transition},
        out_dir=metric_raster_dir,
        vars_to_use=vars_for_transition,
        grid=grid,
        window_size=window_size,
    )
    return grid, mean_paths, metric_paths


def load_metric_memmaps(
    metric_paths: dict[str, dict[str, Path]],
    vars_for_transition: list[str],
    grid: dict,
) -> dict[str, dict[str, np.memmap]]:
    shape = (int(grid["nrows"]), int(grid["ncols"]))
    return {
        var: {
            "lstd": open_memmap(metric_paths["lstd"][var], dtype="float32", mode="r", shape=shape),
            "gmag": open_memmap(metric_paths["gmag"][var], dtype="float32", mode="r", shape=shape),
        }
        for var in vars_for_transition
    }


def attach_raster_transition_metrics(
    df: pd.DataFrame,
    metric_paths: dict[str, dict[str, Path]],
    grid: dict,
    vars_for_transition: list[str],
) -> pd.DataFrame:
    out = df.copy()
    e = pd.to_numeric(out["easting"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(out["northing"], errors="coerce").to_numpy(dtype=float)
    row, col = en_to_rc(
        e,
        n,
        res=float(grid["res"]),
        gx0=int(grid["gx0"]),
        gy1=int(grid["gy1"]),
    )

    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    ok = np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
    metric_maps = load_metric_memmaps(metric_paths, vars_for_transition, grid)

    for var in vars_for_transition:
        lstd = np.full(len(out), np.nan, dtype=np.float32)
        gmag = np.full(len(out), np.nan, dtype=np.float32)
        if ok.any():
            lstd[ok] = metric_maps[var]["lstd"][row[ok], col[ok]]
            gmag[ok] = metric_maps[var]["gmag"][row[ok], col[ok]]
        out[f"{var}__lstd"] = lstd
        out[f"{var}__gmag"] = gmag

    return out


def build_pf_prediction_raster(
    csv_path: Path,
    prob_path: Path,
    model,
    grid: dict,
    metric_paths: dict[str, dict[str, Path]],
    vars_for_transition: list[str],
    feature_names: list[str],
    chunksize: int,
) -> Path:
    log_step("Building full-grid PF susceptibility raster")
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    shape = (nrows, ncols)
    prob_mm = open_memmap(prob_path, dtype="float32", mode="w+", shape=shape)
    prob_mm[:] = np.nan

    metric_maps = load_metric_memmaps(metric_paths, vars_for_transition, grid)
    requested_raw_features = [feat for feat in RAW_FEATURES if feat in feature_names]
    usecols = ["Perma_Distr_map", "easting", "northing"] + requested_raw_features

    for chunk in iter_csv_chunks(
        csv_path,
        usecols=usecols,
        chunksize=chunksize,
        label="prediction raster",
    ):
        chunk = engineer_features(chunk)
        chunk = chunk.loc[chunk["domain"] == "pf"].copy()
        if chunk.empty:
            continue

        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        row, col = en_to_rc(
            e,
            n,
            res=float(grid["res"]),
            gx0=int(grid["gx0"]),
            gy1=int(grid["gy1"]),
        )
        ok = np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
        if not ok.any():
            continue

        features = {
            feat: pd.to_numeric(chunk[feat], errors="coerce").to_numpy(dtype=np.float32)
            for feat in requested_raw_features
        }
        for var in vars_for_transition:
            lstd = np.full(len(chunk), np.nan, dtype=np.float32)
            gmag = np.full(len(chunk), np.nan, dtype=np.float32)
            lstd[ok] = metric_maps[var]["lstd"][row[ok], col[ok]]
            gmag[ok] = metric_maps[var]["gmag"][row[ok], col[ok]]
            features[f"{var}__lstd"] = lstd
            features[f"{var}__gmag"] = gmag

        X_chunk = pd.DataFrame(features, columns=feature_names, dtype=np.float32)
        probs = model.predict_proba(X_chunk.iloc[np.flatnonzero(ok)])[:, 1].astype(np.float32)
        prob_mm[row[ok], col[ok]] = probs

    prob_mm.flush()
    log_step(f"Saved PF susceptibility raster: {prob_path}")
    return prob_path


def axis_profile_from_raster(path: Path, grid: dict, axis: int) -> np.ndarray:
    mm = open_memmap(
        path,
        dtype="float32",
        mode="r",
        shape=(int(grid["nrows"]), int(grid["ncols"])),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile = np.nanmean(mm, axis=axis)
    return np.asarray(profile, dtype=float)


def choose_geo_ticks(coord_profile: np.ndarray, max_ticks: int = 6) -> tuple[np.ndarray, list[str]]:
    valid_idx = np.flatnonzero(np.isfinite(coord_profile))
    if valid_idx.size == 0:
        return np.array([], dtype=int), []

    coord_valid = coord_profile[valid_idx]
    lo = float(coord_valid.min())
    hi = float(coord_valid.max())
    n_ticks = max(2, min(max_ticks, valid_idx.size))
    targets = np.linspace(lo, hi, n_ticks)

    picks = []
    for target in targets:
        near = valid_idx[np.argmin(np.abs(coord_profile[valid_idx] - target))]
        picks.append(int(near))
    picks = np.array(sorted(set(picks)), dtype=int)

    labels = []
    for idx in picks:
        value = coord_profile[idx]
        if not np.isfinite(value):
            labels.append("")
        else:
            labels.append(f"{abs(value):.0f}°{'E' if value >= 0 else 'W'}")
    return picks, labels


def choose_lat_ticks(lat_profile: np.ndarray, max_ticks: int = 6) -> tuple[np.ndarray, list[str]]:
    valid_idx = np.flatnonzero(np.isfinite(lat_profile))
    if valid_idx.size == 0:
        return np.array([], dtype=int), []

    coord_valid = lat_profile[valid_idx]
    lo = float(coord_valid.min())
    hi = float(coord_valid.max())
    n_ticks = max(2, min(max_ticks, valid_idx.size))
    targets = np.linspace(lo, hi, n_ticks)

    picks = []
    for target in targets:
        near = valid_idx[np.argmin(np.abs(lat_profile[valid_idx] - target))]
        picks.append(int(near))
    picks = np.array(sorted(set(picks)), dtype=int)

    labels = []
    for idx in picks:
        value = lat_profile[idx]
        if not np.isfinite(value):
            labels.append("")
        else:
            labels.append(f"{abs(value):.0f}°{'N' if value >= 0 else 'S'}")
    return picks, labels


def apply_projected_geo_ticks(
    ax,
    grid: dict,
    mean_paths: dict[str, Path],
) -> None:
    res = float(grid["res"])
    min_e = float(grid["min_e"])
    max_n = float(grid["max_n"])

    lon_profile = axis_profile_from_raster(mean_paths["longitude"], grid, axis=0)
    lat_profile = axis_profile_from_raster(mean_paths["latitude"], grid, axis=1)

    col_idx, x_labels = choose_geo_ticks(lon_profile)
    row_idx, y_labels = choose_lat_ticks(lat_profile)

    if len(col_idx):
        x_pos = min_e + (col_idx + 0.5) * res
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

    if len(row_idx):
        y_pos = max_n - (row_idx + 0.5) * res
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

    ax.set_xlabel("Longitude (°)", fontweight="bold")
    ax.set_ylabel("Latitude (°)", fontweight="bold")


def build_pf_hotspot_density_raster(
    csv_path: Path,
    density_path: Path,
    *,
    grid: dict,
    hotspot_threshold: float,
    chunksize: int,
    sigma_px: float = HOTSPOT_DENSITY_SIGMA_PX,
) -> Path:
    log_step("Building full-grid PF hotspot-density raster")
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    density_counts = np.zeros((nrows, ncols), dtype=np.float32)

    for chunk in iter_csv_chunks(
        csv_path,
        usecols=["Perma_Distr_map", "easting", "northing", "grad_mag"],
        chunksize=chunksize,
        label="hotspot density",
    ):
        chunk = engineer_features(chunk)
        grad = pd.to_numeric(chunk["grad_mag"], errors="coerce").to_numpy(dtype=float)
        pf = chunk["domain"].to_numpy(dtype=object) == "pf"
        hot = pf & np.isfinite(grad) & (grad >= float(hotspot_threshold))
        if not hot.any():
            continue

        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        row, col = en_to_rc(
            e,
            n,
            res=float(grid["res"]),
            gx0=int(grid["gx0"]),
            gy1=int(grid["gy1"]),
        )
        ok = hot & np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
        if ok.any():
            np.add.at(density_counts, (row[ok], col[ok]), 1.0)

    if np.any(density_counts > 0):
        density = gaussian_filter(density_counts, sigma=float(sigma_px), mode="nearest").astype(np.float32, copy=False)
        max_val = float(np.nanmax(density))
        if np.isfinite(max_val) and max_val > 0:
            density /= max_val
    else:
        density = density_counts

    density_mm = open_memmap(density_path, dtype="float32", mode="w+", shape=(nrows, ncols))
    density_mm[:] = density
    density_mm.flush()
    log_step(f"Saved PF hotspot-density raster: {density_path}")
    return density_path


def build_pf_mask_raster(
    csv_path: Path,
    mask_path: Path,
    *,
    grid: dict,
    chunksize: int,
) -> Path:
    if mask_path.exists():
        log_step(f"Using cached PF mask raster: {mask_path}")
        return mask_path

    log_step("Building full-grid PF mask raster")
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    mask_mm = open_memmap(mask_path, dtype="uint8", mode="w+", shape=(nrows, ncols))
    mask_mm[:] = np.uint8(0)

    for chunk in iter_csv_chunks(
        csv_path,
        usecols=["Perma_Distr_map", "easting", "northing"],
        chunksize=chunksize,
        label="pf mask",
    ):
        pf = pd.to_numeric(chunk["Perma_Distr_map"], errors="coerce").to_numpy(dtype=float) == 1.0
        if not pf.any():
            continue

        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy(dtype=float)
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy(dtype=float)
        row, col = en_to_rc(
            e,
            n,
            res=float(grid["res"]),
            gx0=int(grid["gx0"]),
            gy1=int(grid["gy1"]),
        )
        ok = pf & np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
        if ok.any():
            mask_mm[row[ok], col[ok]] = np.uint8(1)

    mask_mm.flush()
    log_step(f"Saved PF mask raster: {mask_path}")
    return mask_path


def predict_proba_batched(model, X: pd.DataFrame, batch_size: int = 50000) -> np.ndarray:
    out = np.full(len(X), np.nan, dtype=np.float32)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        out[start:stop] = model.predict_proba(X.iloc[start:stop])[:, 1].astype(np.float32)
    return out


def select_zoom_windows(
    pf_df: pd.DataFrame,
    *,
    n_windows: int,
    half_size_m: float,
    min_spacing_m: float,
    min_points: int,
    min_hotspots: int,
    forbidden_boxes: list[tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    ranked = pf_df.sort_values(["susceptibility", "grad_mag"], ascending=[False, False]).reset_index(drop=True)
    windows = []
    centers: list[tuple[float, float]] = []

    e_all = pf_df["easting"].to_numpy(dtype=float)
    n_all = pf_df["northing"].to_numpy(dtype=float)
    hotspot_all = pf_df["is_hotspot_095"].to_numpy(dtype=int)
    score_all = pf_df["susceptibility"].to_numpy(dtype=float)

    for row in ranked.itertuples(index=False):
        if not np.isfinite(row.easting) or not np.isfinite(row.northing):
            continue
        if any((row.easting - cx) ** 2 + (row.northing - cy) ** 2 < min_spacing_m ** 2 for cx, cy in centers):
            continue

        xmin = float(row.easting - half_size_m)
        xmax = float(row.easting + half_size_m)
        ymin = float(row.northing - half_size_m)
        ymax = float(row.northing + half_size_m)
        if forbidden_boxes and any(
            not (xmax < fxmin or xmin > fxmax or ymax < fymin or ymin > fymax)
            for fxmin, fxmax, fymin, fymax in forbidden_boxes
        ):
            continue

        mask = (
            (np.abs(e_all - row.easting) <= half_size_m)
            & (np.abs(n_all - row.northing) <= half_size_m)
        )
        n_points = int(mask.sum())
        n_hot = int(hotspot_all[mask].sum())
        if n_points < min_points or n_hot < min_hotspots:
            continue

        windows.append(
            {
                "window_id": f"W{len(windows) + 1}",
                "center_easting": float(row.easting),
                "center_northing": float(row.northing),
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "n_points": n_points,
                "n_hotspots": n_hot,
                "hotspot_fraction": float(hotspot_all[mask].mean()),
                "mean_susceptibility": float(np.nanmean(score_all[mask])),
                "max_susceptibility": float(np.nanmax(score_all[mask])),
            }
        )
        centers.append((float(row.easting), float(row.northing)))
        if len(windows) >= n_windows:
            break

    if len(windows) < n_windows:
        raise RuntimeError(
            f"Only found {len(windows)} valid zoom windows; requested {n_windows}. "
            "Try reducing the zoom filters."
        )
    return pd.DataFrame(windows)


def crop_window(raster_mm: np.memmap, grid: dict, window: pd.Series) -> tuple[np.ndarray, list[float]] | None:
    res = float(grid["res"])
    min_e = float(grid["min_e"])
    max_n = float(grid["max_n"])
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])

    col0 = max(0, int(np.floor((float(window["xmin"]) - min_e) / res)))
    col1 = min(ncols, int(np.ceil((float(window["xmax"]) - min_e) / res)))
    row0 = max(0, int(np.floor((max_n - float(window["ymax"])) / res)))
    row1 = min(nrows, int(np.ceil((max_n - float(window["ymin"])) / res)))
    if row1 <= row0 or col1 <= col0:
        return None

    sub = np.array(raster_mm[row0:row1, col0:col1], copy=False).astype(np.float32)
    sub_min_e = min_e + col0 * res
    sub_max_n = max_n - row0 * res
    extent = get_extent(
        min_e=sub_min_e,
        max_n=sub_max_n,
        nrows=row1 - row0,
        ncols=col1 - col0,
        res=res,
    )
    return sub, extent


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_panel_A(ax, sample_df: pd.DataFrame, region_thr_grad: float) -> None:
    pf = sample_df.loc[sample_df["domain"] == "pf", "grad_mag_km"].dropna()
    npf = sample_df.loc[sample_df["domain"] == "npf", "grad_mag_km"].dropna()
    all_vals = pd.concat([pf, npf], ignore_index=True)
    region_thr_grad_km = float(region_thr_grad) * GRAD_MAG_DISPLAY_SCALE
    lo, hi = robust_clip(all_vals.to_numpy(dtype=float), 0.5, 99.5)
    bins = np.linspace(lo, hi, 80)

    def probability_weights(values: pd.Series) -> np.ndarray | None:
        if len(values) == 0:
            return None
        return np.full(len(values), 1.0 / len(values), dtype=float)

    ax.hist(
        pf,
        bins=bins,
        weights=probability_weights(pf),
        histtype="step",
        linewidth=1.2,
        color=PF_COLOR,
        label="Permafrost",
    )
    ax.hist(
        npf,
        bins=bins,
        weights=probability_weights(npf),
        histtype="step",
        linewidth=1.2,
        color=NPF_COLOR,
        label="Non-permafrost",
    )
    ax.hist(
        all_vals,
        bins=bins,
        weights=probability_weights(all_vals),
        histtype="step",
        linewidth=3.2,
        color="k",
        label="All",
    )
    ax.axvline(region_thr_grad_km, color="0.35", linestyle="--", linewidth=1.2)
    ax.set_title("Gradient-magnitude regime contrast", fontweight="bold")
    ax.set_xlabel(r"$|\nabla d_u|$ (mm/yr/km)", fontweight="bold")
    ax.set_ylabel("Probability", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    style_open_axes(ax)
    add_panel_label(ax, "A")
    ymax = float(ax.get_ylim()[1])
    ax.text(
        region_thr_grad_km + 0.008 * (hi - lo),
        0.93 * ymax,
        f"Region q={MAIN_THRESHOLD_Q:.2f}",
        ha="left",
        va="top",
        fontsize=8,
        rotation=90,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor="0.6"),
    )

def plot_panel_B(ax, perf_df: pd.DataFrame) -> None:
    model_order = perf_df["model"].drop_duplicates().tolist()
    feature_order = ["raw", "contrast", "combined"]

    roc_mat = np.full((len(model_order), len(feature_order)), np.nan, dtype=float)
    ann = [["" for _ in feature_order] for _ in model_order]

    for i, model in enumerate(model_order):
        for j, featset in enumerate(feature_order):
            sub = perf_df.loc[(perf_df["model"] == model) & (perf_df["feature_set"] == featset)]
            if sub.empty:
                continue
            roc = float(sub["roc_auc"].iloc[0])
            ap = float(sub["ap"].iloc[0])
            roc_mat[i, j] = roc
            ann[i][j] = f"{roc:.2f}\nAP {ap:.2f}"

    vmin = min(0.60, float(np.nanmin(roc_mat)))
    vmax = max(0.70, float(np.nanmax(roc_mat)))
    im = ax.imshow(roc_mat, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels([s.title() for s in feature_order], fontweight="bold")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, fontweight="bold")

    for i in range(len(model_order)):
        for j in range(len(feature_order)):
            if np.isfinite(roc_mat[i, j]):
                txt = ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=8, fontweight="bold")
                txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_title("Model comparison: raw vs contrast vs combined", fontweight="bold")
    ax.set_xlabel("Predictor set", fontweight="bold")
    ax.set_ylabel("Classifier", fontweight="bold")
    style_open_axes(ax)
    add_panel_label(ax, "B")

    cax = ax.inset_axes([-0.20, 0.13, 0.045, 0.72], transform=ax.transAxes)
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_ticks([0.60, 0.65, 0.70])
    cb.ax.tick_params(labelsize=7, length=2)


def plot_panel_C(
    ax_hot,
    ax_pred,
    pf_mask_path: Path,
    hotspot_density_path: Path,
    prob_path: Path,
    grid: dict,
) -> None:
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    stride = choose_stride(nrows, ncols, target_max=1400)
    prob_mm = open_memmap(prob_path, dtype="float32", mode="r", shape=(nrows, ncols))
    hotspot_mm = open_memmap(hotspot_density_path, dtype="float32", mode="r", shape=(nrows, ncols))
    pf_mask_mm = open_memmap(pf_mask_path, dtype="uint8", mode="r", shape=(nrows, ncols))
    prob = np.array(prob_mm[::stride, ::stride], copy=False).astype(np.float32)
    hotspot = np.array(hotspot_mm[::stride, ::stride], copy=False).astype(np.float32)
    pf_mask = np.array(pf_mask_mm[::stride, ::stride], copy=False) == np.uint8(1)
    extent = get_extent(
        min_e=float(grid["min_e"]),
        max_n=float(grid["max_n"]),
        nrows=nrows,
        ncols=ncols,
        res=float(grid["res"]),
    )

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(alpha=0.0)
    norm = Normalize(vmin=0.0, vmax=1.0)
    hotspot_masked = np.ma.masked_invalid(np.where(pf_mask, hotspot, np.nan))
    prob_masked = np.ma.masked_invalid(np.where(pf_mask, prob, np.nan))

    ax_hot.imshow(
        hotspot_masked,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    im = ax_pred.imshow(
        prob_masked,
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax_hot.set_title("Observed Permafrost Hotspot\nKernel Density (0-1 Scaled)", fontweight="bold")
    ax_pred.set_title("Stack Combined Model\nPredicted Susceptibility", fontweight="bold")
    ax_hot.set_xlabel("Easting (m)", fontweight="bold")
    ax_hot.set_ylabel("Northing (m)", fontweight="bold")
    ax_pred.set_xlabel("Easting (m)", fontweight="bold")
    ax_pred.set_ylabel("")
    ax_pred.tick_params(labelleft=False)
    for ax in [ax_hot, ax_pred]:
        style_open_axes(ax)
        ax.set_aspect("equal")
    add_panel_label(ax_hot, "C")

    cax = ax_pred.inset_axes([1.03, 0.22, 0.032, 0.46], transform=ax_pred.transAxes)
    cb = plt.colorbar(im, cax=cax, orientation="vertical")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=7, length=2)


def plot_panel_E(ax, importance_df: pd.DataFrame, top_n: int = 10) -> None:
    top = importance_df.head(top_n).iloc[::-1].copy()
    colors = [
        CONTRAST_COLOR if ("__lstd" in f or "__gmag" in f) else RAW_COLOR
        for f in top["feature"]
    ]
    labels = []
    for f in top["feature"]:
        if "__lstd" in f:
            base = f.replace("__lstd", "")
            labels.append(f"{FEATURE_LABELS.get(base, base)} (lstd)")
        elif "__gmag" in f:
            base = f.replace("__gmag", "")
            labels.append(f"{FEATURE_LABELS.get(base, base)} (gmag)")
        else:
            labels.append(FEATURE_LABELS.get(f, f))

    xerr = top["importance_std"] if "importance_std" in top.columns else None
    ax.barh(
        labels,
        top["importance"],
        xerr=xerr,
        color=colors,
        alpha=0.92,
        ecolor="0.25",
        capsize=3,
    )
    ax.set_title(f"Top {len(top)} susceptibility predictors", fontweight="bold")
    ax.set_xlabel("Permutation importance (AP drop)", fontweight="bold")
    style_open_axes(ax)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    add_panel_label(ax, "D")

    handles = [
        plt.Line2D([0], [0], color=RAW_COLOR, linewidth=6, label="Raw"),
        plt.Line2D([0], [0], color=CONTRAST_COLOR, linewidth=6, label="Contrast"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

def plot_window_raster(
    ax,
    raster_mm: np.memmap,
    pf_mask_mm: np.memmap,
    grid: dict,
    window: pd.Series,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    title_text: str = "",
    ylabel_text: str = "",
    cmap="plasma",
    norm=None,
    border_color: str | None = None,
    stats_text: str | None = None,
    transform=None,
    panel_label: str | None = None,
) :
    cropped = crop_window(raster_mm, grid, window)
    cropped_mask = crop_window(pf_mask_mm, grid, window)
    if cropped is None or cropped_mask is None:
        ax.text(0.5, 0.5, "No raster coverage", transform=ax.transAxes, ha="center", va="center")
        style_open_axes(ax)
        if panel_label:
            add_panel_label(ax, panel_label)
        return None

    sub_map, extent = cropped
    sub_pf_mask, _ = cropped_mask
    if transform is not None:
        sub_map = transform(sub_map)
    sub_map = np.where(sub_pf_mask == np.uint8(1), sub_map, np.nan)
    cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    cmap.set_bad(alpha=0.0)
    im = ax.imshow(
        np.ma.masked_invalid(sub_map),
        extent=extent,
        origin="upper",
        cmap=cmap,
        norm=norm if norm is not None else Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    color = border_color or WINDOW_COLORS[(int(str(window["window_id"]).replace("W", "")) - 1) % len(WINDOW_COLORS)]
    style_open_axes(ax)
    for side, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_edgecolor(color)
        if side in {"top", "right"}:
            spine.set_alpha(0.85)

    ax.set_xlim(float(window["xmin"]), float(window["xmax"]))
    ax.set_ylim(float(window["ymin"]), float(window["ymax"]))
    ax.set_xlabel("Easting (m)" if show_xlabel else "", fontweight="bold")
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(ylabel_text or "Northing (m)", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_title(title_text, fontweight="bold", color=color if title_text else "black")
    if stats_text:
        ax.text(
            0.98,
            0.03,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor=color),
        )
    if panel_label:
        add_panel_label(ax, panel_label)
    return im


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 6 susceptibility modeling")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--sample-pf", type=int, default=90000)
    parser.add_argument("--sample-npf", type=int, default=90000)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--neighbors-trans", type=int, default=21)
    parser.add_argument("--block-size-km", type=float, default=SPATIAL_BLOCK_SIZE_M / 1000.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-missing-frac", type=float, default=MAX_FEATURE_MISSING_FRAC)
    parser.add_argument("--model-n-jobs", type=int, default=DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--exclude-magt", action="store_true")
    parser.add_argument("--n-zoom", type=int, default=DEFAULT_N_ZOOM)
    parser.add_argument("--zoom-size-km", type=float, default=DEFAULT_ZOOM_SIZE_KM)
    parser.add_argument("--min-center-spacing-km", type=float, default=DEFAULT_MIN_CENTER_SPACING_KM)
    parser.add_argument("--min-window-points", type=int, default=DEFAULT_MIN_WINDOW_POINTS)
    parser.add_argument("--min-window-hotspots", type=int, default=DEFAULT_MIN_WINDOW_HOTSPOTS)
    args = parser.parse_args()
    if args.n_zoom < 1 or args.n_zoom > 4:
        raise ValueError("--n-zoom must be between 1 and 4 for the current layout.")

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    block_size_m = float(args.block_size_km) * 1000.0
    model_n_jobs = max(1, int(args.model_n_jobs))
    transition_base_vars = TRANSITION_BASE_VARS.copy()
    if args.exclude_magt:
        transition_base_vars = [var for var in transition_base_vars if var != "magt"]

    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    table_dir = out_dir / "tables"
    for p in [fig_dir, cache_dir, table_dir]:
        p.mkdir(parents=True, exist_ok=True)
    model_cache_path = (
        args.model_cache.resolve()
        if args.model_cache is not None
        else (cache_dir / "models" / "figure6_gradient_mag_model_artifacts.joblib.gz")
    )

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        csv_path = base_dir / "df_all_data_with_wright_du.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Could not find df_all_data_with_wright_du_plus_grad.csv or df_all_data_with_wright_du.csv")

    threshold_cache_path = cache_dir / "figure6_region_thresholds.json"
    thresholds_by_q = resolve_region_thresholds(
        csv_path=csv_path,
        cache_path=threshold_cache_path,
        quantiles=THRESHOLD_SWEEP_QS,
        chunksize=args.chunksize,
    )
    thr_grad_095 = thresholds_by_q[MAIN_THRESHOLD_Q]

    sample_cache = cache_dir / f"figure6_sample_for_susceptibility_pf{args.sample_pf}_npf{args.sample_npf}.csv.gz"
    log_step("Resolving stratified PF/NPF sample")
    df = resolve_sample(
        csv_path=csv_path,
        cache_path=sample_cache,
        sample_pf=args.sample_pf,
        sample_npf=args.sample_npf,
        chunksize=args.chunksize,
    )

    transition_window_size = neighbors_to_window_size(args.neighbors_trans)
    transition_cache_dir = cache_dir / "figure6_transition_rasters"
    log_step("Ensuring transition rasters are available")
    grid, _, metric_paths = ensure_transition_rasters(
        base_dir=base_dir,
        csv_path=csv_path,
        cache_dir=transition_cache_dir,
        vars_for_transition=transition_base_vars,
        chunksize=args.chunksize,
        window_size=transition_window_size,
    )
    log_step("Attaching transition metrics to sampled points")
    df = attach_raster_transition_metrics(
        df=df,
        metric_paths=metric_paths,
        grid=grid,
        vars_for_transition=transition_base_vars,
    )

    contrast_features = []
    for v in transition_base_vars:
        contrast_features.extend([f"{v}__lstd", f"{v}__gmag"])

    raw_features = RAW_FEATURES.copy()
    if args.exclude_magt:
        raw_features = [feature for feature in raw_features if feature != "magt"]
    combined_features = raw_features + contrast_features

    pf_df = df.loc[df["domain"] == "pf"].copy()
    model_cols = ["easting", "northing", "longitude", "latitude", "grad_mag"] + raw_features + contrast_features
    pf_model = pf_df[model_cols].replace([np.inf, -np.inf], np.nan).copy()
    pf_model["block_id"] = make_spatial_block_id(pf_model, block_size_m=block_size_m)
    pf_model = pf_model.dropna(subset=["grad_mag", "block_id"])
    if len(pf_model) < 1000:
        raise RuntimeError("Too few PF rows remain after transition-feature construction.")

    log_step("Selecting spatial block train/test holdout")
    train_idx, test_idx = choose_spatial_holdout_split(
        pf_model,
        test_size=args.test_size,
        block_size_m=block_size_m,
        seed=SEED,
        q_for_validity=MAIN_THRESHOLD_Q,
        threshold_grad=thr_grad_095,
    )

    train_df = pf_model.iloc[train_idx].reset_index(drop=True)
    test_df = pf_model.iloc[test_idx].reset_index(drop=True)

    train_df["is_hotspot_095"] = (train_df["grad_mag"] >= thr_grad_095).astype(int)
    test_df["is_hotspot_095"] = (test_df["grad_mag"] >= thr_grad_095).astype(int)
    pf_model["is_hotspot_095"] = (pf_model["grad_mag"] >= thr_grad_095).astype(int)

    y_train = train_df["is_hotspot_095"].to_numpy(dtype=int)
    y_test = test_df["is_hotspot_095"].to_numpy(dtype=int)
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise RuntimeError("Spatial holdout produced a single-class split under the full-region threshold.")

    feature_sets_unfiltered = {
        "raw": raw_features,
        "contrast": contrast_features,
        "combined": combined_features,
    }
    missing_df = audit_feature_missingness(
        train_df,
        {
            "raw": raw_features,
            "contrast": contrast_features,
        },
    )
    missing_df.to_csv(table_dir / "figure6_feature_missingness.csv", index=False)
    feature_sets, dropped_features = drop_high_missing_features(
        feature_sets_unfiltered,
        missing_df,
        max_missing_frac=args.max_missing_frac,
    )
    raw_features = feature_sets["raw"]
    contrast_features = feature_sets["contrast"]
    combined_features = feature_sets["combined"]
    for set_name, feats in feature_sets.items():
        if not feats:
            raise RuntimeError(f"No predictors remain for feature set '{set_name}' after missingness filtering.")
    if not combined_features:
        raise RuntimeError("All combined predictors were dropped by the missingness filter.")
    if dropped_features:
        log_step(
            "Dropped high-missing features (> "
            f"{args.max_missing_frac:.0%} missing): {', '.join(dropped_features)}"
        )
    else:
        log_step("No features exceeded the missingness threshold")

    cache_signature = build_model_cache_signature(
        csv_path=csv_path,
        sample_cache=sample_cache,
        raw_features=raw_features,
        contrast_features=contrast_features,
        combined_features=combined_features,
        threshold_grad_095=thr_grad_095,
        transition_window_size=transition_window_size,
        transition_base_vars=transition_base_vars,
        neighbors_trans=args.neighbors_trans,
        block_size_m=block_size_m,
        max_missing_frac=args.max_missing_frac,
        model_n_jobs=model_n_jobs,
        test_size=args.test_size,
    )

    cached_artifacts = None if args.force_retrain else load_cached_training_artifacts(model_cache_path, cache_signature)
    loaded_cached_models = cached_artifacts is not None

    y_full = pf_model["is_hotspot_095"].to_numpy(dtype=int)
    groups_train = train_df["block_id"].astype(str).to_numpy()
    groups_full = pf_model["block_id"].astype(str).to_numpy()

    if loaded_cached_models:
        suite_fitted = cached_artifacts["suite_fitted"]
        suite_metrics = cached_artifacts["suite_metrics"]
        suite_probs = cached_artifacts["suite_probs"]
        full_stack = cached_artifacts["full_stack"]
        sweep_df = cached_artifacts["sweep_df"].copy()
        magt_ablation_df = cached_artifacts["magt_ablation_df"].copy()
        log_step(f"Loaded model artifacts: {model_cache_path}")
    else:
        suite_fitted = {}
        suite_metrics = {}
        suite_probs = {}

        for set_name, feats in feature_sets.items():
            log_step(f"Fitting model suite for feature set: {set_name}")
            X_train = make_model_frame(train_df, feats)
            X_test = make_model_frame(test_df, feats)
            fitted, metrics, probs = fit_and_evaluate_suite(
                X_train,
                y_train,
                X_test,
                y_test,
                groups_train=groups_train,
                n_jobs=model_n_jobs,
            )
            if set_name == "combined":
                suite_fitted[set_name] = {"Stack": fitted["Stack"]}
            suite_metrics[set_name] = metrics
            suite_probs[set_name] = probs
            del X_train, X_test, fitted
            gc.collect()

        log_step("Refitting final combined stacked model on all PF sample rows")
        X_full_combined = make_model_frame(pf_model, combined_features)
        pos_rate = max(float(np.mean(y_full)), 1e-6)
        pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
        full_stack = build_models(
            pos_weight,
            stack_cv=FixedGroupKFold(groups_full, n_splits=5),
            n_jobs=model_n_jobs,
        )["Stack"]
        full_stack.fit(X_full_combined, y_full)
        del X_full_combined
        gc.collect()

        log_step("Running full-region threshold robustness sweep")
        sweep_rows = []
        X_train_combined = make_model_frame(train_df, combined_features)
        X_test_combined = make_model_frame(test_df, combined_features)
        for q in THRESHOLD_SWEEP_QS:
            thr_grad = thresholds_by_q[q]
            y_train_q = (train_df["grad_mag"].to_numpy(dtype=float) >= thr_grad).astype(int)
            y_test_q = (test_df["grad_mag"].to_numpy(dtype=float) >= thr_grad).astype(int)

            pos_rate_q = max(float(np.mean(y_train_q)), 1e-6)
            pos_weight_q = max((1.0 - pos_rate_q) / pos_rate_q, 1.0)
            stack_q = build_models(
                pos_weight_q,
                stack_cv=FixedGroupKFold(groups_train, n_splits=5),
                n_jobs=model_n_jobs,
            )["Stack"]
            stack_q.fit(X_train_combined, y_train_q)
            p_q = stack_q.predict_proba(X_test_combined)[:, 1]
            metrics_q = classification_metrics(y_test_q, p_q)

            sweep_rows.append({
                "q": q,
                "threshold_grad_mag": thr_grad,
                "threshold_grad_mag_km": float(thr_grad) * GRAD_MAG_DISPLAY_SCALE,
                "prevalence": float(np.mean(y_test_q)),
                "roc_auc": metrics_q["roc_auc"],
                "ap": metrics_q["ap"],
                "brier": metrics_q["brier"],
            })
            del stack_q
            gc.collect()
        sweep_df = pd.DataFrame(sweep_rows)
        del X_train_combined, X_test_combined
        gc.collect()

        magt_ablation_rows = []
        if not args.exclude_magt and any(
            feature in combined_features for feature in ["magt", "magt__lstd", "magt__gmag"]
        ):
            no_magt_features = [
                feature for feature in combined_features
                if feature not in {"magt", "magt__lstd", "magt__gmag"}
            ]
            if no_magt_features:
                log_step("Running MAGT ablation for the combined stacked model")
                pos_rate = max(float(np.mean(y_train)), 1e-6)
                pos_weight = max((1.0 - pos_rate) / pos_rate, 1.0)
                stack_no_magt = build_models(
                    pos_weight,
                    stack_cv=FixedGroupKFold(groups_train, n_splits=5),
                    n_jobs=model_n_jobs,
                )["Stack"]
                X_train_no_magt = make_model_frame(train_df, no_magt_features)
                X_test_no_magt = make_model_frame(test_df, no_magt_features)
                _, metrics_no_magt, _ = fit_and_evaluate_model(
                    stack_no_magt,
                    X_train_no_magt,
                    y_train,
                    X_test_no_magt,
                    y_test,
                )
                del X_train_no_magt, X_test_no_magt, stack_no_magt
                gc.collect()
                magt_ablation_rows.append({
                    "setting": "combined_without_magt",
                    "n_features": len(no_magt_features),
                    **metrics_no_magt,
                })
        magt_ablation_rows.insert(0, {
            "setting": "combined_with_magt" if not args.exclude_magt else "combined_excluding_magt",
            "n_features": len(combined_features),
            **suite_metrics["combined"]["Stack"],
        })
        magt_ablation_df = pd.DataFrame(magt_ablation_rows)

        save_training_artifacts(
            model_cache_path,
            {
                "cache_signature": cache_signature,
                "suite_fitted": suite_fitted,
                "suite_metrics": suite_metrics,
                "suite_probs": suite_probs,
                "full_stack": full_stack,
                "sweep_df": sweep_df,
                "magt_ablation_df": magt_ablation_df,
            },
        )
        log_step(f"Saved model artifacts: {model_cache_path}")

    if "threshold_grad_mag" in sweep_df.columns and "threshold_grad_mag_km" not in sweep_df.columns:
        sweep_df["threshold_grad_mag_km"] = (
            pd.to_numeric(sweep_df["threshold_grad_mag"], errors="coerce") * GRAD_MAG_DISPLAY_SCALE
        )

    prob_raster_path = cache_dir / "figure6_pf_susceptibility_prob_f32.memmap"
    if prob_raster_path.exists() and loaded_cached_models and not args.force_retrain:
        log_step(f"Using cached PF susceptibility raster: {prob_raster_path}")
    else:
        build_pf_prediction_raster(
            csv_path=csv_path,
            prob_path=prob_raster_path,
            model=full_stack,
            grid=grid,
            metric_paths=metric_paths,
            vars_for_transition=transition_base_vars,
            feature_names=combined_features,
            chunksize=args.chunksize,
        )

    hotspot_density_path = cache_dir / "figure6_pf_hotspot_density_f32.memmap"
    if hotspot_density_path.exists() and loaded_cached_models and not args.force_retrain:
        log_step(f"Using cached PF hotspot-density raster: {hotspot_density_path}")
    else:
        build_pf_hotspot_density_raster(
            csv_path=csv_path,
            density_path=hotspot_density_path,
            grid=grid,
            hotspot_threshold=thr_grad_095,
            chunksize=args.chunksize,
        )

    pf_mask_path = cache_dir / "figure6_pf_mask_u8.memmap"
    build_pf_mask_raster(
        csv_path=csv_path,
        mask_path=pf_mask_path,
        grid=grid,
        chunksize=args.chunksize,
    )

    log_step("Computing permutation importance on the held-out spatial test blocks")
    X_test_combined = make_model_frame(test_df, combined_features)
    max_perm_rows = min(PERMUTATION_MAX_ROWS, len(X_test_combined))
    perm_idx = np.arange(len(X_test_combined))
    if len(perm_idx) > max_perm_rows:
        rng = np.random.default_rng(SEED)
        perm_idx = np.sort(rng.choice(perm_idx, size=max_perm_rows, replace=False))

    perm = permutation_importance(
        suite_fitted["combined"]["Stack"],
        X_test_combined.iloc[perm_idx],
        y_test[perm_idx],
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
        scoring="average_precision",
        n_jobs=model_n_jobs,
    )
    imp_df = (
        pd.DataFrame({
            "feature": combined_features,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # save performance tables
    perf_rows = []
    for featset, model_dict in suite_metrics.items():
        for model_name, m in model_dict.items():
            perf_rows.append({
                "feature_set": featset,
                "model": model_name,
                "roc_auc": m["roc_auc"],
                "ap": m["ap"],
                "brier": m["brier"],
            })
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(table_dir / "figure6_model_performance.csv", index=False)
    imp_df.to_csv(table_dir / "figure6_top8_permutation_importance.csv", index=False)
    sweep_df.to_csv(table_dir / "figure6_threshold_sweep.csv", index=False)
    magt_ablation_df.to_csv(table_dir / "figure6_magt_ablation.csv", index=False)

    log_step("Scoring PF sample for zoom-window selection")
    pf_zoom_df = pf_model.copy()
    X_pf_zoom = make_model_frame(pf_zoom_df, combined_features)
    pf_zoom_df["susceptibility"] = predict_proba_batched(full_stack, X_pf_zoom)
    del X_pf_zoom
    gc.collect()

    zoom_half_size_m = float(args.zoom_size_km) * 500.0 * ZOOM_WINDOW_LINEAR_SCALE
    zoom_min_spacing_m = float(args.min_center_spacing_km) * 1000.0
    try:
        zoom_windows = select_zoom_windows(
            pf_zoom_df,
            n_windows=int(args.n_zoom),
            half_size_m=zoom_half_size_m,
            min_spacing_m=zoom_min_spacing_m,
            min_points=int(args.min_window_points),
            min_hotspots=int(args.min_window_hotspots),
            forbidden_boxes=ZOOM_FORBIDDEN_BOXES,
        )
    except RuntimeError:
        relaxed_points = max(80, int(args.min_window_points) // 2)
        relaxed_hotspots = max(5, int(args.min_window_hotspots) // 2)
        relaxed_spacing = max(120_000.0, 0.8 * zoom_min_spacing_m)
        log_step(
            "Falling back to relaxed zoom-window filters "
            f"(points>={relaxed_points}, hotspots>={relaxed_hotspots}, spacing>={relaxed_spacing / 1000.0:.0f} km)"
        )
        zoom_windows = select_zoom_windows(
            pf_zoom_df,
            n_windows=int(args.n_zoom),
            half_size_m=zoom_half_size_m,
            min_spacing_m=relaxed_spacing,
            min_points=relaxed_points,
            min_hotspots=relaxed_hotspots,
            forbidden_boxes=ZOOM_FORBIDDEN_BOXES,
        )
    zoom_windows.to_csv(table_dir / "figure6_zoom_windows.csv", index=False)

    meta = {
        "pf_threshold_grad_q95": thr_grad_095,
        "pf_threshold_grad_q95_km": float(thr_grad_095) * GRAD_MAG_DISPLAY_SCALE,
        "pf_threshold_log_q95": float(np.log1p(thr_grad_095)),
        "pf_positive_fraction_q95": float(np.mean(y_full)),
        "region_threshold_grad_q95": float(thr_grad_095),
        "region_threshold_grad_q95_km": float(thr_grad_095) * GRAD_MAG_DISPLAY_SCALE,
        "threshold_source": "full_region_all_domains",
        "thresholds_by_q": {str(q): float(v) for q, v in thresholds_by_q.items()},
        "thresholds_by_q_km": {str(q): float(v) * GRAD_MAG_DISPLAY_SCALE for q, v in thresholds_by_q.items()},
        "raw_features": raw_features,
        "contrast_features": contrast_features,
        "transition_base_vars": transition_base_vars,
        "combined_features_n": len(combined_features),
        "dropped_features_high_missing": dropped_features,
        "max_missing_frac": float(args.max_missing_frac),
        "exclude_magt": bool(args.exclude_magt),
        "exclude_vwc": True,
        "block_size_km": float(args.block_size_km),
        "model_n_jobs": int(model_n_jobs),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_blocks": int(train_df["block_id"].nunique()),
        "test_blocks": int(test_df["block_id"].nunique()),
        "transition_window_size": transition_window_size,
        "xgboost_available": HAVE_XGB,
        "model_cache_path": str(model_cache_path),
        "loaded_cached_models": loaded_cached_models,
        "n_zoom": int(len(zoom_windows)),
        "zoom_size_km": float(args.zoom_size_km),
        "selected_zoom_windows": zoom_windows["window_id"].tolist(),
    }
    (cache_dir / "figure6_meta.json").write_text(json.dumps(meta, indent=2))

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), constrained_layout=False)
    gs = GridSpec(
        2, 2,
        figure=fig,
        left=0.05,
        right=0.97,
        top=0.93,
        bottom=0.08,
        wspace=0.20,
        hspace=0.24,
        height_ratios=[0.92, 1.08],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    gsC = gs[1, 0].subgridspec(1, 2, wspace=0.10)
    axC_hot = fig.add_subplot(gsC[0, 0])
    axC_pred = fig.add_subplot(gsC[0, 1])
    axE = fig.add_subplot(gs[1, 1])

    plot_panel_A(axA, df, thr_grad_095)
    plot_panel_B(axB, perf_df)
    plot_panel_C(axC_hot, axC_pred, pf_mask_path, hotspot_density_path, prob_raster_path, grid)
    plot_panel_E(axE, imp_df, top_n=10)

    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    prob_mm = open_memmap(
        prob_raster_path,
        dtype="float32",
        mode="r",
        shape=(int(grid["nrows"]), int(grid["ncols"])),
    )
    hotspot_mm = open_memmap(
        hotspot_density_path,
        dtype="float32",
        mode="r",
        shape=(int(grid["nrows"]), int(grid["ncols"])),
    )
    pf_mask_mm = open_memmap(
        pf_mask_path,
        dtype="uint8",
        mode="r",
        shape=(int(grid["nrows"]), int(grid["ncols"])),
    )

    grad_raster_dir = base_dir / "outputs" / "grad_rasters"
    grad_meta_path = grad_raster_dir / "grid_meta.npz"
    du_path = grad_raster_dir / "du_f32.memmap"
    if not grad_meta_path.exists() or not du_path.exists():
        raise FileNotFoundError(
            f"Missing zoom-context raster inputs: {grad_meta_path} and/or {du_path}"
        )
    grad_meta = np.load(grad_meta_path)
    grad_grid = {
        "res": float(grad_meta["res"]),
        "nrows": int(grad_meta["nrows"]),
        "ncols": int(grad_meta["ncols"]),
        "gx0": int(grad_meta["gx0"]),
        "gy1": int(grad_meta["gy1"]),
        "min_e": float(grad_meta["min_e"]),
        "max_n": float(grad_meta["max_n"]),
    }
    if (
        grad_grid["nrows"] != int(grid["nrows"])
        or grad_grid["ncols"] != int(grid["ncols"])
        or abs(grad_grid["res"] - float(grid["res"])) > 1e-6
    ):
        raise RuntimeError("Zoom-context d_u raster grid does not match the Figure 6 grid.")

    du_mm = open_memmap(
        du_path,
        dtype="float32",
        mode="r",
        shape=(int(grad_grid["nrows"]), int(grad_grid["ncols"])),
    )
    preview_stride = choose_stride(int(grad_grid["nrows"]), int(grad_grid["ncols"]), target_max=1200)
    du_preview = np.array(du_mm[::preview_stride, ::preview_stride], copy=False).astype(np.float32)
    pf_preview = np.array(pf_mask_mm[::preview_stride, ::preview_stride], copy=False) == np.uint8(1)
    du_preview_pf = du_preview[pf_preview & np.isfinite(du_preview)]
    if du_preview_pf.size:
        du_lo, du_hi = robust_clip(du_preview_pf, 2.0, 98.0)
        du_abs = max(abs(du_lo), abs(du_hi), 1e-6)
        abs_du_hi = robust_clip(np.abs(du_preview_pf), 2.0, 98.0)[1]
    else:
        du_abs = 1.0
        abs_du_hi = 1.0
    du_norm = TwoSlopeNorm(vmin=-du_abs, vcenter=0.0, vmax=du_abs)
    abs_du_norm = Normalize(vmin=0.0, vmax=max(abs_du_hi, 1e-6))

    fig_win = plt.figure(figsize=(18, 14), constrained_layout=False)
    gs_win = GridSpec(
        4, 4,
        figure=fig_win,
        left=0.05,
        right=0.93,
        top=0.95,
        bottom=0.07,
        wspace=0.10,
        hspace=0.14,
    )
    row_axes = [[fig_win.add_subplot(gs_win[r, c]) for c in range(4)] for r in range(4)]
    pred_ims = []
    obs_ims = []
    du_ims = []
    abs_du_ims = []

    for idx, ((_, window), color) in enumerate(zip(zoom_windows.iterrows(), WINDOW_COLORS)):
        stats_text = (
            f"n={int(window['n_points'])}\n"
            f"hot={int(window['n_hotspots'])} ({float(window['hotspot_fraction']):.1%})\n"
            f"mean p={float(window['mean_susceptibility']):.2f}"
        )
        pred_ims.append(
            plot_window_raster(
                row_axes[0][idx],
                prob_mm,
                pf_mask_mm,
                grid,
                window,
                show_ylabel=(idx == 0),
                show_xlabel=False,
                title_text=str(window["window_id"]),
                ylabel_text="Predicted\nsusceptibility",
                cmap="plasma",
                norm=Normalize(vmin=0.0, vmax=1.0),
                border_color=color,
                stats_text=stats_text,
            )
        )
        obs_ims.append(
            plot_window_raster(
                row_axes[1][idx],
                hotspot_mm,
                pf_mask_mm,
                grid,
                window,
                show_ylabel=(idx == 0),
                show_xlabel=False,
                ylabel_text="Observed\nhotspot density",
                cmap="plasma",
                norm=Normalize(vmin=0.0, vmax=1.0),
                border_color=color,
            )
        )
        du_ims.append(
            plot_window_raster(
                row_axes[2][idx],
                du_mm,
                pf_mask_mm,
                grad_grid,
                window,
                show_ylabel=(idx == 0),
                show_xlabel=False,
                ylabel_text=r"$d_u$" + "\n(mm/yr)",
                cmap="coolwarm",
                norm=du_norm,
                border_color=color,
            )
        )
        abs_du_ims.append(
            plot_window_raster(
                row_axes[3][idx],
                du_mm,
                pf_mask_mm,
                grad_grid,
                window,
                show_ylabel=(idx == 0),
                show_xlabel=True,
                ylabel_text=r"$|d_u|$" + "\n(mm/yr)",
                cmap="viridis",
                norm=abs_du_norm,
                border_color=color,
                transform=np.abs,
            )
        )

    row_colorbars = [
        (pred_ims, "Predicted susceptibility"),
        (obs_ims, "Observed hotspot density"),
        (du_ims, r"$d_u$ (mm/yr)"),
        (abs_du_ims, r"$|d_u|$ (mm/yr)"),
    ]
    for ims, label in row_colorbars:
        ims_valid = [im for im in ims if im is not None]
        if ims_valid:
            cb = fig_win.colorbar(
                ims_valid[0],
                ax=[im.axes for im in ims_valid],
                location="right",
                fraction=0.02,
                pad=0.01,
            )
            cb.set_label(label, fontweight="bold", fontsize=8)
            cb.ax.tick_params(labelsize=7, length=2)

    out_win_png = fig_dir / f"{WINDOWS_FIG_BASENAME}.png"
    out_win_pdf = fig_dir / f"{WINDOWS_FIG_BASENAME}.pdf"
    fig_win.savefig(out_win_png, bbox_inches="tight", dpi=300)
    fig_win.savefig(out_win_pdf, bbox_inches="tight")
    plt.close(fig_win)

    log_step(f"Saved PNG: {out_png}")
    log_step(f"Saved PDF: {out_pdf}")
    log_step(f"Saved window PNG: {out_win_png}")
    log_step(f"Saved window PDF: {out_win_pdf}")
    log_step(f"Saved performance table: {table_dir / 'figure6_model_performance.csv'}")
    log_step(f"Saved threshold sweep: {table_dir / 'figure6_threshold_sweep.csv'}")
    log_step(f"Saved zoom window table: {table_dir / 'figure6_zoom_windows.csv'}")


if __name__ == "__main__":
    main()
