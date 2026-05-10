#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/_revised_check_npf_extreme_thermal_dependence.py
# Renamed package path: code/analysis_support/npf_transition_thermal_diagnostics.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

import Figure_reorganize_extreme_deformation_susceptibity as figsus
import Figure_reorganize_railway_buffer_analysis as railbuf
import _revised_zou_boundary_utils as zou
import figure6_susceptibility_stacked as fig6
from submission_figure_style import add_panel_label as add_submission_panel_label
from submission_figure_style import FONT


SEED = figsus.SEED
CHUNKSIZE = figsus.CHUNKSIZE
FIG_BASENAME = "_revised_check_npf_extreme_thermal_dependence"
MODEL_ARTIFACT_VERSION = 1
DEFAULT_DPI = 600
ZOU_DISTANCE_MODE_DEFAULT = "lonlat"
BOUNDARY_DISTANCE_COL = "zou_boundary_distance_km"
A4_LANDSCAPE = (11.69, 8.27)
WIDE_SHORT = (11.69, 4.35)
COMPOSITE_FIGSIZE = (11.69, 7.0)
COMPOSITE_WIDTH_RATIOS = [0.45, 0.55]
COMPOSITE_SUBPLOTS_ADJUST = dict(
    left=0.07,
    right=0.97,
    top=0.92,
    bottom=0.10,
    wspace=0.22,
    hspace=0.28,
)
MAGT_ZERO_BAND_LO = -0.3
MAGT_ZERO_BAND_HI = 0.3
MAGT_ZERO_BAND_COLOR = "0.88"
MAGT_ZERO_BAND_ALPHA = 0.5
FINAL_FIGSIZE = (7.5, 10.5)
FINAL_HEIGHT_RATIOS = [0.28, 0.40, 0.32]
FINAL_SUBPLOTS_ADJUST = dict(
    left=0.10,
    right=0.96,
    top=0.93,
    bottom=0.05,
    wspace=0.26,
    hspace=0.30,
)
RIDGE_SPACING = 1.0
RIDGE_OVERLAP = 0.75
RIDGE_MAX_HEIGHT = 0.85
RIDGE_KDE_BW = 0.3
RIDGE_ALL_ALPHA = 0.18
RIDGE_ALL_LINE_ALPHA = 0.40
RIDGE_EXT_ALPHA = 0.50
RIDGE_EXT_LINE_WIDTH = 1.8
MAGT_XLIM = (-3.0, 8.0)
MAGT_XGRID_N = 300
DIST_BIN_PALETTE = [
    "#1b3a4b",
    "#3d6b7e",
    "#7ba3b0",
    "#afc9d2",
    "#d5e2e7",
]
ZERO_C_LINE = dict(color="0.50", linewidth=1.0, linestyle="--", zorder=0)
ZERO_C_BAND_LO = -0.3
ZERO_C_BAND_HI = 0.3
ZERO_C_BAND_COLOR = "#3d6b7e"
ZERO_C_BAND_ALPHA = 0.48
THERMAL_FEATURES = ("magt", "temperature_mean")
THERMAL_LABELS = {
    "magt": "MAGT",
    "temperature_mean": "MAAT",
}
THERMAL_COLORS = {
    "magt": "#2A7F62",
    "temperature_mean": "#A66A3F",
}
FEATURE_LINEAR_TRANSFORMS = {
    "temperature_mean": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
}
FEATURE_UNITS = {
    "magt": "°C",
    "temperature_mean": "°C",
}
DISTANCE_BINS_KM_DEFAULT = (0.0, 1.0, 2.0, 5.0, 10.0, math.inf)
BUFFER_SWEEP_KM_DEFAULT = (0.0, 1.0, 2.0, 5.0, 10.0)
ALE_BINS = 20
ALE_MAX_ROWS = 30_000
ALE_PRED_BATCH_SIZE = 10_000
PANEL_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TARGET_TITLES = {
    "d_u": r"Extreme $d_u$ in non-permafrost",
    "grad_mag_km": r"Extreme $|\nabla d_u|$ in non-permafrost",
}
TARGET_SHORT_LABELS = {
    "d_u": r"Extreme $d_u$",
    "grad_mag_km": r"Extreme $|\nabla d_u|$",
}
OVERLAY_YLABELS = {
    "d_u": "ALE of extreme $d_u$\nsusceptibility",
    "grad_mag_km": "ALE of extreme $|\\nabla d_u|$\nsusceptibility",
}
TARGET_COLORS = {
    "d_u": railbuf.DU_BASE_COLOR,
    "grad_mag_km": railbuf.GRAD_BASE_COLOR,
}
EXCLUSION_CURVE_COLORS = (
    "#234B62",
    "#4F748A",
    "#88A6B5",
    "#B7CBD3",
    "#D8E4E9",
)


def log_step(message: str) -> None:
    print(f"[{FIG_BASENAME}] {message}")


def resolve_csv_path(base_dir: Path) -> Path:
    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        csv_path = base_dir / "df_all_data_with_wright_du.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Could not find df_all_data_with_wright_du_plus_grad.csv or df_all_data_with_wright_du.csv")
    return csv_path


def resolve_project_crs_wkt(
    base_dir: Path,
    explicit_prj: Path | None = None,
) -> str:
    candidates: list[Path] = []
    if explicit_prj is not None:
        candidates.append(explicit_prj.resolve())
    candidates.extend(
        [
            base_dir / "human_features" / "qtec_railway_clip.prj",
            base_dir / "qtec_thaw_lakes" / "TLS_des_clipped.prj",
        ]
    )
    for prj_path in candidates:
        if prj_path.exists():
            return prj_path.read_text(encoding="utf-8", errors="ignore")

    crs_info_path = base_dir / "crs_info.txt"
    if crs_info_path.exists():
        payload = crs_info_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"CRS:\s*(.+?)<br/>\[INFO\] Transform:",
            payload,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
    raise FileNotFoundError(
        "Could not resolve the project CRS from a .prj file or crs_info.txt."
    )


def parse_float_sequence(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in str(raw).split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(math.inf)
        else:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return tuple(values)


def sample_frame(df: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True).copy()


def scale_feature_values_for_plot(feature: str, values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    scale, offset = FEATURE_LINEAR_TRANSFORMS.get(feature, (1.0, 0.0))
    return arr * scale + offset


def feature_axis_label(feature: str) -> str:
    unit = FEATURE_UNITS.get(feature)
    label = THERMAL_LABELS.get(feature, feature)
    if unit:
        return f"{label} ({unit})"
    return label


def add_panel_label(ax, label: str) -> None:
    add_submission_panel_label(ax, label, x=-0.10, y=1.04)


def style_axis(ax) -> None:
    railbuf.style_open_axes(ax)
    railbuf.apply_bold_ticklabels(ax)
    ax.grid(axis="y", color="0.92", linewidth=0.8)


def buffer_tag(buffer_km: float) -> str:
    raw = f"{float(buffer_km):.1f}".rstrip("0").rstrip(".")
    return raw.replace("-", "m").replace(".", "p")


def buffer_label(buffer_km: float) -> str:
    if np.isclose(float(buffer_km), 0.0):
        return "All NPF"
    return f"≥ {float(buffer_km):g} km"


def buffer_label_with_retained(
    buffer_km: float,
    model_df: pd.DataFrame,
    target: str,
) -> str:
    """
    Produce legend labels like 'All NPF (100%)' or '≥ 5 km (55%)'
    so retained-fraction info is embedded without a twin axis.
    """
    row = model_df.loc[
        model_df["target"].eq(target)
        & np.isclose(
            model_df["buffer_km"].to_numpy(dtype=float),
            float(buffer_km),
        )
    ]
    base = buffer_label(buffer_km)
    if row.empty:
        return base
    frac = row.iloc[0].get("retained_frac", np.nan)
    if not np.isfinite(frac):
        return base
    return f"{base} ({frac * 100.0:.0f}%)"


def distance_bin_label(lo: float, hi: float) -> str:
    if np.isinf(hi):
        return f">{float(lo):g}"
    return f"{float(lo):g}–{float(hi):g}"


def rc_to_en(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    res: float,
    gx0: int,
    gy1: int,
) -> tuple[np.ndarray, np.ndarray]:
    cols_i64 = np.asarray(cols, dtype=np.int64)
    rows_i64 = np.asarray(rows, dtype=np.int64)
    easting = (gx0 + cols_i64).astype(np.float64) * float(res)
    northing = (gy1 - rows_i64).astype(np.float64) * float(res)
    return easting, northing


def resolve_zou_boundary_reference(
    *,
    base_dir: Path,
    cache_path: Path,
    mode: str,
    grid: dict[str, object] | None = None,
    target_crs_wkt: str | None = None,
) -> tuple[Path, dict[str, object]]:
    zou_tif = zou.resolve_zou_tif(base_dir)
    boundary_ref = zou.resolve_zou_boundary_reference(
        zou_tif,
        cache_path=cache_path,
        mode=mode,
        grid=grid,
        target_crs_wkt=target_crs_wkt,
    )
    return zou_tif, boundary_ref


def sample_cache_path(cache_dir: Path, *, target: str, sample_pf: int, sample_npf: int) -> Path:
    sample_cache_tag = "negdu" if figsus.NEGATIVE_DU_ONLY else "all_du"
    return cache_dir / (
        f"{figsus.FIG_BASENAME}_{figsus.TARGET_META[target]['table_stub']}_{sample_cache_tag}_sample_pf{sample_pf}_npf{sample_npf}.csv.gz"
    )


def enriched_sample_cache_signature(
    *,
    target: str,
    csv_path: Path,
    base_sample_cache: Path,
    zou_tif: Path,
    boundary_cache_path: Path,
    zou_distance_mode: str,
    transition_window_size: int,
    transition_base_vars: list[str],
    neighbors_trans: int,
) -> dict[str, object]:
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "target": target,
        "csv": fig6.file_signature(csv_path),
        "base_sample_cache": fig6.file_signature(base_sample_cache),
        "zou_tif": zou.path_signature(zou_tif),
        "boundary_cache": zou.path_signature(boundary_cache_path),
        "zou_distance_mode": str(zou_distance_mode),
        "transition_window_size": int(transition_window_size),
        "transition_base_vars": list(transition_base_vars),
        "neighbors_trans": int(neighbors_trans),
    }


def resolve_enriched_target_sample(
    *,
    target: str,
    csv_path: Path,
    cache_dir: Path,
    grid: dict[str, object],
    metric_paths: dict[str, dict[str, Path]],
    transition_base_vars: list[str],
    transition_window_size: int,
    neighbors_trans: int,
    chunksize: int,
    sample_pf: int,
    sample_npf: int,
    zou_tif: Path,
    boundary_ref: dict[str, object],
    boundary_cache_path: Path,
    zou_distance_mode: str,
) -> tuple[pd.DataFrame, Path, Path]:
    base_sample_cache = sample_cache_path(
        cache_dir,
        target=target,
        sample_pf=sample_pf,
        sample_npf=sample_npf,
    )
    cache_path = cache_dir / f"{FIG_BASENAME}_{figsus.TARGET_META[target]['table_stub']}_enriched_sample.joblib.gz"
    signature = enriched_sample_cache_signature(
        target=target,
        csv_path=csv_path,
        base_sample_cache=base_sample_cache,
        zou_tif=zou_tif,
        boundary_cache_path=boundary_cache_path,
        zou_distance_mode=zou_distance_mode,
        transition_window_size=transition_window_size,
        transition_base_vars=transition_base_vars,
        neighbors_trans=neighbors_trans,
    )
    if cache_path.exists():
        try:
            payload = joblib.load(cache_path)
        except Exception as exc:
            log_step(f"Warning: failed to load enriched sample cache {cache_path}: {exc}")
        else:
            if (
                isinstance(payload, dict)
                and {"cache_signature", "df"}.issubset(payload)
                and payload["cache_signature"] == signature
            ):
                return payload["df"], base_sample_cache, cache_path
            log_step(f"Enriched sample cache mismatch at {cache_path}; rebuilding.")

    log_step(f"Resolving sampled PF/NPF rows for {target}")
    sample_df = figsus.resolve_target_sample(
        csv_path=csv_path,
        cache_path=base_sample_cache,
        sample_pf=sample_pf,
        sample_npf=sample_npf,
        target=target,
        chunksize=chunksize,
    )
    log_step(f"Attaching transition metrics for {target}")
    sample_df = fig6.attach_raster_transition_metrics(
        df=sample_df,
        metric_paths=metric_paths,
        grid=grid,
        vars_for_transition=transition_base_vars,
    )
    log_step(f"Sampling Zou domain and Zou-boundary distance for {target}")
    sample_df = zou.attach_zou_domain_and_distance(
        sample_df,
        zou_tif=zou_tif,
        boundary_ref=boundary_ref,
        overwrite_domain=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "cache_signature": signature,
            "df": sample_df,
        },
        cache_path,
        compress=("gzip", 3),
    )
    return sample_df, base_sample_cache, cache_path


def buffer_model_cache_signature(
    *,
    target: str,
    csv_path: Path,
    enriched_sample_cache: Path,
    buffer_km: float,
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
        "domain": "npf",
        "csv": fig6.file_signature(csv_path),
        "enriched_sample_cache": fig6.file_signature(enriched_sample_cache),
        "buffer_km": round(float(buffer_km), 12),
        "combined_features": list(combined_features),
        "transition_window_size": int(transition_window_size),
        "transition_base_vars": list(transition_base_vars),
        "neighbors_trans": int(neighbors_trans),
        "block_size_m": float(block_size_m),
        "max_missing_frac": float(max_missing_frac),
        "model_n_jobs": int(model_n_jobs),
        "test_size": float(test_size),
    }


def compact_domain_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "target": result["target"],
        "domain": result["domain"],
        "y_test": result["y_test"],
        "suite_metrics": result["suite_metrics"],
        "suite_probs": result["suite_probs"],
        "full_stack": result["full_stack"],
        "importance_df": result["importance_df"],
        "missing_df": result["missing_df"],
        "feature_names": result["feature_names"],
        "feature_groups": result["feature_groups"],
        "dropped_features": result["dropped_features"],
        "positive_fraction": result["positive_fraction"],
        "n_rows": result["n_rows"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
    }


def resolve_buffer_model(
    *,
    target: str,
    sample_df: pd.DataFrame,
    csv_path: Path,
    enriched_sample_cache: Path,
    model_cache_dir: Path,
    buffer_km: float,
    combined_features: list[str],
    transition_base_vars: list[str],
    transition_window_size: int,
    neighbors_trans: int,
    block_size_m: float,
    max_missing_frac: float,
    model_n_jobs: int,
    test_size: float,
    force_retrain: bool,
) -> tuple[dict[str, object] | None, pd.DataFrame]:
    subset = sample_df.loc[
        sample_df["domain"].eq("npf")
        & np.isfinite(pd.to_numeric(sample_df["zou_boundary_distance_km"], errors="coerce").to_numpy(dtype=float))
        & (pd.to_numeric(sample_df["zou_boundary_distance_km"], errors="coerce").to_numpy(dtype=float) >= float(buffer_km))
    ].copy().reset_index(drop=True)
    cache_path = model_cache_dir / (
        f"{FIG_BASENAME}_{figsus.TARGET_META[target]['table_stub']}_npf_ge{buffer_tag(buffer_km)}km_artifacts.joblib.gz"
    )
    signature = buffer_model_cache_signature(
        target=target,
        csv_path=csv_path,
        enriched_sample_cache=enriched_sample_cache,
        buffer_km=buffer_km,
        combined_features=combined_features,
        transition_base_vars=transition_base_vars,
        transition_window_size=transition_window_size,
        neighbors_trans=neighbors_trans,
        block_size_m=block_size_m,
        max_missing_frac=max_missing_frac,
        model_n_jobs=model_n_jobs,
        test_size=test_size,
    )
    cached = None if force_retrain else figsus.load_cached_target_artifacts(cache_path, signature)
    if cached is not None:
        out = dict(cached)
        out["model_cache_path"] = cache_path
        return out, subset

    if len(subset) < 1000:
        log_step(f"Skipping {target} at buffer {buffer_km:g} km: only {len(subset)} NPF rows remain.")
        return None, subset

    log_step(f"Fitting NPF buffered model for {target} at {buffer_km:g} km")
    try:
        fit_result = figsus.fit_domain_model(
            subset,
            target=target,
            domain="npf",
            combined_features=combined_features,
            block_size_m=block_size_m,
            test_size=test_size,
            max_missing_frac=max_missing_frac,
            model_n_jobs=model_n_jobs,
        )
    except Exception as exc:
        log_step(f"Skipping {target} at buffer {buffer_km:g} km after fit failure: {exc}")
        return None, subset
    compact = compact_domain_result(fit_result)
    figsus.save_target_artifacts(
        cache_path,
        {
            "cache_signature": signature,
            **compact,
        },
    )
    compact["model_cache_path"] = cache_path
    return compact, subset


def build_ale_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.array([], dtype=float)
    if np.isclose(np.nanmin(vals), np.nanmax(vals)):
        return np.array([], dtype=float)
    edges = np.quantile(vals, np.linspace(0.0, 1.0, int(n_bins) + 1))
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size >= 2:
        return edges
    return np.array([], dtype=float)


def predict_proba_batched(model, X: pd.DataFrame, batch_size: int = ALE_PRED_BATCH_SIZE) -> np.ndarray:
    out = np.full(len(X), np.nan, dtype=np.float32)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        out[start:stop] = model.predict_proba(X.iloc[start:stop])[:, 1].astype(np.float32)
    return out


def compute_ale_curve(
    *,
    X_ref: pd.DataFrame,
    feature: str,
    predict_fn,
    n_bins: int,
) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    if feature not in X_ref.columns or X_ref.empty:
        return pd.DataFrame(columns=["x", "x_plot", "y", "ylo", "yhi", "n"]), None

    vals_all = pd.to_numeric(X_ref[feature], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals_all)
    if not finite.any():
        return pd.DataFrame(columns=["x", "x_plot", "y", "ylo", "yhi", "n"]), None

    X = X_ref.loc[finite].copy().reset_index(drop=True)
    vals = vals_all[finite]
    edges = build_ale_edges(vals, n_bins=n_bins)
    if edges.size < 2:
        return pd.DataFrame(columns=["x", "x_plot", "y", "ylo", "yhi", "n"]), None

    bin_idx = np.searchsorted(edges, vals, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    lowers = edges[bin_idx]
    uppers = edges[bin_idx + 1]

    X_low = X.copy()
    X_high = X.copy()
    X_low[feature] = lowers
    X_high[feature] = uppers

    pred_low = predict_fn(X_low).astype(float)
    pred_high = predict_fn(X_high).astype(float)
    diffs = pred_high - pred_low

    n_bins_eff = len(edges) - 1
    diff_mean = np.full(n_bins_eff, np.nan, dtype=float)
    diff_std = np.full(n_bins_eff, np.nan, dtype=float)
    counts = np.zeros(n_bins_eff, dtype=int)
    for idx in range(n_bins_eff):
        mask = bin_idx == idx
        counts[idx] = int(mask.sum())
        if not mask.any():
            continue
        diff_mean[idx] = float(np.nanmean(diffs[mask]))
        diff_std[idx] = float(np.nanstd(diffs[mask], ddof=1)) if counts[idx] > 1 else 0.0

    valid = counts > 0
    if not valid.any():
        return pd.DataFrame(columns=["x", "x_plot", "y", "ylo", "yhi", "n"]), None

    mids = 0.5 * (edges[:-1] + edges[1:])
    se_bins = np.zeros(n_bins_eff, dtype=float)
    se_bins[valid] = diff_std[valid] / np.sqrt(counts[valid])

    cum_prev = np.concatenate(([0.0], np.cumsum(np.nan_to_num(diff_mean[:-1], nan=0.0))))
    ale = cum_prev + 0.5 * np.nan_to_num(diff_mean, nan=0.0)
    var_prev = np.concatenate(([0.0], np.cumsum(np.square(se_bins[:-1]))))
    ale_se = np.sqrt(var_prev + 0.25 * np.square(se_bins))

    center = float(np.average(ale[valid], weights=counts[valid]))
    ale -= center
    ylo = ale - 1.96 * ale_se
    yhi = ale + 1.96 * ale_se

    curve = pd.DataFrame(
        {
            "x": mids.astype(float),
            "x_plot": scale_feature_values_for_plot(feature, mids),
            "y": ale.astype(float),
            "ylo": ylo.astype(float),
            "yhi": yhi.astype(float),
            "n": counts.astype(int),
        }
    )
    qlo, qhi = np.quantile(vals, [0.02, 0.98])
    support = (
        float(scale_feature_values_for_plot(feature, np.asarray([qlo]))[0]),
        float(scale_feature_values_for_plot(feature, np.asarray([qhi]))[0]),
    )
    return curve, support


def wilson_interval(positives: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = float(positives) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n + (z**2) / (4.0 * n * n), 0.0))
    return max(0.0, center - half), min(1.0, center + half)


def build_distance_rate_table(
    df: pd.DataFrame,
    *,
    target: str,
    distance_bins_km: tuple[float, ...],
) -> pd.DataFrame:
    work = df.loc[
        df["domain"].eq("npf")
        & np.isfinite(pd.to_numeric(df["zou_boundary_distance_km"], errors="coerce").to_numpy(dtype=float))
    ].copy()
    work["label"] = figsus.build_domain_label(work, target=target, domain="npf")
    work = work.loc[work["label"].notna()].copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce").astype(int)
    edges = list(distance_bins_km)
    labels = [distance_bin_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    work["distance_bin"] = pd.cut(
        pd.to_numeric(work["zou_boundary_distance_km"], errors="coerce"),
        bins=edges,
        right=False,
        include_lowest=True,
        labels=labels,
    )
    rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels):
        sub = work.loc[work["distance_bin"].eq(label)]
        positives = int(sub["label"].sum()) if not sub.empty else 0
        n = int(len(sub))
        rate = (positives / n) if n else np.nan
        ci_low, ci_high = wilson_interval(positives, n)
        rows.append(
            {
                "target": target,
                "bin_index": idx,
                "bin_label": label,
                "distance_lo_km": float(edges[idx]),
                "distance_hi_km": float(edges[idx + 1]) if np.isfinite(edges[idx + 1]) else math.inf,
                "n": n,
                "positives": positives,
                "rate": rate,
                "rate_ci_low": ci_low,
                "rate_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def prepare_magt_by_distance(
    sample_df: pd.DataFrame,
    *,
    target: str,
    distance_bins_km: tuple[float, ...],
) -> pd.DataFrame:
    work = sample_df.loc[
        sample_df["domain"].eq("npf")
    ].copy()

    work["magt"] = pd.to_numeric(work["magt"], errors="coerce")
    work["zou_boundary_distance_km"] = pd.to_numeric(
        work["zou_boundary_distance_km"], errors="coerce"
    )
    work = work.loc[
        np.isfinite(work["magt"])
        & np.isfinite(work["zou_boundary_distance_km"])
    ].copy()

    work["label"] = figsus.build_domain_label(
        work, target=target, domain="npf"
    )
    work = work.loc[work["label"].notna()].copy()
    work["is_extreme"] = (
        pd.to_numeric(work["label"], errors="coerce")
        .astype(int)
        .astype(bool)
    )

    edges = list(distance_bins_km)
    bin_labels = [
        distance_bin_label(edges[i], edges[i + 1])
        for i in range(len(edges) - 1)
    ]
    work["distance_bin"] = pd.cut(
        work["zou_boundary_distance_km"],
        bins=edges,
        right=False,
        include_lowest=True,
        labels=bin_labels,
    )
    work = work.loc[work["distance_bin"].notna()].copy()

    return work[
        ["magt", "zou_boundary_distance_km", "distance_bin", "is_extreme"]
    ].reset_index(drop=True)


def build_buffer_palette(buffers_km: tuple[float, ...]) -> dict[float, tuple[float, float, float]]:
    base = "#2F5D73"
    blends = np.linspace(0.08, 0.72, len(buffers_km))
    return {
        float(buffer): railbuf.blend_with_white(base, float(blend))
        for buffer, blend in zip(buffers_km, blends, strict=False)
    }


def plot_distance_rate_figure(
    *,
    distance_df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=WIDE_SHORT, sharey=True)
    plt.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.20, wspace=0.12)
    ymax = pd.to_numeric(distance_df["rate_ci_high"], errors="coerce").to_numpy(dtype=float)
    ymax = ymax[np.isfinite(ymax)]
    ytop = max(0.12, float(np.nanmax(ymax)) * 1.25) if ymax.size else 0.12

    for idx, target in enumerate(("d_u", "grad_mag_km")):
        ax = axes[idx]
        sub = distance_df.loc[distance_df["target"].eq(target)].sort_values("bin_index")
        x = np.arange(len(sub), dtype=float)
        color = TARGET_COLORS[target]
        ax.bar(
            x,
            sub["rate"],
            color=railbuf.blend_with_white(color, 0.68),
            edgecolor=color,
            linewidth=1.0,
        )
        yerr = np.vstack([
            np.maximum(sub["rate"] - sub["rate_ci_low"], 0.0),
            np.maximum(sub["rate_ci_high"] - sub["rate"], 0.0),
        ])
        ax.errorbar(
            x,
            sub["rate"],
            yerr=yerr,
            fmt="o-",
            color=color,
            linewidth=1.8,
            markersize=4.0,
            capsize=3.0,
            zorder=3,
        )
        for xi, rate, n in zip(x, sub["rate"], sub["n"], strict=False):
            if np.isfinite(rate):
                ax.text(
                    xi,
                    rate + 0.015,
                    f"n={int(n):,}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="0.35",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin_label"])
        ax.set_title(TARGET_TITLES[target], fontweight="bold", fontsize=10.0)
        ax.set_xlabel("Distance to Zou PF boundary (km)", fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Extreme rate within NPF", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, ytop)
        add_panel_label(ax, PANEL_LABELS[idx])
        style_axis(ax)

    fig.suptitle("NPF extreme prevalence versus Zou-boundary distance", fontsize=11.5, fontweight="bold", y=0.94)
    fig.text(
        0.5,
        0.08,
        "Rates are computed within the sampled non-permafrost rows used by the susceptibility workflow. Error bars show Wilson 95% intervals.",
        ha="center",
        va="center",
        fontsize=8,
        color="0.30",
    )

    out_png = fig_dir / f"{FIG_BASENAME}_distance_rate_summary.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_distance_rate_summary.pdf"
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def compute_overlay_xlim(curve_df: pd.DataFrame, *, target: str, feature: str) -> tuple[float, float]:
    sub = curve_df.loc[curve_df["target"].eq(target) & curve_df["feature"].eq(feature)]
    vals = pd.to_numeric(sub["x_plot"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.10, 1.0)
        return lo - pad, hi + pad
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def compute_overlay_ylim(curve_df: pd.DataFrame, *, target: str) -> tuple[float, float]:
    sub = curve_df.loc[curve_df["target"].eq(target)]
    vals = []
    for col in ("y", "ylo", "yhi"):
        arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return (-0.02, 0.02)
    merged = np.concatenate(vals + [np.asarray([0.0], dtype=float)])
    lo = float(np.nanmin(merged))
    hi = float(np.nanmax(merged))
    if np.isclose(lo, hi):
        span = max(abs(lo), 0.01)
        return lo - 0.3 * span, hi + 0.3 * span
    pad = 0.12 * (hi - lo)
    return lo - pad, hi + pad


def plot_buffer_summary_figure(
    *,
    model_df: pd.DataFrame,
    thermal_df: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=A4_LANDSCAPE, sharex=True)
    plt.subplots_adjust(left=0.08, right=0.94, top=0.86, bottom=0.12, wspace=0.18, hspace=0.26)

    for row_idx, target in enumerate(("d_u", "grad_mag_km")):
        model_sub = model_df.loc[model_df["target"].eq(target)].sort_values("buffer_km")
        thermal_sub = thermal_df.loc[thermal_df["target"].eq(target)].sort_values("buffer_km")
        target_color = TARGET_COLORS[target]

        ax = axes[row_idx, 0]
        for feature in THERMAL_FEATURES:
            feat_sub = thermal_sub.loc[thermal_sub["feature"].eq(feature)]
            ax.plot(
                feat_sub["buffer_km"],
                feat_sub["importance"],
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                color=THERMAL_COLORS[feature],
                label=THERMAL_LABELS[feature],
            )
        ax.axhline(0.0, color="0.60", linewidth=1.0, linestyle="--", zorder=0.1)
        ax.set_ylabel("Permutation importance", fontweight="bold", color=target_color)
        add_panel_label(ax, PANEL_LABELS[row_idx * 2])
        ax.text(
            0.10,
            0.98,
            TARGET_SHORT_LABELS[target],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            color="0.12",
        )
        style_axis(ax)

        ax_retained = ax.twinx()
        ax_retained.plot(
            model_sub["buffer_km"],
            model_sub["retained_frac"],
            color="0.45",
            linestyle="--",
            linewidth=1.5,
            marker="s",
            markersize=3.8,
            label="Retained NPF fraction",
        )
        ax_retained.set_ylim(0.0, 1.05)
        ax_retained.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_retained.set_ylabel("Retained NPF fraction", fontweight="bold", color="0.40")
        ax_retained.tick_params(axis="y", colors="0.40")
        railbuf.apply_bold_ticklabels(ax_retained)

        ax = axes[row_idx, 1]
        for feature in THERMAL_FEATURES:
            feat_sub = thermal_sub.loc[thermal_sub["feature"].eq(feature)]
            ax.plot(
                feat_sub["buffer_km"],
                feat_sub["ale_span"],
                marker="o",
                linewidth=2.0,
                markersize=4.2,
                color=THERMAL_COLORS[feature],
            )
        ax.axhline(0.0, color="0.60", linewidth=1.0, linestyle="--", zorder=0.1)
        ax.set_ylabel("ALE span", fontweight="bold", color=target_color)
        add_panel_label(ax, PANEL_LABELS[row_idx * 2 + 1])
        style_axis(ax)

    axes[0, 0].set_title("Thermal importance after Zou-boundary exclusion", fontweight="bold", fontsize=10.0)
    axes[0, 1].set_title("Thermal ALE span after Zou-boundary exclusion", fontweight="bold", fontsize=10.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Excluded NPF buffer from Zou boundary (km)", fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=THERMAL_COLORS["magt"], linewidth=2.0, marker="o", label="MAGT"),
        Line2D([0], [0], color=THERMAL_COLORS["temperature_mean"], linewidth=2.0, marker="o", label="MAAT"),
        Line2D([0], [0], color="0.45", linewidth=1.5, linestyle="--", marker="s", label="Retained NPF fraction"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fancybox=False,
    )
    legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.96))
    legend.get_frame().set_edgecolor("0.72")

    fig.text(
        0.5,
        0.05,
        "If MAGT/MAAT importance and ALE span collapse after removing pixels near the Zou boundary, the apparent NPF thermal signal is likely transition leakage.",
        ha="center",
        va="center",
        fontsize=8,
        color="0.30",
    )

    out_png = fig_dir / f"{FIG_BASENAME}_buffer_sweep_summary.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_buffer_sweep_summary.pdf"
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def plot_thermal_ale_overlay_figure(
    *,
    curve_df: pd.DataFrame,
    buffers_km: tuple[float, ...],
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    palette = build_buffer_palette(buffers_km)
    fig, axes = plt.subplots(2, 2, figsize=A4_LANDSCAPE, sharex="col", sharey="row")
    plt.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.12, wspace=0.16, hspace=0.24)

    ylims = {
        target: compute_overlay_ylim(curve_df, target=target)
        for target in ("d_u", "grad_mag_km")
    }

    for row_idx, target in enumerate(("d_u", "grad_mag_km")):
        for col_idx, feature in enumerate(THERMAL_FEATURES):
            ax = axes[row_idx, col_idx]
            for buffer_km in buffers_km:
                sub = curve_df.loc[
                    curve_df["target"].eq(target)
                    & curve_df["feature"].eq(feature)
                    & np.isclose(curve_df["buffer_km"].to_numpy(dtype=float), float(buffer_km))
                ].sort_values("x_plot")
                if sub.empty:
                    continue
                ax.plot(
                    sub["x_plot"],
                    sub["y"],
                    color=palette[float(buffer_km)],
                    linewidth=2.2 if np.isclose(buffer_km, 0.0) else 1.8,
                    label=buffer_label(buffer_km),
                )
            ax.axhline(0.0, color="0.60", linewidth=1.0, linestyle="--", zorder=0.1)
            ax.set_xlim(*compute_overlay_xlim(curve_df, target=target, feature=feature))
            ax.set_ylim(*ylims[target])
            if row_idx == 0:
                ax.set_title(THERMAL_LABELS[feature], fontweight="bold", fontsize=10.0)
            if row_idx == 1:
                ax.set_xlabel(feature_axis_label(feature), fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(OVERLAY_YLABELS[target], fontweight="bold", color=TARGET_COLORS[target])
                add_panel_label(ax, PANEL_LABELS[row_idx * 2 + col_idx])
            else:
                add_panel_label(ax, PANEL_LABELS[row_idx * 2 + col_idx])
                ax.tick_params(axis="y", labelleft=False)
            style_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=palette[float(buffer_km)], linewidth=2.2 if np.isclose(buffer_km, 0.0) else 1.8, label=buffer_label(buffer_km))
        for buffer_km in buffers_km
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(5, len(legend_handles)),
        frameon=True,
        fancybox=False,
    )
    legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.96))
    legend.get_frame().set_edgecolor("0.72")

    fig.suptitle("NPF thermal ALE after progressive Zou-boundary exclusion", fontsize=11.5, fontweight="bold", y=0.94)
    fig.text(
        0.5,
        0.05,
        "Curves are centered ALE from refitted non-permafrost susceptibility models. Stable shapes after boundary exclusion argue against simple transition leakage.",
        ha="center",
        va="center",
        fontsize=8,
        color="0.30",
    )

    out_png = fig_dir / f"{FIG_BASENAME}_thermal_ale_overlay.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_thermal_ale_overlay.pdf"
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def plot_composite_transition_figure(
    *,
    distance_df: pd.DataFrame,
    model_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    buffers_km: tuple[float, ...],
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=COMPOSITE_FIGSIZE,
        gridspec_kw={"width_ratios": COMPOSITE_WIDTH_RATIOS},
    )
    plt.subplots_adjust(**COMPOSITE_SUBPLOTS_ADJUST)

    ymax_vals = pd.to_numeric(distance_df["rate_ci_high"], errors="coerce").to_numpy(dtype=float)
    ymax_vals = ymax_vals[np.isfinite(ymax_vals)]
    bar_ytop = max(0.15, float(np.nanmax(ymax_vals)) * 1.25) if ymax_vals.size else 0.15

    target_to_row = {"d_u": 0, "grad_mag_km": 1}

    for target, row_idx in target_to_row.items():
        ax = axes[row_idx, 0]
        sub = distance_df.loc[
            distance_df["target"].eq(target)
        ].sort_values("bin_index")

        x = np.arange(len(sub), dtype=float)
        color = TARGET_COLORS[target]

        ax.bar(
            x,
            sub["rate"],
            color=railbuf.blend_with_white(color, 0.68),
            edgecolor=color,
            linewidth=1.0,
        )

        yerr = np.vstack([
            np.maximum(sub["rate"] - sub["rate_ci_low"], 0.0),
            np.maximum(sub["rate_ci_high"] - sub["rate"], 0.0),
        ])
        ax.errorbar(
            x,
            sub["rate"],
            yerr=yerr,
            fmt="o-",
            color=color,
            linewidth=1.8,
            markersize=4.0,
            capsize=3.0,
            zorder=3,
        )

        for xi, rate, n in zip(x, sub["rate"], sub["n"], strict=False):
            if np.isfinite(rate):
                ax.text(
                    xi,
                    rate + 0.008,
                    f"n={int(n):,}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="0.35",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin_label"], fontsize=8)
        ax.set_ylim(0.0, bar_ytop)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Extreme rate within NPF", fontweight="bold")
        ax.set_xlabel("Distance to Zou PF boundary (km)", fontweight="bold")

        ax.text(
            0.96,
            0.95,
            TARGET_SHORT_LABELS[target],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color="0.12",
        )

        add_panel_label(ax, PANEL_LABELS[row_idx * 2])
        style_axis(ax)

    feature = "magt"
    palette = build_buffer_palette(buffers_km)

    ylims = {
        target: compute_overlay_ylim(curve_df, target=target)
        for target in ("d_u", "grad_mag_km")
    }

    for target, row_idx in target_to_row.items():
        ax = axes[row_idx, 1]

        ax.axvspan(
            MAGT_ZERO_BAND_LO,
            MAGT_ZERO_BAND_HI,
            color=MAGT_ZERO_BAND_COLOR,
            alpha=MAGT_ZERO_BAND_ALPHA,
            zorder=0,
        )
        ax.text(
            0.0,
            ylims[target][1] * 0.92,
            "0 °C",
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.45",
            style="italic",
        )

        for buffer_km in buffers_km:
            sub = curve_df.loc[
                curve_df["target"].eq(target)
                & curve_df["feature"].eq(feature)
                & np.isclose(
                    curve_df["buffer_km"].to_numpy(dtype=float),
                    float(buffer_km),
                )
            ].sort_values("x_plot")

            if sub.empty:
                continue

            enriched_label = buffer_label_with_retained(
                buffer_km,
                model_df,
                target,
            )
            lw = 2.4 if np.isclose(buffer_km, 0.0) else 1.6
            ax.plot(
                sub["x_plot"],
                sub["y"],
                color=palette[float(buffer_km)],
                linewidth=lw,
                label=enriched_label,
            )

        ax.axhline(0.0, color="0.60", linewidth=1.0, linestyle="--", zorder=0.1)

        ax.set_xlim(
            *compute_overlay_xlim(curve_df, target=target, feature=feature)
        )
        ax.set_ylim(*ylims[target])
        ax.set_xlabel(feature_axis_label(feature), fontweight="bold")
        ax.set_ylabel(
            OVERLAY_YLABELS[target],
            fontweight="bold",
            color=TARGET_COLORS[target],
        )

        ax.legend(
            loc="upper right",
            fontsize=7.0,
            framealpha=0.92,
            edgecolor="0.72",
            handlelength=1.8,
        )

        add_panel_label(ax, PANEL_LABELS[row_idx * 2 + 1])
        style_axis(ax)

    axes[0, 0].set_title(
        "Extreme prevalence vs. Zou-boundary distance",
        fontweight="bold",
        fontsize=9.5,
        pad=8,
    )
    axes[0, 1].set_title(
        "MAGT accumulated local effect under boundary exclusion",
        fontweight="bold",
        fontsize=9.5,
        pad=8,
    )

    fig.text(
        0.5,
        0.965,
        "Transition-zone diagnostic: NPF extreme signal concentration "
        "and thermal-control attribution",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    out_png = fig_dir / f"{FIG_BASENAME}_composite_transition_diagnostic.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_composite_transition_diagnostic.pdf"
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def plot_final_analysis_figure(
    *,
    distance_df: pd.DataFrame,
    model_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    sample_dfs: dict[str, pd.DataFrame],
    distance_bins_km: tuple[float, ...],
    buffers_km: tuple[float, ...],
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    try:
        from scipy.stats import gaussian_kde
    except ModuleNotFoundError:
        gaussian_kde = None

    fig, axes = plt.subplots(
        3,
        2,
        figsize=FINAL_FIGSIZE,
        gridspec_kw={"height_ratios": FINAL_HEIGHT_RATIOS},
    )
    plt.subplots_adjust(**FINAL_SUBPLOTS_ADJUST)

    targets = ("d_u", "grad_mag_km")
    x_grid = np.linspace(MAGT_XLIM[0], MAGT_XLIM[1], MAGT_XGRID_N)
    panel_map = {
        (0, 0): "A", (0, 1): "B",
        (1, 0): "C", (1, 1): "D",
        (2, 0): "E", (2, 1): "F",
    }

    def evaluate_kde(values: pd.Series) -> np.ndarray:
        vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            raise ValueError("Need at least two finite samples for KDE.")
        if np.isclose(np.nanstd(vals, ddof=1), 0.0):
            raise ValueError("KDE requires non-zero variance.")

        if gaussian_kde is not None:
            return gaussian_kde(vals, bw_method=RIDGE_KDE_BW)(x_grid)

        bandwidth = max(float(RIDGE_KDE_BW), 1e-3)
        diffs = (x_grid[:, None] - vals[None, :]) / bandwidth
        kernel = np.exp(-0.5 * np.square(diffs))
        norm = vals.size * bandwidth * math.sqrt(2.0 * math.pi)
        return kernel.sum(axis=1) / norm

    def draw_center_gradient_band(
        ax,
        *,
        x_lo: float,
        x_hi: float,
        x_center: float,
        y_lo: float,
        y_hi: float,
        color: str,
        alpha_max: float,
    ) -> None:
        if not (np.isfinite(x_lo) and np.isfinite(x_hi) and np.isfinite(x_center)):
            return
        if x_hi <= x_lo or y_hi <= y_lo:
            return

        xs = np.linspace(float(x_lo), float(x_hi), 512, dtype=float)
        x_center = float(np.clip(x_center, x_lo, x_hi))
        left_span = max(x_center - x_lo, 1e-6)
        right_span = max(x_hi - x_center, 1e-6)
        alpha = np.where(
            xs <= x_center,
            (xs - x_lo) / left_span,
            (x_hi - xs) / right_span,
        )
        alpha = np.clip(alpha, 0.0, 1.0) * float(alpha_max)

        rgba = np.zeros((2, xs.size, 4), dtype=float)
        rgba[..., :3] = matplotlib.colors.to_rgb(color)
        rgba[..., 3] = alpha[None, :]
        ax.imshow(
            rgba,
            extent=(float(x_lo), float(x_hi), float(y_lo), float(y_hi)),
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            zorder=0.05,
        )

    def add_outer_panel_label(ax, label: str) -> None:
        add_submission_panel_label(ax, label, x=-0.14, y=1.05)

    def add_extreme_share_donut(ax, *, share: float, color: str) -> None:
        if not np.isfinite(share):
            return
        share = float(np.clip(share, 0.0, 1.0))
        inset = ax.inset_axes([0.69, 0.56, 0.26, 0.34])
        if share >= 0.999:
            values = [1.0]
            colors = [color]
        else:
            values = [share, 1.0 - share]
            colors = [color, railbuf.blend_with_white(color, 0.86)]
        inset.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=colors,
            wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 0.7},
        )
        inset.text(
            0.0,
            0.10,
            f"{share * 100.0:.0f}%",
            ha="center",
            va="center",
            fontsize=FONT["annotation"] + 0.2,
            fontweight="bold",
            color="0.18",
        )
        inset.text(
            0.0,
            -0.20,
            "0-5 km",
            ha="center",
            va="center",
            fontsize=6.3,
            color="0.25",
        )
        inset.text(
            0.0,
            -1.16,
            "of all extremes",
            ha="center",
            va="top",
            fontsize=6.1,
            color="0.40",
        )
        inset.set_aspect("equal")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)

    ymax_vals = pd.to_numeric(
        distance_df["rate_ci_high"], errors="coerce"
    ).to_numpy(dtype=float)
    ymax_vals = ymax_vals[np.isfinite(ymax_vals)]
    bar_ytop = max(
        0.15, float(np.nanmax(ymax_vals)) * 1.25
    ) if ymax_vals.size else 0.15

    for col_idx, target in enumerate(targets):
        ax = axes[0, col_idx]
        sub = distance_df.loc[
            distance_df["target"].eq(target)
        ].sort_values("bin_index")

        x = np.arange(len(sub), dtype=float)
        color = TARGET_COLORS[target]

        ax.bar(
            x,
            sub["rate"],
            color=railbuf.blend_with_white(color, 0.68),
            edgecolor=color,
            linewidth=1.0,
        )
        yerr = np.vstack([
            np.maximum(sub["rate"] - sub["rate_ci_low"], 0.0),
            np.maximum(sub["rate_ci_high"] - sub["rate"], 0.0),
        ])
        ax.errorbar(
            x,
            sub["rate"],
            yerr=yerr,
            fmt="o-",
            color=color,
            linewidth=1.6,
            markersize=3.2,
            capsize=2.2,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin_label"], fontsize=FONT["tick"])
        ax.set_ylim(0.0, bar_ytop)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(
            "Distance to boundary (km)",
            fontweight="bold",
            fontsize=FONT["axis"],
        )
        if col_idx == 0:
            ax.set_ylabel(
                "Extreme rate within NPF",
                fontweight="bold",
                fontsize=FONT["axis"],
            )

        positives = pd.to_numeric(sub["positives"], errors="coerce").to_numpy(dtype=float)
        total_positive = float(np.nansum(positives))
        near_positive = float(positives[0]) if positives.size else np.nan
        share = (near_positive / total_positive) if total_positive > 0 else np.nan
        add_extreme_share_donut(ax, share=share, color=color)

        ax.set_title(
            TARGET_SHORT_LABELS[target],
            fontweight="bold",
            fontsize=FONT["title"],
            pad=4,
        )

        add_outer_panel_label(ax, panel_map[(0, col_idx)])
        railbuf.style_open_axes(ax)
        railbuf.apply_bold_ticklabels(ax)
        ax.grid(False)

    for col_idx, target in enumerate(targets):
        ax = axes[1, col_idx]
        magt_work = prepare_magt_by_distance(
            sample_dfs[target],
            target=target,
            distance_bins_km=distance_bins_km,
        )
        bin_labels_ordered = list(
            magt_work["distance_bin"].cat.categories
        )
        n_bins = len(bin_labels_ordered)
        if n_bins == 0:
            ax.set_xlim(*MAGT_XLIM)
            ax.set_xlabel("MAGT (°C)", fontweight="bold", fontsize=FONT["axis"])
            if col_idx == 0:
                ax.set_ylabel(
                    "Distance to boundary",
                    fontweight="bold",
                    fontsize=FONT["axis"],
                )
            add_outer_panel_label(ax, panel_map[(1, col_idx)])
            railbuf.style_open_axes(ax)
            railbuf.apply_bold_ticklabels(ax)
            ax.grid(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            continue

        kdes_all: dict[object, np.ndarray] = {}
        kdes_ext: dict[object, np.ndarray] = {}
        global_max = 0.0

        for bl in bin_labels_ordered:
            sub_all = magt_work.loc[
                magt_work["distance_bin"].eq(bl), "magt"
            ].to_numpy(dtype=float)
            if len(sub_all) >= 30:
                try:
                    density = evaluate_kde(pd.Series(sub_all))
                except (np.linalg.LinAlgError, ValueError):
                    pass
                else:
                    kdes_all[bl] = density
                    global_max = max(global_max, float(np.nanmax(density)))

            sub_ext = magt_work.loc[
                magt_work["distance_bin"].eq(bl)
                & magt_work["is_extreme"],
                "magt",
            ].to_numpy(dtype=float)
            if len(sub_ext) >= 10:
                try:
                    kdes_ext[bl] = evaluate_kde(pd.Series(sub_ext))
                except (np.linalg.LinAlgError, ValueError):
                    pass

        scale = (RIDGE_MAX_HEIGHT * RIDGE_SPACING) / global_max if global_max > 0 else 1.0
        y_offsets: dict[object, float] = {}

        for bin_idx, bl in enumerate(bin_labels_ordered):
            y_off = (n_bins - 1 - bin_idx) * RIDGE_SPACING * (1.0 - RIDGE_OVERLAP)
            y_offsets[bl] = y_off
            color = DIST_BIN_PALETTE[
                min(bin_idx, len(DIST_BIN_PALETTE) - 1)
            ]
            line_style = "-" if bin_idx == 0 else "--"

            if bl in kdes_all:
                d = kdes_all[bl] * scale
                ax.fill_between(
                    x_grid,
                    y_off,
                    y_off + d,
                    color=color,
                    alpha=RIDGE_ALL_ALPHA,
                    zorder=n_bins - bin_idx,
                )
                ax.plot(
                    x_grid,
                    y_off + d,
                    color=color,
                    linewidth=0.8,
                    linestyle=line_style,
                    alpha=RIDGE_ALL_LINE_ALPHA,
                    zorder=n_bins - bin_idx + 0.1,
                )

            if bl in kdes_ext:
                d_ext = kdes_ext[bl] * scale
                ax.fill_between(
                    x_grid,
                    y_off,
                    y_off + d_ext,
                    color=color,
                    alpha=RIDGE_EXT_ALPHA,
                    zorder=n_bins - bin_idx + 0.2,
                )
                ax.plot(
                    x_grid,
                    y_off + d_ext,
                    color=color,
                    linewidth=RIDGE_EXT_LINE_WIDTH,
                    linestyle=line_style,
                    zorder=n_bins - bin_idx + 0.3,
                )

                n_ext = int(
                    magt_work.loc[
                        magt_work["distance_bin"].eq(bl)
                        & magt_work["is_extreme"]
                    ].shape[0]
                )
                peak_idx = int(np.argmax(d_ext))
                peak_x = float(x_grid[peak_idx])
                peak_y = float(y_off + np.nanmax(d_ext))
                ax.text(
                    peak_x + 0.3,
                    peak_y,
                    f"n={n_ext:,}",
                    fontsize=6.4,
                    color=color,
                    ha="left",
                    va="center",
                    zorder=n_bins + 1,
                )

        ax.axvline(0.0, **ZERO_C_LINE)
        ax.text(
            0.0,
            y_offsets[bin_labels_ordered[0]] + RIDGE_MAX_HEIGHT * RIDGE_SPACING + 0.02,
            "0 °C",
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.45",
            style="italic",
        )

        ax.set_yticks(
            [y_offsets[bl] for bl in bin_labels_ordered]
        )
        ax.set_yticklabels(
            [f"{bl} km" for bl in bin_labels_ordered],
            fontsize=FONT["tick"],
            fontweight="bold",
        )
        ax.set_xlim(*MAGT_XLIM)
        ymax_ridge = max([y_offsets[bl] for bl in bin_labels_ordered] + [0.0])
        ax.set_ylim(
            -0.12,
            ymax_ridge + RIDGE_MAX_HEIGHT * RIDGE_SPACING + 0.18,
        )
        ax.set_xlabel("MAGT (°C)", fontweight="bold", fontsize=FONT["axis"])
        if col_idx == 0:
            ax.set_ylabel(
                "Distance to boundary",
                fontweight="bold",
                fontsize=FONT["axis"],
            )

        add_outer_panel_label(ax, panel_map[(1, col_idx)])
        railbuf.style_open_axes(ax)
        railbuf.apply_bold_ticklabels(ax)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    feature = "magt"
    palette = build_buffer_palette(buffers_km)
    ylims = {
        t: compute_overlay_ylim(curve_df, target=t)
        for t in targets
    }

    for col_idx, target in enumerate(targets):
        ax = axes[2, col_idx]
        magt_work = prepare_magt_by_distance(
            sample_dfs[target],
            target=target,
            distance_bins_km=distance_bins_km,
        )
        bin_labels_ordered = list(magt_work["distance_bin"].cat.categories)
        near_label = bin_labels_ordered[0] if bin_labels_ordered else None
        band_interval = None
        band_center = None
        if near_label is not None:
            sub_near = magt_work.loc[
                magt_work["distance_bin"].eq(near_label), "magt"
            ]
            near_vals = pd.to_numeric(sub_near, errors="coerce").to_numpy(dtype=float)
            near_vals = near_vals[np.isfinite(near_vals)]
            if near_vals.size >= 2:
                q_lo, q_hi = np.quantile(near_vals, [0.025, 0.975])
                band_interval = (float(q_lo), float(q_hi))
                try:
                    density_near = evaluate_kde(pd.Series(near_vals))
                except (np.linalg.LinAlgError, ValueError):
                    band_center = float(np.nanmedian(near_vals))
                else:
                    mask_band = (x_grid >= q_lo) & (x_grid <= q_hi)
                    if mask_band.any():
                        x_band = x_grid[mask_band]
                        d_band = density_near[mask_band]
                        band_center = float(x_band[int(np.nanargmax(d_band))])
                    else:
                        band_center = float(np.nanmedian(near_vals))

        for buffer_km in buffers_km:
            sub = curve_df.loc[
                curve_df["target"].eq(target)
                & curve_df["feature"].eq(feature)
                & np.isclose(
                    curve_df["buffer_km"].to_numpy(dtype=float),
                    float(buffer_km),
                )
            ].sort_values("x_plot")

            if sub.empty:
                continue

            enriched_label = buffer_label_with_retained(
                buffer_km, model_df, target,
            )
            lw = 2.2 if np.isclose(buffer_km, 0.0) else 1.3
            ax.plot(
                sub["x_plot"], sub["y"],
                color=palette[float(buffer_km)],
                linewidth=lw,
                label=enriched_label,
            )

        ax.axhline(
            0.0, color="0.60", linewidth=0.8,
            linestyle="--", zorder=0.1,
        )

        ax.set_xlim(*compute_overlay_xlim(
            curve_df, target=target, feature=feature
        ))
        ax.set_ylim(*ylims[target])
        if band_interval is not None and band_center is not None:
            draw_center_gradient_band(
                ax,
                x_lo=band_interval[0],
                x_hi=band_interval[1],
                x_center=band_center,
                y_lo=ylims[target][0],
                y_hi=ylims[target][1],
                color=ZERO_C_BAND_COLOR,
                alpha_max=ZERO_C_BAND_ALPHA,
            )
            ax.text(
                band_center,
                ylims[target][1] * 0.91,
                f"95% 0-5 km MAGT\n{band_interval[0]:.1f} to {band_interval[1]:.1f} °C",
                ha="center",
                va="top",
                fontsize=FONT["annotation"],
                color=DIST_BIN_PALETTE[0],
                linespacing=1.0,
                bbox={"facecolor": (1.0, 1.0, 1.0, 0.78), "edgecolor": "none", "pad": 1.2},
                zorder=4,
            )
        ax.axvline(0.0, **ZERO_C_LINE)
        ax.set_xlabel("MAGT (°C)", fontweight="bold", fontsize=FONT["axis"])
        ax.set_ylabel(
            OVERLAY_YLABELS[target],
            fontweight="bold",
            fontsize=FONT["axis"],
            color=TARGET_COLORS[target],
        )

        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="upper right",
                fontsize=FONT["legend"],
                framealpha=0.90,
                edgecolor="0.72",
                handlelength=1.5,
                labelspacing=0.25,
            )

        ax.text(
            0.0,
            ylims[target][1] * 0.93,
            "0 °C",
            ha="center",
            va="top",
            fontsize=FONT["annotation"],
            color="0.45",
            style="italic",
        )

        add_outer_panel_label(ax, panel_map[(2, col_idx)])
        railbuf.style_open_axes(ax)
        railbuf.apply_bold_ticklabels(ax)
        ax.grid(False)

    out_png = fig_dir / (
        f"{FIG_BASENAME}_final_transition_diagnostic.png"
    )
    out_pdf = fig_dir / (
        f"{FIG_BASENAME}_final_transition_diagnostic.pdf"
    )
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def plot_final_analysis_figure_clean(
    *,
    distance_df: pd.DataFrame,
    model_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    sample_dfs: dict[str, pd.DataFrame],
    distance_bins_km: tuple[float, ...],
    buffers_km: tuple[float, ...],
    fig_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    del model_df, sample_dfs, distance_bins_km

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.16, wspace=0.34, hspace=0.34)

    targets = ("d_u", "grad_mag_km")
    panel_map = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}
    feature = "magt"

    for col_idx, target in enumerate(targets):
        ax = axes[0, col_idx]
        sub = distance_df.loc[distance_df["target"].eq(target)].sort_values("bin_index")
        if sub.empty:
            continue

        x = np.arange(len(sub), dtype=float)
        color = TARGET_COLORS[target]
        sample_share = pd.to_numeric(sub["n"], errors="coerce").to_numpy(dtype=float)
        sample_share = sample_share / np.nansum(sample_share) if np.nansum(sample_share) > 0 else sample_share
        rate = pd.to_numeric(sub["rate"], errors="coerce").to_numpy(dtype=float)
        rate_lo = pd.to_numeric(sub["rate_ci_low"], errors="coerce").to_numpy(dtype=float)
        rate_hi = pd.to_numeric(sub["rate_ci_high"], errors="coerce").to_numpy(dtype=float)

        ax.bar(
            x,
            sample_share,
            color=railbuf.blend_with_white(color, 0.72),
            edgecolor=railbuf.blend_with_white(color, 0.40),
            linewidth=0.9,
            width=0.78,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin_label"], fontsize=FONT["tick"])
        ax.set_xlabel("Distance to boundary (km)", fontweight="bold", fontsize=FONT["axis"])
        ax.set_title(TARGET_SHORT_LABELS[target], fontweight="bold", fontsize=FONT["title"], pad=4)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0.0, max(0.20, float(np.nanmax(sample_share)) * 1.18))
        if col_idx == 0:
            ax.set_ylabel("NPF sample share", fontweight="bold", fontsize=FONT["axis"])

        ax_rate = ax.twinx()
        ax_rate.plot(x, rate, color=color, linewidth=1.8, zorder=3)
        ax_rate.fill_between(x, rate_lo, rate_hi, color=color, alpha=0.12, zorder=2)
        ax_rate.yaxis.set_major_formatter(PercentFormatter(1.0))
        ymax_rate = float(np.nanmax(rate_hi[np.isfinite(rate_hi)])) if np.isfinite(rate_hi).any() else 0.0
        ax_rate.set_ylim(0.0, max(0.06, 1.18 * ymax_rate))
        if col_idx == len(targets) - 1:
            ax_rate.set_ylabel("Extreme share", fontweight="bold", fontsize=FONT["axis"])
        ax_rate.spines["top"].set_visible(False)
        ax_rate.grid(False)

        add_submission_panel_label(ax, panel_map[(0, col_idx)], x=-0.14, y=1.04)
        railbuf.style_open_axes(ax)
        railbuf.apply_bold_ticklabels(ax)
        ax.grid(False)

    exclusion_handles: list[Line2D] = []
    exclusion_labels: list[str] = []
    for col_idx, target in enumerate(targets):
        ax = axes[1, col_idx]
        for lvl_idx, buffer_km in enumerate(buffers_km):
            sub = curve_df.loc[
                curve_df["target"].eq(target)
                & curve_df["feature"].eq(feature)
                & np.isclose(curve_df["buffer_km"].to_numpy(dtype=float), float(buffer_km))
            ].sort_values("x_plot")
            if sub.empty:
                continue

            line = ax.plot(
                sub["x_plot"],
                sub["y"],
                color=EXCLUSION_CURVE_COLORS[min(lvl_idx, len(EXCLUSION_CURVE_COLORS) - 1)],
                linewidth=1.6,
            )[0]
            if col_idx == 0:
                exclusion_handles.append(line)
                exclusion_labels.append(buffer_label(buffer_km))

        ax.axvspan(MAGT_ZERO_BAND_LO, MAGT_ZERO_BAND_HI, color="0.90", alpha=0.45, zorder=0)
        ax.axhline(0.0, color="0.60", linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(0.0, **ZERO_C_LINE)
        ax.set_xlim(*compute_overlay_xlim(curve_df, target=target, feature=feature))
        ax.set_ylim(*compute_overlay_ylim(curve_df, target=target))
        ax.set_xlabel("MAGT (°C)", fontweight="bold", fontsize=FONT["axis"])
        ax.set_ylabel(OVERLAY_YLABELS[target], fontweight="bold", fontsize=FONT["axis"], color=TARGET_COLORS[target])
        add_submission_panel_label(ax, panel_map[(1, col_idx)], x=-0.14, y=1.04)
        railbuf.style_open_axes(ax)
        railbuf.apply_bold_ticklabels(ax)
        ax.grid(False)

    top_legend = fig.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=railbuf.blend_with_white(TARGET_COLORS["d_u"], 0.72),
                edgecolor=railbuf.blend_with_white(TARGET_COLORS["d_u"], 0.40),
                label="NPF sample share",
            ),
            Line2D([0], [0], color="0.15", linewidth=1.8, label="Extreme share"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=True,
        fontsize=FONT["legend"],
    )
    top_legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.96))
    top_legend.get_frame().set_edgecolor("0.72")

    if exclusion_handles:
        lower_legend = fig.legend(
            handles=exclusion_handles,
            labels=exclusion_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=min(3, len(exclusion_handles)),
            frameon=True,
            fontsize=FONT["legend"],
        )
        lower_legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.96))
        lower_legend.get_frame().set_edgecolor("0.72")

    out_png = fig_dir / f"{FIG_BASENAME}_final_transition_diagnostic.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_final_transition_diagnostic.pdf"
    fig.savefig(out_png, dpi=int(dpi), facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the apparent NPF thermal signal is concentrated "
            "near the clean Zou et al. PF boundary."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-pf", type=int, default=figsus.DEFAULT_SAMPLE_PF)
    parser.add_argument("--sample-npf", type=int, default=figsus.DEFAULT_SAMPLE_NPF)
    parser.add_argument("--neighbors-trans", type=int, default=21)
    parser.add_argument("--block-size-km", type=float, default=fig6.SPATIAL_BLOCK_SIZE_M / 1000.0)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-missing-frac", type=float, default=fig6.MAX_FEATURE_MISSING_FRAC)
    parser.add_argument("--model-n-jobs", type=int, default=fig6.DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--distance-bins-km", type=str, default="0,5,10,20,40,inf")
    parser.add_argument("--buffer-sweep-km", type=str, default="0,5,10,20,40")
    parser.add_argument("--ale-bins", type=int, default=ALE_BINS)
    parser.add_argument("--ale-max-rows", type=int, default=ALE_MAX_ROWS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--zou-distance-mode",
        type=str,
        default=ZOU_DISTANCE_MODE_DEFAULT,
        choices=["lonlat", "projected"],
    )
    parser.add_argument("--zou-tif", type=Path, default=None)
    parser.add_argument("--project-prj", type=Path, default=None)
    args = parser.parse_args()

    railbuf.configure_style()

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    fig_dir = out_dir / "figures" / FIG_BASENAME
    table_dir = out_dir / "tables" / FIG_BASENAME
    cache_dir = out_dir / "cache"
    model_cache_dir = args.model_cache_dir.resolve() if args.model_cache_dir is not None else (cache_dir / f"{FIG_BASENAME}_models")
    for path in (fig_dir, table_dir, cache_dir, model_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv_path(base_dir)
    distance_bins_km = parse_float_sequence(args.distance_bins_km)
    buffer_sweep_km = parse_float_sequence(args.buffer_sweep_km)
    block_size_m = float(args.block_size_km) * 1000.0

    raw_features, contrast_features, combined_features, transition_base_vars = figsus.build_feature_sets(exclude_magt=False)
    transition_cache_dir = cache_dir / f"{figsus.FIG_BASENAME}_transition_rasters"
    transition_window_size = fig6.neighbors_to_window_size(args.neighbors_trans)
    log_step("Ensuring transition rasters are available")
    grid, _, metric_paths = fig6.ensure_transition_rasters(
        base_dir=base_dir,
        csv_path=csv_path,
        cache_dir=transition_cache_dir,
        vars_for_transition=transition_base_vars,
        chunksize=int(args.chunksize),
        window_size=transition_window_size,
    )

    zou_tif = (
        args.zou_tif.resolve()
        if args.zou_tif is not None
        else zou.resolve_zou_tif(base_dir)
    )
    if not zou_tif.exists():
        raise FileNotFoundError(f"Zou PF map not found: {zou_tif}")

    boundary_cache_path = cache_dir / f"{FIG_BASENAME}_zou_boundary_reference.joblib.gz"
    target_crs_wkt = None
    if str(args.zou_distance_mode) == "projected":
        project_prj = (
            args.project_prj.resolve()
            if args.project_prj is not None
            else None
        )
        target_crs_wkt = resolve_project_crs_wkt(base_dir, project_prj)

    zou_tif, boundary_ref = resolve_zou_boundary_reference(
        base_dir=base_dir,
        cache_path=boundary_cache_path,
        mode=str(args.zou_distance_mode),
        grid=grid if str(args.zou_distance_mode) == "projected" else None,
        target_crs_wkt=target_crs_wkt,
    )

    distance_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    thermal_rows: list[dict[str, object]] = []
    curve_rows: list[pd.DataFrame] = []
    sample_dfs_by_target: dict[str, pd.DataFrame] = {}
    enriched_cache_by_target: dict[str, str] = {}
    model_cache_paths: dict[str, dict[str, str]] = {"d_u": {}, "grad_mag_km": {}}

    for target in ("d_u", "grad_mag_km"):
        sample_df, base_sample_cache, enriched_sample_cache = resolve_enriched_target_sample(
            target=target,
            csv_path=csv_path,
            cache_dir=cache_dir,
            grid=grid,
            metric_paths=metric_paths,
            transition_base_vars=transition_base_vars,
            transition_window_size=transition_window_size,
            neighbors_trans=int(args.neighbors_trans),
            chunksize=int(args.chunksize),
            sample_pf=int(args.sample_pf),
            sample_npf=int(args.sample_npf),
            zou_tif=zou_tif,
            boundary_ref=boundary_ref,
            boundary_cache_path=boundary_cache_path,
            zou_distance_mode=str(args.zou_distance_mode),
        )
        sample_dfs_by_target[target] = sample_df
        enriched_cache_by_target[target] = str(enriched_sample_cache)

        distance_rows.append(
            build_distance_rate_table(
                sample_df,
                target=target,
                distance_bins_km=distance_bins_km,
            )
        )

        base_npf_rows = int(
            sample_df.loc[
                sample_df["domain"].eq("npf")
                & np.isfinite(pd.to_numeric(sample_df["zou_boundary_distance_km"], errors="coerce").to_numpy(dtype=float))
            ].shape[0]
        )

        for buffer_km in buffer_sweep_km:
            result, subset = resolve_buffer_model(
                target=target,
                sample_df=sample_df,
                csv_path=csv_path,
                enriched_sample_cache=enriched_sample_cache,
                model_cache_dir=model_cache_dir,
                buffer_km=float(buffer_km),
                combined_features=combined_features,
                transition_base_vars=transition_base_vars,
                transition_window_size=transition_window_size,
                neighbors_trans=int(args.neighbors_trans),
                block_size_m=block_size_m,
                max_missing_frac=float(args.max_missing_frac),
                model_n_jobs=max(1, int(args.model_n_jobs)),
                test_size=float(args.test_size),
                force_retrain=bool(args.force_retrain),
            )

            retained_rows = int(len(subset))
            retained_frac = (retained_rows / base_npf_rows) if base_npf_rows else np.nan
            if result is None:
                model_rows.append(
                    {
                        "target": target,
                        "buffer_km": float(buffer_km),
                        "buffer_label": buffer_label(buffer_km),
                        "base_npf_rows": base_npf_rows,
                        "retained_npf_rows": retained_rows,
                        "retained_frac": retained_frac,
                        "roc_auc": np.nan,
                        "ap": np.nan,
                        "brier": np.nan,
                        "positive_fraction": np.nan,
                        "n_rows": retained_rows,
                        "n_train": np.nan,
                        "n_test": np.nan,
                        "model_cache_path": "",
                        "status": "skipped",
                    }
                )
                for feature in THERMAL_FEATURES:
                    thermal_rows.append(
                        {
                            "target": target,
                            "buffer_km": float(buffer_km),
                            "feature": feature,
                            "importance": np.nan,
                            "importance_std": np.nan,
                            "ale_span": np.nan,
                            "ale_min": np.nan,
                            "ale_max": np.nan,
                            "support_lo": np.nan,
                            "support_hi": np.nan,
                            "curve_n_bins": 0,
                            "feature_present": False,
                        }
                    )
                continue

            model_cache_paths[target][buffer_tag(buffer_km)] = str(result["model_cache_path"])
            stack_metrics = result["suite_metrics"].get("Stack", {})
            model_rows.append(
                {
                    "target": target,
                    "buffer_km": float(buffer_km),
                    "buffer_label": buffer_label(buffer_km),
                    "base_npf_rows": base_npf_rows,
                    "retained_npf_rows": retained_rows,
                    "retained_frac": retained_frac,
                    "roc_auc": stack_metrics.get("roc_auc", np.nan),
                    "ap": stack_metrics.get("ap", np.nan),
                    "brier": stack_metrics.get("brier", np.nan),
                    "positive_fraction": result["positive_fraction"],
                    "n_rows": result["n_rows"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "model_cache_path": str(result["model_cache_path"]),
                    "status": "ok",
                }
            )

            model_features = list(result["feature_names"])
            X_background = sample_frame(
                fig6.make_model_frame(subset, model_features),
                max_rows=int(args.ale_max_rows),
            )
            importance_lookup = result["importance_df"].set_index("feature")
            for feature in THERMAL_FEATURES:
                curve, support = compute_ale_curve(
                    X_ref=X_background,
                    feature=feature,
                    predict_fn=lambda X, model=result["full_stack"]: predict_proba_batched(model, X),
                    n_bins=int(args.ale_bins),
                )
                if not curve.empty:
                    curve_rows.append(
                        curve.assign(
                            target=target,
                            buffer_km=float(buffer_km),
                            feature=feature,
                        )
                    )
                importance = float(importance_lookup.at[feature, "importance"]) if feature in importance_lookup.index else np.nan
                importance_std = float(importance_lookup.at[feature, "importance_std"]) if feature in importance_lookup.index else np.nan
                ale_min = float(np.nanmin(curve["y"])) if not curve.empty else np.nan
                ale_max = float(np.nanmax(curve["y"])) if not curve.empty else np.nan
                thermal_rows.append(
                    {
                        "target": target,
                        "buffer_km": float(buffer_km),
                        "feature": feature,
                        "importance": importance,
                        "importance_std": importance_std,
                        "ale_span": (ale_max - ale_min) if np.isfinite(ale_min) and np.isfinite(ale_max) else np.nan,
                        "ale_min": ale_min,
                        "ale_max": ale_max,
                        "support_lo": support[0] if support is not None else np.nan,
                        "support_hi": support[1] if support is not None else np.nan,
                        "curve_n_bins": int(len(curve)),
                        "feature_present": bool(feature in model_features),
                    }
                )

    distance_df = pd.concat(distance_rows, ignore_index=True) if distance_rows else pd.DataFrame()
    model_df = pd.DataFrame(model_rows)
    thermal_df = pd.DataFrame(thermal_rows)
    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    distance_path = table_dir / f"{FIG_BASENAME}_distance_rate_summary.csv"
    model_path = table_dir / f"{FIG_BASENAME}_buffer_model_summary.csv"
    thermal_path = table_dir / f"{FIG_BASENAME}_thermal_summary.csv"
    curves_path = table_dir / f"{FIG_BASENAME}_thermal_ale_curves.csv"
    distance_df.to_csv(distance_path, index=False)
    model_df.to_csv(model_path, index=False)
    thermal_df.to_csv(thermal_path, index=False)
    curve_df.to_csv(curves_path, index=False)

    log_step("Building diagnostic figures")
    distance_fig = plot_distance_rate_figure(distance_df=distance_df, fig_dir=fig_dir, dpi=int(args.dpi))
    buffer_fig = plot_buffer_summary_figure(model_df=model_df, thermal_df=thermal_df, fig_dir=fig_dir, dpi=int(args.dpi))
    ale_fig = plot_thermal_ale_overlay_figure(
        curve_df=curve_df,
        buffers_km=buffer_sweep_km,
        fig_dir=fig_dir,
        dpi=int(args.dpi),
    )
    composite_fig = plot_composite_transition_figure(
        distance_df=distance_df,
        model_df=model_df,
        curve_df=curve_df,
        buffers_km=buffer_sweep_km,
        fig_dir=fig_dir,
        dpi=int(args.dpi),
    )
    log_step(
        f"Saved composite figure: {composite_fig[0]} | {composite_fig[1]}"
    )
    final_fig = plot_final_analysis_figure_clean(
        distance_df=distance_df,
        model_df=model_df,
        curve_df=curve_df,
        sample_dfs=sample_dfs_by_target,
        distance_bins_km=distance_bins_km,
        buffers_km=buffer_sweep_km,
        fig_dir=fig_dir,
        dpi=int(args.dpi),
    )
    log_step(
        f"Saved final transition diagnostic: "
        f"{final_fig[0]} | {final_fig[1]}"
    )

    meta = {
        "figure_outputs": {
            "distance_rate_summary": {"png": str(distance_fig[0]), "pdf": str(distance_fig[1])},
            "buffer_sweep_summary": {"png": str(buffer_fig[0]), "pdf": str(buffer_fig[1])},
            "thermal_ale_overlay": {"png": str(ale_fig[0]), "pdf": str(ale_fig[1])},
            "composite_transition_diagnostic": {"png": str(composite_fig[0]), "pdf": str(composite_fig[1])},
            "final_transition_diagnostic": {"png": str(final_fig[0]), "pdf": str(final_fig[1])},
        },
        "table_outputs": {
            "distance_rate_summary": str(distance_path),
            "buffer_model_summary": str(model_path),
            "thermal_summary": str(thermal_path),
            "thermal_ale_curves": str(curves_path),
        },
        "csv_path": str(csv_path),
        "zou_tif": str(zou_tif),
        "zou_boundary_cache_path": str(boundary_cache_path),
        "zou_distance_mode": str(args.zou_distance_mode),
        "enriched_sample_cache_by_target": enriched_cache_by_target,
        "model_cache_paths": model_cache_paths,
        "distance_bins_km": list(distance_bins_km),
        "buffer_sweep_km": list(buffer_sweep_km),
        "thermal_features": list(THERMAL_FEATURES),
        "ale_bins": int(args.ale_bins),
        "ale_max_rows": int(args.ale_max_rows),
        "negative_du_only": bool(figsus.NEGATIVE_DU_ONLY),
        "transition_window_size": int(transition_window_size),
        "transition_base_vars": transition_base_vars,
        "feature_transforms": {
            "temperature_mean": "(x - 32) * 5 / 9",
        },
    }
    meta_path = cache_dir / f"{FIG_BASENAME}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log_step(f"Saved distance-rate table: {distance_path}")
    log_step(f"Saved model-summary table: {model_path}")
    log_step(f"Saved thermal-summary table: {thermal_path}")
    log_step(f"Saved ALE curve table: {curves_path}")
    log_step(f"Saved Zou boundary cache: {boundary_cache_path}")
    log_step(f"Saved meta JSON: {meta_path}")
    log_step(f"Saved figure PNG/PDF: {distance_fig[0]} | {distance_fig[1]}")
    log_step(f"Saved figure PNG/PDF: {buffer_fig[0]} | {buffer_fig[1]}")
    log_step(f"Saved figure PNG/PDF: {ale_fig[0]} | {ale_fig[1]}")
    log_step(f"Saved figure PNG/PDF: {composite_fig[0]} | {composite_fig[1]}")
    log_step(f"Saved figure PNG/PDF: {final_fig[0]} | {final_fig[1]}")


if __name__ == "__main__":
    main()
