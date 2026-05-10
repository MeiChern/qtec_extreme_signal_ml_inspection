#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/figure7_2_lake_influence_du_gradient.py
# Renamed package path: code/original_project_helpers/lake_influence_context_helpers.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import kendalltau

from figure6_0_transition_metric_review import (
    build_or_load_grid,
    choose_stride,
    en_to_rc,
    get_extent,
    open_memmap,
)


SEED = 42
CHUNKSIZE = 300_000
FIG_BASENAME = "Figure7_2_lake_influence_du_gradient"
SUBPLOT_B_FIG_BASENAME = "Figure7_2_lake_influence_du_gradient_panelB"
DU_COLOR = "#6F94B2"
GRAD_COLOR = "#9B7756"
GRAD_SUSC_COLOR = "#3D7A4A"
DUHOT_SUSC_COLOR = "#7A4AA0"
MAP_BACKGROUND = "#C1C1C1"
MUTED_REDS = LinearSegmentedColormap.from_list(
    "muted_reds",
    ["#FAEEEE", "#F1D5D5", "#E3B7B7", "#CD8F8F", "#A95C5C"],
)
METRIC_LABELS = {
    "d_u": r"$d_u$",
    "grad_mag_km": r"$d_u$ gradient",
    "gradient_hotspot_susceptibility": "Grad hotspot susc.",
    "du_hotspot_susceptibility": r"$d_u$ hotspot susc.",
}
RANK_WARNING = getattr(np, "RankWarning", RuntimeWarning)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def log_step(message: str) -> None:
    print(f"[figure7_2] {message}")


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


def add_horizontal_background_gradient(ax) -> None:
    bg = np.linspace(0.0, 1.0, 512)[None, :]
    bg = np.vstack([bg, bg])
    ax.imshow(
        bg,
        extent=(0.0, 1.0, 0.0, 1.0),
        transform=ax.transAxes,
        origin="lower",
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list(
            "lake_influence_bg",
            ["#E7F0F8", "#F1F4EE", "#F1E1D4"],
        ),
        alpha=0.42,
        zorder=0,
        interpolation="bicubic",
    )


def style_axes(ax, *, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, colors="#333333")
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def iter_csv_chunks(
    csv_path: Path,
    *,
    usecols: list[str],
    chunksize: int,
    progress_every: int = 10,
):
    for idx, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        if idx == 1 or idx % progress_every == 0:
            log_step(f"processed {idx} chunk(s)")
        yield chunk
    log_step("finished reading CSV")


def engineer_features(df: pd.DataFrame, *, copy_df: bool = False) -> pd.DataFrame:
    out = df.copy() if copy_df else df
    numeric_cols = [
        "easting",
        "northing",
        "longitude",
        "latitude",
        "d_u",
        "dudx",
        "dudy",
        "grad_mag",
        "grad_mag_km",
        "Perma_Distr_map",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "grad_mag" in out.columns:
        grad_mag_km = out["grad_mag"] * 1000.0
        if "grad_mag_km" in out.columns:
            out["grad_mag_km"] = out["grad_mag_km"].where(np.isfinite(out["grad_mag_km"]), grad_mag_km)
        else:
            out["grad_mag_km"] = grad_mag_km
    if "Perma_Distr_map" in out.columns:
        out["domain"] = np.where(
            out["Perma_Distr_map"] == 1,
            "pf",
            np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
        )
    out["abs_d_u"] = np.abs(pd.to_numeric(out.get("d_u"), errors="coerce"))
    return out


def build_pf_negative_du_sample(
    csv_path: Path,
    *,
    usecols: list[str],
    desired_n: int,
    chunksize: int,
    seed: int,
) -> pd.DataFrame:
    log_step("building PF negative-d_u sample")
    rng = np.random.default_rng(seed)
    reservoir = pd.DataFrame(columns=usecols + ["_sample_key"])

    for chunk in iter_csv_chunks(csv_path, usecols=usecols, chunksize=chunksize):
        chunk = engineer_features(chunk)
        mask = (
            chunk["domain"].eq("pf")
            & np.isfinite(chunk["easting"])
            & np.isfinite(chunk["northing"])
            & np.isfinite(chunk["d_u"])
            & np.isfinite(chunk["grad_mag"])
            & (chunk["d_u"] < 0.0)
        )
        sub = chunk.loc[mask].copy()
        if sub.empty:
            continue
        sub["_sample_key"] = rng.random(len(sub))
        keep = pd.concat([reservoir, sub], ignore_index=True)
        if len(keep) > desired_n:
            keep = keep.nsmallest(desired_n, "_sample_key").reset_index(drop=True)
        reservoir = keep

    out = reservoir.drop(columns="_sample_key", errors="ignore").copy()
    if out.empty:
        raise RuntimeError("Sampling returned no PF negative-d_u rows.")
    if len(out) > desired_n:
        out = out.sample(desired_n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_lake_influence_config(base_dir: Path, out_dir: Path, csv_path: Path) -> dict[str, object]:
    meta_path = out_dir / "cache" / "lake_influence_layer_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing lake-influence metadata: {meta_path}. "
            "Run generate_lake_influence_layer.py first."
        )
    meta = json.loads(meta_path.read_text())
    outputs = meta.get("outputs", {})
    norm_memmap = outputs.get("norm_memmap")
    if not norm_memmap:
        raise RuntimeError("Lake-influence metadata does not contain outputs.norm_memmap.")
    norm_memmap_path = Path(norm_memmap)
    if not norm_memmap_path.exists():
        raise FileNotFoundError(
            f"Missing normalized lake-influence raster: {norm_memmap_path}. "
            "Run generate_lake_influence_layer.py first."
        )

    grid = build_or_load_grid(base_dir, csv_path)
    meta_grid = meta.get("grid", {})
    if int(meta_grid.get("nrows", grid["nrows"])) != int(grid["nrows"]) or int(
        meta_grid.get("ncols", grid["ncols"])
    ) != int(grid["ncols"]):
        raise RuntimeError("Lake-influence grid shape does not match the project pixel grid.")

    pf_mask_path: Path | None = None
    pf_candidate = meta.get("visualization", {}).get("permafrost_mask_path")
    if pf_candidate:
        cand = Path(pf_candidate)
        if cand.exists():
            pf_mask_path = cand
    if pf_mask_path is None:
        cand = base_dir / "outputs" / "gradient_driver_analysis" / "rasters" / "permafrost_u8.memmap"
        if cand.exists():
            pf_mask_path = cand
    if pf_mask_path is None:
        raise FileNotFoundError("Could not locate the permafrost mask memmap.")

    du_path = base_dir / "outputs" / "grad_rasters" / "du_f32.memmap"
    if not du_path.exists():
        raise FileNotFoundError(f"Missing d_u raster: {du_path}")

    return {
        "meta_path": meta_path,
        "grid": grid,
        "norm_memmap_path": norm_memmap_path,
        "pf_mask_path": pf_mask_path,
        "du_path": du_path,
    }


def load_gradient_hotspot_reference(
    base_dir: Path,
    out_dir: Path,
    csv_path: Path,
) -> dict[str, object]:
    prob_path = out_dir / "cache" / "figure6_pf_susceptibility_prob_f32.memmap"
    meta_path = out_dir / "cache" / "figure6_meta.json"
    if not prob_path.exists():
        raise FileNotFoundError(
            f"Missing gradient-hotspot susceptibility raster: {prob_path}. "
            "Run figure6_susceptibility_stacked.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing Figure 6 metadata: {meta_path}. "
            "Run figure6_susceptibility_stacked.py first."
        )
    return {
        "meta_path": meta_path,
        "grid": build_or_load_grid(base_dir, csv_path),
        "prob_path": prob_path,
    }


def load_du_hotspot_reference(
    base_dir: Path,
    out_dir: Path,
    csv_path: Path,
) -> dict[str, object]:
    meta_path = out_dir / "cache" / "fetch_d_u_hotspot_susceptibity_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing d_u-hotspot susceptibility metadata: {meta_path}. "
            "Run fetch_d_u_hotspot_susceptibity.py first."
        )
    meta = json.loads(meta_path.read_text())
    prob_path = Path(meta.get("du_hotspot_prob_raster", out_dir / "cache" / "fetch_du_hotspot_pf_susceptibility_prob_f32.memmap"))
    if not prob_path.exists():
        raise FileNotFoundError(
            f"Missing d_u-hotspot susceptibility raster: {prob_path}. "
            "Run fetch_d_u_hotspot_susceptibity.py first."
        )
    return {
        "meta_path": meta_path,
        "grid": build_or_load_grid(base_dir, csv_path),
        "prob_path": prob_path,
    }


def sample_raster_at_points(
    raster_path: Path,
    *,
    grid: dict[str, float | int],
    easting: pd.Series,
    northing: pd.Series,
    dtype: str = "float32",
) -> np.ndarray:
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    raster_mm = open_memmap(raster_path, dtype=dtype, mode="r", shape=(nrows, ncols))
    e = pd.to_numeric(easting, errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(northing, errors="coerce").to_numpy(dtype=float)
    row, col = en_to_rc(
        e,
        n,
        res=float(grid["res"]),
        gx0=int(grid["gx0"]),
        gy1=int(grid["gy1"]),
    )
    ok = np.isfinite(e) & np.isfinite(n) & (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)
    out = np.full(len(e), np.nan, dtype=np.float32)
    if ok.any():
        out[ok] = raster_mm[row[ok], col[ok]].astype(np.float32, copy=False)
    return out


def attach_lake_and_susceptibility(
    df: pd.DataFrame,
    *,
    lake_cfg: dict[str, object],
    grad_ref: dict[str, object],
    du_ref: dict[str, object],
) -> pd.DataFrame:
    out = engineer_features(df, copy_df=True)
    out["lake_influence_norm01"] = sample_raster_at_points(
        Path(lake_cfg["norm_memmap_path"]),
        grid=lake_cfg["grid"],
        easting=out["easting"],
        northing=out["northing"],
    )
    out["gradient_hotspot_susceptibility"] = sample_raster_at_points(
        Path(grad_ref["prob_path"]),
        grid=grad_ref["grid"],
        easting=out["easting"],
        northing=out["northing"],
    )
    out["du_hotspot_susceptibility"] = sample_raster_at_points(
        Path(du_ref["prob_path"]),
        grid=du_ref["grid"],
        easting=out["easting"],
        northing=out["northing"],
    )
    return out


def build_smooth_profile(
    df: pd.DataFrame,
    *,
    x_col: str,
    metrics: list[str],
    window_frac: float,
    n_profile_points: int,
) -> pd.DataFrame:
    cols = [x_col] + metrics
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col]).copy()
    work = work.sort_values(x_col).reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No rows available for continuous line analysis.")

    n = len(work)
    window_n = max(301, int(window_frac * n))
    window_n = min(window_n, n)
    if window_n % 2 == 0:
        window_n = max(1, window_n - 1)
    if window_n < 21:
        window_n = min(n, 21 if n >= 21 else n)
    half = max(1, window_n // 2)

    if n <= window_n:
        centers = np.array([n // 2], dtype=int)
    else:
        n_centers = max(25, min(int(n_profile_points), n - window_n + 1))
        centers = np.linspace(half, n - half - 1, n_centers).astype(int)
        centers = np.unique(np.clip(centers, half, n - half - 1))

    rows: list[dict[str, float]] = []
    for center in centers:
        start = max(0, center - half)
        stop = min(n, center + half + 1)
        win = work.iloc[start:stop]
        row = {
            "lake_influence_mean": float(np.nanmean(win[x_col].to_numpy(dtype=float))),
            "lake_influence_median": float(np.nanmedian(win[x_col].to_numpy(dtype=float))),
            "n_window": int(len(win)),
        }
        for metric in metrics:
            vals = pd.to_numeric(win[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                row[f"{metric}_mean"] = float(np.nanmean(vals))
                row[f"{metric}_std"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
        rows.append(row)

    profile = pd.DataFrame(rows)
    profile = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=["lake_influence_mean"]).reset_index(drop=True)
    if profile.empty:
        raise RuntimeError("Continuous profile construction returned no usable points.")
    return profile


def extend_profile_to_unit_interval(
    profile_df: pd.DataFrame,
    *,
    metric_names: list[str],
    x_col: str = "lake_influence_mean",
) -> pd.DataFrame:
    out = profile_df.sort_values(x_col).reset_index(drop=True).copy()
    if out.empty:
        return out

    rows_to_add: list[pd.Series] = []
    first_x = float(out[x_col].iloc[0])
    last_x = float(out[x_col].iloc[-1])

    if np.isfinite(first_x) and first_x > 0.0:
        first_row = out.iloc[0].copy()
        first_row[x_col] = 0.0
        first_row["lake_influence_median"] = 0.0
        rows_to_add.append(first_row)

    if np.isfinite(last_x) and last_x < 1.0:
        last_row = out.iloc[-1].copy()
        last_row[x_col] = 1.0
        last_row["lake_influence_median"] = 1.0
        rows_to_add.append(last_row)

    if rows_to_add:
        out = pd.concat([out, pd.DataFrame(rows_to_add)], ignore_index=True)
        out = out.sort_values(x_col).reset_index(drop=True)

    cols_to_keep = [x_col, "lake_influence_median", "n_window"]
    for metric in metric_names:
        cols_to_keep.extend([f"{metric}_mean", f"{metric}_std"])
    return out.loc[:, [c for c in cols_to_keep if c in out.columns]].copy()


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return np.nan
    if weights is None:
        return float(np.nanmean(vals))
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float(np.nanmean(vals))
    return float(np.average(vals[mask], weights=w[mask]))


def fit_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
        weights = weights[mask]
    else:
        mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([np.nan, np.nan], dtype=float), np.array([], dtype=float)

    constant = weighted_mean(y, weights)
    if len(x) == 1:
        return np.array([0.0, constant], dtype=float), np.full_like(x, constant, dtype=float)

    x_span = float(np.nanmax(x) - np.nanmin(x))
    if (not np.isfinite(x_span)) or x_span <= 1e-10:
        return np.array([0.0, constant], dtype=float), np.full_like(x, constant, dtype=float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=RANK_WARNING)
            if weights is not None and len(weights) == len(x):
                coef = np.polyfit(x, y, 1, w=np.sqrt(np.maximum(weights, 1e-6)))
            else:
                coef = np.polyfit(x, y, 1)
        coef = np.asarray(coef, dtype=float)
        if not np.isfinite(coef).all():
            raise np.linalg.LinAlgError("Non-finite polyfit coefficients")
        pred = np.polyval(coef, x)
        if not np.isfinite(pred).all():
            raise np.linalg.LinAlgError("Non-finite fitted values")
        return coef, np.asarray(pred, dtype=float)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return np.array([0.0, constant], dtype=float), np.full_like(x, constant, dtype=float)


def fit_piecewise_linear(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, object]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if len(x) < 3:
        _, pred = fit_line(x, y, w)
        return {"x": x, "pred": pred, "break_x": np.nan, "n_segments": 1}

    best: dict[str, object] | None = None
    best_sse = np.inf
    for split in range(1, len(x) - 1):
        if split + 1 < 2 or len(x) - split < 2:
            continue
        x_left = x[:split + 1]
        y_left = y[:split + 1]
        w_left = w[:split + 1]
        x_right = x[split:]
        y_right = y[split:]
        w_right = w[split:]
        _, pred_left = fit_line(x_left, y_left, w_left)
        _, pred_right = fit_line(x_right, y_right, w_right)
        pred = np.empty_like(y, dtype=float)
        pred[:split] = pred_left[:-1]
        pred[split:] = pred_right
        sse = float(np.sum(w * (y - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = {
                "x": x,
                "pred": pred,
                "break_x": float(x[split]),
                "n_segments": 2,
            }
    if best is None:
        _, pred = fit_line(x, y, w)
        return {"x": x, "pred": pred, "break_x": np.nan, "n_segments": 1}
    return best


def build_trend_table(sample_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "d_u",
        "grad_mag_km",
        "gradient_hotspot_susceptibility",
        "du_hotspot_susceptibility",
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        raw = sample_df[["lake_influence_norm01", metric]].replace([np.inf, -np.inf], np.nan).dropna()
        tau, p_value = (np.nan, np.nan)
        if len(raw) >= 3 and raw["lake_influence_norm01"].nunique() >= 2:
            tau, p_value = kendalltau(
                raw["lake_influence_norm01"].to_numpy(dtype=float),
                raw[metric].to_numpy(dtype=float),
            )
        fit = fit_piecewise_linear(
            profile_df["lake_influence_mean"].to_numpy(dtype=float),
            profile_df[f"{metric}_mean"].to_numpy(dtype=float),
            weights=profile_df["n_window"].fillna(0).to_numpy(dtype=float),
        )
        rows.append({
            "metric": metric,
            "n_raw": int(len(raw)),
            "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
            "kendall_p": float(p_value) if np.isfinite(p_value) else np.nan,
            "piecewise_break_lake_influence": float(fit["break_x"]) if np.isfinite(fit["break_x"]) else np.nan,
            "piecewise_n_segments": int(fit["n_segments"]),
        })
    return pd.DataFrame(rows)


def load_negative_pf_lake_preview(lake_cfg: dict[str, object]) -> tuple[np.ndarray, list[float]]:
    grid = lake_cfg["grid"]
    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    stride = choose_stride(nrows, ncols, target_max=1100)
    lake_mm = open_memmap(Path(lake_cfg["norm_memmap_path"]), dtype="float32", mode="r", shape=(nrows, ncols))
    pf_mm = open_memmap(Path(lake_cfg["pf_mask_path"]), dtype="uint8", mode="r", shape=(nrows, ncols))
    du_mm = open_memmap(Path(lake_cfg["du_path"]), dtype="float32", mode="r", shape=(nrows, ncols))

    lake_preview = np.array(lake_mm[::stride, ::stride], copy=False).astype(np.float32)
    pf_preview = np.array(pf_mm[::stride, ::stride], copy=False) == np.uint8(1)
    du_preview = np.array(du_mm[::stride, ::stride], copy=False).astype(np.float32)
    valid_mask = pf_preview & np.isfinite(du_preview) & (du_preview < 0.0)
    lake_preview = np.where(valid_mask, lake_preview, np.nan)

    extent = get_extent(
        min_e=float(grid["min_e"]),
        max_n=float(grid["max_n"]),
        nrows=nrows,
        ncols=ncols,
        res=float(grid["res"]),
    )
    return lake_preview, extent


def compute_band(mean_vals: np.ndarray, std_vals: np.ndarray, scale: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    mean_vals = np.asarray(mean_vals, dtype=float)
    std_vals = np.asarray(std_vals, dtype=float)
    band = scale * np.where(np.isfinite(std_vals), std_vals, 0.0)
    return mean_vals - band, mean_vals + band


def format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-3:
        return "<1e-3"
    return f"{p_value:.3f}"


def metric_annotation(trend_df: pd.DataFrame, metric: str) -> str:
    label = METRIC_LABELS.get(metric, metric)
    sub = trend_df.loc[trend_df["metric"] == metric]
    if sub.empty:
        return f"{label}: no trend"
    row = sub.iloc[0]
    tau = row["kendall_tau"]
    p_value = row["kendall_p"]
    break_x = row["piecewise_break_lake_influence"]
    if np.isfinite(break_x):
        return f"{label}: tau={tau:.2f}, p {format_p_value(p_value)}, break={break_x:.2f}"
    return f"{label}: tau={tau:.2f}, p {format_p_value(p_value)}"


def plot_continuous_panel(
    ax,
    profile_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    *,
    bottom_metric: str,
    bottom_label: str,
    bottom_color: str,
    top_metric: str,
    top_label: str,
    top_color: str,
    title: str | None,
    panel_label: str | None,
    force_zero_to_one: bool = False,
    bottom_shade_scale: float = 0.1,
    top_shade_scale: float = 0.1,
    background_gradient: bool = False,
) -> None:
    style_axes(ax, grid_axis="y")
    if background_gradient:
        add_horizontal_background_gradient(ax)
    x = profile_df["lake_influence_mean"].to_numpy(dtype=float)
    w = profile_df["n_window"].to_numpy(dtype=float)

    bottom_mean = profile_df[f"{bottom_metric}_mean"].to_numpy(dtype=float)
    bottom_std = profile_df[f"{bottom_metric}_std"].to_numpy(dtype=float)
    valid_bottom = np.isfinite(x) & np.isfinite(bottom_mean)
    xb = x[valid_bottom]
    yb = bottom_mean[valid_bottom]
    wb = w[valid_bottom]
    lo_b, hi_b = compute_band(yb, bottom_std[valid_bottom], scale=bottom_shade_scale)
    ax.fill_between(xb, lo_b, hi_b, color=bottom_color, alpha=0.18)
    line_bottom, = ax.plot(xb, yb, color=bottom_color, linewidth=2.0, label=bottom_label)
    fit_bottom = fit_piecewise_linear(xb, yb, wb)
    if len(fit_bottom["x"]):
        ax.plot(fit_bottom["x"], fit_bottom["pred"], color=bottom_color, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Lake influence")
    ax.set_ylabel(bottom_label, color=bottom_color)
    ax.tick_params(axis="y", colors=bottom_color)
    ax.spines["left"].set_color(bottom_color)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    if force_zero_to_one:
        ax.set_ylim(0.0, 1.0)

    ax2 = ax.twinx()
    ax2.patch.set_alpha(0.0)
    ax2.spines["top"].set_visible(False)
    top_mean = profile_df[f"{top_metric}_mean"].to_numpy(dtype=float)
    top_std = profile_df[f"{top_metric}_std"].to_numpy(dtype=float)
    valid_top = np.isfinite(x) & np.isfinite(top_mean)
    xt = x[valid_top]
    yt = top_mean[valid_top]
    wt = w[valid_top]
    lo_t, hi_t = compute_band(yt, top_std[valid_top], scale=top_shade_scale)
    ax2.fill_between(xt, lo_t, hi_t, color=top_color, alpha=0.16)
    line_top, = ax2.plot(xt, yt, color=top_color, linewidth=2.0, label=top_label)
    fit_top = fit_piecewise_linear(xt, yt, wt)
    if len(fit_top["x"]):
        ax2.plot(fit_top["x"], fit_top["pred"], color=top_color, linestyle="--", linewidth=1.2)
    ax2.set_ylabel(top_label, color=top_color)
    ax2.tick_params(axis="y", colors=top_color)
    ax2.spines["right"].set_color(top_color)
    if force_zero_to_one:
        ax2.set_ylim(0.0, 1.0)

    if title:
        ax.set_title(title)
    if panel_label:
        add_panel_label(ax, panel_label)
    ax.legend([line_bottom, line_top], [bottom_label, top_label], frameon=False, loc="lower right")
    ax.text(
        0.98,
        0.98,
        "\n".join([
            metric_annotation(trend_df, bottom_metric),
            metric_annotation(trend_df, top_metric),
        ]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.20", facecolor=(1.0, 1.0, 1.0, 0.82), edgecolor="none"),
    )


def plot_map_panel(ax, lake_preview: np.ndarray, extent: list[float]) -> None:
    ax.set_facecolor(MAP_BACKGROUND)
    im = ax.imshow(
        np.ma.masked_invalid(lake_preview),
        extent=extent,
        origin="upper",
        cmap=MUTED_REDS,
        norm=Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("PF pixels with negative $d_u$: continuous lake influence")
    add_panel_label(ax, "A")
    cb = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
    cb.set_label("Lake influence")


def make_figure(
    profile_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    lake_cfg: dict[str, object],
    out_png: Path,
    out_pdf: Path,
) -> None:
    configure_style()
    lake_preview, extent = load_negative_pf_lake_preview(lake_cfg)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]},
    )

    plot_map_panel(axes[0], lake_preview, extent)
    plot_continuous_panel(
        axes[1],
        profile_df,
        trend_df,
        bottom_metric="d_u",
        bottom_label=r"$d_u$ (mm/yr)",
        bottom_color=DU_COLOR,
        top_metric="grad_mag_km",
        top_label=r"$d_u$ gradient (mm/yr/km)",
        top_color=GRAD_COLOR,
        title="Continuous lake-influence relation",
        panel_label="B",
        force_zero_to_one=False,
        bottom_shade_scale=0.10,
        top_shade_scale=0.05,
        background_gradient=True,
    )
    plot_continuous_panel(
        axes[2],
        profile_df,
        trend_df,
        bottom_metric="gradient_hotspot_susceptibility",
        bottom_label="Gradient-hotspot susceptibility",
        bottom_color=GRAD_SUSC_COLOR,
        top_metric="du_hotspot_susceptibility",
        top_label=r"$d_u$-hotspot susceptibility",
        top_color=DUHOT_SUSC_COLOR,
        title="Continuous hotspot-susceptibility relation",
        panel_label="C",
        force_zero_to_one=True,
    )

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def make_subplot_b_figure(
    profile_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.1), constrained_layout=True)
    plot_continuous_panel(
        ax,
        profile_df,
        trend_df,
        bottom_metric="d_u",
        bottom_label=r"$d_u$ (mm/yr)",
        bottom_color=DU_COLOR,
        top_metric="grad_mag_km",
        top_label=r"$d_u$ gradient (mm/yr/km)",
        top_color=GRAD_COLOR,
        title=None,
        panel_label=None,
        force_zero_to_one=False,
        bottom_shade_scale=0.10,
        top_shade_scale=0.05,
        background_gradient=True,
    )
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PF negative-d_u continuous lake-influence analysis with hotspot-susceptibility overlays."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-pf-negative", type=int, default=140_000)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--window-frac", type=float, default=0.05)
    parser.add_argument("--n-profile-points", type=int, default=160)
    parser.add_argument("--force-rebuild-sample", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    cache_dir = out_dir / "cache"
    for path in (fig_dir, table_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing derived pixel table: {csv_path}")

    lake_cfg = load_lake_influence_config(base_dir, out_dir, csv_path)
    grad_ref = load_gradient_hotspot_reference(base_dir, out_dir, csv_path)
    du_ref = load_du_hotspot_reference(base_dir, out_dir, csv_path)

    sample_cache = cache_dir / f"figure7_2_pf_negative_du_sample_n{args.sample_pf_negative}.csv.gz"
    required_cols = {
        "easting",
        "northing",
        "d_u",
        "grad_mag",
        "grad_mag_km",
        "abs_d_u",
        "lake_influence_norm01",
        "gradient_hotspot_susceptibility",
        "du_hotspot_susceptibility",
    }
    if sample_cache.exists() and not args.force_rebuild_sample:
        log_step(f"loading cached sample: {sample_cache}")
        sample_df = pd.read_csv(sample_cache)
        sample_df = engineer_features(sample_df)
        if not required_cols.issubset(sample_df.columns):
            raise RuntimeError(
                f"Cached sample {sample_cache} is missing required columns. "
                "Re-run with --force-rebuild-sample."
            )
    else:
        usecols = [
            "longitude",
            "latitude",
            "easting",
            "northing",
            "Main-Basin",
            "Perma_Distr_map",
            "d_u",
            "dudx",
            "dudy",
            "grad_mag",
            "grad_mag_km",
        ]
        sample_df = build_pf_negative_du_sample(
            csv_path,
            usecols=usecols,
            desired_n=args.sample_pf_negative,
            chunksize=args.chunksize,
            seed=SEED,
        )
        sample_df = attach_lake_and_susceptibility(
            sample_df,
            lake_cfg=lake_cfg,
            grad_ref=grad_ref,
            du_ref=du_ref,
        )
        sample_df.to_csv(sample_cache, index=False, compression="gzip")
        log_step(f"saved sample cache: {sample_cache}")

    sample_df = sample_df.loc[
        np.isfinite(sample_df["lake_influence_norm01"])
        & np.isfinite(sample_df["abs_d_u"])
        & np.isfinite(sample_df["grad_mag_km"])
        & np.isfinite(sample_df["gradient_hotspot_susceptibility"])
        & np.isfinite(sample_df["du_hotspot_susceptibility"])
        & (sample_df["d_u"] < 0.0)
        & sample_df["domain"].eq("pf")
    ].copy()
    if sample_df.empty:
        raise RuntimeError("No PF negative-d_u rows remain after attaching lake influence and susceptibility rasters.")

    metrics = [
        "d_u",
        "grad_mag_km",
        "gradient_hotspot_susceptibility",
        "du_hotspot_susceptibility",
    ]
    profile_df = build_smooth_profile(
        sample_df,
        x_col="lake_influence_norm01",
        metrics=metrics,
        window_frac=float(args.window_frac),
        n_profile_points=int(args.n_profile_points),
    )
    profile_df = extend_profile_to_unit_interval(profile_df, metric_names=metrics)
    trend_df = build_trend_table(sample_df, profile_df)

    profile_path = table_dir / "figure7_2_pf_negative_du_continuous_profile.csv"
    trend_path = table_dir / "figure7_2_pf_negative_du_continuous_trend_tests.csv"
    meta_path = cache_dir / "figure7_2_meta.json"
    profile_df.to_csv(profile_path, index=False)
    trend_df.to_csv(trend_path, index=False)

    fig_png = fig_dir / f"{FIG_BASENAME}.png"
    fig_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    make_figure(profile_df, trend_df, lake_cfg, fig_png, fig_pdf)

    panelb_png = fig_dir / f"{SUBPLOT_B_FIG_BASENAME}.png"
    panelb_pdf = fig_dir / f"{SUBPLOT_B_FIG_BASENAME}.pdf"
    make_subplot_b_figure(profile_df, trend_df, panelb_png, panelb_pdf)

    meta_path.write_text(json.dumps({
        "focus": "PF and d_u<0 only",
        "sample_pf_negative_du": int(len(sample_df)),
        "window_frac": float(args.window_frac),
        "n_profile_points": int(args.n_profile_points),
        "lake_influence_meta_path": str(lake_cfg["meta_path"]),
        "gradient_hotspot_meta_path": str(grad_ref["meta_path"]),
        "du_hotspot_meta_path": str(du_ref["meta_path"]),
        "figure_png": str(fig_png),
        "figure_pdf": str(fig_pdf),
        "subplot_b_png": str(panelb_png),
        "subplot_b_pdf": str(panelb_pdf),
        "profile_table": str(profile_path),
        "trend_table": str(trend_path),
    }, indent=2))

    print(f"Saved sample cache:  {sample_cache}")
    print(f"Saved figure PNG:    {fig_png}")
    print(f"Saved figure PDF:    {fig_pdf}")
    print(f"Saved subplot B PNG: {panelb_png}")
    print(f"Saved subplot B PDF: {panelb_pdf}")
    print(f"Saved profile table: {profile_path}")
    print(f"Saved trend table:   {trend_path}")
    print(f"Saved metadata:      {meta_path}")


if __name__ == "__main__":
    main()
