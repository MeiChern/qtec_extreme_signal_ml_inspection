#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/figure5_2_du_attribution_by_basin.py
# Renamed package path: code/analysis_support/basin_feature_attribution_helpers.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42
CHUNKSIZE = 400_000
FIG_BASENAME = "Figure5_2_du_attribution_by_basin"

TARGET_BASINS = ["Yangtze", "Salween", "Inner Plateau", "Qaidam", "Brahmaputra"]

FEATURES = [
    "longitude",
    "latitude",
    "magt",
    "precipitation_mean",
    "temperature_mean",
    "bulk_density",
    "cf",
    "soc",
    "soil_thickness",
    "vwc35",
    "dem",
    "difpr",
    "dirpr",
    "slope",
    "twi",
    "gpp_mean",
    "ndvi",
]

FEATURE_LABELS = {
    "longitude": "Longitude",
    "latitude": "Latitude",
    "magt": "MAGT",
    "precipitation_mean": "Precipitation",
    "temperature_mean": "MAAT",
    "bulk_density": "Bulk density",
    "cf": "Coarse fragments",
    "soc": "SOC",
    "soil_thickness": "Soil thickness",
    "vwc35": "VWC35",
    "dem": "DEM",
    "difpr": "Diffuse rad.",
    "dirpr": "Direct rad.",
    "slope": "Slope",
    "twi": "TWI",
    "gpp_mean": "GPP",
    "ndvi": "NDVI",
}

PDP_FEATURES = [
    "magt",
    "precipitation_mean",
    "twi",
    "slope",
    "bulk_density",
    "cf",
    "latitude",
    "longitude",
]

MODEL_COLORS = {
    "ALL-Basins": "#222222",
    "Yangtze": "#1f77b4",
    "Salween": "#d62728",
    "Inner Plateau": "#2ca02c",
    "Qaidam": "#ff7f0e",
    "Brahmaputra": "#9467bd",
}

SCATTER_LIMITS = (-15.5, 5.5)
BASIN_DENSITY_CMAP = "Reds"
ALL_BASIN_DENSITY_CMAP = "YlOrRd"

PDP_UNITS = {
    "magt": "°C",
    "precipitation_mean": "mm/yr",
    "twi": "-",
    "cf": "%",
    "slope": "degree",
    "bulk_density": "kg/m³",
    "longitude": "°E",
    "latitude": "°N",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Perma_Distr_map" in out.columns:
        out["Perma_Distr_map"] = pd.to_numeric(out["Perma_Distr_map"], errors="coerce")
    if "d_u" in out.columns:
        out["d_u"] = pd.to_numeric(out["d_u"], errors="coerce")
    if "Main-Basin" in out.columns:
        out["Main-Basin"] = out["Main-Basin"].astype("string").fillna("missing")
    for c in FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["domain"] = np.where(
        out["Perma_Distr_map"] == 1,
        "pf",
        np.where(out["Perma_Distr_map"] == 0, "npf", "other"),
    )
    return out


def robust_clip(arr: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def summarize_pf_count_by_basin(csv_path: Path, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    rows = []
    usecols = ["Perma_Distr_map", "Main-Basin", "d_u"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = engineer_features(chunk)
        sub = chunk.loc[(chunk["domain"] == "pf") & np.isfinite(chunk["d_u"])].copy()
        if sub.empty:
            continue
        tmp = (
            sub["Main-Basin"]
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("Main-Basin")
            .reset_index(name="n_rows")
        )
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby("Main-Basin", as_index=False)["n_rows"].sum()
    return out.sort_values("n_rows", ascending=False).reset_index(drop=True)


def build_pf_sample(
    csv_path: Path,
    usecols: list[str],
    sample_total: int,
    chunksize: int = CHUNKSIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    counts = summarize_pf_count_by_basin(csv_path, chunksize=chunksize)
    total_pf = int(counts["n_rows"].sum())
    if total_pf == 0:
        raise RuntimeError("No permafrost rows found.")

    p = min(1.0, 1.15 * sample_total / total_pf)
    rng = np.random.default_rng(seed)

    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = engineer_features(chunk)
        chunk = chunk.loc[(chunk["domain"] == "pf") & np.isfinite(chunk["d_u"])].copy()
        if chunk.empty:
            continue
        draw = rng.random(len(chunk)) < p
        sub = chunk.loc[draw].copy()
        if not sub.empty:
            parts.append(sub)

    if not parts:
        raise RuntimeError("Permafrost sampling returned no rows.")

    df = pd.concat(parts, ignore_index=True)
    if len(df) > sample_total:
        df = df.sample(sample_total, random_state=seed)
    return df.reset_index(drop=True)


def resolve_sample(
    csv_path: Path,
    cache_path: Path,
    sample_total: int,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    if cache_path.exists():
        required_cols = {"Perma_Distr_map", "Main-Basin", "d_u", *FEATURES}
        cache_cols = set(pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist())
        if required_cols.issubset(cache_cols):
            df = pd.read_csv(cache_path)
            return engineer_features(df)

    usecols = ["Perma_Distr_map", "Main-Basin", "d_u"] + FEATURES
    df = build_pf_sample(
        csv_path=csv_path,
        usecols=usecols,
        sample_total=sample_total,
        chunksize=chunksize,
        seed=SEED,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, compression="gzip")
    return df


def fit_model(df: pd.DataFrame, model_name: str) -> dict:
    cols = FEATURES + ["d_u"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).copy()

    keep_mask = np.isfinite(sub["d_u"].to_numpy(dtype=float))
    sub = sub.loc[keep_mask].copy()
    if len(sub) < 300:
        raise RuntimeError(f"Not enough rows to fit model: {model_name}")

    X = sub[FEATURES].copy()
    y = sub["d_u"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    model = ExtraTreesRegressor(
        n_estimators=800,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features=0.8,
    )
    model.fit(X_train_imp, y_train)
    pred = model.predict(X_test_imp)

    result = {
        "name": model_name,
        "model": model,
        "imputer": imputer,
        "X_train_df": X_train.reset_index(drop=True),
        "X_test_df": X_test.reset_index(drop=True),
        "y_test": y_test,
        "pred_test": pred,
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "feature_importance": pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False),
    }
    return result


def compute_manual_pdp(
    model: ExtraTreesRegressor,
    imputer: SimpleImputer,
    X_ref_df: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
    max_rows: int = 4000,
) -> pd.DataFrame:
    X_ref = X_ref_df.copy().reset_index(drop=True)
    if len(X_ref) > max_rows:
        X_ref = X_ref.sample(max_rows, random_state=SEED).reset_index(drop=True)

    vals = []
    ylo_vals = []
    yhi_vals = []

    for g in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature] = g
        X_imp = imputer.transform(X_tmp)
        pred = model.predict(X_imp)
        y_mean = float(np.mean(pred))
        y_std = float(np.std(pred))
        band = 0.1 * y_std
        vals.append(y_mean)
        ylo_vals.append(y_mean - band)
        yhi_vals.append(y_mean + band)

    return pd.DataFrame(
        {
            "x": grid,
            "y": np.asarray(vals, dtype=float),
            "ylo": np.asarray(ylo_vals, dtype=float),
            "yhi": np.asarray(yhi_vals, dtype=float),
        }
    )


def style_open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", alpha=0.12),
    )


def lighten_color(color: str, blend: float = 0.72) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - blend) * rgb + blend)


def gaussian_kernel1d(sigma: float = 1.3) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_hist2d(hist: np.ndarray, sigma: float = 1.3) -> np.ndarray:
    kernel = gaussian_kernel1d(sigma=sigma)
    smoothed = np.apply_along_axis(
        lambda arr: np.convolve(arr, kernel, mode="same"),
        axis=0,
        arr=hist,
    )
    smoothed = np.apply_along_axis(
        lambda arr: np.convolve(arr, kernel, mode="same"),
        axis=1,
        arr=smoothed,
    )
    return smoothed


def estimate_relative_density(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    hi: float,
    bins: int = 150,
    sigma: float = 1.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= lo)
        & (x <= hi)
        & (y >= lo)
        & (y <= hi)
    )
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size == 0:
        return x, y, np.empty(0, dtype=float)

    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[lo, hi], [lo, hi]],
    )
    density = smooth_hist2d(hist, sigma=sigma)

    ix = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(x_edges) - 2)
    iy = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2)
    point_density = density[ix, iy]

    order = np.argsort(point_density)
    return x[order], y[order], point_density[order]


def plot_obs_pred(
    ax,
    result: dict,
    scatter_cloud: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    scatter_limits: tuple[float, float] = SCATTER_LIMITS,
    density_vmax: float | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    cmap: str = BASIN_DENSITY_CMAP,
):
    obs = np.asarray(result["y_test"], dtype=float)
    pred = np.asarray(result["pred_test"], dtype=float)

    lo, hi = scatter_limits
    if scatter_cloud is None:
        x_sc, y_sc, density_sc = estimate_relative_density(obs, pred, lo=lo, hi=hi)
    else:
        x_sc, y_sc, density_sc = scatter_cloud
    sc = ax.scatter(
        x_sc,
        y_sc,
        c=density_sc,
        cmap=cmap,
        vmin=0.0,
        vmax=density_vmax if density_vmax and density_vmax > 0 else 1.0,
        s=9,
        alpha=0.9,
        linewidths=0,
        rasterized=len(x_sc) > 3000,
    )

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.35", linewidth=1.0)

    fit_mask = (
        np.isfinite(obs)
        & np.isfinite(pred)
        & (obs >= lo)
        & (obs <= hi)
        & (pred >= lo)
        & (pred <= hi)
    )
    if fit_mask.sum() >= 10:
        a, b = np.polyfit(obs[fit_mask], pred[fit_mask], 1)
        xx = np.linspace(lo, hi, 200)
        ax.plot(xx, a * xx + b, linestyle="--", color="#ff7f0e", linewidth=1.2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if show_xlabel:
        ax.set_xlabel("Observed $d_u$ (mm/yr)", fontweight="bold")
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Predicted $d_u$ (mm/yr)", fontweight="bold")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(
        f"{result['name']}\nPredicted vs observed",
        fontweight="bold",
        pad=4,
    )
    ax.text(
        0.97,
        0.04,
        f"$R^2$={result['r2']:.2f}\nMAE={result['mae']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.18", alpha=0.12),
    )
    style_open_axes(ax)
    return sc


def plot_top_importance(ax, result: dict, top_n: int = 10, face_color: str = "#4c78a8") -> None:
    imp = result["feature_importance"].head(top_n).iloc[::-1]
    labels = [FEATURE_LABELS.get(k, k) for k in imp.index]
    y_pos = np.arange(len(imp), dtype=float)
    ax.barh(y_pos, imp.values, color=face_color, alpha=0.95)
    ax.set_title(f"Top {len(imp)} features", fontweight="bold", pad=4)
    x_max = float(np.nanmax(imp.values)) if len(imp) else 1.0
    ax.set_xlim(0.0, max(0.13, 1.08 * x_max))
    ax.set_xlabel("Feature importance", fontweight="bold")
    ax.xaxis.set_label_coords(1.0, -0.09)
    ax.xaxis.label.set_horizontalalignment("right")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=9)
    label_x = ax.get_xlim()[1] - 0.01 * ax.get_xlim()[1]
    for y, label in zip(y_pos, labels):
        ax.text(
            label_x,
            y,
            label,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="0.12",
        )
    style_open_axes(ax)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")


def compute_common_pdp_ylim(
    pdp_curves: dict[str, dict[str, pd.DataFrame]],
    features: list[str],
    model_names: list[str],
) -> tuple[float, float]:
    y_min = np.inf
    y_max = -np.inf

    for feat in features:
        for name in model_names:
            curve = pdp_curves[feat][name]
            vals = curve[["ylo", "yhi"]].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            y_min = min(y_min, float(np.nanmin(vals)))
            y_max = max(y_max, float(np.nanmax(vals)))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return (-1.0, 1.0)

    pad = 0.06 * (y_max - y_min)
    return (y_min - pad, y_max + pad)


def precipitation_tick_formatter(x: float, _pos: int) -> str:
    return f"{x / 10.0:g}"


def slope_tick_formatter(x: float, _pos: int) -> str:
    return f"{x * 10.0:g}"


def build_summary_figure(
    fig_dir: Path,
    results: dict[str, dict],
    model_names: list[str],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(20, 18), constrained_layout=False)
    outer = GridSpec(
        3,
        2,
        figure=fig,
        left=0.045,
        right=0.985,
        top=0.96,
        bottom=0.07,
        wspace=0.18,
        hspace=0.20,
    )

    scatter_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    density_vmax = 0.0
    for name in model_names:
        cloud = estimate_relative_density(
            np.asarray(results[name]["y_test"], dtype=float),
            np.asarray(results[name]["pred_test"], dtype=float),
            lo=SCATTER_LIMITS[0],
            hi=SCATTER_LIMITS[1],
        )
        scatter_data[name] = cloud
        if cloud[2].size:
            density_vmax = max(density_vmax, float(np.nanmax(cloud[2])))

    for idx, name in enumerate(model_names):
        row, col = divmod(idx, 2)
        block = outer[row, col].subgridspec(1, 2, width_ratios=[4.4, 1.4], wspace=0.14)
        ax_sc = fig.add_subplot(block[0, 0])
        ax_imp = fig.add_subplot(block[0, 1])

        plot_obs_pred(
            ax_sc,
            results[name],
            scatter_cloud=scatter_data[name],
            scatter_limits=SCATTER_LIMITS,
            density_vmax=density_vmax,
            show_ylabel=(col == 0),
            show_xlabel=(row == 2),
            cmap=ALL_BASIN_DENSITY_CMAP if name == "ALL-Basins" else BASIN_DENSITY_CMAP,
        )
        plot_top_importance(
            ax_imp,
            results[name],
            top_n=10,
            face_color=lighten_color(MODEL_COLORS.get(name, "#4c78a8")),
        )
        add_panel_label(ax_sc, list("ABCDEF")[idx])

    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def build_pdp_figure(
    fig_dir: Path,
    pdp_curves: dict[str, dict[str, pd.DataFrame]],
    model_names: list[str],
    pdp_ylim: tuple[float, float],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(15, 18), constrained_layout=False)
    outer = GridSpec(
        4,
        2,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.93,
        bottom=0.16,
        wspace=0.15,
        hspace=0.22,
    )

    pdp_titles = {
        "magt": "MAGT",
        "precipitation_mean": "Precipitation",
        "twi": "TWI",
        "cf": "Coarse fragments",
        "slope": "Slope",
        "bulk_density": "Bulk density",
        "longitude": "Longitude",
        "latitude": "Latitude",
    }

    for idx, feat in enumerate(PDP_FEATURES):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(outer[row, col])

        for name in model_names:
            curve = pdp_curves[feat][name]
            color = MODEL_COLORS.get(name, None)
            ax.fill_between(
                curve["x"],
                curve["ylo"],
                curve["yhi"],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(curve["x"], curve["y"], linewidth=1.8, color=color, label=name, zorder=3)

        unit_label = PDP_UNITS.get(feat)
        title = pdp_titles[feat]
        if unit_label and unit_label != "-":
            title = f"{title} ({unit_label})"
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(*pdp_ylim)
        if feat == "precipitation_mean":
            ax.xaxis.set_major_formatter(FuncFormatter(precipitation_tick_formatter))
        if feat == "slope":
            ax.xaxis.set_major_formatter(FuncFormatter(slope_tick_formatter))
        if col == 0:
            ax.set_ylabel("Partial dependence of $d_u$ (mm/yr)", fontweight="bold")
        else:
            ax.tick_params(labelleft=False)
        style_open_axes(ax)
        add_panel_label(ax, list("ABCDEFGH")[idx])

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[name], linewidth=2.2, label=name)
        for name in model_names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(model_names),
        frameon=True,
        fontsize=10,
        borderpad=0.45,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
        facecolor="white",
        edgecolor="0.75",
        framealpha=1.0,
        fancybox=False,
        bbox_to_anchor=(0.5, 0.03),
    )

    out_png = fig_dir / f"{FIG_BASENAME}_pdp.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}_pdp.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 5.2: d_u attribution by basin")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--sample-total", type=int, default=220_000)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        csv_path = base_dir / "df_all_data_with_wright_du.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Could not find df_all_data_with_wright_du_plus_grad.csv or df_all_data_with_wright_du.csv")

    sample_cache = cache_dir / "figure5_2_pf_basin_sample.csv.gz"
    df = resolve_sample(
        csv_path=csv_path,
        cache_path=sample_cache,
        sample_total=args.sample_total,
        chunksize=args.chunksize,
    )

    basin_counts = (
        df["Main-Basin"]
        .astype(str)
        .value_counts()
        .rename_axis("Main-Basin")
        .reset_index(name="n_rows")
    )
    basin_counts.to_csv(table_dir / "figure5_2_basin_counts.csv", index=False)

    available_basins = set(df["Main-Basin"].astype(str).unique().tolist())
    chosen_basins = [b for b in TARGET_BASINS if b in available_basins]
    if len(chosen_basins) < len(TARGET_BASINS):
        fallback = [
            b
            for b in (
            df["Main-Basin"]
            .astype(str)
            .value_counts()
            .loc[lambda s: s.index != "missing"]
            .index.tolist()
            )
            if b not in chosen_basins
        ]
        chosen_basins.extend(
            fallback[: len(TARGET_BASINS) - len(chosen_basins)]
        )

    model_defs = [("ALL-Basins", df.copy())]
    for basin in chosen_basins:
        sub = df.loc[df["Main-Basin"].astype(str) == basin].copy()
        model_defs.append((basin, sub))

    results = {}
    for name, sub in model_defs:
        results[name] = fit_model(sub, name)

    # save feature importance tables
    for name, res in results.items():
        out = res["feature_importance"].rename("importance").reset_index()
        out.columns = ["feature", "importance"]
        out.to_csv(table_dir / f"figure5_2_importance_{name.replace(' ', '_')}.csv", index=False)

    # PDP grids based on all-basin data
    pdp_curves = {}
    for feat in PDP_FEATURES:
        vals = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [5, 95])
        grid = np.linspace(lo, hi, 40)

        pdp_curves[feat] = {}
        for name, res in results.items():
            pdp_curves[feat][name] = compute_manual_pdp(
                model=res["model"],
                imputer=res["imputer"],
                X_ref_df=res["X_train_df"],
                feature=feat,
                grid=grid,
            )

    model_names = ["ALL-Basins"] + chosen_basins

    pdp_ylim = compute_common_pdp_ylim(
        pdp_curves=pdp_curves,
        features=PDP_FEATURES,
        model_names=model_names,
    )
    summary_png, summary_pdf = build_summary_figure(
        fig_dir=fig_dir,
        results=results,
        model_names=model_names,
    )
    pdp_png, pdp_pdf = build_pdp_figure(
        fig_dir=fig_dir,
        pdp_curves=pdp_curves,
        model_names=model_names,
        pdp_ylim=pdp_ylim,
    )

    meta = {
        "chosen_basins": chosen_basins,
        "n_all_basins": int(len(df)),
        "n_by_basin": {
            name: int(len(sub))
            for name, sub in model_defs
        },
        "features": FEATURES,
        "pdp_features": PDP_FEATURES,
        "summary_figure": str(summary_png),
        "pdp_figure": str(pdp_png),
    }
    (cache_dir / "figure5_2_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved summary PNG: {summary_png}")
    print(f"Saved summary PDF: {summary_pdf}")
    print(f"Saved PDP PNG: {pdp_png}")
    print(f"Saved PDP PDF: {pdp_pdf}")
    print(f"Chosen basins: {chosen_basins}")


if __name__ == "__main__":
    main()
