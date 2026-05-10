#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/figure6_0_transition_metric_review.py
# Renamed package path: code/analysis_support/transition_metric_raster_utils.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

CHUNKSIZE = 400_000
FIG_BASENAME = "Figure6_0_transition_metric_review"

# Transition variables used in the sharpness / susceptibility setup
TRANSITION_VARS = [
    "soil_thickness",
    "cf",
    "soc",
    "bulk_density",
    "dem",
    "slope",
    "twi",
    "vwc35",
    "ndvi",
    "magt",
]

_VAR_META: dict[str, tuple[str, str]] = {
    "soil_thickness": ("Soil Thickness", "m"),
    "cf": ("Coarse Fragments", "vol.%"),
    "soc": ("Soil Organic Carbon", "g/kg"),
    "bulk_density": ("Bulk Density", "kg/m³"),
    "dem": ("Elevation (DEM)", "m a.s.l."),
    "slope": ("Slope", "°"),
    "twi": ("Topographic Wetness Index", "–"),
    "vwc35": ("Vol. Water Content (35 cm)", "m³/m³"),
    "ndvi": ("NDVI", "–"),
    "magt": ("MAGT", "°C"),
}


def robust_limits(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def open_memmap(path: Path, dtype: str, mode: str, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def en_to_rc(
    easting: np.ndarray,
    northing: np.ndarray,
    *,
    res: float,
    gx0: int,
    gy1: int,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.rint(np.asarray(easting, dtype=np.float64) / res).astype(np.int64)
    gy = np.rint(np.asarray(northing, dtype=np.float64) / res).astype(np.int64)
    col = (gx - gx0).astype(np.int32)
    row = (gy1 - gy).astype(np.int32)
    return row, col


def get_extent(min_e: float, max_n: float, nrows: int, ncols: int, res: float) -> list[float]:
    left = min_e
    right = min_e + ncols * res
    bottom = max_n - nrows * res
    top = max_n
    return [left, right, bottom, top]


def choose_stride(nrows: int, ncols: int, target_max: int = 900) -> int:
    return max(1, int(max(nrows, ncols) / target_max))


def infer_grid_from_csv(csv_path: Path, chunksize: int = CHUNKSIZE) -> dict:
    e_min = np.inf
    e_max = -np.inf
    n_min = np.inf
    n_max = -np.inf
    e_vals = []
    n_vals = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=["easting", "northing"],
        chunksize=chunksize,
        low_memory=False,
    ):
        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy()
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy()
        ok = np.isfinite(e) & np.isfinite(n)
        if not ok.any():
            continue

        e_ok = e[ok]
        n_ok = n[ok]
        e_min = min(e_min, float(e_ok.min()))
        e_max = max(e_max, float(e_ok.max()))
        n_min = min(n_min, float(n_ok.min()))
        n_max = max(n_max, float(n_ok.max()))

        if len(e_vals) < 200000:
            take = min(50000, len(e_ok))
            e_vals.append(np.sort(np.unique(e_ok[:take])))
            n_vals.append(np.sort(np.unique(n_ok[:take])))

    if not np.isfinite([e_min, e_max, n_min, n_max]).all():
        raise RuntimeError("Could not infer grid bounds from CSV.")

    def min_step(arr_list: list[np.ndarray]) -> float:
        diffs = []
        for arr in arr_list:
            if len(arr) > 1:
                d = np.diff(arr)
                d = d[d > 0]
                if len(d) > 0:
                    diffs.append(d.min())
        if not diffs:
            return 30.0
        return float(np.min(diffs))

    res_e = min_step(e_vals)
    res_n = min_step(n_vals)
    res = float(min(res_e, res_n))

    gx0 = int(np.floor(e_min / res))
    gx1 = int(np.ceil(e_max / res))
    gy0 = int(np.floor(n_min / res))
    gy1 = int(np.ceil(n_max / res))
    ncols = gx1 - gx0 + 1
    nrows = gy1 - gy0 + 1

    return {
        "res": res,
        "gx0": gx0,
        "gy1": gy1,
        "nrows": nrows,
        "ncols": ncols,
        "min_e": e_min,
        "max_n": n_max,
    }


def build_or_load_grid(base_dir: Path, csv_path: Path) -> dict:
    meta_path = base_dir / "outputs" / "grad_rasters" / "grid_meta.npz"
    if meta_path.exists():
        meta = np.load(meta_path)
        return {
            "res": float(meta["res"]),
            "gx0": int(meta["gx0"]),
            "gy1": int(meta["gy1"]),
            "nrows": int(meta["nrows"]),
            "ncols": int(meta["ncols"]),
            "min_e": float(meta["min_e"]),
            "max_n": float(meta["max_n"]),
        }
    return infer_grid_from_csv(csv_path)


def build_raw_raster_cache(
    csv_path: Path,
    out_dir: Path,
    vars_to_use: list[str],
    grid: dict,
    chunksize: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    res = float(grid["res"])
    gx0 = int(grid["gx0"])
    gy1 = int(grid["gy1"])

    sum_paths = {}
    cnt_paths = {}
    sum_maps = {}
    cnt_maps = {}

    for var in vars_to_use:
        sum_path = out_dir / f"{var}_sum_f32.memmap"
        cnt_path = out_dir / f"{var}_cnt_u16.memmap"
        sum_paths[var] = sum_path
        cnt_paths[var] = cnt_path

        sum_mm = open_memmap(sum_path, dtype="float32", mode="w+", shape=(nrows, ncols))
        cnt_mm = open_memmap(cnt_path, dtype="uint16", mode="w+", shape=(nrows, ncols))
        sum_mm[:] = 0.0
        cnt_mm[:] = 0
        sum_maps[var] = sum_mm
        cnt_maps[var] = cnt_mm

    usecols = ["easting", "northing"] + vars_to_use
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        e = pd.to_numeric(chunk["easting"], errors="coerce").to_numpy()
        n = pd.to_numeric(chunk["northing"], errors="coerce").to_numpy()
        row, col = en_to_rc(e, n, res=res, gx0=gx0, gy1=gy1)
        base_ok = (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols)

        for var in vars_to_use:
            vals = pd.to_numeric(chunk[var], errors="coerce").to_numpy(dtype=np.float32)
            ok = base_ok & np.isfinite(vals)
            if not ok.any():
                continue
            rr = row[ok]
            cc = col[ok]
            vv = vals[ok]
            np.add.at(sum_maps[var], (rr, cc), vv)
            np.add.at(cnt_maps[var], (rr, cc), 1)

    for var in vars_to_use:
        sum_maps[var].flush()
        cnt_maps[var].flush()

    return {"sum": sum_paths, "cnt": cnt_paths}


def finalize_mean_rasters(
    cache_paths: dict[str, dict[str, Path]],
    out_dir: Path,
    vars_to_use: list[str],
    grid: dict,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])

    mean_paths = {}
    for var in vars_to_use:
        sum_mm = open_memmap(cache_paths["sum"][var], dtype="float32", mode="r", shape=(nrows, ncols))
        cnt_mm = open_memmap(cache_paths["cnt"][var], dtype="uint16", mode="r", shape=(nrows, ncols))
        mean_path = out_dir / f"{var}_mean_f32.memmap"
        mean_mm = open_memmap(mean_path, dtype="float32", mode="w+", shape=(nrows, ncols))
        mean_mm[:] = np.nan

        mask = cnt_mm > 0
        mean_mm[mask] = sum_mm[mask] / cnt_mm[mask]
        mean_mm.flush()
        mean_paths[var] = mean_path

    return mean_paths


def nan_local_std_and_mean(arr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns local std, local mean, and valid-count arrays using nan-aware moving windows.
    """
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr).astype(np.float32)
    arr0 = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

    area = float(size * size)

    mean_valid = uniform_filter(valid, size=size, mode="nearest")
    mean_sum = uniform_filter(arr0, size=size, mode="nearest")
    mean_sq = uniform_filter(arr0 * arr0, size=size, mode="nearest")

    count = mean_valid * area
    mean = np.full_like(arr0, np.nan, dtype=np.float32)
    var = np.full_like(arr0, np.nan, dtype=np.float32)

    ok = count >= 3
    mean[ok] = mean_sum[ok] / np.maximum(mean_valid[ok], 1e-6)
    ex2 = mean_sq[ok] / np.maximum(mean_valid[ok], 1e-6)
    var[ok] = np.maximum(ex2 - mean[ok] * mean[ok], 0.0)

    lstd = np.full_like(arr0, np.nan, dtype=np.float32)
    lstd[ok] = np.sqrt(var[ok])

    return lstd, mean, count


def derive_transition_metric_rasters(
    mean_paths: dict[str, Path],
    out_dir: Path,
    vars_to_use: list[str],
    grid: dict,
    window_size: int,
) -> dict[str, dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    res = float(grid["res"])

    lstd_paths = {}
    gmag_paths = {}

    for var in vars_to_use:
        lstd_path = out_dir / f"{var}__lstd_f32.memmap"
        gmag_path = out_dir / f"{var}__gmag_f32.memmap"
        lstd_paths[var] = lstd_path
        gmag_paths[var] = gmag_path

        if lstd_path.exists() and gmag_path.exists():
            continue

        arr_mm = open_memmap(mean_paths[var], dtype="float32", mode="r", shape=(nrows, ncols))
        arr = np.array(arr_mm, copy=False).astype(np.float32)

        lstd, local_mean, count = nan_local_std_and_mean(arr, size=window_size)

        # use local_mean as the smoothed surface for gradient estimation
        smooth = np.where(np.isfinite(local_mean), local_mean, 0.0).astype(np.float32)
        gy, gx = np.gradient(smooth, res, res)
        gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

        # mask out places with insufficient neighborhood support
        ok = count >= 3
        gmag = np.where(ok, gmag, np.nan).astype(np.float32)

        lstd_mm = open_memmap(lstd_path, dtype="float32", mode="w+", shape=(nrows, ncols))
        gmag_mm = open_memmap(gmag_path, dtype="float32", mode="w+", shape=(nrows, ncols))
        lstd_mm[:] = lstd
        gmag_mm[:] = gmag
        lstd_mm.flush()
        gmag_mm.flush()

    return {"lstd": lstd_paths, "gmag": gmag_paths}


def make_figure(
    metric_paths: dict[str, dict[str, Path]],
    grid: dict,
    vars_to_use: list[str],
    out_png: Path,
    out_pdf: Path,
) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    nrows_g = int(grid["nrows"])
    ncols_g = int(grid["ncols"])
    res = float(grid["res"])
    min_e = float(grid["min_e"])
    max_n = float(grid["max_n"])
    extent = get_extent(min_e=min_e, max_n=max_n, nrows=nrows_g, ncols=ncols_g, res=res)
    stride = choose_stride(nrows_g, ncols_g, target_max=900)

    first_half = vars_to_use[:5]
    second_half = vars_to_use[5:]

    panel_plan = [
        [("lstd", v) for v in first_half],
        [("gmag", v) for v in first_half],
        [("lstd", v) for v in second_half],
        [("gmag", v) for v in second_half],
    ]

    metric_cmaps = {
        "lstd": "YlOrBr",
        "gmag": "viridis",
    }
    metric_units = {
        "lstd": "local std",
        "gmag": "local gmag",
    }

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }):
        fig, axes = plt.subplots(
            4, 5,
            figsize=(24, 18),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.05,
            right=0.98,
            top=0.94,
            bottom=0.06,
            wspace=0.045,
            hspace=0.24,
        )

        for r in range(4):
            for c in range(5):
                ax = axes[r, c]
                metric_type, var = panel_plan[r][c]
                title, raw_unit = _VAR_META.get(var, (var, ""))

                mm = open_memmap(
                    metric_paths[metric_type][var],
                    dtype="float32",
                    mode="r",
                    shape=(nrows_g, ncols_g),
                )
                arr = np.array(mm[::stride, ::stride], copy=False).astype(np.float64)

                vmin, vmax = robust_limits(arr, 2, 98)

                im = ax.imshow(
                    arr,
                    extent=extent,
                    origin="upper",
                    cmap=metric_cmaps[metric_type],
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )

                ax.set_title(f"{title}\n{metric_units[metric_type]}", pad=4)

                if r == 3:
                    ax.set_xlabel("Easting (m)", labelpad=2)
                else:
                    ax.tick_params(labelbottom=False)

                if c == 0:
                    ax.set_ylabel("Northing (m)", labelpad=2)
                else:
                    ax.tick_params(labelleft=False)

                ax.tick_params(axis="both", which="both", length=2.5)

                cax = inset_axes(
                    ax,
                    width="38%",
                    height="3.5%",
                    loc="lower center",
                    bbox_to_anchor=(0.0, -0.12, 1.0, 1.0),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
                cbar.set_label(raw_unit, fontsize=6, labelpad=1)
                cbar.ax.tick_params(labelsize=6, length=2, pad=1)
                cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        fig.suptitle(
            "Figure 6.0. Transition-structure review: local standard deviation and local gradient magnitude",
            fontsize=14,
            fontweight="bold",
        )
        fig.savefig(out_png, bbox_inches="tight", dpi=300)
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 6.0 transition-metric review")
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    parser.add_argument("--window-size", type=int, default=5, help="Moving window size in raster pixels for local std")
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (base_dir / "outputs" / "deformation_rate_gradient_lake_paper")
    )

    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache" / "figure6_0_transition_review"
    raw_raster_dir = cache_dir / "raw_rasters"
    metric_raster_dir = cache_dir / "metric_rasters"

    for p in [fig_dir, cache_dir, raw_raster_dir, metric_raster_dir]:
        p.mkdir(parents=True, exist_ok=True)

    csv_path = base_dir / "df_all_data_with_wright_du_plus_grad.csv"
    if not csv_path.exists():
        csv_path = base_dir / "df_all_data_with_wright_du.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Could not find df_all_data_with_wright_du_plus_grad.csv or df_all_data_with_wright_du.csv")

    grid = build_or_load_grid(base_dir, csv_path)

    mean_paths = {}
    missing_means = []
    for var in TRANSITION_VARS:
        mean_path = raw_raster_dir / f"{var}_mean_f32.memmap"
        mean_paths[var] = mean_path
        if not mean_path.exists():
            missing_means.append(var)

    if missing_means:
        cache_paths = build_raw_raster_cache(
            csv_path=csv_path,
            out_dir=raw_raster_dir,
            vars_to_use=TRANSITION_VARS,
            grid=grid,
            chunksize=args.chunksize,
        )
        mean_paths = finalize_mean_rasters(
            cache_paths=cache_paths,
            out_dir=raw_raster_dir,
            vars_to_use=TRANSITION_VARS,
            grid=grid,
        )

    metric_paths = derive_transition_metric_rasters(
        mean_paths=mean_paths,
        out_dir=metric_raster_dir,
        vars_to_use=TRANSITION_VARS,
        grid=grid,
        window_size=args.window_size,
    )

    out_png = fig_dir / f"{FIG_BASENAME}.png"
    out_pdf = fig_dir / f"{FIG_BASENAME}.pdf"
    make_figure(metric_paths, grid, TRANSITION_VARS, out_png, out_pdf)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()