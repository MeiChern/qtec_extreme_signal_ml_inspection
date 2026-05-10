# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/submission_build_common.py
# Renamed package path: code/analysis_support/submission_build_common.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

_ENV_PREFIX = Path(sys.prefix)
_PROJ_DIR = _ENV_PREFIX / "share" / "proj"
_GDAL_DIR = _ENV_PREFIX / "share" / "gdal"
if _PROJ_DIR.exists():
    os.environ.setdefault("PROJ_LIB", str(_PROJ_DIR))
if _GDAL_DIR.exists():
    os.environ.setdefault("GDAL_DATA", str(_GDAL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath


ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT_DIR / "pnas_nexus_submission"
RESULTS_DIR = SUBMISSION_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"

SOURCE_DIR = ROOT_DIR / "outputs" / "deformation_rate_gradient_lake_paper"
SOURCE_CACHE_DIR = SOURCE_DIR / "cache"
SOURCE_TABLE_DIR = SOURCE_DIR / "tables"

GRID_META_PATH = ROOT_DIR / "outputs" / "grad_rasters" / "grid_meta.npz"
DU_RASTER_PATH = ROOT_DIR / "outputs" / "grad_rasters" / "du_f32.memmap"
GRAD_RASTER_PATH = ROOT_DIR / "outputs" / "grad_rasters" / "gradmag_f32.memmap"
ENV_RASTER_DIR = SOURCE_CACHE_DIR / "env_review" / "rasters"

for path in (str(ROOT_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from submission_figure_style import EXPORT_DPI, FONT, add_panel_label, apply_style


def ensure_style() -> None:
    apply_style()


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - float(blend)) * rgb + float(blend))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def grid_meta() -> dict[str, float | int]:
    meta = np.load(GRID_META_PATH)
    return {
        "res": float(meta["res"]),
        "nrows": int(meta["nrows"]),
        "ncols": int(meta["ncols"]),
        "gx0": int(meta["gx0"]),
        "gy1": int(meta["gy1"]),
        "min_e": float(meta["min_e"]),
        "max_n": float(meta["max_n"]),
    }


def raster_shape() -> tuple[int, int]:
    meta = grid_meta()
    return int(meta["nrows"]), int(meta["ncols"])


def open_memmap(path: Path, *, dtype: str = "float32") -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=raster_shape())


def load_du_raster() -> np.memmap:
    return open_memmap(DU_RASTER_PATH)


def load_grad_raster() -> np.memmap:
    return open_memmap(GRAD_RASTER_PATH)


def load_env_raster(var_name: str) -> np.memmap:
    return open_memmap(ENV_RASTER_DIR / f"{var_name}_mean_f32.memmap")


def robust_limits(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> tuple[float, float]:
    values = np.asarray(arr, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(values, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def symmetric_limits(arr: np.ndarray, p: float = 98.0) -> tuple[float, float]:
    values = np.asarray(arr, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    hi = float(np.percentile(np.abs(values), p))
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    return (-hi, hi)


def lonlat_extent() -> list[float]:
    candidates = [
        SOURCE_CACHE_DIR / "figure4_lonlat_extent.json",
        SOURCE_CACHE_DIR / "Figure_reorganize_du_du_gradient_ml_features_lonlat_extent.json",
    ]
    for path in candidates:
        if path.exists():
            payload = load_json(path)
            if "extent_lonlat" in payload:
                return [float(v) for v in payload["extent_lonlat"]]
    raise FileNotFoundError("Could not resolve the cached longitude/latitude extent.")


def lonlat_extent_box():
    from shapely.geometry import box

    xmin, xmax, ymin, ymax = lonlat_extent()
    return box(xmin, ymin, xmax, ymax)


def decimate_raster(arr: np.ndarray, *, target_max: int = 1400) -> tuple[np.ndarray, int]:
    stride = max(1, int(max(arr.shape) / int(target_max)))
    return np.asarray(arr[::stride, ::stride]), stride


def masked_raster(arr: np.ndarray, *, zero_is_nodata: bool = False) -> np.ma.MaskedArray:
    values = np.asarray(arr, dtype=float)
    mask = ~np.isfinite(values)
    if zero_is_nodata:
        mask |= np.isclose(values, 0.0)
    return np.ma.masked_array(values, mask=mask)


def _valid_hull_from_lonlat(lon: np.ndarray, lat: np.ndarray):
    from shapely.geometry import MultiPoint

    if lon.size < 3 or lat.size < 3:
        return lonlat_extent_box()
    hull = MultiPoint(np.column_stack([lon, lat])).convex_hull
    if hull.geom_type != "Polygon":
        hull = hull.buffer(0.02)
    return hull.buffer(0.08).intersection(lonlat_extent_box()).buffer(0)


def coverage_hull_lonlat_from_array(arr: np.ndarray, *, stride: int = 24):
    sample = np.asarray(arr[::stride, ::stride], dtype=float)
    valid = np.isfinite(sample)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return lonlat_extent_box()
    rows = rows.astype(np.int64) * stride
    cols = cols.astype(np.int64) * stride
    east, north = rc_to_en(rows, cols)
    to_lonlat = transformers()[0]
    lon, lat = to_lonlat.transform(east, north)
    return _valid_hull_from_lonlat(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


@lru_cache(maxsize=64)
def coverage_hull_lonlat_from_path(path: str, stride: int = 24):
    return coverage_hull_lonlat_from_array(open_memmap(Path(path)), stride=stride)


def clip_gdf_to_hull(gdf, hull):
    import geopandas as gpd

    if gdf.empty:
        return gdf.copy()
    clipper = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
    if gdf.crs != clipper.crs:
        clipper = clipper.to_crs(gdf.crs)
    return gpd.clip(gdf, clipper)


def plot_gdf_with_halo(
    ax,
    gdf,
    *,
    color: str,
    linewidth: float,
    halo_width: float,
    halo_color: str = "white",
    alpha: float = 1.0,
    halo_alpha: float | None = None,
    zorder: float = 3.0,
):
    if gdf.empty:
        return
    if halo_alpha is None:
        halo_alpha = min(alpha, 0.85)
    gdf.plot(ax=ax, color=halo_color, linewidth=halo_width, alpha=halo_alpha, zorder=zorder)
    gdf.plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder + 0.1)


@lru_cache(maxsize=1)
def zou_boundary_gdf_lonlat():
    import geopandas as gpd
    from shapely.geometry import LineString

    lon, lat = zou_boundary_lonlat()
    valid = np.isfinite(lon) & np.isfinite(lat)
    segments = []
    start = None
    for idx, ok in enumerate(valid):
        if ok and start is None:
            start = idx
        if start is not None and (idx == len(valid) - 1 or not valid[idx + 1]):
            seg = np.column_stack([lon[start : idx + 1], lat[start : idx + 1]])
            if len(seg) >= 2:
                segments.append(LineString(seg))
            start = None
    if not segments:
        segments = [LineString(np.column_stack([lon, lat]))]
    return gpd.GeoDataFrame(geometry=segments, crs="EPSG:4326")


def plot_boundary_with_halo(
    ax,
    hull,
    *,
    color: str = "0.15",
    linewidth: float = 0.9,
    halo_width: float = 2.6,
    alpha: float = 0.95,
    zorder: float = 3.0,
):
    boundary = clip_gdf_to_hull(zou_boundary_gdf_lonlat(), hull)
    plot_gdf_with_halo(
        ax,
        boundary,
        color=color,
        linewidth=linewidth,
        halo_width=halo_width,
        alpha=alpha,
        zorder=zorder,
    )


def hull_path(hull) -> MplPath:
    geom = hull if getattr(hull, "geom_type", None) == "Polygon" else hull.convex_hull
    return MplPath(np.asarray(geom.exterior.coords, dtype=float))


def clip_artist_to_hull(ax, artist, hull):
    geom = hull if getattr(hull, "geom_type", None) == "Polygon" else hull.convex_hull
    patch = MplPolygon(
        np.asarray(geom.exterior.coords, dtype=float),
        closed=True,
        facecolor="none",
        edgecolor="none",
        transform=ax.transData,
    )
    ax.add_patch(patch)
    artist.set_clip_path(patch)
    return patch


def mask_points_to_hull(x: np.ndarray, y: np.ndarray, hull) -> np.ndarray:
    path = hull_path(hull)
    points = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    return path.contains_points(points, radius=1e-9)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURE_DIR / f"{stem}.png"
    out_pdf = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(out_png, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def text_halo(width: float = 2.8, color: str = "white") -> list[patheffects.AbstractPathEffect]:
    return [patheffects.withStroke(linewidth=width, foreground=color)]


def geom_halo(width: float = 2.8, color: str = "white") -> list[patheffects.AbstractPathEffect]:
    return [patheffects.Stroke(linewidth=width, foreground=color), patheffects.Normal()]


@lru_cache(maxsize=1)
def project_crs():
    import geopandas as gpd

    return gpd.read_file(ROOT_DIR / "human_features" / "qtec_railway_clip.shp").crs


@lru_cache(maxsize=1)
def transformers():
    from pyproj import Transformer

    proj = project_crs()
    return (
        Transformer.from_crs(proj, "EPSG:4326", always_xy=True),
        Transformer.from_crs("EPSG:4326", proj, always_xy=True),
    )


def en_to_rc(easting: np.ndarray, northing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = grid_meta()
    res = float(meta["res"])
    gx0 = int(meta["gx0"])
    gy1 = int(meta["gy1"])
    gx = np.rint(np.asarray(easting, dtype=float) / res).astype(np.int64)
    gy = np.rint(np.asarray(northing, dtype=float) / res).astype(np.int64)
    col = (gx - gx0).astype(np.int32)
    row = (gy1 - gy).astype(np.int32)
    return row, col


def rc_to_en(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = grid_meta()
    res = float(meta["res"])
    gx0 = int(meta["gx0"])
    gy1 = int(meta["gy1"])
    easting = (gx0 + np.asarray(col, dtype=float)) * res
    northing = (gy1 - np.asarray(row, dtype=float)) * res
    return easting, northing


def crop_raster(
    arr: np.ndarray,
    *,
    center_e: float,
    center_n: float,
    window_km: float,
) -> tuple[np.ndarray, list[float], tuple[int, int, int, int]]:
    meta = grid_meta()
    res = float(meta["res"])
    nrows = int(meta["nrows"])
    ncols = int(meta["ncols"])
    row_center, col_center = en_to_rc(np.asarray([center_e]), np.asarray([center_n]))
    half_px = int(np.ceil((float(window_km) * 500.0) / res))
    row0 = max(0, int(row_center[0]) - half_px)
    row1 = min(nrows, int(row_center[0]) + half_px)
    col0 = max(0, int(col_center[0]) - half_px)
    col1 = min(ncols, int(col_center[0]) + half_px)
    crop = np.asarray(arr[row0:row1, col0:col1])
    x = (np.arange(col0, col1) - int(col_center[0])) * res / 1000.0
    y = (int(row_center[0]) - np.arange(row0, row1)) * res / 1000.0
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    return crop, extent, (row0, row1, col0, col1)


def add_scalebar_lonlat(
    ax,
    *,
    length_km: float,
    lon: float,
    lat: float,
    linewidth: float = 2.0,
    color: str = "0.15",
    label: str | None = None,
) -> None:
    km_per_lon = 111.32 * np.cos(np.deg2rad(float(lat)))
    if not np.isfinite(km_per_lon) or km_per_lon <= 0:
        return
    dlon = float(length_km) / km_per_lon
    ax.plot([lon, lon + dlon], [lat, lat], color=color, linewidth=linewidth, solid_capstyle="butt", zorder=12)
    ax.plot([lon, lon], [lat - 0.03, lat + 0.03], color=color, linewidth=linewidth, zorder=12)
    ax.plot([lon + dlon, lon + dlon], [lat - 0.03, lat + 0.03], color=color, linewidth=linewidth, zorder=12)
    ax.text(
        lon + dlon / 2.0,
        lat + 0.08,
        label or f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=FONT["annotation"],
        color=color,
        path_effects=text_halo(2.4),
        zorder=12,
    )


@lru_cache(maxsize=1)
def load_geo_layers() -> dict[str, object]:
    import geopandas as gpd

    railway_proj = gpd.read_file(ROOT_DIR / "human_features" / "qtec_railway_clip.shp")
    lakes_proj = gpd.read_file(ROOT_DIR / "qtec_thaw_lakes" / "TLS_des_clipped.shp")
    rts = gpd.read_file(ROOT_DIR / "qtp_rts" / "RTS-QTP.shp")
    if rts.crs != railway_proj.crs:
        rts_proj = rts.to_crs(railway_proj.crs)
    else:
        rts_proj = rts

    railway_ll = railway_proj.to_crs("EPSG:4326")
    lakes_ll = lakes_proj.to_crs("EPSG:4326")
    rts_ll = rts_proj.to_crs("EPSG:4326")

    frame_a = gpd.read_file("zip://" + str(ROOT_DIR / "QTP_frames.zip") + "!QTP_frames/S1A_Frames.shp")
    frame_d = gpd.read_file("zip://" + str(ROOT_DIR / "QTP_frames.zip") + "!QTP_frames/S1D_Frames.shp")

    return {
        "railway_proj": railway_proj,
        "railway_ll": railway_ll,
        "lakes_proj": lakes_proj,
        "lakes_ll": lakes_ll,
        "rts_proj": rts_proj,
        "rts_ll": rts_ll,
        "frame_a_ll": frame_a.to_crs("EPSG:4326"),
        "frame_d_ll": frame_d.to_crs("EPSG:4326"),
    }


@lru_cache(maxsize=1)
def zou_boundary_lonlat() -> tuple[np.ndarray, np.ndarray]:
    import _revised_zou_boundary_utils as zou

    cache_path = SOURCE_CACHE_DIR / "_revised_check_pf_extreme_env_dependence_zou_boundary_reference.joblib.gz"
    ref = zou.resolve_zou_boundary_reference(
        zou_tif=ROOT_DIR / "Zou_et_al_permafrost_distribution" / "Perma_Distr_map_TP" / "Perma_Distr_map_coreg.tif",
        cache_path=cache_path,
        mode="lonlat",
    )
    return (
        np.asarray(ref["boundary_lons"], dtype=float),
        np.asarray(ref["boundary_lats"], dtype=float),
    )


@lru_cache(maxsize=1)
def zou_boundary_projected() -> tuple[np.ndarray, np.ndarray]:
    to_proj = transformers()[1]
    lon, lat = zou_boundary_lonlat()
    return to_proj.transform(lon, lat)


def read_joblib_df(path: Path):
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and "df" in payload:
        return payload["df"]
    return payload


@lru_cache(maxsize=32)
def load_zou_mask_lonlat(extent_key: tuple[float, float, float, float], width: int = 900) -> tuple[np.ndarray, list[float]]:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    src_path = ROOT_DIR / "Zou_et_al_permafrost_distribution" / "Perma_Distr_map_TP" / "Perma_Distr_map_coreg.tif"
    xmin, xmax, ymin, ymax = [float(v) for v in extent_key]
    height = max(2, int(round(width * (ymax - ymin) / (xmax - xmin))))
    target = np.full((height, width), -9999, dtype=np.int16)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=target,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            dst_nodata=-9999,
        )
    return target, [xmin, xmax, ymin, ymax]


def plot_smoothed_binary_contour(
    ax,
    mask: np.ndarray,
    *,
    extent: list[float] | tuple[float, float, float, float],
    coverage=None,
    sigma: float = 6.0,
    color: str = "0.22",
    linewidth: float = 0.9,
    alpha: float = 0.95,
    zorder: float = 3.0,
):
    from scipy.ndimage import gaussian_filter

    work = np.asarray(mask, dtype=float)
    if work.size == 0 or not np.isfinite(work).any():
        return
    smooth = gaussian_filter(work, sigma=float(sigma))
    if coverage is not None:
        xmin, xmax, ymin, ymax = [float(v) for v in extent]
        xs = np.linspace(xmin, xmax, smooth.shape[1])
        ys = np.linspace(ymin, ymax, smooth.shape[0])
        xx, yy = np.meshgrid(xs, ys)
        inside = mask_points_to_hull(xx.ravel(), yy.ravel(), coverage).reshape(smooth.shape)
        smooth = np.where(inside, smooth, np.nan)
    ax.contour(
        smooth,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        origin="upper",
        extent=[float(v) for v in extent],
        alpha=alpha,
        zorder=zorder,
    )


def plot_smoothed_zou_boundary(
    ax,
    *,
    extent: list[float] | tuple[float, float, float, float],
    coverage=None,
    width: int = 900,
    sigma: float = 6.0,
    color: str = "0.22",
    linewidth: float = 0.9,
    alpha: float = 0.95,
    zorder: float = 3.0,
):
    zou_mask, extent_vals = load_zou_mask_lonlat(tuple(float(v) for v in extent), width=width)
    plot_smoothed_binary_contour(
        ax,
        zou_mask == 1,
        extent=extent_vals,
        coverage=coverage,
        sigma=sigma,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
