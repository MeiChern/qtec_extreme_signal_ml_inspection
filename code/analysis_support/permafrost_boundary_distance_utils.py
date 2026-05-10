#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/_revised_zou_boundary_utils.py
# Renamed package path: code/analysis_support/permafrost_boundary_distance_utils.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ZOU_UTILS_BASENAME = "_revised_zou_boundary_utils"

# Checked against the local recode.txt / VAT:
#   0 = seasonally frozen ground
#   1 = permafrost
#   2 = unfrozen ground
ZOU_PF_VALUES = (1,)
ZOU_NPF_VALUES = (0, 2)
ZOU_VALID_VALUES = tuple(sorted(set(ZOU_PF_VALUES + ZOU_NPF_VALUES)))

DISTANCE_BINS_KM = (0.0, 5.0, 10.0, 20.0, 40.0, math.inf)

BOUNDARY_MIGRATION_KM = 5.0
TL_ATTRIBUTED_THRESHOLD = 0.05
RTS_ATTRIBUTED_THRESHOLD = 0.05

BOUNDARY_REFERENCE_ARTIFACT_VERSION = 1


def log_step(message: str) -> None:
    print(f"[{ZOU_UTILS_BASENAME}] {message}")


def _require_rasterio():
    try:
        import rasterio
        from rasterio.crs import CRS as RioCRS
        from rasterio.transform import from_origin, rowcol, xy
        from rasterio.warp import Resampling, reproject
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "rasterio is required for the revised Zou-boundary utilities."
        ) from exc
    return rasterio, RioCRS, from_origin, rowcol, xy, Resampling, reproject


def _require_transformer():
    try:
        from pyproj import Transformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyproj is required for the revised Zou-boundary utilities."
        ) from exc
    return Transformer


def _require_nearest_neighbors():
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for projected Zou-boundary distances."
        ) from exc
    return NearestNeighbors


def _require_balltree():
    try:
        from sklearn.neighbors import BallTree
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for lon/lat Zou-boundary distances."
        ) from exc
    return BallTree


def path_signature(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_zou_tif(base_dir: Path) -> Path:
    zou_dir = base_dir / "Zou_et_al_permafrost_distribution" / "Perma_Distr_map_TP"
    raw = zou_dir / "Perma_Distr_map.tif"
    coreg = zou_dir / "Perma_Distr_map_coreg.tif"
    if coreg.exists():
        log_step(f"Using coregistered Zou map: {coreg}")
        return coreg
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Zou PF map not found in {zou_dir}")


def load_zou_permafrost_map(
    zou_tif: Path,
    *,
    grid: dict[str, object],
    target_crs_wkt: str,
) -> np.ndarray:
    """
    Load the Zou et al. map and resample it onto the analysis grid.

    Returns a boolean array where True = permafrost.
    """
    rasterio, RioCRS, from_origin, _rowcol, _xy, Resampling, reproject = _require_rasterio()

    nrows = int(grid["nrows"])
    ncols = int(grid["ncols"])
    res = float(grid["res"])
    min_e = float(grid["min_e"])
    max_n = float(grid["max_n"])

    target_transform = from_origin(min_e, max_n, res, res)
    target_crs = RioCRS.from_wkt(target_crs_wkt)
    dst = np.zeros((nrows, ncols), dtype=np.uint8)

    with rasterio.open(zou_tif) as src:
        log_step(f"Zou raster CRS: {src.crs}")
        log_step(f"Zou raster shape: {src.height} x {src.width}")
        log_step(f"Zou raster res: {src.res}")
        log_step(f"Target CRS: {target_crs}")

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

    unique_vals = np.unique(dst)
    log_step(f"Zou map resampled unique values: {unique_vals.tolist()}")

    pf_mask = np.isin(dst, ZOU_PF_VALUES)
    valid_mask = np.isin(dst, ZOU_VALID_VALUES)
    pf_count = int(pf_mask.sum())
    total_valid = int(valid_mask.sum())
    log_step(
        f"Zou PF pixels: {pf_count:,} "
        f"({100.0 * pf_count / max(total_valid, 1):.1f}% of valid)"
    )
    return pf_mask


def sample_zou_at_lonlat(
    zou_tif: Path,
    *,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample the Zou map at lon/lat coordinates.
    """
    rasterio, _RioCRS, _from_origin, rowcol, _xy, _Resampling, _reproject = _require_rasterio()
    Transformer = _require_transformer()

    lon_arr = np.asarray(lons, dtype=float)
    lat_arr = np.asarray(lats, dtype=float)
    result = np.full(len(lon_arr), np.nan, dtype=float)
    valid_ll = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    if not valid_ll.any():
        return result

    with rasterio.open(zou_tif) as src:
        if src.crs is None:
            raise RuntimeError(f"Zou raster has no CRS: {zou_tif}")

        if src.crs.is_geographic:
            xs = lon_arr[valid_ll]
            ys = lat_arr[valid_ll]
        else:
            transformer = Transformer.from_crs(
                "EPSG:4326",
                src.crs,
                always_xy=True,
            )
            xs, ys = transformer.transform(lon_arr[valid_ll], lat_arr[valid_ll])

        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        data = src.read(1)
        nrows, ncols = data.shape
        within = (
            (rows >= 0)
            & (rows < nrows)
            & (cols >= 0)
            & (cols < ncols)
        )
        valid_idx = np.flatnonzero(valid_ll)
        result[valid_idx[within]] = data[rows[within], cols[within]].astype(float)

    return result


def classify_zou_domain_at_points(
    zou_tif: Path,
    *,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Return per-point Zou-domain labels: 'pf', 'npf', or 'unknown'.
    """
    vals = sample_zou_at_lonlat(zou_tif, lons=lons, lats=lats)
    domain = np.full(len(vals), "unknown", dtype=object)

    finite = np.isfinite(vals)
    vals_int = np.full(len(vals), -9999, dtype=int)
    vals_int[finite] = np.rint(vals[finite]).astype(int)

    pf_mask = np.isin(vals_int, ZOU_PF_VALUES)
    npf_mask = finite & np.isin(vals_int, ZOU_NPF_VALUES)
    fallback_npf = finite & ~pf_mask & (vals_int >= 0)

    domain[pf_mask] = "pf"
    domain[npf_mask | fallback_npf] = "npf"
    return domain


def extract_zou_boundary_coords(
    zou_pf_mask: np.ndarray,
    *,
    grid: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract projected boundary-cell centroids from the clean Zou PF mask.
    """
    res = float(grid["res"])
    gx0 = int(grid["gx0"])
    gy1 = int(grid["gy1"])

    padded = np.pad(
        np.asarray(zou_pf_mask, dtype=bool),
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=False,
    )
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    boundary = center & (~up | ~down | ~left | ~right)
    rr, cc = np.nonzero(boundary)
    if rr.size == 0:
        raise RuntimeError("No Zou PF boundary cells found.")

    easting = (gx0 + cc.astype(np.int64)).astype(np.float64) * res
    northing = (gy1 - rr.astype(np.int64)).astype(np.float64) * res
    log_step(f"Zou boundary: {len(easting):,} cells (projected)")
    return easting.astype(np.float32), northing.astype(np.float32)


def extract_zou_boundary_lonlat(
    zou_tif: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Zou PF boundary cell centroids directly in lon/lat.
    """
    rasterio, _RioCRS, _from_origin, _rowcol, xy, _Resampling, _reproject = _require_rasterio()
    Transformer = _require_transformer()

    with rasterio.open(zou_tif) as src:
        if src.crs is None:
            raise RuntimeError(f"Zou raster has no CRS: {zou_tif}")

        data = src.read(1)
        pf_mask = np.isin(data, ZOU_PF_VALUES)

        padded = np.pad(
            pf_mask,
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=False,
        )
        center = padded[1:-1, 1:-1]
        boundary = center & (
            ~padded[:-2, 1:-1]
            | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2]
            | ~padded[1:-1, 2:]
        )
        rr, cc = np.nonzero(boundary)
        if rr.size == 0:
            raise RuntimeError("No Zou boundary cells found.")

        xs, ys = xy(src.transform, rr, cc, offset="center")
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)

        if src.crs.is_geographic:
            lons = xs
            lats = ys
        else:
            transformer = Transformer.from_crs(
                src.crs,
                "EPSG:4326",
                always_xy=True,
            )
            lons, lats = transformer.transform(xs, ys)

    log_step(f"Zou boundary: {len(lons):,} cells in lon/lat")
    return np.asarray(lons, dtype=np.float32), np.asarray(lats, dtype=np.float32)


def compute_zou_boundary_distance_projected(
    easting: np.ndarray,
    northing: np.ndarray,
    *,
    boundary_easting: np.ndarray,
    boundary_northing: np.ndarray,
) -> np.ndarray:
    """
    Euclidean projected distance to the clean Zou boundary, in km.
    """
    NearestNeighbors = _require_nearest_neighbors()

    coords_boundary = np.column_stack(
        [boundary_easting, boundary_northing]
    ).astype(np.float32)
    if coords_boundary.size == 0:
        raise RuntimeError("Zou projected boundary coordinates are empty.")

    e = np.asarray(easting, dtype=float)
    n = np.asarray(northing, dtype=float)
    ok = np.isfinite(e) & np.isfinite(n)
    dist_m = np.full(len(e), np.nan, dtype=np.float32)
    if ok.any():
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(coords_boundary)
        d, _ = nbrs.kneighbors(
            np.column_stack([e[ok], n[ok]]).astype(np.float32),
            return_distance=True,
        )
        dist_m[ok] = d[:, 0].astype(np.float32)
    return dist_m / np.float32(1000.0)


def compute_zou_boundary_distance_lonlat(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    boundary_lons: np.ndarray,
    boundary_lats: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance to the clean Zou boundary, in km.
    """
    BallTree = _require_balltree()

    lon_arr = np.asarray(lons, dtype=float)
    lat_arr = np.asarray(lats, dtype=float)
    boundary_lon_arr = np.asarray(boundary_lons, dtype=float)
    boundary_lat_arr = np.asarray(boundary_lats, dtype=float)

    boundary_ok = np.isfinite(boundary_lon_arr) & np.isfinite(boundary_lat_arr)
    if not boundary_ok.any():
        raise RuntimeError("Zou lon/lat boundary coordinates are empty.")

    boundary_rad = np.deg2rad(
        np.column_stack([boundary_lat_arr[boundary_ok], boundary_lon_arr[boundary_ok]])
    ).astype(np.float64)
    points_rad = np.deg2rad(
        np.column_stack([lat_arr, lon_arr])
    ).astype(np.float64)

    ok = np.all(np.isfinite(points_rad), axis=1)
    dist_km = np.full(len(lon_arr), np.nan, dtype=np.float32)
    if ok.any():
        tree = BallTree(boundary_rad, metric="haversine")
        d, _ = tree.query(points_rad[ok], k=1)
        dist_km[ok] = (d[:, 0] * 6371.0).astype(np.float32)
    return dist_km


def build_zou_boundary_reference_signature(
    zou_tif: Path,
    *,
    mode: str,
    grid: dict[str, object] | None = None,
    target_crs_wkt: str | None = None,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "artifact_version": BOUNDARY_REFERENCE_ARTIFACT_VERSION,
        "zou_tif": path_signature(zou_tif),
        "mode": str(mode),
    }
    if grid is not None:
        signature["grid"] = {
            "res": float(grid["res"]),
            "gx0": int(grid["gx0"]),
            "gy1": int(grid["gy1"]),
            "nrows": int(grid["nrows"]),
            "ncols": int(grid["ncols"]),
            "min_e": float(grid["min_e"]),
            "max_n": float(grid["max_n"]),
        }
    if target_crs_wkt is not None:
        signature["target_crs_wkt"] = str(target_crs_wkt)
    return signature


def resolve_zou_boundary_reference(
    zou_tif: Path,
    *,
    cache_path: Path,
    mode: str = "lonlat",
    grid: dict[str, object] | None = None,
    target_crs_wkt: str | None = None,
) -> dict[str, object]:
    """
    Resolve and cache Zou boundary coordinates in either lon/lat or projected mode.
    """
    if str(mode) not in {"lonlat", "projected"}:
        raise ValueError(f"Unsupported Zou boundary mode: {mode}")

    signature = build_zou_boundary_reference_signature(
        zou_tif,
        mode=mode,
        grid=grid,
        target_crs_wkt=target_crs_wkt,
    )
    if cache_path.exists():
        try:
            payload = joblib.load(cache_path)
        except Exception as exc:
            log_step(f"Warning: failed to load Zou boundary cache {cache_path}: {exc}")
        else:
            if (
                isinstance(payload, dict)
                and payload.get("cache_signature") == signature
                and payload.get("mode") == mode
            ):
                return payload
            log_step(f"Zou boundary cache mismatch at {cache_path}; rebuilding.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "lonlat":
        boundary_lons, boundary_lats = extract_zou_boundary_lonlat(zou_tif)
        payload = {
            "cache_signature": signature,
            "mode": "lonlat",
            "zou_tif": str(zou_tif),
            "boundary_lons": boundary_lons,
            "boundary_lats": boundary_lats,
            "n_boundary_cells": int(len(boundary_lons)),
        }
    else:
        if grid is None or target_crs_wkt is None:
            raise ValueError("Projected Zou boundary mode requires grid and target_crs_wkt.")
        pf_mask = load_zou_permafrost_map(
            zou_tif,
            grid=grid,
            target_crs_wkt=target_crs_wkt,
        )
        boundary_easting, boundary_northing = extract_zou_boundary_coords(
            pf_mask,
            grid=grid,
        )
        payload = {
            "cache_signature": signature,
            "mode": "projected",
            "zou_tif": str(zou_tif),
            "boundary_easting": boundary_easting,
            "boundary_northing": boundary_northing,
            "n_boundary_cells": int(len(boundary_easting)),
        }

    joblib.dump(payload, cache_path, compress=("gzip", 3))
    return payload


def attach_zou_domain_and_distance(
    df: pd.DataFrame,
    *,
    zou_tif: Path,
    boundary_ref: dict[str, object],
    overwrite_domain: bool = True,
) -> pd.DataFrame:
    """
    Attach `zou_domain`, `zou_boundary_distance_km`, and `zou_boundary_distance_m`.
    """
    out = df.copy()
    lon_arr = pd.to_numeric(out["longitude"], errors="coerce").to_numpy(dtype=float)
    lat_arr = pd.to_numeric(out["latitude"], errors="coerce").to_numpy(dtype=float)

    zou_domain = classify_zou_domain_at_points(
        zou_tif,
        lons=lon_arr,
        lats=lat_arr,
    )
    out["zou_domain"] = zou_domain
    if overwrite_domain:
        out["domain"] = zou_domain

    mode = str(boundary_ref.get("mode"))
    if mode == "lonlat":
        dist_km = compute_zou_boundary_distance_lonlat(
            lon_arr,
            lat_arr,
            boundary_lons=np.asarray(boundary_ref["boundary_lons"], dtype=float),
            boundary_lats=np.asarray(boundary_ref["boundary_lats"], dtype=float),
        )
    elif mode == "projected":
        easting = pd.to_numeric(out["easting"], errors="coerce").to_numpy(dtype=float)
        northing = pd.to_numeric(out["northing"], errors="coerce").to_numpy(dtype=float)
        dist_km = compute_zou_boundary_distance_projected(
            easting,
            northing,
            boundary_easting=np.asarray(boundary_ref["boundary_easting"], dtype=float),
            boundary_northing=np.asarray(boundary_ref["boundary_northing"], dtype=float),
        )
    else:
        raise ValueError(f"Unsupported Zou boundary reference mode: {mode}")

    out["zou_boundary_distance_km"] = dist_km
    out["zou_boundary_distance_m"] = dist_km * np.float32(1000.0)
    return out


def classify_pf_extreme_population(
    df: pd.DataFrame,
    *,
    zou_distance_col: str = "zou_boundary_distance_km",
    tl_influence_col: str = "lake_influence_norm01",
    rts_influence_col: str = "rts_influence_norm01",
) -> pd.Series:
    """
    Classify PF rows into boundary-migration / interior-attributed / interior-unexplained.
    """
    zou_dist = pd.to_numeric(df[zou_distance_col], errors="coerce").to_numpy(dtype=float)
    tl = pd.to_numeric(df[tl_influence_col], errors="coerce").to_numpy(dtype=float)
    rts = pd.to_numeric(df[rts_influence_col], errors="coerce").to_numpy(dtype=float)

    pop = np.full(len(df), "unclassified", dtype=object)

    boundary_mask = np.isfinite(zou_dist) & (zou_dist < float(BOUNDARY_MIGRATION_KM))
    pop[boundary_mask] = "boundary_migration"

    interior_mask = np.isfinite(zou_dist) & ~boundary_mask
    near_feature = (
        (np.where(np.isfinite(tl), tl, 0.0) >= float(TL_ATTRIBUTED_THRESHOLD))
        | (np.where(np.isfinite(rts), rts, 0.0) >= float(RTS_ATTRIBUTED_THRESHOLD))
    )
    pop[interior_mask & near_feature] = "interior_attributed"
    pop[interior_mask & ~near_feature] = "interior_unexplained"

    return pd.Series(pop, index=df.index, name="pf_population")
