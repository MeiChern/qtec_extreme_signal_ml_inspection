#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/_tlrts_distance_utils.py
# Renamed package path: code/analysis_support/thermokarst_lake_slump_distance_utils.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import CRS

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("geopandas is required for TL/RTS distance utilities.") from exc

import figure6_susceptibility_stacked as fig6
import figure7_2_lake_influence_du_gradient as lakefig


DISTANCE_UTILS_BASENAME = "_tlrts_distance_utils"
DEFAULT_DISTANCE_CHUNK_SIZE = 50_000


def log_step(message: str) -> None:
    print(f"[{DISTANCE_UTILS_BASENAME}] {message}")


def _make_valid_series(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    try:
        from shapely import make_valid as shapely_make_valid
    except Exception:
        shapely_make_valid = None

    if shapely_make_valid is not None:
        return geoms.apply(lambda geom: shapely_make_valid(geom) if geom is not None else geom)
    return geoms.buffer(0)


def repair_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    *,
    label: str,
) -> gpd.GeoDataFrame:
    out = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if out.empty:
        raise RuntimeError(f"No valid geometries found after filtering empties for {label}.")

    invalid_mask = ~out.geometry.is_valid
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        log_step(f"Repairing {invalid_count} invalid geometries in {label}")
        out.geometry = _make_valid_series(out.geometry)
        out = out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()

        still_invalid = ~out.geometry.is_valid
        if bool(still_invalid.any()):
            retry_count = int(still_invalid.sum())
            log_step(f"Retrying repair for {retry_count} geometries in {label} with buffer(0)")
            out.loc[still_invalid, "geometry"] = out.loc[still_invalid, "geometry"].buffer(0)
            out = out.loc[out.geometry.notna() & ~out.geometry.is_empty].copy()

    final_invalid = int((~out.geometry.is_valid).sum())
    if final_invalid > 0:
        raise RuntimeError(f"{label} still has {final_invalid} invalid geometries after repair.")

    return out.reset_index(drop=True)


def load_feature_gdf(
    shp_path: Path,
    *,
    project_crs: CRS,
    type_col: str | None = None,
    type_values: tuple[str, ...] | None = None,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise RuntimeError(f"No valid geometries found in {shp_path}")

    if type_col is not None and type_values is not None:
        if type_col not in gdf.columns:
            raise RuntimeError(
                f"Requested filtering by {type_col}, but it is missing from {shp_path}"
            )
        gdf[type_col] = gdf[type_col].astype(str).str.strip().str.upper()
        gdf = gdf.loc[gdf[type_col].isin(type_values)].copy()
        if gdf.empty:
            raise RuntimeError(
                f"No features remain in {shp_path} after filtering {type_col} to {list(type_values)}"
            )

    if gdf.crs is None:
        raise RuntimeError(f"Missing CRS on {shp_path}")
    source_crs = CRS.from_user_input(gdf.crs)
    if not source_crs.equals(project_crs):
        gdf = gdf.to_crs(project_crs)
    return repair_invalid_geometries(gdf, label=str(shp_path))


def union_all_geometries(gdf: gpd.GeoDataFrame):
    geom_accessor = gdf.geometry
    if hasattr(geom_accessor, "union_all"):
        try:
            return geom_accessor.union_all()
        except Exception:
            repaired = repair_invalid_geometries(gdf, label="union input")
            return repaired.geometry.union_all()
    return geom_accessor.unary_union


def compute_distance_to_geometry_m(
    easting: np.ndarray,
    northing: np.ndarray,
    *,
    feature_geom,
    chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
) -> np.ndarray:
    out = np.full(len(easting), np.nan, dtype=np.float32)
    if getattr(feature_geom, "is_empty", False):
        return out

    e = np.asarray(easting, dtype=float)
    n = np.asarray(northing, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(e) & np.isfinite(n))
    if valid_idx.size == 0:
        return out

    for start in range(0, valid_idx.size, int(chunk_size)):
        idx = valid_idx[start : start + int(chunk_size)]
        points = gpd.GeoSeries(
            gpd.points_from_xy(e[idx], n[idx]),
            index=idx,
        )
        out[idx] = points.distance(feature_geom).to_numpy(dtype=np.float32)
    return out


def attach_tlrts_distance_columns(
    df: pd.DataFrame,
    *,
    lake_geom,
    rts_geom,
    chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
) -> pd.DataFrame:
    out = lakefig.engineer_features(df, copy_df=True)
    e = pd.to_numeric(out["easting"], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(out["northing"], errors="coerce").to_numpy(dtype=float)

    tl_dist_m = compute_distance_to_geometry_m(
        e,
        n,
        feature_geom=lake_geom,
        chunk_size=chunk_size,
    )
    rts_dist_m = compute_distance_to_geometry_m(
        e,
        n,
        feature_geom=rts_geom,
        chunk_size=chunk_size,
    )
    out["tl_distance_km"] = tl_dist_m / np.float32(1000.0)
    out["rts_distance_km"] = rts_dist_m / np.float32(1000.0)

    dist_stack = np.column_stack([out["tl_distance_km"], out["rts_distance_km"]])
    dist_both_nan = ~np.isfinite(dist_stack).any(axis=1)
    with np.errstate(all="ignore"):
        combined_dist = np.nanmin(dist_stack, axis=1)
    combined_dist[dist_both_nan] = np.nan
    out["tlrts_distance_km"] = combined_dist.astype(np.float32)
    out["tlrts_distance_m"] = out["tlrts_distance_km"] * np.float32(1000.0)
    return out


def build_distance_cache_signature(
    *,
    base_cache_path: Path,
    lakes_shp: Path,
    rts_shp: Path,
    rts_types: tuple[str, ...] | None,
    chunk_size: int,
) -> dict[str, object]:
    return {
        "artifact_version": 1,
        "base_cache": fig6.file_signature(base_cache_path),
        "lakes_shp": fig6.file_signature(lakes_shp),
        "rts_shp": fig6.file_signature(rts_shp),
        "rts_types": None if rts_types is None else list(rts_types),
        "chunk_size": int(chunk_size),
    }


def resolve_distance_cache(
    *,
    cache_path: Path,
    signature: dict[str, object],
) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        payload = joblib.load(cache_path)
    except Exception as exc:
        log_step(f"Warning: failed to load TL/RTS distance cache {cache_path}: {exc}")
        return None
    if isinstance(payload, dict) and payload.get("cache_signature") == signature:
        return payload.get("df")
    log_step(f"TL/RTS distance cache mismatch at {cache_path}; rebuilding.")
    return None


def save_distance_cache(
    *,
    cache_path: Path,
    signature: dict[str, object],
    df: pd.DataFrame,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "cache_signature": signature,
            "df": df,
        },
        cache_path,
        compress=("gzip", 3),
    )
