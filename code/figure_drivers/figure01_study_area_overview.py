#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/Figure01_study_area_overview.py
# Renamed package path: code/figure_drivers/figure01_study_area_overview.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

from io import BytesIO
import math
import re
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds
from shapely.geometry import box
from shapely.ops import transform

from submission_build_common import (
    CACHE_DIR,
    FONT,
    ROOT_DIR,
    add_panel_label,
    blend_with_white,
    clip_artist_to_hull,
    ensure_style,
    save_figure,
)


FIG_STEM = "Figure01_study_area_overview"
WAYBACK_DIR = Path("/mnt/d/OneDriveBackUps/OneDrive/Desktop/Wayback_Imagery")
WEB_MERCATOR = "EPSG:3857"
LONLAT = "EPSG:4326"
BUFFER_SAMPLE_FILE = "20221102_Fenghuoshan_North_Suspicious_Region.tpkx"
PF_RASTER_PATH = ROOT_DIR / "Zou_et_al_permafrost_distribution" / "Perma_Distr_map_TP" / "Perma_Distr_map.tif"
ESRI_HILLSHADE_BASE_URL = "https://tiledbasemaps.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer"
ESRI_HILLSHADE_URL = f"{ESRI_HILLSHADE_BASE_URL}/tile/{{z}}/{{y}}/{{x}}"
ESRI_HILLSHADE_CACHE_DIR = CACHE_DIR / "esri_world_hillshade"
PANEL_A_HILLSHADE_PAD_DEG = 0.1

QTP_FACE = blend_with_white("#D8D3C7", 0.72)
PF_FACE = blend_with_white("#5A8F63", 0.48)
NPF_FACE = blend_with_white("#9A6A49", 0.50)
S1A_EDGE = "#356899"
S1D_EDGE = "#A86521"
TL_COLOR = "#4E7A84"
TL_BOUNDARY_COLOR = "#1677FF"
RTS_COLOR = "#D84C6F"
RAIL_COLOR = "0.08"
QTEC_COLOR = "#972D15"
BUFFER_FACE = blend_with_white("#808080", 0.84)
BUFFER_EDGE = "0.35"
MET_SITE_COLOR = "0.05"
FRAME_DASHES = (7, 4)
RAIL_DASHES = (8, 5)
WINDOW_PANEL_ASPECT = 1.65

STATION_LABEL_SPECS = {
    "Tuotuohe": (-6.0, 4.0, "right"),
    "Wudaoliang": (6.0, 4.0, "left"),
    "Xiaozaohuo": (6.0, 4.0, "left"),
    "Golmud": (6.0, -9.0, "left"),
    "Banga": (-6.0, 4.0, "right"),
    "Nagqu": (6.0, 4.0, "left"),
    "Lhasa": (-6.0, -10.0, "right"),
    "Shannan": (6.0, -10.0, "left"),
}

WINDOW_SPECS = (
    {
        "title": "Thermokarst lake window 1",
        "subtitle": "Fenghuoshan North",
        "filename": "20221102_Fenghuoshan_North_Suspicious_Region.tpkx",
        "kind": "tl",
        "frame_color": "#296A86",
        "panel_label": "C",
        "crop_lonlat": {
            "lon_min": 92.9045253,
            "lon_max": 92.9314747,
            "lat_min": 34.8132954,
            "lat_max": 34.8267040,
        },
        "crop_aspect": WINDOW_PANEL_ASPECT,
        "scalebar_km": 0.5,
    },
    {
        "title": "Thermokarst lake window 2",
        "subtitle": "Yangshiping North",
        "filename": "Wayback_Yangshiping_north_TL_20221102.tpkx",
        "kind": "tl",
        "frame_color": "#3C7C85",
        "panel_label": "D",
        "crop_lonlat": {
            "lon_min": 92.1452310,
            "lon_max": 92.1718410,
            "lat_min": 33.7500000,
            "lat_max": 33.7634083,
        },
        "crop_aspect": WINDOW_PANEL_ASPECT,
        "scalebar_km": 0.5,
    },
    {
        "title": "Thaw slump window",
        "subtitle": "Fenghuoshan North",
        "filename": "20221102_Fenghuoshan_North_Suspicious_Region.tpkx",
        "kind": "rts",
        "frame_color": RTS_COLOR,
        "panel_label": "E",
        "crop_lonlat": {
            "lon_min": 92.8824320,
            "lon_max": 92.8878147,
            "lat_min": 34.7075327,
            "lat_max": 34.7102144,
        },
        "crop_aspect": WINDOW_PANEL_ASPECT,
        "scalebar_km": 0.1,
    },
)


def _drop_z(geom):
    if geom is None or geom.is_empty:
        return geom
    if not getattr(geom, "has_z", False):
        return geom
    return transform(lambda x, y, z=None: (x, y), geom)


def _geometry_union(gdf: gpd.GeoDataFrame):
    union_all = getattr(gdf.geometry, "union_all", None)
    if callable(union_all):
        return union_all()
    return gdf.unary_union


def _clip_to_geometry(gdf: gpd.GeoDataFrame, geom) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    return gpd.clip(gdf, gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs))


def _extent_from_bounds(bounds: np.ndarray | list[float], *, xpad: float, ypad: float) -> list[float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    return [xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad]


def _merge_bounds(bounds_list: list[list[float] | tuple[float, float, float, float]], *, xpad: float, ypad: float) -> list[float]:
    xmin = min(float(bounds[0]) for bounds in bounds_list)
    ymin = min(float(bounds[1]) for bounds in bounds_list)
    xmax = max(float(bounds[2]) for bounds in bounds_list)
    ymax = max(float(bounds[3]) for bounds in bounds_list)
    return [xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad]


def _set_window_panel(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("0.15")


@lru_cache(maxsize=1)
def mercator_transformers() -> tuple[Transformer, Transformer]:
    return (
        Transformer.from_crs(WEB_MERCATOR, LONLAT, always_xy=True),
        Transformer.from_crs(LONLAT, WEB_MERCATOR, always_xy=True),
    )


def _choose_step(span: float, candidates: list[float]) -> float:
    for step in candidates:
        if span / step <= 6.0:
            return step
    return candidates[-1]


def _format_lon(value: float, step: float) -> str:
    digits = 0 if step >= 1.0 else (1 if step >= 0.5 else 2)
    return f"{value:.{digits}f}°E"


def _format_lat(value: float, step: float) -> str:
    digits = 0 if step >= 1.0 else (1 if step >= 0.5 else 2)
    return f"{value:.{digits}f}°N"


def apply_lonlat_axes(
    ax,
    extent: list[float],
    *,
    show_grid: bool = False,
    xlabel: str = "Longitude",
    ylabel: str = "Latitude",
    lon_ticks_deg: list[float] | None = None,
    lat_ticks_deg: list[float] | None = None,
    lon_formatter=None,
    lat_formatter=None,
) -> None:
    to_ll, from_ll = mercator_transformers()
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    lon0, lat0 = to_ll.transform(xmin, ymin)
    lon1, lat1 = to_ll.transform(xmax, ymax)
    lon_lo, lon_hi = sorted([lon0, lon1])
    lat_lo, lat_hi = sorted([lat0, lat1])
    lon_mid = 0.5 * (lon_lo + lon_hi)
    lat_mid = 0.5 * (lat_lo + lat_hi)

    lon_step = _choose_step(lon_hi - lon_lo, [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    lat_step = _choose_step(lat_hi - lat_lo, [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0])

    if lon_ticks_deg is None:
        lon_ticks_deg = np.arange(math.ceil(lon_lo / lon_step) * lon_step, lon_hi + lon_step * 0.25, lon_step)
    else:
        lon_ticks_deg = np.asarray(lon_ticks_deg, dtype=float)
        if len(lon_ticks_deg) >= 2:
            lon_step = float(np.min(np.diff(lon_ticks_deg)))
    if lat_ticks_deg is None:
        lat_ticks_deg = np.arange(math.ceil(lat_lo / lat_step) * lat_step, lat_hi + lat_step * 0.25, lat_step)
    else:
        lat_ticks_deg = np.asarray(lat_ticks_deg, dtype=float)
        if len(lat_ticks_deg) >= 2:
            lat_step = float(np.min(np.diff(lat_ticks_deg)))

    x_ticks = [from_ll.transform(float(lon), lat_mid)[0] for lon in lon_ticks_deg]
    y_ticks = [from_ll.transform(lon_mid, float(lat))[1] for lat in lat_ticks_deg]

    if lon_formatter is None:
        lon_formatter = lambda lon: _format_lon(float(lon), lon_step)
    if lat_formatter is None:
        lat_formatter = lambda lat: _format_lat(float(lat), lat_step)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([lon_formatter(float(lon)) for lon in lon_ticks_deg], fontsize=FONT["tick"])
    ax.set_yticklabels([lat_formatter(float(lat)) for lat in lat_ticks_deg], fontsize=FONT["tick"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(color="0.88", linewidth=0.5)
    else:
        ax.grid(False)


def add_north_arrow(ax, *, x: float = 0.95, y: float = 0.93, size: float = 0.10) -> None:
    text = ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=FONT["annotation"] + 1,
        fontweight="bold",
        color="0.08",
        arrowprops={"arrowstyle": "-|>", "color": "0.08", "linewidth": 1.6},
        zorder=12,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])


def add_scalebar_projected(
    ax,
    *,
    length_km: float,
    x: float,
    y: float,
    ref_lat: float,
    color: str = "0.12",
    halo: bool = True,
) -> None:
    cos_lat = max(0.15, math.cos(math.radians(float(ref_lat))))
    length_proj = float(length_km) * 1000.0 / cos_lat
    _, _, ymin, ymax = ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]
    tick_height = 0.012 * (ymax - ymin)
    ax.plot([x, x + length_proj], [y, y], color=color, linewidth=2.0, solid_capstyle="butt", zorder=12)
    ax.plot([x, x], [y - tick_height, y + tick_height], color=color, linewidth=2.0, zorder=12)
    ax.plot([x + length_proj, x + length_proj], [y - tick_height, y + tick_height], color=color, linewidth=2.0, zorder=12)
    label = ax.text(
        x + 0.5 * length_proj,
        y + 1.6 * tick_height,
        f"{int(round(length_km * 1000))} m" if length_km < 1.0 else f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=FONT["annotation"],
        color=color,
        zorder=12,
    )
    if halo:
        label.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    x_tile = (float(lon) + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_tile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x_tile, y_tile


def _tile_to_lonlat(x_tile: float, y_tile: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = float(x_tile) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y_tile) / n))))
    return lon, lat


def _esri_tile_cache_path(x_tile: int, y_tile: int, zoom: int) -> Path:
    return ESRI_HILLSHADE_CACHE_DIR / f"z{int(zoom)}" / f"x{int(x_tile)}" / f"y{int(y_tile)}.jpg"


@lru_cache(maxsize=96)
def _fetch_esri_tile(x_tile: int, y_tile: int, zoom: int) -> np.ndarray | None:
    cache_path = _esri_tile_cache_path(x_tile, y_tile, zoom)
    if cache_path.exists():
        try:
            image = Image.open(cache_path).convert("RGBA")
        except Exception:
            pass
        else:
            return np.asarray(image, dtype=np.uint8)

    url = ESRI_HILLSHADE_URL.format(z=int(zoom), y=int(y_tile), x=int(x_tile))
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read()
            image = Image.open(BytesIO(payload)).convert("RGBA")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
    except Exception:
        return None
    return np.asarray(image, dtype=np.uint8)


def lonlat_bbox_for_extent(extent: list[float], *, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
    to_ll, _ = mercator_transformers()
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    lon0, lat0 = to_ll.transform(xmin, ymin)
    lon1, lat1 = to_ll.transform(xmax, ymax)
    lon_min, lon_max = sorted([float(lon0), float(lon1)])
    lat_min, lat_max = sorted([float(lat0), float(lat1)])
    pad = float(pad_deg)
    return lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad


def esri_tile_ranges_for_bbox(
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    zoom: int,
) -> tuple[range, range]:
    x0f, y1f = _lonlat_to_tile(lon_min, lat_min, zoom)
    x1f, y0f = _lonlat_to_tile(lon_max, lat_max, zoom)
    x_tiles = range(max(0, int(math.floor(x0f))), max(0, int(math.floor(x1f))) + 1)
    y_tiles = range(max(0, int(math.floor(y0f))), max(0, int(math.floor(y1f))) + 1)
    return x_tiles, y_tiles


def draw_esri_hillshade(
    ax,
    *,
    extent: list[float],
    zoom: int = 5,
    alpha: float = 0.26,
    zorder: float = -1,
    clip_geom=None,
) -> bool:
    to_ll, from_ll = mercator_transformers()
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    lon0, lat0 = to_ll.transform(xmin, ymin)
    lon1, lat1 = to_ll.transform(xmax, ymax)
    lon_lo, lon_hi = sorted([lon0, lon1])
    lat_lo, lat_hi = sorted([lat0, lat1])
    x0f, y1f = _lonlat_to_tile(lon_lo, lat_lo, zoom)
    x1f, y0f = _lonlat_to_tile(lon_hi, lat_hi, zoom)
    x_tiles = list(range(max(0, int(math.floor(x0f))), max(0, int(math.floor(x1f))) + 1))
    y_tiles = list(range(max(0, int(math.floor(y0f))), max(0, int(math.floor(y1f))) + 1))
    if not x_tiles or not y_tiles:
        return False

    rows: list[np.ndarray] = []
    for y_tile in y_tiles:
        row_tiles: list[np.ndarray] = []
        for x_tile in x_tiles:
            tile = _fetch_esri_tile(x_tile, y_tile, zoom)
            if tile is None:
                return False
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))

    image = np.concatenate(rows, axis=0).astype(np.float32) / 255.0
    image[..., :3] = np.clip((image[..., :3] - 0.5) * 1.22 + 0.5, 0.0, 1.0)

    lon_w, lat_n = _tile_to_lonlat(x_tiles[0], y_tiles[0], zoom)
    lon_e, lat_s = _tile_to_lonlat(x_tiles[-1] + 1, y_tiles[-1] + 1, zoom)
    x_w, y_n = from_ll.transform(lon_w, lat_n)
    x_e, y_s = from_ll.transform(lon_e, lat_s)
    artist = ax.imshow(
        image,
        extent=[float(x_w), float(x_e), float(y_s), float(y_n)],
        origin="upper",
        alpha=alpha,
        zorder=zorder,
    )
    if clip_geom is not None:
        clip_artist_to_hull(ax, artist, clip_geom)
    return True


def lonlat_extent_to_mercator(*, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> list[float]:
    _, from_ll = mercator_transformers()
    x0, y0 = from_ll.transform(float(lon_min), float(lat_min))
    x1, y1 = from_ll.transform(float(lon_max), float(lat_max))
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    return [float(xmin), float(xmax), float(ymin), float(ymax)]


def date_from_tpkx_name(filename: str) -> str:
    match = re.search(r"(?<!\d)((?:19|20)\d{6})(?!\d)", filename)
    if match is None:
        return ""
    return match.group(1)


def resolve_station_labels(station_gdf: gpd.GeoDataFrame):
    for col in ("english_st", "english_station_name", "station_na", "station_name", "NAME"):
        if col not in station_gdf.columns:
            continue
        labels = station_gdf[col].astype(str).str.strip()
        labels = labels.mask(labels.eq("")).mask(labels.str.lower().isin({"nan", "none"}))
        if labels.notna().any():
            return labels
    if "station_id" in station_gdf.columns:
        return station_gdf["station_id"].map(lambda value: f"Site {value}")
    raise RuntimeError("Could not resolve a site-label field from the meteoro station shapefile.")


def plot_dashed_railway(ax, railway: gpd.GeoDataFrame, *, linewidth: float, halo_width: float, zorder: float) -> None:
    if railway.empty:
        return
    railway.plot(ax=ax, color="white", linewidth=halo_width, alpha=1.0, zorder=zorder)
    railway.plot(
        ax=ax,
        color=RAIL_COLOR,
        linewidth=linewidth,
        alpha=1.0,
        linestyle=(0, RAIL_DASHES),
        zorder=zorder + 0.1,
    )


def draw_pf_background(ax, *, extent: list[float], clip_geom=None, width: int, alpha: float = 1.0) -> None:
    pf, _ = load_pf_background(extent, width=width)
    pf_mask = np.ma.masked_where(pf != 1, pf)
    npf_mask = np.ma.masked_where(~np.isin(pf, [0, 2]), pf)
    npf_cmap = plt.matplotlib.colors.ListedColormap([NPF_FACE])
    pf_cmap = plt.matplotlib.colors.ListedColormap([PF_FACE])
    npf_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    pf_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    npf_im = ax.imshow(npf_mask, extent=extent, origin="upper", cmap=npf_cmap, alpha=alpha, zorder=1)
    pf_im = ax.imshow(pf_mask, extent=extent, origin="upper", cmap=pf_cmap, alpha=alpha, zorder=2)
    if clip_geom is not None:
        clip_artist_to_hull(ax, npf_im, clip_geom)
        clip_artist_to_hull(ax, pf_im, clip_geom)


def style_translucent_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.50)
    frame.set_edgecolor("none")
    frame.set_linewidth(0.0)


@lru_cache(maxsize=1)
def load_layers() -> dict[str, gpd.GeoDataFrame]:
    def _read(path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf.geometry = gdf.geometry.apply(_drop_z)
        if gdf.crs != WEB_MERCATOR:
            gdf = gdf.to_crs(WEB_MERCATOR)
        return gdf

    meteoro = _read(ROOT_DIR / "human_features" / "qtec_meteoro_station_sites.shp")
    meteoro["site_label"] = resolve_station_labels(meteoro)
    meteoro = meteoro.loc[meteoro["site_label"].notna()].copy()

    return {
        "qtp_aoi": _read(ROOT_DIR / "human_features" / "qtp_aoi.shp"),
        "qtec_aoi": _read(ROOT_DIR / "human_features" / "qtec_aoi.shp"),
        "railway": _read(ROOT_DIR / "human_features" / "qtp_railway.shp"),
        "frame_a": _read(ROOT_DIR / "human_features" / "qtec_s1a.shp"),
        "frame_d": _read(ROOT_DIR / "human_features" / "qtec_s1d.shp"),
        "lakes": _read(ROOT_DIR / "qtec_thaw_lakes" / "TLS_des_clipped.shp"),
        "rts": _read(ROOT_DIR / "qtp_rts" / "RTS-QTP.shp"),
        "meteoro": meteoro,
    }


def load_pf_background(extent: list[float], width: int = 1600) -> tuple[np.ndarray, list[float]]:
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    height = max(2, int(round(width * (ymax - ymin) / (xmax - xmin))))
    target = np.full((height, width), -9999, dtype=np.int16)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    with rasterio.open(PF_RASTER_PATH) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=target,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=WEB_MERCATOR,
            resampling=Resampling.nearest,
            dst_nodata=-9999,
        )
    return target, [xmin, xmax, ymin, ymax]


def read_wayback_rgb(path: Path, *, width: int = 560, extent: list[float] | None = None) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(path) as src:
        if extent is None:
            left, right, bottom, top = src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top
            read_window = None
        else:
            left, right, bottom, top = [float(v) for v in extent]
            read_window = window_from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        height = max(2, int(round(float(width) * (top - bottom) / (right - left))))
        data = src.read(
            indexes=[1, 2, 3],
            window=read_window,
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
            boundless=True,
        )
    return np.moveaxis(data, 0, -1), [float(left), float(right), float(bottom), float(top)]


@lru_cache(maxsize=1)
def window_specs() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in WINDOW_SPECS:
        path = WAYBACK_DIR / str(spec["filename"])
        if path.exists():
            item = dict(spec)
            item["path"] = path
            item["date"] = date_from_tpkx_name(path.name)
            out.append(item)
    return out


def _extent_from_window(bounds: list[float] | tuple[float, float, float, float], *, width_frac: float, height_frac: float) -> list[float]:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    width = xmax - xmin
    height = ymax - ymin
    target_width = width * float(width_frac)
    target_height = height * float(height_frac)
    return [xmin, xmin + target_width, ymin, ymin + target_height]


def _expand_extent_to_aspect(extent: list[float], target_aspect: float) -> list[float]:
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return extent

    current_aspect = width / height
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    if current_aspect < float(target_aspect):
        width = height * float(target_aspect)
    else:
        height = width / float(target_aspect)
    return [center_x - 0.5 * width, center_x + 0.5 * width, center_y - 0.5 * height, center_y + 0.5 * height]


def _clamp_extent(extent: list[float], bounds: list[float] | tuple[float, float, float, float]) -> list[float]:
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    bxmin, bymin, bxmax, bymax = [float(v) for v in bounds]
    width = min(xmax - xmin, bxmax - bxmin)
    height = min(ymax - ymin, bymax - bymin)
    xmin = min(max(xmin, bxmin), bxmax - width)
    ymin = min(max(ymin, bymin), bymax - height)
    return [xmin, xmin + width, ymin, ymin + height]


def _window_point_layer(kind: str, layers: dict[str, gpd.GeoDataFrame], chip_geom) -> gpd.GeoDataFrame:
    if kind == "tl":
        points = layers["lakes"].copy()
    else:
        points = layers["rts"].copy()
    points.geometry = points.geometry.centroid
    return points.loc[points.geometry.intersects(chip_geom)].copy()


def _search_window_focus(
    chip_bounds: list[float],
    points: gpd.GeoDataFrame,
    railway: gpd.GeoDataFrame,
    *,
    crop_frac: float,
    railway_bonus: float,
) -> list[float] | None:
    xmin, xmax, ymin, ymax = [float(v) for v in chip_bounds]
    chip_width = xmax - xmin
    chip_height = ymax - ymin
    window_width = chip_width * float(crop_frac)
    window_height = chip_height * float(crop_frac)
    step_x = max(1_500.0, 0.18 * window_width)
    step_y = max(1_500.0, 0.18 * window_height)
    if points.empty and railway.empty:
        return None

    point_x = points.geometry.x.to_numpy(dtype=float) if not points.empty else np.empty(0, dtype=float)
    point_y = points.geometry.y.to_numpy(dtype=float) if not points.empty else np.empty(0, dtype=float)
    x_candidates = np.arange(xmin, max(xmin, xmax - window_width) + step_x * 0.5, step_x)
    y_candidates = np.arange(ymin, max(ymin, ymax - window_height) + step_y * 0.5, step_y)

    best_score = None
    best_extent = None
    for x0 in x_candidates:
        x1 = min(xmax, x0 + window_width)
        x0 = x1 - window_width
        point_x_mask = (point_x >= x0) & (point_x <= x1)
        for y0 in y_candidates:
            y1 = min(ymax, y0 + window_height)
            y0 = y1 - window_height
            point_count = int(np.count_nonzero(point_x_mask & (point_y >= y0) & (point_y <= y1)))
            window_geom = box(x0, y0, x1, y1)
            railway_hit = 0.0
            if not railway.empty and railway.intersects(window_geom).any():
                railway_hit = railway_bonus
            score = float(point_count) + railway_hit
            if best_score is None or score > best_score:
                best_score = score
                best_extent = [float(x0), float(x1), float(y0), float(y1)]

    return best_extent


def choose_window_crop(spec: dict[str, object], layers: dict[str, gpd.GeoDataFrame], full_extent: list[float]) -> list[float]:
    crop_lonlat = spec.get("crop_lonlat")
    if isinstance(crop_lonlat, dict):
        fixed_extent = lonlat_extent_to_mercator(
            lon_min=float(crop_lonlat["lon_min"]),
            lon_max=float(crop_lonlat["lon_max"]),
            lat_min=float(crop_lonlat["lat_min"]),
            lat_max=float(crop_lonlat["lat_max"]),
        )
        if "crop_aspect" in spec:
            fixed_extent = _expand_extent_to_aspect(fixed_extent, float(spec["crop_aspect"]))
        return _clamp_extent(fixed_extent, [full_extent[0], full_extent[2], full_extent[1], full_extent[3]])

    chip_geom = box(full_extent[0], full_extent[2], full_extent[1], full_extent[3])
    points = _window_point_layer(str(spec["kind"]), layers, chip_geom)
    railway = _clip_to_geometry(layers["railway"], chip_geom)
    crop_frac = float(spec.get("crop_frac", 0.28))
    focus_extent = _search_window_focus(
        full_extent,
        points,
        railway,
        crop_frac=crop_frac,
        railway_bonus=6.0 if str(spec["kind"]) == "tl" else 0.0,
    )

    if focus_extent is None:
        focus_extent = _extent_from_window(
            [full_extent[0], full_extent[2], full_extent[1], full_extent[3]],
            width_frac=crop_frac,
            height_frac=crop_frac,
        )

    if not points.empty:
        pxmin, pymin, pxmax, pymax = [float(v) for v in points.total_bounds]
        focus_center_x = 0.5 * (focus_extent[0] + focus_extent[1])
        focus_center_y = 0.5 * (focus_extent[2] + focus_extent[3])
        point_center_x = 0.5 * (pxmin + pxmax)
        point_center_y = 0.5 * (pymin + pymax)
        nudge = 0.25 if str(spec.get("kind", "")) == "rts" else 0.10
        y_offset = 0.18 * (focus_extent[3] - focus_extent[2]) if str(spec.get("kind", "")) == "rts" else 0.0
        focus_extent = [
            focus_extent[0] + nudge * (point_center_x - focus_center_x),
            focus_extent[1] + nudge * (point_center_x - focus_center_x),
            focus_extent[2] + nudge * (point_center_y - focus_center_y) + y_offset,
            focus_extent[3] + nudge * (point_center_y - focus_center_y) + y_offset,
        ]

    return _clamp_extent(focus_extent, [full_extent[0], full_extent[2], full_extent[1], full_extent[3]])


def prepare_windows(layers: dict[str, gpd.GeoDataFrame]) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for spec in window_specs():
        item = dict(spec)
        with rasterio.open(Path(item["path"])) as src:
            full_extent = [float(src.bounds.left), float(src.bounds.right), float(src.bounds.bottom), float(src.bounds.top)]
        item["full_extent"] = full_extent
        item["crop_extent"] = choose_window_crop(item, layers, full_extent)
        prepared.append(item)
    return prepared


def _representative_panel_b_window(lakes: gpd.GeoDataFrame, rts: gpd.GeoDataFrame) -> list[float] | None:
    if lakes.empty or rts.empty:
        return None

    lake_x = lakes.geometry.x.to_numpy(dtype=float)
    lake_y = lakes.geometry.y.to_numpy(dtype=float)
    rts_x = rts.geometry.x.to_numpy(dtype=float)
    rts_y = rts.geometry.y.to_numpy(dtype=float)

    xs = np.concatenate([lake_x, rts_x])
    ys = np.concatenate([lake_y, rts_y])
    x_window = 30_000.0
    y_window = 45_000.0
    step = 5_000.0

    best_score = None
    best_bounds = None
    for x0 in np.arange(xs.min(), xs.max() + step, step):
        x1 = x0 + x_window
        lake_x_mask = (lake_x >= x0) & (lake_x <= x1)
        rts_x_mask = (rts_x >= x0) & (rts_x <= x1)
        if not lake_x_mask.any() or not rts_x_mask.any():
            continue
        for y0 in np.arange(ys.min(), ys.max() + step, step):
            y1 = y0 + y_window
            lake_count = np.count_nonzero(lake_x_mask & (lake_y >= y0) & (lake_y <= y1))
            rts_count = np.count_nonzero(rts_x_mask & (rts_y >= y0) & (rts_y <= y1))
            if lake_count == 0 or rts_count == 0:
                continue
            score = lake_count + 12 * rts_count
            if best_score is None or score > best_score:
                best_score = score
                best_bounds = [float(x0), float(y0), float(x1), float(y1)]

    if best_bounds is None:
        return None
    return _extent_from_bounds(best_bounds, xpad=2_500.0, ypad=3_000.0)


def choose_panel_b_extent(layers: dict[str, gpd.GeoDataFrame], windows: list[dict[str, object]] | None = None) -> list[float]:
    railway_buf = _geometry_union(layers["railway"]).buffer(5_000.0)
    if windows:
        window_geom = _geometry_union(
            gpd.GeoDataFrame(
                geometry=[box(extent[0], extent[2], extent[1], extent[3]) for extent in (spec["crop_extent"] for spec in windows)],
                crs=WEB_MERCATOR,
            )
        )
        focus_geom = railway_buf.intersection(window_geom.buffer(18_000.0))
        if focus_geom.is_empty:
            focus_geom = window_geom.buffer(18_000.0)
    else:
        focus_geom = None

    if focus_geom is None:
        sample_path = WAYBACK_DIR / BUFFER_SAMPLE_FILE
        if sample_path.exists():
            with rasterio.open(sample_path) as src:
                focus_geom = railway_buf.intersection(box(*src.bounds))
        else:
            focus_geom = railway_buf

    if focus_geom.is_empty:
        focus_geom = railway_buf

    lakes = layers["lakes"].copy()
    lakes.geometry = lakes.geometry.centroid
    rts = layers["rts"].copy()
    rts.geometry = rts.geometry.centroid

    local_lakes = lakes.loc[lakes.geometry.intersects(focus_geom)].copy()
    local_rts = rts.loc[rts.geometry.intersects(focus_geom)].copy()

    representative_extent = None if windows else _representative_panel_b_window(local_lakes, local_rts)
    if representative_extent is not None:
        return representative_extent

    bounds_list: list[list[float] | tuple[float, float, float, float]] = [focus_geom.bounds]
    if windows:
        bounds_list.extend([(extent[0], extent[2], extent[1], extent[3]) for extent in (spec["crop_extent"] for spec in windows)])
    if not local_lakes.empty:
        bounds_list.append(local_lakes.total_bounds)
    if not local_rts.empty:
        bounds_list.append(local_rts.total_bounds)
    return _merge_bounds(bounds_list, xpad=10_000.0, ypad=10_000.0)


def draw_meteoro_sites(ax, sites: gpd.GeoDataFrame, *, extent: list[float]) -> None:
    panel_geom = box(extent[0], extent[2], extent[1], extent[3])
    sites = sites.loc[sites.geometry.intersects(panel_geom)].copy()
    if sites.empty:
        return

    ax.scatter(
        sites.geometry.x.to_numpy(dtype=float),
        sites.geometry.y.to_numpy(dtype=float),
        s=26,
        marker="s",
        color=MET_SITE_COLOR,
        edgecolors="white",
        linewidths=0.6,
        zorder=6.5,
    )

    for site in sites.itertuples(index=False):
        label = str(site.site_label).strip()
        dx, dy, ha = STATION_LABEL_SPECS.get(label, (5.0, 3.0, "left"))
        text = ax.annotate(
            label,
            xy=(float(site.geometry.x), float(site.geometry.y)),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=FONT["annotation"] - 0.3,
            color=MET_SITE_COLOR,
            zorder=6.8,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])


def compute_main_extent(layers: dict[str, gpd.GeoDataFrame]) -> list[float]:
    return _extent_from_bounds(layers["qtp_aoi"].total_bounds, xpad=90_000.0, ypad=80_000.0)


BUFFER_EXTENT_LONLAT = dict(lon_min=91.4, lon_max=93.6, lat_min=33.2, lat_max=35.6)


def compute_buffer_extent() -> list[float]:
    return lonlat_extent_to_mercator(**BUFFER_EXTENT_LONLAT)


def draw_main_panel(ax, *, layers: dict[str, gpd.GeoDataFrame], extent: list[float]) -> None:
    qtp_aoi = layers["qtp_aoi"]
    qtec_aoi = layers["qtec_aoi"]
    qtp_geom = _geometry_union(qtp_aoi).buffer(0)

    layer_alpha = 0.70
    has_hillshade = draw_esri_hillshade(ax, extent=extent, zoom=5, alpha=1.0, zorder=-1)
    qtp_fill = (1.0, 1.0, 1.0, 0.0) if has_hillshade else QTP_FACE
    qtp_aoi.plot(ax=ax, facecolor=qtp_fill, edgecolor="0.72", linewidth=0.9, alpha=layer_alpha, zorder=0)
    draw_pf_background(ax, extent=extent, clip_geom=qtp_geom, width=2100, alpha=layer_alpha)
    if has_hillshade:
        draw_esri_hillshade(ax, extent=extent, zoom=5, alpha=0.30, zorder=2.6)

    frame_a = _clip_to_geometry(layers["frame_a"], qtp_geom)
    frame_d = _clip_to_geometry(layers["frame_d"], qtp_geom)
    if not frame_a.empty:
        frame_a.plot(ax=ax, facecolor="none", edgecolor=S1A_EDGE, linewidth=0.8, linestyle=(0, FRAME_DASHES), alpha=layer_alpha, zorder=3)
    if not frame_d.empty:
        frame_d.plot(ax=ax, facecolor="none", edgecolor=S1D_EDGE, linewidth=0.8, linestyle=(0, FRAME_DASHES), alpha=layer_alpha, zorder=3)

    railway = _clip_to_geometry(layers["railway"], qtp_geom)
    plot_dashed_railway(ax, railway, linewidth=1.9, halo_width=4.0, zorder=4)
    qtec_aoi.plot(ax=ax, facecolor="none", edgecolor=QTEC_COLOR, linewidth=2.0, alpha=layer_alpha, zorder=5)
    qtp_aoi.plot(ax=ax, facecolor="none", edgecolor="0.60", linewidth=0.9, alpha=layer_alpha, zorder=5.5)
    draw_meteoro_sites(ax, layers["meteoro"], extent=extent)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    apply_lonlat_axes(ax, extent, show_grid=False, xlabel="Longitude", ylabel="Latitude")
    ax.set_title("Corridor study area", fontsize=FONT["title"], pad=6)

    to_ll, from_ll = mercator_transformers()
    sy = extent[2] + 0.265 * (extent[3] - extent[2])
    _, ref_lat = to_ll.transform(0.5 * (extent[0] + extent[1]), sy)
    sx = extent[0] + 0.055 * (extent[1] - extent[0])
    add_scalebar_projected(ax, length_km=500.0, x=sx, y=sy, ref_lat=ref_lat)
    add_north_arrow(ax, x=0.935, y=0.955, size=0.10)

    legend = ax.legend(
        handles=[
            Patch(facecolor=PF_FACE, edgecolor="none", label="Permafrost"),
            Patch(facecolor=NPF_FACE, edgecolor="none", label="Non-permafrost"),
            Patch(facecolor="none", edgecolor=S1A_EDGE, linewidth=1.0, linestyle=(0, FRAME_DASHES), label="Ascending frames"),
            Patch(facecolor="none", edgecolor=S1D_EDGE, linewidth=1.0, linestyle=(0, FRAME_DASHES), label="Descending frames"),
            Line2D([0], [0], color=RAIL_COLOR, linewidth=1.8, linestyle=(0, RAIL_DASHES), label="Qinghai-Tibet railway"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.006),
        ncol=1,
        frameon=True,
        fancybox=False,
        facecolor="white",
        framealpha=0.50,
        fontsize=FONT["annotation"] - 0.2,
        handlelength=2.0,
    )
    style_translucent_legend(legend)


def draw_buffer_panel(ax, *, layers: dict[str, gpd.GeoDataFrame], windows: list[dict[str, object]], extent: list[float]) -> None:
    panel_geom = box(extent[0], extent[2], extent[1], extent[3])
    qtp_geom = _geometry_union(layers["qtp_aoi"]).buffer(0)
    railway_clip = _clip_to_geometry(layers["railway"], panel_geom)
    buffer_geom = _geometry_union(layers["railway"]).buffer(5_000.0).intersection(panel_geom)
    buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs=WEB_MERCATOR)

    lakes = layers["lakes"].copy()
    lakes.geometry = lakes.geometry.centroid
    lakes = lakes.loc[lakes.geometry.intersects(buffer_geom)].copy()

    rts = layers["rts"].copy()
    rts.geometry = rts.geometry.centroid
    rts = rts.loc[rts.geometry.intersects(buffer_geom)].copy()

    layers["qtp_aoi"].plot(ax=ax, facecolor=QTP_FACE, edgecolor="none", linewidth=0.0, zorder=0)
    draw_pf_background(ax, extent=extent, clip_geom=None, width=1200, alpha=0.96)
    if not buffer_gdf.empty and not buffer_geom.is_empty:
        buffer_gdf.plot(ax=ax, facecolor=BUFFER_FACE, edgecolor=BUFFER_EDGE, linewidth=1.0, alpha=0.24, zorder=3)
    plot_dashed_railway(ax, railway_clip, linewidth=1.8, halo_width=3.8, zorder=5)
    draw_meteoro_sites(ax, layers["meteoro"], extent=extent)

    if not lakes.empty:
        ax.scatter(
            lakes.geometry.x.to_numpy(dtype=float),
            lakes.geometry.y.to_numpy(dtype=float),
            s=20,
            color=TL_COLOR,
            alpha=0.70,
            linewidths=0.16,
            edgecolors="white",
            zorder=6,
        )
    if not rts.empty:
        ax.scatter(
            rts.geometry.x.to_numpy(dtype=float),
            rts.geometry.y.to_numpy(dtype=float),
            s=42,
            color=RTS_COLOR,
            alpha=0.80,
            marker="^",
            linewidths=0.18,
            edgecolors="white",
            zorder=7,
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    apply_lonlat_axes(
        ax,
        extent,
        show_grid=False,
        xlabel="Longitude",
        ylabel="Latitude",
        lon_ticks_deg=[91.5, 92.0, 92.5, 93.0, 93.5],
        lat_ticks_deg=[34.0, 35.0],
        lat_formatter=lambda lat: f"{int(round(float(lat)))}N",
    )
    ax.xaxis.set_label_coords(0.5, -0.070)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_title("5 km railway buffer", fontsize=FONT["title"], pad=6)

    to_ll, _ = mercator_transformers()
    sy_sb = extent[2] + 0.255 * (extent[3] - extent[2])
    right_x = extent[1] - 0.055 * (extent[1] - extent[0])
    _, ref_lat = to_ll.transform(right_x, sy_sb)
    scale_length_proj = 50_000.0 / max(0.15, math.cos(math.radians(ref_lat)))
    sx = right_x - scale_length_proj
    add_scalebar_projected(ax, length_km=50.0, x=sx, y=sy_sb, ref_lat=ref_lat)
    add_north_arrow(ax, x=0.08, y=0.89, size=0.10)

    legend_b = ax.legend(
        handles=[
            Patch(facecolor=BUFFER_FACE, edgecolor=BUFFER_EDGE, label="5 km railway buffer"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TL_COLOR, markeredgecolor="white", markersize=6, label="Thermokarst lakes"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=RTS_COLOR, markeredgecolor="white", markersize=6, label="Thaw slumps"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.03),
        ncol=1,
        frameon=True,
        fancybox=False,
        facecolor="white",
        framealpha=0.50,
        fontsize=FONT["annotation"] - 0.2,
        handlelength=2.0,
    )
    style_translucent_legend(legend_b)

    for spec in windows:
        crop = [float(v) for v in spec["crop_extent"]]
        rect = Rectangle(
            (crop[0], crop[2]),
            crop[1] - crop[0],
            crop[3] - crop[2],
            fill=False,
            edgecolor=str(spec["frame_color"]),
            linewidth=1.6,
            zorder=8.4,
        )
        ax.add_patch(rect)
        text = ax.text(
            crop[0] + 0.015 * (extent[1] - extent[0]),
            crop[3] - 0.020 * (extent[3] - extent[2]),
            str(spec["panel_label"]),
            fontsize=FONT["annotation"] + 0.2,
            fontweight="bold",
            color=str(spec["frame_color"]),
            zorder=8.6,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white")])


def draw_window_panel(ax, *, layers: dict[str, gpd.GeoDataFrame], spec: dict[str, object]) -> None:
    image, extent = read_wayback_rgb(Path(spec["path"]), width=620, extent=[float(v) for v in spec["crop_extent"]])
    chip_geom = box(extent[0], extent[2], extent[1], extent[3])
    ax.imshow(image, extent=extent, origin="upper", zorder=0)

    railway = _clip_to_geometry(layers["railway"], chip_geom)
    if not railway.empty:
        plot_dashed_railway(ax, railway, linewidth=1.3, halo_width=2.8, zorder=2)

    if spec["kind"] == "tl":
        lake_polys = _clip_to_geometry(layers["lakes"], chip_geom)
        if not lake_polys.empty:
            lake_polys.plot(
                ax=ax,
                facecolor="none",
                edgecolor=TL_BOUNDARY_COLOR,
                linewidth=1.05,
                alpha=0.95,
                zorder=3.0,
            )
            lake_pts = lake_polys.copy()
            lake_pts.geometry = lake_pts.geometry.centroid
            ax.scatter(
                lake_pts.geometry.x.to_numpy(dtype=float),
                lake_pts.geometry.y.to_numpy(dtype=float),
                s=8,
                color=TL_BOUNDARY_COLOR,
                alpha=0.68,
                linewidths=0.15,
                edgecolors="white",
                zorder=3.2,
            )
    else:
        rts_poly = _clip_to_geometry(layers["rts"], chip_geom)
        if not rts_poly.empty:
            rts_poly.plot(ax=ax, facecolor="none", edgecolor=RTS_COLOR, linewidth=0.9, alpha=0.90, zorder=3)
            rts_pts = rts_poly.copy()
            rts_pts.geometry = rts_pts.geometry.centroid
            ax.scatter(
                rts_pts.geometry.x.to_numpy(dtype=float),
                rts_pts.geometry.y.to_numpy(dtype=float),
                s=34,
                color=RTS_COLOR,
                alpha=0.78,
                marker="^",
                linewidths=0.2,
                edgecolors="white",
                zorder=3.2,
            )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor(str(spec["frame_color"]))
    ax.set_title(str(spec["title"]), fontsize=FONT["title"], pad=6, color=str(spec["frame_color"]))

    to_ll, _ = mercator_transformers()
    _, ref_lat = to_ll.transform(0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3]))
    scale_km = float(spec.get("scalebar_km", 0.5 if str(spec.get("kind")) == "rts" else 1.0))
    sb_x = extent[0] + 0.05 * (extent[1] - extent[0])
    sb_y = extent[3] - 0.12 * (extent[3] - extent[2])
    add_scalebar_projected(ax, length_km=scale_km, x=sb_x, y=sb_y, ref_lat=ref_lat, color="white", halo=False)

    date_label = str(spec.get("date", "")).strip()
    if date_label:
        label = ax.text(
            extent[1] - 0.035 * (extent[1] - extent[0]),
            extent[3] - 0.045 * (extent[3] - extent[2]),
            date_label,
            ha="right",
            va="top",
            fontsize=FONT["annotation"] + 0.1,
            fontweight="bold",
            color="white",
            zorder=13,
        )
        label.set_path_effects([pe.withStroke(linewidth=2.4, foreground="0.08")])


def main() -> None:
    ensure_style()
    layers = load_layers()
    windows = prepare_windows(layers)

    extent_a = compute_main_extent(layers)
    extent_b = compute_buffer_extent()
    aspect_a = (extent_a[1] - extent_a[0]) / (extent_a[3] - extent_a[2])
    aspect_b = (extent_b[1] - extent_b[0]) / (extent_b[3] - extent_b[2])

    fig = plt.figure(figsize=(12.4, 8.9))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.82, 0.92],
        hspace=0.11,
    )
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[aspect_a, aspect_b], wspace=0.02)
    ax_main = fig.add_subplot(top[0, 0])
    ax_buffer = fig.add_subplot(top[0, 1])
    bottom = outer[1, 0].subgridspec(1, 3, wspace=0.09)

    draw_main_panel(ax_main, layers=layers, extent=extent_a)
    draw_buffer_panel(ax_buffer, layers=layers, windows=windows, extent=extent_b)

    for idx in range(3):
        ax = fig.add_subplot(bottom[0, idx])
        if idx < len(windows):
            draw_window_panel(ax, layers=layers, spec=windows[idx])
        else:
            _set_window_panel(ax)
            ax.text(0.5, 0.5, "Missing imagery", transform=ax.transAxes, ha="center", va="center")

    add_panel_label(ax_main, "A", x=-0.01, y=1.01)
    add_panel_label(ax_buffer, "B", x=-0.10, y=1.01)
    for idx, label in enumerate(("C", "D", "E")):
        add_panel_label(fig.axes[2 + idx], label, x=-0.03, y=1.01)

    # Force panels A and B to share the same rendered height after aspect boxing.
    fig.canvas.draw()
    pos_a = ax_main.get_position()
    pos_b = ax_buffer.get_position()
    fig_w, fig_h = fig.get_size_inches()
    fig_hw = fig_h / fig_w
    gap = pos_b.x0 - pos_a.x1
    width_budget = pos_b.x1 - pos_a.x0 - gap
    common_h = min(
        pos_a.height,
        pos_b.height,
        width_budget / (fig_hw * (aspect_a + aspect_b)),
    )
    width_a = common_h * fig_hw * aspect_a
    width_b = common_h * fig_hw * aspect_b
    common_y0 = min(pos_a.y0, pos_b.y0)
    ax_main.set_position([pos_a.x0, common_y0, width_a, common_h])
    ax_buffer.set_position([pos_b.x1 - width_b, common_y0, width_b, common_h])

    save_figure(fig, FIG_STEM)


if __name__ == "__main__":
    main()
