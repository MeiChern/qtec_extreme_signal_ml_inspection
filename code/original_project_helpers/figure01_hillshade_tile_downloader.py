#!/usr/bin/env python3
# Packaged reviewer inspection copy.
# Original source: pnas_nexus_submission/scripts/download_figure01_hillshade_tiles.py
# Renamed package path: code/original_project_helpers/figure01_hillshade_tile_downloader.py
# See README.md and REPRODUCIBILITY_LIMITS.md for package scope and rerun limits.

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from Figure01_study_area_overview import (
    ESRI_HILLSHADE_BASE_URL,
    PANEL_A_HILLSHADE_PAD_DEG,
    _esri_tile_cache_path,
    _fetch_esri_tile,
    compute_main_extent,
    esri_tile_ranges_for_bbox,
    load_layers,
    lonlat_bbox_for_extent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ESRI World Hillshade tiles covering Figure 1 panel A plus a lon/lat pad."
    )
    parser.add_argument("--zoom", type=int, default=5, help="ArcGIS XYZ tile zoom level.")
    parser.add_argument("--pad-deg", type=float, default=PANEL_A_HILLSHADE_PAD_DEG, help="Padding in degrees.")
    parser.add_argument("--overwrite", action="store_true", help="Refetch tiles even when they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the bbox and tile list without downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = load_layers()
    extent = compute_main_extent(layers)
    lon_min, lon_max, lat_min, lat_max = lonlat_bbox_for_extent(extent, pad_deg=args.pad_deg)
    x_tiles, y_tiles = esri_tile_ranges_for_bbox(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        zoom=args.zoom,
    )
    tile_ids = [(x, y) for y in y_tiles for x in x_tiles]

    print(f"Service: {ESRI_HILLSHADE_BASE_URL}")
    print(f"Panel A bbox + {args.pad_deg:g} deg: lon {lon_min:.8f} {lon_max:.8f}, lat {lat_min:.8f} {lat_max:.8f}")
    print(
        f"Zoom {args.zoom}: x {x_tiles.start}-{x_tiles.stop - 1}, "
        f"y {y_tiles.start}-{y_tiles.stop - 1}, total {len(tile_ids)} tiles"
    )

    if args.dry_run:
        for x_tile, y_tile in tile_ids:
            print(f"tile/{args.zoom}/{y_tile}/{x_tile} -> {_esri_tile_cache_path(x_tile, y_tile, args.zoom)}")
        return

    failures: list[tuple[int, int, Path]] = []
    for x_tile, y_tile in tile_ids:
        cache_path = _esri_tile_cache_path(x_tile, y_tile, args.zoom)
        if args.overwrite and cache_path.exists():
            cache_path.unlink()
        existed_before = cache_path.exists()
        tile = _fetch_esri_tile(x_tile, y_tile, args.zoom)
        if tile is None or not cache_path.exists():
            failures.append((x_tile, y_tile, cache_path))
            print(f"failed tile/{args.zoom}/{y_tile}/{x_tile} -> {cache_path}")
            continue
        status = "cached" if existed_before else "downloaded"
        print(f"{status} tile/{args.zoom}/{y_tile}/{x_tile} -> {cache_path}")

    if failures:
        raise SystemExit(f"{len(failures)} tile downloads failed")


if __name__ == "__main__":
    main()
