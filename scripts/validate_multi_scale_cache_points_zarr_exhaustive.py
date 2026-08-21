"""Developer-only exhaustive validation for a staged Zarr points cache.

This script is intentionally outside the installed ``napari_harpy`` package.
Normal cache publication uses the compact path-only validator in
``writer.staging_validation``. Run this tool only for format or algorithm
changes, release qualification, or investigation of suspected corruption.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CacheAttributes
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import TARGET_POINTS_PER_BUCKET
from napari_harpy.core.multi_scale_cache_points_zarr.models import _TileDescriptor
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _BucketReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import _validate_bucket
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.writer.exact import (
    _read_and_annotate_row_group,
    _source_row_group_read_specs,
)
from napari_harpy.core.multi_scale_cache_points_zarr.writer.staging_validation import (
    _ManifestInventory,
    _read_manifest_inventory,
    _validate_staged_cache,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively validate staged Zarr point payloads and optional source equivalence."
    )
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--temporary-directory-root", type=Path, required=True)
    parser.add_argument("--source-spatialdata-path", type=Path)
    parser.add_argument("--points-name")
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--value", default="gene")
    args = parser.parse_args()
    if (args.source_spatialdata_path is None) != (args.points_name is None):
        parser.error("--source-spatialdata-path and --points-name must be supplied together.")
    return args


def _validate_cache_exhaustive(
    cache_root: Path,
    *,
    source: ParquetPointsSource | None,
    temporary_directory_root: Path,
) -> None:
    """Run complete payload, identity, nesting, and optional source checks."""
    _validate_staged_cache(cache_root)
    if not isinstance(temporary_directory_root, Path) or not temporary_directory_root.is_dir():
        raise ValueError("`temporary_directory_root` must be an existing pathlib.Path directory.")
    cache_resolved = cache_root.resolve()
    temporary_resolved = temporary_directory_root.resolve()
    if (
        cache_resolved == temporary_resolved
        or cache_resolved in temporary_resolved.parents
        or temporary_resolved in cache_resolved.parents
    ):
        raise ValueError("Cache and exhaustive-validation temporary roots must be separate directory trees.")

    with _CatalogReader(cache_root) as reader:
        attributes = reader.attributes
        inventory = _read_manifest_inventory(reader)
    for buckets in inventory.buckets_by_level:
        for bucket in buckets:
            result = _validate_bucket(cache_root, level=bucket.level, bucket_id=bucket.bucket_id)
            if result.tile_descriptors != bucket.descriptors:
                raise ValueError("Exhaustive bucket descriptors disagree with the persisted manifest.")

    with tempfile.TemporaryDirectory(prefix="harpy-zarr-validation-", dir=temporary_directory_root) as scratch:
        scratch_root = Path(scratch)
        _validate_exact_point_ids_and_coordinates(cache_root, attributes, inventory, scratch_root)
        _validate_cross_level_payloads(cache_root, attributes, inventory, scratch_root)
        if source is not None:
            _validate_exact_against_source(cache_root, attributes, inventory, source, scratch_root)


def _validate_exact_point_ids_and_coordinates(
    cache_root: Path,
    attributes: _CacheAttributes,
    inventory: _ManifestInventory,
    scratch_root: Path,
) -> None:
    """Prove Exact point-ID coverage with a disk-backed external bitmap."""
    row_count = attributes.source.row_count
    seen_path = scratch_root / "exact-seen.u1"
    seen = np.memmap(seen_path, mode="w+", dtype=np.uint8, shape=(row_count,))
    seen[:] = 0
    for descriptor, payload in _iter_level_payloads(cache_root, inventory, level=0):
        _validate_tile_relative_coordinates(payload.x_rel, payload.y_rel, attributes.levels[0].tile_size)
        point_ids = payload.point_id
        if int(point_ids.max()) >= row_count:
            raise ValueError("Exact point ID lies outside the canonical source-row interval.")
        if len(np.unique(point_ids)) != len(point_ids) or bool(seen[point_ids].any()):
            raise ValueError("Exact point IDs are not globally unique.")
        seen[point_ids] = 1
        del descriptor, payload
    if not bool(seen.all()):
        raise ValueError("Exact point IDs do not cover every canonical source row.")
    del seen
    seen_path.unlink()


def _validate_cross_level_payloads(
    cache_root: Path,
    attributes: _CacheAttributes,
    inventory: _ManifestInventory,
    scratch_root: Path,
) -> None:
    """Prove immediate-coarser payloads retain unchanged finer-level points."""
    row_count = attributes.source.row_count
    for coarser_level in range(1, len(attributes.levels)):
        present_path = scratch_root / f"level-{coarser_level - 1}-present.u1"
        values_path = scratch_root / f"level-{coarser_level - 1}-value.u4"
        x_path = scratch_root / f"level-{coarser_level - 1}-x.f8"
        y_path = scratch_root / f"level-{coarser_level - 1}-y.f8"
        coarser_seen_path = scratch_root / f"level-{coarser_level}-seen.u1"
        present = np.memmap(present_path, mode="w+", dtype=np.uint8, shape=(row_count,))
        values = np.memmap(values_path, mode="w+", dtype=np.uint32, shape=(row_count,))
        x_source = np.memmap(x_path, mode="w+", dtype=np.float64, shape=(row_count,))
        y_source = np.memmap(y_path, mode="w+", dtype=np.float64, shape=(row_count,))
        coarser_seen = np.memmap(coarser_seen_path, mode="w+", dtype=np.uint8, shape=(row_count,))
        present[:] = 0
        coarser_seen[:] = 0
        finer_metadata = attributes.levels[coarser_level - 1]
        for descriptor, payload in _iter_level_payloads(cache_root, inventory, level=coarser_level - 1):
            _validate_tile_relative_coordinates(payload.x_rel, payload.y_rel, finer_metadata.tile_size)
            ids = payload.point_id
            if int(ids.max()) >= row_count or len(np.unique(ids)) != len(ids) or bool(present[ids].any()):
                raise ValueError("A finer cache level contains invalid or duplicate point IDs.")
            present[ids] = 1
            values[ids] = payload.value_id
            x_source[ids] = attributes.geometry.x_origin + descriptor.tile_x * finer_metadata.tile_size + payload.x_rel
            y_source[ids] = attributes.geometry.y_origin + descriptor.tile_y * finer_metadata.tile_size + payload.y_rel

        coarser_metadata = attributes.levels[coarser_level]
        tolerance = _coordinate_tolerance(coarser_metadata.tile_size)
        for descriptor, payload in _iter_level_payloads(cache_root, inventory, level=coarser_level):
            _validate_tile_relative_coordinates(payload.x_rel, payload.y_rel, coarser_metadata.tile_size)
            ids = payload.point_id
            if (
                int(ids.max()) >= row_count
                or len(np.unique(ids)) != len(ids)
                or bool(coarser_seen[ids].any())
                or not bool(present[ids].all())
            ):
                raise ValueError("A coarser cache level contains an invalid, duplicate, or absent point ID.")
            coarser_seen[ids] = 1
            observed_x = attributes.geometry.x_origin + descriptor.tile_x * coarser_metadata.tile_size + payload.x_rel
            observed_y = attributes.geometry.y_origin + descriptor.tile_y * coarser_metadata.tile_size + payload.y_rel
            if not (
                np.array_equal(payload.value_id, values[ids])
                and np.allclose(observed_x, x_source[ids], rtol=0.0, atol=tolerance)
                and np.allclose(observed_y, y_source[ids], rtol=0.0, atol=tolerance)
            ):
                raise ValueError("A coarser cache level changed retained point values or coordinates.")
        del present, values, x_source, y_source, coarser_seen
        for path in (present_path, values_path, x_path, y_path, coarser_seen_path):
            path.unlink()


def _validate_exact_against_source(
    cache_root: Path,
    attributes: _CacheAttributes,
    inventory: _ManifestInventory,
    source: ParquetPointsSource,
    scratch_root: Path,
) -> None:
    """Freshly validate and compare the canonical source with Exact by point ID."""
    validated = validate_parquet_points_source(source)
    if (
        validated.source_signature != attributes.source.signature
        or validated.row_count != attributes.source.row_count
        or tuple(validated.value_table["value"].to_pylist()) != attributes.value_names
    ):
        raise ValueError("Canonical source identity no longer matches the staged cache.")

    row_count = attributes.source.row_count
    value_path = scratch_root / "exact-value.u4"
    x_path = scratch_root / "exact-x.f8"
    y_path = scratch_root / "exact-y.f8"
    cache_value = np.memmap(value_path, mode="w+", dtype=np.uint32, shape=(row_count,))
    cache_x = np.memmap(x_path, mode="w+", dtype=np.float64, shape=(row_count,))
    cache_y = np.memmap(y_path, mode="w+", dtype=np.float64, shape=(row_count,))
    exact = attributes.levels[0]
    for descriptor, payload in _iter_level_payloads(cache_root, inventory, level=0):
        ids = payload.point_id
        cache_value[ids] = payload.value_id
        cache_x[ids] = attributes.geometry.x_origin + descriptor.tile_x * exact.tile_size + payload.x_rel
        cache_y[ids] = attributes.geometry.y_origin + descriptor.tile_y * exact.tile_size + payload.y_rel

    value_labels = tuple(validated.value_table["value"].to_pylist())
    bucket_count = max(
        1,
        math.ceil(exact.point_count_upper_bound / TARGET_POINTS_PER_BUCKET),
    )
    for spec in _source_row_group_read_specs(validated):
        frame = _read_and_annotate_row_group(
            spec,
            source_root=validated.source.parquet_path,
            x_column=validated.source.columns.x,
            y_column=validated.source.columns.y,
            value_column=validated.source.columns.value,
            x_origin=attributes.geometry.x_origin,
            y_origin=attributes.geometry.y_origin,
            tile_size=exact.tile_size,
            grid_width=exact.grid_width,
            grid_height=exact.grid_height,
            bucket_count=bucket_count,
            value_labels_by_id=value_labels,
            validated_row_count=validated.row_count,
        )
        ids = frame["point_id"].to_numpy(dtype=np.uint64, copy=False)
        source_x = (
            attributes.geometry.x_origin
            + frame["tile_x"].to_numpy(dtype=np.uint64, copy=False) * exact.tile_size
            + frame["x_rel"].to_numpy(dtype=np.float32, copy=False)
        )
        source_y = (
            attributes.geometry.y_origin
            + frame["tile_y"].to_numpy(dtype=np.uint64, copy=False) * exact.tile_size
            + frame["y_rel"].to_numpy(dtype=np.float32, copy=False)
        )
        if not (
            np.array_equal(cache_value[ids], frame["value_id"].to_numpy(dtype=np.uint32, copy=False))
            and np.allclose(cache_x[ids], source_x, rtol=0.0, atol=_coordinate_tolerance(exact.tile_size))
            and np.allclose(cache_y[ids], source_y, rtol=0.0, atol=_coordinate_tolerance(exact.tile_size))
        ):
            raise ValueError("Exact payload does not match canonical source rows.")
    del cache_value, cache_x, cache_y
    for path in (value_path, x_path, y_path):
        path.unlink()


def _iter_level_payloads(
    cache_root: Path,
    inventory: _ManifestInventory,
    *,
    level: int,
) -> Iterator[tuple[_TileDescriptor, _PointPayload]]:
    """Yield complete tile payloads while keeping only one bucket handle open."""
    for bucket in inventory.buckets_by_level[level]:
        with _BucketReader(cache_root, level=level, bucket_id=bucket.bucket_id) as reader:
            for descriptor in bucket.descriptors:
                yield descriptor, reader.read_construction_payload(descriptor)


def _validate_tile_relative_coordinates(x_rel: np.ndarray, y_rel: np.ndarray, tile_size: int) -> None:
    tolerance = _coordinate_tolerance(tile_size)
    if not (
        bool(np.isfinite(x_rel).all())
        and bool(np.isfinite(y_rel).all())
        and not bool((x_rel < 0).any())
        and not bool((y_rel < 0).any())
        and not bool((x_rel > tile_size + tolerance).any())
        and not bool((y_rel > tile_size + tolerance).any())
    ):
        raise ValueError("Tile-relative point coordinate lies outside its logical tile.")


def _coordinate_tolerance(tile_size: int) -> float:
    return float(np.spacing(np.float32(tile_size)))


def main() -> None:
    """Run the developer-only exhaustive validator from command-line arguments."""
    args = _parse_args()
    source = None
    if args.source_spatialdata_path is not None:
        source = ParquetPointsSource(
            spatialdata_path=args.source_spatialdata_path,
            points_name=args.points_name,
            columns=PointColumnSelection(x=args.x, y=args.y, value=args.value),
        )
    started = perf_counter()
    _validate_cache_exhaustive(
        args.cache_root,
        source=source,
        temporary_directory_root=args.temporary_directory_root,
    )
    print(f"Exhaustive validation succeeded in {perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
