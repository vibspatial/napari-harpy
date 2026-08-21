from __future__ import annotations

from itertools import chain
from pathlib import Path

import numpy as np

from napari_harpy import __version__ as napari_harpy_version
from napari_harpy.core.multi_scale_cache_points_zarr.build_plan import _PointsCacheBuildPlan
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_STAGING,
    _BuildMetadata,
    _CacheAttributes,
    _CatalogMetadata,
    _CatalogWriteSettings,
    _GeometryMetadata,
    _LevelMetadata,
    _SourceMetadata,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import (
    BUCKET_HASH_METHOD,
    TARGET_POINTS_PER_BUCKET,
    _bucket_count_for_level,
)
from napari_harpy.core.multi_scale_cache_points_zarr.sampling import (
    SAMPLED_TILE_MICROGRID_EDGE,
    SAMPLING_METHOD,
    SAMPLING_SEED,
)
from napari_harpy.core.multi_scale_cache_points_zarr.source.models import ValidatedPointsSource
from napari_harpy.core.multi_scale_cache_points_zarr.source.signature import _normalized_arrow_type
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import (
    _iter_bucket_range_batches,
    _read_bucket_storage_settings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_writer import (
    _CatalogWriter,
    _ValueTilesWriteSummary,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _LevelWriteResult,
    _ZarrWriteSettings,
)


def _write_staged_cache_catalog(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    level_results: tuple[_LevelWriteResult, ...],
    *,
    staging_root: Path,
    cache_generation_id: str,
    settings: _CatalogWriteSettings,
) -> None:
    """Write and reconcile one complete Zarr-only cache catalog.

    The level writers have already finalized every point bucket below the
    unpublished staging root. This operation adds the cache root and ancestor
    groups, values, tile manifest, and the derived value-to-tile inverted index.
    It reads compact bucket indexes only and never decodes point payload arrays
    or canonical source rows.
    """
    if not isinstance(validated, ValidatedPointsSource):
        raise ValueError("`validated` must be ValidatedPointsSource.")
    if not isinstance(plan, _PointsCacheBuildPlan):
        raise ValueError("`plan` must be _PointsCacheBuildPlan.")
    if not isinstance(settings, _CatalogWriteSettings):
        raise ValueError("`settings` must be _CatalogWriteSettings.")
    _require_existing_staging_root(staging_root)
    _require_level_results_match_plan(validated, plan, level_results)
    _require_catalog_targets_absent(staging_root)
    _require_bucket_inventory_matches_results(staging_root, level_results)

    first_bucket = level_results[0].buckets[0]
    zarr_settings = _read_bucket_storage_settings(staging_root, first_bucket)
    for result in level_results:
        for bucket in result.buckets:
            if _read_bucket_storage_settings(staging_root, bucket) != zarr_settings:
                raise ValueError("Every cache bucket must use one common physical settings profile.")

    value_names = tuple(validated.value_table["value"].to_pylist())
    value_counts = np.ascontiguousarray(
        validated.value_table["n_points"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.uint64,
    )
    if len(value_names) != len(value_counts) or int(value_counts.sum(dtype=np.uint64)) != validated.row_count:
        raise ValueError("Validated value labels and counts do not reconcile to source rows.")

    (
        level_indptr,
        bucket_id,
        bucket_tile_index,
        tile_x,
        tile_y,
        manifest_n_points,
        bucket_manifest_indexes,
    ) = _build_manifest_arrays(plan, level_results)
    manifest_row_count = len(bucket_id)
    value_tile_row_count = sum(result.range_count for result in level_results)
    catalog_metadata = _CatalogMetadata(
        value_count=len(value_names),
        level_count=len(level_results),
        manifest_row_count=manifest_row_count,
        value_tile_row_count=value_tile_row_count,
        settings=settings,
    )

    with _CatalogWriter(
        staging_root,
        level_count=len(level_results),
        value_count=len(value_names),
        manifest_row_count=manifest_row_count,
        value_tile_row_count=value_tile_row_count,
        zarr_settings=zarr_settings,
        catalog_settings=settings,
    ) as writer:
        writer.write_value_counts(value_counts)
        writer.write_manifest(
            level_indptr=level_indptr,
            bucket_id=bucket_id,
            bucket_tile_index=bucket_tile_index,
            tile_x=tile_x,
            tile_y=tile_y,
            n_points=manifest_n_points,
        )
        # Keep one lazy stream per level while concatenating its bucket streams;
        # only the currently consumed bucket store is opened by the reader.
        batches_by_level = tuple(
            chain.from_iterable(
                _iter_bucket_range_batches(
                    staging_root,
                    bucket,
                    bucket_manifest_indexes[(bucket.level, bucket.bucket_id)],
                    batch_rows=settings.value_tile_chunk_rows,
                    expected_settings=zarr_settings,
                )
                for bucket in result.buckets
            )
            for result in level_results
        )
        summary = writer.write_value_tiles_by_level(
            batches_by_level,
            level_indptr=level_indptr,
            expected_level_row_counts=tuple(result.range_count for result in level_results),
            value_count=len(value_names),
            output_batch_rows=settings.value_tile_chunk_rows,
        )
        _reconcile_catalog(
            validated,
            level_results,
            value_counts=value_counts,
            manifest_n_points=manifest_n_points,
            value_tile_row_count=value_tile_row_count,
            summary=summary,
        )
        attributes = _build_cache_attributes(
            validated,
            plan,
            level_results,
            cache_generation_id=cache_generation_id,
            zarr_settings=zarr_settings,
            value_names=value_names,
            catalog_metadata=catalog_metadata,
        )
        writer.finalize(attributes)


def _build_manifest_arrays(
    plan: _PointsCacheBuildPlan,
    level_results: tuple[_LevelWriteResult, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], np.ndarray],
]:
    """Flatten finalized tile descriptors into the cache-wide manifest.

    Manifest rows follow ``(level, tile_y, tile_x)`` order, while every bucket's
    compact tile and sparse-range arrays use ``bucket_tile_index`` order. This
    function constructs both the parallel persisted manifest arrays and a
    transient dense translation between those two orderings.

    Parameters
    ----------
    plan
        Complete build plan whose level grids bound every manifest tile.
    level_results
        Finalized level results in serialized level order. Within each level,
        tile descriptors are flattened into global ``(tile_y, tile_x)`` order.

    Returns
    -------
    level_indptr
        Manifest row boundaries for each level. Level ``L`` occupies
        ``level_indptr[L]:level_indptr[L + 1]``.
    bucket_id, bucket_tile_index, tile_x, tile_y, n_points
        Parallel manifest arrays describing the physical address, logical grid
        position, and stored point total of every nonempty tile.
    bucket_manifest_indexes
        Transient mapping from ``(level, bucket_id)`` to a dense array whose
        position is ``bucket_tile_index`` and whose value is the corresponding
        cache-wide manifest row. In other words::

            bucket_manifest_indexes[(level, bucket_id)][bucket_tile_index]
                == manifest_index

    Examples
    --------
    Suppose level 0 contains three tiles in global spatial order, but hashing
    distributed them across two buckets. The shared array position is the
    implicit global manifest index::

        implicit       bucket_   bucket_tile_   tile_   tile_   n_
        manifest_index id        index          x       y       points
        ----------------------------------------------------------------
        0              1         0              0       0       120
        1              0         0              1       0        85
        2              1         1              2       0       103

    The transient translations are then::

        bucket_manifest_indexes[(0, 0)] = [1]
        #                         │  │     └─ local tile 0 maps to manifest row 1
        #                         │  └─────── bucket 0
        #                         └────────── level 0

        bucket_manifest_indexes[(0, 1)] = [0, 2]
        #                         │  │     │  └─ local tile 1 maps to manifest row 2
        #                         │  │     └──── local tile 0 maps to manifest row 0
        #                         │  └────────── bucket 1
        #                         └───────────── level 0

    A sparse range belonging to bucket 1's local tile 1 can therefore be
    recorded in ``value_tiles`` as global manifest row 2.
    """
    manifest_row_count = sum(result.tile_count for result in level_results)
    level_indptr = np.empty(len(level_results) + 1, dtype=np.uint64)
    level_indptr[0] = 0
    bucket_id = np.empty(manifest_row_count, dtype=np.uint32)  # Physical bucket containing tile ``i``.
    bucket_tile_index = np.empty(manifest_row_count, dtype=np.uint32)  # Tile ``i``'s local bucket index.
    tile_x = np.empty(manifest_row_count, dtype=np.uint32)  # Tile ``i``'s logical x-grid position.
    tile_y = np.empty(manifest_row_count, dtype=np.uint32)  # Tile ``i``'s logical y-grid position.
    n_points = np.empty(manifest_row_count, dtype=np.uint64)  # Total stored points for tile ``i``.
    address_to_manifest_row: dict[tuple[int, int, int], int] = {}
    cursor = 0
    for level_plan, result in zip(plan.levels, level_results, strict=True):
        descriptors = result.tile_descriptors
        for descriptor in descriptors:
            if descriptor.tile_x >= level_plan.grid_width or descriptor.tile_y >= level_plan.grid_height:
                raise ValueError("Manifest tile lies outside its planned level grid.")
            address = (descriptor.level, descriptor.bucket_id, descriptor.bucket_tile_index)
            if address in address_to_manifest_row:
                raise ValueError("Manifest contains a duplicate bucket-local tile address.")
            address_to_manifest_row[address] = cursor
            bucket_id[cursor] = descriptor.bucket_id
            bucket_tile_index[cursor] = descriptor.bucket_tile_index
            tile_x[cursor] = descriptor.tile_x
            tile_y[cursor] = descriptor.tile_y
            n_points[cursor] = descriptor.n_points
            cursor += 1
        level_indptr[level_plan.level + 1] = cursor
    if cursor != manifest_row_count:
        raise RuntimeError("Manifest construction did not fill its declared rows.")

    bucket_manifest_indexes: dict[tuple[int, int], np.ndarray] = {}
    for result in level_results:
        for bucket in result.buckets:
            indexes = np.fromiter(
                (
                    address_to_manifest_row[(descriptor.level, descriptor.bucket_id, descriptor.bucket_tile_index)]
                    for descriptor in bucket.tile_descriptors
                ),
                dtype=np.uint64,
                count=len(bucket.tile_descriptors),
            )
            bucket_manifest_indexes[(bucket.level, bucket.bucket_id)] = np.ascontiguousarray(indexes)
    return (
        np.ascontiguousarray(level_indptr),
        np.ascontiguousarray(bucket_id),
        np.ascontiguousarray(bucket_tile_index),
        np.ascontiguousarray(tile_x),
        np.ascontiguousarray(tile_y),
        np.ascontiguousarray(n_points),
        bucket_manifest_indexes,
    )


def _build_cache_attributes(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    level_results: tuple[_LevelWriteResult, ...],
    *,
    cache_generation_id: str,
    zarr_settings: _ZarrWriteSettings,
    value_names: tuple[str, ...],
    catalog_metadata: _CatalogMetadata,
) -> _CacheAttributes:
    columns = validated.source.columns
    selected_schema = tuple(
        {
            "role": role,
            "name": name,
            "nullable": validated.selected_schema.field(name).nullable,
            "type": _normalized_arrow_type(validated.selected_schema.field(name).type),
        }
        for role, name in (("x", columns.x), ("y", columns.y), ("value", columns.value))
    )
    source = _SourceMetadata(
        points_name=validated.source.points_name,
        element_path=validated.source.element_path,
        row_count=validated.row_count,
        x_column=columns.x,
        y_column=columns.y,
        value_column=columns.value,
        selected_schema=selected_schema,
        signature_method=validated.source_signature_method,
        signature=validated.source_signature,
        value_normalization_method=validated.value_normalization_method,
        point_id_policy=validated.point_id_policy,
    )
    bounds = validated.bounds
    geometry = _GeometryMetadata(
        x_origin=float(plan.x_origin),
        y_origin=float(plan.y_origin),
        x_min=float(bounds.x_min),
        x_max=float(bounds.x_max),
        y_min=float(bounds.y_min),
        y_max=float(bounds.y_max),
    )
    build = _BuildMetadata(
        leaf_tile_size=plan.leaf_tile_size,
        overview_point_budget=plan.overview_point_budget,
        target_points_per_bucket=TARGET_POINTS_PER_BUCKET,
        bucket_hash_method=BUCKET_HASH_METHOD,
        sampling_method=SAMPLING_METHOD,
        sampling_seed=SAMPLING_SEED,
        sampling_microgrid_edge=SAMPLED_TILE_MICROGRID_EDGE,
    )
    levels = tuple(
        _LevelMetadata(
            level=level_plan.level,
            kind=level_plan.kind.value,
            tile_size=level_plan.tile_size,
            grid_width=level_plan.grid_width,
            grid_height=level_plan.grid_height,
            max_points_per_tile=level_plan.max_points_per_tile,
            point_count_upper_bound=level_plan.point_count_upper_bound,
            bucket_count=result.bucket_count,
            tile_count=result.tile_count,
            point_count=result.point_count,
            range_count=result.range_count,
            relative_directory=level_plan.relative_directory,
        )
        for level_plan, result in zip(plan.levels, level_results, strict=True)
    )
    return _CacheAttributes(
        cache_generation_id=cache_generation_id,
        publication_state=PUBLICATION_STATE_STAGING,
        created_by_version=napari_harpy_version,
        zarr_settings=zarr_settings,
        source=source,
        geometry=geometry,
        build=build,
        levels=levels,
        value_names=value_names,
        catalog=catalog_metadata,
    )


def _reconcile_catalog(
    validated: ValidatedPointsSource,
    level_results: tuple[_LevelWriteResult, ...],
    *,
    value_counts: np.ndarray,
    manifest_n_points: np.ndarray,
    value_tile_row_count: int,
    summary: _ValueTilesWriteSummary,
) -> None:
    if not isinstance(summary, _ValueTilesWriteSummary):
        raise ValueError("`summary` must be _ValueTilesWriteSummary.")
    expected_level_counts = np.asarray([result.point_count for result in level_results], dtype=np.uint64)
    if summary.row_count != value_tile_row_count:
        raise RuntimeError("Value-tile rows do not reconcile to bucket range totals.")
    if not np.array_equal(summary.manifest_n_points, manifest_n_points):
        raise RuntimeError("Value-tile counts do not reconcile to manifest tile totals.")
    if not np.array_equal(summary.level_n_points, expected_level_counts):
        raise RuntimeError("Value-tile counts do not reconcile to level point totals.")
    if not np.array_equal(summary.exact_value_n_points, value_counts):
        raise RuntimeError("Exact value-tile counts do not reconcile to validated value totals.")
    if int(summary.level_n_points[0]) != validated.row_count:
        raise RuntimeError("Exact catalog total does not match the validated source row count.")


def _require_level_results_match_plan(
    validated: ValidatedPointsSource,
    plan: _PointsCacheBuildPlan,
    level_results: object,
) -> None:
    """Require completed level results to match the source and build plan.

    Results must cover every planned level in serialized order, remain within
    planned bucket, point-count, and grid bounds, and preserve every Exact
    source row.
    """
    if not isinstance(level_results, tuple) or len(level_results) != len(plan.levels):
        raise ValueError("`level_results` must contain exactly one result per planned level.")
    if not all(isinstance(result, _LevelWriteResult) for result in level_results):
        raise ValueError("Every level result must be _LevelWriteResult.")
    for level_plan, result in zip(plan.levels, level_results, strict=True):
        if result.level != level_plan.level:
            raise ValueError("Level results must follow planned serialized order.")
        planned_bucket_count = _bucket_count_for_level(level_plan)
        if any(bucket.bucket_id >= planned_bucket_count for bucket in result.buckets):
            raise ValueError("A result bucket ID lies outside the planned hash space.")
        if result.point_count > level_plan.point_count_upper_bound:
            raise ValueError("A level result exceeds its planned point-count bound.")
        if any(
            descriptor.tile_x >= level_plan.grid_width or descriptor.tile_y >= level_plan.grid_height
            for descriptor in result.tile_descriptors
        ):
            raise ValueError("A level result contains an out-of-grid tile.")
    if level_results[0].point_count != validated.row_count:
        raise ValueError("Exact result points do not match validated source rows.")


def _require_bucket_inventory_matches_results(
    staging_root: Path,
    level_results: tuple[_LevelWriteResult, ...],
) -> None:
    """Require staged bucket directories to match completed level results.

    The comparison spans every level and rejects both missing and unexpected
    bucket stores. Bucket contents and physical layouts are validated
    separately.
    """
    expected = {(staging_root / bucket.bucket_path).resolve() for result in level_results for bucket in result.buckets}
    observed = {path.resolve() for path in (staging_root / "levels").rglob("bucket-*.zarr") if path.is_dir()}
    if observed != expected:
        raise ValueError("Staged bucket paths do not match completed level results exactly.")


def _require_catalog_targets_absent(staging_root: Path) -> None:
    for relative_path in ("zarr.json", "values", "manifest", "value_tiles"):
        if (staging_root / relative_path).exists():
            raise FileExistsError(f"Catalog target already exists: {relative_path}.")


def _require_existing_staging_root(staging_root: Path) -> None:
    if not isinstance(staging_root, Path) or not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
