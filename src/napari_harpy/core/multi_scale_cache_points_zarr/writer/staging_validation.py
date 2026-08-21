"""Independently validate a complete, unpublished Zarr cache generation.

Construction and independent validation intentionally use different container
hierarchies::

    Construction
        _LevelWriteResult -> _BucketWriteResult -> _TileDescriptor

    Independent validation
        _ManifestInventory -> _ManifestLevel -> _ManifestBucket -> _TileDescriptor

Writer results report finalized physical output retained during construction.
The manifest inventory instead reconstructs expected logical structure solely
from the reopened catalog, including the global manifest indexes needed to
validate bucket sparse ranges. Only ``_TileDescriptor`` is shared because it
has the same persisted tile-address meaning on both sides of that boundary.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_N_POINTS,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    PUBLICATION_STATE_STAGING,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    _CacheAttributes,
)
from napari_harpy.core.multi_scale_cache_points_zarr.hashing import _tile_bucket_ids
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import (
    _CatalogReader,
    _iter_compact_bucket_range_batches,
)

_BRIDGE_MAX_POINTS_PER_TILE = 4_096


@dataclass(frozen=True)
class _ManifestBucket:
    """Manifest-derived description of one expected physical bucket."""

    level: int
    bucket_id: int
    descriptors: tuple[_TileDescriptor, ...]
    manifest_indexes: np.ndarray

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)
        if not isinstance(self.descriptors, tuple) or not self.descriptors:
            raise ValueError("A manifest bucket must contain a nonempty descriptor tuple.")
        if not all(isinstance(descriptor, _TileDescriptor) for descriptor in self.descriptors):
            raise ValueError("Every manifest-bucket descriptor must be a _TileDescriptor.")
        if any(
            (descriptor.level, descriptor.bucket_id) != (self.level, self.bucket_id) for descriptor in self.descriptors
        ):
            raise ValueError("Every manifest-bucket descriptor must belong to the stated bucket.")
        if tuple(descriptor.bucket_tile_index for descriptor in self.descriptors) != tuple(
            range(len(self.descriptors))
        ):
            raise ValueError("Manifest-bucket tile indexes must be contiguous from zero.")
        if self.descriptors != tuple(
            sorted(self.descriptors, key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x))
        ):
            raise ValueError("Manifest-bucket descriptors must follow (tile_y, tile_x) order.")
        if (
            not isinstance(self.manifest_indexes, np.ndarray)
            or self.manifest_indexes.dtype != np.dtype(np.uint64)
            or self.manifest_indexes.shape != (len(self.descriptors),)
        ):
            raise ValueError("Manifest bucket indexes must align with its descriptors.")
        if not self.manifest_indexes.flags.c_contiguous:
            raise ValueError("Manifest bucket indexes must be C-contiguous.")
        read_only = self.manifest_indexes.view()
        read_only.flags.writeable = False
        object.__setattr__(self, "manifest_indexes", read_only)


@dataclass(frozen=True)
class _ManifestLevel:
    """Group the manifest-derived physical buckets belonging to one level."""

    level: int
    buckets: tuple[_ManifestBucket, ...]

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        if not isinstance(self.buckets, tuple) or not self.buckets:
            raise ValueError("A manifest level must contain at least one bucket.")
        if not all(isinstance(bucket, _ManifestBucket) for bucket in self.buckets):
            raise ValueError("Every manifest-level bucket must be a _ManifestBucket.")
        if any(bucket.level != self.level for bucket in self.buckets):
            raise ValueError("Every manifest-level bucket must belong to the stated level.")
        bucket_ids = tuple(bucket.bucket_id for bucket in self.buckets)
        if bucket_ids != tuple(sorted(bucket_ids)) or len(set(bucket_ids)) != len(bucket_ids):
            raise ValueError("Manifest-level bucket IDs must be ordered and unique.")

    @property
    def descriptors(self) -> tuple[_TileDescriptor, ...]:
        """Return this level's bucket-owned descriptors in manifest tile order."""
        return tuple(
            sorted(
                (descriptor for bucket in self.buckets for descriptor in bucket.descriptors),
                key=lambda descriptor: (descriptor.tile_y, descriptor.tile_x),
            )
        )


@dataclass(frozen=True)
class _ManifestInventory:
    """Persisted manifest hierarchy reconstructed without writer results.

    This is the validation-side counterpart of the construction-time
    ``_LevelWriteResult`` hierarchy, not an interchangeable write result.
    """

    levels: tuple[_ManifestLevel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.levels, tuple) or not self.levels:
            raise ValueError("A manifest inventory must contain at least one level.")
        if not all(isinstance(level, _ManifestLevel) for level in self.levels):
            raise ValueError("Every manifest inventory level must be a _ManifestLevel.")
        if tuple(level.level for level in self.levels) != tuple(range(len(self.levels))):
            raise ValueError("Manifest inventory levels must be consecutive from zero.")


def _validate_staged_cache(staging_root: Path) -> None:
    """Validate one staged cache without reading point payload or source rows.

    This is the mandatory publication-tier validator. Its only source of truth
    is the reopened staged generation: no build plan, writer result, or source
    object is accepted from the caller.

    Its logical validation responsibilities are split as follows::

        _CatalogReader.validate_contents()
            Catalog arrays are internally consistent.

        _validate_persisted_build()
            Manifest and root metadata follow the construction policy.

        _validate_bucket_ranges_against_catalog()
            Physical bucket metadata and sparse ranges agree with the catalog.
    """
    _require_staging_root(staging_root)
    _validate_staging_artifacts(staging_root)
    with _CatalogReader(staging_root) as reader:
        if reader.attributes.publication_state != PUBLICATION_STATE_STAGING:
            raise ValueError("Staged validation requires publication_state='staging'.")
        reader.validate_contents()
        inventory = _read_manifest_inventory(reader)
        _validate_persisted_build(reader.attributes, inventory)
        _validate_bucket_ranges_against_catalog(reader, inventory, staging_root=staging_root)
    # Catch an unexpected sidecar created while validation had stores open.
    _validate_staging_artifacts(staging_root)


def _read_manifest_inventory(reader: _CatalogReader) -> _ManifestInventory:
    """Reconstruct ordered tile descriptors and physical buckets from manifest arrays."""
    level_indptr = np.asarray(reader.array(MANIFEST_LEVEL_INDPTR)[:], dtype=np.uint64)
    bucket_id = np.asarray(reader.array(MANIFEST_BUCKET_ID)[:], dtype=np.uint32)
    bucket_tile_index = np.asarray(reader.array(MANIFEST_BUCKET_TILE_INDEX)[:], dtype=np.uint32)
    tile_x = np.asarray(reader.array(MANIFEST_TILE_X)[:], dtype=np.uint32)
    tile_y = np.asarray(reader.array(MANIFEST_TILE_Y)[:], dtype=np.uint32)
    n_points = np.asarray(reader.array(MANIFEST_N_POINTS)[:], dtype=np.uint64)

    levels: list[_ManifestLevel] = []
    for level, (stored_start, stored_stop) in enumerate(zip(level_indptr[:-1], level_indptr[1:], strict=True)):
        start = int(stored_start)
        stop = int(stored_stop)
        descriptors = tuple(
            _TileDescriptor(
                level=level,
                bucket_id=int(bucket_id[index]),
                bucket_tile_index=int(bucket_tile_index[index]),
                tile_x=int(tile_x[index]),
                tile_y=int(tile_y[index]),
                n_points=int(n_points[index]),
            )
            for index in range(start, stop)
        )
        grouped: dict[int, list[tuple[int, _TileDescriptor]]] = defaultdict(list)
        for manifest_index, descriptor in zip(range(start, stop), descriptors, strict=True):
            grouped[descriptor.bucket_id].append((manifest_index, descriptor))
        level_buckets: list[_ManifestBucket] = []
        for current_bucket_id in sorted(grouped):
            rows = sorted(grouped[current_bucket_id], key=lambda item: item[1].bucket_tile_index)
            level_buckets.append(
                _ManifestBucket(
                    level=level,
                    bucket_id=current_bucket_id,
                    descriptors=tuple(descriptor for _, descriptor in rows),
                    manifest_indexes=np.ascontiguousarray(
                        np.asarray([manifest_index for manifest_index, _ in rows], dtype=np.uint64)
                    ),
                )
            )
        levels.append(_ManifestLevel(level=level, buckets=tuple(level_buckets)))
    return _ManifestInventory(levels=tuple(levels))


def _validate_persisted_build(attributes: _CacheAttributes, inventory: _ManifestInventory) -> None:
    """Require the manifest and root metadata to follow the construction policy.

    The two inputs are reconstructed from separate persisted parts of the
    reopened cache::

        root Zarr attributes                 manifest arrays
                |                                  |
                v                                  v
        _CacheAttributes                    _ManifestInventory
        declared build policy <--- compare ---> observed tile structure

    The comparison validates the planned hierarchy, observed tile counts, and
    deterministic hashing without accepting writer results or reading point
    payload arrays.
    """
    levels = attributes.levels
    geometry = attributes.geometry
    build = attributes.build
    expected_origin_x = float(math.floor(geometry.x_min / build.leaf_tile_size) * build.leaf_tile_size)
    expected_origin_y = float(math.floor(geometry.y_min / build.leaf_tile_size) * build.leaf_tile_size)
    if (geometry.x_origin, geometry.y_origin) != (expected_origin_x, expected_origin_y):
        raise ValueError("Cache geometry origins do not follow the aligned-leaf policy.")

    expected_levels = _expected_level_plan(attributes)
    if len(expected_levels) != len(levels):
        raise ValueError("Persisted cache level count does not follow the build policy.")
    for observed, expected in zip(levels, expected_levels, strict=True):
        observed_plan = (
            observed.level,
            observed.kind,
            observed.tile_size,
            observed.grid_width,
            observed.grid_height,
            observed.max_points_per_tile,
            observed.point_count_upper_bound,
        )
        if observed_plan != expected:
            raise ValueError(f"Persisted cache level {observed.level} does not follow the build policy.")

    coordinate_counts: list[dict[tuple[int, int], int]] = []
    for metadata, manifest_level in zip(levels, inventory.levels, strict=True):
        descriptors = manifest_level.descriptors
        counts = {(tile.tile_x, tile.tile_y): tile.n_points for tile in descriptors}
        coordinate_counts.append(counts)
        if len(counts) != len(descriptors):
            raise ValueError(f"Level {metadata.level} contains duplicate manifest coordinates.")
        if any(x >= metadata.grid_width or y >= metadata.grid_height for x, y in counts):
            raise ValueError(f"Level {metadata.level} contains a tile outside its stored grid.")
        if metadata.max_points_per_tile is not None and any(
            count > metadata.max_points_per_tile for count in counts.values()
        ):
            raise ValueError(f"Level {metadata.level} contains a tile above its capacity.")

        planned_bucket_count = max(
            1,
            math.ceil(metadata.point_count_upper_bound / build.target_points_per_bucket),
        )
        tile_x = np.ascontiguousarray(np.asarray([tile.tile_x for tile in descriptors], dtype=np.uint32))
        tile_y = np.ascontiguousarray(np.asarray([tile.tile_y for tile in descriptors], dtype=np.uint32))
        expected_bucket_ids = _tile_bucket_ids(tile_x, tile_y, bucket_count=planned_bucket_count)
        observed_bucket_ids = np.asarray([tile.bucket_id for tile in descriptors], dtype=np.uint64)
        if not np.array_equal(expected_bucket_ids, observed_bucket_ids):
            raise ValueError(f"Level {metadata.level} manifest bucket IDs violate the tile hash policy.")
        if len(set(observed_bucket_ids.tolist())) != metadata.bucket_count:
            raise ValueError(f"Level {metadata.level} physical bucket count is inconsistent.")

    if len(levels) > 1:
        if coordinate_counts[1].keys() != coordinate_counts[0].keys():
            raise ValueError("Bridge and Exact nonempty tile coordinates differ.")
        bridge_capacity = levels[1].max_points_per_tile
        if bridge_capacity is None:
            raise ValueError("Bridge level is missing its per-tile capacity.")
        for coordinate, exact_count in coordinate_counts[0].items():
            if coordinate_counts[1][coordinate] != min(exact_count, bridge_capacity):
                raise ValueError("Bridge tile count does not follow its Exact input and capacity.")

    for level in range(2, len(levels)):
        expected_counts: dict[tuple[int, int], int] = defaultdict(int)
        for (finer_x, finer_y), count in coordinate_counts[level - 1].items():
            expected_counts[(finer_x // 2, finer_y // 2)] += count
        capacity = levels[level].max_points_per_tile
        if capacity is None:
            raise ValueError(f"Spatial level {level} is missing its per-tile capacity.")
        expected_counts = {coordinate: min(count, capacity) for coordinate, count in expected_counts.items()}
        if coordinate_counts[level] != expected_counts:
            raise ValueError(f"Spatial level {level} does not match its immediate finer tiles and capacity.")

    observed_totals = [level.point_count for level in levels]
    if any(coarser > finer for finer, coarser in zip(observed_totals, observed_totals[1:], strict=False)):
        raise ValueError("Observed cache level point totals increase toward coarser levels.")
    if observed_totals[-1] > build.overview_point_budget:
        raise ValueError("Terminal cache level exceeds the overview point budget.")


def _expected_level_plan(attributes: _CacheAttributes) -> tuple[tuple[object, ...], ...]:
    """Recompute the versioned logical plan solely from persisted source/build facts."""
    geometry = attributes.geometry
    build = attributes.build

    def grid_shape(tile_size: int) -> tuple[int, int]:
        return (
            math.floor((geometry.x_max - geometry.x_origin) / tile_size) + 1,
            math.floor((geometry.y_max - geometry.y_origin) / tile_size) + 1,
        )

    grid_width, grid_height = grid_shape(build.leaf_tile_size)
    levels: list[tuple[object, ...]] = [
        (
            0,
            "exact",
            build.leaf_tile_size,
            grid_width,
            grid_height,
            None,
            attributes.source.row_count,
        )
    ]
    if attributes.source.row_count <= build.overview_point_budget:
        return tuple(levels)

    tile_size = build.leaf_tile_size
    scheduled_capacity = _BRIDGE_MAX_POINTS_PER_TILE
    kind = "bridge"
    while True:
        level = len(levels)
        grid_width, grid_height = grid_shape(tile_size)
        capacity = scheduled_capacity
        upper_bound = min(int(levels[-1][-1]), grid_width * grid_height * capacity)
        if upper_bound > build.overview_point_budget and grid_width == 1 and grid_height == 1:
            capacity = build.overview_point_budget
            upper_bound = build.overview_point_budget
        levels.append((level, kind, tile_size, grid_width, grid_height, capacity, upper_bound))
        if upper_bound <= build.overview_point_budget:
            return tuple(levels)
        kind = "spatial"
        tile_size *= 2
        scheduled_capacity *= 2


def _validate_bucket_ranges_against_catalog(
    reader: _CatalogReader,
    inventory: _ManifestInventory,
    *,
    staging_root: Path,
) -> None:
    """Require physical bucket metadata and sparse ranges to agree with the catalog.

    The two independently read representations are reconciled as follows::

        physical bucket sparse ranges             catalog value_tiles
                      |                                  |
                      v                                  v
        (value_id, manifest_index, n_points)       implicit value_id,
                                                   manifest_index, n_points
                      +---- record-for-record comparison ----+

    The catalog's ``value_id`` is implicit in ``value_tiles/indptr`` rather
    than repeated in each row. Bucket sparse ranges are therefore reordered by
    ``(value_id, manifest_index)`` before comparison. Point payload arrays are
    not read.
    """
    attributes = reader.attributes
    level_indptr = np.asarray(reader.array(MANIFEST_LEVEL_INDPTR)[:], dtype=np.uint64)
    value_indptr = np.asarray(reader.array(VALUE_TILES_INDPTR)[:], dtype=np.uint64)
    exact_value_totals = np.zeros(attributes.catalog.value_count, dtype=np.uint64)

    for metadata, manifest_level in zip(attributes.levels, inventory.levels, strict=True):
        level = manifest_level.level
        buckets = manifest_level.buckets
        value_id = np.empty(metadata.range_count, dtype=np.uint32)
        manifest_index = np.empty(metadata.range_count, dtype=np.uint64)
        n_points = np.empty(metadata.range_count, dtype=np.uint64)
        cursor = 0
        # Physical ranges are bucket-local, whereas ``value_tiles`` is ordered
        # cache-wide by value within each level. Reconstruct the complete level
        # before sorting it into the catalog's comparison order::
        #
        #     bucket 0 ranges --\
        #     bucket 1 ranges ---+--> level-wide compact records
        #     bucket 2 ranges --/                 |
        #                                        v
        #                              sort by value_id,
        #                              then manifest_index
        #                                        |
        #                                        v
        #                              compare with value_tiles
        for bucket in buckets:
            for batch in _iter_compact_bucket_range_batches(
                staging_root,
                level=level,
                bucket_id=bucket.bucket_id,
                expected_descriptors=bucket.descriptors,
                manifest_indexes=bucket.manifest_indexes,
                batch_rows=attributes.catalog.settings.value_tile_chunk_rows,
                expected_settings=attributes.zarr_settings,
            ):
                stop = cursor + batch.row_count
                if stop > metadata.range_count:
                    raise ValueError(f"Level {level} bucket ranges exceed the stored range total.")
                value_id[cursor:stop] = batch.value_id
                manifest_index[cursor:stop] = batch.manifest_index
                n_points[cursor:stop] = batch.n_points
                cursor = stop
        if cursor != metadata.range_count:
            raise ValueError(f"Level {level} bucket ranges do not match the stored range total.")
        if bool((value_id >= attributes.catalog.value_count).any()):
            raise ValueError(f"Level {level} bucket ranges contain an unknown value ID.")

        # 1. Summary reconciliation
        # -------------------------
        # Sum physical ranges by manifest tile and compare their aggregate
        # counts with ``manifest/n_points``. For Exact, also accumulate physical
        # counts by value; that summary is checked after the level loop completes.
        level_manifest_start = int(level_indptr[level])
        level_manifest_stop = int(level_indptr[level + 1])
        derived_manifest_totals = np.zeros(level_manifest_stop - level_manifest_start, dtype=np.uint64)
        np.add.at(derived_manifest_totals, manifest_index - np.uint64(level_manifest_start), n_points)
        expected_manifest_totals = np.asarray(
            reader.array(MANIFEST_N_POINTS)[level_manifest_start:level_manifest_stop],
            dtype=np.uint64,
        )
        if not np.array_equal(derived_manifest_totals, expected_manifest_totals):
            raise ValueError(f"Level {level} bucket ranges do not reconcile to manifest point totals.")
        if level == 0:
            np.add.at(exact_value_totals, value_id, n_points)

        # 2. Record-for-record reconciliation
        # -----------------------------------
        # Reorder the physical range records into the catalog's
        # ``(value_id, manifest_index)`` order, then require every logical field
        # to match the corresponding ``value_tiles`` record exactly.
        order = np.lexsort((manifest_index, value_id))
        expected_start = int(value_indptr[level, 0])
        expected_stop = int(value_indptr[level, -1])
        if expected_stop - expected_start != metadata.range_count:
            raise ValueError(f"Level {level} value-tile interval does not match its range total.")
        batch_rows = attributes.catalog.settings.value_tile_chunk_rows
        # The complete level arrays and their sort permutation are already in
        # memory. Compare in batches so reordered physical fields, catalog rows,
        # positions, and inferred value IDs are not also materialized for the
        # complete level; this bounds comparison overhead, not the level sort.
        for local_start in range(0, metadata.range_count, batch_rows):
            local_stop = min(local_start + batch_rows, metadata.range_count)
            selected = order[local_start:local_stop]
            positions = np.arange(expected_start + local_start, expected_start + local_stop, dtype=np.uint64)
            expected_values = np.searchsorted(value_indptr[level], positions, side="right") - 1
            expected_manifest = np.asarray(
                reader.array(VALUE_TILES_MANIFEST_INDEX)[expected_start + local_start : expected_start + local_stop],
                dtype=np.uint64,
            )
            expected_counts = np.asarray(
                reader.array(VALUE_TILES_N_POINTS)[expected_start + local_start : expected_start + local_stop],
                dtype=np.uint64,
            )
            if not (
                np.array_equal(value_id[selected], expected_values.astype(np.uint32))
                and np.array_equal(manifest_index[selected], expected_manifest)
                and np.array_equal(n_points[selected], expected_counts)
            ):
                raise ValueError(f"Level {level} bucket sparse ranges disagree with value_tiles.")

    stored_exact_value_totals = np.asarray(reader.array("values/n_points")[:], dtype=np.uint64)
    if not np.array_equal(exact_value_totals, stored_exact_value_totals):
        raise ValueError("Exact bucket sparse ranges do not reconcile to canonical value totals.")


def _validate_staging_artifacts(staging_root: Path) -> None:
    """Require a clean, unpublished, Zarr-only generation tree.

    The staging root must contain exactly::

        staging_root/
          zarr.json
          levels/
          values/
          manifest/
          value_tiles/

    Descendants must be Zarr groups or arrays identified by ``zarr.json``, or
    numeric chunk/shard keys below a ``c/`` directory. Reject sidecars,
    construction scratch, symbolic links, and unexplained nodes.
    Logical Zarr contents and array layouts are validated by their dedicated
    readers rather than by this filesystem-level check.
    """
    allowed_root_entries = {"zarr.json", "levels", "values", "manifest", "value_tiles"}
    observed_root_entries = {path.name for path in staging_root.iterdir()}
    if observed_root_entries != allowed_root_entries:
        raise ValueError("Staged cache root contains missing or unexpected artifacts.")

    for path in staging_root.rglob("*"):
        relative = path.relative_to(staging_root)
        if path.is_symlink():
            raise ValueError(f"Staged cache contains an unexpected symbolic link: {relative.as_posix()}.")
        if path.is_file():
            if path.name == "zarr.json":
                continue
            if "c" in relative.parts:
                chunk_parts = relative.parts[relative.parts.index("c") + 1 :]
                if chunk_parts and all(part.isdecimal() for part in chunk_parts):
                    continue
            raise ValueError(f"Staged cache contains an unexpected file: {relative.as_posix()}.")
        if path.is_dir() and not (path / "zarr.json").is_file():
            if "c" in relative.parts:
                chunk_parts = relative.parts[relative.parts.index("c") + 1 :]
                if all(part.isdecimal() for part in chunk_parts):
                    continue
            raise ValueError(f"Staged cache contains an unexpected directory: {relative.as_posix()}.")


def _require_staging_root(staging_root: Path) -> None:
    if not isinstance(staging_root, Path) or not staging_root.is_dir():
        raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
