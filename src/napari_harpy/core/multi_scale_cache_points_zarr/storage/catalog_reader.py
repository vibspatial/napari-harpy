from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    CATALOG_ARRAY_DTYPES,
    LEVELS_GROUP,
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_GROUP,
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_N_POINTS,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    VALUE_TILES_GROUP,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    VALUES_GROUP,
    VALUES_N_POINTS,
    _CacheAttributes,
    _parse_cache_attributes,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import _parse_root_attributes
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import (
    _strict_array,
    _validate_array_layout,
    _validate_array_layouts,
    _validate_hierarchy,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import (
    _BucketWriteResult,
    _ZarrWriteSettings,
)

_CATALOG_ARRAY_PATHS = (
    VALUES_N_POINTS,
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    MANIFEST_N_POINTS,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
)


@dataclass(frozen=True, eq=False)
class _RangeRecordBatch:
    """Hold one bounded batch from one level's sortable tile/value records.

    Level identity belongs to the containing level stream rather than being
    repeated for every record in the batch.
    """

    value_id: np.ndarray
    manifest_index: np.ndarray
    n_points: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            ("value_id", self.value_id, np.dtype(np.uint32)),
            ("manifest_index", self.manifest_index, np.dtype(np.uint64)),
            ("n_points", self.n_points, np.dtype(np.uint64)),
        )
        row_count: int | None = None
        for name, array, dtype in arrays:
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 1
                or array.dtype != dtype
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"`{name}` must be a one-dimensional C-contiguous {dtype.name} array.")
            if row_count is None:
                row_count = len(array)
            elif len(array) != row_count:
                raise ValueError("Range-record arrays must have equal lengths.")
        if row_count == 0:
            raise ValueError("A range-record batch must not be empty.")
        if bool((self.n_points == 0).any()):
            raise ValueError("Every range-record point count must be positive.")
        for name, array, _ in arrays:
            view = array.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    @property
    def row_count(self) -> int:
        return len(self.value_id)


class _CatalogReader:
    """Open a self-describing cache root and validate its frozen catalog layout."""

    def __init__(self, cache_root: Path) -> None:
        self._cache_root = cache_root
        self._store: LocalStore | None = None
        self._root: zarr.Group | None = None
        self._attributes: _CacheAttributes | None = None
        self._arrays: dict[str, zarr.Array] = {}

    @property
    def attributes(self) -> _CacheAttributes:
        if self._attributes is None:
            raise RuntimeError("Catalog reader is not open.")
        return self._attributes

    def __enter__(self) -> _CatalogReader:
        if self._store is not None:
            raise RuntimeError("A catalog reader can be entered only once.")
        if not isinstance(self._cache_root, Path) or not self._cache_root.is_dir():
            raise FileNotFoundError("Cache root does not exist.")
        self._store = LocalStore(self._cache_root, read_only=True)
        try:
            self._root = zarr.open_group(
                store=self._store,
                mode="r",
                zarr_format=3,
                use_consolidated=False,
            )
            self._attributes = _parse_cache_attributes(dict(self._root.attrs))
            self._validate_hierarchy()
            self._arrays = {name: self._strict_array(name) for name in _CATALOG_ARRAY_PATHS}
            self._validate_layouts()
        except Exception:
            self._close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self._close()
        return False

    def array(self, name: str) -> zarr.Array:
        """Return one strict catalog array by its frozen cache-relative path."""
        try:
            return self._arrays[name]
        except KeyError as error:
            raise ValueError(f"Unknown or unopened catalog array: {name}.") from error

    def validate_contents(self) -> None:
        """Stream and reconcile logical catalog contents without reading points."""
        attributes = self.attributes
        catalog = attributes.catalog
        value_counts = np.asarray(self.array(VALUES_N_POINTS)[:], dtype=np.uint64)
        if bool((value_counts == 0).any()) or int(value_counts.sum(dtype=np.uint64)) != attributes.source.row_count:
            raise ValueError("Catalog value totals do not reconcile to Exact source rows.")

        level_indptr = np.asarray(self.array(MANIFEST_LEVEL_INDPTR)[:], dtype=np.uint64)
        if (
            int(level_indptr[0]) != 0
            or int(level_indptr[-1]) != catalog.manifest_row_count
            or bool((level_indptr[1:] <= level_indptr[:-1]).any())
        ):
            raise ValueError("Manifest level pointers are invalid.")
        manifest = {
            name: np.asarray(self.array(name)[:], dtype=CATALOG_ARRAY_DTYPES[name])
            for name in (
                MANIFEST_BUCKET_ID,
                MANIFEST_BUCKET_TILE_INDEX,
                MANIFEST_TILE_X,
                MANIFEST_TILE_Y,
                MANIFEST_N_POINTS,
            )
        }
        if bool((manifest[MANIFEST_N_POINTS] == 0).any()):
            raise ValueError("Manifest point counts must be positive.")
        addresses: set[tuple[int, int, int]] = set()
        levels_group = self._root_or_raise()[LEVELS_GROUP]
        if not isinstance(levels_group, zarr.Group):
            raise ValueError("Cache levels node is not a group.")
        for level, metadata in enumerate(attributes.levels):
            start = int(level_indptr[level])
            stop = int(level_indptr[level + 1])
            if stop - start != metadata.tile_count:
                raise ValueError("Manifest level rows do not match root level metadata.")
            coordinates = tuple(
                zip(
                    manifest[MANIFEST_TILE_X][start:stop].tolist(),
                    manifest[MANIFEST_TILE_Y][start:stop].tolist(),
                    strict=True,
                )
            )
            if coordinates != tuple(sorted(coordinates, key=lambda coordinate: (coordinate[1], coordinate[0]))):
                raise ValueError("Manifest coordinates are not ordered by (tile_y, tile_x).")
            if len(set(coordinates)) != len(coordinates) or any(
                tile_x >= metadata.grid_width or tile_y >= metadata.grid_height for tile_x, tile_y in coordinates
            ):
                raise ValueError("Manifest coordinates are duplicate or outside the level grid.")
            bucket_ids = manifest[MANIFEST_BUCKET_ID][start:stop]
            bucket_indexes = manifest[MANIFEST_BUCKET_TILE_INDEX][start:stop]
            planned_bucket_count = max(
                1,
                (metadata.point_count_upper_bound + attributes.build.target_points_per_bucket - 1)
                // attributes.build.target_points_per_bucket,
            )
            if bool((bucket_ids >= planned_bucket_count).any()):
                raise ValueError("Manifest bucket identity lies outside the planned hash space.")
            for bucket_id, bucket_index in zip(bucket_ids.tolist(), bucket_indexes.tolist(), strict=True):
                address = (level, bucket_id, bucket_index)
                if address in addresses:
                    raise ValueError("Manifest contains a duplicate bucket-local address.")
                addresses.add(address)
            if len(set(bucket_ids.tolist())) != metadata.bucket_count:
                raise ValueError("Manifest bucket identities do not match root level metadata.")
            for bucket_id_value in set(bucket_ids.tolist()):
                indexes = np.sort(bucket_indexes[bucket_ids == bucket_id_value])
                if not np.array_equal(indexes, np.arange(len(indexes), dtype=np.uint32)):
                    raise ValueError("Manifest bucket-local indexes are not contiguous from zero.")
            if int(manifest[MANIFEST_N_POINTS][start:stop].sum(dtype=np.uint64)) != metadata.point_count:
                raise ValueError("Manifest point counts do not match root level metadata.")
            level_group = levels_group[f"level_{level}"]
            if not isinstance(level_group, zarr.Group):
                raise ValueError("Serialized cache level is not a Zarr group.")
            expected_buckets = {f"bucket-{bucket_id:03d}.zarr" for bucket_id in set(bucket_ids.tolist())}
            if (
                set(level_group.group_keys()) != expected_buckets
                or set(level_group.array_keys())
                or dict(level_group.attrs)
            ):
                raise ValueError("Serialized level bucket children do not match the manifest.")

        indptr = np.asarray(self.array(VALUE_TILES_INDPTR)[:], dtype=np.uint64)
        if int(indptr[0, 0]) != 0 or int(indptr[-1, -1]) != catalog.value_tile_row_count:
            raise ValueError("Value-tile pointers have invalid terminals.")
        if bool((indptr[:, 1:] < indptr[:, :-1]).any()) or bool((indptr[1:, 0] != indptr[:-1, -1]).any()):
            raise ValueError("Value-tile pointers are not nondecreasing and level-continuous.")
        flat_indptr = np.empty(catalog.level_count * catalog.value_count + 1, dtype=np.uint64)
        for level in range(catalog.level_count):
            start = level * catalog.value_count
            flat_indptr[start : start + catalog.value_count + 1] = indptr[level]

        manifest_totals = np.zeros(catalog.manifest_row_count, dtype=np.uint64)
        level_totals = np.zeros(catalog.level_count, dtype=np.uint64)
        exact_value_totals = np.zeros(catalog.value_count, dtype=np.uint64)
        previous_key = -1
        previous_manifest = -1
        batch_rows = catalog.settings.value_tile_chunk_rows
        for batch_start in range(0, catalog.value_tile_row_count, batch_rows):
            batch_stop = min(batch_start + batch_rows, catalog.value_tile_row_count)
            manifest_index = np.asarray(
                self.array(VALUE_TILES_MANIFEST_INDEX)[batch_start:batch_stop],
                dtype=np.uint64,
            )
            n_points = np.asarray(self.array(VALUE_TILES_N_POINTS)[batch_start:batch_stop], dtype=np.uint64)
            if bool((n_points == 0).any()) or int(manifest_index.max()) >= catalog.manifest_row_count:
                raise ValueError("Value-tile rows contain invalid references or counts.")
            positions = np.arange(batch_start, batch_stop, dtype=np.uint64)
            flat_keys = np.searchsorted(flat_indptr, positions, side="right") - 1
            levels = flat_keys // catalog.value_count
            values = flat_keys % catalog.value_count
            level_starts = level_indptr[levels]
            level_stops = level_indptr[levels + 1]
            if bool(((manifest_index < level_starts) | (manifest_index >= level_stops)).any()):
                raise ValueError("Value-tile manifest reference belongs to the wrong level.")
            same_key = flat_keys[1:] == flat_keys[:-1]
            if bool((manifest_index[1:][same_key] <= manifest_index[:-1][same_key]).any()):
                raise ValueError("Value-tile manifest indexes are not strictly ordered within a value.")
            if int(flat_keys[0]) == previous_key and int(manifest_index[0]) <= previous_manifest:
                raise ValueError("Value-tile ordering is invalid across a read-batch boundary.")
            previous_key = int(flat_keys[-1])
            previous_manifest = int(manifest_index[-1])
            np.add.at(manifest_totals, manifest_index, n_points)
            np.add.at(level_totals, levels, n_points)
            exact = levels == 0
            np.add.at(exact_value_totals, values[exact], n_points[exact])

        if not np.array_equal(manifest_totals, manifest[MANIFEST_N_POINTS]):
            raise ValueError("Value-tile counts do not reconcile to manifest tile totals.")
        if not np.array_equal(
            level_totals,
            np.asarray([level.point_count for level in attributes.levels], dtype=np.uint64),
        ):
            raise ValueError("Value-tile counts do not reconcile to level totals.")
        if not np.array_equal(exact_value_totals, value_counts):
            raise ValueError("Exact value-tile counts do not reconcile to canonical value totals.")

    def _validate_hierarchy(self) -> None:
        root = self._root_or_raise()
        if set(root.group_keys()) != {LEVELS_GROUP, VALUES_GROUP, MANIFEST_GROUP, VALUE_TILES_GROUP}:
            raise ValueError("Cache root contains missing or unexpected Zarr groups.")
        if set(root.array_keys()):
            raise ValueError("Cache root must not contain arrays directly.")
        levels = root[LEVELS_GROUP]
        if not isinstance(levels, zarr.Group):
            raise ValueError("Cache levels node is not a group.")
        expected_levels = {f"level_{level}" for level in range(self.attributes.catalog.level_count)}
        if set(levels.group_keys()) != expected_levels or set(levels.array_keys()) or dict(levels.attrs):
            raise ValueError("Cache level hierarchy does not match root metadata.")

        expected_arrays = {
            VALUES_GROUP: {"n_points"},
            MANIFEST_GROUP: {"level_indptr", "bucket_id", "bucket_tile_index", "tile_x", "tile_y", "n_points"},
            VALUE_TILES_GROUP: {"indptr", "manifest_index", "n_points"},
        }
        for group_name, array_names in expected_arrays.items():
            group = root[group_name]
            if not isinstance(group, zarr.Group):
                raise ValueError(f"Catalog node is not a group: {group_name}.")
            if set(group.array_keys()) != array_names or set(group.group_keys()) or dict(group.attrs):
                raise ValueError(f"Catalog group has the wrong children or attributes: {group_name}.")

    def _validate_layouts(self) -> None:
        attributes = self.attributes
        catalog = attributes.catalog
        settings = catalog.settings
        codec_id = attributes.zarr_settings.codec_id
        fixed = {
            VALUES_N_POINTS: ((catalog.value_count,), (catalog.value_count,)),
            MANIFEST_LEVEL_INDPTR: ((catalog.level_count + 1,), (catalog.level_count + 1,)),
            VALUE_TILES_INDPTR: (
                (catalog.level_count, catalog.value_count + 1),
                (catalog.level_count, catalog.value_count + 1),
            ),
        }
        for name, (shape, chunks) in fixed.items():
            _validate_array_layout(
                self._arrays[name],
                name=name,
                dtype=CATALOG_ARRAY_DTYPES[name],
                shape=shape,
                chunks=chunks,
                shards=None,
                codec_id=codec_id,
            )

        for name in (
            MANIFEST_BUCKET_ID,
            MANIFEST_BUCKET_TILE_INDEX,
            MANIFEST_TILE_X,
            MANIFEST_TILE_Y,
            MANIFEST_N_POINTS,
        ):
            _validate_array_layout(
                self._arrays[name],
                name=name,
                dtype=CATALOG_ARRAY_DTYPES[name],
                shape=(catalog.manifest_row_count,),
                chunks=(settings.manifest_chunk_rows,),
                shards=(settings.manifest_shard_rows,),
                codec_id=codec_id,
            )
        for name in (VALUE_TILES_MANIFEST_INDEX, VALUE_TILES_N_POINTS):
            _validate_array_layout(
                self._arrays[name],
                name=name,
                dtype=CATALOG_ARRAY_DTYPES[name],
                shape=(catalog.value_tile_row_count,),
                chunks=(settings.value_tile_chunk_rows,),
                shards=(settings.value_tile_shard_rows,),
                codec_id=codec_id,
            )

    def _strict_array(self, name: str) -> zarr.Array:
        root = self._root_or_raise()
        node = root[name]
        if not isinstance(node, zarr.Array):
            raise ValueError(f"Required catalog node is not an array: {name}.")
        if dict(node.attrs):
            raise ValueError(f"Catalog arrays must not contain attributes: {name}.")
        return node.with_config({"read_missing_chunks": False})

    def _root_or_raise(self) -> zarr.Group:
        if self._root is None:
            raise RuntimeError("Catalog root is not open.")
        return self._root

    def _close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._root = None
        self._attributes = None
        self._arrays = {}


def _iter_bucket_range_batches(
    cache_root: Path,
    bucket_result: _BucketWriteResult,
    manifest_indexes: np.ndarray,
    *,
    batch_rows: int,
    expected_settings: _ZarrWriteSettings,
) -> Iterator[_RangeRecordBatch]:
    """Yield validated compact bucket ranges without decoding point payloads.

    ``manifest_indexes[i]`` is the global manifest row assigned to bucket-local
    tile ``i``. Range rows are read in bounded contiguous slices, mapped through
    ``tile_indptr`` to that tile's manifest row, and returned as sortable
    ``(value_id, manifest_index, n_points)`` records within the bucket's
    validated level stream. Validation carries value-order and row-coverage
    state across slice boundaries.
    """
    if not isinstance(cache_root, Path) or not cache_root.is_dir():
        raise ValueError("`cache_root` must be an existing pathlib.Path directory.")
    if not isinstance(bucket_result, _BucketWriteResult):
        raise ValueError("`bucket_result` must be _BucketWriteResult.")
    if not isinstance(expected_settings, _ZarrWriteSettings):
        raise ValueError("`expected_settings` must be _ZarrWriteSettings.")
    _require_integer_in_range(batch_rows, "batch_rows", minimum=1, maximum=_INT64_MAX)
    if (
        not isinstance(manifest_indexes, np.ndarray)
        or manifest_indexes.dtype != np.dtype(np.uint64)
        or manifest_indexes.shape != (len(bucket_result.tile_descriptors),)
        or not manifest_indexes.flags.c_contiguous
    ):
        raise ValueError("`manifest_indexes` must align with the bucket's tile descriptors.")

    store = LocalStore(cache_root / bucket_result.bucket_path, read_only=True)
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        _validate_hierarchy(root)
        attributes = _parse_root_attributes(
            dict(root.attrs),
            expected_level=bucket_result.level,
            expected_bucket_id=bucket_result.bucket_id,
        )
        arrays = {
            name: _strict_array(root, name)
            for name in (
                "location",
                "point_id",
                "value_id",
                "tile_x",
                "tile_y",
                "tile_offset",
                "ranges/tile_indptr",
                "ranges/value_id",
                "ranges/row_start",
                "ranges/row_count",
            )
        }
        # Validate the reopened store at this reader boundary rather than trust
        # an optional earlier catalog preflight. This keeps the iterator safe as
        # a standalone storage primitive and detects changes before array reads.
        _validate_array_layouts(arrays, attributes)
        observed_settings = _ZarrWriteSettings(
            point_chunk_rows=arrays["value_id"].chunks[0],
            point_shard_rows=arrays["value_id"].shards[0],  # type: ignore[index]
            range_chunk_rows=arrays["ranges/value_id"].chunks[0],
            range_shard_rows=arrays["ranges/value_id"].shards[0],  # type: ignore[index]
            codec_id=attributes.codec_id,
        )
        if observed_settings != expected_settings:
            raise ValueError("Bucket physical settings do not match the cache-wide backend profile.")
        if (attributes.point_count, attributes.range_count) != (
            bucket_result.point_count,
            bucket_result.range_count,
        ):
            raise ValueError("Bucket physical totals do not match its finalized result.")

        tile_x = np.asarray(arrays["tile_x"][:], dtype=np.uint32)
        tile_y = np.asarray(arrays["tile_y"][:], dtype=np.uint32)
        tile_offset = np.asarray(arrays["tile_offset"][:], dtype=np.uint64)
        tile_indptr = np.asarray(arrays["ranges/tile_indptr"][:], dtype=np.uint64)
        descriptors = bucket_result.tile_descriptors
        if tile_offset[0] != 0 or tile_offset[-1] != attributes.point_count:
            raise ValueError("Bucket tile offsets have invalid terminals.")
        if tile_indptr[0] != 0 or tile_indptr[-1] != attributes.range_count:
            raise ValueError("Bucket tile range pointers have invalid terminals.")
        if bool((tile_offset[1:] <= tile_offset[:-1]).any()) or bool((tile_indptr[1:] <= tile_indptr[:-1]).any()):
            raise ValueError("Bucket tile pointers must be strictly increasing.")
        for index, descriptor in enumerate(descriptors):
            if (
                descriptor.bucket_tile_index != index
                or int(tile_x[index]) != descriptor.tile_x
                or int(tile_y[index]) != descriptor.tile_y
                or int(tile_offset[index + 1] - tile_offset[index]) != descriptor.n_points
            ):
                raise ValueError("Bucket compact tile arrays do not match finalized descriptors.")

        previous_tile = -1
        previous_value = -1
        expected_row_start = 0
        range_total = attributes.range_count
        for batch_start in range(0, range_total, batch_rows):
            batch_stop = min(batch_start + batch_rows, range_total)
            values = np.asarray(arrays["ranges/value_id"][batch_start:batch_stop], dtype=np.uint32)
            row_starts = np.asarray(arrays["ranges/row_start"][batch_start:batch_stop], dtype=np.uint64)
            row_counts = np.asarray(arrays["ranges/row_count"][batch_start:batch_stop], dtype=np.uint64)
            if bool((row_counts == 0).any()):
                raise ValueError("Bucket sparse range counts must be positive.")
            range_indexes = np.arange(batch_start, batch_stop, dtype=np.uint64)
            tile_indexes = np.searchsorted(tile_indptr, range_indexes, side="right") - 1
            boundaries = np.flatnonzero(
                np.concatenate((np.array([True]), tile_indexes[1:] != tile_indexes[:-1], np.array([True])))
            )
            for segment_start, segment_stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                tile_index = int(tile_indexes[segment_start])
                segment_values = values[segment_start:segment_stop]
                segment_starts = row_starts[segment_start:segment_stop]
                segment_counts = row_counts[segment_start:segment_stop]
                first_global_range = batch_start + int(segment_start)
                last_global_range = batch_start + int(segment_stop)
                if tile_index != previous_tile:
                    if tile_index != previous_tile + 1:
                        raise ValueError("Bucket range rows do not follow tile order.")
                    previous_value = -1
                    expected_row_start = int(tile_offset[tile_index])
                if int(segment_values[0]) <= previous_value or bool((segment_values[1:] <= segment_values[:-1]).any()):
                    raise ValueError("Bucket range values are not strictly ordered within a tile.")
                if int(segment_starts[0]) != expected_row_start or bool(
                    (segment_starts[1:] != segment_starts[:-1] + segment_counts[:-1]).any()
                ):
                    raise ValueError("Bucket sparse ranges do not cover contiguous point rows.")
                expected_row_start = int(segment_starts[-1] + segment_counts[-1])
                previous_value = int(segment_values[-1])
                previous_tile = tile_index
                if last_global_range == int(tile_indptr[tile_index + 1]) and expected_row_start != int(
                    tile_offset[tile_index + 1]
                ):
                    raise ValueError("Bucket sparse ranges do not terminate at the tile boundary.")
                if first_global_range < int(tile_indptr[tile_index]):
                    raise ValueError("Bucket range segment starts before its tile pointer.")

            yield _RangeRecordBatch(
                value_id=np.ascontiguousarray(values),
                manifest_index=np.ascontiguousarray(manifest_indexes[tile_indexes]),
                n_points=np.ascontiguousarray(row_counts),
            )
        if previous_tile != len(descriptors) - 1 or expected_row_start != attributes.point_count:
            raise ValueError("Bucket range iteration did not cover every tile and point row.")
    finally:
        store.close()


def _read_bucket_storage_settings(
    cache_root: Path,
    bucket_result: _BucketWriteResult,
) -> _ZarrWriteSettings:
    """Read and validate one bucket's authoritative physical row settings."""
    if not isinstance(cache_root, Path) or not cache_root.is_dir():
        raise ValueError("`cache_root` must be an existing pathlib.Path directory.")
    if not isinstance(bucket_result, _BucketWriteResult):
        raise ValueError("`bucket_result` must be _BucketWriteResult.")
    store = LocalStore(cache_root / bucket_result.bucket_path, read_only=True)
    try:
        root = zarr.open_group(store=store, mode="r", zarr_format=3, use_consolidated=False)
        _validate_hierarchy(root)
        attributes = _parse_root_attributes(
            dict(root.attrs),
            expected_level=bucket_result.level,
            expected_bucket_id=bucket_result.bucket_id,
        )
        arrays = {
            name: _strict_array(root, name)
            for name in (
                "location",
                "point_id",
                "value_id",
                "tile_x",
                "tile_y",
                "tile_offset",
                "ranges/tile_indptr",
                "ranges/value_id",
                "ranges/row_start",
                "ranges/row_count",
            )
        }
        _validate_array_layouts(arrays, attributes)
        point_shards = arrays["value_id"].shards
        range_shards = arrays["ranges/value_id"].shards
        if point_shards is None or range_shards is None:
            raise ValueError("Bucket point and range arrays must be sharded.")
        return _ZarrWriteSettings(
            # Layout validation above proves that ``location``, ``point_id``,
            # and point-level ``value_id`` use identical chunk and shard
            # boundaries along their first, point-row axis. Use the
            # one-dimensional ``value_id`` metadata as their canonical source.
            point_chunk_rows=arrays["value_id"].chunks[0],
            point_shard_rows=point_shards[0],
            range_chunk_rows=arrays["ranges/value_id"].chunks[0],
            range_shard_rows=range_shards[0],
            codec_id=attributes.codec_id,
        )
    finally:
        store.close()
