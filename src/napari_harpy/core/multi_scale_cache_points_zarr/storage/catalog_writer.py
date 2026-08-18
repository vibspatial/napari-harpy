from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import zarr
from zarr.codecs import BytesCodec
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
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
    _CatalogWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    _CHUNK_KEY_ENCODING,
    _compressors,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _RangeRecordBatch
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings


@dataclass(frozen=True)
class _ValueTilesWriteSummary:
    """Return catalog accounting used to reconcile the level-wise sort."""

    indptr: np.ndarray
    manifest_n_points: np.ndarray
    level_n_points: np.ndarray
    exact_value_n_points: np.ndarray
    row_count: int

    def __post_init__(self) -> None:
        for name, array in (
            ("indptr", self.indptr),
            ("manifest_n_points", self.manifest_n_points),
            ("level_n_points", self.level_n_points),
            ("exact_value_n_points", self.exact_value_n_points),
        ):
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.uint64) or not array.flags.c_contiguous:
                raise ValueError(f"`{name}` must be a C-contiguous uint64 array.")
            view = array.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)
        _require_integer_in_range(self.row_count, "row_count", minimum=1, maximum=_INT64_MAX)


class _CatalogWriter:
    """Create and fill the cache-wide Zarr catalog in unpublished staging.

    The staging directory already contains independently finalized bucket
    stores, but no root or ancestor group metadata. Entering creates that group
    hierarchy and all fixed-shape catalog arrays without touching bucket data.
    Root semantic attributes are written only by ``finalize`` after every array
    has been filled and reconciled.
    """

    def __init__(
        self,
        staging_root: Path,
        *,
        level_count: int,
        value_count: int,
        manifest_row_count: int,
        value_tile_row_count: int,
        zarr_settings: _ZarrWriteSettings,
        catalog_settings: _CatalogWriteSettings,
    ) -> None:
        if not isinstance(staging_root, Path):
            raise ValueError("`staging_root` must be pathlib.Path.")
        for name, value in (
            ("level_count", level_count),
            ("value_count", value_count),
            ("manifest_row_count", manifest_row_count),
            ("value_tile_row_count", value_tile_row_count),
        ):
            _require_integer_in_range(value, name, minimum=1, maximum=_INT64_MAX)
        if not isinstance(zarr_settings, _ZarrWriteSettings):
            raise ValueError("`zarr_settings` must be _ZarrWriteSettings.")
        if not isinstance(catalog_settings, _CatalogWriteSettings):
            raise ValueError("`catalog_settings` must be _CatalogWriteSettings.")
        self._staging_root = staging_root
        self._level_count = level_count
        self._value_count = value_count
        self._manifest_row_count = manifest_row_count
        self._value_tile_row_count = value_tile_row_count
        self._zarr_settings = zarr_settings
        self._catalog_settings = catalog_settings
        self._store: LocalStore | None = None
        self._root: zarr.Group | None = None
        self._arrays: dict[str, zarr.Array] = {}
        self._value_tile_cursor = 0
        self._values_written = False
        self._manifest_written = False
        self._indptr_written = False
        self._finalized = False

    def __enter__(self) -> _CatalogWriter:
        if self._store is not None:
            raise RuntimeError("A catalog writer can be entered only once.")
        if not isinstance(self._staging_root, Path) or not self._staging_root.is_dir():
            raise ValueError("`staging_root` must be an existing pathlib.Path directory.")
        if (self._staging_root / "zarr.json").exists():
            raise FileExistsError("The staged cache root already has Zarr group metadata.")
        for path in (VALUES_GROUP, MANIFEST_GROUP, VALUE_TILES_GROUP):
            if (self._staging_root / path).exists():
                raise FileExistsError(f"Catalog target already exists: {path}.")

        self._store = LocalStore(self._staging_root, read_only=False)
        try:
            self._root = zarr.open_group(
                store=self._store,
                mode="a",
                zarr_format=3,
                use_consolidated=False,
            )
            self._create_hierarchy()
            self._create_arrays()
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

    def write_value_counts(self, n_points: np.ndarray) -> None:
        """Write exact canonical value totals once."""
        self._require_array(n_points, "n_points", dtype=np.dtype(np.uint64), shape=(self._value_count,))
        if self._values_written:
            raise RuntimeError("Catalog values have already been written.")
        self._array(VALUES_N_POINTS)[:] = n_points
        self._values_written = True

    def write_manifest(
        self,
        *,
        level_indptr: np.ndarray,
        bucket_id: np.ndarray,
        bucket_tile_index: np.ndarray,
        tile_x: np.ndarray,
        tile_y: np.ndarray,
        n_points: np.ndarray,
    ) -> None:
        """Write the complete aligned manifest once."""
        self._require_array(
            level_indptr,
            "level_indptr",
            dtype=np.dtype(np.uint64),
            shape=(self._level_count + 1,),
        )
        rows = self._manifest_row_count
        for name, array, dtype in (
            ("bucket_id", bucket_id, np.dtype(np.uint32)),
            ("bucket_tile_index", bucket_tile_index, np.dtype(np.uint32)),
            ("tile_x", tile_x, np.dtype(np.uint32)),
            ("tile_y", tile_y, np.dtype(np.uint32)),
            ("n_points", n_points, np.dtype(np.uint64)),
        ):
            self._require_array(array, name, dtype=dtype, shape=(rows,))
        if self._manifest_written:
            raise RuntimeError("Catalog manifest has already been written.")
        self._array(MANIFEST_LEVEL_INDPTR)[:] = level_indptr
        self._array(MANIFEST_BUCKET_ID)[:] = bucket_id
        self._array(MANIFEST_BUCKET_TILE_INDEX)[:] = bucket_tile_index
        self._array(MANIFEST_TILE_X)[:] = tile_x
        self._array(MANIFEST_TILE_Y)[:] = tile_y
        self._array(MANIFEST_N_POINTS)[:] = n_points
        self._manifest_written = True

    def append_value_tiles(self, manifest_index: np.ndarray, n_points: np.ndarray) -> None:
        """Append one aligned globally ordered value-to-tile output batch."""
        if not isinstance(manifest_index, np.ndarray) or not isinstance(n_points, np.ndarray):
            raise ValueError("Value-tile batches must be NumPy arrays.")
        if (
            manifest_index.ndim != 1
            or manifest_index.dtype != np.dtype(np.uint64)
            or n_points.ndim != 1
            or n_points.dtype != np.dtype(np.uint64)
            or manifest_index.shape != n_points.shape
            or not manifest_index.flags.c_contiguous
            or not n_points.flags.c_contiguous
        ):
            raise ValueError("Value-tile batches must be aligned C-contiguous uint64 arrays.")
        if len(manifest_index) == 0:
            return
        stop = self._value_tile_cursor + len(manifest_index)
        if stop > self._value_tile_row_count:
            raise ValueError("Value-tile output exceeds its declared row count.")
        self._array(VALUE_TILES_MANIFEST_INDEX)[self._value_tile_cursor : stop] = manifest_index
        self._array(VALUE_TILES_N_POINTS)[self._value_tile_cursor : stop] = n_points
        self._value_tile_cursor = stop

    def write_value_tile_indptr(self, indptr: np.ndarray) -> None:
        """Write the final two-dimensional level/value pointer table once."""
        self._require_array(
            indptr,
            "value_tiles/indptr",
            dtype=np.dtype(np.uint64),
            shape=(self._level_count, self._value_count + 1),
        )
        if self._indptr_written:
            raise RuntimeError("Value-tile pointers have already been written.")
        self._array(VALUE_TILES_INDPTR)[:, :] = indptr
        self._indptr_written = True

    def finalize(self, attributes: _CacheAttributes) -> None:
        """Write semantic root attributes after every physical row is present."""
        if not isinstance(attributes, _CacheAttributes):
            raise ValueError("`attributes` must be _CacheAttributes.")
        if self._finalized:
            raise RuntimeError("Catalog writer has already been finalized.")
        if not self._values_written or not self._manifest_written or not self._indptr_written:
            raise RuntimeError("Cannot finalize before values, manifest, and pointers are written.")
        if self._value_tile_cursor != self._value_tile_row_count:
            raise RuntimeError("Value-tile output rows are incomplete.")
        if (
            attributes.zarr_settings != self._zarr_settings
            or attributes.catalog.level_count != self._level_count
            or attributes.catalog.value_count != self._value_count
            or attributes.catalog.manifest_row_count != self._manifest_row_count
            or attributes.catalog.value_tile_row_count != self._value_tile_row_count
            or attributes.catalog.settings.manifest_chunk_rows != self._catalog_settings.manifest_chunk_rows
            or attributes.catalog.settings.manifest_shard_rows != self._catalog_settings.manifest_shard_rows
            or attributes.catalog.settings.value_tile_chunk_rows != self._catalog_settings.value_tile_chunk_rows
            or attributes.catalog.settings.value_tile_shard_rows != self._catalog_settings.value_tile_shard_rows
        ):
            raise ValueError("Cache attributes do not match the physical catalog writer contract.")
        root = self._root_or_raise()
        if dict(root.attrs):
            raise RuntimeError("Cache root attributes must remain empty until finalization.")
        root.update_attributes(attributes.to_dict())
        self._finalized = True

    def _create_hierarchy(self) -> None:
        root = self._root_or_raise()
        levels = root.create_group("levels")
        for level in range(self._level_count):
            levels.create_group(f"level_{level}")
        root.create_group(VALUES_GROUP)
        root.create_group(MANIFEST_GROUP)
        root.create_group(VALUE_TILES_GROUP)

    def _create_arrays(self) -> None:
        root = self._root_or_raise()
        compressors = _compressors(self._zarr_settings.codec_id)
        common = {
            "compressors": compressors,
            "serializer": BytesCodec(endian="little"),
            "fill_value": 0,
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "config": {"write_empty_chunks": True},
        }
        values = root[VALUES_GROUP]
        manifest = root[MANIFEST_GROUP]
        value_tiles = root[VALUE_TILES_GROUP]
        if not all(isinstance(group, zarr.Group) for group in (values, manifest, value_tiles)):
            raise RuntimeError("Catalog groups were not created.")

        self._arrays[VALUES_N_POINTS] = values.create_array(
            "n_points",
            shape=(self._value_count,),
            dtype=np.uint64,
            chunks=(self._value_count,),
            **common,
        )
        self._arrays[MANIFEST_LEVEL_INDPTR] = manifest.create_array(
            "level_indptr",
            shape=(self._level_count + 1,),
            dtype=np.uint64,
            chunks=(self._level_count + 1,),
            **common,
        )
        manifest_rows = self._manifest_row_count
        manifest_chunks = (self._catalog_settings.manifest_chunk_rows,)
        manifest_shards = (self._catalog_settings.manifest_shard_rows,)
        for name, dtype in (
            ("bucket_id", np.uint32),
            ("bucket_tile_index", np.uint32),
            ("tile_x", np.uint32),
            ("tile_y", np.uint32),
            ("n_points", np.uint64),
        ):
            self._arrays[f"{MANIFEST_GROUP}/{name}"] = manifest.create_array(
                name,
                shape=(manifest_rows,),
                dtype=dtype,
                chunks=manifest_chunks,
                shards=manifest_shards,
                **common,
            )

        self._arrays[VALUE_TILES_INDPTR] = value_tiles.create_array(
            "indptr",
            shape=(self._level_count, self._value_count + 1),
            dtype=np.uint64,
            chunks=(self._level_count, self._value_count + 1),
            **common,
        )
        value_rows = self._value_tile_row_count
        value_chunks = (self._catalog_settings.value_tile_chunk_rows,)
        value_shards = (self._catalog_settings.value_tile_shard_rows,)
        for name in ("manifest_index", "n_points"):
            self._arrays[f"{VALUE_TILES_GROUP}/{name}"] = value_tiles.create_array(
                name,
                shape=(value_rows,),
                dtype=np.uint64,
                chunks=value_chunks,
                shards=value_shards,
                **common,
            )

    def _array(self, name: str) -> zarr.Array:
        try:
            return self._arrays[name]
        except KeyError as error:
            raise RuntimeError(f"Catalog array is not open: {name}.") from error

    def _root_or_raise(self) -> zarr.Group:
        if self._root is None:
            raise RuntimeError("Catalog root is not open.")
        return self._root

    @staticmethod
    def _require_array(
        value: object,
        name: str,
        *,
        dtype: np.dtype,
        shape: tuple[int, ...],
    ) -> None:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != dtype
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"`{name}` must be a C-contiguous {dtype.name} array with shape {shape}.")

    def _close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._root = None
        self._arrays = {}


def _write_value_tiles_by_level(
    batches_by_level: tuple[Iterable[_RangeRecordBatch], ...],
    writer: _CatalogWriter,
    *,
    level_indptr: np.ndarray,
    expected_level_row_counts: tuple[int, ...],
    value_count: int,
    output_batch_rows: int,
) -> _ValueTilesWriteSummary:
    """Sort compact range records one level at a time and write the index.

    The persisted key order is ``(level, value_id, manifest_index)``. Levels
    occupy disjoint, ascending output regions, so sorting each complete level by
    ``(value_id, manifest_index)`` is equivalent to one cache-wide sort. Keeping
    only one level's three compact arrays and NumPy permutation in memory avoids
    temporary sorted stores while bounding peak allocation by the largest level
    rather than the complete cache.

    Sorted output is emitted in small contiguous batches. The full ordered
    ``manifest_index`` and ``n_points`` arrays are therefore never materialized
    as additional level-sized copies.
    """
    if not isinstance(writer, _CatalogWriter):
        raise ValueError("`writer` must be _CatalogWriter.")
    if not isinstance(expected_level_row_counts, tuple):
        raise ValueError("`expected_level_row_counts` must be a tuple.")
    if not isinstance(batches_by_level, tuple) or len(batches_by_level) != len(expected_level_row_counts):
        raise ValueError("`batches_by_level` must contain one iterable per expected level.")
    level_count = len(expected_level_row_counts)
    if level_count == 0:
        raise ValueError("At least one cache level is required.")
    for level, row_count in enumerate(expected_level_row_counts):
        _require_integer_in_range(row_count, f"expected_level_row_counts[{level}]", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(value_count, "value_count", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(output_batch_rows, "output_batch_rows", minimum=1, maximum=_INT64_MAX)
    if (
        not isinstance(level_indptr, np.ndarray)
        or level_indptr.dtype != np.dtype(np.uint64)
        or level_indptr.shape != (level_count + 1,)
        or not level_indptr.flags.c_contiguous
        or int(level_indptr[0]) != 0
        or bool((level_indptr[1:] <= level_indptr[:-1]).any())
    ):
        raise ValueError("`level_indptr` must be a valid C-contiguous uint64 manifest pointer array.")

    manifest_row_count = int(level_indptr[-1])
    manifest_counts = np.zeros(manifest_row_count, dtype=np.uint64)
    level_counts = np.zeros(level_count, dtype=np.uint64)
    exact_value_counts = np.zeros(value_count, dtype=np.uint64)
    indptr = np.empty((level_count, value_count + 1), dtype=np.uint64)
    rows_written = 0

    for level, (batches, expected_row_count) in enumerate(
        zip(batches_by_level, expected_level_row_counts, strict=True)
    ):
        value_id, manifest_index, n_points = _collect_level_range_records(
            batches,
            level=level,
            expected_row_count=expected_row_count,
            value_count=value_count,
            manifest_start=int(level_indptr[level]),
            manifest_stop=int(level_indptr[level + 1]),
        )
        np.add.at(manifest_counts, manifest_index, n_points)
        level_counts[level] = n_points.sum(dtype=np.uint64)
        if level == 0:
            np.add.at(exact_value_counts, value_id, n_points)

        # ``np.lexsort`` uses its last key as primary: value groups first,
        # followed by strictly increasing manifest rows inside each group.
        order = np.lexsort((manifest_index, value_id))
        _write_ordered_level(
            writer,
            value_id=value_id,
            manifest_index=manifest_index,
            n_points=n_points,
            order=order,
            output_batch_rows=output_batch_rows,
        )

        entry_counts = np.bincount(value_id, minlength=value_count).astype(np.uint64, copy=False)
        indptr[level, 0] = rows_written
        np.cumsum(entry_counts, out=indptr[level, 1:])
        indptr[level, 1:] += np.uint64(rows_written)
        rows_written += expected_row_count
        # Release this complete level before the next collector allocates its
        # arrays; otherwise Python loop locals would overlap adjacent levels.
        del value_id, manifest_index, n_points, order, entry_counts

    summary = _ValueTilesWriteSummary(
        indptr=np.ascontiguousarray(indptr),
        manifest_n_points=np.ascontiguousarray(manifest_counts),
        level_n_points=np.ascontiguousarray(level_counts),
        exact_value_n_points=np.ascontiguousarray(exact_value_counts),
        row_count=rows_written,
    )
    writer.write_value_tile_indptr(summary.indptr)
    return summary


def _collect_level_range_records(
    batches: Iterable[_RangeRecordBatch],
    *,
    level: int,
    expected_row_count: int,
    value_count: int,
    manifest_start: int,
    manifest_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize exactly one level's compact records in traversal order."""
    value_id = np.empty(expected_row_count, dtype=np.uint32)
    manifest_index = np.empty(expected_row_count, dtype=np.uint64)
    n_points = np.empty(expected_row_count, dtype=np.uint64)
    cursor = 0
    for batch in batches:
        if not isinstance(batch, _RangeRecordBatch):
            raise ValueError("Range-record iterator yielded an invalid batch.")
        if not bool((batch.level == level).all()):
            raise ValueError("Range-record batch belongs to the wrong cache level.")
        if int(batch.value_id.max()) >= value_count:
            raise ValueError("Range-record value ID is outside the catalog.")
        if bool(((batch.manifest_index < manifest_start) | (batch.manifest_index >= manifest_stop)).any()):
            raise ValueError("Range-record manifest index belongs to the wrong cache level.")
        stop = cursor + batch.row_count
        if stop > expected_row_count:
            raise ValueError("Range-record level exceeds its declared total.")
        value_id[cursor:stop] = batch.value_id
        manifest_index[cursor:stop] = batch.manifest_index
        n_points[cursor:stop] = batch.n_points
        cursor = stop
    if cursor != expected_row_count:
        raise ValueError("Range-record level does not match its declared total.")
    return value_id, manifest_index, n_points


def _write_ordered_level(
    writer: _CatalogWriter,
    *,
    value_id: np.ndarray,
    manifest_index: np.ndarray,
    n_points: np.ndarray,
    order: np.ndarray,
    output_batch_rows: int,
) -> None:
    """Write one sorted level without constructing full ordered array copies."""
    previous_value: int | None = None
    previous_manifest: int | None = None
    for start in range(0, len(order), output_batch_rows):
        indexes = order[start : start + output_batch_rows]
        ordered_values = np.ascontiguousarray(value_id[indexes])
        ordered_manifest = np.ascontiguousarray(manifest_index[indexes])
        ordered_counts = np.ascontiguousarray(n_points[indexes])
        same_value = ordered_values[1:] == ordered_values[:-1]
        if bool((ordered_manifest[1:][same_value] <= ordered_manifest[:-1][same_value]).any()):
            raise ValueError("Duplicate (level, value_id, manifest_index) record.")
        first_value = int(ordered_values[0])
        first_manifest = int(ordered_manifest[0])
        if previous_value == first_value and previous_manifest is not None and first_manifest <= previous_manifest:
            raise ValueError("Duplicate (level, value_id, manifest_index) record.")
        writer.append_value_tiles(ordered_manifest, ordered_counts)
        previous_value = int(ordered_values[-1])
        previous_manifest = int(ordered_manifest[-1])
