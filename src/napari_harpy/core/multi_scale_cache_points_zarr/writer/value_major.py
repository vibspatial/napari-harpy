"""Construct the mandatory all-level value-major location sidecar."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    _CacheAttributes,
    _parse_cache_attributes,
    _ValueMajorMetadata,
    _ValueMajorWriteSettings,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import _INT64_MAX, _require_integer_in_range
from napari_harpy.core.multi_scale_cache_points_zarr.storage._paths import VALUE_MAJOR_GROUP, level_name
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    MANIFEST_BUCKET_ID,
    VALUE_MAJOR_LOCATION_ARRAY,
    VALUE_MAJOR_POINT_INDPTR_ARRAY,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    ZARR_FORMAT_VERSION,
    ZARR_READ_MISSING_CHUNKS,
    ZARR_USE_CONSOLIDATED,
    _array_creation_options,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache


@dataclass(frozen=True)
class _RangeFragmentBatch:
    """Describe a point-bounded consecutive section of value-major ranges.

    Each aligned record retains the address of consecutive locations in one
    original tile-major bucket. ``manifest_index`` identifies the source tile's
    manifest row; that manifest row supplies the bucket ID. ``row_start`` is the
    first row in that bucket's point arrays, and ``row_count`` is the number of
    consecutive rows. The level is supplied separately when the locations are
    read. During value-major construction, this address is used specifically to
    read rows from the bucket's ``location`` array.

    This is an I/O batching construct, not a per-value container. A fragment may
    cross from one value's ranges into the next because point-count boundaries
    are independent of value boundaries. Value IDs remain implicit in the
    preserved value-major record order and the final ``value_point_indptr``;
    splitting changes neither that order nor the number of points per value.

    For example::

        manifest_index = 12
            -> manifest/bucket_id = 4
        row_start = 10
        row_count = 3
            -> bucket-004.zarr/location[[10, 11, 12], :]

    A fragment need not cover an entire input range. When a ``max_points``
    boundary falls inside a source range, ``_split_range_records_by_points()``
    derives ``row_start`` and ``row_count`` values describing only the portion
    assigned to that fragment. The original range metadata and bucket contents
    remain unchanged.

    Attributes
    ----------
    manifest_index
        Manifest row for each source range, used to resolve its bucket.
    row_start
        Bucket-global first row of each source range.
    row_count
        Number of consecutive source rows in each range.
    """

    manifest_index: np.ndarray
    row_start: np.ndarray
    row_count: np.ndarray

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_index", self.manifest_index),
            ("row_start", self.row_start),
            ("row_count", self.row_count),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype(np.uint64)
                or value.ndim != 1
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"`{name}` must be a one-dimensional C-contiguous uint64 array.")
        if not (len(self.manifest_index) == len(self.row_start) == len(self.row_count)) or len(self.row_count) == 0:
            raise ValueError("Range-fragment arrays must be nonempty and aligned.")
        if bool((self.row_count == 0).any()):
            raise ValueError("Range fragments must contain at least one point.")

    @property
    def point_count(self) -> int:
        return int(self.row_count.sum(dtype=np.uint64))


class _ShardBufferedLocationWriter:
    """Buffer sequential locations so each physical location shard is written once."""

    def __init__(self, array: zarr.Array, *, point_count: int, buffer_rows: int) -> None:
        self._array = array
        self._point_count = point_count
        self._buffer = np.empty((min(point_count, buffer_rows), 2), dtype=np.float32)
        self._buffered = 0
        self._written = 0

    def append(self, locations: np.ndarray) -> None:
        if (
            not isinstance(locations, np.ndarray)
            or locations.dtype != np.dtype(np.float32)
            or locations.ndim != 2
            or locations.shape[1:] != (2,)
            or not locations.flags.c_contiguous
        ):
            raise ValueError("Value-major locations must be a C-contiguous (N, 2) float32 array.")
        source_start = 0
        while source_start < len(locations):
            available = len(self._buffer) - self._buffered
            copied = min(available, len(locations) - source_start)
            source_stop = source_start + copied
            buffer_stop = self._buffered + copied
            self._buffer[self._buffered : buffer_stop] = locations[source_start:source_stop]
            self._buffered = buffer_stop
            source_start = source_stop
            if self._buffered == len(self._buffer):
                self._flush()

    def finalize(self) -> None:
        if self._buffered:
            self._flush()
        if self._written != self._point_count:
            raise RuntimeError("Value-major location rows do not match the declared level total.")

    def _flush(self) -> None:
        stop = self._written + self._buffered
        if stop > self._point_count:
            raise RuntimeError("Value-major location output exceeds the declared level total.")
        self._array[self._written : stop, :] = self._buffer[: self._buffered]
        self._written = stop
        self._buffered = 0


def _write_value_major_sidecars(
    staging_root: Path,
    attributes: _CacheAttributes,
    *,
    ordered_row_start: np.ndarray,
    value_tile_indptr: np.ndarray,
    level_value_n_points: np.ndarray,
    write_settings: _ValueMajorWriteSettings,
    max_open_value_major_readers: int | None = None,
) -> None:
    """Transpose tile-major locations into all mandatory value-major levels.

    ``ordered_row_start`` is the generation-owned construction index produced
    by the catalog transpose. It aligns one bucket-global source address with
    every persisted ``value_tiles`` row, but is never published in the cache.
    Locations are read and staged in bounded point batches; bucket readers
    are retained in a level-scoped cache. By default, every source-bucket reader
    encountered for the active level remains open until that level is complete.
    A positive ``max_open_value_major_readers`` applies an explicit upper bound
    on retained-reader capacity for unusually large bucket inventories.
    ``write_settings`` also supplies the construction-only point-batch limit;
    its physical fields must match the compact profile already published in
    ``attributes``.

    For each aligned record, ``manifest_index`` resolves the source tile and
    bucket, ``ordered_row_start`` resolves the first row in that bucket's point
    arrays, and ``value_tiles/n_points`` gives the consecutive row count. These
    three fields are sufficient to retrieve and append the corresponding
    locations in value-major order.
    """
    if not isinstance(attributes, _CacheAttributes):
        raise ValueError("`attributes` must be _CacheAttributes.")
    if not isinstance(write_settings, _ValueMajorWriteSettings):
        raise ValueError("`write_settings` must be _ValueMajorWriteSettings.")
    sidecar_metadata = attributes.value_major
    if sidecar_metadata != _ValueMajorMetadata.from_write_settings(write_settings):
        raise ValueError("Value-major write settings disagree with the published physical profile.")
    if max_open_value_major_readers is not None:
        _require_integer_in_range(
            max_open_value_major_readers,
            "max_open_value_major_readers",
            minimum=1,
            maximum=_INT64_MAX,
        )
    catalog = attributes.catalog
    if (
        not isinstance(ordered_row_start, np.ndarray)
        or ordered_row_start.dtype != np.dtype(np.uint64)
        or ordered_row_start.shape != (catalog.value_tile_row_count,)
        or not ordered_row_start.flags.c_contiguous
    ):
        raise ValueError("`ordered_row_start` does not match the value-tile catalog.")
    if (
        not isinstance(value_tile_indptr, np.ndarray)
        or value_tile_indptr.dtype != np.dtype(np.uint64)
        or value_tile_indptr.shape != (catalog.level_count, catalog.value_count + 1)
    ):
        raise ValueError("`value_tile_indptr` does not match the cache dimensions.")
    if (
        not isinstance(level_value_n_points, np.ndarray)
        or level_value_n_points.dtype != np.dtype(np.uint64)
        or level_value_n_points.shape != (catalog.level_count, catalog.value_count)
    ):
        raise ValueError("`level_value_n_points` does not match the cache dimensions.")

    with LocalStore(staging_root, read_only=False) as store:
        root = zarr.open_group(
            store=store,
            mode="r+",
            zarr_format=ZARR_FORMAT_VERSION,
            use_consolidated=ZARR_USE_CONSOLIDATED,
        )
        if _parse_cache_attributes(dict(root.attrs)) != attributes:
            raise ValueError("Staged root metadata changed before value-major construction.")
        if VALUE_MAJOR_GROUP in root:
            raise FileExistsError("Value-major sidecar already exists.")
        value_major = root.create_group(VALUE_MAJOR_GROUP)
        manifest_bucket_id = np.asarray(root[MANIFEST_BUCKET_ID][:], dtype=np.uint32)
        manifest_index_array = root[VALUE_TILES_MANIFEST_INDEX].with_config(
            {"read_missing_chunks": ZARR_READ_MISSING_CHUNKS}
        )
        n_points_array = root[VALUE_TILES_N_POINTS].with_config({"read_missing_chunks": ZARR_READ_MISSING_CHUNKS})

        for level, level_metadata in enumerate(attributes.levels):
            level_group = value_major.create_group(level_name(level))
            location, point_indptr = _create_level_arrays(
                level_group,
                point_count=level_metadata.point_count,
                value_count=catalog.value_count,
                metadata=sidecar_metadata,
                codec_id=attributes.zarr_settings.codec_id,
            )
            pointer = np.empty(catalog.value_count + 1, dtype=np.uint64)
            pointer[0] = 0
            np.cumsum(level_value_n_points[level], out=pointer[1:])
            if int(pointer[-1]) != level_metadata.point_count:
                raise RuntimeError("Value-major value totals do not match the level point count.")
            # Locations are appended below in value-major order. Persist
            # these boundaries so location[pointer[v]:pointer[v + 1]] belongs
            # to value ID v; no point-level value_id sidecar is required.
            point_indptr[:] = pointer

            reader_capacity = (
                level_metadata.bucket_count
                if max_open_value_major_readers is None
                else min(max_open_value_major_readers, level_metadata.bucket_count)
            )
            with _BucketReaderCache(staging_root, max_open_readers=reader_capacity) as readers:
                location_writer = _ShardBufferedLocationWriter(
                    location,
                    point_count=level_metadata.point_count,
                    buffer_rows=sidecar_metadata.point_shard_rows,
                )
                record_start = int(value_tile_indptr[level, 0])
                record_stop = int(value_tile_indptr[level, -1])
                for batch_start in range(
                    record_start,
                    record_stop,
                    catalog.settings.value_tile_chunk_rows,
                ):
                    batch_stop = min(batch_start + catalog.settings.value_tile_chunk_rows, record_stop)
                    manifest_index = np.asarray(
                        manifest_index_array[batch_start:batch_stop],
                        dtype=np.uint64,
                    )
                    row_start = np.asarray(ordered_row_start[batch_start:batch_stop], dtype=np.uint64)
                    row_count = np.asarray(n_points_array[batch_start:batch_stop], dtype=np.uint64)
                    for fragments in _split_range_records_by_points(
                        manifest_index,
                        row_start,
                        row_count,
                        max_points=write_settings.construction_batch_points,
                    ):
                        location_writer.append(
                            _read_fragment_locations(
                                fragments,
                                level=level,
                                manifest_bucket_id=manifest_bucket_id,
                                readers=readers,
                            )
                        )
                location_writer.finalize()


def _create_level_arrays(
    group: zarr.Group,
    *,
    point_count: int,
    value_count: int,
    metadata: _ValueMajorMetadata,
    codec_id: str,
) -> tuple[zarr.Array, zarr.Array]:
    common = _array_creation_options(codec_id)
    location = group.create_array(
        VALUE_MAJOR_LOCATION_ARRAY,
        shape=(point_count, 2),
        dtype=np.float32,
        chunks=(metadata.point_chunk_rows, 2),
        shards=(metadata.point_shard_rows, 2),
        **common,
    )
    point_indptr = group.create_array(
        VALUE_MAJOR_POINT_INDPTR_ARRAY,
        shape=(value_count + 1,),
        dtype=np.uint64,
        chunks=(value_count + 1,),
        **common,
    )
    return location, point_indptr


def _split_range_records_by_points(
    manifest_index: np.ndarray,
    row_start: np.ndarray,
    row_count: np.ndarray,
    *,
    max_points: int,
) -> Iterator[_RangeFragmentBatch]:
    """Split aligned range records into point-bounded fragments.

    The input arrays describe consecutive source ranges, but the number of
    records does not bound the number of points represented by those records.
    This function therefore partitions their conceptual concatenated point
    stream into fragments containing at most ``max_points``. Individual source
    ranges are split when necessary.

    Only the range metadata is transformed. Locations are read later by
    ``_read_fragment_locations()``, using the returned ``manifest_index``,
    ``row_start``, and ``row_count`` arrays. Input record order is preserved.

    For example, source ranges ``[10:15)`` and ``[20:22)`` contain seven
    points. With ``max_points=3``, they become::

        [10:13)
        [13:15), [20:21)
        [21:22)

    Parameters
    ----------
    manifest_index
        Manifest index for each source range.
    row_start
        First row of each range in its tile-major bucket array.
    row_count
        Number of consecutive points represented by each range.
    max_points
        Maximum total number of points represented by one yielded fragment.

    Yields
    ------
    _RangeFragmentBatch
        Aligned range metadata describing at most ``max_points`` points.
    """
    _require_integer_in_range(max_points, "max_points", minimum=1, maximum=_INT64_MAX)
    for name, value in (
        ("manifest_index", manifest_index),
        ("row_start", row_start),
        ("row_count", row_count),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(np.uint64)
            or value.ndim != 1
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"`{name}` must be a one-dimensional C-contiguous uint64 array.")
    if not (manifest_index.shape == row_start.shape == row_count.shape) or len(row_count) == 0:
        raise ValueError("Range-record arrays must be nonempty and aligned.")
    if bool((row_count == 0).any()):
        raise ValueError("Value-major source ranges must be nonempty.")
    record_indptr = np.empty(len(row_count) + 1, dtype=np.uint64)
    record_indptr[0] = 0
    np.cumsum(row_count, out=record_indptr[1:])
    total = int(record_indptr[-1])
    for point_start in range(0, total, max_points):
        point_stop = min(point_start + max_points, total)
        first = int(np.searchsorted(record_indptr, point_start, side="right") - 1)
        last = int(np.searchsorted(record_indptr, point_stop - 1, side="right"))
        # These must be owning copies: boundary adjustments below must never
        # mutate the caller's catalog block through a contiguous slice view.
        fragment_manifest = np.array(manifest_index[first:last], copy=True, order="C")
        fragment_start = np.array(row_start[first:last], copy=True, order="C")
        fragment_count = np.array(row_count[first:last], copy=True, order="C")
        leading = point_start - int(record_indptr[first])
        if leading:
            fragment_start[0] += np.uint64(leading)
            fragment_count[0] -= np.uint64(leading)
        excess = int(fragment_count.sum(dtype=np.uint64)) - (point_stop - point_start)
        if excess:
            fragment_count[-1] -= np.uint64(excess)
        yield _RangeFragmentBatch(
            manifest_index=fragment_manifest,
            row_start=fragment_start,
            row_count=fragment_count,
        )


def _read_fragment_locations(
    fragments: _RangeFragmentBatch,
    *,
    level: int,
    manifest_bucket_id: np.ndarray,
    readers: _BucketReaderCache,
) -> np.ndarray:
    """Read tile-major bucket locations and restore value-major record order.

    Each fragment record identifies a manifest row and a consecutive bucket-row
    interval. ``manifest_bucket_id`` maps that manifest row to its physical
    bucket. Together with ``level``, the bucket ID selects the original
    tile-major Zarr array from which ``row_start:row_start + row_count`` is read::

        fragments.manifest_index
                    |
                    v
        manifest/bucket_id + level
                    |
                    v
        tile_major/level_<level>/bucket-<bucket_id>.zarr/location[source_rows, :]
                    |
                    v
        scatter into the fragments' original value-major record order
                    |
                    v
        returned location batch
                    |
                    v
        value_major/level_<level>/location

    Records are grouped by bucket and sorted by their source ``row_start`` so
    each Zarr orthogonal row selector is strictly increasing. The fetched rows
    are then scattered back into the incoming fragment order before returning;
    the caller can therefore append the result sequentially without losing the
    value-major ordering. This function reads locations only. Value IDs remain
    implicit in that ordering and the sidecar's ``value_point_indptr``.

    Parameters
    ----------
    fragments
        Point-bounded source range records in value-major order.
    level
        Cache level containing the original tile-major buckets.
    manifest_bucket_id
        Mapping from global manifest row to physical bucket ID.
    readers
        Bounded cache of readers for the original tile-major bucket Zarrs.

    Returns
    -------
    np.ndarray
        C-contiguous ``float32`` locations in the incoming fragment order.
    """
    point_count = fragments.point_count
    output = np.empty((point_count, 2), dtype=np.float32)
    fragment_output_start = np.empty(len(fragments.row_count), dtype=np.uint64)
    fragment_output_start[0] = 0
    if len(fragment_output_start) > 1:
        np.cumsum(fragments.row_count[:-1], out=fragment_output_start[1:])
    bucket_id = manifest_bucket_id[fragments.manifest_index]
    for current_bucket_id in np.unique(bucket_id):
        selected = np.flatnonzero(bucket_id == current_bucket_id)
        # Source intervals follow tile-major bucket rows, while selected records
        # arrived in value-major catalog order. Sort reads by source address so
        # the exact row selector is increasing, then scatter back to output order.
        order = selected[np.argsort(fragments.row_start[selected], kind="stable")]
        starts = np.asarray(fragments.row_start[order], dtype=np.int64)
        counts = np.asarray(fragments.row_count[order], dtype=np.int64)
        stops = starts + counts
        if bool((starts[1:] < stops[:-1]).any()):
            raise ValueError("Value-major construction encountered overlapping source ranges.")
        # Expand matching source and destination intervals in the same
        # sorted-record order: source_rows[i] is written to output_rows[i].
        source_rows = _expand_ranges(starts, counts)
        output_rows = _expand_ranges(
            np.asarray(fragment_output_start[order], dtype=np.int64),
            counts,
        )
        locations = readers.get(level=level, bucket_id=int(current_bucket_id)).read_location_rows(source_rows)
        output[output_rows] = locations
    return output


def _expand_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Expand aligned starts/counts into one exact C-contiguous int64 selector."""
    total = int(counts.sum(dtype=np.int64))
    interval_offsets = np.empty(len(counts), dtype=np.int64)
    interval_offsets[0] = 0
    if len(counts) > 1:
        np.cumsum(counts[:-1], out=interval_offsets[1:])
    adjustments = np.repeat(starts - interval_offsets, counts)
    return np.arange(total, dtype=np.int64) + adjustments
