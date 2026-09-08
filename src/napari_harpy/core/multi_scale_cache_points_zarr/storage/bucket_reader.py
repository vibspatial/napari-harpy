from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _bucket_path,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.payload import _PointPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage._row_selection import (
    _build_exact_row_selection,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    TILE_MAJOR_BUCKET_ARRAY_PATHS,
    TILE_MAJOR_LOCATION,
    TILE_MAJOR_POINT_ID,
    TILE_MAJOR_TILE_OFFSET,
    TILE_MAJOR_TILE_X,
    TILE_MAJOR_TILE_Y,
    TILE_MAJOR_VALUE_ID,
    ZARR_FORMAT_VERSION,
    ZARR_READ_MISSING_CHUNKS,
    ZARR_USE_CONSOLIDATED,
    _BucketAttributes,
    _parse_root_attributes,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_validation import (
    _validate_array_layouts,
    _validate_hierarchy,
)


@dataclass(frozen=True)
class _PointDisplayPayload:
    """Return the aligned point arrays needed for visualization."""

    location: npt.NDArray[np.float32]
    value_id: npt.NDArray[np.uint32]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.location, np.ndarray)
            or self.location.dtype != np.dtype(np.float32)
            or self.location.ndim != 2
            or self.location.shape[1:] != (2,)
            or not self.location.flags.c_contiguous
        ):
            raise ValueError("`location` must be a C-contiguous (N, 2) float32 array.")
        if (
            not isinstance(self.value_id, np.ndarray)
            or self.value_id.dtype != np.dtype(np.uint32)
            or self.value_id.ndim != 1
            or not self.value_id.flags.c_contiguous
            or len(self.value_id) != len(self.location)
            or len(self.value_id) == 0
        ):
            raise ValueError("`value_id` must be a nonempty aligned C-contiguous uint32 array.")
        location = self.location.view()
        location.flags.writeable = False
        value_id = self.value_id.view()
        value_id.flags.writeable = False
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "value_id", value_id)


class _BucketReader:
    """Reuse strict read-only handles for construction and display payloads.

    Parameters
    ----------
    cache_root
        Cache-generation root containing the canonical bucket path.
    level
        Serialized level of the bucket to open.
    bucket_id
        Serialized bucket identifier within ``level``.

    Attributes
    ----------
    _tile_descriptors : tuple[_TileDescriptor, ...] | None
        Accepted complete-tile addressing for this bucket. The immutable tuple
        contains every nonempty tile in ``bucket_tile_index`` order, not only
        tiles requested by a viewport. It stores tile identities and point-row
        intervals, not point payloads.

        ``None`` means no descriptor tuple has been accepted, not that the
        bucket is empty. ``set_tile_descriptors()`` installs the tuple only
        after validating it against the opened bucket. In the viewer path,
        ``_PointsCacheReader`` supplies its existing per-bucket tuple; neither
        the tuple nor its descriptor objects are copied.

        Complete-tile display reads use the accepted descriptors for direct
        tile-index lookup without rereading stored tile pointers. Closing the
        reader releases its reference to the tuple.

    Notes
    -----
    The context opens one bucket once and configures every array to fail on a
    missing chunk or shard. Full-tile construction payloads read every tile row
    including mandatory point IDs; value-major construction can instead read
    exact location-only row selections. Every complete-tile display read requires
    the validated descriptors described above, including standalone callers.
    Display payloads omit point IDs and never access per-value sparse ranges;
    any diagnostic value filtering happens in the caller after complete reads.
    """

    def __init__(self, cache_root: str | Path, *, level: int, bucket_id: int) -> None:
        _require_integer_in_range(level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(bucket_id, "bucket_id", maximum=_UINT32_MAX)
        self._cache_root = Path(cache_root)
        self._level = level
        self._bucket_id = bucket_id
        self._target = self._cache_root / _bucket_path(level=level, bucket_id=bucket_id)
        self._store: LocalStore | None = None
        self._root: zarr.Group | None = None
        self._attributes: _BucketAttributes | None = None
        self._arrays: dict[str, zarr.Array] = {}
        # Accepted complete-tile addressing; installed only after bucket validation.
        self._tile_descriptors: tuple[_TileDescriptor, ...] | None = None
        self._entered = False
        self._open = False

    def __enter__(self) -> _BucketReader:
        if self._entered:
            raise RuntimeError("A bucket reader can be entered only once.")
        self._entered = True
        if not self._target.exists():
            raise FileNotFoundError(f"Zarr bucket does not exist: {self._target}")
        try:
            self._store = LocalStore(self._target, read_only=True)
            self._root = zarr.open_group(
                store=self._store,
                mode="r",
                zarr_format=ZARR_FORMAT_VERSION,
                use_consolidated=ZARR_USE_CONSOLIDATED,
            )
            _validate_hierarchy(self._root)
            self._attributes = _parse_root_attributes(
                dict(self._root.attrs),
                expected_level=self._level,
                expected_bucket_id=self._bucket_id,
            )
            self._arrays = {name: self._strict_array(name) for name in TILE_MAJOR_BUCKET_ARRAY_PATHS}
            _validate_array_layouts(self._arrays, self._attributes)
        except Exception:
            self._close()
            raise
        self._open = True
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

    def read_construction_payload(self, descriptor: _TileDescriptor) -> _PointPayload:
        """Read all aligned rows and mandatory point IDs for cache construction."""
        start, stop = self._construction_tile_interval(descriptor)
        location = np.ascontiguousarray(self._array(TILE_MAJOR_LOCATION)[start:stop, :], dtype=np.float32)
        value_id = np.ascontiguousarray(self._array(TILE_MAJOR_VALUE_ID)[start:stop], dtype=np.uint32)
        point_id = np.ascontiguousarray(self._array(TILE_MAJOR_POINT_ID)[start:stop], dtype=np.uint64)
        return _PointPayload(
            x_rel=np.ascontiguousarray(location[:, 0]),
            y_rel=np.ascontiguousarray(location[:, 1]),
            value_id=value_id,
            point_id=point_id,
        )

    def read_location_rows(self, row_selection: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
        """Read an exact increasing location-row selection for cache construction.

        This does not require installed complete-tile descriptors. The mandatory
        value-major writer already owns validated bucket-global row addresses
        carried forward from ``ranges/row_start`` and needs only locations.
        """
        if (
            not isinstance(row_selection, np.ndarray)
            or row_selection.dtype != np.dtype(np.int64)
            or row_selection.ndim != 1
            or len(row_selection) == 0
            or not row_selection.flags.c_contiguous
        ):
            raise ValueError("`row_selection` must be a nonempty C-contiguous int64 array.")
        point_count = self._attributes_or_raise().point_count
        if (
            int(row_selection[0]) < 0
            or int(row_selection[-1]) >= point_count
            or bool((row_selection[1:] <= row_selection[:-1]).any())
        ):
            raise ValueError("Location rows must be strictly increasing and in bounds.")
        locations = np.ascontiguousarray(
            self._array(TILE_MAJOR_LOCATION).get_orthogonal_selection((row_selection, slice(None))),
            dtype=np.float32,
        )
        if locations.shape != (len(row_selection), 2):
            raise RuntimeError("Bucket location selection returned an unexpected shape.")
        return locations

    def read_complete_display_payload(self, descriptor: _TileDescriptor) -> _PointDisplayPayload:
        """Read one complete tile-major display payload through the batched reader.

        Return every point's location and value ID, without point IDs or value
        filtering. The complete bucket descriptor tuple must already be installed.
        """
        return self.read_complete_display_payloads((descriptor,))[0]

    def read_complete_display_payloads(
        self,
        descriptors: tuple[_TileDescriptor, ...],
    ) -> tuple[_PointDisplayPayload, ...]:
        """Read complete tile-major tiles in one coordinated bucket operation.

        Each descriptor supplies a validated bucket-global point interval.
        Touching intervals use one basic slice; disjoint intervals use one exact
        C-contiguous int64 row selector. Both aligned Zarr arrays, ``location``
        and point-level ``value_id``, receive that same orthogonal selection.
        Point IDs and per-value sparse ranges are never read.

        Parameters
        ----------
        descriptors
            Nonempty tuple of requested tiles in increasing bucket-local tile
            order. Install the complete bucket tuple with ``set_tile_descriptors()``
            first; this request may then contain any ordered subset of its tiles.
            Every point in each requested tile is returned, without value filtering.

        Returns
        -------
        tuple of _PointDisplayPayload
            Nonempty, immutable payloads in request order. Per-tile arrays are
            views into the shared batch allocations, not additional point copies.

        Notes
        -----
        The transient ``batch_tile_indptr`` partitions the returned point arrays
        by request. It is unrelated to persisted sparse-range pointers.
        """
        self._require_open()
        if not isinstance(descriptors, tuple) or not descriptors:
            raise ValueError("`descriptors` must be a nonempty tuple.")
        if any(not isinstance(descriptor, _TileDescriptor) for descriptor in descriptors):
            raise ValueError("Every complete display request must be a _TileDescriptor.")

        batch_tile_indptr = np.empty(len(descriptors) + 1, dtype=np.uint64)
        batch_tile_indptr[0] = 0
        intervals: list[tuple[int, int]] = []
        rows_resolved = 0
        previous_bucket_tile_index: int | None = None
        for request_index, descriptor in enumerate(descriptors):
            interval = self.resolve_complete_tile_interval(descriptor)
            bucket_tile_index = descriptor.bucket_tile_index
            if previous_bucket_tile_index is not None and bucket_tile_index <= previous_bucket_tile_index:
                raise ValueError("Display requests must follow increasing bucket-local tile order.")
            previous_bucket_tile_index = bucket_tile_index
            intervals.append(interval)
            rows_resolved += interval[1] - interval[0]
            batch_tile_indptr[request_index + 1] = rows_resolved

        row_selection = _exact_row_selection(
            intervals,
            point_count=self._attributes_or_raise().point_count,
            expected_row_count=rows_resolved,
        )
        location = np.ascontiguousarray(
            self._array(TILE_MAJOR_LOCATION).get_orthogonal_selection((row_selection, slice(None))),
            dtype=np.float32,
        )
        value_id = np.ascontiguousarray(
            self._array(TILE_MAJOR_VALUE_ID).get_orthogonal_selection((row_selection,)),
            dtype=np.uint32,
        )
        if location.shape != (rows_resolved, 2) or value_id.shape != (rows_resolved,):
            raise RuntimeError("Bucket display selection returned unexpected aligned array shapes.")

        # Partition the shared Zarr results without fetching or copying rows
        # again: indptr [0, 3, 8] gives request payloads [0:3] and [3:8].
        payloads: list[_PointDisplayPayload] = []
        # Each requested tile has n_points >= 1, and no value filtering occurs
        # here, so start < stop for every payload. Empty filtered results are
        # handled by callers, not by this complete-tile reader.
        for tile_start, tile_stop in zip(batch_tile_indptr[:-1], batch_tile_indptr[1:], strict=True):
            start, stop = int(tile_start), int(tile_stop)
            payloads.append(_PointDisplayPayload(location=location[start:stop, :], value_id=value_id[start:stop]))
        return tuple(payloads)

    def set_tile_descriptors(self, descriptors: tuple[_TileDescriptor, ...]) -> None:
        """Validate manifest-derived addressing once, without reading sparse ranges.

        Installation is atomic: a corrupt bucket never acquires accepted
        addressing. Reusing the identical immutable tuple performs no IO.
        The tuple borrows the cache reader's existing descriptor objects; no
        parallel derived offset array is retained after storage comparison.
        """
        self._require_open()
        if not isinstance(descriptors, tuple) or not descriptors:
            raise ValueError("Complete-tile descriptors must be a nonempty tuple.")
        if self._tile_descriptors is descriptors:
            return
        if self._tile_descriptors is not None:
            raise ValueError("Complete-tile addressing cannot be replaced on an open bucket reader.")
        attributes = self._attributes_or_raise()
        if len(descriptors) != attributes.tile_count:
            raise ValueError("Manifest complete-tile counts disagree with the bucket.")
        # _PointsCacheReader._load_runtime_indexes() already groups descriptors
        # by bucket and constructs contiguous tile indexes and point intervals.
        # Repeat those consistency checks here so this reader does not rely on
        # every caller having obtained its descriptors through that method.
        row_start = 0
        previous_coordinate: tuple[int, int] | None = None
        for bucket_tile_index, descriptor in enumerate(descriptors):
            if self._require_bucket_tile_index(descriptor) != bucket_tile_index:
                raise ValueError("Bucket-local tile indexes must be contiguous from zero.")
            if descriptor.bucket_row_start != row_start:
                raise ValueError("Complete-tile point intervals must be contiguous from zero.")
            coordinate = (descriptor.tile_y, descriptor.tile_x)
            if previous_coordinate is not None and coordinate <= previous_coordinate:
                raise ValueError("Complete-tile coordinates must follow unique (tile_y, tile_x) order.")
            previous_coordinate = coordinate
            row_start += descriptor.n_points
        if row_start != attributes.point_count:
            raise ValueError("Manifest complete-tile counts disagree with the bucket.")
        # Descriptors come from the manifest; the offsets and coordinates below
        # come independently from the opened bucket. Compare them before accepting
        # the tuple: constructing valid descriptors alone does not establish
        # that the manifest and the opened bucket agree on tile addressing.
        stored_offsets = np.asarray(self._array(TILE_MAJOR_TILE_OFFSET)[:], dtype=np.uint64)
        expected_offsets = [descriptor.bucket_row_start for descriptor in descriptors]
        expected_offsets.append(row_start)
        if not np.array_equal(stored_offsets, np.asarray(expected_offsets, dtype=np.uint64)):
            raise ValueError("Manifest complete-tile offsets disagree with the bucket.")
        for name, expected in (
            (TILE_MAJOR_TILE_X, [descriptor.tile_x for descriptor in descriptors]),
            (TILE_MAJOR_TILE_Y, [descriptor.tile_y for descriptor in descriptors]),
        ):
            if not np.array_equal(self._array(name)[:], expected):
                raise ValueError("Manifest tile coordinates disagree with the bucket.")
        self._tile_descriptors = descriptors

    def resolve_complete_tile_interval(self, descriptor: _TileDescriptor) -> tuple[int, int]:
        """Resolve one complete tile using the bucket's accepted descriptor tuple."""
        self._require_open()
        bucket_tile_index = self._require_bucket_tile_index(descriptor)
        if self._tile_descriptors is None:
            raise RuntimeError("Complete-tile descriptors are not installed; call set_tile_descriptors() first.")
        accepted = self._tile_descriptors[bucket_tile_index]
        # Constant-time tile-index lookup prevents a different descriptor from
        # borrowing validation. Production reuses the identical object;
        # independently constructed, value-equal descriptors are also valid.
        if descriptor is not accepted and descriptor != accepted:
            raise ValueError("Requested tile descriptor disagrees with accepted bucket addressing.")
        return descriptor.bucket_row_start, descriptor.bucket_row_start + descriptor.n_points

    def _construction_tile_interval(self, descriptor: _TileDescriptor) -> tuple[int, int]:
        """Resolve and verify one tile's bucket-global point-row interval.

        The descriptor's bucket identity, bucket-local tile index, coordinates,
        row start, and point count must agree with the opened bucket and its
        ``tile_offset`` array before the half-open ``(start, stop)`` interval
        is returned.

        Parameters
        ----------
        descriptor
            Identity and expected complete point interval of the logical tile.

        Returns
        -------
        start, stop
            Half-open row bounds into the aligned bucket-wide point arrays.
        """
        self._require_open()
        if not isinstance(descriptor, _TileDescriptor):
            raise ValueError("`descriptor` must be a _TileDescriptor.")
        if (descriptor.level, descriptor.bucket_id) != (self._level, self._bucket_id):
            raise ValueError("Tile descriptor belongs to a different bucket.")
        attributes = self._attributes_or_raise()
        bucket_tile_index = descriptor.bucket_tile_index
        if bucket_tile_index >= attributes.tile_count:
            raise ValueError("Tile descriptor's bucket-local tile index is out of bounds.")
        stored_coordinates = (
            int(self._array(TILE_MAJOR_TILE_X)[bucket_tile_index]),
            int(self._array(TILE_MAJOR_TILE_Y)[bucket_tile_index]),
        )
        if stored_coordinates != (descriptor.tile_x, descriptor.tile_y):
            raise ValueError("Tile descriptor coordinates disagree with the bucket index.")
        offsets = np.asarray(
            self._array(TILE_MAJOR_TILE_OFFSET)[bucket_tile_index : bucket_tile_index + 2], dtype=np.uint64
        )
        start, stop = (int(value) for value in offsets)
        if not 0 <= start < stop <= attributes.point_count:
            raise ValueError("Tile point offsets are invalid.")
        if start != descriptor.bucket_row_start:
            raise ValueError("Tile descriptor row start disagrees with the bucket offsets.")
        if stop - start != descriptor.n_points:
            raise ValueError("Tile descriptor count disagrees with the bucket offsets.")
        return start, stop

    def _require_bucket_tile_index(self, descriptor: _TileDescriptor) -> int:
        if not isinstance(descriptor, _TileDescriptor):
            raise ValueError("`descriptor` must be a _TileDescriptor.")
        if (descriptor.level, descriptor.bucket_id) != (self._level, self._bucket_id):
            raise ValueError("Tile descriptor belongs to a different bucket.")
        bucket_tile_index = descriptor.bucket_tile_index
        if bucket_tile_index >= self._attributes_or_raise().tile_count:
            raise ValueError("Tile descriptor's bucket-local tile index is out of bounds.")
        return bucket_tile_index

    def _strict_array(self, name: str) -> zarr.Array:
        """Return a required array configured to reject missing chunks.

        A missing physical chunk indicates an incomplete or corrupt cache and
        must fail rather than silently yielding the array's fill value.
        """
        root = self._root
        if root is None:
            raise RuntimeError("Bucket Zarr group is not open.")
        node = root[name]
        if not isinstance(node, zarr.Array):
            raise ValueError(f"Required bucket node is not an array: {name}.")
        return node.with_config({"read_missing_chunks": ZARR_READ_MISSING_CHUNKS})

    def _array(self, name: str) -> zarr.Array:
        self._require_open()
        try:
            return self._arrays[name]
        except KeyError as error:
            raise RuntimeError(f"Required bucket array is not open: {name}.") from error

    def _attributes_or_raise(self) -> _BucketAttributes:
        if self._attributes is None:
            raise RuntimeError("Bucket attributes are not open.")
        return self._attributes

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("Bucket reader is not open.")

    def _close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None
        self._root = None
        self._attributes = None
        self._arrays = {}
        self._tile_descriptors = None
        self._open = False


def _exact_row_selection(
    intervals: Iterable[tuple[int, int]],
    *,
    point_count: int,
    expected_row_count: int,
) -> slice | npt.NDArray[np.int64]:
    """Return the cheapest exact row selector for one bucket batch.

    The input consists of ordered, nonoverlapping, half-open bucket row
    intervals. Selection follows this policy::

        exact half-open intervals
                   |
                   v
        merge only intervals whose boundaries touch
                   |
                   v
        one resulting interval?
            yes -> slice(start, stop)
            no  -> exact C-contiguous int64 row array

    Touching intervals can be merged without selecting unrelated rows. Gaps are
    never filled merely to form a larger slice. Consequently, the returned
    selector always addresses exactly ``expected_row_count`` point rows.

    The slice specialization avoids allocating an ``int64`` row array and lets
    Zarr use its cheaper contiguous-selection path for complete tiles and other
    genuinely contiguous batches. Disjoint selections still use one orthogonal
    integer selector so Zarr can coordinate the complete bucket batch without
    materializing the rows inside its gaps.

    Intervals must be nonempty, ordered, nonoverlapping, and contained within
    the bucket's point arrays. This function also reconciles their total length
    with ``expected_row_count`` before returning a selector.

    This is the bucket point-row counterpart of
    ``_exact_value_tile_row_selection`` in the cache-level reader. Their
    domain-specific validation remains separate; both delegate the validated
    interval transformation to the shared storage utility.
    """
    interval_iterator = iter(intervals)
    try:
        first_interval = next(interval_iterator)
    except StopIteration:
        raise ValueError("`intervals` must be nonempty.") from None

    _require_integer_in_range(point_count, "point_count", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(expected_row_count, "expected_row_count", minimum=1, maximum=_INT64_MAX)
    validated_intervals: list[tuple[int, int]] = []
    observed_row_count = 0
    previous_stop: int | None = None
    for interval in chain((first_interval,), interval_iterator):
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("Every point interval must be a (start, stop) pair.")
        start, stop = interval
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop <= point_count:
            raise ValueError("Point intervals must be nonempty and lie inside the bucket point arrays.")
        if previous_stop is not None and start < previous_stop:
            raise ValueError("Point intervals must be ordered and nonoverlapping.")
        observed_row_count += stop - start
        validated_intervals.append((start, stop))
        previous_stop = stop
    if observed_row_count != expected_row_count:
        raise ValueError("Point intervals do not reconcile to the expected batch row count.")
    return _build_exact_row_selection(validated_intervals)
