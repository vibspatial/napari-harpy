from __future__ import annotations

from collections.abc import Iterable, Sequence
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
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
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


@dataclass(frozen=True)
class _ResolvedSelectedValueRange:
    """Retain one selected value's labelled bucket-global point range."""

    value_id: int
    row_start: int
    row_count: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.value_id, "value_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.row_start, "row_start", maximum=_INT64_MAX)
        _require_integer_in_range(self.row_count, "row_count", minimum=1, maximum=_INT64_MAX)
        if self.row_start > _INT64_MAX - self.row_count:
            raise ValueError("Selected point range exceeds the supported row domain.")

    @property
    def row_stop(self) -> int:
        """Return the exclusive bucket-global point-row endpoint."""
        return self.row_start + self.row_count

    @property
    def interval(self) -> tuple[int, int]:
        """Return the unlabelled half-open interval used for physical selection."""
        return self.row_start, self.row_stop


@dataclass(frozen=True)
class _BucketLookupIndex:
    """Retain one bucket's immutable metadata-to-point-row lookup.

    Keeping these arrays resident is a deliberate runtime policy. Every
    visualization request needs them to translate logical tile and value
    selections into exact point-array intervals. Loading them once avoids
    repeating small Zarr selections and shard decoding during viewport and
    value-selection changes.

    Only lookup metadata is retained. Point coordinates and point-level values
    remain chunked on disk, and cache-level loading enforces an explicit memory
    budget.

    The resident fields correspond to the persisted bucket arrays as follows::

        Resident field          Zarr array
        -----------------------------------------------
        tile_offset             tile_offset
        tile_indptr             ranges/tile_indptr
        range_value_id          ranges/value_id
        range_row_start         ranges/row_start
        range_row_count         ranges/row_count
    """

    level: int
    bucket_id: int
    tile_offset: npt.NDArray[np.uint64]
    tile_indptr: npt.NDArray[np.uint64]
    range_value_id: npt.NDArray[np.uint32]
    range_row_start: npt.NDArray[np.uint64]
    range_row_count: npt.NDArray[np.uint64]

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)
        arrays = (
            ("tile_offset", self.tile_offset, np.uint64),
            ("tile_indptr", self.tile_indptr, np.uint64),
            ("range_value_id", self.range_value_id, np.uint32),
            ("range_row_start", self.range_row_start, np.uint64),
            ("range_row_count", self.range_row_count, np.uint64),
        )
        for name, array, dtype in arrays:
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.dtype(dtype)
                or array.ndim != 1
                or not array.flags.c_contiguous
            ):
                raise ValueError(f"`{name}` must be a one-dimensional C-contiguous {np.dtype(dtype).name} array.")
            read_only = array.view()
            read_only.flags.writeable = False
            object.__setattr__(self, name, read_only)
        if len(self.tile_offset) != len(self.tile_indptr):
            raise ValueError("Tile point and range pointer arrays must have equal lengths.")
        if len(self.tile_offset) < 2:
            raise ValueError("Bucket lookup pointers must describe at least one tile.")
        if not (len(self.range_value_id) == len(self.range_row_start) == len(self.range_row_count)):
            raise ValueError("Sparse range arrays must be row-aligned.")

    @property
    def tile_count(self) -> int:
        """Return the number of indexed logical tiles."""
        return len(self.tile_offset) - 1

    @property
    def resident_bytes(self) -> int:
        """Return bytes in the five retained NumPy buffers."""
        return sum(
            array.nbytes
            for array in (
                self.tile_offset,
                self.tile_indptr,
                self.range_value_id,
                self.range_row_start,
                self.range_row_count,
            )
        )


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

    Notes
    -----
    The context opens one bucket once and configures every array to fail on a
    missing chunk or shard. Construction payloads read every tile row including
    mandatory point IDs. Display payloads require explicit lookup-index loading, omit
    point IDs, and resolve complete or selected intervals exclusively from the
    resident tile pointers and sparse ranges.
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
        self._lookup_index: _BucketLookupIndex | None = None
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
                zarr_format=3,
                use_consolidated=False,
            )
            _validate_hierarchy(self._root)
            self._attributes = _parse_root_attributes(
                dict(self._root.attrs),
                expected_level=self._level,
                expected_bucket_id=self._bucket_id,
            )
            self._arrays = {
                name: self._strict_array(name)
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
        location = np.ascontiguousarray(self._array("location")[start:stop, :], dtype=np.float32)
        value_id = np.ascontiguousarray(self._array("value_id")[start:stop], dtype=np.uint32)
        point_id = np.ascontiguousarray(self._array("point_id")[start:stop], dtype=np.uint64)
        return _PointPayload(
            x_rel=np.ascontiguousarray(location[:, 0]),
            y_rel=np.ascontiguousarray(location[:, 1]),
            value_id=value_id,
            point_id=point_id,
        )

    def read_display_payload(
        self,
        descriptor: _TileDescriptor,
        selected_value_ids: npt.NDArray[np.uint32] | None = None,
    ) -> _PointDisplayPayload | None:
        """Read one display payload through the canonical plural batch path."""
        return self.read_display_payloads(((descriptor, selected_value_ids),))[0]

    def read_display_payloads(
        self,
        requests: tuple[tuple[_TileDescriptor, npt.NDArray[np.uint32] | None], ...],
    ) -> tuple[_PointDisplayPayload | None, ...]:
        """Read requested logical tiles in one coordinated operation for this bucket.

        Every request first resolves to exact bucket-global point intervals from
        the resident lookup index. Physically touching intervals become one
        basic slice; otherwise one exact C-contiguous ``int64`` row selector is
        used.

        ``location`` is always read from Zarr using that orthogonal row selection.
        All requests in one call use the same selection mode. In all-values mode
        (``selected_value_ids=None``), the same selector is also applied to the
        point-level ``value_id`` Zarr array. In selected-values mode, the aligned
        output IDs are reconstructed from the labelled resident ranges and the
        point-level ``value_id`` array is not accessed. Point IDs are never selected.

        Parameters
        ----------
        requests
            Nonempty tuple of ``(descriptor, selected_value_ids)`` pairs from
            this bucket in increasing bucket-local tile order. ``None`` selects
            every point in that tile; otherwise the IDs must be strictly
            increasing and unique. Every request in one call must use the same
            selection mode: either all ``selected_value_ids`` are ``None``, or
            every request provides a nonempty selected-value array. The arrays
            may differ between requests because values occur in different tiles;
            only their presence or absence must agree within one call.

        Returns
        -------
        tuple of _PointDisplayPayload or None
            Results aligned with ``requests``. ``None`` denotes a selected-value
            request for which the logical tile contains no requested value.

        Notes
        -----
        ``load_lookup_index()`` must have completed first. The transient
        ``batch_tile_indptr`` constructed here partitions the returned point
        arrays by request; it is unrelated to the persisted sparse-range
        ``tile_indptr``.
        """
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("`requests` must be a nonempty tuple.")
        if any(not isinstance(request, tuple) or len(request) != 2 for request in requests):
            raise ValueError("Every display request must be a (descriptor, selected_value_ids) pair.")

        is_subset_mode = requests[0][1] is not None
        if any((request[1] is not None) != is_subset_mode for request in requests[1:]):
            raise ValueError(
                "Display requests must be homogeneous: every `selected_value_ids` must be None "
                "or every request must provide selected value IDs."
            )

        batch_tile_indptr = np.empty(len(requests) + 1, dtype=np.uint64)
        batch_tile_indptr[0] = 0
        complete_intervals: list[tuple[int, int]] = []
        selected_ranges: list[_ResolvedSelectedValueRange] = []
        rows_resolved = 0
        previous_tile_index: int | None = None
        for request_index, request in enumerate(requests):
            descriptor, selected_value_ids = request
            if selected_value_ids is None:
                interval = self.resolve_complete_tile_interval(descriptor)
                complete_intervals.append(interval)
                tile_row_count = interval[1] - interval[0]
            else:
                resolved = self.resolve_selected_tile_intervals(descriptor, selected_value_ids)
                if resolved is None:
                    tile_row_count = 0
                else:
                    resolved_ranges, tile_row_count = resolved
                    selected_ranges.extend(resolved_ranges)
            tile_index = descriptor.bucket_tile_index
            if previous_tile_index is not None and tile_index <= previous_tile_index:
                raise ValueError("Display requests must follow increasing bucket-local tile order.")
            previous_tile_index = tile_index
            rows_resolved += tile_row_count
            batch_tile_indptr[request_index + 1] = rows_resolved

        if rows_resolved == 0:
            return (None,) * len(requests)

        row_intervals: Iterable[tuple[int, int]]
        if is_subset_mode:
            row_intervals = (selected_range.interval for selected_range in selected_ranges)
        else:
            row_intervals = complete_intervals
        row_selection = _exact_row_selection(
            row_intervals,
            point_count=self._attributes_or_raise().point_count,
            expected_row_count=rows_resolved,
        )
        location = np.ascontiguousarray(
            self._array("location").get_orthogonal_selection((row_selection, slice(None))),
            dtype=np.float32,
        )
        # Selected ranges already carry canonical value IDs in row-selection
        # order. Synthesizing the aligned IDs is intentional: reading the
        # point-level `value_id` array here would reintroduce the sparse
        # many-chunk decoding bottleneck that selected-value reads avoid.
        if is_subset_mode:
            value_id = _synthesize_selected_value_ids(
                selected_ranges,
                expected_row_count=rows_resolved,
            )
        else:
            value_id = np.ascontiguousarray(
                self._array("value_id").get_orthogonal_selection((row_selection,)),
                dtype=np.uint32,
            )
        if location.shape != (rows_resolved, 2) or value_id.shape != (rows_resolved,):
            raise RuntimeError("Bucket display selection returned unexpected aligned array shapes.")

        # Split the combined Zarr result back into request-aligned tile payloads.
        # For example, indptr [0, 3, 3, 8] maps the three requests to rows
        # [0:3], no selected rows, and [3:8]. These NumPy slices are views whose
        # backing storage remains the shared batch arrays; no point rows are
        # fetched again or copied merely to create the per-tile payloads.
        payloads: list[_PointDisplayPayload | None] = []
        for tile_start, tile_stop in zip(batch_tile_indptr[:-1], batch_tile_indptr[1:], strict=True):
            start = int(tile_start)
            stop = int(tile_stop)
            if start == stop:
                payloads.append(None)
                continue
            payloads.append(
                _PointDisplayPayload(
                    location=location[start:stop, :],
                    value_id=value_id[start:stop],
                )
            )
        return tuple(payloads)

    @property
    def projected_lookup_bytes(self) -> int:
        """Return resident bytes required by this bucket's lookup arrays."""
        attributes = self._attributes_or_raise()
        pointer_bytes = 2 * (attributes.tile_count + 1) * np.dtype(np.uint64).itemsize
        range_bytes = attributes.range_count * (np.dtype(np.uint32).itemsize + 2 * np.dtype(np.uint64).itemsize)
        return pointer_bytes + range_bytes

    @property
    def resident_lookup_bytes(self) -> int:
        """Return currently retained lookup bytes, or zero before loading."""
        return 0 if self._lookup_index is None else self._lookup_index.resident_bytes

    @property
    def lookup_index_loaded(self) -> bool:
        """Return whether the bucket's immutable lookup metadata is resident."""
        return self._lookup_index is not None

    def load_lookup_index(self) -> None:
        """Load and retain this bucket's trusted lookup metadata once.

        Publication-time validation already reconciled the logical contents.
        Runtime loading therefore copies only the five lookup arrays and never
        selects coordinates or point payload arrays.
        """
        self._require_open()
        if self._lookup_index is not None:
            return

        tile_offset = np.ascontiguousarray(self._array("tile_offset")[:], dtype=np.uint64)
        tile_indptr = np.ascontiguousarray(self._array("ranges/tile_indptr")[:], dtype=np.uint64)
        range_value_id = np.ascontiguousarray(self._array("ranges/value_id")[:], dtype=np.uint32)
        range_row_start = np.ascontiguousarray(self._array("ranges/row_start")[:], dtype=np.uint64)
        range_row_count = np.ascontiguousarray(self._array("ranges/row_count")[:], dtype=np.uint64)
        lookup = _BucketLookupIndex(
            level=self._level,
            bucket_id=self._bucket_id,
            tile_offset=tile_offset,
            tile_indptr=tile_indptr,
            range_value_id=range_value_id,
            range_row_start=range_row_start,
            range_row_count=range_row_count,
        )
        if lookup.resident_bytes != self.projected_lookup_bytes:
            raise RuntimeError("Bucket lookup bytes differ from the preflight projection.")
        self._lookup_index = lookup

    def release_lookup_index(self) -> None:
        """Release resident lookup buffers while keeping Zarr handles open."""
        self._lookup_index = None

    def resolve_complete_tile_interval(self, descriptor: _TileDescriptor) -> tuple[int, int]:
        """Resolve one complete tile using only the resident lookup index."""
        index = self._lookup_index_or_raise()
        tile_index = self._require_lookup_tile_index(descriptor, index)
        start = int(index.tile_offset[tile_index])
        stop = int(index.tile_offset[tile_index + 1])
        if stop - start != descriptor.n_points:
            raise ValueError("Tile descriptor count disagrees with the resident bucket offsets.")
        return start, stop

    def resolve_selected_tile_intervals(
        self,
        descriptor: _TileDescriptor,
        selected_value_ids: npt.NDArray[np.uint32],
    ) -> tuple[tuple[_ResolvedSelectedValueRange, ...], int] | None:
        """Resolve labelled selected point ranges using only resident lookup arrays."""
        self._require_selected_value_ids(selected_value_ids)
        index = self._lookup_index_or_raise()
        tile_index = self._require_lookup_tile_index(descriptor, index)
        tile_start = int(index.tile_offset[tile_index])
        tile_stop = int(index.tile_offset[tile_index + 1])
        range_start = int(index.tile_indptr[tile_index])
        range_stop = int(index.tile_indptr[tile_index + 1])
        range_values = index.range_value_id[range_start:range_stop]
        positions = np.searchsorted(range_values, selected_value_ids)
        in_bounds = positions < len(range_values)
        matches = np.zeros(len(selected_value_ids), dtype=np.bool_)
        matches[in_bounds] = range_values[positions[in_bounds]] == selected_value_ids[in_bounds]
        selected_positions = positions[matches]
        if len(selected_positions) == 0:
            return None

        selected_values = range_values[selected_positions]
        row_starts = index.range_row_start[range_start:range_stop][selected_positions]
        row_counts = index.range_row_count[range_start:range_stop][selected_positions]
        # Each selected sparse range becomes a half-open, bucket-global row
        # interval into the aligned point arrays while retaining the canonical
        # value and count that will reconstruct its output IDs. These are neither
        # range-array indexes nor tile-local offsets.
        resolved_ranges = tuple(
            _ResolvedSelectedValueRange(
                value_id=int(value_id),
                row_start=int(row_start),
                row_count=int(row_count),
            )
            for value_id, row_start, row_count in zip(
                selected_values,
                row_starts,
                row_counts,
                strict=True,
            )
        )
        if any(
            selected_range.row_start < tile_start or selected_range.row_stop > tile_stop
            for selected_range in resolved_ranges
        ):
            raise ValueError("Selected sparse ranges are outside the logical tile interval.")
        return resolved_ranges, sum(selected_range.row_count for selected_range in resolved_ranges)

    def _construction_tile_interval(self, descriptor: _TileDescriptor) -> tuple[int, int]:
        """Resolve and verify one tile's bucket-global point-row interval.

        The descriptor's bucket identity, local tile index, coordinates, and
        point count must agree with the opened bucket and its ``tile_offset``
        array before the half-open ``(start, stop)`` interval is returned.

        Parameters
        ----------
        descriptor
            Compact identity and expected point count of the logical tile.

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
        index = descriptor.bucket_tile_index
        if index >= attributes.tile_count:
            raise ValueError("Tile descriptor bucket-local index is out of bounds.")
        stored_coordinates = (
            int(self._array("tile_x")[index]),
            int(self._array("tile_y")[index]),
        )
        if stored_coordinates != (descriptor.tile_x, descriptor.tile_y):
            raise ValueError("Tile descriptor coordinates disagree with the bucket index.")
        offsets = np.asarray(self._array("tile_offset")[index : index + 2], dtype=np.uint64)
        start, stop = (int(value) for value in offsets)
        if not 0 <= start < stop <= attributes.point_count:
            raise ValueError("Tile point offsets are invalid.")
        if stop - start != descriptor.n_points:
            raise ValueError("Tile descriptor count disagrees with the bucket offsets.")
        return start, stop

    def _require_lookup_tile_index(self, descriptor: _TileDescriptor, index: _BucketLookupIndex) -> int:
        if not isinstance(descriptor, _TileDescriptor):
            raise ValueError("`descriptor` must be a _TileDescriptor.")
        if (descriptor.level, descriptor.bucket_id) != (index.level, index.bucket_id):
            raise ValueError("Tile descriptor belongs to a different bucket.")
        tile_index = descriptor.bucket_tile_index
        if tile_index >= index.tile_count:
            raise ValueError("Tile descriptor bucket-local index is out of bounds.")
        return tile_index

    @staticmethod
    def _require_selected_value_ids(value: npt.NDArray[np.uint32]) -> None:
        if not isinstance(value, np.ndarray):
            raise ValueError("`selected_value_ids` must be a NumPy array.")
        if value.ndim != 1 or value.dtype != np.dtype(np.uint32) or not value.flags.c_contiguous:
            raise ValueError("`selected_value_ids` must be a one-dimensional C-contiguous uint32 array.")
        if len(value) == 0:
            raise ValueError("`selected_value_ids` must contain at least one value ID.")
        if bool((value[1:] <= value[:-1]).any()):
            raise ValueError("`selected_value_ids` must be strictly increasing and unique.")

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
        return node.with_config({"read_missing_chunks": False})

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

    def _lookup_index_or_raise(self) -> _BucketLookupIndex:
        self._require_open()
        if self._lookup_index is None:
            raise RuntimeError("Bucket lookup index is not loaded; prime it before display reads.")
        return self._lookup_index

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
        self._lookup_index = None
        self._open = False


def _synthesize_selected_value_ids(
    resolved_ranges: Sequence[_ResolvedSelectedValueRange],
    *,
    expected_row_count: int,
) -> npt.NDArray[np.uint32]:
    """Construct aligned point-level value IDs from labelled selected ranges."""
    if not resolved_ranges:
        raise ValueError("`resolved_ranges` must be nonempty.")
    _require_integer_in_range(
        expected_row_count,
        "expected_row_count",
        minimum=1,
        maximum=_INT64_MAX,
    )
    value_id = np.empty(expected_row_count, dtype=np.uint32)
    cursor = 0
    for selected_range in resolved_ranges:
        stop = cursor + selected_range.row_count
        if stop > expected_row_count:
            raise ValueError("Selected range counts exceed the expected output row count.")
        value_id[cursor:stop] = selected_range.value_id
        cursor = stop
    if cursor != expected_row_count:
        raise ValueError("Selected range counts do not reconcile to the expected output row count.")
    return value_id


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
    ``_exact_value_tile_row_selection`` in the cache-level reader. The helpers
    deliberately remain separate because this function validates untyped point
    row pairs, while its counterpart validates value-major catalog intervals.
    """
    interval_iterator = iter(intervals)
    try:
        first_interval = next(interval_iterator)
    except StopIteration:
        raise ValueError("`intervals` must be nonempty.") from None

    _require_integer_in_range(point_count, "point_count", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(expected_row_count, "expected_row_count", minimum=1, maximum=_INT64_MAX)
    merged: list[tuple[int, int]] = []
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
        if previous_stop is not None and start == previous_stop:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
        previous_stop = stop
    if observed_row_count != expected_row_count:
        raise ValueError("Point intervals do not reconcile to the expected batch row count.")

    if len(merged) == 1:
        start, stop = merged[0]
        return slice(start, stop)

    # Begin with output positions and shift each destination segment in place to
    # its bucket-global interval. This fills one selector allocation without
    # constructing one Python integer per returned point.
    selected_rows = np.arange(observed_row_count, dtype=np.int64)
    cursor = 0
    for start, stop in merged:
        count = stop - start
        selected_rows[cursor : cursor + count] += start - cursor
        cursor += count
    return selected_rows
