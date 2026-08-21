from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt
import zarr
from zarr.storage import LocalStore

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
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
class _PointReadPlan:
    """Describe exact rows and minimal coalesced Zarr read envelopes."""

    intervals: tuple[tuple[int, int], ...]
    blocks: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _BucketLookupIndex:
    """Retain one bucket's immutable metadata-to-point-row lookup.

    Keeping these arrays resident is a deliberate runtime policy. Every
    visualization request needs them to translate logical tile and value
    selections into exact point-array intervals. Loading them once avoids
    repeating small Zarr selections and shard decoding during viewport and
    value-selection changes.

    Only lookup metadata is retained. Point coordinates and point-level values
    remain chunked on disk, and cache-level priming enforces an explicit memory
    budget.
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
    mandatory point IDs. Display payloads require explicit lookup priming, omit
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
        plan = self._point_read_plan(((start, stop),))
        location, value_id, point_id = self._read_aligned_rows(plan, include_point_id=True)
        if point_id is None:
            raise RuntimeError("Construction reads must include point IDs.")
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
        """Read exact display rows while decoding each touched chunk once.

        ``None`` reads every point row in the logical tile. For selected values,
        sparse range records identify the exact bucket-global rows, while Zarr
        decompresses at inner-chunk granularity. Reading every range separately
        could therefore decode the same chunk repeatedly. The selected path
        performs the following conversion::

            selected sparse ranges
                -> exact point-row intervals
                -> touched inner-chunk IDs
                -> groups of overlapping or consecutive touched chunks
                -> one minimal row-envelope read per group and aligned array
                -> exact selected rows extracted from those blocks

        A coalesced read may include unselected rows between selected intervals,
        but its outer bounds remain the first and last exact selected rows. Only
        exact selected rows enter the returned arrays. Complete and selected
        reads slice only aligned ``location`` and ``value_id`` arrays; the
        point-ID payload is never read or decoded.

        Parameters
        ----------
        descriptor
            Identity and expected point count of the logical tile.
        selected_value_ids
            Optional strictly increasing unique value IDs to retrieve. ``None``
            reads every value in the tile.

        Returns
        -------
        _PointDisplayPayload or None
            Exact aligned display rows, or ``None`` when a selected-value
            request finds no requested value in the tile.

        Notes
        -----
        ``load_lookup_index()`` must have completed before this method is
        called. There is no viewport-time fallback to Zarr lookup arrays.
        """
        if selected_value_ids is None:
            intervals = (self.resolve_complete_tile_interval(descriptor),)
        else:
            resolved = self.resolve_selected_tile_intervals(descriptor, selected_value_ids)
            if resolved is None:
                return None
            intervals, _ = resolved
        plan = self._point_read_plan(intervals)
        location, value_id, point_id = self._read_aligned_rows(plan, include_point_id=False)
        if point_id is not None:
            raise RuntimeError("Visualization reads must not include point IDs.")
        return _PointDisplayPayload(
            location=location,
            value_id=value_id,
        )

    @property
    def projected_lookup_bytes(self) -> int:
        """Return resident bytes required by this bucket's lookup arrays."""
        attributes = self._attributes_or_raise()
        pointer_bytes = 2 * (attributes.tile_count + 1) * np.dtype(np.uint64).itemsize
        range_bytes = attributes.range_count * (
            np.dtype(np.uint32).itemsize + 2 * np.dtype(np.uint64).itemsize
        )
        return pointer_bytes + range_bytes

    @property
    def resident_lookup_bytes(self) -> int:
        """Return currently retained lookup bytes, or zero before priming."""
        return 0 if self._lookup_index is None else self._lookup_index.resident_bytes

    @property
    def lookup_index_loaded(self) -> bool:
        """Return whether the bucket's immutable lookup metadata is resident."""
        return self._lookup_index is not None

    def load_lookup_index(self) -> None:
        """Load and retain this bucket's trusted lookup metadata once.

        Publication-time validation already reconciled the logical contents.
        Runtime priming therefore copies only the five lookup arrays and never
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
    ) -> tuple[tuple[tuple[int, int], ...], int] | None:
        """Resolve exact selected point rows using only resident lookup arrays."""
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

        row_starts = index.range_row_start[range_start:range_stop][selected_positions]
        row_counts = index.range_row_count[range_start:range_stop][selected_positions]
        # Each selected sparse range becomes a half-open, bucket-global row
        # interval into the aligned point arrays. These are neither range-array
        # indexes nor tile-local offsets.
        intervals = tuple((int(start), int(start + count)) for start, count in zip(row_starts, row_counts, strict=True))
        if any(start < tile_start or stop > tile_stop or start >= stop for start, stop in intervals):
            raise ValueError("Selected sparse ranges are outside the logical tile interval.")
        return intervals, sum(stop - start for start, stop in intervals)

    def _point_read_plan(self, intervals: tuple[tuple[int, int], ...]) -> _PointReadPlan:
        """Plan aligned Zarr reads for exact bucket-global row intervals.

        The caller supplies the exact nonempty point rows that must be returned
        from one logical tile. This method coalesces intervals whose touched
        chunks overlap or are consecutive into minimal Zarr slice envelopes,
        preventing repeated decoding of the same inner chunk while leaving
        gaps containing untouched chunks unread.

        The returned ``intervals`` remain the exact output rows, whereas
        ``blocks`` are the physical slices consumed by ``_read_aligned_rows``.
        No point payload arrays are read here.

        Parameters
        ----------
        intervals
            Exact half-open row bounds into the aligned bucket-wide point
            arrays. Callers have already verified that they are nonempty and
            lie inside one logical tile.

        Returns
        -------
        _PointReadPlan
            Exact output intervals and coalesced physical read blocks.
        """
        value_array = self._array("value_id")
        chunk_rows = value_array.chunks[0]
        blocks = _coalesced_read_blocks_for_intervals(intervals, chunk_rows=chunk_rows)
        return _PointReadPlan(
            intervals=intervals,
            blocks=blocks,
        )

    def _read_aligned_rows(
        self,
        plan: _PointReadPlan,
        *,
        include_point_id: bool,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32], npt.NDArray[np.uint64] | None]:
        location_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] | None = [] if include_point_id else None
        # Read each coalesced envelope once to avoid repeated Zarr chunk
        # decoding, then append only its exact selected intervals.
        for block_start, block_stop in plan.blocks:
            location = np.asarray(self._array("location")[block_start:block_stop, :], dtype=np.float32)
            values = np.asarray(self._array("value_id")[block_start:block_stop], dtype=np.uint32)
            points = (
                np.asarray(self._array("point_id")[block_start:block_stop], dtype=np.uint64)
                if include_point_id
                else None
            )
            for interval_start, interval_stop in plan.intervals:
                if interval_stop <= block_start or interval_start >= block_stop:
                    continue
                local_start = max(interval_start, block_start) - block_start
                local_stop = min(interval_stop, block_stop) - block_start
                location_parts.append(np.ascontiguousarray(location[local_start:local_stop, :]))
                value_parts.append(np.ascontiguousarray(values[local_start:local_stop]))
                if point_parts is not None:
                    if points is None:
                        raise RuntimeError("Point-ID rows were not read for a construction payload.")
                    point_parts.append(np.ascontiguousarray(points[local_start:local_stop]))

        return (
            _concatenate_location_parts(location_parts),
            _concatenate_parts(value_parts, np.uint32),
            None if point_parts is None else _concatenate_parts(point_parts, np.uint64),
        )

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

def _coalesced_read_blocks_for_intervals(
    intervals: tuple[tuple[int, int], ...],
    *,
    chunk_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Return minimal row envelopes for connected touched-chunk runs.

    The input intervals are half-open bucket-global point-row bounds. Overlapping
    touched-chunk spans share one read to prevent repeated decoding; consecutive
    spans also share one read to reduce Zarr slice requests, while gaps between
    untouched chunks remain separate. Each returned block retains the first and
    last exact interval bounds rather than expanding them to the outer chunk
    edges. The caller has already verified that the nonempty intervals remain
    within one logical tile.

    Examples
    --------
    With four rows per chunk, ``(1, 2)`` and ``(3, 4)`` both touch chunk 0,
    so they become one exact read envelope, ``(1, 4)``. The interval
    ``(8, 9)`` touches chunk 2 and remains separate because chunk 1 is not
    touched. The resulting blocks are therefore ``((1, 4), (8, 9))``.
    """
    ordered = sorted(intervals, key=lambda interval: interval[0])
    block_start, block_stop = ordered[0]
    last_chunk = (block_stop - 1) // chunk_rows
    blocks: list[tuple[int, int]] = []
    for start, stop in ordered[1:]:
        first_chunk = start // chunk_rows
        interval_last_chunk = (stop - 1) // chunk_rows
        if first_chunk > last_chunk + 1:
            blocks.append((block_start, block_stop))
            block_start, block_stop = start, stop
            last_chunk = interval_last_chunk
            continue
        block_stop = max(block_stop, stop)
        last_chunk = max(last_chunk, interval_last_chunk)
    blocks.append((block_start, block_stop))
    return tuple(blocks)


def _concatenate_parts(parts: list[np.ndarray], dtype: npt.DTypeLike) -> np.ndarray:
    if len(parts) == 1:
        return np.ascontiguousarray(parts[0], dtype=dtype)
    return np.ascontiguousarray(np.concatenate(parts), dtype=dtype)


def _concatenate_location_parts(parts: list[np.ndarray]) -> npt.NDArray[np.float32]:
    if len(parts) == 1:
        return np.ascontiguousarray(parts[0], dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
