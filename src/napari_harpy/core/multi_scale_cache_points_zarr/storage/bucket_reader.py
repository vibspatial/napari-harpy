from __future__ import annotations

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


class _BucketReader:
    """Reuse strict read-only handles for complete and selected bucket reads.

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
    missing chunk or shard. Complete reads use ``tile_offset``. Selected reads
    resolve sparse ranges through ``tile_indptr`` and deduplicate the touched
    inner point chunks before constructing a complete ``_PointPayload``.
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

    def read_complete(self, descriptor: _TileDescriptor) -> _PointPayload:
        """Read all aligned rows for one verified logical tile."""
        start, stop = self._tile_interval(descriptor)
        location = np.asarray(self._array("location")[start:stop, :], dtype=np.float32)
        return _PointPayload(
            x_rel=np.ascontiguousarray(location[:, 0]),
            y_rel=np.ascontiguousarray(location[:, 1]),
            value_id=np.ascontiguousarray(self._array("value_id")[start:stop], dtype=np.uint32),
            point_id=np.ascontiguousarray(self._array("point_id")[start:stop], dtype=np.uint64),
        )

    def read_selected(
        self,
        descriptor: _TileDescriptor,
        selected_value_ids: npt.NDArray[np.uint32],
    ) -> _PointPayload | None:
        """Read exact selected value runs while decoding each touched point chunk once."""
        self._require_selected_value_ids(selected_value_ids)
        tile_start, tile_stop = self._tile_interval(descriptor)
        tile_index = descriptor.bucket_tile_index
        range_bounds = np.asarray(
            self._array("ranges/tile_indptr")[tile_index : tile_index + 2],
            dtype=np.uint64,
        )
        range_start, range_stop = (int(value) for value in range_bounds)
        if not 0 <= range_start < range_stop <= self._attributes_or_raise().range_count:
            raise ValueError("Tile range pointers are invalid.")

        range_values = np.asarray(
            self._array("ranges/value_id")[range_start:range_stop],
            dtype=np.uint32,
        )
        positions = np.searchsorted(range_values, selected_value_ids)
        in_bounds = positions < len(range_values)
        matches = np.zeros(len(selected_value_ids), dtype=np.bool_)
        matches[in_bounds] = range_values[positions[in_bounds]] == selected_value_ids[in_bounds]
        selected_positions = positions[matches]
        if len(selected_positions) == 0:
            return None

        row_starts = np.asarray(
            self._array("ranges/row_start")[range_start:range_stop],
            dtype=np.uint64,
        )[selected_positions]
        row_counts = np.asarray(
            self._array("ranges/row_count")[range_start:range_stop],
            dtype=np.uint64,
        )[selected_positions]
        intervals = tuple(
            (int(start), int(start + count))
            for start, count in zip(row_starts, row_counts, strict=True)
        )
        if any(start < tile_start or stop > tile_stop or start >= stop for start, stop in intervals):
            raise ValueError("Selected sparse ranges are outside the logical tile interval.")

        attributes = self._attributes_or_raise()
        blocks = _chunk_blocks_for_intervals(
            intervals,
            chunk_rows=attributes.settings.point_chunk_rows,
            point_count=attributes.point_count,
        )
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        for block_start, block_stop in blocks:
            location = np.asarray(self._array("location")[block_start:block_stop, :], dtype=np.float32)
            values = np.asarray(self._array("value_id")[block_start:block_stop], dtype=np.uint32)
            points = np.asarray(self._array("point_id")[block_start:block_stop], dtype=np.uint64)
            for interval_start, interval_stop in intervals:
                if interval_stop <= block_start or interval_start >= block_stop:
                    continue
                local_start = max(interval_start, block_start) - block_start
                local_stop = min(interval_stop, block_stop) - block_start
                x_parts.append(np.ascontiguousarray(location[local_start:local_stop, 0]))
                y_parts.append(np.ascontiguousarray(location[local_start:local_stop, 1]))
                value_parts.append(np.ascontiguousarray(values[local_start:local_stop]))
                point_parts.append(np.ascontiguousarray(points[local_start:local_stop]))

        return _PointPayload(
            x_rel=_concatenate_parts(x_parts, np.float32),
            y_rel=_concatenate_parts(y_parts, np.float32),
            value_id=_concatenate_parts(value_parts, np.uint32),
            point_id=_concatenate_parts(point_parts, np.uint64),
        )

    def _tile_interval(self, descriptor: _TileDescriptor) -> tuple[int, int]:
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
        self._open = False


def _chunk_blocks_for_intervals(
    intervals: tuple[tuple[int, int], ...],
    *,
    chunk_rows: int,
    point_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return maximal consecutive inner-chunk blocks touched by row intervals."""
    chunk_ids: set[int] = set()
    for start, stop in intervals:
        chunk_ids.update(range(start // chunk_rows, (stop - 1) // chunk_rows + 1))
    ordered = sorted(chunk_ids)
    blocks: list[tuple[int, int]] = []
    first = previous = ordered[0]
    for chunk_id in ordered[1:]:
        if chunk_id != previous + 1:
            blocks.append((first * chunk_rows, min((previous + 1) * chunk_rows, point_count)))
            first = chunk_id
        previous = chunk_id
    blocks.append((first * chunk_rows, min((previous + 1) * chunk_rows, point_count)))
    return tuple(blocks)


def _concatenate_parts(parts: list[np.ndarray], dtype: npt.DTypeLike) -> np.ndarray:
    if len(parts) == 1:
        return np.ascontiguousarray(parts[0], dtype=dtype)
    return np.ascontiguousarray(np.concatenate(parts), dtype=dtype)
