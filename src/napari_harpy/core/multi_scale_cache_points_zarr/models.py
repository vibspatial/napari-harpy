from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from napari_harpy.core.multi_scale_cache_points_zarr.storage._paths import tile_major_bucket_path

_INT16_MAX = 2**15 - 1
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1
type _SerializedLevelKind = Literal["exact", "bridge", "spatial"]


def _expected_level_kind(level: int) -> _SerializedLevelKind:
    """Return the one valid semantic kind for a serialized cache level."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    if level == 0:
        return "exact"
    if level == 1:
        return "bridge"
    return "spatial"


@dataclass(frozen=True)
class _TileDescriptor:
    """Identify one nonempty logical tile in one finalized Zarr bucket.

    Parameters
    ----------
    level
        Non-negative serialized cache-level number containing the tile.
    bucket_id
        Deterministic identifier of the Zarr bucket containing the tile. Together
        with ``level``, it determines the canonical ``bucket_path`` property.
    bucket_tile_index
        Zero-based index of this tile among all nonempty tiles in its bucket,
        after ordering them by ``(tile_y, tile_x)``. For index ``i``, the bucket
        stores this tile's identity at ``tile_x[i]`` and ``tile_y[i]``, its
        complete point interval at ``tile_offset[i:i + 2]``, and its sparse
        value-range interval at ``tile_indptr[i:i + 2]``. It is not a point
        offset, chunk number, shard number, or Parquet row group.
    bucket_row_start
        Zero-based start in the bucket's aligned tile-major ``location``,
        point-level ``value_id``, and ``point_id`` arrays. Together with
        ``n_points``, it gives the complete tile interval
        ``[bucket_row_start, bucket_row_start + n_points)``. This is not a
        spatial origin, byte offset, or row address in the value-major cache.
    tile_x
        Logical x index of the tile in this cache level's aligned tile grid.
    tile_y
        Logical y index of the tile in this cache level's aligned tile grid.
    n_points
        Number of stored points in the complete logical tile. It must equal
        ``tile_offset[i + 1] - tile_offset[i]`` in the finalized bucket, where
        ``i`` is ``bucket_tile_index``.

    Notes
    -----
    ``bucket_path`` is derived canonically from ``level`` and ``bucket_id`` so
    those integer fields are the only stored source of bucket identity.

    The writer obtains row starts from its planned bucket offsets. On reopening,
    manifest readers derive them once from complete bucket counts, including
    offscreen tiles. Counts ``[3, 5, 2]`` produce row starts ``[0, 3, 8]``;
    tile index 1 therefore occupies rows ``[3:8)``. ``bucket_tile_index`` has
    the same name in the descriptor and the persisted manifest; no row-start
    column is stored in the manifest.

    Before normal tile-major display reads, the bucket reader validates the complete
    descriptor tuple against stored offsets and tile coordinates. Subsequent
    reads use the accepted descriptor directly, without retaining another
    derived offset array. Construction and diagnostic reads independently
    reconcile their addresses with persisted bucket pointers.

    Value membership is deliberately absent. A tile can contain a variable and
    potentially large number of distinct ``value_id`` values; duplicating them
    here would turn the compact descriptor into a second sparse index. For
    diagnostic subset reads, ``bucket_tile_index`` locates records through
    ``tile_indptr[i:i + 2]``, and those records store each present value together
    with its point-row start and count.
    """

    level: int
    bucket_id: int
    bucket_tile_index: int
    bucket_row_start: int
    tile_x: int
    tile_y: int
    n_points: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.bucket_tile_index, "bucket_tile_index", maximum=_UINT32_MAX)
        _require_integer_in_range(self.bucket_row_start, "bucket_row_start", maximum=_INT64_MAX)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.n_points, "n_points", minimum=1, maximum=_INT64_MAX)
        if self.bucket_row_start + self.n_points > _INT64_MAX:
            raise ValueError("Tile point interval exceeds the supported row domain.")

    @property
    def bucket_path(self) -> str:
        """Return the canonical cache-relative Zarr path for this bucket."""
        return _bucket_path(level=self.level, bucket_id=self.bucket_id)


def _bucket_path(*, level: int, bucket_id: int) -> str:
    """Return the canonical path derived from serialized bucket identity."""
    _require_integer_in_range(level, "level", maximum=_INT16_MAX)
    _require_integer_in_range(bucket_id, "bucket_id", maximum=_UINT32_MAX)
    return tile_major_bucket_path(level=level, bucket_id=bucket_id)


def _require_integer_in_range(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"`{name}` must be an integer in the range [{minimum}, {maximum}].")
    return value
