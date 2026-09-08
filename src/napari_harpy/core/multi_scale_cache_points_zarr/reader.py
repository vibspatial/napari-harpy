"""Read one published multiscale Zarr points cache for visualization.

This module is deliberately independent of napari. A later adapter translates
camera and canvas state into :class:`_IntrinsicViewport` and supplies an
effective point budget. The reader owns only cache-level planning and payload
access.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import TracebackType
from typing import Literal

import numpy as np
import numpy.typing as npt
import zarr

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    PUBLICATION_STATE_COMPLETE,
    _CacheAttributes,
    _LevelMetadata,
)
from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _INT16_MAX,
    _INT64_MAX,
    _UINT32_MAX,
    _require_integer_in_range,
    _TileDescriptor,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._row_selection import (
    _build_exact_row_selection,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage._schema import (
    CATALOG_ARRAY_DTYPES,
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_N_POINTS,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    VALUES_N_POINTS,
    value_major_location,
    value_major_point_indptr,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import (
    _BucketReader,
    _PointDisplayPayload,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CacheRootReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache
from napari_harpy.core.multi_scale_cache_points_zarr.storage.value_major_reader import _ValueMajorLocationReader

_ViewportReadRoute = Literal["tile_major_all_values", "value_major_subset"]


@dataclass(frozen=True)
class _IntrinsicViewport:
    """Represent one half-open rectangle in intrinsic transcript coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite real number.")
            object.__setattr__(self, name, float(value))
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("An intrinsic viewport must have positive width and height.")


@dataclass(frozen=True)
class _CacheLevelInfo:
    """Expose one serialized level without leaking cache-format models."""

    level: int
    kind: str
    tile_size: int
    grid_width: int
    grid_height: int
    max_points_per_tile: int | None
    bucket_count: int
    tile_count: int
    point_count: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        if self.kind not in {"exact", "bridge", "spatial"}:
            raise ValueError("`kind` must be exact, bridge, or spatial.")
        for name, maximum in (
            ("tile_size", _INT64_MAX),
            ("grid_width", _UINT32_MAX + 1),
            ("grid_height", _UINT32_MAX + 1),
            ("bucket_count", _UINT32_MAX + 1),
            ("tile_count", _INT64_MAX),
            ("point_count", _INT64_MAX),
        ):
            _require_integer_in_range(getattr(self, name), name, minimum=1, maximum=maximum)
        if self.max_points_per_tile is not None:
            _require_integer_in_range(
                self.max_points_per_tile,
                "max_points_per_tile",
                minimum=1,
                maximum=_INT64_MAX,
            )


@dataclass(frozen=True)
class _CacheDatasetInfo:
    """Expose immutable cache information needed by a transcript viewer."""

    cache_generation_id: str
    points_name: str
    value_column: str
    value_names: tuple[str, ...]
    x_origin: float
    y_origin: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    levels: tuple[_CacheLevelInfo, ...]
    overview_point_budget: int

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        if not self.points_name or not self.value_column:
            raise ValueError("Dataset point and value-column names must be nonempty.")
        if not self.value_names or any(not isinstance(value, str) or not value for value in self.value_names):
            raise ValueError("`value_names` must contain nonempty strings.")
        if len(set(self.value_names)) != len(self.value_names):
            raise ValueError("`value_names` must be unique.")
        for name in ("x_origin", "y_origin", "x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite float.")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Dataset bounds must not be inverted.")
        if (
            not isinstance(self.levels, tuple)
            or not self.levels
            or not all(isinstance(level, _CacheLevelInfo) for level in self.levels)
            or tuple(level.level for level in self.levels) != tuple(range(len(self.levels)))
        ):
            raise ValueError("`levels` must contain consecutive cache-level summaries.")
        _require_integer_in_range(
            self.overview_point_budget,
            "overview_point_budget",
            minimum=1,
            maximum=_INT64_MAX,
        )


@dataclass(frozen=True)
class _PlannedTileRead:
    """Retain one positive tile's logical coordinates and physical identity.

    Value membership stays in the viewport plan's shared selected-level index,
    not in a separate array per tile. Physical value-major blocks are resolved
    from that index only after CPU residency identifies the missing tiles.
    """

    level: int
    tile_x: int
    tile_y: int
    manifest_row: int
    bucket_id: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.manifest_row, "manifest_row", maximum=_INT64_MAX)
        _require_integer_in_range(self.bucket_id, "bucket_id", maximum=_UINT32_MAX)

    @property
    def tile_key(self) -> tuple[int, int, int]:
        """Return the hashable logical identity ``(level, tile_x, tile_y)``."""
        return self.level, self.tile_x, self.tile_y


@dataclass(frozen=True)
class _ViewportReadPlan:
    """Describe an immutable, generation-bound viewport read without payload IO.

    ``requested_value_ids`` records the complete value selection once for the
    plan. ``route`` fixes its physical payload before any read begins. A
    selected-value plan retains the exact immutable level index from which its
    positive tiles were derived, so sidecar addressing never consults mutable
    reader selection state or reloads catalog records.
    """

    cache_generation_id: str
    requested_value_ids: tuple[int, ...] | None
    level: int
    requests: tuple[_PlannedTileRead, ...]
    route: _ViewportReadRoute
    selected_value_level_index: _SelectedValueLevelIndex | None

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        _require_requested_value_ids(self.requested_value_ids)
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        if self.route not in ("tile_major_all_values", "value_major_subset"):
            raise ValueError("Viewport plan has an unsupported physical route.")
        if self.route == "tile_major_all_values":
            if self.requested_value_ids is not None or self.selected_value_level_index is not None:
                raise ValueError("An all-values route cannot retain a selected-value index.")
        elif (
            self.requested_value_ids is None
            or not isinstance(self.selected_value_level_index, _SelectedValueLevelIndex)
            or len(self.selected_value_level_index.value_indptr) != len(self.requested_value_ids) + 1
        ):
            raise ValueError("A value-major route requires its aligned selected-value level index.")
        if not isinstance(self.requests, tuple) or not all(
            isinstance(request, _PlannedTileRead) for request in self.requests
        ):
            raise ValueError("`requests` must be a tuple of _PlannedTileRead values.")
        keys = tuple(request.tile_key for request in self.requests)
        if len(set(keys)) != len(keys):
            raise ValueError("Viewport plan tile keys must be unique.")
        if tuple((tile_y, tile_x) for _, tile_x, tile_y in keys) != tuple(
            sorted((tile_y, tile_x) for _, tile_x, tile_y in keys)
        ):
            raise ValueError("Viewport plan tiles must follow manifest spatial order.")
        if any(level != self.level for level, _, _ in keys):
            raise ValueError("Every viewport plan request must belong to the plan level.")

    @property
    def tile_keys(self) -> tuple[tuple[int, int, int], ...]:
        """Return positive logical tiles in manifest spatial order."""
        return tuple(request.tile_key for request in self.requests)

    @property
    def required_bucket_keys(self) -> tuple[tuple[int, int], ...]:
        """Return sorted physical bucket addresses needed by the complete plan."""
        return tuple(sorted({(self.level, request.bucket_id) for request in self.requests}))


@dataclass(frozen=True)
class _TileReadResult:
    """Return display rows for one nonempty logical tile.

    Parameters
    ----------
    level
        Cache level containing the tile.
    tile_x, tile_y
        Logical tile coordinates within the level grid.
    tile_size
        Intrinsic width and height of the tile. Although derived from
        ``level``, it is included so consumers can position the returned
        tile-relative coordinates without accessing private level metadata.
    location
        Tile-relative ``(x, y)`` coordinates with shape ``(N, 2)``.
    value_id
        Value identifier aligned with each location row.

    Notes
    -----
    Given the cache origin, intrinsic coordinates are reconstructed as::

        x = x_origin + tile_x * tile_size + location[:, 0]
        y = y_origin + tile_y * tile_size + location[:, 1]
    """

    level: int
    tile_x: int
    tile_y: int
    tile_size: int
    location: npt.NDArray[np.float32]
    value_id: npt.NDArray[np.uint32]

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(self.tile_x, "tile_x", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_y, "tile_y", maximum=_UINT32_MAX)
        _require_integer_in_range(self.tile_size, "tile_size", minimum=1, maximum=_INT64_MAX)
        _require_display_arrays(self.location, self.value_id)
        location = self.location.view()
        location.flags.writeable = False
        value_id = self.value_id.view()
        value_id.flags.writeable = False
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "value_id", value_id)


@dataclass(frozen=True)
class _ViewportReadResult:
    """Return ordered positive tiles for one viewport."""

    level: int
    tiles: tuple[_TileReadResult, ...]

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        if not isinstance(self.tiles, tuple) or not all(isinstance(tile, _TileReadResult) for tile in self.tiles):
            raise ValueError("`tiles` must be a tuple of _TileReadResult values.")
        if any(tile.level != self.level for tile in self.tiles):
            raise ValueError("Every viewport tile must belong to the selected level.")
        if tuple((tile.tile_y, tile.tile_x) for tile in self.tiles) != tuple(
            sorted((tile.tile_y, tile.tile_x) for tile in self.tiles)
        ):
            raise ValueError("Viewport tiles must follow manifest spatial order.")


@dataclass(frozen=True)
class _LevelSelection:
    """Return one catalog-only LOD decision together with its evidence.

    Parameters
    ----------
    level
        Selected serialized cache level.
    estimated_point_count
        Complete positive-tile rows estimated for the request at ``level``.
    positive_visible_tile_count
        Intersecting manifest tiles contributing at least one estimated row.
    within_budget
        Whether ``estimated_point_count`` satisfies the runtime point budget.
    omitted_value_ids
        For a value-filtered request, sorted IDs that had a positive Exact
        visible count but zero visible count at ``level``. An empty array means
        no Exact-visible selected value was omitted. ``None`` means no value
        filter was supplied.
    """

    level: int
    estimated_point_count: int
    positive_visible_tile_count: int
    within_budget: bool
    omitted_value_ids: npt.NDArray[np.uint32] | None

    def __post_init__(self) -> None:
        _require_integer_in_range(self.level, "level", maximum=_INT16_MAX)
        _require_integer_in_range(
            self.estimated_point_count,
            "estimated_point_count",
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(
            self.positive_visible_tile_count,
            "positive_visible_tile_count",
            maximum=_INT64_MAX,
        )
        if (self.estimated_point_count == 0) != (self.positive_visible_tile_count == 0):
            raise ValueError("Estimated points and positive tiles must be empty together.")
        if not isinstance(self.within_budget, bool):
            raise ValueError("`within_budget` must be bool.")
        if self.omitted_value_ids is None:
            return
        if (
            not isinstance(self.omitted_value_ids, np.ndarray)
            or self.omitted_value_ids.ndim != 1
            or self.omitted_value_ids.dtype != np.dtype(np.uint32)
        ):
            raise ValueError("`omitted_value_ids` must be a one-dimensional uint32 array or None.")
        if bool((self.omitted_value_ids[1:] <= self.omitted_value_ids[:-1]).any()):
            raise ValueError("`omitted_value_ids` must be strictly increasing.")
        omitted_value_ids = np.ascontiguousarray(self.omitted_value_ids).view()
        omitted_value_ids.flags.writeable = False
        object.__setattr__(self, "omitted_value_ids", omitted_value_ids)


@dataclass(frozen=True)
class _SelectedValueLevelIndex:
    """Retain one level's selected value-to-tile catalog records in memory.

    ``value_indptr`` partitions the aligned ``manifest_index`` and ``n_points``
    arrays by selected-value position. Empty intervals deliberately preserve a
    selected value that has no serialized records at this level.
    """

    value_indptr: npt.NDArray[np.uint64]
    manifest_index: npt.NDArray[np.uint64]
    n_points: npt.NDArray[np.uint64]

    def __post_init__(self) -> None:
        value_indptr = _read_only_index_array(self.value_indptr, "value_indptr", np.uint64)
        manifest_index = _read_only_index_array(self.manifest_index, "manifest_index", np.uint64)
        n_points = _read_only_index_array(self.n_points, "n_points", np.uint64)
        if len(value_indptr) == 0 or int(value_indptr[0]) != 0:
            raise ValueError("`value_indptr` must be nonempty and start at zero.")
        if bool((value_indptr[1:] < value_indptr[:-1]).any()) or int(value_indptr[-1]) != len(manifest_index):
            raise ValueError("`value_indptr` must be nondecreasing and terminate at the record count.")
        if len(n_points) != len(manifest_index) or bool((n_points == 0).any()):
            raise ValueError("`n_points` must contain one positive count per manifest record.")
        for start, stop in pairwise(value_indptr.tolist()):
            if stop - start > 1 and bool((manifest_index[start + 1 : stop] <= manifest_index[start : stop - 1]).any()):
                raise ValueError("Manifest rows must be strictly increasing within every selected value.")
        object.__setattr__(self, "value_indptr", value_indptr)
        object.__setattr__(self, "manifest_index", manifest_index)
        object.__setattr__(self, "n_points", n_points)

    @property
    def resident_bytes(self) -> int:
        """Return bytes in the three retained NumPy buffers."""
        return self.value_indptr.nbytes + self.manifest_index.nbytes + self.n_points.nbytes


@dataclass(frozen=True)
class _SelectedValueIndex:
    """Retain one proper subset's immutable value-to-tile index in memory.

    This object is an auxiliary catalog index, not the logical selection
    itself. It exists only for a nonempty proper subset of the cache
    vocabulary. Reader APIs represent an all-values selection by the absence
    of this auxiliary index (``None``), allowing that path to use the resident
    manifest and tile-major payload without loading every ``value_tiles``
    record.

    Parameters
    ----------
    cache_generation_id
        Cache generation that owns the selected catalog records. This prevents
        an index loaded from one immutable generation from being reused with
        another.
    value_ids
        Nonempty, strictly increasing canonical ``uint32`` IDs for the proper
        subset. Position ``i`` identifies the corresponding interval in every
        level's ``value_indptr``.
    levels
        One level index for every consecutive serialized cache level. Each
        level retains all ``value_tiles/manifest_index`` and aligned
        ``value_tiles/n_points`` records for the selected values, including
        records outside the current viewport. Viewport planning intersects
        these resident records with visible manifest rows; value-major reads
        also use the complete ordered counts to derive per-tile sidecar
        offsets.

    Notes
    -----
    Point locations, point-level value IDs, and unselected value-to-tile
    records are not retained here. The contained NumPy arrays are made
    read-only so the same generation-bound index can be reused across LOD
    selection, viewport planning, and payload reads until the selection
    changes.
    """

    cache_generation_id: str
    value_ids: npt.NDArray[np.uint32]
    levels: tuple[_SelectedValueLevelIndex, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cache_generation_id, str):
            raise ValueError("`cache_generation_id` must be a canonical UUID string.")
        try:
            parsed_uuid = uuid.UUID(self.cache_generation_id)
        except ValueError as error:
            raise ValueError("`cache_generation_id` must be a canonical UUID string.") from error
        if str(parsed_uuid) != self.cache_generation_id:
            raise ValueError("`cache_generation_id` must be a canonical lowercase UUID string.")
        value_ids = _read_only_index_array(self.value_ids, "value_ids", np.uint32)
        if len(value_ids) == 0 or bool((value_ids[1:] <= value_ids[:-1]).any()):
            raise ValueError("`value_ids` must be nonempty and strictly increasing.")
        if not isinstance(self.levels, tuple) or not self.levels:
            raise ValueError("`levels` must be a nonempty tuple.")
        if not all(isinstance(level, _SelectedValueLevelIndex) for level in self.levels):
            raise ValueError("Every selected-value level must be a _SelectedValueLevelIndex.")
        if any(len(level.value_indptr) != len(value_ids) + 1 for level in self.levels):
            raise ValueError("Every level index must preserve every selected-value interval.")
        object.__setattr__(self, "value_ids", value_ids)

    @property
    def resident_bytes(self) -> int:
        """Return bytes in all retained NumPy buffers."""
        return self.value_ids.nbytes + sum(level.resident_bytes for level in self.levels)


@dataclass(frozen=True)
class _ValueMajorReadBlock:
    """Address one selected value/tile block in ``value_major/location``.

    ``row_start`` and ``row_count`` identify the physical source rows.
    ``manifest_row`` identifies the logical destination tile, and ``value_id``
    labels the rows when the tile-oriented output is assembled.
    """

    value_id: int
    manifest_row: int
    row_start: int
    row_count: int

    def __post_init__(self) -> None:
        _require_integer_in_range(self.value_id, "value_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.manifest_row, "manifest_row", maximum=_INT64_MAX)
        _require_integer_in_range(self.row_start, "row_start", maximum=_INT64_MAX)
        _require_integer_in_range(self.row_count, "row_count", minimum=1, maximum=_INT64_MAX)
        if self.row_start > _INT64_MAX - self.row_count:
            raise ValueError("Value-major read block exceeds the supported row domain.")

    @property
    def interval(self) -> tuple[int, int]:
        """Return the block's half-open ``value_major/location`` interval."""
        return self.row_start, self.row_start + self.row_count


@dataclass(frozen=True)
class _ValueTileInterval:
    """Identify one requested value's exact half-open catalog rows."""

    selected_value_position: int
    value_id: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        _require_integer_in_range(
            self.selected_value_position,
            "selected_value_position",
            maximum=_INT64_MAX,
        )
        _require_integer_in_range(self.value_id, "value_id", maximum=_UINT32_MAX)
        _require_integer_in_range(self.start, "start", maximum=_INT64_MAX)
        _require_integer_in_range(self.stop, "stop", minimum=1, maximum=_INT64_MAX)
        if self.start >= self.stop:
            raise ValueError("`start` must be smaller than `stop`.")


class _PointsCacheReader:
    """Coordinate metadata and point reads for one completed cache generation.

    Cache-wide lookup metadata
    --------------------------
    Cache-wide metadata used to plan and interpret reads from tile-major and
    value-major storage:

    - ``manifest/*``: tile coordinates, physical bucket addresses, and point counts.
    - ``values/n_points``: per-value totals.
    - ``value_tiles/*``: manifest tiles containing each value, with their point counts.

    Memory residency
    ----------------
    Resident data means materialized data in memory, not merely Zarr array
    references. The table names cache-relative arrays whose contents are loaded
    into NumPy arrays; the persisted arrays remain on disk::

        Boundary                   Cache fields loaded into memory                     Retained by
        -------------------------  --------------------------------------------------  ------------------
        Startup (complete arrays)  manifest/level_indptr                               _PointsCacheReader
                                   manifest/bucket_id
                                   manifest/bucket_tile_index
                                   manifest/tile_x
                                   manifest/tile_y
                                   manifest/n_points
                                   values/n_points
                                   value_tiles/indptr
        Startup (every level)      value_major/level_N/value_point_indptr              _PointsCacheReader
        New selected-value subset  value_tiles/manifest_index                          Caller
        (selected rows)            value_tiles/n_points
        Tile-major payload         tile_major/level_N/bucket-....zarr/location         Caller
        (requested tile rows)      tile_major/level_N/bucket-....zarr/value_id
        Value-major payload        value_major/level_N/location                        Caller
        (requested intervals)

    ``level_N`` and ``bucket-....zarr`` stand for the relevant level and bucket.
    The table lists persisted sources; additional runtime data is derived:

    - Startup constructs ``_TileDescriptor`` objects and lookup mappings from
      the stored manifest arrays. These Python objects exist only in memory
      and are rebuilt whenever the reader opens the cache.
    - Selected rows from ``value_tiles/manifest_index`` and
      ``value_tiles/n_points`` populate the per-level records retained by
      ``_SelectedValueIndex``, which also retains the selected value IDs. Each
      contained ``_SelectedValueLevelIndex`` stores derived ``value_indptr``
      pointers that partition its in-memory ``manifest_index`` and ``n_points``
      arrays by selected-value position.
    - Value-major payloads reconstruct aligned point-level value IDs. There is
      no point-level ``value_id`` array in value-major storage.

    ``load_selected_value_index()`` supplies records for the selected values
    across every level, including tiles outside the viewport. While the selected
    value IDs remain unchanged, pan and zoom reuse the existing
    ``_SelectedValueIndex``. They neither reload its records from
    ``value_tiles/manifest_index`` and ``value_tiles/n_points`` nor construct
    a new ``_SelectedValueIndex``. CPU tile residency is managed outside this
    reader.

    A ``zarr.Array`` object refers to stored data, not an in-memory NumPy copy of
    its point payload. Opening that object or constructing a location-reader
    wrapper does not decode locations; point rows are read only when requested.

    Reader ownership
    ----------------
    Owned readers and shared references::

        _PointsCacheReader
            +-- _CacheRootReader
            |     Parsed cache-root attributes (_CacheAttributes)
            |     Zarr array objects (cache-relative paths):
            |       manifest/*
            |       values/n_points
            |       value_tiles/*
            |       value_major/level_N/location
            |       value_major/level_N/value_point_indptr
            +-- _ValueMajorLocationReader per level
            |     borrows the location Zarr object from _CacheRootReader
            |     (no additional store)
            +-- _BucketReaderCache
                  (level, bucket_id) -> lazily opened _BucketReader
                    Bucket identity: level and bucket_id
                    Path: tile_major/level_N/bucket-....zarr
                    Opened bucket store and root Zarr group
                    Parsed bucket attributes (_BucketAttributes)
                    Zarr array objects (cache-relative paths):
                      tile_major/level_N/bucket-....zarr/location
                      tile_major/level_N/bucket-....zarr/point_id
                      tile_major/level_N/bucket-....zarr/value_id
                      tile_major/level_N/bucket-....zarr/tile_x
                      tile_major/level_N/bucket-....zarr/tile_y
                      tile_major/level_N/bucket-....zarr/tile_offset
                      tile_major/level_N/bucket-....zarr/ranges/*
                    Accepted tuple of _TileDescriptor objects (once installed):
                      borrowed from _PointsCacheReader._descriptors_by_bucket
                      neither the tuple nor its descriptors are copied

    - ``_descriptors`` retains one immutable ``_TileDescriptor`` per manifest row,
      in manifest order. ``_descriptors_by_bucket`` groups the same objects without
      copying them.
    - ``bucket_tile_index`` matches the persisted manifest column.
      ``bucket_row_start`` is derived once from all preceding tile counts in the
      same level/bucket, including tiles outside the viewport. No parallel derived
      offset array is retained.
    - Bucket readers retain opened Zarr array references and accepted descriptors,
      not decoded point payloads or resident sparse-range indexes.

    Read routing and lifecycle
    --------------------------
    - Startup validates root and array layouts, without replaying independent
      staged or exhaustive validation.
    - First complete-tile use validates the full descriptor tuple against stored
      offsets, coordinates, and totals before accepting it atomically. Warm reads
      reuse that tuple without rereading addressing metadata.
    - ``read_planned_tiles()`` routes all values to batched complete tile-major reads,
      and proper subsets to value-major intervals. Both return ordered logical tiles;
      value-major locations are regrouped by tile with reconstructed aligned IDs.
    - Diagnostic ``read_tile(value_ids=...)`` reads a complete tile-major payload,
      then filters both arrays in memory. No display path loads bucket sparse ranges.
    - Closure closes owned readers/stores and releases metadata, array wrappers,
      and descriptor references. Returned selections and tiles remain caller-owned.
    """

    def __init__(self, cache_root: str | Path) -> None:
        self._cache_root = Path(cache_root)
        self._stack: ExitStack | None = None
        self._catalog: _CacheRootReader | None = None
        self._bucket_cache: _BucketReaderCache | None = None
        self._attributes: _CacheAttributes | None = None
        self._dataset_info: _CacheDatasetInfo | None = None
        self._manifest_level_indptr: npt.NDArray[np.uint64] | None = None
        self._manifest_bucket_id: npt.NDArray[np.uint32] | None = None
        self._manifest_bucket_tile_index: npt.NDArray[np.uint32] | None = None
        self._manifest_tile_x: npt.NDArray[np.uint32] | None = None
        self._manifest_tile_y: npt.NDArray[np.uint32] | None = None
        self._manifest_n_points: npt.NDArray[np.uint64] | None = None
        self._value_tiles_indptr: npt.NDArray[np.uint64] | None = None
        self._value_n_points: npt.NDArray[np.uint64] | None = None
        self._value_major_point_indptr: tuple[npt.NDArray[np.uint64], ...] = ()
        self._value_major_readers: tuple[_ValueMajorLocationReader, ...] = ()
        self._descriptors: tuple[_TileDescriptor, ...] = ()
        self._descriptors_by_bucket: dict[tuple[int, int], tuple[_TileDescriptor, ...]] = {}
        self._manifest_row_by_tile: dict[tuple[int, int, int], int] = {}
        self._resident_index_bytes = 0
        self._entered = False
        self._open = False

    def __enter__(self) -> _PointsCacheReader:
        if self._entered:
            raise RuntimeError("A points cache reader can be entered only once.")
        self._entered = True
        stack = ExitStack()
        try:
            catalog = stack.enter_context(_CacheRootReader(self._cache_root))
            attributes = catalog.attributes
            if attributes.publication_state != PUBLICATION_STATE_COMPLETE:
                raise ValueError("Cache root publication_state is not 'complete'.")
            # Retain bucket metadata after its first payload read. Entering this
            # cache does not open buckets or read sparse-range payloads.
            bucket_cache = stack.enter_context(
                _BucketReaderCache(
                    self._cache_root,
                    max_open_readers=sum(level.bucket_count for level in attributes.levels),
                )
            )
            self._catalog = catalog
            self._attributes = attributes
            self._bucket_cache = bucket_cache
            self._dataset_info = _dataset_info_from_attributes(attributes)
            self._load_runtime_indexes()
        except Exception:
            stack.close()
            self._clear_open_state()
            raise
        self._stack = stack
        self._open = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        stack = self._stack
        self._stack = None
        try:
            if stack is not None:
                stack.close()
        finally:
            self._clear_open_state()
        return False

    @property
    def value_names(self) -> tuple[str, ...]:
        """Return canonical value labels in implicit value-ID order."""
        return self._attributes_or_raise().value_names

    @property
    def dataset_info(self) -> _CacheDatasetInfo:
        """Return the immutable viewer-facing description of this cache."""
        self._require_open()
        if self._dataset_info is None:
            raise RuntimeError("Cache dataset information is not loaded.")
        return self._dataset_info

    @property
    def cache_generation_id(self) -> str:
        """Return the opened completed generation UUID."""
        return self._attributes_or_raise().cache_generation_id

    @property
    def level_count(self) -> int:
        """Return the number of serialized cache levels."""
        return len(self._attributes_or_raise().levels)

    @property
    def resident_index_bytes(self) -> int:
        """Return retained catalog and value-major pointer NumPy-array bytes.

        Python descriptors, including their complete-tile row starts, and
        grouping containers are excluded. Use descriptor count and process
        RSS alongside this array-only measurement when evaluating memory.
        """
        self._require_open()
        return self._resident_index_bytes

    @property
    def tile_descriptor_count(self) -> int:
        """Return the number of shared Python tile descriptors retained by this reader."""
        self._require_open()
        return len(self._descriptors)

    @property
    def resident_value_major_pointer_bytes(self) -> int:
        """Return bytes in the compact per-level value-major point pointers."""
        self._require_open()
        return sum(pointer.nbytes for pointer in self._value_major_point_indptr)

    @property
    def open_bucket_reader_count(self) -> int:
        """Return the number of lazily entered bucket readers."""
        return self._bucket_cache_or_raise().open_reader_count

    def read_tile(
        self,
        level: int,
        tile_x: int,
        tile_y: int,
        *,
        value_ids: npt.NDArray[np.uint32] | None = None,
    ) -> _TileReadResult | None:
        """Conveniently read one explicitly addressed tile.

        This singleton API is retained for diagnostics, acceptance checks, and
        callers that genuinely need exactly one known logical tile. It delegates
        to the canonical bucket-batch reader as a one-request batch and does not
        define a separate physical-read path.

        Do not loop over this method to fetch viewport or other multi-tile data.
        Such consumers must use :meth:`read_viewport`, which groups all logical
        requests by physical layout and preserves coordinated Zarr selection.

        This diagnostic method always reads a complete tile-major display payload
        after descriptor validation, then applies the same membership mask to
        locations and point-level value IDs when ``value_ids`` is supplied.
        It does not use sparse ranges or the value-major cache. Filtering can
        therefore read substantially more points than it returns. A missing tile
        or a tile with no matching points returns ``None``.
        """
        metadata = self._require_level(level)
        _require_integer_in_range(tile_x, "tile_x", maximum=metadata.grid_width - 1)
        _require_integer_in_range(tile_y, "tile_y", maximum=metadata.grid_height - 1)
        value_ids = self._require_value_ids(value_ids)
        manifest_row = self._manifest_row_by_tile.get((level, tile_x, tile_y))
        if manifest_row is None:
            return None
        descriptor = self._descriptors[manifest_row]
        bucket_reader = self._get_bucket_reader_for_complete_display(level=level, bucket_id=descriptor.bucket_id)
        payload = bucket_reader.read_complete_display_payload(descriptor)
        if value_ids is not None:
            matches = np.isin(payload.value_id, value_ids)
            if not bool(matches.any()):
                return None
            payload = _PointDisplayPayload(
                location=payload.location[matches],
                value_id=payload.value_id[matches],
            )
        return self._tile_result(descriptor, payload)

    def load_selected_value_index(
        self,
        value_ids: npt.NDArray[np.uint32],
        *,
        max_resident_bytes: int | None,
    ) -> _SelectedValueIndex:
        """Read and retain selected value-to-tile records for every level.

        This is the explicit selected-value catalog-I/O boundary. The returned
        immutable index is independent of a viewport and can be reused by every
        subsequent pan, zoom, LOD decision, and viewport payload request.
        Load a new index only when the selected value IDs change; ordinary
        viewport changes reuse this in-memory representation instead of
        reconstructing it from the catalog.

        The resident representation is compact relative to point payloads: it
        retains only selected value-to-tile manifest rows, aligned point counts,
        and per-value pointers—not point locations or point-level value IDs.
        Its exact NumPy-buffer footprint is projected before either large
        catalog array is read. A configured ``max_resident_bytes`` is checked
        against that projection.

        Parameters
        ----------
        value_ids
            Nonempty sorted unique canonical value IDs forming a proper subset
            of the complete cache vocabulary.
        max_resident_bytes
            Maximum retained NumPy-buffer bytes allowed for the loaded index.
            ``None`` disables the configured limit; projection and exact
            post-load byte reconciliation still run.

        Returns
        -------
        _SelectedValueIndex
            Generation-bound in-memory proper-subset index.

        Notes
        -----
        The production cache-session caller normalizes a complete-vocabulary
        tuple to its explicit all-values state before calling this method. A
        direct caller that supplies the complete vocabulary has violated that
        boundary and is rejected rather than receiving a second representation
        of the all-values state.
        """
        value_ids = self._require_value_ids(value_ids)
        if value_ids is None:
            raise ValueError("`value_ids` must be supplied when loading a selected-value index.")
        if max_resident_bytes is not None:
            _require_integer_in_range(
                max_resident_bytes,
                "max_resident_bytes",
                minimum=1,
                maximum=_INT64_MAX,
            )
        if len(value_ids) == len(self.value_names):
            raise ValueError(
                "`value_ids` must be a proper subset; normalize the complete vocabulary "
                "to the all-values path before loading an index."
            )

        pointers = self._value_tiles_indptr_or_raise()
        indexes = value_ids.astype(np.int64, copy=False)
        record_counts = pointers[:, indexes + 1] - pointers[:, indexes]
        projected_bytes = value_ids.nbytes + pointers.shape[0] * (len(value_ids) + 1) * np.dtype(np.uint64).itemsize
        projected_bytes += int(record_counts.sum(dtype=np.uint64)) * 2 * np.dtype(np.uint64).itemsize
        if max_resident_bytes is not None and projected_bytes > max_resident_bytes:
            raise ValueError(
                f"Selected-value index requires {projected_bytes} resident bytes, "
                f"exceeding `max_resident_bytes={max_resident_bytes}`."
            )

        catalog = self._catalog_or_raise()
        manifest_array = catalog.array(VALUE_TILES_MANIFEST_INDEX)
        point_count_array = catalog.array(VALUE_TILES_N_POINTS)
        levels = tuple(
            self._load_selected_value_level_index(
                level,
                value_ids,
                record_counts[level],
                manifest_array=manifest_array,
                point_count_array=point_count_array,
            )
            for level in range(self.level_count)
        )
        value_index = _SelectedValueIndex(
            cache_generation_id=self.cache_generation_id,
            value_ids=value_ids,
            levels=levels,
        )
        if value_index.resident_bytes != projected_bytes:
            raise RuntimeError("Selected-value index bytes differ from the preflight projection.")
        return value_index

    def plan_viewport(
        self,
        level: int,
        viewport: _IntrinsicViewport,
        *,
        value_index: _SelectedValueIndex | None = None,
    ) -> _ViewportReadPlan:
        """Plan positive viewport tiles entirely from resident catalog indexes.

        Parameters
        ----------
        level
            Serialized level to read.
        viewport
            Half-open intrinsic-coordinate viewport used for complete-tile
            intersection.
        value_index
            Loaded selected-value index, or ``None`` for all values.

        Returns
        -------
        _ViewportReadPlan
            Immutable generation- and selection-bound read instructions in
            manifest spatial order.

        Notes
        -----
        Planning reads no point payload and opens no bucket. It retains private
        manifest identity, the chosen physical route, and the selected level
        index needed for later sidecar addressing, while callers operate only
        on logical tile keys.
        """
        self._require_level(level)
        value_index = self._require_selected_value_index(value_index)
        requested_value_ids = (
            None if value_index is None else tuple(int(value_id) for value_id in value_index.value_ids)
        )
        visible_rows = self._visible_manifest_rows(level, viewport)

        # Planning needs only the positive-tile union, not the selected values
        # in each tile. Keep their shared level index for later missing-tile
        # addressing; even a fully resident viewport must build this plan.
        positive_rows = (
            visible_rows
            if value_index is None
            else self._positive_visible_manifest_rows(level, visible_rows, value_index)
        )
        planned = tuple(
            _PlannedTileRead(
                level=level,
                tile_x=self._descriptors[manifest_row].tile_x,
                tile_y=self._descriptors[manifest_row].tile_y,
                manifest_row=manifest_row,
                bucket_id=self._descriptors[manifest_row].bucket_id,
            )
            for manifest_row in positive_rows.tolist()
        )
        return _ViewportReadPlan(
            cache_generation_id=self.cache_generation_id,
            requested_value_ids=requested_value_ids,
            level=level,
            requests=planned,
            route="tile_major_all_values" if value_index is None else "value_major_subset",
            selected_value_level_index=None if value_index is None else value_index.levels[level],
        )

    def read_planned_tiles(
        self,
        plan: _ViewportReadPlan,
        tile_keys_to_read: tuple[tuple[int, int, int], ...],
        *,
        raise_if_cancelled: Callable[[], None] | None = None,
    ) -> _ViewportReadResult:
        """Read a unique subset of one viewport plan in original plan order.

        ``tile_keys_to_read`` identifies the plan subset to read physically.
        An empty subset performs no payload IO. ``plan.route`` selects complete
        tile-major reads for ``tile_major_all_values`` or the selected level's
        value-major sidecar for ``value_major_subset``. Tile-major reads batch
        one orthogonal Zarr selection per array and bucket.
        """
        self._require_viewport_plan_compatible(plan)
        if raise_if_cancelled is not None and not callable(raise_if_cancelled):
            raise ValueError("`raise_if_cancelled` must be callable or None.")
        if not isinstance(tile_keys_to_read, tuple) or any(
            not isinstance(key, tuple)
            or len(key) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) for value in key)
            for key in tile_keys_to_read
        ):
            raise ValueError("`tile_keys_to_read` must contain (level, tile_x, tile_y) integer tuples.")
        if len(set(tile_keys_to_read)) != len(tile_keys_to_read):
            raise ValueError("`tile_keys_to_read` must not contain duplicates.")
        tile_keys_to_read_set = set(tile_keys_to_read)
        unknown = tile_keys_to_read_set - set(plan.tile_keys)
        if unknown:
            raise ValueError("`tile_keys_to_read` contains a tile absent from the viewport plan.")
        manifest_rows = tuple(
            request.manifest_row for request in plan.requests if request.tile_key in tile_keys_to_read_set
        )
        if not manifest_rows:
            return _ViewportReadResult(level=plan.level, tiles=())
        if plan.route == "value_major_subset":
            # Plan identity was validated above. The physical helper receives
            # only the selected-level facts and manifest rows still missing
            # from CPU residency, never the complete viewport's tile requests.
            assert plan.requested_value_ids is not None
            assert plan.selected_value_level_index is not None
            return self._read_value_major_requests(
                level=plan.level,
                requested_value_ids=plan.requested_value_ids,
                selected_value_level_index=plan.selected_value_level_index,
                manifest_rows=manifest_rows,
                raise_if_cancelled=raise_if_cancelled,
            )
        return self._read_complete_tile_major_requests(
            plan.level,
            manifest_rows,
            raise_if_cancelled=raise_if_cancelled,
        )

    def read_viewport(
        self,
        level: int,
        viewport: _IntrinsicViewport,
        *,
        value_index: _SelectedValueIndex | None = None,
    ) -> _ViewportReadResult:
        """Conveniently plan and read every positive tile in one viewport."""
        plan = self.plan_viewport(level, viewport, value_index=value_index)
        return self.read_planned_tiles(plan, plan.tile_keys)

    def select_level(
        self,
        viewport: _IntrinsicViewport,
        point_budget: int,
        *,
        value_index: _SelectedValueIndex | None = None,
    ) -> _LevelSelection:
        """Choose the finest eligible visible level within the point budget.

        Parameters
        ----------
        viewport
            Intrinsic-coordinate viewport used to identify intersecting tiles.
        point_budget
            Maximum estimated visible point count for a successful selection.
        value_index
            Immutable in-memory index returned by
            :meth:`load_selected_value_index`, or ``None`` for all values. When
            supplied, only represented requested values contribute to each
            level's estimate; missing values do not make that level ineligible.

        Returns
        -------
        _LevelSelection
            Selected level, its catalog-derived visible point and positive-tile
            estimates, whether the estimate satisfies ``point_budget``, and any
            Exact-visible selected value IDs omitted at that level.

        Notes
        -----
        **Selected-index lifecycle.** ``load_selected_value_index()`` is the
        explicit catalog-I/O boundary used when the selected value IDs change.
        The caller deliberately keeps its result in memory and reuses it across
        subsequent pan, zoom, level-selection, and viewport-read operations.
        This method consumes that resident index but never reconstructs it or
        rereads the cache-wide ``value_tiles`` arrays. Keeping this preparation
        outside the viewport hot path is what makes filtered level selection
        responsive as the camera changes.

        **Level-choice policy.** Evaluate serialized levels from Exact toward the
        coarsest level. At each level, sum visible points for the requested values
        represented there. Return the first estimate at most ``point_budget``;
        sampled omission of a requested value does not make a level ineligible.
        This is deliberate: if every Exact-visible value had to survive, one rare
        value lost during sampling could invalidate every coarser level and force
        the entire multi-value request back to Exact. That would make the render
        budget ineffective and could require reading millions of points merely
        to retain one rare value. The selected level therefore follows the budget,
        while ``omitted_value_ids`` reports the values sacrificed at that LOD. If
        no level fits, return the coarsest level with ``within_budget=False``.

        **Why a value count can reappear at a coarser level.** Level estimates
        count complete logical tiles that intersect the viewport; they do not
        clip individual points to the viewport. Coarser tiles cover larger
        spatial footprints, so one can contain an existing value from an Exact
        tile that did not intersect the viewport. The one-dimensional example
        below makes that tile-footprint effect explicit::

            Exact tiles, size 10

            viewport [0, 5)
            [--------)
            +----------+----------+
            | tile 0   | tile 1   |
            | no A     | A exists |
            +----------+----------+
            0         10         20

            coarser tile, size 20

            viewport [0, 5)
            [--------)
            +---------------------+
            | coarser tile        |
            | includes sampled A  |
            +---------------------+
            0                    20

        At Exact, the viewport intersects only tile 0, where A is absent. At the
        coarser level, the same viewport intersects one tile assembled from both
        Exact footprints, including the existing A from tile 1. The pyramid has
        not created A; the complete-tile estimate has widened. This is why
        selected counts need not change monotonically with level and why a
        coarser appearance is not treated as catalog corruption.

        Level selection performs all of this work from resident manifest arrays
        and, when filtered, the selected-value index. It does not read catalog
        Zarr payloads, open bucket stores, or read point payloads.
        """
        _require_integer_in_range(point_budget, "point_budget", minimum=1, maximum=_INT64_MAX)
        value_index = self._require_selected_value_index(value_index)
        attributes = self._attributes_or_raise()

        if value_index is None:
            candidates: list[_LevelSelection] = []
            for metadata in attributes.levels:
                rows = self._visible_manifest_rows(metadata.level, viewport)
                point_count = int(self._manifest_n_points_or_raise()[rows].sum(dtype=np.uint64))
                candidate = _LevelSelection(
                    level=metadata.level,
                    estimated_point_count=point_count,
                    positive_visible_tile_count=len(rows),
                    within_budget=point_count <= point_budget,
                    omitted_value_ids=None,
                )
                candidates.append(candidate)
                if candidate.within_budget:
                    return candidate
            return candidates[-1]

        exact_present_values: npt.NDArray[np.bool_] | None = None
        # Retain the most recently evaluated candidate. If no level fits, the
        # completed loop leaves this pointing to the coarsest serialized level.
        fallback: _LevelSelection | None = None
        for metadata in attributes.levels:
            rows = self._visible_manifest_rows(metadata.level, viewport)
            point_count_by_value, positive_visible_tile_count = self._selected_value_manifest_summary(
                metadata.level,
                rows,
                value_index,
            )
            point_count = int(point_count_by_value.sum(dtype=np.uint64))
            if exact_present_values is None:
                exact_present_values = point_count_by_value > 0
            omitted_value_ids = np.ascontiguousarray(
                value_index.value_ids[exact_present_values & (point_count_by_value == 0)]
            )
            candidate = _LevelSelection(
                level=metadata.level,
                estimated_point_count=point_count,
                positive_visible_tile_count=positive_visible_tile_count,
                within_budget=point_count <= point_budget,
                omitted_value_ids=omitted_value_ids,
            )
            fallback = candidate
            if candidate.within_budget:
                # Avoid intersecting selected-value records for coarser levels once
                # the finest valid fit is known.
                return candidate

        if fallback is None:
            raise RuntimeError("Cache has no serialized levels.")
        return fallback

    def _load_runtime_indexes(self) -> None:
        """Materialize the compact catalog state needed for runtime planning.

        Load the small manifest, catalog pointers, value totals, and per-level
        value-major point pointers as read-only NumPy arrays. Construct
        manifest-row-aligned tile descriptors and an O(1) mapping from logical
        tile coordinates to manifest rows. Each frozen descriptor receives its
        complete point-row start from the preceding counts in its bucket;
        no additional derived offset array is retained. This requires neither
        opening bucket stores nor loading their sparse per-value ranges.

        Point payloads and the potentially large value-tile record arrays
        remain on disk and are read only for requested tiles and values.
        """
        catalog = self._catalog_or_raise()
        arrays = {
            name: _read_only_array(catalog, name)
            for name in (
                MANIFEST_LEVEL_INDPTR,
                MANIFEST_BUCKET_ID,
                MANIFEST_BUCKET_TILE_INDEX,
                MANIFEST_TILE_X,
                MANIFEST_TILE_Y,
                MANIFEST_N_POINTS,
                VALUE_TILES_INDPTR,
                VALUES_N_POINTS,
            )
        }
        self._manifest_level_indptr = arrays[MANIFEST_LEVEL_INDPTR]
        self._manifest_bucket_id = arrays[MANIFEST_BUCKET_ID]
        self._manifest_bucket_tile_index = arrays[MANIFEST_BUCKET_TILE_INDEX]
        self._manifest_tile_x = arrays[MANIFEST_TILE_X]
        self._manifest_tile_y = arrays[MANIFEST_TILE_Y]
        self._manifest_n_points = arrays[MANIFEST_N_POINTS]
        self._value_tiles_indptr = arrays[VALUE_TILES_INDPTR]
        self._value_n_points = arrays[VALUES_N_POINTS]
        self._value_major_point_indptr = tuple(
            _read_only_array(catalog, value_major_point_indptr(level), dtype=np.uint64)
            for level in range(self.level_count)
        )
        self._value_major_readers = tuple(
            _ValueMajorLocationReader(catalog.array(value_major_location(level))) for level in range(self.level_count)
        )
        for level, (metadata, pointer) in enumerate(
            zip(self._attributes_or_raise().levels, self._value_major_point_indptr, strict=True)
        ):
            if (
                int(pointer[0]) != 0
                or int(pointer[-1]) != metadata.point_count
                or bool((pointer[1:] < pointer[:-1]).any())
            ):
                raise ValueError(f"Value-major point pointers are invalid at level {level}.")
        self._resident_index_bytes = sum(array.nbytes for array in arrays.values()) + sum(
            pointer.nbytes for pointer in self._value_major_point_indptr
        )

        level_indptr = self._manifest_level_indptr
        descriptors: list[_TileDescriptor] = []
        descriptors_by_bucket: dict[tuple[int, int], list[_TileDescriptor]] = {}
        lookup: dict[tuple[int, int, int], int] = {}
        attributes = self._attributes_or_raise()
        for level, metadata in enumerate(attributes.levels):
            start = int(level_indptr[level])
            stop = int(level_indptr[level + 1])
            for manifest_row in range(start, stop):
                bucket_id = int(self._manifest_bucket_id[manifest_row])
                bucket_descriptors = descriptors_by_bucket.setdefault((level, bucket_id), [])
                # This indexes the tile within its bucket, not the point rows
                # occupied by that tile in the bucket's payload arrays.
                bucket_tile_index = int(self._manifest_bucket_tile_index[manifest_row])
                if bucket_tile_index != len(bucket_descriptors):
                    raise ValueError("Manifest tiles must follow contiguous bucket-local tile indexes.")
                # Include every preceding tile in this bucket, not just visible
                # or missing ones. Counts [3, 5, 2] give starts [0, 3, 8].
                previous = bucket_descriptors[-1] if bucket_descriptors else None
                row_start = previous.bucket_row_start + previous.n_points if previous is not None else 0
                descriptor = _TileDescriptor(
                    level=level,
                    bucket_id=bucket_id,
                    bucket_tile_index=bucket_tile_index,
                    bucket_row_start=row_start,
                    tile_x=int(self._manifest_tile_x[manifest_row]),
                    tile_y=int(self._manifest_tile_y[manifest_row]),
                    n_points=int(self._manifest_n_points[manifest_row]),
                )
                key = (level, descriptor.tile_x, descriptor.tile_y)
                if key in lookup:
                    raise ValueError("Manifest contains duplicate logical tile coordinates.")
                if descriptor.tile_x >= metadata.grid_width or descriptor.tile_y >= metadata.grid_height:
                    raise ValueError("Manifest tile lies outside its declared level grid.")
                lookup[key] = manifest_row
                descriptors.append(descriptor)
                bucket_descriptors.append(descriptor)
        if len(descriptors) != len(self._manifest_n_points):
            raise ValueError("Manifest pointers do not cover every resident manifest row.")
        self._descriptors = tuple(descriptors)
        # Manifest rows follow (level, tile_y, tile_x), and bucket-local tile
        # indexes follow (tile_y, tile_x). Grouping preserves that order, so no
        # sort is needed. The tile-index check above rejects inconsistent ordering.
        # Both collections refer to the same immutable descriptor objects.
        self._descriptors_by_bucket = {
            key: tuple(bucket_descriptors) for key, bucket_descriptors in descriptors_by_bucket.items()
        }
        bucket_counts_by_level = [0] * self.level_count
        for level, _bucket_id in self._descriptors_by_bucket:
            bucket_counts_by_level[level] += 1
        # Empty hash buckets are not serialized, so physical bucket IDs can
        # have gaps. Count the distinct buckets rather than bounding their IDs
        # by the number of nonempty buckets in the level metadata.
        if bucket_counts_by_level != [metadata.bucket_count for metadata in attributes.levels]:
            raise ValueError("Manifest bucket counts disagree with level metadata.")
        self._manifest_row_by_tile = lookup

    def _visible_manifest_rows(
        self,
        level: int,
        viewport: _IntrinsicViewport,
    ) -> npt.NDArray[np.int64]:
        """Return global manifest rows for nonempty tiles intersecting a viewport.

        This is a tile-bounding-box lookup over resident catalog arrays. It does
        not read point payloads, and an intersecting tile may extend beyond the
        exact viewport boundary.
        """
        metadata = self._require_level(level)
        if not isinstance(viewport, _IntrinsicViewport):
            raise ValueError("`viewport` must be _IntrinsicViewport.")
        clipped = self._clip_viewport(viewport)
        if clipped is None:
            return np.empty(0, dtype=np.int64)
        # Manifest rows are contiguous by level. These pointers select the
        # global half-open row interval containing this level's nonempty tiles.
        level_indptr = self._manifest_level_indptr_or_raise()
        start = int(level_indptr[level])
        stop = int(level_indptr[level + 1])
        tile_x = self._manifest_tile_x_or_raise()[start:stop].astype(np.float64, copy=False)
        tile_y = self._manifest_tile_y_or_raise()[start:stop].astype(np.float64, copy=False)
        # Convert tile-grid coordinates to half-open intrinsic bounds and apply
        # the vectorized rectangle-intersection test to every manifest tile.
        x_start = self._attributes_or_raise().geometry.x_origin + tile_x * metadata.tile_size
        y_start = self._attributes_or_raise().geometry.y_origin + tile_y * metadata.tile_size
        mask = (
            (x_start < clipped.x_max)
            & (x_start + metadata.tile_size > clipped.x_min)
            & (y_start < clipped.y_max)
            & (y_start + metadata.tile_size > clipped.y_min)
        )
        return np.ascontiguousarray(np.flatnonzero(mask) + start, dtype=np.int64)

    def _clip_viewport(self, viewport: _IntrinsicViewport) -> _IntrinsicViewport | None:
        """Clip a half-open viewport to the observed source-point geometry.

        The cache geometry stores inclusive extrema observed in the validated
        source, whereas viewports use half-open bounds. Expand each observed
        maximum by one representable float so points exactly at that maximum
        remain inside the clipped viewport. Return ``None`` when the requested
        viewport is completely disjoint from the observed geometry.
        """
        geometry = self._attributes_or_raise().geometry
        if (
            viewport.x_max <= geometry.x_min
            or viewport.x_min > geometry.x_max
            or viewport.y_max <= geometry.y_min
            or viewport.y_min > geometry.y_max
        ):
            return None
        # Source maxima are observed point coordinates, hence closed bounds.
        # Expand them by one representable float so they can participate in the
        # reader's half-open viewport convention.
        x_max = math.nextafter(geometry.x_max, math.inf)
        y_max = math.nextafter(geometry.y_max, math.inf)
        return _IntrinsicViewport(
            max(viewport.x_min, geometry.x_min),
            max(viewport.y_min, geometry.y_min),
            min(viewport.x_max, x_max),
            min(viewport.y_max, y_max),
        )

    def _load_selected_value_level_index(
        self,
        level: int,
        value_ids: npt.NDArray[np.uint32],
        record_counts: npt.NDArray[np.uint64],
        *,
        manifest_array: zarr.Array,
        point_count_array: zarr.Array,
    ) -> _SelectedValueLevelIndex:
        """Load one immutable level index from selected catalog records.

        ``record_counts`` is aligned with ``value_ids`` and gives the number of
        value-to-tile records retained for each selected value at this level.
        Its cumulative sum defines the level-index ``value_indptr`` and exact
        output-array sizes.

        The on-disk catalog is value-major and ``value_ids`` is sorted. Resolve
        its exact selected intervals to one basic slice when they become
        contiguous, or one C-contiguous ``int64`` row selector when gaps remain.
        Each aligned catalog array receives that selector once. Zarr owns chunk
        and shard processing; the application never materializes unselected
        rows merely to join the intervals into a broad envelope.

        The returned level contains compact read-only ``value_indptr``,
        ``manifest_index``, and ``n_points`` arrays grouped by selected-value
        position.
        """
        # Convert the per-selected-value record counts into a level-local CSR
        # pointer table. Unlike the cache-wide on-disk value_tiles/indptr, this
        # indexes only the selected values inside the compact in-memory arrays:
        # value position i owns manifest_index[value_indptr[i]:value_indptr[i + 1]]
        # and the aligned n_points rows. Equal pointers preserve an empty value.
        value_indptr = np.empty(len(value_ids) + 1, dtype=np.uint64)
        value_indptr[0] = 0
        np.cumsum(record_counts, out=value_indptr[1:])
        expected_row_count = int(value_indptr[-1])
        intervals = self._value_tile_intervals(level, value_ids)
        if not intervals:
            if expected_row_count != 0:
                raise RuntimeError("Selected-value intervals do not reconcile to the projected record count.")
            manifest_index = np.empty(0, dtype=np.uint64)
            n_points = np.empty(0, dtype=np.uint64)
            return _SelectedValueLevelIndex(value_indptr, manifest_index, n_points)

        row_selection = _exact_value_tile_row_selection(
            intervals,
            catalog_row_count=self._attributes_or_raise().catalog.value_tile_row_count,
            expected_row_count=expected_row_count,
        )
        manifest_index = np.ascontiguousarray(
            manifest_array.get_orthogonal_selection((row_selection,)),
            dtype=np.uint64,
        )
        n_points = np.ascontiguousarray(
            point_count_array.get_orthogonal_selection((row_selection,)),
            dtype=np.uint64,
        )
        if manifest_index.shape != (expected_row_count,) or n_points.shape != (expected_row_count,):
            raise ValueError("Selected catalog reads returned unexpected aligned shapes.")

        level_start = int(self._manifest_level_indptr_or_raise()[level])
        level_stop = int(self._manifest_level_indptr_or_raise()[level + 1])
        if (
            bool((n_points == 0).any())
            or bool((manifest_index < level_start).any())
            or bool((manifest_index >= level_stop).any())
        ):
            raise ValueError("Encountered invalid value-tile records while loading the index.")
        return _SelectedValueLevelIndex(value_indptr, manifest_index, n_points)

    def _positive_visible_manifest_rows(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_index: _SelectedValueIndex,
    ) -> npt.NDArray[np.int64]:
        """Return the sorted union of visible tiles containing selected values.

        The resident selected-value index already owns the value-to-tile
        relation. Planning needs only a boolean per visible tile: a tile is
        positive when at least one selected value occurs in it. It does not
        need a dictionary of tile-specific value IDs or one array per tile.

        For example, values occurring in visible manifest rows [10, 30] and
        [20, 30] produce [10, 20, 30], with tile 30 included only once.
        No catalog or point-payload IO is performed.
        """
        visible = np.asarray(visible_rows, dtype=np.uint64)
        positive_visible = np.zeros(len(visible), dtype=bool)
        for _, visible_positions, _ in self._iter_selected_value_matches(
            level, visible, value_index, include_point_counts=False
        ):
            positive_visible[visible_positions] = True
        return visible_rows[positive_visible]

    def _selected_value_manifest_summary(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_index: _SelectedValueIndex,
    ) -> tuple[npt.NDArray[np.uint64], int]:
        """Return indexed counts and the positive-tile union needed for LOD.

        Level selection needs visible point totals and the number of distinct
        positive tiles, but not the value IDs applicable to each tile. This
        summary-only path therefore retains only aligned per-value counts and
        a boolean mask of positive visible tiles.

        Parameters
        ----------
        level
            Cache level whose resident value-to-tile records are summarized.
        visible_rows
            Sorted global manifest rows for logical tiles intersecting the
            viewport.
        value_index
            Generation-validated immutable selected-value index.

        Returns
        -------
        counts_by_value : numpy.ndarray
            Visible ``uint64`` point totals aligned with
            ``value_index.value_ids``.
        positive_visible_tile_count : int
            Number of distinct visible manifest tiles containing at least one
            selected value. A tile containing several selected values is counted
            once.

        Notes
        -----
        This operation reads no catalog Zarr array, bucket, or point payload.
        Sampled-away values retain a zero in ``counts_by_value``, preserving the
        selected-value alignment needed for omission evidence.

        Examples
        --------
        Suppose selected values have these ``(manifest row, point count)``
        records at this level::

            value 0: [(100, 3), (102, 8), (104, 1)]
            value 1: [(101, 6), (104, 2)]

        With visible manifest rows ``[101, 102, 104]``, the result is::

            counts_by_value = np.array([9, 8], dtype=np.uint64)
            positive_visible_tile_count = 3

        Value ``0`` contributes ``8 + 1`` points and value ``1`` contributes
        ``6 + 2``. Manifest row ``104`` contains both selected values, but it
        contributes only once to the positive-tile union ``{101, 102, 104}``.
        """
        visible = np.asarray(visible_rows, dtype=np.uint64)
        counts_by_value = np.zeros(len(value_index.value_ids), dtype=np.uint64)
        positive_visible = np.zeros(len(visible), dtype=np.bool_)
        for selected_position, visible_positions, n_points in self._iter_selected_value_matches(
            level,
            visible,
            value_index,
            include_point_counts=True,
        ):
            assert n_points is not None
            counts_by_value[selected_position] = n_points.sum(dtype=np.uint64)
            positive_visible[visible_positions] = True
        return counts_by_value, int(np.count_nonzero(positive_visible))

    def _iter_selected_value_matches(
        self,
        level: int,
        visible: npt.NDArray[np.uint64],
        value_index: _SelectedValueIndex,
        *,
        include_point_counts: bool,
    ) -> Iterator[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.uint64] | None]]:
        """Yield each selected value's visible tile positions and optional counts.

        Intersect one immutable level index with the resident
        visible manifest rows. This is the shared in-memory primitive behind LOD
        summaries and the positive-tile union used by viewport planning.

        Parameters
        ----------
        level
            Cache level whose resident value-to-tile records are queried.
        visible
            Sorted global manifest rows for logical tiles intersecting the
            viewport. Positions in this array identify visible tiles within the
            current request.
        value_index
            Generation-validated selected-value index. Its level-local
            ``value_indptr`` partitions records by position in ``value_ids``.
        include_point_counts
            Whether to gather the matched point counts for LOD estimation.
            Positive-tile planning passes ``False``: it needs only the matching
            positions, so the iterator does not access ``level_index.n_points``
            or allocate a filtered point-count array.

        Yields
        ------
        selected_value_position : int
            Position of the represented value in ``value_index.value_ids``.
        visible_positions : numpy.ndarray
            ``int64`` positions into ``visible`` for tiles containing that value.
        n_points : numpy.ndarray or None
            Aligned positive ``uint64`` point counts for those value/tile records
            when requested. ``None`` means counts were not requested, not that
            the matched tiles contain zero points.

        Notes
        -----
        Empty indexed value intervals and values with no visible tiles produce
        no yield. Yielded arrays are C-contiguous; when counts are requested,
        they are aligned with the visible positions. This method reads no Zarr
        catalog array, opens no bucket, and reads no point payload; all inputs
        were materialized by ``load_selected_value_index``.

        Examples
        --------
        Suppose the method receives::

            level = 2
            visible = np.array([101, 102, 104], dtype=np.uint64)
            value_index.value_ids = np.array([10, 42], dtype=np.uint32)
            include_point_counts = True

        and the level index contains these records:

        | Selected position | Value ID | Manifest row | Points |
        |---:|---:|---:|---:|
        | 0 | 10 | 100 | 3 |
        | 0 | 10 | 102 | 8 |
        | 0 | 10 | 104 | 1 |
        | 1 | 42 | 101 | 6 |
        | 1 | 42 | 104 | 2 |

        Manifest row ``100`` is not visible and is discarded. The two yields
        are equivalent to::

            (
                0,
                np.array([1, 2], dtype=np.int64),
                np.array([8, 1], dtype=np.uint64),
            )
            (
                1,
                np.array([0, 2], dtype=np.int64),
                np.array([6, 2], dtype=np.uint64),
            )

        Thus ``visible[visible_positions]`` recovers the global manifest rows,
        while ``value_index.value_ids[selected_value_position]`` recovers
        the corresponding canonical value ID. With ``include_point_counts=False``,
        the same matches are yielded, but each tuple's third item is ``None``.
        """
        if len(visible) == 0:
            return
        level_index = value_index.levels[level]
        level_start = int(self._manifest_level_indptr_or_raise()[level])
        level_stop = int(self._manifest_level_indptr_or_raise()[level + 1])
        relative_visible = visible.astype(np.int64, copy=False) - level_start
        if bool((relative_visible < 0).any()) or bool((relative_visible >= level_stop - level_start).any()):
            raise ValueError("Visible manifest rows lie outside the requested level.")
        visible_position_by_level_row = np.full(level_stop - level_start, -1, dtype=np.int64)
        visible_position_by_level_row[relative_visible] = np.arange(len(visible), dtype=np.int64)

        # For each selected value:
        # 1. find all indexed manifest tiles containing it;
        # 2. map those tiles to positions in the current viewport;
        # 3. discard tiles outside the viewport;
        # 4. yield visible positions, gathering aligned point counts only when
        #    requested by the caller.
        for selected_position, (start, stop) in enumerate(pairwise(level_index.value_indptr.tolist())):
            if start == stop:
                continue
            manifest_index = level_index.manifest_index[start:stop]
            positions = visible_position_by_level_row[manifest_index - np.uint64(level_start)]
            matches = positions >= 0
            if bool(matches.any()):
                n_points = None
                if include_point_counts:
                    n_points = np.ascontiguousarray(level_index.n_points[start:stop][matches], dtype=np.uint64)
                yield (
                    selected_position,
                    np.ascontiguousarray(positions[matches], dtype=np.int64),
                    n_points,
                )

    def _value_tile_intervals(
        self,
        level: int,
        value_ids: npt.NDArray[np.uint32],
    ) -> tuple[_ValueTileInterval, ...]:
        """Resolve requested values to nonempty exact catalog intervals."""
        pointers = self._value_tiles_indptr_or_raise()
        indexes = value_ids.astype(np.int64, copy=False)
        starts = pointers[level, indexes]
        stops = pointers[level, indexes + 1]
        row_count = self._attributes_or_raise().catalog.value_tile_row_count
        if bool((starts > stops).any()) or bool((stops > row_count).any()):
            raise ValueError("Value-tile pointers are outside the catalog arrays.")
        return tuple(
            _ValueTileInterval(
                selected_value_position=selected_value_position,
                value_id=int(value_id),
                start=int(start),
                stop=int(stop),
            )
            for selected_value_position, (value_id, start, stop) in enumerate(
                zip(value_ids.tolist(), starts.tolist(), stops.tolist(), strict=True)
            )
            if start < stop
        )

    def _read_value_major_requests(
        self,
        *,
        level: int,
        requested_value_ids: tuple[int, ...],
        selected_value_level_index: _SelectedValueLevelIndex,
        manifest_rows: tuple[int, ...],
        raise_if_cancelled: Callable[[], None] | None,
    ) -> _ViewportReadResult:
        """Read selected logical tiles from one level's value-major storage.

        The read combines three sources:

        A. ``manifest_rows`` supplies the sorted missing tiles that still
           require physical payload I/O. The plan-aware dispatcher has already
           excluded tiles outside the viewport or in CPU residency.
        B. ``selected_value_level_index``, the in-memory projection of
           ``value_tiles``, maps each selected value to all manifest tiles
           containing it and their point counts. It contains every value/tile
           record for those values at this level, including records outside the
           viewport.
        C. The value-major storage supplies the physical ``location`` rows,
           grouped by value without repeating the tile identity alongside
           every point.

        ``requested_value_ids`` names the values partitioned by Source B's
        ``value_indptr`` in the same positional order. The helper receives
        these already validated selected-level facts, not a viewport plan or
        a second per-tile copy of the value membership relation.

        For example::

            missing manifest rows (supplies requested tiles, Source A):
                manifest tiles {10, 30}

            in-memory value_tiles index (answers which tiles contain V and their point counts, Source B):
                value V -> [(manifest 10, 3 points),
                            (manifest 20, 5 points),
                            (manifest 30, 2 points)]

            value-major storage (supplies V's physical location rows, Source C):
                value V -> location[100:110]

        The reader intersects the requested manifest rows ``{10, 30}`` with
        the value's indexed manifest rows ``{10, 20, 30}``. The complete
        ``n_points`` records, provided by the in-memory ``value_tiles`` index,
        establish the physical blocks in ``value_major/location``::

            manifest 10 -> location[100:103]
            manifest 20 -> location[103:108]
            manifest 30 -> location[108:110]

        It then reads only ``[100:103]`` and ``[108:110]``. Manifest 20 is not
        read, but its point count is still required to calculate manifest 30's
        offset. The same applies to records outside the viewport or already in
        CPU residency.

        This method does not access the physical tile-major buckets. It reads
        locations in value-major order, synthesizes their value IDs, and
        scatters them into the requested manifest-tile order. Within each
        returned tile, value blocks remain in increasing canonical value-ID
        order and retain their existing point order.
        """
        level_index = selected_value_level_index

        # 1. Index the requested logical tiles (Source A)
        # ------------------------------------------------
        request_position_by_manifest = {manifest_row: position for position, manifest_row in enumerate(manifest_rows)}
        if len(request_position_by_manifest) != len(manifest_rows):
            raise ValueError("Value-major requests must contain unique manifest rows.")
        missing_manifest = np.asarray(manifest_rows, dtype=np.uint64)

        # 2. Resolve physical value-major location blocks (Sources A, B, and C)
        # ----------------------------------------------------------------------
        # Source C above: these pointers delimit each value's complete interval
        # in ``value_major/location``.
        point_indptr = self._value_major_point_indptr_or_raise(level)
        blocks: list[_ValueMajorReadBlock] = []
        for selected_position, value_id in enumerate(requested_value_ids):
            record_start = int(level_index.value_indptr[selected_position])
            record_stop = int(level_index.value_indptr[selected_position + 1])
            # Source B above: `record_manifest` contains the global manifest
            # row indices of every tile containing `value_id` at this level,
            # including tiles outside the viewport or already in CPU residency.
            # `record_n_points[i]` is this value's point count in the tile
            # identified by `record_manifest[i]`. Keep the complete sequences:
            # skipped tiles still contribute to later value-major offsets.
            record_manifest = level_index.manifest_index[record_start:record_stop]
            record_n_points = level_index.n_points[record_start:record_stop]

            # Intersect missing tiles directly with this value's complete
            # records. Missing tiles containing only other selected values
            # need no block here; neither do values absent from this level.
            positions = np.searchsorted(record_manifest, missing_manifest)
            in_bounds = positions < len(record_manifest)
            positions = positions[in_bounds]
            matched_manifests = missing_manifest[in_bounds]
            matches = record_manifest[positions] == matched_manifests
            positions = positions[matches]
            matched_manifests = matched_manifests[matches]
            if len(positions) == 0:
                continue

            value_row_start = int(point_indptr[value_id])
            value_row_stop = int(point_indptr[value_id + 1])

            # Combine Sources B and C: convert the per-manifest point counts
            # into absolute ``value_major/location`` boundaries. Point blocks
            # follow the same increasing manifest order as ``record_manifest``
            # and its aligned ``record_n_points``. Thus, for value/tile record i:
            #
            #   record_sidecar_indptr[i]
            #       = point_indptr[value_id] + sum(record_n_points[:i])
            #
            # and ``record_manifest[i]`` owns the corresponding interval:
            #
            #   value_major/location[
            #       record_sidecar_indptr[i]:record_sidecar_indptr[i + 1]
            #   ]
            #
            record_sidecar_indptr = np.empty(len(record_n_points) + 1, dtype=np.uint64)
            record_sidecar_indptr[0] = np.uint64(value_row_start)
            np.cumsum(record_n_points, out=record_sidecar_indptr[1:])
            record_sidecar_indptr[1:] += np.uint64(value_row_start)
            if int(record_sidecar_indptr[-1]) != value_row_stop:
                raise ValueError("Selected value-to-tile counts do not reconcile to value-major pointers.")

            for manifest_row, position in zip(matched_manifests.tolist(), positions.tolist(), strict=True):
                blocks.append(
                    _ValueMajorReadBlock(
                        value_id=value_id,
                        manifest_row=manifest_row,
                        row_start=int(record_sidecar_indptr[position]),
                        row_count=int(record_n_points[position]),
                    )
                )

        if not blocks:
            raise ValueError("A nonempty value-major request resolved no sidecar rows.")
        if any(current.row_start < previous.interval[1] for previous, current in pairwise(blocks)):
            raise ValueError("Value-major read blocks must follow nonoverlapping sidecar order.")

        # 3. Read the resolved locations in value-major order (Source C)
        # ----------------------------------------------------------------
        selected_row_count = sum(block.row_count for block in blocks)
        locations = self._value_major_reader_or_raise(level).read_intervals(
            tuple(block.interval for block in blocks),
            expected_row_count=selected_row_count,
            raise_if_cancelled=raise_if_cancelled,
        )

        # 4. Scatter value-major rows into tile-oriented output
        # -------------------------------------------------------
        tile_counts = np.zeros(len(manifest_rows), dtype=np.uint64)
        for block in blocks:
            tile_counts[request_position_by_manifest[block.manifest_row]] += np.uint64(block.row_count)
        tile_indptr = np.empty(len(manifest_rows) + 1, dtype=np.uint64)
        tile_indptr[0] = 0
        np.cumsum(tile_counts, out=tile_indptr[1:])
        if int(tile_indptr[-1]) != selected_row_count or bool((tile_counts == 0).any()):
            raise RuntimeError("Value-major blocks do not reconcile to the requested logical tiles.")

        # Locations read in value-major order are grouped by value:
        #
        #   locations:
        #       value 1, tile 10
        #       value 1, tile 30
        #       value 700, tile 10
        #       value 700, tile 30
        #
        # Scatter them into the tile-oriented layout expected by CPU residency:
        #
        #   ordered_locations:
        #       tile 10:
        #           value 1
        #           value 700
        #       tile 30:
        #           value 1
        #           value 700
        #
        # ``tile_cursor`` tracks the next unwritten row in each tile interval.
        # Value IDs are synthesized because ``value_major/location`` stores
        # only locations.
        ordered_locations = np.empty_like(locations)
        ordered_value_ids = np.empty(selected_row_count, dtype=np.uint32)
        tile_cursor = tile_indptr[:-1].copy()
        source_start = 0
        for block in blocks:
            source_stop = source_start + block.row_count
            request_position = request_position_by_manifest[block.manifest_row]
            output_start = int(tile_cursor[request_position])
            output_stop = output_start + block.row_count
            ordered_locations[output_start:output_stop] = locations[source_start:source_stop]
            ordered_value_ids[output_start:output_stop] = np.uint32(block.value_id)
            tile_cursor[request_position] = np.uint64(output_stop)
            source_start = source_stop
        if source_start != selected_row_count or not np.array_equal(tile_cursor, tile_indptr[1:]):
            raise RuntimeError("Value-major scatter did not fill every logical tile interval.")

        # 5. Build logical tile results in request order
        # ------------------------------------------------
        tiles = tuple(
            self._tile_result(
                self._descriptors[manifest_row],
                _PointDisplayPayload(
                    location=ordered_locations[int(start) : int(stop)],
                    value_id=ordered_value_ids[int(start) : int(stop)],
                ),
            )
            for manifest_row, start, stop in zip(manifest_rows, tile_indptr[:-1], tile_indptr[1:], strict=True)
        )
        return _ViewportReadResult(level=level, tiles=tiles)

    def _read_complete_tile_major_requests(
        self,
        level: int,
        manifest_rows: tuple[int, ...],
        *,
        raise_if_cancelled: Callable[[], None] | None,
    ) -> _ViewportReadResult:
        """Read complete manifest-addressed tiles through one batch per bucket.

        Every supplied manifest row identifies one complete logical tile at
        ``level``, without a per-tile value selection. This method groups
        those rows by physical bucket and makes exactly one
        ``read_complete_display_payloads`` call for each nonempty bucket group. The call
        contains every requested tile in that bucket; the bucket reader performs
        coordinated point-array selections and returns one result per tile. The
        resulting complete tile payloads are restored to the original manifest
        request order::

            manifest tile requests
                    -> group by (level, bucket_id)
                    -> one reader call per bucket, containing all its tiles
                    -> one payload per logical tile
                    -> restore original request order

        Buckets are processed sequentially. Batching here concerns the tiles
        within each bucket and does not introduce cross-bucket concurrency.
        Cancellation is checked before and after each bucket batch; an active
        Zarr operation cannot be interrupted, but later buckets are not read
        after cancellation is observed.

        This bucket-local batching is deliberate. Reading every tile through a
        separate Zarr selection would repeat selection dispatch and make the
        caller wait for one tile before requesting the next. One bucket batch
        instead presents all exact requested rows to each aligned point array at
        once, allowing Zarr to coordinate work across the touched chunks and
        shards. The batch never spans buckets because each bucket is an
        independent Zarr store with its own reader and row-coordinate space.
        """
        # Group logical tile requests by physical bucket so each bucket reader is
        # acquired once; restore the original manifest-request order after reading.
        grouped: dict[int, list[int]] = {}
        for manifest_row in manifest_rows:
            descriptor = self._descriptors[manifest_row]
            grouped.setdefault(descriptor.bucket_id, []).append(manifest_row)

        results: dict[int, _TileReadResult] = {}
        for bucket_id, bucket_manifest_rows in grouped.items():
            if raise_if_cancelled is not None:
                raise_if_cancelled()
            bucket_reader = self._get_bucket_reader_for_complete_display(level=level, bucket_id=bucket_id)
            # This is one physical-reader call for the bucket, not one call per
            # tile. Its tuple retains every logical tile request in the group.
            # Manifest rows are converted to tile descriptors for the bucket API.
            payloads = bucket_reader.read_complete_display_payloads(
                tuple(self._descriptors[manifest_row] for manifest_row in bucket_manifest_rows)
            )
            if raise_if_cancelled is not None:
                raise_if_cancelled()
            for manifest_row, payload in zip(bucket_manifest_rows, payloads, strict=True):
                descriptor = self._descriptors[manifest_row]
                results[manifest_row] = self._tile_result(descriptor, payload)

        ordered_tiles = tuple(results[manifest_row] for manifest_row in manifest_rows)
        return _ViewportReadResult(level=level, tiles=ordered_tiles)

    def _tile_result(
        self,
        descriptor: _TileDescriptor,
        payload: _PointDisplayPayload,
    ) -> _TileReadResult:
        return _TileReadResult(
            level=descriptor.level,
            tile_x=descriptor.tile_x,
            tile_y=descriptor.tile_y,
            tile_size=self._attributes_or_raise().levels[descriptor.level].tile_size,
            location=payload.location,
            value_id=payload.value_id,
        )

    def _require_value_ids(
        self,
        value_ids: npt.NDArray[np.uint32] | None,
    ) -> npt.NDArray[np.uint32] | None:
        if value_ids is None:
            return None
        if (
            not isinstance(value_ids, np.ndarray)
            or value_ids.dtype != np.dtype(np.uint32)
            or value_ids.ndim != 1
            or not value_ids.flags.c_contiguous
            or len(value_ids) == 0
        ):
            raise ValueError("`value_ids` must be a nonempty one-dimensional C-contiguous uint32 array.")
        if bool((value_ids[1:] <= value_ids[:-1]).any()):
            raise ValueError("`value_ids` must be strictly increasing and unique.")
        if int(value_ids[-1]) >= len(self.value_names):
            raise ValueError("`value_ids` contains an ID outside the serialized vocabulary.")
        return value_ids

    def _require_selected_value_index(
        self,
        value_index: _SelectedValueIndex | None,
    ) -> _SelectedValueIndex | None:
        if value_index is None:
            return None
        if not isinstance(value_index, _SelectedValueIndex):
            raise ValueError("`value_index` must be _SelectedValueIndex or None.")
        if value_index.cache_generation_id != self.cache_generation_id:
            raise ValueError("Selected-value index belongs to another cache generation.")
        if len(value_index.levels) != self.level_count:
            raise ValueError("Selected-value index has the wrong number of cache levels.")
        if int(value_index.value_ids[-1]) >= len(self.value_names):
            raise ValueError("Selected-value index contains an ID outside the serialized vocabulary.")
        return value_index

    def _require_viewport_plan_compatible(self, plan: _ViewportReadPlan) -> None:
        """Require a plan compatible with the currently opened cache.

        ``_ViewportReadPlan`` owns its internal request consistency. This check
        only binds that immutable plan to the active generation, serialized
        levels, and value vocabulary without rescanning every planned tile.
        """
        if not isinstance(plan, _ViewportReadPlan):
            raise ValueError("`plan` must be _ViewportReadPlan.")
        if plan.cache_generation_id != self.cache_generation_id:
            raise ValueError("Viewport plan belongs to another cache generation.")
        self._require_level(plan.level)
        if plan.requested_value_ids is not None and plan.requested_value_ids[-1] >= len(self.value_names):
            raise ValueError("Viewport plan selection contains an ID outside the serialized vocabulary.")
        if plan.route == "value_major_subset":
            level_index = plan.selected_value_level_index
            if level_index is None or plan.requested_value_ids is None:
                raise ValueError("Value-major viewport plan is missing its selected-value identity.")
            level_start = int(self._manifest_level_indptr_or_raise()[plan.level])
            level_stop = int(self._manifest_level_indptr_or_raise()[plan.level + 1])
            if bool((level_index.manifest_index < level_start).any()) or bool(
                (level_index.manifest_index >= level_stop).any()
            ):
                raise ValueError("Selected-value level index does not belong to the viewport plan level.")

    def _require_level(self, level: int) -> _LevelMetadata:
        attributes = self._attributes_or_raise()
        _require_integer_in_range(level, "level", maximum=len(attributes.levels) - 1)
        return attributes.levels[level]

    def _catalog_or_raise(self) -> _CacheRootReader:
        self._require_open_or_initializing()
        if self._catalog is None:
            raise RuntimeError("Cache-root reader is not open.")
        return self._catalog

    def _get_bucket_reader_for_complete_display(self, *, level: int, bucket_id: int) -> _BucketReader:
        """Get a cached bucket reader ready for complete-tile display reads.

        Open the bucket lazily and ensure its complete descriptor tuple is
        validated and installed. Subsequent calls reuse the reader and accepted
        tuple without repeating descriptor-validation IO. This method does not
        read point payloads.
        """
        bucket_reader = self._bucket_cache_or_raise().get(level=level, bucket_id=bucket_id)
        key = (level, bucket_id)
        bucket_reader.set_tile_descriptors(self._descriptors_by_bucket[key])
        return bucket_reader

    def _bucket_cache_or_raise(self) -> _BucketReaderCache:
        self._require_open()
        if self._bucket_cache is None:
            raise RuntimeError("Bucket reader cache is not open.")
        return self._bucket_cache

    def _attributes_or_raise(self) -> _CacheAttributes:
        self._require_open_or_initializing()
        if self._attributes is None:
            raise RuntimeError("Cache attributes are not open.")
        return self._attributes

    def _manifest_level_indptr_or_raise(self) -> npt.NDArray[np.uint64]:
        if self._manifest_level_indptr is None:
            raise RuntimeError("Manifest level pointers are not loaded.")
        return self._manifest_level_indptr

    def _manifest_tile_x_or_raise(self) -> npt.NDArray[np.uint32]:
        if self._manifest_tile_x is None:
            raise RuntimeError("Manifest tile x coordinates are not loaded.")
        return self._manifest_tile_x

    def _manifest_tile_y_or_raise(self) -> npt.NDArray[np.uint32]:
        if self._manifest_tile_y is None:
            raise RuntimeError("Manifest tile y coordinates are not loaded.")
        return self._manifest_tile_y

    def _manifest_n_points_or_raise(self) -> npt.NDArray[np.uint64]:
        if self._manifest_n_points is None:
            raise RuntimeError("Manifest point counts are not loaded.")
        return self._manifest_n_points

    def _value_tiles_indptr_or_raise(self) -> npt.NDArray[np.uint64]:
        if self._value_tiles_indptr is None:
            raise RuntimeError("Value-tile pointers are not loaded.")
        return self._value_tiles_indptr

    def _value_major_point_indptr_or_raise(self, level: int) -> npt.NDArray[np.uint64]:
        self._require_level(level)
        if len(self._value_major_point_indptr) != self.level_count:
            raise RuntimeError("Value-major point pointers are not loaded.")
        return self._value_major_point_indptr[level]

    def _value_major_reader_or_raise(self, level: int) -> _ValueMajorLocationReader:
        self._require_level(level)
        if len(self._value_major_readers) != self.level_count:
            raise RuntimeError("Value-major location readers are not ready.")
        return self._value_major_readers[level]

    def _require_open_or_initializing(self) -> None:
        if not self._open and self._catalog is None:
            raise RuntimeError("Points cache reader is not open.")

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("Points cache reader is not open.")

    def _clear_open_state(self) -> None:
        self._catalog = None
        self._bucket_cache = None
        self._attributes = None
        self._dataset_info = None
        self._manifest_level_indptr = None
        self._manifest_bucket_id = None
        self._manifest_bucket_tile_index = None
        self._manifest_tile_x = None
        self._manifest_tile_y = None
        self._manifest_n_points = None
        self._value_tiles_indptr = None
        self._value_n_points = None
        self._value_major_point_indptr = ()
        self._value_major_readers = ()
        self._descriptors = ()
        self._descriptors_by_bucket = {}
        self._manifest_row_by_tile = {}
        self._resident_index_bytes = 0
        self._open = False


def _read_cache_dataset_info(cache_root: str | Path) -> _CacheDatasetInfo:
    """Read the small semantic descriptor of one completed cache generation.

    Unlike entering :class:`_PointsCacheReader`, this seam does not materialize
    catalog indexes, open bucket stores, or read point payloads. It is intended
    for application discovery before a long-lived worker-owned reader exists.
    """
    with _CacheRootReader(Path(cache_root)) as catalog:
        attributes = catalog.attributes
        if attributes.publication_state != PUBLICATION_STATE_COMPLETE:
            raise ValueError("Cache root publication_state is not 'complete'.")
        return _dataset_info_from_attributes(attributes)


def _dataset_info_from_attributes(attributes: _CacheAttributes) -> _CacheDatasetInfo:
    """Copy validated cache metadata into the narrow viewer-facing contract."""
    return _CacheDatasetInfo(
        cache_generation_id=attributes.cache_generation_id,
        points_name=attributes.source.points_name,
        value_column=attributes.source.value_column,
        value_names=attributes.value_names,
        x_origin=attributes.geometry.x_origin,
        y_origin=attributes.geometry.y_origin,
        x_min=attributes.geometry.x_min,
        x_max=attributes.geometry.x_max,
        y_min=attributes.geometry.y_min,
        y_max=attributes.geometry.y_max,
        levels=tuple(
            _CacheLevelInfo(
                level=level.level,
                kind=level.kind,
                tile_size=level.tile_size,
                grid_width=level.grid_width,
                grid_height=level.grid_height,
                max_points_per_tile=level.max_points_per_tile,
                bucket_count=level.bucket_count,
                tile_count=level.tile_count,
                point_count=level.point_count,
            )
            for level in attributes.levels
        ),
        overview_point_budget=attributes.build.overview_point_budget,
    )


def _require_cache_generation_id(value: object) -> None:
    """Require the canonical UUID form used by published cache generations."""
    if not isinstance(value, str):
        raise ValueError("`cache_generation_id` must be a canonical UUID string.")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError("`cache_generation_id` must be a canonical UUID string.") from error
    if str(parsed) != value:
        raise ValueError("`cache_generation_id` must be a canonical lowercase UUID string.")


def _require_requested_value_ids(value: object) -> None:
    """Require the canonical all-values or requested-value plan identity."""
    if value is None:
        return
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            not isinstance(value_id, int) or isinstance(value_id, bool) or not 0 <= value_id <= _UINT32_MAX
            for value_id in value
        )
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError("`requested_value_ids` must be None or a nonempty sorted unique value-ID tuple.")


def _read_only_array(
    catalog: _CacheRootReader,
    name: str,
    *,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray:
    array = np.ascontiguousarray(
        np.asarray(catalog.array(name)[:], dtype=CATALOG_ARRAY_DTYPES[name] if dtype is None else dtype)
    )
    array.flags.writeable = False
    return array


def _read_only_index_array(array: object, name: str, dtype: npt.DTypeLike) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != expected_dtype
        or array.ndim != 1
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"`{name}` must be a one-dimensional C-contiguous {expected_dtype.name} array.")
    # Selected-value indexes can outlive the caller's input arrays. Own the buffer
    # before freezing it so later caller mutation cannot alter the index.
    read_only = np.array(array, dtype=expected_dtype, order="C", copy=True)
    read_only.flags.writeable = False
    return read_only


def _exact_value_tile_row_selection(
    intervals: tuple[_ValueTileInterval, ...],
    *,
    catalog_row_count: int,
    expected_row_count: int,
) -> slice | npt.NDArray[np.int64]:
    """Return the cheapest exact row selector for one selected catalog level.

    The input contains one ordered, nonoverlapping, value-major half-open
    interval for every selected value represented at the level. Selection
    follows this policy::

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
    selector always addresses exactly ``expected_row_count`` catalog rows.

    The slice specialization avoids allocating an ``int64`` row array and lets
    Zarr use its cheaper contiguous-selection path for genuinely contiguous
    values. Disjoint selections use one orthogonal integer selector so Zarr can
    coordinate the complete level selection without materializing unselected
    value rows inside its gaps. Chunk and shard geometry remain Zarr concerns.

    Intervals must be nonempty, ordered by selected value, nonoverlapping, and
    contained within the catalog arrays. This function also reconciles their
    total length with ``expected_row_count`` before returning a selector.

    This is the selected-catalog counterpart of ``_exact_row_selection`` in
    ``storage.bucket_reader``. Their domain-specific validation remains
    separate; both delegate the validated interval transformation to the shared
    storage utility.
    """
    _require_integer_in_range(catalog_row_count, "catalog_row_count", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(expected_row_count, "expected_row_count", minimum=1, maximum=_INT64_MAX)
    if not intervals:
        raise ValueError("`intervals` must be nonempty.")
    if any(
        not isinstance(interval, _ValueTileInterval)
        or interval.selected_value_position < 0
        or interval.value_id < 0
        or interval.start < 0
        or interval.start >= interval.stop
        or interval.stop > catalog_row_count
        for interval in intervals
    ):
        raise ValueError("Catalog intervals must be nonempty and lie inside the catalog arrays.")
    if any(
        current.selected_value_position <= previous.selected_value_position
        or current.value_id <= previous.value_id
        or current.start < previous.stop
        for previous, current in pairwise(intervals)
    ):
        raise ValueError("Catalog intervals must follow selected-value and nonoverlapping row order.")

    observed_row_count = 0
    for interval in intervals:
        observed_row_count += interval.stop - interval.start
    if observed_row_count != expected_row_count:
        raise ValueError("Catalog intervals do not reconcile to the expected selected record count.")
    return _build_exact_row_selection(tuple((interval.start, interval.stop) for interval in intervals))


def _require_display_arrays(location: object, value_id: object) -> None:
    if (
        not isinstance(location, np.ndarray)
        or location.dtype != np.dtype(np.float32)
        or location.ndim != 2
        or location.shape[1:] != (2,)
        or not location.flags.c_contiguous
        or len(location) == 0
    ):
        raise ValueError("`location` must be a nonempty C-contiguous (N, 2) float32 array.")
    if (
        not isinstance(value_id, np.ndarray)
        or value_id.dtype != np.dtype(np.uint32)
        or value_id.ndim != 1
        or not value_id.flags.c_contiguous
        or len(value_id) != len(location)
    ):
        raise ValueError("`value_id` must be an aligned C-contiguous uint32 array.")
