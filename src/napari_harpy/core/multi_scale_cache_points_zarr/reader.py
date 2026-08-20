"""Read one published multiscale Zarr points cache for visualization.

This module is deliberately independent of napari. A later adapter translates
camera and canvas state into :class:`_IntrinsicViewport` and supplies an
effective point budget. The reader owns only cache-level planning and payload
access.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import (
    CATALOG_ARRAY_DTYPES,
    MANIFEST_BUCKET_ID,
    MANIFEST_BUCKET_TILE_INDEX,
    MANIFEST_LEVEL_INDPTR,
    MANIFEST_N_POINTS,
    MANIFEST_TILE_X,
    MANIFEST_TILE_Y,
    PUBLICATION_STATE_COMPLETE,
    VALUE_TILES_INDPTR,
    VALUE_TILES_MANIFEST_INDEX,
    VALUE_TILES_N_POINTS,
    VALUES_N_POINTS,
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
from napari_harpy.core.multi_scale_cache_points_zarr.storage.bucket_reader import _PointDisplayPayload
from napari_harpy.core.multi_scale_cache_points_zarr.storage.catalog_reader import _CatalogReader
from napari_harpy.core.multi_scale_cache_points_zarr.storage.reader_cache import _BucketReaderCache


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
class _ValueTileInterval:
    """Identify one requested value's exact half-open catalog rows."""

    selected_value_position: int
    value_id: int
    start: int
    stop: int


@dataclass(frozen=True)
class _ValueTileCatalogEnvelope:
    """Describe one shard-bounded contiguous catalog read.

    The first interval's start and final interval's stop define the shared row
    envelope sliced once from each parallel ``value_tiles`` array. Unselected
    gaps between the exact fragments are discarded after the read.
    """

    intervals: tuple[_ValueTileInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intervals, tuple) or not self.intervals:
            raise ValueError("`intervals` must be a nonempty tuple.")
        if not all(isinstance(interval, _ValueTileInterval) for interval in self.intervals):
            raise ValueError("Every catalog-envelope interval must be a _ValueTileInterval.")
        if any(
            interval.selected_value_position < 0
            or interval.value_id < 0
            or interval.start < 0
            or interval.start >= interval.stop
            for interval in self.intervals
        ):
            raise ValueError("Catalog-envelope intervals must contain valid nonnegative half-open rows.")
        if any(
            current.selected_value_position <= previous.selected_value_position
            or current.value_id <= previous.value_id
            or current.start < previous.stop
            for previous, current in pairwise(self.intervals)
        ):
            raise ValueError("Catalog-envelope intervals must be ordered and nonoverlapping.")

    @property
    def start(self) -> int:
        """Return the first exact fragment's catalog-row start."""
        return self.intervals[0].start

    @property
    def stop(self) -> int:
        """Return the final exact fragment's catalog-row stop."""
        return self.intervals[-1].stop


class _PointsCacheReader:
    """Read one trusted, completed Zarr cache generation.

    Entering validates the frozen root and array layouts, then materializes only
    the compact manifest, value pointer table, and value totals. It deliberately
    does not replay complete staged validation. Bucket stores are opened lazily
    and retained for this reader's lifetime.
    """

    def __init__(self, cache_root: str | Path) -> None:
        self._cache_root = Path(cache_root)
        self._stack: ExitStack | None = None
        self._catalog: _CatalogReader | None = None
        self._bucket_cache: _BucketReaderCache | None = None
        self._attributes: _CacheAttributes | None = None
        self._manifest_level_indptr: npt.NDArray[np.uint64] | None = None
        self._manifest_bucket_id: npt.NDArray[np.uint32] | None = None
        self._manifest_bucket_tile_index: npt.NDArray[np.uint32] | None = None
        self._manifest_tile_x: npt.NDArray[np.uint32] | None = None
        self._manifest_tile_y: npt.NDArray[np.uint32] | None = None
        self._manifest_n_points: npt.NDArray[np.uint64] | None = None
        self._value_tiles_indptr: npt.NDArray[np.uint64] | None = None
        self._value_n_points: npt.NDArray[np.uint64] | None = None
        self._descriptors: tuple[_TileDescriptor, ...] = ()
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
            catalog = stack.enter_context(_CatalogReader(self._cache_root))
            attributes = catalog.attributes
            if attributes.publication_state != PUBLICATION_STATE_COMPLETE:
                raise ValueError("Cache root publication_state is not 'complete'.")
            bucket_cache = stack.enter_context(
                _BucketReaderCache(
                    self._cache_root,
                    max_open_readers=sum(level.bucket_count for level in attributes.levels),
                )
            )
            self._catalog = catalog
            self._attributes = attributes
            self._bucket_cache = bucket_cache
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
    def cache_generation_id(self) -> str:
        """Return the opened completed generation UUID."""
        return self._attributes_or_raise().cache_generation_id

    @property
    def level_count(self) -> int:
        """Return the number of serialized cache levels."""
        return len(self._attributes_or_raise().levels)

    @property
    def resident_index_bytes(self) -> int:
        """Return bytes in the compact resident NumPy catalog arrays."""
        self._require_open()
        return self._resident_index_bytes

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
        """Read all or value-filtered display rows for one logical tile."""
        metadata = self._require_level(level)
        _require_integer_in_range(tile_x, "tile_x", maximum=metadata.grid_width - 1)
        _require_integer_in_range(tile_y, "tile_y", maximum=metadata.grid_height - 1)
        value_ids = self._require_value_ids(value_ids)
        manifest_row = self._manifest_row_by_tile.get((level, tile_x, tile_y))
        if manifest_row is None:
            return None
        descriptor = self._descriptors[manifest_row]
        reader = self._bucket_cache_or_raise().get(level=level, bucket_id=descriptor.bucket_id)
        payload = reader.read_display_payload(descriptor, value_ids)
        if payload is None:
            return None
        return self._tile_result(descriptor, payload)

    def read_viewport(
        self,
        level: int,
        viewport: _IntrinsicViewport,
        *,
        value_ids: npt.NDArray[np.uint32] | None = None,
    ) -> _ViewportReadResult:
        """Read all applicable rows of every logical tile intersecting a viewport."""
        self._require_level(level)
        value_ids = self._require_value_ids(value_ids)
        visible_rows = self._visible_manifest_rows(level, viewport)
        if len(visible_rows) == 0:
            return _ViewportReadResult(level=level, tiles=())

        # Each request pairs a global manifest row with either None for all
        # values or the selected value IDs known to be present in that tile.
        if value_ids is None:
            requests = tuple((int(row), None) for row in visible_rows)
        else:
            # Use the cache-wide value_tiles index to discard visible tiles
            # containing none of the selection and identify the requested values
            # present in each retained tile.
            value_ids_by_manifest_row = self._value_filtered_manifest(level, visible_rows, value_ids)
            requests = tuple(
                (manifest_row, value_ids_by_manifest_row[manifest_row])
                for manifest_row in sorted(value_ids_by_manifest_row)
            )
        if not requests:
            return _ViewportReadResult(level=level, tiles=())
        return self._read_manifest_requests(level, requests)

    def select_level(
        self,
        viewport: _IntrinsicViewport,
        point_budget: int,
        *,
        value_ids: npt.NDArray[np.uint32] | None = None,
    ) -> _LevelSelection:
        """Choose the finest eligible visible level within the point budget.

        Parameters
        ----------
        viewport
            Intrinsic-coordinate viewport used to identify intersecting tiles.
        point_budget
            Maximum estimated visible point count for a successful selection.
        value_ids
            Optional sorted unique value IDs. When supplied, only represented
            requested values contribute to each level's estimate; missing
            values do not make that level ineligible.

        Returns
        -------
        _LevelSelection
            Selected level, its catalog-derived visible point and positive-tile
            estimates, whether the estimate satisfies ``point_budget``, and any
            Exact-visible selected value IDs omitted at that level.

        Notes
        -----
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

        Level selection performs all of this work from catalog metadata. It does
        not open bucket stores or read point payloads.
        """
        _require_integer_in_range(point_budget, "point_budget", minimum=1, maximum=_INT64_MAX)
        value_ids = self._require_value_ids(value_ids)
        attributes = self._attributes_or_raise()

        if value_ids is None:
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
            point_count_by_value, positive_visible_tile_count = self._value_filtered_manifest_summary(
                metadata.level,
                rows,
                value_ids,
            )
            point_count = int(point_count_by_value.sum(dtype=np.uint64))
            if exact_present_values is None:
                exact_present_values = point_count_by_value > 0
            omitted_value_ids = np.ascontiguousarray(value_ids[exact_present_values & (point_count_by_value == 0)])
            candidate = _LevelSelection(
                level=metadata.level,
                estimated_point_count=point_count,
                positive_visible_tile_count=positive_visible_tile_count,
                within_budget=point_count <= point_budget,
                omitted_value_ids=omitted_value_ids,
            )
            fallback = candidate
            if candidate.within_budget:
                # Avoid slicing value_tiles and constructing tile selections
                # for coarser levels once the finest valid fit is known.
                return candidate

        if fallback is None:
            raise RuntimeError("Cache has no serialized levels.")
        return fallback

    def _load_runtime_indexes(self) -> None:
        """Materialize the compact catalog state needed for runtime planning.

        Load the small manifest, level-pointer, value-pointer, and value-total
        arrays as read-only NumPy arrays. Construct manifest-row-aligned tile
        descriptors and an O(1) mapping from logical tile coordinates to
        manifest rows.

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
        self._resident_index_bytes = sum(array.nbytes for array in arrays.values())

        level_indptr = self._manifest_level_indptr
        descriptors: list[_TileDescriptor] = []
        lookup: dict[tuple[int, int, int], int] = {}
        attributes = self._attributes_or_raise()
        for level, metadata in enumerate(attributes.levels):
            start = int(level_indptr[level])
            stop = int(level_indptr[level + 1])
            for manifest_row in range(start, stop):
                descriptor = _TileDescriptor(
                    level=level,
                    bucket_id=int(self._manifest_bucket_id[manifest_row]),
                    bucket_tile_index=int(self._manifest_bucket_tile_index[manifest_row]),
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
        if len(descriptors) != len(self._manifest_n_points):
            raise ValueError("Manifest pointers do not cover every resident manifest row.")
        self._descriptors = tuple(descriptors)
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

    def _value_filtered_manifest(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_ids: npt.NDArray[np.uint32],
    ) -> dict[int, npt.NDArray[np.uint32]]:
        """Resolve value-to-tile discovery before bucket-local sparse ranges.

        The two-index flow is::

            selected value and level
                -> value_tiles/indptr
                -> manifest rows containing the value
                -> manifest bucket address
                -> ranges/tile_indptr
                -> value-specific row_start and row_count
                -> exact point rows

        This method owns only the cache-wide first half. `_BucketReader` owns
        the bucket-local second half after a positive manifest tile is known.

        Returns
        -------
        value_ids_by_manifest_row : dict[int, numpy.ndarray]
            Requested value IDs present in each positive visible manifest row.

        Examples
        --------
        Suppose requested value IDs ``[0, 1]`` have these level-wide
        ``value_tiles`` records:

        | Value | Manifest row | Points |
        |---:|---:|---:|
        | 0 | 100 | 3 |
        | 0 | 102 | 8 |
        | 0 | 104 | 1 |
        | 1 | 101 | 6 |
        | 1 | 104 | 2 |

        If the viewport intersects manifest rows ``[101, 102, 104]``, row
        ``100`` is discarded. The result is equivalent to::

            value_ids_by_manifest_row = {
                101: np.array([1], dtype=np.uint32),
                102: np.array([0], dtype=np.uint32),
                104: np.array([0, 1], dtype=np.uint32),
            }
        """
        visible = np.asarray(visible_rows, dtype=np.uint64)
        by_row: dict[int, list[int]] = {}
        for selected_value_positions, visible_positions, _ in self._iter_value_tile_matches(
            level,
            visible,
            value_ids,
        ):
            matched_value_ids = value_ids[selected_value_positions]
            for position, value_id in zip(visible_positions.tolist(), matched_value_ids.tolist(), strict=True):
                by_row.setdefault(int(visible[position]), []).append(value_id)

        return {row: np.ascontiguousarray(values, dtype=np.uint32) for row, values in by_row.items()}

    def _value_filtered_manifest_summary(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_ids: npt.NDArray[np.uint32],
    ) -> tuple[npt.NDArray[np.uint64], int]:
        """Return counts and the positive-tile union needed for LOD choice.

        Unlike viewport reading, level selection does not need to retain which
        values occur in each tile. Keep that per-tile mapping and its many small
        arrays out of the summary-only path.

        Returns
        -------
        counts_by_value : numpy.ndarray
            Visible point totals aligned with the requested ``value_ids``.
        positive_visible_tile_count : int
            Number of distinct visible manifest tiles containing at least one
            requested value. A tile containing several requested values is
            counted once.

        Examples
        --------
        Suppose requested value IDs ``[0, 1]`` have these records and the
        viewport intersects manifest rows ``[101, 102, 104]``:

        | Value | Manifest row | Points |
        |---:|---:|---:|
        | 0 | 100 | 3 |
        | 0 | 102 | 8 |
        | 0 | 104 | 1 |
        | 1 | 101 | 6 |
        | 1 | 104 | 2 |

        Row ``100`` is outside the viewport. The returned summary is equivalent
        to::

            counts_by_value = np.array([9, 8], dtype=np.uint64)
            positive_visible_tile_count = 3

        Value ``0`` contributes ``8 + 1`` points and value ``1`` contributes
        ``6 + 2``. Manifest row ``104`` contains both requested values, but it
        contributes only once to the positive-tile union ``{101, 102, 104}``.
        """
        visible = np.asarray(visible_rows, dtype=np.uint64)
        counts_by_value = np.zeros(len(value_ids), dtype=np.uint64)
        positive_visible = np.zeros(len(visible), dtype=np.bool_)
        for selected_value_positions, visible_positions, n_points in self._iter_value_tile_matches(
            level,
            visible,
            value_ids,
        ):
            np.add.at(counts_by_value, selected_value_positions, n_points)
            positive_visible[visible_positions] = True
        return np.ascontiguousarray(counts_by_value), int(np.count_nonzero(positive_visible))

    def _iter_value_tile_matches(
        self,
        level: int,
        visible: npt.NDArray[np.uint64],
        value_ids: npt.NDArray[np.uint32],
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.uint64]]]:
        """Yield selected-value positions, visible-row positions, and counts.

        Resolve all requested value intervals up front and read their connected
        catalog chunks in shard-bounded envelopes. Each envelope is intersected with
        the resident visible-manifest rows once, rather than repeating one Zarr
        selection and one viewport search for every requested value.

        Parameters
        ----------
        level
            Cache level whose value-to-tile catalog records are queried.
        visible
            Sorted global manifest rows for logical tiles intersecting the
            viewport. Positions in this array identify visible tiles within the
            current request.
        value_ids
            Sorted unique requested value IDs. Positions in this array preserve
            caller order for per-value count accumulation.

        Yields
        ------
        selected_value_positions : numpy.ndarray
            ``int64`` positions into ``value_ids``, one for every matching
            value/tile catalog record in the yielded envelope.
        visible_positions : numpy.ndarray
            Aligned ``int64`` positions into ``visible``.
        n_points : numpy.ndarray
            Aligned positive ``uint64`` point counts for those value/tile
            records.

        Notes
        -----
        The three arrays in each yielded tuple are row-aligned. Physical Zarr
        envelope boundaries may split one logical result across several yields;
        concatenating corresponding arrays produces the complete result.

        Examples
        --------
        Suppose the method receives::

            level = 2
            visible = np.array([101, 102, 104], dtype=np.uint64)
            value_ids = np.array([10, 42], dtype=np.uint32)

        and level 2 contains these catalog records:

        | Value ID | Manifest row | Points |
        |---:|---:|---:|
        | 10 | 100 | 3 |
        | 10 | 102 | 8 |
        | 10 | 104 | 1 |
        | 42 | 101 | 6 |
        | 42 | 104 | 2 |

        Manifest row ``100`` is not visible and is discarded. Ignoring possible
        physical envelope boundaries, the yielded arrays are equivalent to::

            selected_value_positions = np.array([0, 0, 1, 1], dtype=np.int64)
            visible_positions = np.array([1, 2, 0, 2], dtype=np.int64)
            n_points = np.array([8, 1, 6, 2], dtype=np.uint64)

        For example, the first aligned record resolves as::

            value_ids[selected_value_positions[0]] == 10
            visible[visible_positions[0]] == 102
            n_points[0] == 8

        It therefore means that value ``10`` occurs in visible manifest tile
        ``102`` with eight points. Manifest tile ``104`` appears twice because
        it contains both requested values.
        """
        if len(visible) == 0:
            return
        intervals = self._value_tile_intervals(level, value_ids)
        if not intervals:
            return
        attributes = self._attributes_or_raise()
        settings = attributes.catalog.settings
        envelopes = _value_tile_catalog_envelopes(
            intervals,
            chunk_rows=settings.value_tile_chunk_rows,
            shard_rows=settings.value_tile_shard_rows,
        )
        catalog = self._catalog_or_raise()
        manifest_array = catalog.array(VALUE_TILES_MANIFEST_INDEX)
        point_count_array = catalog.array(VALUE_TILES_N_POINTS)
        level_start = int(self._manifest_level_indptr_or_raise()[level])
        level_stop = int(self._manifest_level_indptr_or_raise()[level + 1])
        relative_visible = visible.astype(np.int64, copy=False) - level_start
        if bool((relative_visible < 0).any()) or bool((relative_visible >= level_stop - level_start).any()):
            raise ValueError("Visible manifest rows lie outside the requested level.")
        # One manifest level is compact (one row per nonempty logical tile),
        # whereas many value records can refer to each tile. Build this small
        # direct map once so every envelope can intersect its records in O(N)
        # instead of repeatedly binary-searching the visible rows.
        visible_position_by_level_row = np.full(level_stop - level_start, -1, dtype=np.int64)
        visible_position_by_level_row[relative_visible] = np.arange(len(visible), dtype=np.int64)
        previous_manifest = np.zeros(len(value_ids), dtype=np.uint64)
        has_previous = np.zeros(len(value_ids), dtype=np.bool_)

        for envelope in envelopes:
            manifest_index = np.asarray(
                manifest_array[envelope.start : envelope.stop],
                dtype=np.uint64,
            )
            n_points = np.asarray(
                point_count_array[envelope.start : envelope.stop],
                dtype=np.uint64,
            )
            interval_lengths = np.empty(len(envelope.intervals), dtype=np.int64)
            for interval_index, interval in enumerate(envelope.intervals):
                local_start = interval.start - envelope.start
                local_stop = interval.stop - envelope.start
                interval_manifest = manifest_index[local_start:local_stop]
                interval_counts = n_points[local_start:local_stop]
                selected_value_position = interval.selected_value_position
                if (
                    bool((interval_counts == 0).any())
                    or bool((interval_manifest < level_start).any())
                    or bool((interval_manifest >= level_stop).any())
                    or bool((interval_manifest[1:] <= interval_manifest[:-1]).any())
                    or (
                        has_previous[selected_value_position]
                        and int(interval_manifest[0]) <= int(previous_manifest[selected_value_position])
                    )
                ):
                    raise ValueError("Encountered invalid value-tile records during lookup.")
                previous_manifest[selected_value_position] = interval_manifest[-1]
                has_previous[selected_value_position] = True
                interval_lengths[interval_index] = local_stop - local_start

            selected_value_positions = np.repeat(
                np.fromiter(
                    (interval.selected_value_position for interval in envelope.intervals),
                    dtype=np.int64,
                    count=len(envelope.intervals),
                ),
                interval_lengths,
            )
            selected_row_count = int(interval_lengths.sum(dtype=np.int64))
            if selected_row_count == len(manifest_index):
                selected_manifest = manifest_index
                selected_counts = n_points
            else:
                selected_manifest = np.concatenate(
                    tuple(
                        manifest_index[interval.start - envelope.start : interval.stop - envelope.start]
                        for interval in envelope.intervals
                    )
                )
                selected_counts = np.concatenate(
                    tuple(
                        n_points[interval.start - envelope.start : interval.stop - envelope.start]
                        for interval in envelope.intervals
                    )
                )

            positions = visible_position_by_level_row[selected_manifest - np.uint64(level_start)]
            matches = positions >= 0
            if bool(matches.any()):
                yield (
                    np.ascontiguousarray(selected_value_positions[matches], dtype=np.int64),
                    np.ascontiguousarray(positions[matches], dtype=np.int64),
                    np.ascontiguousarray(selected_counts[matches], dtype=np.uint64),
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

    def _read_manifest_requests(
        self,
        level: int,
        requests: tuple[tuple[int, npt.NDArray[np.uint32] | None], ...],
    ) -> _ViewportReadResult:
        grouped: dict[tuple[int, int], list[tuple[int, npt.NDArray[np.uint32] | None]]] = {}
        for manifest_row, selected in requests:
            descriptor = self._descriptors[manifest_row]
            grouped.setdefault((descriptor.level, descriptor.bucket_id), []).append((manifest_row, selected))

        results: dict[int, _TileReadResult] = {}
        for (bucket_level, bucket_id), bucket_requests in grouped.items():
            reader = self._bucket_cache_or_raise().get(level=bucket_level, bucket_id=bucket_id)
            for manifest_row, selected in bucket_requests:
                descriptor = self._descriptors[manifest_row]
                payload = reader.read_display_payload(descriptor, selected)
                if payload is None:
                    raise ValueError("Catalog selected a tile whose bucket contains none of the requested values.")
                results[manifest_row] = self._tile_result(descriptor, payload)

        ordered_tiles = tuple(results[manifest_row] for manifest_row, _ in requests)
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

    def _require_level(self, level: int) -> _LevelMetadata:
        attributes = self._attributes_or_raise()
        _require_integer_in_range(level, "level", maximum=len(attributes.levels) - 1)
        return attributes.levels[level]

    def _catalog_or_raise(self) -> _CatalogReader:
        self._require_open_or_initializing()
        if self._catalog is None:
            raise RuntimeError("Catalog reader is not open.")
        return self._catalog

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
        self._manifest_level_indptr = None
        self._manifest_bucket_id = None
        self._manifest_bucket_tile_index = None
        self._manifest_tile_x = None
        self._manifest_tile_y = None
        self._manifest_n_points = None
        self._value_tiles_indptr = None
        self._value_n_points = None
        self._descriptors = ()
        self._manifest_row_by_tile = {}
        self._resident_index_bytes = 0
        self._open = False


def _read_only_array(catalog: _CatalogReader, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(catalog.array(name)[:], dtype=CATALOG_ARRAY_DTYPES[name]))
    array.flags.writeable = False
    return array


def _value_tile_catalog_envelopes(
    intervals: tuple[_ValueTileInterval, ...],
    *,
    chunk_rows: int,
    shard_rows: int,
) -> tuple[_ValueTileCatalogEnvelope, ...]:
    """Plan minimal connected catalog envelopes within physical shards.

    Split logical value intervals at shard boundaries, then combine fragments
    whose touched inner chunks overlap or are consecutive inside the same shard.
    Returned envelopes retain their first and last exact row bounds rather than
    expanding to complete chunk or shard edges.

    Notes
    -----
    Zarr can correctly serve a slice spanning several shards. The one-shard
    boundary is instead a deliberate resource policy. A cross-shard slice must
    still access each independent shard, consult its index, and decode the
    relevant inner chunks from that shard. Zarr then assembles the extracted
    rows into one returned array, so crossing the boundary can create a larger
    temporary result without eliminating the underlying shard accesses or
    chunk decoding. Keeping each envelope inside one shard gives every individual
    Zarr slice a deterministic base bound. With the default 1,048,576-row
    shards, the two parallel ``uint64`` result arrays contain at most 16 MiB for
    one envelope before exact-fragment indexes, masks, and other working arrays are
    considered. This is not a strict process-peak bound: generator loop locals
    from the preceding envelope may remain referenced until they are overwritten,
    and consumer output has its own memory cost.

    This bound applies to temporary decoded catalog envelopes, not to any
    retained output assembled from their exact selected fragments. Blocks are
    also not padded to shard edges. For a shard boundary at row 8, one logical
    interval ``[7:10]`` becomes exact fragments ``[7:8]`` and ``[8:10]``.
    Processing them as successive envelopes avoids one cross-shard returned slice
    without changing the logical interval.
    """
    _require_integer_in_range(chunk_rows, "chunk_rows", minimum=1, maximum=_INT64_MAX)
    _require_integer_in_range(shard_rows, "shard_rows", minimum=1, maximum=_INT64_MAX)
    if shard_rows % chunk_rows:
        raise ValueError("`shard_rows` must be a multiple of `chunk_rows`.")
    if not intervals:
        return ()
    if any(
        not isinstance(interval, _ValueTileInterval)
        or interval.selected_value_position < 0
        or interval.value_id < 0
        or interval.start < 0
        or interval.start >= interval.stop
        for interval in intervals
    ):
        raise ValueError("Catalog intervals must contain valid nonnegative half-open rows.")
    if any(
        current.selected_value_position <= previous.selected_value_position
        or current.value_id <= previous.value_id
        or current.start < previous.stop
        for previous, current in pairwise(intervals)
    ):
        raise ValueError("Catalog intervals must follow selected-value and nonoverlapping row order.")

    fragments: list[_ValueTileInterval] = []
    for interval in intervals:
        cursor = interval.start
        while cursor < interval.stop:
            shard_stop = ((cursor // shard_rows) + 1) * shard_rows
            fragment_stop = min(interval.stop, shard_stop)
            fragments.append(
                _ValueTileInterval(
                    selected_value_position=interval.selected_value_position,
                    value_id=interval.value_id,
                    start=cursor,
                    stop=fragment_stop,
                )
            )
            cursor = fragment_stop

    block_stop = fragments[0].stop
    block_shard = fragments[0].start // shard_rows
    last_chunk = (block_stop - 1) // chunk_rows
    block_intervals = [fragments[0]]
    envelopes: list[_ValueTileCatalogEnvelope] = []
    for fragment in fragments[1:]:
        fragment_shard = fragment.start // shard_rows
        first_chunk = fragment.start // chunk_rows
        fragment_last_chunk = (fragment.stop - 1) // chunk_rows
        if fragment_shard != block_shard or first_chunk > last_chunk + 1:
            envelopes.append(
                _ValueTileCatalogEnvelope(
                    intervals=tuple(block_intervals),
                )
            )
            block_stop = fragment.stop
            block_shard = fragment_shard
            last_chunk = fragment_last_chunk
            block_intervals = [fragment]
            continue
        block_stop = max(block_stop, fragment.stop)
        last_chunk = max(last_chunk, fragment_last_chunk)
        block_intervals.append(fragment)
    envelopes.append(
        _ValueTileCatalogEnvelope(
            intervals=tuple(block_intervals),
        )
    )
    return tuple(envelopes)


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
