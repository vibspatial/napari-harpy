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

import numpy as np
import numpy.typing as npt
import zarr

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
    """Retain an immutable generation-bound selected-value index in memory."""

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
    """Read one trusted, completed Zarr cache generation.

    Entering validates the frozen root and array layouts, then materializes only
    the compact manifest, value pointer table, and value totals. It deliberately
    does not replay complete staged validation. Bucket stores are opened lazily
    or by explicit lookup priming and retained for this reader's lifetime.
    Display payload reads require their bucket lookup indexes to be resident;
    they never load lookup metadata implicitly on the viewport path.
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

    @property
    def loaded_bucket_lookup_index_count(self) -> int:
        """Return the number of buckets with resident lookup metadata."""
        return self._bucket_cache_or_raise().loaded_lookup_index_count

    @property
    def resident_bucket_lookup_bytes(self) -> int:
        """Return bytes retained by all loaded bucket lookup indexes."""
        return self._bucket_cache_or_raise().resident_lookup_bytes

    def project_bucket_lookup_index_bytes(
        self,
        *,
        levels: tuple[int, ...] | None = None,
        bucket_keys: tuple[tuple[int, int], ...] | None = None,
    ) -> int:
        """Return exact resident bytes for a requested bucket set.

        This opens the requested bucket readers and validates their Zarr layouts
        so their declared tile and range counts are available. It does not read
        tile pointers, sparse ranges, or point payload arrays.
        """
        keys = self._requested_bucket_keys(levels=levels, bucket_keys=bucket_keys)
        cache = self._bucket_cache_or_raise()
        return sum(cache.get(level=level, bucket_id=bucket_id).projected_lookup_bytes for level, bucket_id in keys)

    def load_bucket_lookup_indexes(
        self,
        *,
        max_resident_bytes: int,
        levels: tuple[int, ...] | None = None,
        bucket_keys: tuple[tuple[int, int], ...] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Explicitly load an immutable, byte-bounded bucket lookup set.

        The complete requested set is projected before any large lookup array is
        read. Loading is atomic with respect to indexes introduced by this call:
        if one bucket or the progress callback fails, those new indexes are
        released while indexes resident before the call remain available.

        Parameters
        ----------
        max_resident_bytes
            Positive upper bound for all cached bucket lookup-index array bytes
            after the operation, including indexes cached by earlier calls.
        levels
            Optional sorted unique levels to load. ``None`` with no
            ``bucket_keys`` requests every serialized bucket.
        bucket_keys
            Optional sorted unique ``(level, bucket_id)`` addresses. It is
            mutually exclusive with ``levels``.
        progress
            Optional callback receiving ``(completed_buckets, total_buckets)``
            after every requested bucket becomes ready.

        Returns
        -------
        int
            Exact bytes retained by all loaded bucket lookup indexes.
        """
        _require_integer_in_range(
            max_resident_bytes,
            "max_resident_bytes",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if progress is not None and not callable(progress):
            raise ValueError("`progress` must be callable or None.")
        keys = self._requested_bucket_keys(levels=levels, bucket_keys=bucket_keys)
        cache = self._bucket_cache_or_raise()
        readers = {
            key: cache.get(level=key[0], bucket_id=key[1])
            for key in keys
        }
        projected_total = cache.resident_lookup_bytes + sum(
            reader.projected_lookup_bytes
            for reader in readers.values()
            if not reader.lookup_index_loaded
        )
        if projected_total > max_resident_bytes:
            raise ValueError(
                f"Bucket lookup indexes require {projected_total} resident bytes, "
                f"exceeding `max_resident_bytes={max_resident_bytes}`."
            )

        newly_loaded: list[tuple[int, int]] = []
        try:
            for completed, key in enumerate(keys, start=1):
                reader = readers[key]
                if not reader.lookup_index_loaded:
                    reader.load_lookup_index()
                    newly_loaded.append(key)
                if progress is not None:
                    progress(completed, len(keys))
        except Exception:
            cache.release_lookup_indexes(tuple(newly_loaded))
            raise
        if cache.resident_lookup_bytes != projected_total:
            cache.release_lookup_indexes(tuple(newly_loaded))
            raise RuntimeError("Resident bucket lookup bytes differ from the preflight projection.")
        return cache.resident_lookup_bytes

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
        requests by bucket and preserves coordinated Zarr selection. The target
        bucket's lookup index must have been explicitly primed before this call.
        """
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

    def load_selected_value_index(
        self,
        value_ids: npt.NDArray[np.uint32],
        *,
        max_resident_bytes: int,
    ) -> _SelectedValueIndex | None:
        """Read and retain selected value-to-tile records for every level.

        This is the explicit selected-value catalog-I/O boundary. The returned
        immutable index is independent of a viewport and can be reused by every
        subsequent pan, zoom, LOD decision, and viewport payload request.
        Load a new index only when the selected value IDs change; ordinary
        viewport changes reuse this in-memory representation instead of
        reconstructing it from the catalog.

        The resident representation is compact relative to point payloads: it
        retains only selected value-to-tile manifest rows, aligned point counts,
        and per-value pointers—not point coordinates or point-level value IDs.
        Its exact NumPy-buffer footprint is projected and checked against
        ``max_resident_bytes`` before either large catalog array is read.

        Parameters
        ----------
        value_ids
            Nonempty sorted unique canonical value IDs.
        max_resident_bytes
            Maximum retained NumPy-buffer bytes allowed for the loaded index.

        Returns
        -------
        _SelectedValueIndex or None
            Generation-bound in-memory index, or ``None`` when all canonical
            values were selected and the resident all-values path should be used.
        """
        value_ids = self._require_value_ids(value_ids)
        if value_ids is None:
            raise ValueError("`value_ids` must be supplied when loading a selected-value index.")
        _require_integer_in_range(
            max_resident_bytes,
            "max_resident_bytes",
            minimum=1,
            maximum=_INT64_MAX,
        )
        if len(value_ids) == len(self.value_names):
            return None

        pointers = self._value_tiles_indptr_or_raise()
        indexes = value_ids.astype(np.int64, copy=False)
        record_counts = pointers[:, indexes + 1] - pointers[:, indexes]
        projected_bytes = value_ids.nbytes + pointers.shape[0] * (len(value_ids) + 1) * np.dtype(np.uint64).itemsize
        projected_bytes += int(record_counts.sum(dtype=np.uint64)) * 2 * np.dtype(np.uint64).itemsize
        if projected_bytes > max_resident_bytes:
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

    def read_viewport(
        self,
        level: int,
        viewport: _IntrinsicViewport,
        *,
        value_index: _SelectedValueIndex | None = None,
    ) -> _ViewportReadResult:
        """Read positive logical tiles intersecting one intrinsic viewport.

        Parameters
        ----------
        level
            Serialized level to read.
        viewport
            Half-open intrinsic-coordinate viewport used for complete-tile
            intersection.
        value_index
            Loaded selected-value index, or ``None`` for all values.

        Notes
        -----
        Positive-tile discovery uses only resident manifest and selected-index arrays.
        Every positive bucket's lookup index must already be resident. The
        subsequent bucket-local point payload reads remain I/O by design.
        """
        self._require_level(level)
        value_index = self._require_selected_value_index(value_index)
        visible_rows = self._visible_manifest_rows(level, viewport)
        if len(visible_rows) == 0:
            return _ViewportReadResult(level=level, tiles=())

        # Each request pairs a global manifest row with either None for all
        # values or the selected value IDs known to be present in that tile.
        if value_index is None:
            requests = tuple((int(row), None) for row in visible_rows)
        else:
            # Intersect the immutable selected-value index with the visible manifest
            # rows. Catalog Zarr I/O is forbidden on this viewport-time path.
            value_ids_by_manifest_row = self._selected_value_manifest(level, visible_rows, value_index)
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
        descriptors_by_bucket: dict[tuple[int, int], list[_TileDescriptor]] = {}
        for descriptor in self._descriptors:
            descriptors_by_bucket.setdefault((descriptor.level, descriptor.bucket_id), []).append(descriptor)
        self._descriptors_by_bucket = {
            key: tuple(bucket_descriptors)
            for key, bucket_descriptors in descriptors_by_bucket.items()
        }
        self._manifest_row_by_tile = lookup

    def _requested_bucket_keys(
        self,
        *,
        levels: tuple[int, ...] | None,
        bucket_keys: tuple[tuple[int, int], ...] | None,
    ) -> tuple[tuple[int, int], ...]:
        """Normalize one all-level, level-scoped, or bucket-scoped prime request."""
        self._require_open()
        if levels is not None and bucket_keys is not None:
            raise ValueError("`levels` and `bucket_keys` are mutually exclusive.")
        available = self._descriptors_by_bucket
        if levels is None and bucket_keys is None:
            return tuple(sorted(available))
        if levels is not None:
            if (
                not isinstance(levels, tuple)
                or not levels
                or any(not isinstance(level, int) or isinstance(level, bool) for level in levels)
                or levels != tuple(sorted(set(levels)))
            ):
                raise ValueError("`levels` must be a nonempty sorted unique tuple of integers.")
            for level in levels:
                self._require_level(level)
            selected_levels = set(levels)
            return tuple(key for key in sorted(available) if key[0] in selected_levels)

        if bucket_keys is None:
            raise RuntimeError("Bucket-key normalization reached an impossible state.")
        if (
            not isinstance(bucket_keys, tuple)
            or not bucket_keys
            or any(
                not isinstance(key, tuple)
                or len(key) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in key)
                for key in bucket_keys
            )
            or bucket_keys != tuple(sorted(set(bucket_keys)))
        ):
            raise ValueError("`bucket_keys` must be a nonempty sorted unique tuple of (level, bucket_id) pairs.")
        unknown = tuple(key for key in bucket_keys if key not in available)
        if unknown:
            raise ValueError(f"`bucket_keys` contains an unknown bucket address: {unknown[0]}.")
        return bucket_keys

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

    def _selected_value_manifest(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_index: _SelectedValueIndex,
    ) -> dict[int, npt.NDArray[np.uint32]]:
        """Map positive visible manifest rows to their selected value IDs.

        This method performs the in-memory value-to-tile discovery needed before
        bucket-local sparse-range lookup. The complete selected read flow is::

            resident selected-value level records
                -> visible manifest rows containing each selected value
                -> manifest bucket address
                -> ranges/tile_indptr
                -> value-specific row_start and row_count
                -> exact point rows

        This method owns only the in-memory cache-wide first half. `_BucketReader`
        owns the bucket-local second half after a positive manifest tile and its
        applicable selected values are known.

        Parameters
        ----------
        level
            Cache level whose resident value-to-tile records are queried.
        visible_rows
            Sorted global manifest rows for logical tiles intersecting the
            viewport.
        value_index
            Generation-validated immutable selected-value index.

        Returns
        -------
        value_ids_by_manifest_row : dict[int, numpy.ndarray]
            Requested value IDs present in each positive visible manifest row.
            Each value array is sorted, unique, C-contiguous, and ``uint32``.

        Notes
        -----
        This operation reads no catalog Zarr array, bucket, or point payload.
        The returned mapping is consumed later to prune empty tiles and request
        only applicable bucket-local sparse value ranges.

        Examples
        --------
        Suppose the selected-value index contains value IDs ``[0, 1]`` and these
        level records:

        | Value | Manifest row | Points |
        |---:|---:|---:|
        | 0 | 100 | 3 |
        | 0 | 102 | 8 |
        | 0 | 104 | 1 |
        | 1 | 101 | 6 |
        | 1 | 104 | 2 |

        If ``visible_rows`` is ``[101, 102, 104]``, manifest row ``100`` is
        outside the viewport and is discarded. The result is equivalent to::

            value_ids_by_manifest_row = {
                101: np.array([1], dtype=np.uint32),
                102: np.array([0], dtype=np.uint32),
                104: np.array([0, 1], dtype=np.uint32),
            }
        """
        visible = np.asarray(visible_rows, dtype=np.uint64)
        by_row: dict[int, list[int]] = {}
        for selected_position, visible_positions, _ in self._iter_selected_value_matches(
            level,
            visible,
            value_index,
        ):
            value_id = int(value_index.value_ids[selected_position])
            for position in visible_positions.tolist():
                by_row.setdefault(int(visible[position]), []).append(value_id)
        return {row: np.ascontiguousarray(values, dtype=np.uint32) for row, values in by_row.items()}

    def _selected_value_manifest_summary(
        self,
        level: int,
        visible_rows: npt.NDArray[np.int64],
        value_index: _SelectedValueIndex,
    ) -> tuple[npt.NDArray[np.uint64], int]:
        """Return indexed counts and the positive-tile union needed for LOD.

        Level selection needs visible point totals and the number of distinct
        positive tiles, but not the value IDs applicable to each tile. This
        summary-only path therefore avoids constructing the dictionary and many
        small arrays produced by `_selected_value_manifest`.

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
        Using the same records as `_selected_value_manifest`, with selected
        value IDs ``[0, 1]`` and visible manifest rows ``[101, 102, 104]``, the
        result is equivalent to::

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
        ):
            counts_by_value[selected_position] = n_points.sum(dtype=np.uint64)
            positive_visible[visible_positions] = True
        return counts_by_value, int(np.count_nonzero(positive_visible))

    def _iter_selected_value_matches(
        self,
        level: int,
        visible: npt.NDArray[np.uint64],
        value_index: _SelectedValueIndex,
    ) -> Iterator[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.uint64]]]:
        """Yield each selected value's visible tile positions and point counts.

        Intersect one immutable level index with the resident
        visible manifest rows. This is the shared in-memory primitive behind LOD
        summaries and the manifest-row-to-value mapping used by viewport reads.

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

        Yields
        ------
        selected_value_position : int
            Position of the represented value in ``value_index.value_ids``.
        visible_positions : numpy.ndarray
            ``int64`` positions into ``visible`` for tiles containing that value.
        n_points : numpy.ndarray
            Aligned positive ``uint64`` point counts for those value/tile records.

        Notes
        -----
        Empty indexed value intervals and values with no visible tiles produce
        no yield. The two yielded arrays are row-aligned and C-contiguous. This
        method reads no Zarr catalog array, opens no bucket, and reads no point
        payload; all inputs were materialized by ``load_selected_value_index``.

        Examples
        --------
        Suppose the method receives::

            level = 2
            visible = np.array([101, 102, 104], dtype=np.uint64)
            value_index.value_ids = np.array([10, 42], dtype=np.uint32)

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
        the corresponding canonical value ID.
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
        # 4. yield visible positions and their aligned point counts.
        for selected_position, (start, stop) in enumerate(pairwise(level_index.value_indptr.tolist())):
            if start == stop:
                continue
            manifest_index = level_index.manifest_index[start:stop]
            positions = visible_position_by_level_row[manifest_index - np.uint64(level_start)]
            matches = positions >= 0
            if bool(matches.any()):
                yield (
                    selected_position,
                    np.ascontiguousarray(positions[matches], dtype=np.int64),
                    np.ascontiguousarray(level_index.n_points[start:stop][matches], dtype=np.uint64),
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
        """Read manifest-addressed tiles through one batched call per bucket.

        A manifest request remains one logical tile request. This method groups
        those requests by physical ``(level, bucket_id)`` and makes exactly one
        ``read_display_payloads`` call for each nonempty bucket group. The call
        contains every requested tile in that bucket; the bucket reader performs
        coordinated point-array selections and returns one result per tile. The
        resulting tile payloads are finally restored to the original manifest
        request order::

            manifest tile requests
                    -> group by (level, bucket_id)
                    -> one reader call per bucket, containing all its tiles
                    -> one payload per logical tile
                    -> restore original request order

        Buckets are processed sequentially. Batching here concerns the tiles
        within each bucket and does not introduce cross-bucket concurrency.

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
        grouped: dict[tuple[int, int], list[tuple[int, npt.NDArray[np.uint32] | None]]] = {}
        for manifest_row, selected in requests:
            descriptor = self._descriptors[manifest_row]
            grouped.setdefault((descriptor.level, descriptor.bucket_id), []).append((manifest_row, selected))

        results: dict[int, _TileReadResult] = {}
        for (bucket_level, bucket_id), bucket_requests in grouped.items():
            reader = self._bucket_cache_or_raise().get(level=bucket_level, bucket_id=bucket_id)
            # This is one physical-reader call for the bucket, not one call per
            # tile. Its tuple retains every logical tile request in the group.
            # Manifest rows remain a catalog concern, so the physical reader
            # receives only bucket-local descriptors and selections.
            payloads = reader.read_display_payloads(
                tuple((self._descriptors[manifest_row], selected) for manifest_row, selected in bucket_requests)
            )
            for (manifest_row, _), payload in zip(bucket_requests, payloads, strict=True):
                if payload is None:
                    raise ValueError("Catalog selected a tile whose bucket contains none of the requested values.")
                descriptor = self._descriptors[manifest_row]
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
        self._descriptors_by_bucket = {}
        self._manifest_row_by_tile = {}
        self._resident_index_bytes = 0
        self._open = False


def _read_only_array(catalog: _CatalogReader, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(catalog.array(name)[:], dtype=CATALOG_ARRAY_DTYPES[name]))
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
    ``storage.bucket_reader``. The helpers deliberately remain separate because
    this function validates value-major catalog intervals, while its counterpart
    validates untyped bucket point-row pairs.
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

    merged: list[tuple[int, int]] = []
    observed_row_count = 0
    for interval in intervals:
        observed_row_count += interval.stop - interval.start
        if merged and interval.start == merged[-1][1]:
            merged[-1] = (merged[-1][0], interval.stop)
        else:
            merged.append((interval.start, interval.stop))
    if observed_row_count != expected_row_count:
        raise ValueError("Catalog intervals do not reconcile to the expected selected record count.")

    if len(merged) == 1:
        start, stop = merged[0]
        return slice(start, stop)

    # Fill one exact selector without constructing one Python integer per
    # catalog row. Each destination segment is shifted to its source interval.
    selected_rows = np.arange(observed_row_count, dtype=np.int64)
    cursor = 0
    for start, stop in merged:
        count = stop - start
        selected_rows[cursor : cursor + count] += start - cursor
        cursor += count
    return selected_rows


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
