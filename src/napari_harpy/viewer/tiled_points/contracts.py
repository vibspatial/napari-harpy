"""Immutable viewer-side tiled-points layer values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import numpy as np
import numpy.typing as npt

DEFAULT_HARD_RENDER_POINT_BUDGET = 100_000
DEFAULT_TARGET_PIXELS_PER_POINT = 9.0
_UINT32_MAX = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class TiledPointsDatasetReference:
    """Identify one logical tiled-points dataset without storing point rows.

    Parameters
    ----------
    cache_generation_id
        UUID of the completed cache generation represented by the layer.
    points_name
        Name of the source SpatialData points element.
    value_column
        Source column represented by cache ``value_id`` rows.
    x_min, x_max, y_min, y_max
        Complete observed intrinsic-coordinate bounds of the cache.
    """

    cache_generation_id: str
    points_name: str
    value_column: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        if not isinstance(self.points_name, str) or not self.points_name:
            raise ValueError("`points_name` must be a nonempty string.")
        if not isinstance(self.value_column, str) or not self.value_column:
            raise ValueError("`value_column` must be a nonempty string.")
        for name in ("x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite number.")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Dataset minima must not exceed maxima.")


@dataclass(frozen=True)
class TiledPointsLayerStatus:
    """Describe the current cache-backed tiled-points display state."""

    level: int | None = None
    level_kind: Literal["exact", "bridge", "spatial"] | None = None
    rendered_point_count: int = 0
    rendered_tile_count: int = 0
    message: str = "Idle"
    sampled: bool = False
    omitted_value_ids: tuple[int, ...] = ()

    @property
    def level_label(self) -> str:
        """Return the canonical presentation label for the active cache level."""
        if self.level is None:
            return "—"
        if self.level_kind == "exact":
            return "Exact"
        if self.level_kind == "bridge":
            return "Bridge"
        return f"Spatial L{self.level}"

    def __post_init__(self) -> None:
        if (self.level is None) != (self.level_kind is None):
            raise ValueError("`level` and `level_kind` must either both be present or both be absent.")
        if self.level is not None:
            if not isinstance(self.level, int) or isinstance(self.level, bool) or self.level < 0:
                raise ValueError("`level` must be a nonnegative integer or None.")
            expected_kind = "exact" if self.level == 0 else "bridge" if self.level == 1 else "spatial"
            if self.level_kind != expected_kind:
                raise ValueError("`level_kind` does not match the serialized cache level.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("`message` must be a nonempty string.")
        for name in ("rendered_point_count", "rendered_tile_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"`{name}` must be a nonnegative integer.")
        if not isinstance(self.sampled, bool):
            raise ValueError("`sampled` must be bool.")
        if (
            not isinstance(self.omitted_value_ids, tuple)
            or any(
                not isinstance(value_id, int) or isinstance(value_id, bool) or value_id < 0
                for value_id in self.omitted_value_ids
            )
            or tuple(sorted(set(self.omitted_value_ids))) != self.omitted_value_ids
        ):
            raise ValueError("`omitted_value_ids` must contain sorted unique nonnegative integers.")


@dataclass(frozen=True)
class TiledPointsViewportState:
    """Describe one normalized napari viewport in intrinsic cache coordinates.

    The tiled-points layer produces this immutable state from a napari draw or
    a viewer-budget change and emits it through ``layer.events.viewport``. The
    GUI-side viewport coordinator consumes it; cache workers do not derive
    napari geometry themselves.

    The hard and screen-density budgets are retained as viewer-side diagnostic
    evidence. ``effective_point_budget`` is their derived minimum. Later cache
    planning consumes only the intrinsic bounds and that effective budget.
    """

    displayed_axes: tuple[int, int]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    canvas_width: int
    canvas_height: int
    hard_render_point_budget: int
    screen_density_budget: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.displayed_axes, tuple)
            or len(self.displayed_axes) != 2
            or any(not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 for axis in self.displayed_axes)
            or len(set(self.displayed_axes)) != 2
        ):
            raise ValueError("`displayed_axes` must contain two unique nonnegative integers.")
        for name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite float.")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Viewport bounds must have positive width and height.")
        for name in (
            "canvas_width",
            "canvas_height",
            "hard_render_point_budget",
            "screen_density_budget",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")

    @property
    def effective_point_budget(self) -> int:
        """Return the stricter of the hard and screen-density budgets."""
        return min(self.hard_render_point_budget, self.screen_density_budget)


@dataclass(frozen=True)
class TileResidencyKey:
    """Identify one decoded logical tile for one cache and value selection."""

    cache_generation_id: str
    requested_value_ids: tuple[int, ...] | None
    level: int
    tile_x: int
    tile_y: int

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        _require_value_ids(self.requested_value_ids, "requested_value_ids")
        for name in ("level", "tile_x", "tile_y"):
            _require_nonnegative_integer(getattr(self, name), name)

    @property
    def logical_tile_key(self) -> tuple[int, int, int]:
        """Return the core reader key ``(level, tile_x, tile_y)``."""
        return self.level, self.tile_x, self.tile_y


@dataclass(frozen=True)
class TiledPointsRenderTile:
    """Carry one immutable decoded tile payload across the renderer boundary.

    The viewer runtime gives each newly decoded render tile independent
    ``location`` and ``value_id`` backing allocations before CPU residency.
    This dataclass validates those arrays and installs read-only views without
    copying their point buffers again.
    """

    key: TileResidencyKey
    tile_size: int
    location: npt.NDArray[np.float32]
    value_id: npt.NDArray[np.uint32]

    def __post_init__(self) -> None:
        if not isinstance(self.key, TileResidencyKey):
            raise ValueError("`key` must be TileResidencyKey.")
        _require_positive_integer(self.tile_size, "tile_size")
        if (
            not isinstance(self.location, np.ndarray)
            or self.location.ndim != 2
            or self.location.shape[1] != 2
            or self.location.dtype != np.dtype(np.float32)
            or not self.location.flags.c_contiguous
            or len(self.location) == 0
        ):
            raise ValueError("`location` must be a C-contiguous float32 array with shape (N, 2).")
        if (
            not isinstance(self.value_id, np.ndarray)
            or self.value_id.ndim != 1
            or self.value_id.dtype != np.dtype(np.uint32)
            or not self.value_id.flags.c_contiguous
            or len(self.value_id) != len(self.location)
        ):
            raise ValueError("`value_id` must be a C-contiguous uint32 array aligned with `location`.")
        if not self.location.flags.owndata:
            raise ValueError("`location` must own its backing allocation.")
        if not self.value_id.flags.owndata:
            raise ValueError("`value_id` must own its backing allocation.")
        location = self.location.view()
        location.flags.writeable = False
        value_id = self.value_id.view()
        value_id.flags.writeable = False
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "value_id", value_id)

    @property
    def point_count(self) -> int:
        """Return the number of aligned display rows."""
        return len(self.value_id)

    @property
    def resident_bytes(self) -> int:
        """Return bytes retained by the two decoded point arrays."""
        return self.location.nbytes + self.value_id.nbytes


@dataclass(frozen=True)
class TiledPointsRenderSnapshot:
    """Describe one coherent generation-bound set of active render tiles.

    A snapshot is the complete render state for one viewport, not merely the
    tiles newly read from Zarr. The worker combines CPU-resident and newly read
    tiles in the plan's spatial order before constructing it::

        planned viewport tiles
                |
        already resident --+
                          +--> complete ordered tile set
        newly read --------+
                                  |
                                  v
                      TiledPointsRenderSnapshot
                                  |
                         worker -> GUI boundary
                                  |
                                  v
                      atomically replace visual state

    Request and selection generations let the GUI reject an obsolete snapshot
    without mixing it with a newer visual state. Level fields report the LOD
    decision and omitted selected values. A within-budget snapshot carries all
    active tile payloads; an over-budget snapshot deliberately carries none.

    The frozen snapshot and its immutable tiles form one validated thread-boundary
    value. Construction verifies that tiles share the snapshot's cache,
    selection, and level, and that their decoded point total reconciles with
    ``estimated_point_count``.
    """

    cache_generation_id: str
    request_generation: int
    selection_generation: int
    requested_value_ids: tuple[int, ...] | None
    level: int
    level_kind: Literal["exact", "bridge", "spatial"]
    within_budget: bool
    estimated_point_count: int
    omitted_value_ids: tuple[int, ...]
    tiles: tuple[TiledPointsRenderTile, ...]

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        _require_positive_integer(self.request_generation, "request_generation")
        _require_nonnegative_integer(self.selection_generation, "selection_generation")
        _require_value_ids(self.requested_value_ids, "requested_value_ids")
        _require_nonnegative_integer(self.level, "level")
        expected_kind = "exact" if self.level == 0 else "bridge" if self.level == 1 else "spatial"
        if self.level_kind != expected_kind:
            raise ValueError("`level_kind` does not match `level`.")
        if not isinstance(self.within_budget, bool):
            raise ValueError("`within_budget` must be bool.")
        _require_nonnegative_integer(self.estimated_point_count, "estimated_point_count")
        _require_value_ids(self.omitted_value_ids, "omitted_value_ids", allow_none=False, allow_empty=True)
        if not isinstance(self.tiles, tuple) or not all(isinstance(tile, TiledPointsRenderTile) for tile in self.tiles):
            raise ValueError("`tiles` must be a tuple of TiledPointsRenderTile values.")
        keys = tuple(tile.key for tile in self.tiles)
        if len(set(keys)) != len(keys):
            raise ValueError("Snapshot tile residency keys must be unique.")
        if tuple((key.tile_y, key.tile_x) for key in keys) != tuple(sorted((key.tile_y, key.tile_x) for key in keys)):
            raise ValueError("Snapshot tiles must follow spatial (tile_y, tile_x) order.")
        if any(
            key.cache_generation_id != self.cache_generation_id
            or key.requested_value_ids != self.requested_value_ids
            or key.level != self.level
            for key in keys
        ):
            raise ValueError("Every snapshot tile must match its cache, selection, and level.")
        if not self.within_budget and self.tiles:
            raise ValueError("An over-budget snapshot must not contain point payloads.")
        if self.within_budget and self.rendered_point_count != self.estimated_point_count:
            raise ValueError("Within-budget snapshot payloads must reconcile to the estimated point count.")

    @property
    def rendered_point_count(self) -> int:
        """Return the complete active decoded point count."""
        return sum(tile.point_count for tile in self.tiles)

    @property
    def rendered_tile_count(self) -> int:
        """Return the number of active logical tiles."""
        return len(self.tiles)


@dataclass(frozen=True)
class _ViewportRequest:
    """Carry one GUI-stamped viewport request to the cache worker.

    The coordinator creates this finalized request only when its pending
    viewport can be dispatched and the session has a committed value
    selection. One immutable object then crosses the Qt thread boundary::

        GUI thread                         worker thread
        ----------                         -------------
        construct _ViewportRequest
                `-- Qt queued signal ----> read immutable request

    The request and selection generations let the GUI reject obsolete results;
    ``requested_value_ids`` also lets the worker verify that the request agrees
    with its resident selected-value index.
    """

    request_generation: int
    selection_generation: int
    requested_value_ids: tuple[int, ...] | None
    viewport: TiledPointsViewportState

    def __post_init__(self) -> None:
        _require_positive_integer(self.request_generation, "request_generation")
        _require_nonnegative_integer(self.selection_generation, "selection_generation")
        _require_value_ids(self.requested_value_ids, "requested_value_ids")
        if not isinstance(self.viewport, TiledPointsViewportState):
            raise ValueError("`viewport` must be TiledPointsViewportState.")


def _require_cache_generation_id(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("`cache_generation_id` must be a UUID string.")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError("`cache_generation_id` must be a UUID string.") from error
    if str(parsed) != value:
        raise ValueError("`cache_generation_id` must use canonical UUID spelling.")


def _require_value_ids(
    value: object,
    name: str,
    *,
    allow_none: bool = True,
    allow_empty: bool = False,
) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"`{name}` must be a tuple.")
    if (
        not isinstance(value, tuple)
        or (not value and not allow_empty)
        or any(
            not isinstance(value_id, int) or isinstance(value_id, bool) or not 0 <= value_id <= _UINT32_MAX
            for value_id in value
        )
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError(f"`{name}` must contain sorted unique nonnegative uint32 integers.")


def _require_nonnegative_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"`{name}` must be a nonnegative integer.")


def _require_positive_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
