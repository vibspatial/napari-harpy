"""Immutable viewer-side tiled-points layer values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final
from uuid import UUID

import numpy as np
import numpy.typing as npt

from napari_harpy.core.multi_scale_cache_points_zarr.models import (
    _expected_level_kind,
    _SerializedLevelKind,
)

DEFAULT_HARD_RENDER_POINT_BUDGET = 100_000
DEFAULT_TARGET_PIXELS_PER_POINT = 9.0
TILED_POINTS_VERTEX_DTYPE: Final = np.dtype([("a_position", np.float32, (2,)), ("a_value_id", np.float32)])
_UINT32_MAX = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class TiledPointsDatasetReference:
    """Identify one logical tiled-points dataset without storing point rows.

    The napari layer keeps neither eager point payloads nor a cache reader on
    the GUI thread. This reference therefore carries the stable dataset metadata
    needed to interpret runtime tiles: ``value_count`` validates that the
    presentation palette has exactly one row per canonical cache value, while
    ``x_origin`` and ``y_origin`` are consumed by
    ``VispyTiledPointsLayer._on_matrix_change()``. That method precomposes the
    shared cache origin into the layer's root transform, reconstructing intrinsic
    coordinates while the one packed snapshot VBO retains smaller cache-relative
    float32 positions.

    Parameters
    ----------
    cache_generation_id
        UUID of the completed cache generation represented by the layer.
    points_name
        Name of the source SpatialData points element.
    value_column
        Source column represented by cache ``value_id`` rows.
    value_count
        Complete canonical cache-vocabulary size. This belongs to the dataset
        contract rather than being inferred from the presentation palette.
    x_origin, y_origin
        Intrinsic origin shared by every serialized tile grid. The VisPy layer
        precomposes it into the root layer transform; together with a tile's
        logical coordinates and size, this positions tile-local point rows in
        the complete intrinsic dataset coordinate system without rewriting the
        point buffers as large absolute float32 coordinates.
    x_min, x_max, y_min, y_max
        Complete observed intrinsic-coordinate bounds of the cache.
    """

    cache_generation_id: str
    points_name: str
    value_column: str
    value_count: int
    x_origin: float
    y_origin: float
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
        if (
            not isinstance(self.value_count, int)
            or isinstance(self.value_count, bool)
            or not 1 <= self.value_count <= _UINT32_MAX + 1
        ):
            raise ValueError("`value_count` must be a positive integer addressable by uint32 value IDs.")
        for name in ("x_origin", "y_origin", "x_min", "x_max", "y_min", "y_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"`{name}` must be a finite number.")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Dataset minima must not exceed maxima.")
        if self.x_origin > self.x_min or self.y_origin > self.y_min:
            raise ValueError("Dataset origins must not exceed their corresponding observed minima.")


@dataclass(frozen=True)
class TiledPointsLayerStatus:
    """Describe the current cache-backed tiled-points display state."""

    level: int | None = None
    level_kind: _SerializedLevelKind | None = None
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
            expected_kind = _expected_level_kind(self.level)
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
    """Identify one logical tile in worker-owned decoded CPU residency.

    This is not a physical Zarr address. It combines the published cache
    generation, requested value selection, and logical tile coordinates so a
    decoded CPU payload is reused only for the dataset and selection that
    produced it. CPU residency maps this key to a ``TiledPointsRenderTile``;
    the renderer consumes only the worker-prepared complete render batch and
    does not retain these keys. ``logical_tile_key`` exposes the smaller
    ``(level, tile_x, tile_y)`` identity expected by the core cache reader.
    """

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
    """Retain one immutable decoded tile payload in worker CPU residency.

    The viewer runtime gives each newly decoded render tile independent
    ``location`` and ``value_id`` backing allocations before CPU residency.
    This dataclass validates those arrays and installs read-only views without
    copying their point buffers again. Logical tiles remain worker-local for CPU
    residency and render-batch construction; only the separately packed
    ``TiledPointsRenderBatch`` crosses to the GUI renderer.

    Parameters
    ----------
    key
        Cache-generation, value-selection, level, and logical-tile identity
        used by CPU residency.
    tile_size
        Intrinsic width and height of the logical tile, used to position its
        tile-relative coordinates.
    location
        Independently owned C-contiguous ``float32`` coordinates with shape
        ``(N, 2)``, expressed relative to the logical tile origin.
    value_id
        Independently owned C-contiguous ``uint32`` value IDs with shape
        ``(N,)``, aligned row-for-row with ``location``.
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
class TiledPointsRenderBatch:
    """Carry one immutable renderer-ready vertex allocation across Qt.

    Parameters
    ----------
    vertices
        One owning, C-contiguous, read-only structured array using
        ``TILED_POINTS_VERTEX_DTYPE``. ``a_position`` contains cache-relative
        ``(x, y)`` coordinates with logical tile offsets already folded in;
        ``a_value_id`` contains exact float32 palette indexes.

    Notes
    -----
    The worker owns construction and validation of this physical payload. Qt
    transports the enclosing snapshot as a Python object reference, and the
    GUI renderer reads this allocation only long enough to stage the stable
    VisPy vertex buffer with copy-safe lifetime semantics.
    """

    vertices: npt.NDArray[np.void]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.vertices, np.ndarray)
            or self.vertices.ndim != 1
            or self.vertices.dtype != TILED_POINTS_VERTEX_DTYPE
            or not self.vertices.flags.c_contiguous
            or not self.vertices.flags.owndata
            or self.vertices.flags.writeable
            or self.vertices.nbytes != len(self.vertices) * TILED_POINTS_VERTEX_DTYPE.itemsize
        ):
            raise ValueError(
                "`vertices` must be one owning, read-only, C-contiguous canonical tiled-points vertex array."
            )

    @classmethod
    def empty(cls) -> TiledPointsRenderBatch:
        """Return a valid owning immutable zero-row render batch."""
        vertices = np.empty(0, dtype=TILED_POINTS_VERTEX_DTYPE)
        vertices.flags.writeable = False
        return cls(vertices)

    @property
    def point_count(self) -> int:
        """Return the number of packed renderer rows in O(1)."""
        return len(self.vertices)

    @property
    def nbytes(self) -> int:
        """Return the logical packed vertex-payload byte count in O(1)."""
        return self.vertices.nbytes


@dataclass(frozen=True)
class TiledPointsRenderSnapshot:
    """Describe one coherent generation-bound renderer result.

    A snapshot is the complete render state for one viewport, not merely the
    tiles newly read from Zarr. The worker combines CPU-resident and newly read
    tiles in the plan's spatial order before constructing it::

        planned viewport tiles
                |
        already resident --+
                          +--> worker-local ordered tiles
        newly read --------+              |
                                           v
                                 validate tile order
                                           |
                                           v
                                 pack_render_tiles()
                                 |-- preflight and allocate one vertex array
                                 |-- fold tile-grid offsets into cache-relative a_position
                                 |-- copy value IDs into a_value_id
                                 `-- validate and make the allocation read-only
                                           |
                                           v
                              TiledPointsRenderBatch
                              `-- immutable vertices
                                           |
                                           v
                           TiledPointsRenderSnapshot
                           |-- render_batch: TiledPointsRenderBatch
                           |-- request/selection generations
                           |-- LOD and budget result
                           |-- rendered_tile_count
                           `-- omitted_value_ids
                                           |
                                           v
                                  worker -> GUI boundary
                              queued reference delivery
                                           |
                                           v
                         VispyTiledPointsLayer.apply_snapshot()
                                           |
                                           v
                                  replace the stable VBO payload
                                           |
                                           v
                              atomically activate the visual state

    Request and selection generations let the GUI reject an obsolete snapshot
    without mixing it with a newer visual state. Level fields report the LOD
    decision and omitted selected values. A within-budget snapshot carries one
    immutable packed render batch; an over-budget snapshot carries an empty
    batch and only describes why the active visual was retained. Decoded logical
    tiles remain worker-local and are released or retained by CPU residency after
    packing rather than crossing the GUI boundary.

    Parameters
    ----------
    cache_generation_id
        Identity of the published cache generation that produced the snapshot.
    request_generation
        Monotonic viewport-request generation used to reject stale results.
    selection_generation
        Value-selection generation against which the request was evaluated.
    requested_value_ids
        Selected value IDs, or ``None`` when all values were requested.
    level
        Serialized cache level chosen for the viewport.
    level_kind
        Semantic kind of ``level``: Exact, Bridge, or spatial.
    within_budget
        Whether the selected level satisfies the effective point budget.
    estimated_point_count
        Catalog-derived point count for the complete snapshot.
    omitted_value_ids
        Requested values present in the Exact viewport but absent from the
        selected sampled level. These are sampling omissions, not evidence of
        biological absence.
    rendered_tile_count
        Number of logical tiles packed into the render batch. This is zero when
        ``within_budget`` is false.
    render_batch
        Worker-prepared renderer payload for the same complete tile set. An
        over-budget snapshot carries a valid empty batch even when its estimate
        is nonzero.
    """

    cache_generation_id: str
    request_generation: int
    selection_generation: int
    requested_value_ids: tuple[int, ...] | None
    level: int
    level_kind: _SerializedLevelKind
    within_budget: bool
    estimated_point_count: int
    omitted_value_ids: tuple[int, ...]
    rendered_tile_count: int
    render_batch: TiledPointsRenderBatch

    def __post_init__(self) -> None:
        _require_cache_generation_id(self.cache_generation_id)
        _require_positive_integer(self.request_generation, "request_generation")
        _require_nonnegative_integer(self.selection_generation, "selection_generation")
        _require_value_ids(self.requested_value_ids, "requested_value_ids")
        _require_nonnegative_integer(self.level, "level")
        expected_kind = _expected_level_kind(self.level)
        if self.level_kind != expected_kind:
            raise ValueError("`level_kind` does not match `level`.")
        if not isinstance(self.within_budget, bool):
            raise ValueError("`within_budget` must be bool.")
        _require_nonnegative_integer(self.estimated_point_count, "estimated_point_count")
        _require_value_ids(self.omitted_value_ids, "omitted_value_ids", allow_none=False, allow_empty=True)
        if self.requested_value_ids is None:
            if self.omitted_value_ids:
                raise ValueError("An all-values snapshot cannot report omitted value IDs.")
        elif not set(self.omitted_value_ids).issubset(self.requested_value_ids):
            raise ValueError("`omitted_value_ids` must be a subset of `requested_value_ids`.")
        if not isinstance(self.render_batch, TiledPointsRenderBatch):
            raise ValueError("`render_batch` must be TiledPointsRenderBatch.")
        _require_nonnegative_integer(self.rendered_tile_count, "rendered_tile_count")
        if not self.within_budget:
            if self.rendered_tile_count or self.render_batch.point_count:
                raise ValueError("An over-budget snapshot must not contain point payloads.")
        elif (
            self.render_batch.point_count != self.estimated_point_count
            or (self.rendered_tile_count == 0) != (self.render_batch.point_count == 0)
            or self.rendered_tile_count > self.render_batch.point_count
        ):
            raise ValueError(
                "Within-budget rendered tile count and render batch must reconcile to the estimated point count."
            )

    @property
    def rendered_point_count(self) -> int:
        """Return the validated packed point count without iterating tiles."""
        return self.render_batch.point_count

    @property
    def all_exact_present_values_omitted(self) -> bool:
        """Return whether sampling removed every Exact-present selected value.

        Requested values already absent from the Exact viewport are not sampled
        omissions. A nonempty omission tuple together with a within-budget,
        zero-point selected snapshot therefore identifies the complete sampled
        omission case that needs an explicit viewer status.
        """
        return (
            self.within_budget
            and self.requested_value_ids is not None
            and self.estimated_point_count == 0
            and bool(self.omitted_value_ids)
        )


@dataclass(frozen=True)
class TiledPointsRenderResult:
    """Acknowledge whether the renderer applied one generation-bound snapshot.

    Parameters
    ----------
    request_generation
        Viewport-request generation copied from the candidate snapshot.
    selection_generation
        Value-selection generation copied from the candidate snapshot.
    applied
        Whether the renderer atomically activated the candidate tile set.
    """

    request_generation: int
    selection_generation: int
    applied: bool

    def __post_init__(self) -> None:
        _require_positive_integer(self.request_generation, "request_generation")
        _require_nonnegative_integer(self.selection_generation, "selection_generation")
        if not isinstance(self.applied, bool):
            raise ValueError("`applied` must be bool.")


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
