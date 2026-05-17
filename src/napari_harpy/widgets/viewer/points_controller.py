from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from napari_harpy._points_value_index import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_RENDER_POINT_BUDGET,
    DEFAULT_X,
    DEFAULT_Y,
    PointsValueSelection,
    PointsValueTable,
    _ValidatedPointsElement,
    build_points_value_table,
    load_points,
    validate_points_element_for_value_selection,
)
from napari_harpy.viewer.adapter import PointsLayerIdentity

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


thread_worker = _resolve_thread_worker()

PointsStatusKind = Literal["info", "warning", "success", "error"]
POINTS_IDLE_STATUS = "Points: choose a points element and index column."


class PointsControllerState(Enum):
    """State machine for direct points value loading."""

    NO_SDATA = "no_sdata"
    NO_POINTS_ELEMENT = "no_points_element"
    LOADING_VALUES = "loading_values"
    VALUES_READY = "values_ready"
    LOADING_SELECTION = "loading_selection"
    LOADED_SELECTION = "loaded_selection"
    LOAD_FAILED = "load_failed"


@dataclass(frozen=True)
class PointsValueSourceJob:
    """Immutable payload copied on the main thread and consumed by a value-table worker."""

    job_id: int
    sdata: SpatialData
    points_name: str
    coordinate_system: str
    index_column: str
    x: str = DEFAULT_X
    y: str = DEFAULT_Y
    transcript_id: str | None = None


@dataclass(frozen=True)
class PointsLoadJob:
    """Immutable payload copied on the main thread and consumed by a selected-points worker."""

    job_id: int
    value_source: PointsValueSource
    values: Sequence[str] | Literal["all"]
    render_point_budget: int = DEFAULT_RENDER_POINT_BUDGET
    random_state: int | None = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class PointsValueSource:
    """Validated points source plus its in-memory value table."""

    identity: PointsLayerIdentity
    validated: _ValidatedPointsElement
    value_table: PointsValueTable

    def __post_init__(self) -> None:
        if self.identity.points_name != self.validated.points_name:
            raise ValueError(
                "`identity.points_name` and `validated.points_name` must match. "
                f"Got {self.identity.points_name!r} and {self.validated.points_name!r}."
            )
        if self.identity.index_column != self.validated.index_column:
            raise ValueError(
                "`identity.index_column` and `validated.index_column` must match. "
                f"Got {self.identity.index_column!r} and {self.validated.index_column!r}."
            )
        if self.validated.index_column != self.value_table.index_column:
            raise ValueError(
                "`validated.index_column` and `value_table.index_column` must match. "
                f"Got {self.validated.index_column!r} and {self.value_table.index_column!r}."
            )


@dataclass(frozen=True)
class PointsLoadResult:
    """Selected points loaded by the controller, before napari layer application."""

    identity: PointsLayerIdentity
    selection: PointsValueSelection
    value_table: PointsValueTable

    def __post_init__(self) -> None:
        if self.identity.index_column != self.selection.index_column:
            raise ValueError(
                "`identity.index_column` and `selection.index_column` must match. "
                f"Got {self.identity.index_column!r} and {self.selection.index_column!r}."
            )
        if self.selection.index_column != self.value_table.index_column:
            raise ValueError(
                "`selection.index_column` and `value_table.index_column` must match. "
                f"Got {self.selection.index_column!r} and {self.value_table.index_column!r}."
            )


@thread_worker(start_thread=False, ignore_errors=True)
def _run_points_value_source_job(job: PointsValueSourceJob) -> PointsValueSource:
    """Validate a points source and build its value table in a worker thread.

    This is the expensive source-preparation step for direct points value
    selection. It validates that the selected SpatialData points element has
    usable coordinate and index columns, then builds the in-memory
    ``PointsValueTable`` containing one row per normalized index value and its
    point count.

    The work can trigger Dask computation over the full points element, so it
    must not run on the Qt main thread. Keeping it in this worker lets the
    widget stay responsive while the controller later reuses the resulting
    ``PointsValueSource`` for many value selections from the same source.
    """
    validated = validate_points_element_for_value_selection(
        job.sdata,
        job.points_name,
        x=job.x,
        y=job.y,
        index_column=job.index_column,
        transcript_id=job.transcript_id,
    )
    value_table = build_points_value_table(validated)
    return PointsValueSource(
        identity=PointsLayerIdentity(
            sdata=job.sdata,
            points_name=job.points_name,
            coordinate_system=job.coordinate_system,
            index_column=job.index_column,
        ),
        validated=validated,
        value_table=value_table,
    )


@thread_worker(start_thread=False, ignore_errors=True)
def _run_points_load_job(job: PointsLoadJob) -> PointsLoadResult:
    selection = load_points(
        job.value_source.validated,
        job.value_source.value_table,
        job.values,
        render_point_budget=job.render_point_budget,
        random_state=job.random_state,
    )
    return PointsLoadResult(
        identity=job.value_source.identity,
        selection=selection,
        value_table=job.value_source.value_table,
    )


class PointsController:
    """Manage async direct points value loading for the viewer widget."""

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed

        self._sdata: SpatialData | None = None
        self._points_name: str | None = None
        self._coordinate_system: str | None = None
        self._index_column: str | None = None
        self._x = DEFAULT_X
        self._y = DEFAULT_Y
        self._transcript_id: str | None = None

        self._current_value_source: PointsValueSource | None = None
        self._current_load_result: PointsLoadResult | None = None

        self._latest_value_job_id = 0
        self._active_value_worker_job_id: int | None = None
        self._active_value_worker: Any | None = None
        self._latest_load_job_id = 0
        self._active_load_worker_job_id: int | None = None
        self._active_load_worker: Any | None = None

        self._state = PointsControllerState.NO_SDATA
        self._status_message = POINTS_IDLE_STATUS
        self._status_kind: PointsStatusKind = "warning"

    @property
    def state(self) -> PointsControllerState:
        """Return the current points controller state."""
        return self._state

    @property
    def status_message(self) -> str:
        """Return the latest user-facing points status message."""
        return self._status_message

    @property
    def status_kind(self) -> PointsStatusKind:
        """Return the latest status level: info, warning, success, or error."""
        return self._status_kind

    @property
    def current_value_source(self) -> PointsValueSource | None:
        """Return the latest validated source and value table, if available."""
        return self._current_value_source

    @property
    def current_load_result(self) -> PointsLoadResult | None:
        """Return the latest selected-points load result, if available."""
        return self._current_load_result

    @property
    def is_loading_values(self) -> bool:
        """Return whether value-table construction is currently running."""
        return self._active_value_worker is not None

    @property
    def is_loading(self) -> bool:
        """Return whether selected-point loading is currently running."""
        return self._active_load_worker is not None

    @property
    def is_building(self) -> bool:
        """Return whether an optional cache build is currently running."""
        return False

    @property
    def cache_status(self) -> str:
        """Return the optional cache status for the current direct-first implementation."""
        return "not_available"

    @property
    def can_load_values(self) -> bool:
        """Return whether the controller has enough bound input to load values."""
        return (
            self._sdata is not None
            and self._points_name is not None
            and self._coordinate_system is not None
            and self._index_column is not None
            and not self.is_loading_values
        )

    @property
    def can_visualize(self) -> bool:
        """Return whether selected values can be loaded from the current source."""
        return self._current_value_source is not None and not self.is_loading

    @property
    def can_build_cache(self) -> bool:
        """Return whether direct-first MVP supports cache building."""
        return False

    @property
    def can_rebuild_cache(self) -> bool:
        """Return whether direct-first MVP supports cache rebuilding."""
        return False

    def bind_source(
        self,
        sdata: SpatialData | None,
        points_name: str | None,
        coordinate_system: str | None,
        index_column: str | None,
        *,
        x: str = DEFAULT_X,
        y: str = DEFAULT_Y,
        transcript_id: str | None = None,
    ) -> bool:
        """Bind the controller to the points source selected in the widget."""
        normalized_points_name = _normalize_optional_text(points_name)
        normalized_coordinate_system = _normalize_optional_text(coordinate_system)
        normalized_index_column = _normalize_optional_text(index_column)
        normalized_x = _normalize_required_text(x, DEFAULT_X)
        normalized_y = _normalize_required_text(y, DEFAULT_Y)
        normalized_transcript_id = _normalize_optional_text(transcript_id)

        changed = (
            sdata is not self._sdata
            or normalized_points_name != self._points_name
            or normalized_coordinate_system != self._coordinate_system
            or normalized_index_column != self._index_column
            or normalized_x != self._x
            or normalized_y != self._y
            or normalized_transcript_id != self._transcript_id
        )

        self._sdata = sdata
        self._points_name = normalized_points_name
        self._coordinate_system = normalized_coordinate_system
        self._index_column = normalized_index_column
        self._x = normalized_x
        self._y = normalized_y
        self._transcript_id = normalized_transcript_id

        if changed:
            self._current_value_source = None
            self._current_load_result = None
            self._cancel_value_worker()
            self._cancel_load_worker()

        self._update_bound_status()
        return changed

    def load_values(self) -> bool:
        """Launch async validation and direct value-table construction."""
        job = self._prepare_value_source_job()
        if job is None:
            return False

        self._latest_value_job_id = job.job_id
        self._current_value_source = None
        self._current_load_result = None
        self._cancel_value_worker()
        self._cancel_load_worker()

        worker = self._create_value_source_worker(job)
        self._active_value_worker = worker
        self._active_value_worker_job_id = job.job_id
        worker.returned.connect(partial(self._on_value_worker_returned, job.job_id))
        worker.errored.connect(partial(self._on_value_worker_errored, job.job_id))
        worker.finished.connect(partial(self._on_value_worker_finished, job.job_id))
        self._set_state_status(
            PointsControllerState.LOADING_VALUES,
            f"Points: loading values for `{job.points_name}` by `{job.index_column}`.",
            kind="info",
        )
        worker.start()
        return True

    def load_selection(
        self,
        values: Sequence[str] | Literal["all"],
        *,
        render_point_budget: int,
        random_state: int | None = DEFAULT_RANDOM_STATE,
    ) -> bool:
        """Launch async selected-point loading from the current value source."""
        job = self._prepare_load_job(
            values,
            render_point_budget=render_point_budget,
            random_state=random_state,
        )
        if job is None:
            return False

        self._latest_load_job_id = job.job_id
        self._current_load_result = None
        self._cancel_load_worker()

        worker = self._create_points_load_worker(job)
        self._active_load_worker = worker
        self._active_load_worker_job_id = job.job_id
        worker.returned.connect(partial(self._on_load_worker_returned, job.job_id))
        worker.errored.connect(partial(self._on_load_worker_errored, job.job_id))
        worker.finished.connect(partial(self._on_load_worker_finished, job.job_id))
        self._set_state_status(
            PointsControllerState.LOADING_SELECTION,
            f"Points: loading selected `{job.value_source.value_table.index_column}` values.",
            kind="info",
        )
        worker.start()
        return True

    def shutdown(self) -> None:
        """Cancel active async work before the owning widget is destroyed."""
        self._cancel_value_worker()
        self._cancel_load_worker()

    def _prepare_value_source_job(self) -> PointsValueSourceJob | None:
        if self._sdata is None:
            self._set_state_status(
                PointsControllerState.NO_SDATA,
                "Points: load a SpatialData object before loading values.",
                kind="warning",
            )
            return None
        if self._points_name is None or self._coordinate_system is None:
            self._set_state_status(
                PointsControllerState.NO_POINTS_ELEMENT,
                "Points: choose a points element and coordinate system.",
                kind="warning",
            )
            return None
        if self._index_column is None:
            self._set_state_status(
                PointsControllerState.LOAD_FAILED,
                "Points: choose a non-empty index column.",
                kind="error",
            )
            return None

        return PointsValueSourceJob(
            job_id=self._latest_value_job_id + 1,
            sdata=self._sdata,
            points_name=self._points_name,
            coordinate_system=self._coordinate_system,
            index_column=self._index_column,
            x=self._x,
            y=self._y,
            transcript_id=self._transcript_id,
        )

    def _prepare_load_job(
        self,
        values: Sequence[str] | Literal["all"],
        *,
        render_point_budget: int,
        random_state: int | None,
    ) -> PointsLoadJob | None:
        value_source = self._current_value_source
        if value_source is None:
            self._set_state_status(
                PointsControllerState.LOAD_FAILED,
                "Points: load values before visualizing a selection.",
                kind="error",
            )
            return None
        if isinstance(render_point_budget, bool) or not isinstance(render_point_budget, int) or render_point_budget <= 0:
            self._set_state_status(
                PointsControllerState.LOAD_FAILED,
                "Points: render point budget must be a positive integer.",
                kind="error",
            )
            return None

        return PointsLoadJob(
            job_id=self._latest_load_job_id + 1,
            value_source=value_source,
            values=values,
            render_point_budget=int(render_point_budget),
            random_state=random_state,
        )

    def _create_value_source_worker(self, job: PointsValueSourceJob) -> Any:
        return _run_points_value_source_job(job)

    def _create_points_load_worker(self, job: PointsLoadJob) -> Any:
        return _run_points_load_job(job)

    def _on_value_worker_returned(self, job_id: int, result: PointsValueSource) -> None:
        if job_id != self._latest_value_job_id or job_id != self._active_value_worker_job_id:
            return

        self._current_value_source = result
        self._current_load_result = None
        value_count = len(result.value_table.values)
        self._set_state_status(
            PointsControllerState.VALUES_READY,
            f"Points: loaded {value_count:,} values from `{result.identity.points_name}`.",
            kind="success",
        )

    def _on_value_worker_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_value_job_id or job_id != self._active_value_worker_job_id:
            return

        self._current_value_source = None
        self._current_load_result = None
        self._set_state_status(
            PointsControllerState.LOAD_FAILED,
            f"Points: value loading failed: {error}",
            kind="error",
        )

    def _on_value_worker_finished(self, job_id: int) -> None:
        if job_id != self._active_value_worker_job_id:
            return

        self._active_value_worker = None
        self._active_value_worker_job_id = None
        self._notify_state_changed()

    def _on_load_worker_returned(self, job_id: int, result: PointsLoadResult) -> None:
        if job_id != self._latest_load_job_id or job_id != self._active_load_worker_job_id:
            return

        self._current_load_result = result
        selection = result.selection
        if selection.is_sampled and selection.warning:
            self._set_state_status(
                PointsControllerState.LOADED_SELECTION,
                f"Points: {selection.warning}",
                kind="warning",
            )
        else:
            self._set_state_status(
                PointsControllerState.LOADED_SELECTION,
                f"Points: loaded {selection.loaded_count:,} selected points.",
                kind="success",
            )

    def _on_load_worker_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_load_job_id or job_id != self._active_load_worker_job_id:
            return

        self._current_load_result = None
        self._set_state_status(
            PointsControllerState.LOAD_FAILED,
            f"Points: selection loading failed: {error}",
            kind="error",
        )

    def _on_load_worker_finished(self, job_id: int) -> None:
        if job_id != self._active_load_worker_job_id:
            return

        self._active_load_worker = None
        self._active_load_worker_job_id = None
        self._notify_state_changed()

    def _update_bound_status(self) -> None:
        if self._sdata is None:
            self._set_state_status(
                PointsControllerState.NO_SDATA,
                "Points: load a SpatialData object.",
                kind="warning",
            )
            return
        if self._points_name is None or self._coordinate_system is None:
            self._set_state_status(
                PointsControllerState.NO_POINTS_ELEMENT,
                "Points: choose a points element and coordinate system.",
                kind="warning",
            )
            return
        if self._index_column is None:
            self._set_state_status(
                PointsControllerState.LOAD_FAILED,
                "Points: choose a non-empty index column.",
                kind="error",
            )
            return
        if self._current_value_source is not None:
            self._set_state_status(
                PointsControllerState.VALUES_READY,
                f"Points: values are ready for `{self._points_name}`.",
                kind="success",
            )
            return

        self._set_state_status(
            PointsControllerState.NO_POINTS_ELEMENT,
            f"Points: ready to load values for `{self._points_name}`.",
            kind="info",
        )

    def _set_state_status(
        self,
        state: PointsControllerState,
        message: str,
        *,
        kind: PointsStatusKind,
    ) -> None:
        self._state = state
        self._status_message = message
        self._status_kind = kind
        self._notify_state_changed()

    def _notify_state_changed(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _cancel_value_worker(self) -> None:
        if self._active_value_worker is None:
            return

        quit_worker = getattr(self._active_value_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_value_worker = None
        self._active_value_worker_job_id = None

    def _cancel_load_worker(self) -> None:
        if self._active_load_worker is None:
            return

        quit_worker = getattr(self._active_load_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_load_worker = None
        self._active_load_worker_job_id = None


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _normalize_required_text(value: str, default: str) -> str:
    if not isinstance(value, str):
        return default
    return value.strip() or default
