"""Cache-backed points controller used by the napari-harpy Viewer panel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from napari.utils.events import Event

from napari_harpy.core.multi_scale_cache_points_zarr.cache_location import (
    points_cache_path,
    points_element_path,
)
from napari_harpy.core.multi_scale_cache_points_zarr.reader import _read_cache_dataset_info
from napari_harpy.viewer.adapter import TiledPointsLayerBinding, ViewerAdapter
from napari_harpy.viewer.tiled_points.application import (
    TiledPointsApplicationSettings,
    TiledPointsCacheDescriptor,
)
from napari_harpy.viewer.tiled_points.runtime.cache_session import _CacheSessionState

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - sandboxed import fallback  # noqa: BLE001
        from superqt.utils import thread_worker
    return thread_worker


thread_worker = _resolve_thread_worker()
PointsStatusKind = Literal["info", "warning", "success", "error"]


class TiledPointsControllerState(StrEnum):
    """Identify application-visible cache-backed points state."""

    UNBOUND = "unbound"
    LOADING_DESCRIPTOR = "loading_descriptor"
    DESCRIPTOR_READY = "descriptor_ready"
    LAYER_ACTIVE = "layer_active"
    FAILED = "failed"


@dataclass(frozen=True)
class _CacheDescriptorJob:
    job_id: int
    sdata: SpatialData
    points_name: str
    coordinate_system: str
    value_column: str


@thread_worker(start_thread=False, ignore_errors=True)
def _run_cache_descriptor_job(job: _CacheDescriptorJob) -> TiledPointsCacheDescriptor:
    """Resolve and read only the nested completed cache's semantic descriptor."""
    return _load_cache_descriptor(job)


def _load_cache_descriptor(job: _CacheDescriptorJob) -> TiledPointsCacheDescriptor:
    """Synchronously resolve one descriptor; the worker is its production caller."""
    if not job.sdata.is_backed() or job.sdata.path is None or not isinstance(job.sdata.path, Path):
        raise ValueError("Points visualization requires a SpatialData object backed by a local Zarr store.")
    if job.points_name not in job.sdata.points:
        raise ValueError(f"Points element {job.points_name!r} is not available in the SpatialData object.")
    expected_element_path = points_element_path(job.points_name)
    observed_paths = tuple(job.sdata.locate_element(job.sdata.points[job.points_name]))
    if observed_paths != (expected_element_path,):
        raise ValueError(
            f"Points element {job.points_name!r} must resolve exactly to {expected_element_path!r}; "
            f"observed {observed_paths!r}."
        )
    cache_root = points_cache_path(job.sdata.path, job.points_name)
    info = _read_cache_dataset_info(cache_root)
    if info.points_name != job.points_name:
        raise ValueError(f"Cache {cache_root} describes points element {info.points_name!r}, not {job.points_name!r}.")
    if info.value_column != job.value_column:
        raise ValueError(
            f"Selected value column {job.value_column!r} does not match cache value column "
            f"{info.value_column!r} at {cache_root}."
        )
    return TiledPointsCacheDescriptor(cache_root=cache_root, dataset_info=info)


class TiledPointsController:
    """Bind the existing points panel to persistent cache-backed napari layers."""

    def __init__(
        self,
        viewer_adapter: ViewerAdapter,
        *,
        settings: TiledPointsApplicationSettings = TiledPointsApplicationSettings(),
        on_state_changed: Callable[[], None] | None = None,
        on_values_loaded: Callable[[tuple[str, ...]], None] | None = None,
    ) -> None:
        self._viewer_adapter = viewer_adapter
        self._settings = settings
        self._on_state_changed = on_state_changed
        self._on_values_loaded = on_values_loaded
        self._sdata: SpatialData | None = None
        self._points_name: str | None = None
        self._coordinate_system: str | None = None
        self._value_column: str | None = None
        self._descriptor: TiledPointsCacheDescriptor | None = None
        self._active_binding: TiledPointsLayerBinding | None = None
        self._latest_job_id = 0
        self._active_worker_job_id: int | None = None
        self._active_worker: Any | None = None
        self._state = TiledPointsControllerState.UNBOUND
        self._status_message = "Points: choose a points element and index column."
        self._status_kind: PointsStatusKind = "warning"

    @property
    def state(self) -> TiledPointsControllerState:
        return self._state

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def status_kind(self) -> PointsStatusKind:
        return self._status_kind

    @property
    def is_loading_values(self) -> bool:
        return self._active_worker is not None

    @property
    def is_loading(self) -> bool:
        binding = self._active_binding
        return binding is not None and binding.runtime.state in {
            _CacheSessionState.STARTING,
            _CacheSessionState.LOADING_BUCKET_INDEXES,
            _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX,
        }

    @property
    def can_visualize(self) -> bool:
        return self._descriptor is not None

    @property
    def descriptor(self) -> TiledPointsCacheDescriptor | None:
        return self._descriptor

    def bind_source(
        self,
        sdata: SpatialData | None,
        points_name: str | None,
        coordinate_system: str | None,
        value_column: str | None,
    ) -> bool:
        """Bind a panel source and asynchronously read its small cache descriptor."""
        normalized = tuple(_optional_text(value) for value in (points_name, coordinate_system, value_column))
        points_name, coordinate_system, value_column = normalized
        changed = (
            sdata is not self._sdata
            or points_name != self._points_name
            or coordinate_system != self._coordinate_system
            or value_column != self._value_column
        )
        if not changed:
            return False
        self._cancel_worker()
        self._sdata = sdata
        self._points_name = points_name
        self._coordinate_system = coordinate_system
        self._value_column = value_column
        self._descriptor = None
        self._set_active_binding(None)
        if sdata is None or points_name is None or coordinate_system is None or value_column is None:
            self._set_status(
                TiledPointsControllerState.UNBOUND,
                "Points: choose a SpatialData object, points element, coordinate system, and index column.",
                "warning",
            )
            self._notify_values(())
            return True
        self._latest_job_id += 1
        job = _CacheDescriptorJob(
            self._latest_job_id,
            sdata,
            points_name,
            coordinate_system,
            value_column,
        )
        worker = _run_cache_descriptor_job(job)
        self._active_worker = worker
        self._active_worker_job_id = job.job_id
        worker.returned.connect(partial(self._on_descriptor_returned, job.job_id))
        worker.errored.connect(partial(self._on_descriptor_errored, job.job_id))
        worker.finished.connect(partial(self._on_descriptor_finished, job.job_id))
        self._set_status(
            TiledPointsControllerState.LOADING_DESCRIPTOR,
            f'Points: opening cache metadata for "{points_name}".',
            "info",
        )
        worker.start()
        return True

    def apply_selection(
        self,
        values: Sequence[str] | Literal["all"],
        *,
        render_point_budget: int,
    ) -> bool:
        """Create or update the persistent tiled layer for the current source."""
        descriptor = self._descriptor
        if descriptor is None or self._sdata is None or self._points_name is None or self._coordinate_system is None:
            self._set_status(TiledPointsControllerState.FAILED, "Points: cache metadata is not ready.", "error")
            return False
        if (
            not isinstance(render_point_budget, int)
            or isinstance(render_point_budget, bool)
            or render_point_budget <= 0
        ):
            self._set_status(TiledPointsControllerState.FAILED, "Points: render budget must be positive.", "error")
            return False
        try:
            normalized_values: tuple[str, ...] | str = "all" if values == "all" else tuple(values)
            requested_value_ids = descriptor.requested_value_ids(normalized_values)
            result = self._viewer_adapter.ensure_tiled_points_layer(
                sdata=self._sdata,
                points_name=self._points_name,
                coordinate_system=self._coordinate_system,
                descriptor=descriptor,
                requested_value_ids=requested_value_ids,
                hard_render_point_budget=render_point_budget,
                settings=self._settings,
            )
        except Exception as error:  # noqa: BLE001
            self._set_status(TiledPointsControllerState.FAILED, f"Points: {error}", "error")
            return False
        self._set_active_binding(result.binding)
        self._viewer_adapter.activate_layer(result.binding.layer)
        action = "created" if result.created else "updated"
        self._set_status(
            TiledPointsControllerState.LAYER_ACTIVE,
            f'Points: {action} tiled layer "{self._points_name}"; {result.binding.layer.display_status.message}.',
            "success",
        )
        return True

    def shutdown(self) -> None:
        """Cancel descriptor work and close every tiled runtime owned by the adapter."""
        self._cancel_worker()
        self._viewer_adapter.close_tiled_points_runtimes()

    def _on_descriptor_returned(self, job_id: int, descriptor: TiledPointsCacheDescriptor) -> None:
        if job_id != self._latest_job_id or job_id != self._active_worker_job_id:
            return
        self._descriptor = descriptor
        self._set_status(
            TiledPointsControllerState.DESCRIPTOR_READY,
            f'Points: {len(descriptor.value_names):,} cached values are ready for "{descriptor.dataset_info.points_name}".',
            "success",
        )
        self._notify_values(descriptor.value_names)

    def _on_descriptor_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_job_id or job_id != self._active_worker_job_id:
            return
        self._descriptor = None
        self._notify_values(())
        self._set_status(TiledPointsControllerState.FAILED, f"Points: cache metadata failed: {error}", "error")

    def _on_descriptor_finished(self, job_id: int) -> None:
        if job_id != self._active_worker_job_id:
            return
        self._active_worker = None
        self._active_worker_job_id = None
        self._notify_state()

    def _set_active_binding(self, binding: TiledPointsLayerBinding | None) -> None:
        previous = self._active_binding
        if previous is not None:
            try:
                previous.layer.events.display_status.disconnect(self._on_layer_status)
            except (RuntimeError, TypeError):
                pass
        self._active_binding = binding
        if binding is not None:
            binding.layer.events.display_status.connect(self._on_layer_status)

    def _on_layer_status(self, event: Event) -> None:
        binding = self._active_binding
        if binding is None or event.source is not binding.layer:
            return
        status = binding.layer.display_status
        details = status.message
        if status.level is not None:
            details += (
                f" · {status.level_label} · {status.rendered_point_count:,} points "
                f"in {status.rendered_tile_count:,} tiles"
            )
        descriptor = self._descriptor
        if status.omitted_value_ids and descriptor is not None:
            omitted_names = tuple(descriptor.value_names[value_id] for value_id in status.omitted_value_ids)
            details += f" · omitted selected values: {', '.join(omitted_names)}"
        normalized_message = status.message.casefold()
        if "failed" in normalized_message:
            kind: PointsStatusKind = "error"
        elif status.omitted_value_ids or "exceeds" in normalized_message:
            kind = "warning"
        elif normalized_message.startswith(("opening", "loading", "updating")):
            kind = "info"
        else:
            kind = "success"
        self._set_status(TiledPointsControllerState.LAYER_ACTIVE, f"Points: {details}", kind)

    def _cancel_worker(self) -> None:
        worker = self._active_worker
        if worker is not None:
            quit_worker = getattr(worker, "quit", None)
            if callable(quit_worker):
                quit_worker()
        self._active_worker = None
        self._active_worker_job_id = None

    def _set_status(self, state: TiledPointsControllerState, message: str, kind: PointsStatusKind) -> None:
        self._state = state
        self._status_message = message
        self._status_kind = kind
        self._notify_state()

    def _notify_state(self) -> None:
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_values(self, values: tuple[str, ...]) -> None:
        if self._on_values_loaded is not None:
            self._on_values_loaded(values)


def _optional_text(value: str | None) -> str | None:
    return value.strip() or None if isinstance(value, str) else None
