"""Own one long-lived points-cache reader on a dedicated Qt thread."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from qtpy.QtCore import QObject, QThread, Signal, Slot

from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _CacheDatasetInfo,
    _IntrinsicViewport,
    _PointsCacheReader,
    _SelectedValueIndex,
)
from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderBatch,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
    _ViewportRequest,
)
from napari_harpy.viewer.tiled_points.render_batch import pack_render_tiles
from napari_harpy.viewer.tiled_points.runtime.residency import _CpuTileResidency

_UINT32_MAX = np.iinfo(np.uint32).max
_FailurePhase = Literal[
    "startup", "bucket_index_projection", "bucket_index_loading", "selection", "viewport", "shutdown"
]
_ReaderFactory = Callable[[Path], _PointsCacheReader]


class _CacheSessionState(StrEnum):
    """Identify one GUI-visible cache-session lifecycle state."""

    NEW = "new"
    STARTING = "starting"
    LOADING_BUCKET_INDEXES = "loading_bucket_indexes"
    READY = "ready"
    UPDATING_SELECTED_VALUE_INDEX = "updating_selected_value_index"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class _CacheSessionSettings:
    """Bound metadata and decoded tile payloads retained by one cache session.

    Parameters
    ----------
    max_bucket_lookup_bytes
        Maximum total resident bytes for the five tile/range arrays represented
        by all loaded bucket lookup indexes. ``None`` disables this configured
        preflight limit without disabling byte projection or accounting.
    max_selected_value_index_bytes
        Maximum resident bytes for the current selected-value catalog index.
        ``None`` disables this configured preflight limit.
    max_cpu_tile_bytes
        Positive byte limit for the evicting decoded point-payload LRU.
    max_vertex_payload_bytes
        Positive byte limit for one worker-prepared renderer vertex payload.
    """

    max_bucket_lookup_bytes: int | None
    max_selected_value_index_bytes: int | None
    max_cpu_tile_bytes: int
    max_vertex_payload_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_bucket_lookup_bytes", "max_selected_value_index_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer or None.")
        for name in ("max_cpu_tile_bytes", "max_vertex_payload_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")


@dataclass(frozen=True)
class _CacheSessionFailure:
    """Describe one worker failure without transporting a live traceback."""

    phase: _FailurePhase
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if self.phase not in (
            "startup",
            "bucket_index_projection",
            "bucket_index_loading",
            "selection",
            "viewport",
            "shutdown",
        ):
            raise ValueError("Unsupported cache-session failure phase.")
        for name in ("exception_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"`{name}` must be a nonempty string.")


class _SessionCancelled(RuntimeError):
    """Stop active worker work without reporting cancellation as a failure."""


class _TiledPointsCacheWorker(QObject):
    """Own cache IO and reader state on one dedicated Qt worker thread.

    The GUI-thread ``_TiledPointsCacheSession`` constructs this object, moves it
    to a ``QThread``, and then starts that thread. Reader construction happens in
    :meth:`start`, after the move, so the reader and all opened Zarr resources
    are created, used, and closed on the worker thread. Neither the live reader
    nor the resident selected-value index crosses the GUI facade.

    Communication in both directions uses Qt signals and slots::

        GUI thread                              worker thread
        ----------                              -------------
        session.start()       --------------->  start()
        selection requested  --------------->  update_selected_value_index()
        viewport requested   --------------->  read_viewport_snapshot()
        close requested      --------------->  close()

        session handlers     <---------------  state/progress/result signals
        thread.quit()        <---------------  finished

    These connections are queued according to QObject thread affinity. They
    keep cache operations off the GUI thread while returning only immutable
    descriptions and snapshots, selected value IDs, byte counts, and structured
    failures.

    Notes
    -----
    The session and worker share the thread-safe ``cancellation`` event supplied
    to ``__init__``. Closing needs both this event and a queued ``close()`` slot::

        GUI thread                         worker thread
        ----------                         -------------
        session.close()
          cancellation.set()  ---------->  _require_not_cancelled()
          emit close signal                raises _SessionCancelled
                                                    |
                                                    v
                                            close reader and finish

    A busy worker cannot execute the queued ``close()`` slot until its current
    slot returns. Cancellation checkpoints let that active operation observe the
    event, raise ``_SessionCancelled``, and enter cleanup without publishing a
    late result. This is cooperative cancellation: it stops between operations
    or bucket reads, but it cannot interrupt a Zarr call already in progress.

    Conversely, an idle worker has no active checkpoint at which to observe the
    passive event. The queued ``close()`` slot therefore remains necessary to
    execute cleanup on the reader's owning thread. Active cancellation and idle
    closure both converge on idempotent :meth:`_shutdown`, which closes the
    reader once, clears worker-resident state, and emits ``finished`` so the
    session can terminate the thread and publish ``CLOSED``.
    """

    state_changed = Signal(object)
    dataset_available = Signal(object)
    bucket_index_progress = Signal(int, int)
    ready = Signal(int, int)
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    finished = Signal()

    def __init__(
        self,
        cache_root: Path,
        settings: _CacheSessionSettings,
        cancellation: threading.Event,
        reader_factory: _ReaderFactory,
    ) -> None:
        super().__init__()
        self._cache_root = cache_root
        self._settings = settings
        self._cancellation = cancellation
        self._reader_factory = reader_factory
        self._reader: _PointsCacheReader | None = None
        self._selected_value_ids: tuple[int, ...] | None = None
        self._selected_value_index: _SelectedValueIndex | None = None
        self._cpu_tile_residency = _CpuTileResidency(settings.max_cpu_tile_bytes)
        self._finished = False

    @Slot()
    def start(self) -> None:
        """Enter and prime the reader before announcing readiness."""
        phase: _FailurePhase = "startup"
        try:
            self._require_not_cancelled()
            # Construct the reader here so it and all opened Zarr resources are
            # owned by the worker thread.
            reader = self._reader_factory(self._cache_root)
            self._require_not_cancelled()
            reader.__enter__()
            self._reader = reader
            self.dataset_available.emit(reader.dataset_info)

            self._require_not_cancelled()
            phase = "bucket_index_projection"
            projected_bytes = reader.project_bucket_lookup_index_bytes()
            max_lookup_bytes = self._settings.max_bucket_lookup_bytes
            if max_lookup_bytes is not None and projected_bytes > max_lookup_bytes:
                raise ValueError(
                    f"Bucket lookup indexes require {projected_bytes} resident bytes, exceeding "
                    f"max_bucket_lookup_bytes={max_lookup_bytes}."
                )

            self._require_not_cancelled()
            phase = "bucket_index_loading"
            self.state_changed.emit(_CacheSessionState.LOADING_BUCKET_INDEXES)
            # Omitting `levels` and `bucket_keys` loads every serialized bucket
            # across all levels. Each retains five point-row lookup arrays:
            # `tile_offset` and `ranges/{tile_indptr,value_id,row_start,row_count}`. See
            # `storage.bucket_reader._BucketLookupIndex` for the complete contract.
            # Point coordinates and point-level values remain on disk.
            resident_bucket_index_bytes = reader.load_bucket_lookup_indexes(
                max_resident_bytes=max_lookup_bytes,
                progress=self._on_bucket_index_progress,
            )
            self._require_not_cancelled()
            self.state_changed.emit(_CacheSessionState.READY)
            self.ready.emit(projected_bytes, resident_bucket_index_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed during {}.", phase)
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception(phase, error))
            self._shutdown(emit_closing=False)

    @Slot(object)
    def update_selected_value_index(self, requested_value_ids: tuple[int, ...] | None) -> None:
        """Update the worker-resident selected-value index for a new selection.

        This operation does not necessarily load data. An unchanged normalized
        selection reuses the current index, while the all-values selection
        (``None``) clears that index. Only a changed proper subset loads a new
        selected-value index from the catalog. The replacement is committed
        only after loading succeeds, so a recoverable failure leaves the
        previous selection active.
        """
        if self._finished:
            return
        reader = self._reader
        if reader is None:
            error = RuntimeError("Cache reader is not ready.")
            logger.error("Tiled-points worker received a selection request without a live cache reader.")
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception("selection", error))
            self._shutdown(emit_closing=False)
            return
        try:
            self._require_not_cancelled()
            requested_value_ids = _normalize_all_values(
                requested_value_ids,
                value_count=len(reader.value_names),
            )
            if requested_value_ids == self._selected_value_ids:
                resident_bytes = 0 if self._selected_value_index is None else self._selected_value_index.resident_bytes
                self.state_changed.emit(_CacheSessionState.READY)
                self.value_selection_ready.emit(requested_value_ids, resident_bytes)
                return

            if requested_value_ids is None:
                value_index = None
            else:
                value_index = reader.load_selected_value_index(
                    np.asarray(requested_value_ids, dtype=np.uint32),
                    max_resident_bytes=self._settings.max_selected_value_index_bytes,
                )
                if value_index is None:
                    requested_value_ids = None

            self._require_not_cancelled()
            self._selected_value_ids = requested_value_ids
            self._selected_value_index = value_index
            resident_bytes = 0 if value_index is None else value_index.resident_bytes
            self.state_changed.emit(_CacheSessionState.READY)
            self.value_selection_ready.emit(requested_value_ids, resident_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while loading a selected-value index.")
            self._report_recoverable_selection_failure(error)

    @Slot()
    def close(self) -> None:
        """Close the reader and finish this worker exactly once."""
        self._shutdown(emit_closing=True)

    @Slot(object)
    def read_viewport_snapshot(self, request: _ViewportRequest) -> None:
        """Build one generation-bound snapshot with resident or newly read tiles."""
        if self._finished:
            return
        reader = self._reader
        if reader is None:
            self._report_viewport_failure(request, RuntimeError("Cache reader is not ready."))
            return
        try:
            self._require_not_cancelled()
            if not isinstance(request, _ViewportRequest):
                raise ValueError("`request` must be _ViewportRequest.")
            if request.requested_value_ids != self._selected_value_ids:
                raise ValueError("Viewport request value IDs do not match the worker's committed selection.")
            snapshot = _read_viewport_snapshot(
                reader,
                self._selected_value_index,
                self._cpu_tile_residency,
                request,
                max_vertex_payload_bytes=self._settings.max_vertex_payload_bytes,
                check_cancelled=self._require_not_cancelled,
            )
            self._require_not_cancelled()
            self.viewport_ready.emit(snapshot)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while reading a viewport snapshot.")
            self._report_viewport_failure(request, error)

    def _on_bucket_index_progress(self, completed_buckets: int, total_buckets: int) -> None:
        self._require_not_cancelled()
        self.bucket_index_progress.emit(completed_buckets, total_buckets)

    def _require_not_cancelled(self) -> None:
        """Stop active work when the GUI has requested session closure.

        The thread-safe event can be observed while the worker's queued
        ``close()`` slot is still waiting for the current slot to return. This
        lets long-running work stop at its next checkpoint; reader cleanup
        remains on the worker thread.
        """
        if self._cancellation.is_set():
            raise _SessionCancelled

    def _report_recoverable_selection_failure(self, error: Exception) -> None:
        self.failed.emit(_failure_from_exception("selection", error))
        self.state_changed.emit(_CacheSessionState.READY)

    def _report_viewport_failure(self, request: object, error: Exception) -> None:
        request_generation = request.request_generation if isinstance(request, _ViewportRequest) else 0
        self.viewport_failed.emit(request_generation, _failure_from_exception("viewport", error))

    def _shutdown(self, *, emit_closing: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if emit_closing:
            self.state_changed.emit(_CacheSessionState.CLOSING)

        reader = self._reader
        self._reader = None
        self._selected_value_ids = None
        self._selected_value_index = None
        self._cpu_tile_residency.clear()
        try:
            if reader is not None:
                reader.__exit__(None, None, None)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while closing the reader.")
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception("shutdown", error))
        finally:
            self.finished.emit()


class _TiledPointsCacheSession(QObject):
    """Expose one worker-owned cache reader to the GUI thread.

    Construction is passive. :meth:`start` creates the dedicated reader thread;
    :meth:`close` is terminal and idempotent. The reader and selected-value
    index never cross this facade.
    """

    state_changed = Signal(object)
    dataset_available = Signal(object)
    bucket_index_progress = Signal(int, int)
    ready = Signal()
    value_selection_ready = Signal(object, int)
    viewport_ready = Signal(object)
    viewport_failed = Signal(int, object)
    failed = Signal(object)
    closed = Signal()

    # Queue a value-ID selection change on the worker; `value_selection_ready`
    # announces the committed result back to the GUI thread.
    _value_selection_change_requested = Signal(object)
    _viewport_requested = Signal(object)
    _close_requested = Signal()

    def __init__(
        self,
        cache_root: str | Path,
        settings: _CacheSessionSettings,
        *,
        reader_factory: _ReaderFactory = _PointsCacheReader,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(settings, _CacheSessionSettings):
            raise ValueError("`settings` must be _CacheSessionSettings.")
        if not callable(reader_factory):
            raise ValueError("`reader_factory` must be callable.")
        self._cache_root = Path(cache_root)
        self._settings = settings
        self._reader_factory = reader_factory
        self._state = _CacheSessionState.NEW
        self._dataset_info: _CacheDatasetInfo | None = None
        self._selected_value_ids: tuple[int, ...] | None = None
        self._projected_lookup_bytes: int | None = None
        self._resident_lookup_bytes: int | None = None
        self._cancellation = threading.Event()
        self._thread: QThread | None = None
        self._worker: _TiledPointsCacheWorker | None = None

    @property
    def state(self) -> _CacheSessionState:
        """Return the current GUI-side session state."""
        return self._state

    @property
    def dataset_info(self) -> _CacheDatasetInfo | None:
        """Return immutable opened-cache information when available."""
        return self._dataset_info

    @property
    def selected_value_ids(self) -> tuple[int, ...] | None:
        """Return the successfully applied value IDs; ``None`` means all values."""
        return self._selected_value_ids

    @property
    def projected_lookup_bytes(self) -> int | None:
        """Return the complete projected lookup footprint after startup."""
        return self._projected_lookup_bytes

    @property
    def resident_lookup_bytes(self) -> int | None:
        """Return the complete resident lookup footprint after startup."""
        return self._resident_lookup_bytes

    def start(self) -> None:
        """Create the worker thread and begin guarded cache startup."""
        if self._state is not _CacheSessionState.NEW:
            raise RuntimeError("A cache session can be started only from NEW.")

        thread = QThread(self)
        worker = _TiledPointsCacheWorker(
            self._cache_root,
            self._settings,
            self._cancellation,
            self._reader_factory,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.start)
        self._value_selection_change_requested.connect(worker.update_selected_value_index)
        self._viewport_requested.connect(worker.read_viewport_snapshot)
        self._close_requested.connect(worker.close)
        worker.state_changed.connect(self._on_worker_state_changed)
        worker.dataset_available.connect(self._on_dataset_available)
        worker.bucket_index_progress.connect(self._on_bucket_index_progress)
        worker.ready.connect(self._on_ready)
        worker.value_selection_ready.connect(self._on_value_selection_ready)
        worker.viewport_ready.connect(self._on_viewport_ready)
        worker.viewport_failed.connect(self._on_viewport_failed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._set_state(_CacheSessionState.STARTING)
        thread.start()

    def set_selected_value_ids(self, requested_value_ids: tuple[int, ...] | None) -> bool:
        """Queue one selected-value index replacement from the ready state.

        ``None`` selects all canonical values. A tuple represents one sorted,
        unique, nonempty subset of canonical value IDs.
        """
        if self._state is not _CacheSessionState.READY:
            raise RuntimeError("Value selection can change only while the cache session is READY.")
        requested_value_ids = _require_requested_value_ids(requested_value_ids)
        if requested_value_ids == self._selected_value_ids:
            return False
        self._set_state(_CacheSessionState.UPDATING_SELECTED_VALUE_INDEX)
        self._value_selection_change_requested.emit(requested_value_ids)
        return True

    def request_viewport(self, request: _ViewportRequest) -> None:
        """Queue one coordinator-stamped viewport request on the reader worker."""
        if self._state is not _CacheSessionState.READY:
            raise RuntimeError("A viewport can be requested only while the cache session is READY.")
        if not isinstance(request, _ViewportRequest):
            raise ValueError("`request` must be _ViewportRequest.")
        if request.requested_value_ids != self._selected_value_ids:
            raise ValueError("Viewport request value IDs do not match the committed session selection.")
        self._viewport_requested.emit(request)

    def close(self) -> bool:
        """Request terminal owner-thread closure exactly once."""
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return False
        if self._state is _CacheSessionState.NEW:
            self._set_state(_CacheSessionState.CLOSING)
            self._set_state(_CacheSessionState.CLOSED)
            self.closed.emit()
            return True

        # The event stops active work; the queued slot closes an idle worker on
        # its owning thread. Both paths converge on idempotent worker shutdown.
        self._cancellation.set()
        self._set_state(_CacheSessionState.CLOSING)
        self._close_requested.emit()
        return True

    @Slot(object)
    def _on_worker_state_changed(self, state: _CacheSessionState) -> None:
        if self._state is _CacheSessionState.CLOSED:
            return
        if self._state is _CacheSessionState.CLOSING and state is not _CacheSessionState.FAILED:
            return
        self._set_state(state)

    @Slot(object)
    def _on_dataset_available(self, dataset_info: _CacheDatasetInfo) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._dataset_info = dataset_info
        self.dataset_available.emit(dataset_info)

    @Slot(int, int)
    def _on_bucket_index_progress(self, completed_buckets: int, total_buckets: int) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self.bucket_index_progress.emit(completed_buckets, total_buckets)

    @Slot(int, int)
    def _on_ready(self, projected_lookup_bytes: int, resident_lookup_bytes: int) -> None:
        if self._state is not _CacheSessionState.READY:
            return
        self._projected_lookup_bytes = projected_lookup_bytes
        self._resident_lookup_bytes = resident_lookup_bytes
        self.ready.emit()

    @Slot(object, int)
    def _on_value_selection_ready(self, selected_value_ids: tuple[int, ...] | None, resident_bytes: int) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._selected_value_ids = selected_value_ids
        self.value_selection_ready.emit(selected_value_ids, resident_bytes)

    @Slot(object)
    def _on_viewport_ready(self, snapshot: TiledPointsRenderSnapshot) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self.viewport_ready.emit(snapshot)

    @Slot(int, object)
    def _on_viewport_failed(self, request_generation: int, failure: _CacheSessionFailure) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self.failed.emit(failure)
        self.viewport_failed.emit(request_generation, failure)

    @Slot(object)
    def _on_failed(self, failure: _CacheSessionFailure) -> None:
        if self._state is _CacheSessionState.CLOSED:
            return
        self.failed.emit(failure)

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._set_state(_CacheSessionState.CLOSED)
        self.closed.emit()

    def _set_state(self, state: _CacheSessionState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state)


def _read_viewport_snapshot(
    reader: _PointsCacheReader,
    selected_value_index: _SelectedValueIndex | None,
    residency: _CpuTileResidency,
    request: _ViewportRequest,
    *,
    max_vertex_payload_bytes: int,
    check_cancelled: Callable[[], None],
) -> TiledPointsRenderSnapshot:
    """Plan one viewport and assemble its complete immutable render snapshot.

    Reuse CPU-resident tiles, read only residency misses, and restore the
    complete tile plan's spatial order before returning. If no serialized level
    satisfies the runtime budget, return a metadata-only over-budget snapshot
    without reading point payloads.
    """
    viewport = _IntrinsicViewport(
        request.viewport.x_min,
        request.viewport.y_min,
        request.viewport.x_max,
        request.viewport.y_max,
    )
    level_selection = reader.select_level(
        viewport,
        request.viewport.effective_point_budget,
        value_index=selected_value_index,
    )
    check_cancelled()
    dataset_info = reader.dataset_info
    level_kind = _level_kind(level_selection.level, dataset_info.levels[level_selection.level].kind)
    omitted_value_ids = (
        ()
        if level_selection.omitted_value_ids is None
        else tuple(int(value_id) for value_id in level_selection.omitted_value_ids)
    )
    if not level_selection.within_budget:
        return TiledPointsRenderSnapshot(
            cache_generation_id=dataset_info.cache_generation_id,
            request_generation=request.request_generation,
            selection_generation=request.selection_generation,
            requested_value_ids=request.requested_value_ids,
            level=level_selection.level,
            level_kind=level_kind,
            within_budget=False,
            estimated_point_count=level_selection.estimated_point_count,
            omitted_value_ids=omitted_value_ids,
            rendered_tile_count=0,
            render_batch=TiledPointsRenderBatch.empty(),
        )

    plan = reader.plan_viewport(level_selection.level, viewport, value_index=selected_value_index)
    check_cancelled()
    if plan.requested_value_ids != request.requested_value_ids:
        raise RuntimeError("Viewport plan selection differs from its generation-bound request.")
    keys = tuple(
        TileResidencyKey(
            cache_generation_id=dataset_info.cache_generation_id,
            requested_value_ids=request.requested_value_ids,
            level=level,
            tile_x=tile_x,
            tile_y=tile_y,
        )
        for level, tile_x, tile_y in plan.tile_keys
    )
    payloads_by_key: dict[TileResidencyKey, TiledPointsRenderTile] = {}
    missing_keys: list[TileResidencyKey] = []
    for key in keys:
        tile = residency.get(key)
        if tile is None:
            missing_keys.append(key)
        else:
            payloads_by_key[key] = tile
    resident_keys = tuple(payloads_by_key)

    new_tiles: tuple[TiledPointsRenderTile, ...] = ()
    if missing_keys:
        result = reader.read_planned_tiles(
            plan,
            tuple(key.logical_tile_key for key in missing_keys),
        )
        check_cancelled()
        key_by_logical_tile = {key.logical_tile_key: key for key in missing_keys}
        # Bucket reads expose per-tile views into shared batch allocations.
        # Copy once at the viewer-residency boundary so every render tile owns
        # exactly the point-array bytes accounted by `_CpuTileResidency`.
        new_tiles = tuple(
            TiledPointsRenderTile(
                key=key_by_logical_tile[(tile.level, tile.tile_x, tile.tile_y)],
                tile_size=tile.tile_size,
                location=tile.location.copy(order="C"),
                value_id=tile.value_id.copy(order="C"),
            )
            for tile in result.tiles
        )
        if {tile.key for tile in new_tiles} != set(missing_keys):
            raise RuntimeError("Viewport subset read did not return every requested nonresident tile.")
        payloads_by_key.update((tile.key, tile) for tile in new_tiles)

    ordered_tiles = tuple(payloads_by_key[key] for key in keys)
    _require_ordered_render_tiles(
        ordered_tiles,
        cache_generation_id=dataset_info.cache_generation_id,
        requested_value_ids=request.requested_value_ids,
        level=level_selection.level,
    )
    if new_tiles:
        residency.retain(new_tiles, protected_keys=resident_keys)
    render_batch = pack_render_tiles(
        ordered_tiles,
        point_count=level_selection.estimated_point_count,
        value_count=len(dataset_info.value_names),
        max_vertex_payload_bytes=max_vertex_payload_bytes,
        check_cancelled=check_cancelled,
    )
    check_cancelled()
    return TiledPointsRenderSnapshot(
        cache_generation_id=dataset_info.cache_generation_id,
        request_generation=request.request_generation,
        selection_generation=request.selection_generation,
        requested_value_ids=request.requested_value_ids,
        level=level_selection.level,
        level_kind=level_kind,
        within_budget=True,
        estimated_point_count=level_selection.estimated_point_count,
        omitted_value_ids=omitted_value_ids,
        rendered_tile_count=len(ordered_tiles),
        render_batch=render_batch,
    )


def _require_ordered_render_tiles(
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    cache_generation_id: str,
    requested_value_ids: tuple[int, ...] | None,
    level: int,
) -> None:
    """Validate the complete worker-local tile order before packing it."""
    keys = tuple(tile.key for tile in tiles)
    if len(set(keys)) != len(keys):
        raise RuntimeError("Snapshot tile residency keys must be unique.")
    coordinates = tuple((key.tile_y, key.tile_x) for key in keys)
    if coordinates != tuple(sorted(coordinates)):
        raise RuntimeError("Snapshot tiles must follow spatial (tile_y, tile_x) order.")
    if any(
        key.cache_generation_id != cache_generation_id
        or key.requested_value_ids != requested_value_ids
        or key.level != level
        for key in keys
    ):
        raise RuntimeError("Every snapshot tile must match its cache, selection, and level.")


def _level_kind(level: int, serialized_kind: str) -> Literal["exact", "bridge", "spatial"]:
    expected: Literal["exact", "bridge", "spatial"] = "exact" if level == 0 else "bridge" if level == 1 else "spatial"
    if serialized_kind != expected:
        raise RuntimeError("Serialized cache level kind is inconsistent with its level index.")
    return expected


def _require_requested_value_ids(requested_value_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if requested_value_ids is None:
        return None
    if (
        not isinstance(requested_value_ids, tuple)
        or not requested_value_ids
        or any(
            not isinstance(value_id, int) or isinstance(value_id, bool) or not 0 <= value_id <= _UINT32_MAX
            for value_id in requested_value_ids
        )
        or tuple(sorted(set(requested_value_ids))) != requested_value_ids
    ):
        raise ValueError("`requested_value_ids` must be None or sorted unique nonnegative uint32 integers.")
    return requested_value_ids


def _normalize_all_values(value_ids: tuple[int, ...] | None, *, value_count: int) -> tuple[int, ...] | None:
    if value_ids == tuple(range(value_count)):
        return None
    return value_ids


def _failure_from_exception(phase: _FailurePhase, error: Exception) -> _CacheSessionFailure:
    exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return _CacheSessionFailure(
        phase=phase,
        exception_type=exception_type,
        message=str(error) or repr(error),
    )
