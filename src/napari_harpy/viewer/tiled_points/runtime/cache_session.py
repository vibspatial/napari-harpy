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
    _PointsCacheReader,
    _SelectedValueIndex,
)

_UINT32_MAX = np.iinfo(np.uint32).max
_FailurePhase = Literal["startup", "bucket_index_projection", "bucket_index_loading", "selection", "shutdown"]
_ReaderFactory = Callable[[Path], _PointsCacheReader]


class _CacheSessionState(StrEnum):
    """Identify one GUI-visible cache-session lifecycle state."""

    NEW = "new"
    STARTING = "starting"
    LOADING_BUCKET_INDEXES = "loading_bucket_indexes"
    READY = "ready"
    LOADING_SELECTION = "loading_selection"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class _CacheSessionSettings:
    """Bound lookup metadata retained by one cache session.

    Parameters
    ----------
    max_bucket_lookup_bytes
        Maximum total resident bytes for the five tile/range arrays represented
        by all loaded bucket lookup indexes. ``None`` disables this configured
        preflight limit without disabling byte projection or accounting.
    max_selected_value_index_bytes
        Maximum resident bytes for the current selected-value catalog index.
        ``None`` disables this configured preflight limit.
    """

    max_bucket_lookup_bytes: int | None
    max_selected_value_index_bytes: int | None

    def __post_init__(self) -> None:
        for name in ("max_bucket_lookup_bytes", "max_selected_value_index_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer or None.")


@dataclass(frozen=True)
class _CacheSessionFailure:
    """Describe one worker failure without transporting a live traceback."""

    phase: _FailurePhase
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if self.phase not in ("startup", "bucket_index_projection", "bucket_index_loading", "selection", "shutdown"):
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
        selection requested  --------------->  load_selection()
        close requested      --------------->  close()

        session handlers     <---------------  state/progress/result signals
        thread.quit()        <---------------  finished

    These connections are queued according to QObject thread affinity. They
    keep cache operations off the GUI thread while returning only immutable
    descriptions, selection identities, byte counts, and structured failures.

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
    selection_ready = Signal(object, int)
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
        self._selection_identity: tuple[int, ...] | None = None
        self._selected_value_index: _SelectedValueIndex | None = None
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
            resident_bytes = reader.load_bucket_lookup_indexes(
                max_resident_bytes=max_lookup_bytes,
                progress=self._on_bucket_index_progress,
            )
            self._require_not_cancelled()
            self.state_changed.emit(_CacheSessionState.READY)
            self.ready.emit(projected_bytes, resident_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed during {}.", phase)
            self.state_changed.emit(_CacheSessionState.FAILED)
            self.failed.emit(_failure_from_exception(phase, error))
            self._shutdown(emit_closing=False)

    @Slot(object)
    def load_selection(self, requested_value_ids: tuple[int, ...] | None) -> None:
        """Replace the worker-resident selected-value index."""
        if self._finished:
            return
        reader = self._reader
        if reader is None:
            self._report_recoverable_selection_failure(RuntimeError("Cache reader is not ready."))
            return
        try:
            self._require_not_cancelled()
            selection_identity = _normalize_all_values(requested_value_ids, value_count=len(reader.value_names))
            if selection_identity == self._selection_identity:
                resident_bytes = 0 if self._selected_value_index is None else self._selected_value_index.resident_bytes
                self.state_changed.emit(_CacheSessionState.READY)
                self.selection_ready.emit(selection_identity, resident_bytes)
                return

            if selection_identity is None:
                value_index = None
            else:
                value_index = reader.load_selected_value_index(
                    np.asarray(selection_identity, dtype=np.uint32),
                    max_resident_bytes=self._settings.max_selected_value_index_bytes,
                )
                if value_index is None:
                    selection_identity = None

            self._require_not_cancelled()
            self._selection_identity = selection_identity
            self._selected_value_index = value_index
            resident_bytes = 0 if value_index is None else value_index.resident_bytes
            self.state_changed.emit(_CacheSessionState.READY)
            self.selection_ready.emit(selection_identity, resident_bytes)
        except _SessionCancelled:
            self._shutdown(emit_closing=True)
        except Exception as error:  # noqa: BLE001
            logger.exception("Tiled-points cache session failed while loading a selected-value index.")
            self._report_recoverable_selection_failure(error)

    @Slot()
    def close(self) -> None:
        """Close the reader and finish this worker exactly once."""
        self._shutdown(emit_closing=True)

    def _on_bucket_index_progress(self, completed_buckets: int, total_buckets: int) -> None:
        self._require_not_cancelled()
        self.bucket_index_progress.emit(completed_buckets, total_buckets)

    def _require_not_cancelled(self) -> None:
        """Stop the active worker operation after a GUI-side close request."""
        if self._cancellation.is_set():
            raise _SessionCancelled

    def _report_recoverable_selection_failure(self, error: Exception) -> None:
        self.failed.emit(_failure_from_exception("selection", error))
        self.state_changed.emit(_CacheSessionState.READY)

    def _shutdown(self, *, emit_closing: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if emit_closing:
            self.state_changed.emit(_CacheSessionState.CLOSING)

        reader = self._reader
        self._reader = None
        self._selection_identity = None
        self._selected_value_index = None
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
    selection_ready = Signal(object, int)
    failed = Signal(object)
    closed = Signal()

    _selection_requested = Signal(object)
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
        self._selection_identity: tuple[int, ...] | None = None
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
    def selection_identity(self) -> tuple[int, ...] | None:
        """Return the current canonical selection; ``None`` means all values."""
        return self._selection_identity

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
        self._selection_requested.connect(worker.load_selection)
        self._close_requested.connect(worker.close)
        worker.state_changed.connect(self._on_worker_state_changed)
        worker.dataset_available.connect(self._on_dataset_available)
        worker.bucket_index_progress.connect(self._on_bucket_index_progress)
        worker.ready.connect(self._on_ready)
        worker.selection_ready.connect(self._on_selection_ready)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        self._set_state(_CacheSessionState.STARTING)
        thread.start()

    def set_selected_value_ids(self, value_ids: tuple[int, ...] | None) -> bool:
        """Queue one selected-value index replacement from the ready state.

        ``None`` selects all canonical values. A tuple represents one sorted,
        unique, nonempty subset of canonical value IDs.
        """
        if self._state is not _CacheSessionState.READY:
            raise RuntimeError("Value selection can change only while the cache session is READY.")
        selection_identity = _require_selection_identity(value_ids)
        if selection_identity == self._selection_identity:
            return False
        self._set_state(_CacheSessionState.LOADING_SELECTION)
        self._selection_requested.emit(selection_identity)
        return True

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
    def _on_selection_ready(self, selection_identity: tuple[int, ...] | None, resident_bytes: int) -> None:
        if self._state in (_CacheSessionState.CLOSING, _CacheSessionState.CLOSED):
            return
        self._selection_identity = selection_identity
        self.selection_ready.emit(selection_identity, resident_bytes)

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


def _require_selection_identity(value_ids: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if value_ids is None:
        return None
    if (
        not isinstance(value_ids, tuple)
        or not value_ids
        or any(
            not isinstance(value_id, int) or isinstance(value_id, bool) or not 0 <= value_id <= _UINT32_MAX
            for value_id in value_ids
        )
        or tuple(sorted(set(value_ids))) != value_ids
    ):
        raise ValueError("`value_ids` must be None or sorted unique nonnegative uint32 integers.")
    return value_ids


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
