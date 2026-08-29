from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import napari_harpy.viewer.tiled_points.runtime.cache_session as cache_session_module
from napari_harpy.core.multi_scale_cache_points_zarr.reader import (
    _LevelSelection,
    _PlannedTileRead,
    _TileReadResult,
    _ViewportReadPlan,
    _ViewportReadResult,
)
from napari_harpy.viewer.tiled_points.contracts import TiledPointsViewportState, _ViewportRequest
from napari_harpy.viewer.tiled_points.runtime.cache_session import (
    _CacheSessionFailure,
    _CacheSessionSettings,
    _CacheSessionState,
    _TiledPointsCacheSession,
    _TiledPointsCacheWorker,
)


@dataclass
class _ReaderProbe:
    projected_bytes: int = 64
    resident_bytes: int = 64
    bucket_count: int = 3
    operations: list[tuple[str, int]] = field(default_factory=list)
    selection_calls: list[tuple[int, ...]] = field(default_factory=list)
    bucket_lookup_limits: list[int | None] = field(default_factory=list)
    selected_value_limits: list[int | None] = field(default_factory=list)
    fail_selection: bool = False
    pause_bucket_index_loading: bool = False
    bucket_index_loading_paused: threading.Event = field(default_factory=threading.Event)
    resume_bucket_index_loading: threading.Event = field(default_factory=threading.Event)
    pause_selection: bool = False
    selection_paused: threading.Event = field(default_factory=threading.Event)
    resume_selection: threading.Event = field(default_factory=threading.Event)
    pause_construction: bool = False
    construction_paused: threading.Event = field(default_factory=threading.Event)
    resume_construction: threading.Event = field(default_factory=threading.Event)
    planned_tile_x: tuple[int, ...] = (0, 1)
    over_budget: bool = False
    viewport_reads: list[tuple[tuple[int, int, int], ...]] = field(default_factory=list)
    last_location_batch: np.ndarray | None = None
    last_value_id_batch: np.ndarray | None = None

    def record(self, operation: str) -> None:
        self.operations.append((operation, threading.get_ident()))


@dataclass(frozen=True)
class _FakeSelectedValueIndex:
    resident_bytes: int
    value_ids: np.ndarray


class _ControllableReader:
    value_names = ("A", "B", "C")

    def __init__(self, cache_root: Path, probe: _ReaderProbe) -> None:
        del cache_root
        self._probe = probe
        self.dataset_info = _FakeDatasetInfo()
        probe.record("construct")
        if probe.pause_construction:
            probe.construction_paused.set()
            assert probe.resume_construction.wait(timeout=5)

    def __enter__(self) -> _ControllableReader:
        self._probe.record("enter")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self._probe.record("exit")
        return False

    def project_bucket_lookup_index_bytes(self) -> int:
        self._probe.record("project")
        return self._probe.projected_bytes

    def load_bucket_lookup_indexes(
        self,
        *,
        max_resident_bytes: int | None,
        progress: Callable[[int, int], None],
    ) -> int:
        self._probe.record("load_bucket_indexes")
        self._probe.bucket_lookup_limits.append(max_resident_bytes)
        assert max_resident_bytes is None or max_resident_bytes >= self._probe.resident_bytes
        for completed in range(1, self._probe.bucket_count + 1):
            progress(completed, self._probe.bucket_count)
            if completed == 1 and self._probe.pause_bucket_index_loading:
                self._probe.bucket_index_loading_paused.set()
                assert self._probe.resume_bucket_index_loading.wait(timeout=5)
        return self._probe.resident_bytes

    def load_selected_value_index(
        self,
        value_ids: np.ndarray,
        *,
        max_resident_bytes: int | None,
    ) -> _FakeSelectedValueIndex:
        self._probe.record("load_selection")
        self._probe.selected_value_limits.append(max_resident_bytes)
        selection = tuple(int(value_id) for value_id in value_ids)
        self._probe.selection_calls.append(selection)
        if self._probe.pause_selection:
            self._probe.selection_paused.set()
            assert self._probe.resume_selection.wait(timeout=5)
        if self._probe.fail_selection:
            raise ValueError("selection does not fit")
        return _FakeSelectedValueIndex(
            24 if max_resident_bytes is None else min(max_resident_bytes, 24),
            value_ids.copy(),
        )

    def select_level(self, viewport: object, point_budget: int, *, value_index: object) -> _LevelSelection:
        del viewport, point_budget, value_index
        point_count = len(self._probe.planned_tile_x)
        return _LevelSelection(
            level=0,
            estimated_point_count=point_count,
            positive_visible_tile_count=point_count,
            within_budget=not self._probe.over_budget,
            omitted_value_ids=None,
        )

    def plan_viewport(self, level: int, viewport: object, *, value_index: object) -> _ViewportReadPlan:
        del viewport
        requested_value_ids = (
            None if value_index is None else tuple(int(value_id) for value_id in value_index.value_ids)
        )
        return _ViewportReadPlan(
            cache_generation_id=_GENERATION_ID,
            requested_value_ids=requested_value_ids,
            level=level,
            requests=tuple(
                _PlannedTileRead(level, tile_x, 0, tile_x, 0, None if value_index is None else value_index.value_ids)
                for tile_x in self._probe.planned_tile_x
            ),
        )

    def read_planned_tiles(
        self,
        plan: _ViewportReadPlan,
        tile_keys_to_read: tuple[tuple[int, int, int], ...],
    ) -> _ViewportReadResult:
        self._probe.record("read_viewport")
        self._probe.viewport_reads.append(tile_keys_to_read)
        location_batch = np.asarray(
            [[float(tile_x), float(tile_y)] for _, tile_x, tile_y in tile_keys_to_read],
            dtype=np.float32,
        )
        value_id_batch = np.zeros(len(tile_keys_to_read), dtype=np.uint32)
        self._probe.last_location_batch = location_batch
        self._probe.last_value_id_batch = value_id_batch
        return _ViewportReadResult(
            level=plan.level,
            tiles=tuple(
                _TileReadResult(
                    level=level,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_size=10,
                    location=location_batch[index : index + 1, :],
                    value_id=value_id_batch[index : index + 1],
                )
                for index, (level, tile_x, tile_y) in enumerate(tile_keys_to_read)
            ),
        )


_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


@dataclass(frozen=True)
class _FakeLevelInfo:
    kind: str = "exact"


@dataclass(frozen=True)
class _FakeDatasetInfo:
    cache_generation_id: str = _GENERATION_ID
    value_names: tuple[str, ...] = ("A", "B", "C")
    levels: tuple[_FakeLevelInfo, ...] = (_FakeLevelInfo(),)


def _session(
    probe: _ReaderProbe,
    *,
    max_bucket_lookup_bytes: int | None = 1_000,
    max_selected_value_index_bytes: int | None = 1_000,
    max_cpu_tile_bytes: int = 1_000,
    max_vertex_payload_bytes: int = 1_000_000,
) -> _TiledPointsCacheSession:
    return _TiledPointsCacheSession(
        Path("unused.zarr"),
        _CacheSessionSettings(
            max_bucket_lookup_bytes=max_bucket_lookup_bytes,
            max_selected_value_index_bytes=max_selected_value_index_bytes,
            max_cpu_tile_bytes=max_cpu_tile_bytes,
            max_vertex_payload_bytes=max_vertex_payload_bytes,
        ),
        reader_factory=lambda cache_root: _ControllableReader(cache_root, probe),
    )


def _start_ready(session: _TiledPointsCacheSession, qtbot) -> None:
    with qtbot.waitSignal(session.ready, timeout=5_000):
        session.start()
    assert session.state is _CacheSessionState.READY


def _close(session: _TiledPointsCacheSession, qtbot) -> None:
    if session.state is _CacheSessionState.CLOSED:
        return
    with qtbot.waitSignal(session.closed, timeout=5_000):
        session.close()
    assert session.state is _CacheSessionState.CLOSED


def _viewport_request(request_generation: int) -> _ViewportRequest:
    return _ViewportRequest(
        request_generation=request_generation,
        selection_generation=0,
        requested_value_ids=None,
        viewport=TiledPointsViewportState(
            displayed_axes=(0, 1),
            x_min=0.0,
            y_min=0.0,
            x_max=30.0,
            y_max=10.0,
            canvas_width=100,
            canvas_height=100,
            hard_render_point_budget=100,
            screen_density_budget=100,
        ),
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_cpu_tile_bytes", None),
        ("max_cpu_tile_bytes", True),
        ("max_cpu_tile_bytes", 0),
        ("max_cpu_tile_bytes", -1),
        ("max_vertex_payload_bytes", None),
        ("max_vertex_payload_bytes", True),
        ("max_vertex_payload_bytes", 0),
        ("max_vertex_payload_bytes", -1),
    ],
)
def test_session_settings_require_positive_worker_allocation_limits(name: str, value: object) -> None:
    values = {"max_cpu_tile_bytes": 1_000, "max_vertex_payload_bytes": 1_000}
    values[name] = value
    with pytest.raises(ValueError, match=name):
        _CacheSessionSettings(
            max_bucket_lookup_bytes=None,
            max_selected_value_index_bytes=None,
            **values,  # type: ignore[arg-type]
        )


def test_session_owns_reader_on_one_worker_thread_and_reuses_selection(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)
    gui_thread_id = threading.get_ident()
    callback_thread_ids: list[int] = []
    progress: list[tuple[int, int]] = []
    states: list[_CacheSessionState] = []
    session.ready.connect(lambda: callback_thread_ids.append(threading.get_ident()))
    session.value_selection_ready.connect(lambda _selection, _bytes: callback_thread_ids.append(threading.get_ident()))
    session.bucket_index_progress.connect(lambda completed, total: progress.append((completed, total)))
    session.state_changed.connect(states.append)

    try:
        _start_ready(session, qtbot)
        assert progress == [(1, 3), (2, 3), (3, 3)]
        assert session.projected_lookup_bytes == 64
        assert session.resident_lookup_bytes == 64

        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            assert session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids == (0,)
        assert not session.set_selected_value_ids((0,))
        assert probe.selection_calls == [(0,)]

        # Returning to all values drops the selected index without another
        # catalog-index load.
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            assert session.set_selected_value_ids(None)
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids is None
        assert probe.selection_calls == [(0,)]
    finally:
        _close(session, qtbot)

    worker_ids = {thread_id for _, thread_id in probe.operations}
    assert len(worker_ids) == 1
    assert worker_ids != {gui_thread_id}
    assert callback_thread_ids and set(callback_thread_ids) == {gui_thread_id}
    assert [operation for operation, _ in probe.operations] == [
        "construct",
        "enter",
        "project",
        "load_bucket_indexes",
        "load_selection",
        "exit",
    ]
    assert states == [
        _CacheSessionState.STARTING,
        _CacheSessionState.LOADING_BUCKET_INDEXES,
        _CacheSessionState.READY,
        _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX,
        _CacheSessionState.READY,
        _CacheSessionState.UPDATING_SELECTED_VALUE_INDEX,
        _CacheSessionState.READY,
        _CacheSessionState.CLOSING,
        _CacheSessionState.CLOSED,
    ]


def test_session_rejects_bucket_index_projection_before_loading_arrays(qtbot) -> None:
    probe = _ReaderProbe(projected_bytes=2_000)
    session = _session(probe, max_bucket_lookup_bytes=1_000)
    failures: list[_CacheSessionFailure] = []
    session.failed.connect(failures.append)

    with qtbot.waitSignal(session.closed, timeout=5_000):
        session.start()

    assert session.state is _CacheSessionState.CLOSED
    assert len(failures) == 1
    assert failures[0].phase == "bucket_index_projection"
    assert [operation for operation, _ in probe.operations] == ["construct", "enter", "project", "exit"]


def test_session_propagates_absent_lookup_and_selection_limits(qtbot) -> None:
    probe = _ReaderProbe(projected_bytes=2_000, resident_bytes=2_000)
    session = _session(
        probe,
        max_bucket_lookup_bytes=None,
        max_selected_value_index_bytes=None,
    )

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert probe.bucket_lookup_limits == [None]
        assert probe.selected_value_limits == [None]
    finally:
        _close(session, qtbot)


def test_selection_failure_retains_previous_ready_selection(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)
    failures: list[_CacheSessionFailure] = []
    session.failed.connect(failures.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)

        probe.fail_selection = True
        with qtbot.waitSignal(session.failed, timeout=5_000):
            session.set_selected_value_ids((1,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)

        assert failures[-1].phase == "selection"
        assert session.selected_value_ids == (0,)
        assert probe.selection_calls == [(0,), (1,)]
    finally:
        _close(session, qtbot)


def test_worker_treats_selection_without_reader_as_fatal() -> None:
    worker = _TiledPointsCacheWorker(
        Path("unused.zarr"),
        _CacheSessionSettings(
            max_bucket_lookup_bytes=None,
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000,
            max_vertex_payload_bytes=1_000_000,
        ),
        threading.Event(),
        lambda cache_root: _ControllableReader(cache_root, _ReaderProbe()),
    )
    states: list[_CacheSessionState] = []
    failures: list[_CacheSessionFailure] = []
    finished: list[None] = []
    worker.state_changed.connect(states.append)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda: finished.append(None))

    worker.update_selected_value_index((0,))

    assert states == [_CacheSessionState.FAILED]
    assert len(failures) == 1
    assert failures[0].phase == "selection"
    assert failures[0].message == "Cache reader is not ready."
    assert finished == [None]


def test_close_during_bucket_index_loading_rolls_into_owner_thread_shutdown(qtbot) -> None:
    probe = _ReaderProbe(pause_bucket_index_loading=True)
    session = _session(probe)
    ready_events: list[None] = []

    def close_after_first_bucket(completed: int, total: int) -> None:
        del total
        if completed == 1:
            session.close()
            probe.resume_bucket_index_loading.set()

    session.bucket_index_progress.connect(close_after_first_bucket)
    session.ready.connect(lambda: ready_events.append(None))

    with qtbot.waitSignal(session.closed, timeout=5_000):
        session.start()

    assert probe.bucket_index_loading_paused.is_set()
    assert ready_events == []
    assert session.state is _CacheSessionState.CLOSED
    assert [operation for operation, _ in probe.operations][-1] == "exit"


def test_close_during_selected_index_load_does_not_publish_late_selection(qtbot) -> None:
    probe = _ReaderProbe(pause_selection=True)
    session = _session(probe)
    published: list[object] = []
    session.value_selection_ready.connect(lambda selection, _bytes: published.append(selection))

    _start_ready(session, qtbot)
    session.set_selected_value_ids((0,))
    qtbot.waitUntil(probe.selection_paused.is_set, timeout=5_000)
    assert session.close()
    probe.resume_selection.set()
    qtbot.waitUntil(lambda: session.state is _CacheSessionState.CLOSED, timeout=5_000)

    assert published == []
    assert [operation for operation, _ in probe.operations][-1] == "exit"


def test_close_during_reader_construction_prevents_reader_entry(qtbot) -> None:
    probe = _ReaderProbe(pause_construction=True)
    session = _session(probe)

    session.start()
    qtbot.waitUntil(probe.construction_paused.is_set, timeout=5_000)
    assert session.close()
    probe.resume_construction.set()
    qtbot.waitUntil(lambda: session.state is _CacheSessionState.CLOSED, timeout=5_000)

    assert [operation for operation, _ in probe.operations] == ["construct"]


def test_passive_session_closes_idempotently_without_starting_reader(qtbot) -> None:
    probe = _ReaderProbe()
    session = _session(probe)

    assert session.state is _CacheSessionState.NEW
    assert probe.operations == []
    with qtbot.waitSignal(session.closed, timeout=1_000):
        assert session.close()
    assert not session.close()
    assert session.state is _CacheSessionState.CLOSED
    assert probe.operations == []
    with pytest.raises(RuntimeError, match="started only from NEW"):
        session.start()


def test_session_reads_only_nonresident_tiles_and_returns_complete_snapshots(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_cpu_tile_bytes=24)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        probe.planned_tile_x = (1, 2)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(2))

        assert probe.viewport_reads == [((0, 0, 0), (0, 1, 0)), ((0, 2, 0),)]
        assert snapshots[-1].rendered_tile_count == 2
        assert snapshots[-1].rendered_point_count == 2
        np.testing.assert_array_equal(
            snapshots[-1].render_batch.vertices["a_position"],
            np.asarray(((11.0, 0.0), (22.0, 0.0)), dtype=np.float32),
        )
    finally:
        _close(session, qtbot)


def test_session_residency_detaches_tiles_from_shared_reader_batches(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_cpu_tile_bytes=1_000)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        location_batch = probe.last_location_batch
        value_id_batch = probe.last_value_id_batch
        assert location_batch is not None
        assert value_id_batch is not None
        first_vertices = snapshots[-1].render_batch.vertices.copy()
        location_batch[:] = -100
        value_id_batch[:] = 2

        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(2))

        assert probe.viewport_reads == [((0, 0, 0), (0, 1, 0))]
        np.testing.assert_array_equal(snapshots[-1].render_batch.vertices, first_vertices)
    finally:
        _close(session, qtbot)


def test_session_rejects_over_budget_viewport_before_point_io(qtbot) -> None:
    probe = _ReaderProbe(over_budget=True)
    session = _session(probe)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        assert not snapshots[-1].within_budget
        assert snapshots[-1].rendered_tile_count == 0
        assert snapshots[-1].render_batch.point_count == 0
        assert probe.viewport_reads == []
    finally:
        _close(session, qtbot)


def test_oversized_tile_is_returned_transiently_but_not_retained(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0,))
    session = _session(probe, max_cpu_tile_bytes=1)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        for generation in (1, 2):
            with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
                session.request_viewport(_viewport_request(generation))

        assert len(snapshots) == 2
        assert all(snapshot.rendered_point_count == 1 for snapshot in snapshots)
        assert probe.viewport_reads == [((0, 0, 0),), ((0, 0, 0),)]
    finally:
        _close(session, qtbot)


def test_worker_rejects_vertex_payload_capacity_before_batch_allocation(qtbot) -> None:
    probe = _ReaderProbe(planned_tile_x=(0, 1))
    session = _session(probe, max_vertex_payload_bytes=12)
    failures: list[_CacheSessionFailure] = []
    snapshots: list[object] = []
    session.viewport_failed.connect(lambda _generation, failure: failures.append(failure))
    session.viewport_ready.connect(snapshots.append)

    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_failed, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        assert snapshots == []
        assert len(failures) == 1
        assert failures[0].phase == "viewport"
        assert "max_vertex_payload_bytes=12" in failures[0].message
    finally:
        _close(session, qtbot)


def test_worker_rejects_nonspatial_tile_order_before_packing(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _ReaderProbe(planned_tile_x=(1, 0))
    session = _session(probe)
    failures: list[_CacheSessionFailure] = []
    pack_called = False

    def _unexpected_pack(*args, **kwargs):
        nonlocal pack_called
        del args, kwargs
        pack_called = True
        raise AssertionError("packing should not be attempted")

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _unexpected_pack)
    session.viewport_failed.connect(lambda _generation, failure: failures.append(failure))
    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_failed, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        assert not pack_called
        assert len(failures) == 1
        assert failures[0].phase == "viewport"
        assert "spatial" in failures[0].message
    finally:
        _close(session, qtbot)


def test_queued_viewport_delivery_preserves_maximum_budget_vertex_allocation_identity(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _ReaderProbe(planned_tile_x=tuple(range(100)))
    session = _session(probe, max_cpu_tile_bytes=10_000, max_vertex_payload_bytes=1_200)
    worker_vertex_ids: list[int] = []
    original_pack = cache_session_module.pack_render_tiles

    def _record_pack(*args, **kwargs):
        batch = original_pack(*args, **kwargs)
        worker_vertex_ids.append(id(batch.vertices))
        return batch

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _record_pack)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    try:
        _start_ready(session, qtbot)
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(_viewport_request(1))

        snapshot = snapshots[-1]
        assert snapshot.render_batch.point_count == 100
        assert snapshot.render_batch.nbytes == 1_200
        assert id(snapshot.render_batch.vertices) == worker_vertex_ids[-1]
    finally:
        _close(session, qtbot)


@pytest.mark.parametrize("pause_check", [1, 2], ids=["before-allocation", "during-fragmented-pack"])
def test_terminal_close_cancels_worker_render_batch_packing(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    pause_check: int,
) -> None:
    probe = _ReaderProbe(planned_tile_x=tuple(range(129)))
    session = _session(probe, max_cpu_tile_bytes=10_000, max_vertex_payload_bytes=10_000)
    pack_paused = threading.Event()
    resume_pack = threading.Event()
    original_pack = cache_session_module.pack_render_tiles

    def _pause_pack(*args, check_cancelled, **kwargs):
        checks = 0

        def _check() -> None:
            nonlocal checks
            checks += 1
            if checks == pause_check:
                pack_paused.set()
                assert resume_pack.wait(timeout=5)
            check_cancelled()

        return original_pack(*args, check_cancelled=_check, **kwargs)

    monkeypatch.setattr(cache_session_module, "pack_render_tiles", _pause_pack)
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    _start_ready(session, qtbot)
    session.request_viewport(_viewport_request(1))
    qtbot.waitUntil(pack_paused.is_set, timeout=5_000)
    assert session.close()
    with qtbot.waitSignal(session.closed, timeout=5_000):
        resume_pack.set()

    assert snapshots == []
    assert session.state is _CacheSessionState.CLOSED


def test_real_cache_session_opens_primes_and_loads_selection(real_cache_root: Path, qtbot) -> None:
    session = _TiledPointsCacheSession(
        real_cache_root,
        _CacheSessionSettings(
            max_bucket_lookup_bytes=None,
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000,
            max_vertex_payload_bytes=1_000_000,
        ),
    )
    try:
        _start_ready(session, qtbot)
        assert session.dataset_info is not None
        assert session.dataset_info.value_names == ("A", "B")
        assert session.projected_lookup_bytes == session.resident_lookup_bytes

        with qtbot.waitSignal(session.value_selection_ready, timeout=5_000):
            session.set_selected_value_ids((0,))
        qtbot.waitUntil(lambda: session.state is _CacheSessionState.READY)
        assert session.selected_value_ids == (0,)
    finally:
        _close(session, qtbot)


def test_real_cache_session_builds_generation_bound_viewport_snapshot(real_cache_root: Path, qtbot) -> None:
    session = _TiledPointsCacheSession(
        real_cache_root,
        _CacheSessionSettings(
            max_bucket_lookup_bytes=None,
            max_selected_value_index_bytes=None,
            max_cpu_tile_bytes=1_000_000,
            max_vertex_payload_bytes=1_000_000,
        ),
    )
    snapshots: list[object] = []
    session.viewport_ready.connect(snapshots.append)
    try:
        _start_ready(session, qtbot)
        request = _ViewportRequest(
            request_generation=1,
            selection_generation=0,
            requested_value_ids=None,
            viewport=TiledPointsViewportState(
                displayed_axes=(0, 1),
                x_min=0.0,
                y_min=0.0,
                x_max=20.0,
                y_max=10.0,
                canvas_width=100,
                canvas_height=100,
                hard_render_point_budget=100,
                screen_density_budget=100,
            ),
        )
        with qtbot.waitSignal(session.viewport_ready, timeout=5_000):
            session.request_viewport(request)

        snapshot = snapshots[-1]
        assert snapshot.request_generation == 1
        assert snapshot.within_budget
        assert snapshot.rendered_point_count == 4
        assert snapshot.rendered_tile_count == 2
    finally:
        _close(session, qtbot)
