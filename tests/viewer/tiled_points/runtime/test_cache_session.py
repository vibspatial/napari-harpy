from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_harpy.core.multi_scale_cache_points_zarr.builder import (
    _build_points_cache_zarr,
    _PointsCacheBuilderConfig,
)
from napari_harpy.core.multi_scale_cache_points_zarr.cache_format import _CatalogWriteSettings
from napari_harpy.core.multi_scale_cache_points_zarr.source import (
    ParquetPointsSource,
    PointColumnSelection,
    validate_parquet_points_source,
)
from napari_harpy.core.multi_scale_cache_points_zarr.storage.models import _ZarrWriteSettings
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

    def record(self, operation: str) -> None:
        self.operations.append((operation, threading.get_ident()))


@dataclass(frozen=True)
class _FakeSelectedValueIndex:
    resident_bytes: int


class _ControllableReader:
    value_names = ("A", "B", "C")

    def __init__(self, cache_root: Path, probe: _ReaderProbe) -> None:
        del cache_root
        self._probe = probe
        self.dataset_info = object()
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
        return _FakeSelectedValueIndex(24 if max_resident_bytes is None else min(max_resident_bytes, 24))


def _session(
    probe: _ReaderProbe,
    *,
    max_bucket_lookup_bytes: int | None = 1_000,
    max_selected_value_index_bytes: int | None = 1_000,
) -> _TiledPointsCacheSession:
    return _TiledPointsCacheSession(
        Path("unused.zarr"),
        _CacheSessionSettings(
            max_bucket_lookup_bytes=max_bucket_lookup_bytes,
            max_selected_value_index_bytes=max_selected_value_index_bytes,
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


@pytest.fixture(scope="module")
def real_cache_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiled-points-session")
    source = ParquetPointsSource(
        spatialdata_path=root / "source.zarr",
        points_name="transcripts",
        columns=PointColumnSelection(x="x", y="y", value="gene"),
    )
    source.parquet_path.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "x": pa.array([1.0, 3.0, 2.0, 11.0], type=pa.float64()),
                "y": pa.array([1.0, 2.0, 3.0, 1.0], type=pa.float64()),
                "gene": pa.array(["A", "B", "A", "B"]),
            }
        ),
        source.parquet_path / "part.0.parquet",
        row_group_size=2,
    )
    validated = validate_parquet_points_source(source, max_batch_rows=2)
    temporary_root = root / "temporary"
    temporary_root.mkdir()
    return _build_points_cache_zarr(
        validated,
        output_path=root / "transcripts_vis_zarr",
        temporary_directory_root=temporary_root,
        config=_PointsCacheBuilderConfig(
            leaf_tile_size=10,
            overview_point_budget=10,
            dask_worker_count=2,
            zarr_settings=_ZarrWriteSettings(2, 4, 2, 4, "zstd-v1"),
            catalog_settings=_CatalogWriteSettings(
                manifest_chunk_rows=2,
                manifest_shard_rows=4,
                value_tile_chunk_rows=2,
                value_tile_shard_rows=4,
            ),
        ),
    )


def test_real_cache_session_opens_primes_and_loads_selection(real_cache_root: Path, qtbot) -> None:
    session = _TiledPointsCacheSession(
        real_cache_root,
        _CacheSessionSettings(
            max_bucket_lookup_bytes=None,
            max_selected_value_index_bytes=None,
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
