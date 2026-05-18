from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from napari_harpy._points_value_index import (
    DEFAULT_RANDOM_STATE,
    PointsValueSelection,
    PointsValueTable,
    _ValidatedPointsElement,
)
from napari_harpy.viewer.adapter import PointsLayerIdentity
from napari_harpy.widgets.viewer.points_controller import (
    PointsController,
    PointsControllerState,
    PointsLoadResult,
    PointsValueSource,
    PointsValueSourceJob,
)


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        self._callbacks.append(callback)

    def emit(self, *args: object) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _FakeWorker:
    def __init__(self) -> None:
        self.returned = _FakeSignal()
        self.errored = _FakeSignal()
        self.finished = _FakeSignal()
        self.start_count = 0
        self.quit_count = 0

    def start(self) -> None:
        self.start_count += 1

    def quit(self) -> None:
        self.quit_count += 1


class _DummySpatialData:
    def __init__(self, points: dict[str, object] | None = None, *, path: Path | None = None) -> None:
        self.points = {} if points is None else points
        self.path = path

    def is_backed(self) -> bool:
        return self.path is not None


def _points_dataframe(data: dict[str, object], *, npartitions: int = 1) -> dd.DataFrame:
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(pd.DataFrame(data), npartitions=npartitions)


def _example_sdata() -> _DummySpatialData:
    return _DummySpatialData(
        {
            "transcripts": _points_dataframe(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [3.0, 4.0, 5.0],
                    "gene": ["AAMP", "AXL", "MALAT1"],
                }
            )
        }
    )


def _example_validated(sdata: object, *, points_name: str = "transcripts") -> _ValidatedPointsElement:
    return _ValidatedPointsElement(
        points=_points_dataframe(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [3.0, 4.0, 5.0],
                "gene": ["AAMP", "AXL", "MALAT1"],
            }
        ),
        points_name=points_name,
        source_path=None,
        source_n_points=3,
        x="x",
        y="y",
        index_column="gene",
        points_id=None,
    )


def _example_value_table() -> PointsValueTable:
    return PointsValueTable(
        values=pd.DataFrame(
            {
                "value_id": pd.Series([0, 1, 2], dtype="uint32"),
                "value": ["AAMP", "AXL", "MALAT1"],
                "n_points": pd.Series([1, 1, 1], dtype="uint64"),
            }
        ),
        index_column="gene",
        total_count=3,
    )


def _example_value_source(sdata: object, *, points_name: str = "transcripts") -> PointsValueSource:
    validated = _example_validated(sdata, points_name=points_name)
    return PointsValueSource(
        identity=PointsLayerIdentity(
            sdata=sdata,
            points_name=points_name,
            coordinate_system="global",
            index_column="gene",
        ),
        validated=validated,
        value_table=_example_value_table(),
    )


def _example_selection(
    values: list[str] | None = None,
    *,
    total_count: int | None = None,
    render_point_budget: int = 100_000,
    is_sampled: bool = False,
    warning: str | None = None,
) -> PointsValueSelection:
    row_values = ["AAMP", "AXL"] if values is None else values
    selected_values = tuple(dict.fromkeys(row_values))
    value_id_by_value = {value: index for index, value in enumerate(selected_values)}
    return PointsValueSelection(
        coordinates=np.asarray(
            [[float(index), float(index + 10)] for index in range(len(row_values))],
            dtype="float32",
        ),
        features=pd.DataFrame(
            {
                "gene": pd.Categorical(row_values, categories=list(selected_values)),
                "value_id": pd.Series([value_id_by_value[value] for value in row_values], dtype="uint32"),
            }
        ),
        index_column="gene",
        selected_values=selected_values,
        selected_value_ids=tuple(value_id_by_value[value] for value in selected_values),
        selection_mode="values",
        total_count=len(row_values) if total_count is None else total_count,
        render_point_budget=render_point_budget,
        is_sampled=is_sampled,
        warning=warning,
    )


def _example_load_result(value_source: PointsValueSource, selection: PointsValueSelection | None = None) -> PointsLoadResult:
    return PointsLoadResult(
        identity=value_source.identity,
        selection=_example_selection() if selection is None else selection,
        value_table=value_source.value_table,
    )


def test_points_controller_initializes_without_cache() -> None:
    controller = PointsController()

    assert controller.state is PointsControllerState.NO_SDATA
    assert controller.current_value_source is None
    assert controller.current_load_result is None
    assert controller.can_build_cache is False
    assert controller.can_rebuild_cache is False
    assert controller.cache_status == "not_available"


def test_points_controller_schedules_value_loading(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    workers: list[_FakeWorker] = []
    jobs: list[PointsValueSourceJob] = []

    def create_worker(job: PointsValueSourceJob) -> _FakeWorker:
        jobs.append(job)
        worker = _FakeWorker()
        workers.append(worker)
        return worker

    monkeypatch.setattr(controller, "_create_value_source_worker", create_worker)

    changed = controller.bind_source(sdata, "transcripts", "global", "gene")
    started = controller.load_value_source()

    assert changed is True
    assert started is True
    assert controller.state is PointsControllerState.LOADING_VALUES
    assert len(jobs) == 1
    assert jobs[0].points_name == "transcripts"
    assert jobs[0].coordinate_system == "global"
    assert jobs[0].index_column == "gene"
    assert workers[0].start_count == 1
    assert controller.current_value_source is None


def test_points_controller_stores_successful_value_source(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    worker = _FakeWorker()
    value_source = _example_value_source(sdata)
    monkeypatch.setattr(controller, "_create_value_source_worker", lambda job: worker)

    controller.bind_source(sdata, "transcripts", "global", "gene")
    controller.load_value_source()
    worker.returned.emit(value_source)
    worker.finished.emit()

    assert controller.current_value_source == value_source
    assert controller.current_load_result is None
    assert controller.state is PointsControllerState.VALUES_READY
    assert controller.status_kind == "success"
    assert controller.is_loading_values is False


def test_points_controller_notifies_value_source_loaded_once_on_value_return(monkeypatch) -> None:
    loaded_sources: list[PointsValueSource] = []
    state_change_count = 0

    def on_state_changed() -> None:
        nonlocal state_change_count
        state_change_count += 1

    controller = PointsController(
        on_state_changed=on_state_changed,
        on_value_source_loaded=loaded_sources.append,
    )
    sdata = _example_sdata()
    worker = _FakeWorker()
    value_source = _example_value_source(sdata)
    monkeypatch.setattr(controller, "_create_value_source_worker", lambda job: worker)

    controller.bind_source(sdata, "transcripts", "global", "gene")
    controller.load_value_source()
    worker.returned.emit(value_source)
    worker.finished.emit()

    assert loaded_sources == [value_source]
    assert state_change_count >= 4


def test_points_controller_failed_value_loading_enters_load_failed(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    worker = _FakeWorker()
    monkeypatch.setattr(controller, "_create_value_source_worker", lambda job: worker)

    controller.bind_source(sdata, "transcripts", "global", "missing")
    controller.load_value_source()
    worker.errored.emit(ValueError("Column `missing` is unavailable."))
    worker.finished.emit()

    assert controller.current_value_source is None
    assert controller.state is PointsControllerState.LOAD_FAILED
    assert controller.status_kind == "error"
    assert "missing" in controller.status_message


def test_points_controller_missing_index_column_enters_load_failed() -> None:
    controller = PointsController()

    controller.bind_source(_example_sdata(), "transcripts", "global", None)

    assert controller.state is PointsControllerState.LOAD_FAILED
    assert controller.can_load_values is False
    assert controller.status_kind == "error"


def test_points_controller_schedules_selection_loading(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    worker = _FakeWorker()
    jobs = []
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: jobs.append(job) or worker)

    started = controller.load_selection(["AAMP"], render_point_budget=10, random_state=DEFAULT_RANDOM_STATE)

    assert started is True
    assert controller.state is PointsControllerState.LOADING_SELECTION
    assert len(jobs) == 1
    assert jobs[0].value_source == value_source
    assert jobs[0].values == ["AAMP"]
    assert jobs[0].render_point_budget == 10
    assert jobs[0].random_state == DEFAULT_RANDOM_STATE
    assert worker.start_count == 1


def test_points_controller_stores_successful_load_result(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    result = _example_load_result(value_source)
    worker = _FakeWorker()
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: worker)

    controller.load_selection(["AAMP", "AXL"], render_point_budget=100_000)
    worker.returned.emit(result)
    worker.finished.emit()

    assert controller.current_load_result == result
    assert controller.state is PointsControllerState.LOADED_SELECTION
    assert controller.status_kind == "success"
    assert controller.is_loading is False


def test_points_controller_notifies_points_loaded_once_on_load_return(monkeypatch) -> None:
    loaded_results: list[PointsLoadResult] = []
    state_change_count = 0

    def on_state_changed() -> None:
        nonlocal state_change_count
        state_change_count += 1

    controller = PointsController(
        on_state_changed=on_state_changed,
        on_points_loaded=loaded_results.append,
    )
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    result = _example_load_result(value_source)
    worker = _FakeWorker()
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: worker)

    controller.load_selection(["AAMP", "AXL"], render_point_budget=100_000)
    worker.returned.emit(result)
    worker.finished.emit()

    assert loaded_results == [result]
    assert state_change_count >= 3


def test_points_controller_uses_warning_status_for_sampled_selection(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    selection = _example_selection(
        ["AAMP"],
        total_count=10,
        render_point_budget=1,
        is_sampled=True,
        warning="Showing 1 of 10 selected points.",
    )
    result = _example_load_result(value_source, selection)
    worker = _FakeWorker()
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: worker)

    controller.load_selection(["AAMP"], render_point_budget=1)
    worker.returned.emit(result)

    assert controller.current_load_result == result
    assert controller.state is PointsControllerState.LOADED_SELECTION
    assert controller.status_kind == "warning"
    assert "Showing 1 of 10" in controller.status_message


def test_points_controller_rebinding_cancels_active_workers(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_worker = _FakeWorker()
    load_worker = _FakeWorker()
    controller._current_value_source = _example_value_source(sdata)
    monkeypatch.setattr(controller, "_create_value_source_worker", lambda job: value_worker)
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: load_worker)

    controller.bind_source(sdata, "transcripts", "global", "gene")
    controller.load_value_source()
    controller._current_value_source = _example_value_source(sdata)
    controller.load_selection(["AAMP"], render_point_budget=10)
    controller.bind_source(sdata, "other_transcripts", "global", "gene")

    assert value_worker.quit_count == 1
    assert load_worker.quit_count == 1
    assert controller.current_value_source is None
    assert controller.current_load_result is None


def test_points_controller_new_selection_load_replaces_active_load_worker(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    workers = [_FakeWorker(), _FakeWorker()]
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: workers.pop(0))

    controller.load_selection(["AAMP"], render_point_budget=10)
    first_worker = controller._active_load_worker
    controller.load_selection(["AXL"], render_point_budget=10)

    assert first_worker is not None
    assert first_worker.quit_count == 1
    assert controller._active_load_worker is not first_worker


def test_points_controller_ignores_stale_load_results(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_source = _example_value_source(sdata)
    first_worker = _FakeWorker()
    second_worker = _FakeWorker()
    worker_queue = [first_worker, second_worker]
    controller._current_value_source = value_source
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: worker_queue.pop(0))

    controller.load_selection(["AAMP"], render_point_budget=10)
    controller.load_selection(["AXL"], render_point_budget=10)
    second_result = _example_load_result(value_source, _example_selection(["AXL"]))
    first_result = _example_load_result(value_source, _example_selection(["AAMP"]))

    first_worker.returned.emit(first_result)
    second_worker.returned.emit(second_result)

    assert controller.current_load_result == second_result


def test_points_controller_shutdown_cancels_workers(monkeypatch) -> None:
    controller = PointsController()
    sdata = _example_sdata()
    value_worker = _FakeWorker()
    load_worker = _FakeWorker()
    controller._current_value_source = _example_value_source(sdata)
    monkeypatch.setattr(controller, "_create_value_source_worker", lambda job: value_worker)
    monkeypatch.setattr(controller, "_create_points_load_worker", lambda job: load_worker)

    controller.bind_source(sdata, "transcripts", "global", "gene")
    controller.load_value_source()
    controller._current_value_source = _example_value_source(sdata)
    controller.load_selection(["AAMP"], render_point_budget=10)
    controller.shutdown()

    assert value_worker.quit_count == 1
    assert load_worker.quit_count == 1
