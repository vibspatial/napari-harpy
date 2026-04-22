from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

from napari_harpy._feature_extraction import (
    FEATURE_EXTRACTION_IDLE_STATUS,
    FeatureExtractionController,
    FeatureExtractionJob,
    FeatureExtractionResult,
    _run_feature_extraction_job,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: FeatureExtractionResult | None = None) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        assert self._result is not None
        self.returned.emit(self._result)
        self.finished.emit()

    def emit_errored(self, error: Exception) -> None:
        self.errored.emit(error)
        self.finished.emit()


def test_feature_extraction_controller_starts_idle() -> None:
    controller = FeatureExtractionController()

    assert controller.status_message == FEATURE_EXTRACTION_IDLE_STATUS
    assert controller.status_kind == "warning"
    assert controller.is_running is False
    assert controller.can_calculate is False


def test_feature_extraction_controller_bind_is_passive_and_ready_for_existing_table(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area", "perimeter"],
        "feature_matrix_1",
    )

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.is_running is False
    assert controller.can_calculate is True


def test_feature_extraction_controller_blocks_when_no_table_is_selected(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        None,
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose an annotation table linked to the selected segmentation."


def test_feature_extraction_controller_requires_image_for_intensity_features(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose an image before calculating intensity features."


def test_feature_extraction_controller_requires_coordinate_system(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        None,
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose a coordinate system."
    assert controller._prepare_feature_extraction_job(8) is None


def test_feature_extraction_controller_bind_returns_false_for_unchanged_context(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    first_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )
    second_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert first_changed is True
    assert second_changed is False


def test_feature_extraction_controller_prepares_immutable_job_payload(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area", "mean"],
        "feature_matrix_1",
        overwrite_feature_key=True,
    )

    job = controller._prepare_feature_extraction_job(7)

    assert isinstance(job, FeatureExtractionJob)
    assert job.job_id == 7
    assert job.sdata is sdata_blobs
    assert job.label_name == "blobs_labels"
    assert job.image_name == "blobs_image"
    assert job.channels is None
    assert job.table_name == "table"
    assert job.coordinate_system == "global"
    assert job.feature_names == ("mean", "area")
    assert job.feature_key == "feature_matrix_1"
    assert job.overwrite_feature_key is True


def test_feature_extraction_controller_bind_stores_selected_channels_in_job(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        channels=("1", "2"),
    )

    job = controller._prepare_feature_extraction_job(8)

    assert isinstance(job, FeatureExtractionJob)
    assert job.channels == ("1", "2")


def test_feature_extraction_controller_bind_rejects_duplicate_selected_channels(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    with pytest.raises(ValueError, match="Duplicate channel selection is not allowed"):
        controller.bind(
            sdata_blobs,
            "blobs_labels",
            "blobs_image",
            "table",
            "global",
            ["mean", "area"],
            "feature_matrix_1",
            channels=("1", "1"),
        )


def test_feature_extraction_controller_morphology_only_job_keeps_channels_none(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
        channels=None,
    )

    job = controller._prepare_feature_extraction_job(9)

    assert isinstance(job, FeatureExtractionJob)
    assert job.image_name is None
    assert job.channels is None


def test_feature_extraction_controller_notifies_table_state_change_on_success(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[str] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            label_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
        )
    )

    controller = FeatureExtractionController(
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    assert controller.status_kind == "info"
    assert deferred_worker.started is True

    deferred_worker.emit_returned()

    assert table_state_changes == ["changed"]
    assert controller.status_kind == "success"
    assert (
        controller.status_message
        == "Feature extraction: wrote `feature_matrix_1` into table `table` as `.obsm['feature_matrix_1']`."
    )
    assert controller.is_running is False


def test_feature_extraction_controller_calculate_accepts_one_shot_overwrite_override(
    sdata_blobs: SpatialData,
) -> None:
    captured_overwrite_flags: list[bool] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            label_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
        )
    )

    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        overwrite_feature_key=False,
    )

    def capture_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        captured_overwrite_flags.append(job.overwrite_feature_key)
        return deferred_worker

    controller._create_feature_extraction_worker = capture_worker  # type: ignore[method-assign]

    launched = controller.calculate(overwrite_feature_key=True)

    assert launched is True
    assert captured_overwrite_flags == [True]


def test_feature_extraction_controller_calculate_launches_job_with_selected_channels(
    sdata_blobs: SpatialData,
) -> None:
    captured_channels: list[tuple[int | str, ...] | None] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            label_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
        )
    )

    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
        channels=("0", "2"),
    )

    def capture_worker(job: FeatureExtractionJob) -> _DeferredWorker:
        captured_channels.append(job.channels)
        return deferred_worker

    controller._create_feature_extraction_worker = capture_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    assert captured_channels == [("0", "2")]


def test_run_feature_extraction_job_passes_channels_to_harpy(monkeypatch, sdata_blobs: SpatialData) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_add_feature_matrix(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "harpy",
        SimpleNamespace(tb=SimpleNamespace(add_feature_matrix=fake_add_feature_matrix)),
    )

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=4,
            sdata=sdata_blobs,
            label_name="blobs_labels",
            image_name="blobs_image",
            channels=("0", "2"),
            table_name="table",
            coordinate_system="global",
            feature_names=("mean", "area"),
            feature_key="feature_matrix_1",
            overwrite_feature_key=False,
        )
    )

    assert captured_kwargs["channels"] == ["0", "2"]
    assert result == FeatureExtractionResult(
        job_id=4,
        label_name="blobs_labels",
        table_name="table",
        feature_key="feature_matrix_1",
    )


def test_feature_extraction_controller_propagates_worker_errors(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[str] = []
    deferred_worker = _DeferredWorker()

    controller = FeatureExtractionController(
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    deferred_worker.emit_errored(RuntimeError("boom"))

    assert table_state_changes == []
    assert controller.status_kind == "error"
    assert controller.status_message == "Feature extraction: calculation failed: boom"
    assert controller.is_running is False


def test_feature_extraction_controller_drops_stale_results_after_rebinding(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[str] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            label_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
        )
    )

    controller = FeatureExtractionController(
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_1",
    )
    controller._create_feature_extraction_worker = lambda job: deferred_worker  # type: ignore[method-assign]

    launched = controller.calculate()

    assert launched is True
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        ["mean", "area"],
        "feature_matrix_2",
    )

    assert deferred_worker.quit_called is True
    assert controller.status_message == "Feature extraction: ready to calculate."

    deferred_worker.emit_returned()

    assert table_state_changes == []
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
