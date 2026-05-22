from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

from napari_harpy._app_state import FeatureMatrixWrittenEvent
from napari_harpy.widgets.feature_extraction.controller import (
    FEATURE_EXTRACTION_IDLE_STATUS,
    FeatureExtractionBindingState,
    FeatureExtractionController,
    FeatureExtractionJob,
    FeatureExtractionRequest,
    FeatureExtractionResult,
    FeatureExtractionTriplet,
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


def test_feature_extraction_request_maps_create_table_target_to_harpy_parameters() -> None:
    request = FeatureExtractionRequest(
        triplets=(),
        table_name="features_table",
        create_table=True,
        feature_names=("area",),
        feature_key="features",
    )

    assert request.harpy_table_name is None
    assert request.harpy_output_table_name == "features_table"


def test_feature_extraction_request_maps_existing_table_target_to_harpy_parameters() -> None:
    request = FeatureExtractionRequest(
        triplets=(),
        table_name="table",
        create_table=False,
        feature_names=("area",),
        feature_key="features",
    )

    assert request.harpy_table_name == "table"
    assert request.harpy_output_table_name is None


def test_feature_extraction_request_rejects_invalid_table_target_state() -> None:
    with pytest.raises(ValueError, match="requires a table name"):
        FeatureExtractionRequest(
            triplets=(),
            table_name="",
            create_table=True,
            feature_names=("area",),
            feature_key="features",
        )

    with pytest.raises(ValueError, match="Cannot overwrite a feature key while creating a new table"):
        FeatureExtractionRequest(
            triplets=(),
            table_name="features_table",
            create_table=True,
            feature_names=("area",),
            feature_key="features",
            overwrite_feature_key=True,
        )


def test_feature_extraction_binding_state_allows_incomplete_table_target() -> None:
    binding_state = FeatureExtractionBindingState(
        sdata=None,
        triplets=(),
        table_name=None,
        create_table=True,
        feature_names=(),
        feature_key=None,
    )

    assert binding_state.harpy_table_name is None
    assert binding_state.harpy_output_table_name is None


def test_feature_extraction_binding_state_rejects_empty_table_name() -> None:
    with pytest.raises(ValueError, match="binding table name cannot be empty"):
        FeatureExtractionBindingState(
            sdata=None,
            triplets=(),
            table_name=" ",
            create_table=False,
            feature_names=(),
            feature_key=None,
        )


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


def test_feature_extraction_controller_rejects_spatialdata_invalid_feature_key(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "features tst",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert "choose a valid feature matrix key" in controller.status_message
    assert "alphanumeric characters, underscores, dots and hyphens" in controller.status_message
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


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
    assert (
        controller.status_message
        == "Feature extraction: choose an annotation table linked to the selected labels element."
    )


def test_feature_extraction_controller_bind_is_ready_for_create_table(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features_table",
        "global",
        ["area", "perimeter"],
        "feature_matrix_1",
        create_table=True,
    )

    job = controller._prepare_feature_extraction_job(8, overwrite_feature_key=False)

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.can_calculate is True
    assert isinstance(job, FeatureExtractionJob)
    assert job.change_kind == "created"
    assert job.request.create_table is True
    assert job.request.table_name == "features_table"
    assert job.request.harpy_table_name is None
    assert job.request.harpy_output_table_name == "features_table"


def test_feature_extraction_controller_blocks_create_table_name_collision(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == (
        "Feature extraction: table `table` already exists. Choose a different table name."
    )
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_blocks_invalid_create_table_name(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert "choose a valid table name" in controller.status_message
    assert "alphanumeric characters, underscores, dots and hyphens" in controller.status_message
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_blocks_create_table_feature_key_overwrite_override(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()
    controller.bind(
        sdata_blobs,
        "blobs_labels",
        None,
        "features_table",
        "global",
        ["area"],
        "feature_matrix_1",
        create_table=True,
    )

    launched = controller.calculate(overwrite_feature_key=True)

    assert launched is False
    assert controller.status_kind == "warning"
    assert controller.status_message == (
        "Feature extraction: cannot overwrite a feature key while creating a new table."
    )


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
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


def test_feature_extraction_controller_requires_segmentation_mask(sdata_blobs: SpatialData) -> None:
    controller = FeatureExtractionController()

    controller.bind(
        sdata_blobs,
        None,
        None,
        "table",
        "global",
        ["area"],
        "feature_matrix_1",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert controller.status_message == "Feature extraction: choose a labels element."
    assert controller._prepare_feature_extraction_job(8, overwrite_feature_key=False) is None


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
    )

    job = controller._prepare_feature_extraction_job(7, overwrite_feature_key=True)

    assert isinstance(job, FeatureExtractionJob)
    assert job.job_id == 7
    assert job.sdata is sdata_blobs
    assert job.labels_name == "blobs_labels"
    assert job.image_name == "blobs_image"
    assert job.channels is None
    assert job.table_name == "table"
    assert job.coordinate_system == "global"
    assert job.feature_names == ("mean", "area")
    assert job.feature_key == "feature_matrix_1"
    assert job.overwrite_feature_key is True
    assert job.request.triplets == (
        FeatureExtractionTriplet(
            coordinate_system="global",
            labels_name="blobs_labels",
            image_name="blobs_image",
            channels=None,
        ),
    )


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

    job = controller._prepare_feature_extraction_job(8, overwrite_feature_key=False)

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

    job = controller._prepare_feature_extraction_job(9, overwrite_feature_key=False)

    assert isinstance(job, FeatureExtractionJob)
    assert job.image_name is None
    assert job.channels is None


def test_feature_extraction_controller_bind_batch_is_ready_for_multi_target_request(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    context_changed = controller.bind_batch(
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name=None,
            ),
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name=None,
            ),
        ),
        "table",
        ["area", "perimeter"],
        "feature_matrix_batch",
    )

    assert context_changed is True
    assert controller.status_message == "Feature extraction: ready to calculate."
    assert controller.status_kind == "success"
    assert controller.can_calculate is True


def test_feature_extraction_controller_exposes_binding_state_snapshot(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()
    triplets = (
        FeatureExtractionTriplet(
            coordinate_system="global",
            labels_name="blobs_labels",
            image_name="blobs_image",
            channels=("0", "2"),
        ),
    )

    controller.bind_batch(
        sdata_blobs,
        triplets,
        "table",
        ["mean"],
        "feature_matrix_batch",
    )

    binding_state = controller.binding_state

    assert binding_state.sdata is sdata_blobs
    assert binding_state.triplets == triplets
    assert binding_state.table_name == "table"
    assert binding_state.feature_names == ("mean",)
    assert binding_state.feature_key == "feature_matrix_batch"


def test_feature_extraction_controller_bind_batch_rejects_mixed_channel_selections_for_intensity_features(
    sdata_blobs: SpatialData,
) -> None:
    controller = FeatureExtractionController()

    controller.bind_batch(
        sdata_blobs,
        (
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_labels",
                image_name="blobs_image",
                channels=("0", "1"),
            ),
            FeatureExtractionTriplet(
                coordinate_system="global",
                labels_name="blobs_multiscale_labels",
                image_name="blobs_multiscale_image",
                channels=("2",),
            ),
        ),
        "table",
        ["mean"],
        "feature_matrix_batch",
    )

    assert controller.can_calculate is False
    assert controller.status_kind == "warning"
    assert (
        controller.status_message
        == "Feature extraction: all selected extraction targets must currently use the same channel selection."
    )


def test_feature_extraction_controller_notifies_table_state_change_on_success(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[FeatureExtractionResult] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_name="blobs_labels",
        table_name="table",
        feature_key="feature_matrix_1",
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_table_state_changed=table_state_changes.append,
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

    assert table_state_changes == [result]
    assert controller.status_kind == "success"
    assert (
        controller.status_message
        == "Feature extraction: wrote `feature_matrix_1` into table `table` as `.obsm['feature_matrix_1']` "
        "with metadata in `.uns['feature_matrices']['feature_matrix_1']`."
    )
    assert controller.is_running is False


def test_feature_extraction_controller_notifies_feature_matrix_written_on_success(sdata_blobs: SpatialData) -> None:
    written_events: list[FeatureMatrixWrittenEvent] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
            change_kind="created",
        )
    )

    controller = FeatureExtractionController(
        on_feature_matrix_written=lambda event: written_events.append(event),
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

    deferred_worker.emit_returned()

    assert written_events == [
        FeatureMatrixWrittenEvent(
            sdata=sdata_blobs,
            table_name="table",
            feature_key="feature_matrix_1",
            change_kind="created",
        )
    ]


def test_feature_extraction_controller_calculate_accepts_one_shot_overwrite_override(
    sdata_blobs: SpatialData,
) -> None:
    captured_overwrite_flags: list[bool] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_name="blobs_labels",
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
            labels_name="blobs_labels",
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
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name="blobs_image",
                        channels=("0", "2"),
                    ),
                ),
                table_name="table",
                create_table=False,
                feature_names=("mean", "area"),
                feature_key="feature_matrix_1",
                overwrite_feature_key=False,
            ),
        )
    )

    assert captured_kwargs["channels"] == ["0", "2"]
    assert captured_kwargs["feature_matrices_key"] == "feature_matrices"
    assert captured_kwargs["output_table_name"] is None
    assert captured_kwargs["overwrite_output_table"] is False
    assert result == FeatureExtractionResult(
        job_id=4,
        labels_name="blobs_labels",
        table_name="table",
        feature_key="feature_matrix_1",
    )


def test_run_feature_extraction_job_submits_multi_target_request_to_harpy(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
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
            job_id=5,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name="blobs_image",
                        channels=("0", "2"),
                    ),
                    FeatureExtractionTriplet(
                        coordinate_system="aligned",
                        labels_name="blobs_multiscale_labels",
                        image_name="blobs_multiscale_image",
                        channels=("0", "2"),
                    ),
                ),
                table_name="table",
                create_table=False,
                feature_names=("mean", "area"),
                feature_key="feature_matrix_batch",
                overwrite_feature_key=True,
            ),
        )
    )

    assert captured_kwargs["labels_name"] == ["blobs_labels", "blobs_multiscale_labels"]
    assert captured_kwargs["image_name"] == ["blobs_image", "blobs_multiscale_image"]
    assert captured_kwargs["table_name"] == "table"
    assert captured_kwargs["output_table_name"] is None
    assert captured_kwargs["to_coordinate_system"] == ["global", "aligned"]
    assert captured_kwargs["channels"] == ["0", "2"]
    assert captured_kwargs["overwrite_output_table"] is False
    assert captured_kwargs["overwrite_feature_key"] is True
    assert result == FeatureExtractionResult(
        job_id=5,
        labels_name=None,
        table_name="table",
        feature_key="feature_matrix_batch",
        triplet_count=2,
    )


def test_run_feature_extraction_job_fills_one_feature_matrix_for_multiple_regions(
    monkeypatch,
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    feature_key = "feature_matrix_multi_region"

    def fake_add_feature_matrix(**kwargs):
        assert kwargs["labels_name"] == ["blobs_labels", "blobs_labels_2"]
        assert kwargs["table_name"] == "table_multi"
        assert kwargs["feature_key"] == feature_key

        region_values = table.obs["region"].astype("string").to_numpy()
        features = np.full((table.n_obs, 1), np.nan, dtype=np.float64)
        features[region_values == "blobs_labels", 0] = 1.0
        features[region_values == "blobs_labels_2", 0] = 2.0
        table.obsm[feature_key] = features

    monkeypatch.setitem(
        sys.modules,
        "harpy",
        SimpleNamespace(tb=SimpleNamespace(add_feature_matrix=fake_add_feature_matrix)),
    )

    result = _run_feature_extraction_job.__wrapped__(
        FeatureExtractionJob(
            job_id=6,
            sdata=sdata_blobs_multi_region,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name=None,
                    ),
                    FeatureExtractionTriplet(
                        coordinate_system="global_1",
                        labels_name="blobs_labels_2",
                        image_name=None,
                    ),
                ),
                table_name="table_multi",
                create_table=False,
                feature_names=("area",),
                feature_key=feature_key,
                overwrite_feature_key=True,
            ),
        )
    )

    features = np.asarray(table.obsm[feature_key], dtype=np.float64)
    region_values = table.obs["region"].astype("string").to_numpy()

    assert result == FeatureExtractionResult(
        job_id=6,
        labels_name=None,
        table_name="table_multi",
        feature_key=feature_key,
        triplet_count=2,
    )
    assert features.shape == (table.n_obs, 1)
    assert np.all(features[region_values == "blobs_labels", 0] == 1.0)
    assert np.all(features[region_values == "blobs_labels_2", 0] == 2.0)


def test_run_feature_extraction_job_submits_create_table_request_to_harpy(
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
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
            job_id=7,
            sdata=sdata_blobs,
            request=FeatureExtractionRequest(
                triplets=(
                    FeatureExtractionTriplet(
                        coordinate_system="global",
                        labels_name="blobs_labels",
                        image_name=None,
                    ),
                ),
                table_name="features_table",
                create_table=True,
                feature_names=("area",),
                feature_key="feature_matrix_1",
                overwrite_feature_key=False,
            ),
        )
    )

    assert captured_kwargs["table_name"] is None
    assert captured_kwargs["output_table_name"] == "features_table"
    assert captured_kwargs["overwrite_output_table"] is False
    assert captured_kwargs["overwrite_feature_key"] is False
    assert result == FeatureExtractionResult(
        job_id=7,
        labels_name="blobs_labels",
        table_name="features_table",
        feature_key="feature_matrix_1",
    )


def test_feature_extraction_controller_propagates_worker_errors(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[str] = []
    deferred_worker = _DeferredWorker()

    controller = FeatureExtractionController(
        on_table_state_changed=lambda result: table_state_changes.append(result.table_name),
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


def test_feature_extraction_controller_notifies_table_state_before_feature_matrix_written(
    sdata_blobs: SpatialData,
) -> None:
    notification_order: list[str] = []
    result = FeatureExtractionResult(
        job_id=1,
        labels_name="blobs_labels",
        table_name="table",
        feature_key="feature_matrix_1",
        change_kind="created",
    )
    deferred_worker = _DeferredWorker(result)

    controller = FeatureExtractionController(
        on_table_state_changed=lambda table_result: notification_order.append(
            f"table:{table_result.table_name}"
        ),
        on_feature_matrix_written=lambda event: notification_order.append(f"event:{event.table_name}"),
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

    assert controller.calculate() is True
    deferred_worker.emit_returned()

    assert notification_order == ["table:table", "event:table"]


def test_feature_extraction_controller_drops_stale_results_after_rebinding(sdata_blobs: SpatialData) -> None:
    table_state_changes: list[str] = []
    deferred_worker = _DeferredWorker(
        FeatureExtractionResult(
            job_id=1,
            labels_name="blobs_labels",
            table_name="table",
            feature_key="feature_matrix_1",
        )
    )

    controller = FeatureExtractionController(
        on_table_state_changed=lambda result: table_state_changes.append(result.table_name),
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
