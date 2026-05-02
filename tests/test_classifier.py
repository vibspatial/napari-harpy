from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal
from spatialdata import SpatialData

import napari_harpy._classifier as classifier_module
from napari_harpy._annotation import USER_CLASS_COLUMN
from napari_harpy._class_palette import default_class_colors
from napari_harpy._classifier import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
    ClassifierController,
)


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: classifier_module.ClassifierJobResult) -> None:
        super().__init__()
        self._result = result
        self.started = False
        self.quit_called = False

    def start(self) -> None:
        self.started = True

    def quit(self) -> None:
        self.quit_called = True

    def emit_returned(self) -> None:
        self.returned.emit(self._result)
        self.finished.emit()


def _set_deterministic_features(sdata: SpatialData, *, table_name: str = "table") -> None:
    table = sdata[table_name]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    first_feature = (instance_ids > 13).astype(np.float64)
    second_feature = instance_ids.astype(np.float64) / instance_ids.max()
    table.obsm["features_1"] = np.column_stack([first_feature, second_feature])


def _set_user_classes(sdata: SpatialData, class_by_instance: dict[int, int], *, table_name: str = "table") -> None:
    table = sdata[table_name]
    values = np.array([class_by_instance.get(int(instance_id), 0) for instance_id in table.obs["instance_id"]])
    categories = sorted({0, *values.tolist()})
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)


def _set_user_classes_by_region(
    sdata: SpatialData,
    class_by_region_instance: dict[tuple[str, int], int],
    *,
    table_name: str = "table_multi",
) -> None:
    table = sdata[table_name]
    region_values = table.obs["region"].astype("string").to_numpy()
    instance_values = table.obs["instance_id"].to_numpy(dtype=np.int64)
    values = np.array(
        [
            class_by_region_instance.get((str(region), int(instance_id)), 0)
            for region, instance_id in zip(region_values, instance_values, strict=True)
        ],
        dtype=np.int64,
    )
    categories = sorted({0, *values.tolist()})
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)


def _set_invalid_feature_rows_for_region(
    sdata: SpatialData,
    *,
    region_name: str,
    table_name: str = "table_multi",
    feature_key: str = "features_1",
) -> None:
    table = sdata[table_name]
    feature_matrix = np.asarray(table.obsm[feature_key], dtype=np.float64).copy()
    region_mask = (table.obs["region"].astype("string") == region_name).to_numpy(dtype=bool, copy=False)
    feature_matrix[region_mask, :] = np.nan
    table.obsm[feature_key] = feature_matrix


def _resolved_scope(
    positions: np.ndarray | list[int] | tuple[int, ...],
    *,
    label_name: str = "blobs_labels",
    mode: classifier_module.ClassifierScopeMode = "selected_segmentation_only",
    regions: tuple[str, ...] | None = None,
    n_rows_in_regions: int | None = None,
) -> classifier_module.ResolvedClassifierScope:
    table_row_positions = np.asarray(positions, dtype=np.int64)
    return classifier_module.ResolvedClassifierScope(
        mode=mode,
        regions=(label_name,) if regions is None else regions,
        table_row_positions=table_row_positions,
        n_rows_in_regions=int(table_row_positions.size if n_rows_in_regions is None else n_rows_in_regions),
    )


def test_classifier_controller_trains_on_labeled_rows_and_predicts_active_objects(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table_state_changes: list[str] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]
    qtbot.waitUntil(lambda: table.obs[PRED_CLASS_COLUMN].astype("string").ne("0").any(), timeout=5000)

    pred_class = table.obs.set_index("instance_id")[PRED_CLASS_COLUMN]
    pred_confidence = table.obs.set_index("instance_id")[PRED_CONFIDENCE_COLUMN]

    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert list(table.obs[PRED_CLASS_COLUMN].cat.categories) == [0, 1, 2]
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([0, 1, 2])
    assert pred_class.loc[1] == 1
    assert pred_class.loc[5] == 1
    assert pred_class.loc[24] == 2
    assert pred_class.loc[26] == 2
    assert pd.to_numeric(pred_class.astype("string"), errors="coerce").min() >= 1
    assert pred_confidence.between(0.0, 1.0).all()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["eligible"] is True
    assert table.uns[CLASSIFIER_CONFIG_KEY]["trained"] is True
    assert table.uns[CLASSIFIER_CONFIG_KEY]["class_labels_seen"] == [1, 2]
    assert controller.status_kind == "success"
    assert table_state_changes == ["changed"]


def test_multi_region_classifier_fixture_duplicates_instance_ids_only_across_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    table = sdata_blobs_multi_region["table_multi"]
    first_region_rows = table.obs.loc[table.obs["region"].astype("string") == "blobs_labels"]
    second_region_rows = table.obs.loc[table.obs["region"].astype("string") == "blobs_labels_2"]

    assert first_region_rows["instance_id"].is_unique
    assert second_region_rows["instance_id"].is_unique
    assert set(first_region_rows["instance_id"].tolist()) == set(second_region_rows["instance_id"].tolist())


def test_classifier_controller_selected_region_only_training_scope_ignores_labels_in_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels_2", 1): 1,
            ("blobs_labels_2", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        training_scope="selected_segmentation_only",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels",)
    assert job.training_scope.n_rows_in_regions == table.n_obs // 2
    assert job.training_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs // 2
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 1
    assert job.summary.labeled_count == 0
    assert job.summary.eligible is False
    assert "Need at least 2 labeled samples" in job.summary.reason


def test_classifier_controller_defaults_training_scope_to_all_and_can_use_labels_from_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels_2", 1): 1,
            ("blobs_labels_2", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert job.training_scope.n_rows_in_regions == table.n_obs
    assert job.training_scope.n_eligible_rows == table.n_obs
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.prediction_scope.n_rows_in_regions == table.n_obs // 2
    assert job.prediction_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.labeled_count == 4
    assert job.summary.eligible is True


def test_classifier_controller_default_table_wide_training_excludes_invalid_rows_from_other_regions(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels", 24): 2,
            ("blobs_labels", 25): 2,
        },
    )
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
    )

    job = controller._prepare_classifier_job(1)

    assert job is not None
    assert job.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert job.training_scope.n_rows_in_regions == table.n_obs
    assert job.training_scope.n_eligible_rows == table.n_obs // 2
    assert job.training_scope.n_excluded_feature_invalid_rows == table.n_obs // 2
    assert job.prediction_scope.regions == ("blobs_labels",)
    assert job.prediction_scope.n_eligible_rows == table.n_obs // 2
    assert job.summary.resolved_training_row_count == table.n_obs // 2
    assert job.summary.resolved_prediction_row_count == table.n_obs // 2
    assert job.summary.training_region_count == 2
    assert job.summary.eligible is True


def test_classifier_controller_prediction_scope_all_clears_invalid_rows_in_scope(
    qtbot, sdata_blobs_multi_region: SpatialData
) -> None:
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels", 24): 2,
            ("blobs_labels", 25): 2,
        },
    )
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[0, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        prediction_scope="all",
    )
    controller.schedule_retrain(immediate=True)

    region_values = table.obs["region"].astype("string")
    valid_rows = region_values == "blobs_labels"
    invalid_rows = region_values == "blobs_labels_2"

    qtbot.waitUntil(
        lambda: (
            CLASSIFIER_CONFIG_KEY in table.uns
            and table.uns[CLASSIFIER_CONFIG_KEY].get("trained") is True
            and controller.status_kind == "success"
        ),
        timeout=5000,
    )

    assert table.obs.loc[valid_rows, PRED_CLASS_COLUMN].astype("string").ne("0").all()
    assert table.obs.loc[valid_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[invalid_rows, PRED_CLASS_COLUMN].eq(0).all()
    assert table.obs.loc[invalid_rows, PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "all"
    assert table.uns[CLASSIFIER_CONFIG_KEY]["prediction_regions"] == ["blobs_labels", "blobs_labels_2"]
    assert table.uns[CLASSIFIER_CONFIG_KEY]["n_predicted_rows"] == int(valid_rows.sum())
    assert table.uns[CLASSIFIER_CONFIG_KEY]["training_scope"] == "all"
    assert table.uns[CLASSIFIER_CONFIG_KEY]["n_training_rows"] == int(valid_rows.sum())


def test_classifier_controller_describes_current_preparation_without_building_worker_arrays(
    monkeypatch, sdata_blobs_multi_region: SpatialData
) -> None:
    _set_user_classes_by_region(
        sdata_blobs_multi_region,
        {
            ("blobs_labels", 1): 1,
            ("blobs_labels", 2): 1,
            ("blobs_labels_2", 24): 2,
            ("blobs_labels_2", 25): 2,
        },
    )

    def fail_slice_feature_rows(feature_matrix, positions):
        del feature_matrix, positions
        raise AssertionError("preparation summary should not construct worker feature arrays")

    monkeypatch.setattr(classifier_module, "_slice_feature_rows", fail_slice_feature_rows)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(
        sdata_blobs_multi_region,
        "blobs_labels",
        "table_multi",
        "features_1",
        prediction_scope="all",
    )

    summary = controller.describe_current_preparation()

    assert summary is not None
    assert summary.training_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert summary.prediction_scope.regions == ("blobs_labels", "blobs_labels_2")
    assert summary.resolved_prediction_row_count == sdata_blobs_multi_region["table_multi"].n_obs
    assert summary.labeled_count == 4
    assert summary.eligible is True


def test_classifier_controller_resets_predictions_when_only_one_class_is_labeled(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1})
    table_state_changes: list[str] = []

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]

    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert table.obs[PRED_CLASS_COLUMN].eq(0).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([0])
    assert table.uns[CLASSIFIER_CONFIG_KEY]["eligible"] is False
    assert "two labeled classes" in table.uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert controller.status_kind == "warning"
    assert table_state_changes == ["changed"]


def test_classifier_controller_validates_feature_matrix_shape(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 24: 2})
    table = sdata_blobs["table"]

    def raise_shape_error(feature_matrix, n_obs, *, copy=True):
        del feature_matrix, n_obs, copy
        raise ValueError("Feature matrix has 25 rows but the table has 26 observations.")

    monkeypatch.setattr(classifier_module, "_normalize_feature_matrix", raise_shape_error)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    assert isinstance(table.obs[PRED_CLASS_COLUMN].dtype, pd.CategoricalDtype)
    assert table.obs[PRED_CLASS_COLUMN].eq(0).all()
    assert table.obs[PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[PRED_CLASS_COLORS_KEY] == default_class_colors([0])
    assert "rows but the table has" in table.uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert controller.status_kind == "warning"


def test_classifier_controller_drops_stale_results(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    call_log: list[int] = []

    def fake_fit(job):
        call_log.append(job.job_id)
        if job.job_id == 1:
            time.sleep(0.2)
            pred_class = np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64)
        else:
            pred_class = np.full(job.prediction_scope.table_row_positions.shape, 2, dtype=np.int64)

        return classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            pred_classes=pred_class,
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]
    qtbot.waitUntil(
        lambda: table.obs[PRED_CLASS_COLUMN].nunique() == 1 and table.obs[PRED_CLASS_COLUMN].iloc[0] == 2, timeout=5000
    )

    assert call_log == [1, 2]
    assert table.obs[PRED_CLASS_COLUMN].eq(2).all()
    assert controller.status_kind == "success"


def test_classifier_controller_bind_is_passive_until_marked_dirty(qtbot, monkeypatch, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})

    call_log: list[int] = []
    table_state_changes: list[str] = []

    def fake_fit(job):
        call_log.append(job.job_id)
        return classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    context_changed = controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    assert context_changed is True
    assert controller.is_dirty is False
    assert controller.status_message == "Classifier: model is up to date."
    assert call_log == []
    assert table_state_changes == []

    controller.mark_dirty(reason="the feature matrix changed")

    assert controller.is_dirty is True
    assert "Classifier: model is stale because the feature matrix changed" == controller.status_message

    controller.retrain_now()
    qtbot.waitUntil(lambda: call_log == [1], timeout=5000)
    assert controller.is_dirty is False
    assert table_state_changes == ["changed"]


def test_classifier_controller_notifies_table_state_change_when_training_errors(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table_state_changes: list[str] = []

    def raise_fit_error(job):
        del job
        raise RuntimeError("boom")

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", raise_fit_error)

    controller = ClassifierController(
        debounce_interval_ms=0,
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    qtbot.waitUntil(lambda: controller.status_kind == "error", timeout=5000)

    assert table_state_changes == ["changed"]
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["trained"] is False
    assert "boom" in sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["reason"]
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["training_scope"] == "all"
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["prediction_scope"] == "selected_segmentation_only"


def test_classifier_controller_freeze_for_reload_cancels_pending_debounce(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    created_job_ids: list[int] = []

    def fake_create_training_worker(self, job):
        del self
        created_job_ids.append(job.job_id)
        raise AssertionError("reload freeze should cancel the debounce before a worker is created")

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(
        debounce_interval_ms=50,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")

    controller.schedule_retrain()
    controller.freeze_for_reload()
    qtbot.wait(150)

    assert created_job_ids == []
    assert controller.is_training is False
    assert controller.is_dirty is True
    assert controller.status_message == "Classifier: model is stale. Click Train Classifier to refresh predictions."


def test_classifier_controller_invalidates_pending_work_for_selected_feature_matrix_overwrite(
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    result = classifier_module.ClassifierJobResult(
        job_id=1,
        feature_key="features_1",
        label_name="blobs_labels",
        table_name="table",
        pred_classes=np.array([1, 2], dtype=np.int64),
        pred_confidences=np.array([0.9, 0.8], dtype=np.float64),
        trained_at="2026-04-13T09:00:00+00:00",
        model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
        summary=classifier_module.ClassifierPreparationSummary(
            training_scope=_resolved_scope([0, 1]),
            prediction_scope=_resolved_scope([0, 1]),
            eligible=True,
            reason="Ready to train.",
            resolved_training_row_count=2,
            resolved_prediction_row_count=2,
            training_region_count=1,
            labeled_count=2,
            class_labels=(1, 2),
            n_features=2,
        ),
    )
    worker = _DeferredWorker(result)

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller._create_training_worker = lambda job: worker  # type: ignore[method-assign]

    assert controller.schedule_retrain(immediate=True) is True
    assert controller.is_training is True

    invalidated = controller.invalidate_for_feature_matrix_overwrite("features_1")

    assert invalidated is True
    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller.is_dirty is True
    assert "overwritten" in controller.status_message


def test_classifier_controller_reset_after_reload_ignores_late_worker_results(
    monkeypatch, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    workers: dict[int, _DeferredWorker] = {}

    def fake_create_training_worker(self, job):
        del self
        result = classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            pred_classes=np.full(job.prediction_scope.table_row_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.prediction_scope.table_row_positions.shape, 0.91, dtype=np.float64),
            trained_at="2026-04-13T09:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            summary=job.summary,
        )
        worker = _DeferredWorker(result)
        workers[job.job_id] = worker
        return worker

    monkeypatch.setattr(ClassifierController, "_create_training_worker", fake_create_training_worker)

    controller = ClassifierController(
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.retrain_now()

    worker = workers[1]
    assert worker.started is True
    assert controller.is_training is True

    controller.freeze_for_reload()

    table = sdata_blobs["table"]
    disk_predictions = np.full(table.n_obs, 2, dtype=np.int64)
    disk_confidences = np.full(table.n_obs, 0.77, dtype=np.float64)
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(disk_predictions, categories=[0, 2])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(
        disk_confidences,
        index=table.obs.index,
        dtype="float64",
    )
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "selected_segmentation_only",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }

    controller.reset_after_reload()

    assert worker.quit_called is True
    assert controller.is_training is False
    assert controller.is_dirty is False
    assert (
        controller.status_message
        == f"Classifier: model is up to date. Loaded predictions for {table.n_obs} objects from disk."
    )


def test_classifier_controller_reset_after_reload_marks_stale_when_prediction_scope_differs(
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.uns[CLASSIFIER_CONFIG_KEY] = {
        "model_type": "RandomForestClassifier",
        "feature_key": "features_1",
        "table_name": "table",
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-04-13T09:00:00+00:00",
        "n_labeled_objects": 4,
        "n_features": 2,
        "class_labels_seen": [1, 2],
        "rf_params": dict(classifier_module.RANDOM_FOREST_PARAMS),
        "training_scope": "all",
        "training_regions": ["blobs_labels"],
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": ["blobs_labels"],
        "n_predicted_rows": int(table.n_obs),
    }

    controller = ClassifierController(debounce_interval_ms=0)
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.reset_after_reload()

    assert controller.is_dirty is True
    assert "prediction scope `all`" in controller.status_message
