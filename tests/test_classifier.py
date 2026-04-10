from __future__ import annotations

import time

import numpy as np
import pandas as pd
from spatialdata import SpatialData

import napari_harpy._classifier as classifier_module
from napari_harpy._annotation import USER_CLASS_COLUMN
from napari_harpy._classifier import (
    CLASSIFIER_CONFIG_KEY,
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
    ClassifierController,
)
from napari_harpy._class_palette import default_class_colors
from napari_harpy._spatialdata import SpatialDataAdapter


def _set_deterministic_features(sdata: SpatialData) -> None:
    table = sdata["table"]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    first_feature = (instance_ids > 13).astype(np.float64)
    second_feature = instance_ids.astype(np.float64) / instance_ids.max()
    table.obsm["features_1"] = np.column_stack([first_feature, second_feature])


def _set_user_classes(sdata: SpatialData, class_by_instance: dict[int, int]) -> None:
    table = sdata["table"]
    values = np.array([class_by_instance.get(int(instance_id), 0) for instance_id in table.obs["instance_id"]])
    categories = sorted({0, *values.tolist()})
    table.obs[USER_CLASS_COLUMN] = pd.Categorical(values, categories=categories)


def test_classifier_controller_trains_on_labeled_rows_and_predicts_active_objects(qtbot, sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1, 24: 2, 25: 2})
    table_state_changes: list[str] = []

    controller = ClassifierController(
        SpatialDataAdapter(),
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


def test_classifier_controller_resets_predictions_when_only_one_class_is_labeled(
    qtbot, sdata_blobs: SpatialData
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_user_classes(sdata_blobs, {1: 1, 2: 1})
    table_state_changes: list[str] = []

    controller = ClassifierController(
        SpatialDataAdapter(),
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

    def raise_shape_error(feature_matrix, n_obs):
        del feature_matrix, n_obs
        raise ValueError("Feature matrix has 25 rows but the table has 26 observations.")

    monkeypatch.setattr(classifier_module, "_normalize_feature_matrix", raise_shape_error)

    controller = ClassifierController(
        SpatialDataAdapter(),
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
            pred_class = np.full(job.active_positions.shape, 1, dtype=np.int64)
        else:
            pred_class = np.full(job.active_positions.shape, 2, dtype=np.int64)

        return classifier_module.ClassifierJobResult(
            job_id=job.job_id,
            feature_key=job.feature_key,
            label_name=job.label_name,
            table_name=job.table_name,
            active_positions=job.active_positions,
            pred_classes=pred_class,
            pred_confidences=np.full(job.active_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            eligibility=job.eligibility,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        SpatialDataAdapter(),
        debounce_interval_ms=0,
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)
    controller.schedule_retrain(immediate=True)

    table = sdata_blobs["table"]
    qtbot.waitUntil(lambda: table.obs[PRED_CLASS_COLUMN].nunique() == 1 and table.obs[PRED_CLASS_COLUMN].iloc[0] == 2, timeout=5000)

    assert call_log == [1, 2]
    assert table.obs[PRED_CLASS_COLUMN].eq(2).all()
    assert controller.status_kind == "success"


def test_classifier_controller_bind_is_passive_until_marked_dirty(
    qtbot, monkeypatch, sdata_blobs: SpatialData
) -> None:
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
            active_positions=job.active_positions,
            pred_classes=np.full(job.active_positions.shape, 1, dtype=np.int64),
            pred_confidences=np.full(job.active_positions.shape, 0.9, dtype=np.float64),
            trained_at="2026-04-08T12:00:00+00:00",
            model_params=dict(classifier_module.RANDOM_FOREST_PARAMS),
            eligibility=job.eligibility,
        )

    monkeypatch.setattr(classifier_module, "_fit_classifier_job", fake_fit)

    controller = ClassifierController(
        SpatialDataAdapter(),
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
        SpatialDataAdapter(),
        debounce_interval_ms=0,
        on_table_state_changed=lambda: table_state_changes.append("changed"),
    )
    controller.bind(sdata_blobs, "blobs_labels", "table", "features_1")
    controller.schedule_retrain(immediate=True)

    qtbot.waitUntil(lambda: controller.status_kind == "error", timeout=5000)

    assert table_state_changes == ["changed"]
    assert sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["trained"] is False
    assert "boom" in sdata_blobs["table"].uns[CLASSIFIER_CONFIG_KEY]["reason"]
