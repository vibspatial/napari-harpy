from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from spatialdata import SpatialData, read_zarr

from napari_harpy import headless
from napari_harpy._classifier_core import (
    CLASSIFIER_APPLY_CONFIG_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy._classifier_export import (
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    ClassifierExportBundle,
    write_classifier_export_bundle,
)

_FEATURE_MATRICES_KEY = "feature_matrices"


def _set_deterministic_features(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
) -> None:
    table = sdata[table_name]
    instance_ids = table.obs["instance_id"].to_numpy(dtype=np.int64)
    first_feature = (instance_ids > 13).astype(np.float64)
    second_feature = instance_ids.astype(np.float64) / instance_ids.max()
    table.obsm[feature_key] = np.column_stack([first_feature, second_feature])


def _set_feature_metadata(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
    feature_columns: tuple[str, ...] = ("is_large", "instance_fraction"),
    features: tuple[str, ...] = ("synthetic_size", "synthetic_position"),
) -> None:
    table = sdata[table_name]
    table.uns.setdefault(_FEATURE_MATRICES_KEY, {})[feature_key] = {
        "feature_columns": list(feature_columns),
        "schema_version": 1,
        "backend": "numpy",
        "dtype": str(np.asarray(table.obsm[feature_key]).dtype),
        "source_label": "blobs_labels",
        "source_image": None,
        "coordinate_system": "global",
        "features": list(features),
    }


def _make_classifier_bundle(
    sdata: SpatialData,
    *,
    table_name: str = "table",
    feature_key: str = "features_1",
) -> ClassifierExportBundle:
    table = sdata[table_name]
    feature_matrix = np.asarray(table.obsm[feature_key], dtype=np.float64)
    train_labels = np.where(feature_matrix[:, 0] > 0.5, 2, 1).astype(np.int64)
    classifier = RandomForestClassifier(n_estimators=25, random_state=0, n_jobs=1)
    classifier.fit(feature_matrix, train_labels)
    regions = _table_regions(table)
    classifier_config = {
        "model_type": "RandomForestClassifier",
        "feature_key": feature_key,
        "table_name": table_name,
        "roi_mode": "none",
        "trained": True,
        "eligible": True,
        "reason": "Ready to train.",
        "training_timestamp": "2026-05-05T09:00:00+00:00",
        "n_labeled_objects": int(table.n_obs),
        "n_features": int(feature_matrix.shape[1]),
        "class_labels_seen": [1, 2],
        "rf_params": {"n_estimators": 25, "random_state": 0, "n_jobs": 1},
        "training_scope": "all",
        "training_regions": list(regions),
        "n_training_rows": int(table.n_obs),
        "prediction_scope": "all",
        "prediction_regions": list(regions),
        "n_predicted_rows": int(table.n_obs),
    }
    return ClassifierExportBundle(
        schema_version=CLASSIFIER_EXPORT_SCHEMA_VERSION,
        created_at="2026-05-05T09:05:00+00:00",
        napari_harpy_version="0.0.0-test",
        sklearn_version=None,
        estimator=classifier,
        source_classifier_config=classifier_config,
        source_feature_metadata=dict(table.uns[_FEATURE_MATRICES_KEY][feature_key]),
    )


def _table_regions(table) -> tuple[str, ...]:
    regions = table.obs["region"]
    if isinstance(regions.dtype, pd.CategoricalDtype):
        return tuple(str(region) for region in regions.cat.categories)
    return tuple(str(region) for region in pd.unique(regions))


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


def test_apply_classifier_from_path_writes_predictions_and_apply_config(
    tmp_path: Path,
    sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    classifier_path = tmp_path / "classifier.harpy-classifier.joblib"
    write_classifier_export_bundle(classifier_path, bundle)

    result = headless.apply_classifier_from_path(sdata_blobs, classifier_path, table_name="table")

    table = sdata_blobs["table"]
    feature_matrix = np.asarray(table.obsm["features_1"], dtype=np.float64)
    expected_classes = bundle.estimator.predict(feature_matrix)
    expected_confidences = bundle.estimator.predict_proba(feature_matrix).max(axis=1)
    config = table.uns[CLASSIFIER_APPLY_CONFIG_KEY]

    assert result.table_name == "table"
    assert result.feature_key == "features_1"
    assert result.prediction_regions == ("blobs_labels",)
    assert result.n_predicted_rows == table.n_obs
    assert result.n_skipped_feature_invalid_rows == 0
    assert table.obs[PRED_CLASS_COLUMN].astype("string").astype(int).to_numpy().tolist() == expected_classes.tolist()
    assert np.allclose(table.obs[PRED_CONFIDENCE_COLUMN].to_numpy(dtype=np.float64), expected_confidences)
    assert config["applied"] is True
    assert config["classifier_path"] == str(classifier_path)
    assert config["source_classifier_config"] == bundle.source_classifier_config
    assert config["table_name"] == "table"
    assert config["feature_key"] == "features_1"
    assert config["prediction_regions"] == ["blobs_labels"]
    assert config["pred_class_column"] == PRED_CLASS_COLUMN
    assert config["pred_confidence_column"] == PRED_CONFIDENCE_COLUMN


def test_apply_classifier_bundle_can_write_custom_prediction_columns(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)

    result = headless.apply_classifier(
        sdata_blobs,
        bundle,
        table_name="table",
        pred_class_column="headless_class",
        pred_confidence_column="headless_confidence",
    )

    table = sdata_blobs["table"]
    assert result.pred_class_column == "headless_class"
    assert result.pred_confidence_column == "headless_confidence"
    assert "headless_class" in table.obs
    assert "headless_confidence" in table.obs
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["classifier_path"] is None
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["pred_class_column"] == "headless_class"


def test_apply_classifier_from_path_persists_backed_prediction_state(
    tmp_path: Path,
    backed_sdata_blobs: SpatialData,
) -> None:
    _set_deterministic_features(backed_sdata_blobs)
    _set_feature_metadata(backed_sdata_blobs)
    bundle = _make_classifier_bundle(backed_sdata_blobs)
    classifier_path = tmp_path / "classifier.harpy-classifier.joblib"
    write_classifier_export_bundle(classifier_path, bundle)

    result = headless.apply_classifier_from_path(
        backed_sdata_blobs,
        classifier_path,
        table_name="table",
        pred_class_column="headless_class",
        pred_confidence_column="headless_confidence",
    )

    table = backed_sdata_blobs["table"]
    reread = read_zarr(backed_sdata_blobs.path)
    disk_table = reread["table"]
    config = disk_table.uns[CLASSIFIER_APPLY_CONFIG_KEY]

    assert result.table_name == "table"
    assert "headless_class" in disk_table.obs
    assert "headless_confidence" in disk_table.obs
    assert "headless_class_colors" in disk_table.uns
    assert disk_table.obs["headless_class"].astype("string").tolist() == table.obs["headless_class"].astype(
        "string"
    ).tolist()
    assert np.allclose(
        disk_table.obs["headless_confidence"].to_numpy(dtype=np.float64),
        table.obs["headless_confidence"].to_numpy(dtype=np.float64),
    )
    assert config["classifier_path"] == str(classifier_path)
    assert config["table_name"] == "table"
    assert config["feature_key"] == "features_1"
    assert config["pred_class_column"] == "headless_class"
    assert config["pred_confidence_column"] == "headless_confidence"


def test_apply_classifier_rejects_missing_feature_key(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)

    with pytest.raises(ValueError, match="not available in `.obsm`"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table", feature_key="missing_features")


def test_apply_classifier_rejects_missing_feature_metadata(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    sdata_blobs["table"].uns.pop(_FEATURE_MATRICES_KEY)

    with pytest.raises(ValueError, match="missing Harpy metadata"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_feature_matrix_shape_drift(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    table = sdata_blobs["table"]
    table.obsm["features_1"] = np.column_stack(
        [np.asarray(table.obsm["features_1"], dtype=np.float64), np.zeros(table.n_obs)]
    )

    with pytest.raises(ValueError, match="metadata"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_rejects_feature_column_order_mismatch(sdata_blobs: SpatialData) -> None:
    _set_deterministic_features(sdata_blobs)
    _set_feature_metadata(sdata_blobs)
    bundle = _make_classifier_bundle(sdata_blobs)
    sdata_blobs["table"].uns[_FEATURE_MATRICES_KEY]["features_1"]["feature_columns"] = [
        "instance_fraction",
        "is_large",
    ]

    with pytest.raises(ValueError, match="do not match"):
        headless.apply_classifier(sdata_blobs, bundle, table_name="table")


def test_apply_classifier_selected_region_leaves_other_regions_unchanged(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs_multi_region, table_name="table_multi")
    _set_feature_metadata(sdata_blobs_multi_region, table_name="table_multi")
    bundle = _make_classifier_bundle(sdata_blobs_multi_region, table_name="table_multi")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[0, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")

    result = headless.apply_classifier(
        sdata_blobs_multi_region,
        bundle,
        table_name="table_multi",
        prediction_regions=["blobs_labels"],
    )

    region_values = table.obs["region"].astype("string")
    selected_rows = region_values == "blobs_labels"
    hidden_rows = region_values == "blobs_labels_2"

    assert result.prediction_regions == ("blobs_labels",)
    assert result.n_predicted_rows == int(selected_rows.sum())
    assert table.obs.loc[selected_rows, PRED_CLASS_COLUMN].astype("string").ne("9").all()
    assert table.obs.loc[selected_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[hidden_rows, PRED_CLASS_COLUMN].astype("string").eq("9").all()
    assert table.obs.loc[hidden_rows, PRED_CONFIDENCE_COLUMN].eq(0.55).all()


def test_apply_classifier_clears_invalid_rows_in_prediction_scope(
    sdata_blobs_multi_region: SpatialData,
) -> None:
    _set_deterministic_features(sdata_blobs_multi_region, table_name="table_multi")
    _set_feature_metadata(sdata_blobs_multi_region, table_name="table_multi")
    bundle = _make_classifier_bundle(sdata_blobs_multi_region, table_name="table_multi")
    _set_invalid_feature_rows_for_region(sdata_blobs_multi_region, region_name="blobs_labels_2")
    table = sdata_blobs_multi_region["table_multi"]
    table.obs[PRED_CLASS_COLUMN] = pd.Categorical(np.full(table.n_obs, 9, dtype=np.int64), categories=[0, 9])
    table.obs[PRED_CONFIDENCE_COLUMN] = pd.Series(np.full(table.n_obs, 0.55), index=table.obs.index, dtype="float64")

    result = headless.apply_classifier(sdata_blobs_multi_region, bundle, table_name="table_multi")

    region_values = table.obs["region"].astype("string")
    valid_rows = region_values == "blobs_labels"
    invalid_rows = region_values == "blobs_labels_2"

    assert result.prediction_regions == ("blobs_labels", "blobs_labels_2")
    assert result.n_predicted_rows == int(valid_rows.sum())
    assert result.n_skipped_feature_invalid_rows == int(invalid_rows.sum())
    assert table.obs.loc[valid_rows, PRED_CLASS_COLUMN].astype("string").ne("0").all()
    assert table.obs.loc[valid_rows, PRED_CONFIDENCE_COLUMN].between(0.0, 1.0).all()
    assert table.obs.loc[invalid_rows, PRED_CLASS_COLUMN].eq(0).all()
    assert table.obs.loc[invalid_rows, PRED_CONFIDENCE_COLUMN].isna().all()
    assert table.uns[CLASSIFIER_APPLY_CONFIG_KEY]["n_skipped_feature_invalid_rows"] == int(invalid_rows.sum())


def test_headless_module_avoids_direct_interactive_classifier_imports() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "napari_harpy" / "headless.py").read_text()
    import_modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            import_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            import_modules.add(node.module)

    assert "napari_harpy._classifier" not in import_modules
    assert not any(module.startswith("napari_harpy.widgets") for module in import_modules)
    assert "napari" not in import_modules
    assert "qtpy" not in import_modules
    assert "thread_worker" not in source
