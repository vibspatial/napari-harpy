from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.utils.colormaps import DirectLabelColormap
from spatialdata import SpatialData

import napari_harpy.widgets.object_classification.viewer_styling as viewer_styling_module
from napari_harpy.core.annotation import (
    UNLABELED_COLOR,
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
)
from napari_harpy.viewer.labels_colormap import CompactCategoricalLabelColormap
from napari_harpy.widgets.object_classification.annotation_controller import UserClassAnnotationChange
from napari_harpy.widgets.object_classification.controller import (
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy.widgets.object_classification.viewer_styling import (
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
    ViewerStylingController,
)


class _FakeLabelsLayer:
    def __init__(self) -> None:
        self.colormap: DirectLabelColormap | None = None
        self.features = pd.DataFrame()
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class _FakeViewerAdapter:
    def __init__(self, layer: _FakeLabelsLayer) -> None:
        self._layer = layer

    def get_loaded_primary_labels_layer(self, *args: Any, **kwargs: Any) -> _FakeLabelsLayer:
        del args, kwargs
        return self._layer


def _make_controller(sdata: SpatialData, layer: _FakeLabelsLayer) -> ViewerStylingController:
    controller = ViewerStylingController(_FakeViewerAdapter(layer))
    controller.bind(sdata, "blobs_labels", "table")
    return controller


def _feature_rows(
    user_class_by_instance: dict[int, int],
    *,
    pred_class_by_instance: dict[int, int] | None = None,
    pred_confidence_by_instance: dict[int, float] | None = None,
) -> pd.DataFrame:
    instance_ids = sorted(user_class_by_instance)
    pred_class_by_instance = pred_class_by_instance or {}
    pred_confidence_by_instance = pred_confidence_by_instance or {}
    return pd.DataFrame(
        {
            USER_CLASS_COLUMN: [user_class_by_instance[instance_id] for instance_id in instance_ids],
            PRED_CLASS_COLUMN: [pred_class_by_instance.get(instance_id, 0) for instance_id in instance_ids],
            PRED_CONFIDENCE_COLUMN: [
                pred_confidence_by_instance.get(instance_id, np.nan) for instance_id in instance_ids
            ],
        },
        index=pd.Index(instance_ids, name="index", dtype="int64"),
    )


def _set_user_classes(
    sdata: SpatialData,
    class_by_instance: dict[int, int],
    *,
    categories: list[int],
    colors: list[str] | None = None,
) -> None:
    table = sdata["table"]
    values = [
        class_by_instance.get(int(instance_id), 0)
        if str(region) == "blobs_labels"
        else 0
        for region, instance_id in zip(table.obs["region"], table.obs["instance_id"], strict=True)
    ]
    table.obs[USER_CLASS_COLUMN] = pd.Series(
        pd.Categorical(values, categories=categories),
        index=table.obs.index,
        name=USER_CLASS_COLUMN,
    )
    table.uns[USER_CLASS_COLORS_KEY] = colors or [UNLABELED_COLOR, "#ff0000"][: len(categories)]


def _set_pred_classes(
    sdata: SpatialData,
    class_by_instance: dict[int, int],
    *,
    categories: list[int],
    colors: list[str] | None = None,
) -> None:
    table = sdata["table"]
    values = [
        class_by_instance.get(int(instance_id), 0)
        if str(region) == "blobs_labels"
        else 0
        for region, instance_id in zip(table.obs["region"], table.obs["instance_id"], strict=True)
    ]
    table.obs[PRED_CLASS_COLUMN] = pd.Series(
        pd.Categorical(values, categories=categories),
        index=table.obs.index,
        name=PRED_CLASS_COLUMN,
    )
    table.uns[PRED_CLASS_COLORS_KEY] = colors or [UNLABELED_COLOR, "#00ff00"][: len(categories)]


def _expected_rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=np.float32)


def test_refresh_reuses_one_region_feature_snapshot(monkeypatch: pytest.MonkeyPatch, sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4, 6: 0})
    calls = 0

    def fake_feature_rows() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return feature_rows.copy()

    monkeypatch.setattr(controller, "_get_region_feature_rows", fake_feature_rows)

    controller.refresh()

    assert calls == 1
    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    assert len(layer.colormap.color_dict) <= 3
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba("#ff0000"))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(UNLABELED_COLOR))
    assert layer.refresh_count == 0
    assert USER_CLASS_COLUMN in layer.features.columns


def test_user_class_color_lookup_uses_valid_categorical_without_full_normalization(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4, 6: 0})

    def fail_normalization(*args: Any, **kwargs: Any) -> pd.Series:
        del args, kwargs
        raise AssertionError("valid categorical user_class should not be normalized")

    monkeypatch.setattr(viewer_styling_module, "normalize_class_values", fail_normalization)

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba("#ff0000"))
    np.testing.assert_allclose(layer.colormap.map(6), _expected_rgba(UNLABELED_COLOR))


def test_pred_class_color_lookup_uses_valid_categorical_without_full_normalization(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_pred_classes(sdata_blobs, {5: 2, 6: 2}, categories=[0, 2], colors=[UNLABELED_COLOR, "#00ff00"])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: 0, 5: 0, 6: 0},
        pred_class_by_instance={1: 0, 5: 2, 6: 2},
    )

    def fail_normalization(*args: Any, **kwargs: Any) -> pd.Series:
        del args, kwargs
        raise AssertionError("valid categorical pred_class should not be normalized")

    monkeypatch.setattr(viewer_styling_module, "normalize_class_values", fail_normalization)

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(UNLABELED_COLOR))
    np.testing.assert_allclose(layer.colormap.map(5), _expected_rgba("#00ff00"))


def test_class_value_reader_uses_vectorized_integer_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    values = pd.Series(np.asarray([0, 4, 4, 2], dtype=np.int64))

    def fail_scalar_missing_check(*args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        raise AssertionError("integer class ids should use the vectorized path")

    monkeypatch.setattr(viewer_styling_module.pd, "isna", fail_scalar_missing_check)

    assert viewer_styling_module._read_class_values_without_normalizing(values, unlabeled_class=0) == {0, 2, 4}


def test_user_class_color_lookup_falls_back_for_invalid_table_state(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"]
    table.obs[USER_CLASS_COLUMN] = pd.Series(0, index=table.obs.index, name=USER_CLASS_COLUMN, dtype="int64")
    table.uns[USER_CLASS_COLORS_KEY] = [UNLABELED_COLOR]
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4})
    original_normalize = viewer_styling_module.normalize_class_values
    normalized_columns: list[str] = []

    def record_normalization(values: pd.Series, *, column_name: str, unlabeled_class: int = 0) -> pd.Series:
        normalized_columns.append(column_name)
        return original_normalize(values, column_name=column_name, unlabeled_class=unlabeled_class)

    monkeypatch.setattr(viewer_styling_module, "normalize_class_values", record_normalization)

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert USER_CLASS_COLUMN in normalized_columns
    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(UNLABELED_COLOR))
    assert not np.allclose(layer.colormap.map(5), layer.colormap.map(0))


def test_refresh_layer_methods_still_work_without_precomputed_feature_rows(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4})
    calls = 0

    def fake_feature_rows() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return feature_rows.copy()

    monkeypatch.setattr(controller, "_get_region_feature_rows", fake_feature_rows)

    controller.refresh_layer_colors()
    controller.refresh_layer_features()

    assert calls == 2
    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    assert USER_CLASS_COLUMN in layer.features.columns


def test_pred_class_coloring_keeps_explicit_entries_for_unlabeled_predictions(sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {}, categories=[0])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: 0, 5: 0, 6: 0},
        pred_class_by_instance={1: 0, 5: 2, 6: 2},
    )

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, CompactCategoricalLabelColormap)
    assert len(layer.colormap.color_dict) <= 3
    np.testing.assert_allclose(layer.colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(layer.colormap.map(1), _expected_rgba(UNLABELED_COLOR))
    np.testing.assert_allclose(layer.colormap.map(5), layer.colormap.map(6))
    assert layer.refresh_count == 0
    assert not np.allclose(layer.colormap.map(1), layer.colormap.map(0))


def test_pred_confidence_coloring_uses_vectorized_numeric_rgba_without_explicit_refresh(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    """Verify pred_confidence uses one vectorized real-colormap call and no explicit refresh."""
    class _RecordingColormap:
        def __init__(self, colormap: Any) -> None:
            self._colormap = colormap
            self.calls: list[np.ndarray] = []

        def __call__(self, values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            self.calls.append(values.copy())
            return self._colormap(values)

    _set_user_classes(sdata_blobs, {}, categories=[0])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CONFIDENCE)
    feature_rows = _feature_rows(
        {1: 0, 5: 0, 6: 0},
        pred_confidence_by_instance={1: 0.0, 5: 1.0},
    )
    real_colormap = viewer_styling_module.colormaps[viewer_styling_module.PRED_CONFIDENCE_COLORMAP]
    recording_colormap = _RecordingColormap(real_colormap)
    monkeypatch.setattr(
        viewer_styling_module,
        "colormaps",
        {viewer_styling_module.PRED_CONFIDENCE_COLORMAP: recording_colormap},
    )

    controller.refresh_layer_colors(feature_rows=feature_rows)

    assert isinstance(layer.colormap, DirectLabelColormap)
    assert set(layer.colormap.color_dict) == {None, 0, 1, 5, 6}
    assert isinstance(layer.colormap.color_dict[None], np.ndarray)
    assert isinstance(layer.colormap.color_dict[0], np.ndarray)
    assert isinstance(layer.colormap.color_dict[1], np.ndarray)
    assert isinstance(layer.colormap.color_dict[6], np.ndarray)
    assert len(recording_colormap.calls) == 1
    np.testing.assert_allclose(recording_colormap.calls[0], np.asarray([0.0, 1.0, 0.0]))
    expected_colors = np.asarray(real_colormap(np.asarray([0.0, 1.0])), dtype=np.float32)
    np.testing.assert_allclose(
        layer.colormap.color_dict[1],
        expected_colors[0],
    )
    np.testing.assert_allclose(
        layer.colormap.color_dict[5],
        expected_colors[1],
    )
    assert layer.colormap.color_dict[6].shape == (4,)
    assert not np.allclose(layer.colormap.color_dict[6], layer.colormap.color_dict[1])
    assert layer.refresh_count == 0


def test_row_scoped_user_class_annotation_defers_compact_colormap_to_full_refresh(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4, 6: 0})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    original_features = layer.features.copy()

    handled = controller.refresh_user_class_colormap_and_feature(UserClassAnnotationChange(instance_id=6, class_id=4))

    assert handled is False
    assert layer.colormap is original_colormap
    pd.testing.assert_frame_equal(layer.features, original_features)


def test_row_scoped_user_class_annotation_clear_defers_compact_colormap_to_full_refresh(
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4, 6: 0})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    original_features = layer.features.copy()

    handled = controller.refresh_user_class_colormap_and_feature(UserClassAnnotationChange(instance_id=5, class_id=0))

    assert handled is False
    assert layer.colormap is original_colormap
    pd.testing.assert_frame_equal(layer.features, original_features)


def test_row_scoped_user_class_feature_refresh_keeps_prediction_colormap(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    controller.set_color_by(COLOR_BY_PRED_CLASS)
    feature_rows = _feature_rows(
        {1: 0, 5: 4, 6: 0},
        pred_class_by_instance={1: 0, 5: 2, 6: 2},
    )
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_colormap = layer.colormap
    original_refresh_count = layer.refresh_count

    def fail_full_feature_rows() -> pd.DataFrame:
        raise AssertionError("feature-only annotation refresh must not rebuild all feature rows")

    monkeypatch.setattr(controller, "_get_region_feature_rows", fail_full_feature_rows)

    handled = controller.refresh_user_class_feature(UserClassAnnotationChange(instance_id=6, class_id=4))

    assert handled is True
    assert layer.colormap is original_colormap
    assert layer.refresh_count == original_refresh_count
    assert layer.features.set_index("index").loc[6, USER_CLASS_COLUMN] == 4
    assert layer.features.set_index("index").loc[1, USER_CLASS_COLUMN] == 0


def test_row_scoped_user_class_annotation_returns_false_for_missing_feature_row(sdata_blobs: SpatialData) -> None:
    _set_user_classes(sdata_blobs, {5: 4}, categories=[0, 4])
    layer = _FakeLabelsLayer()
    controller = _make_controller(sdata_blobs, layer)
    feature_rows = _feature_rows({1: 0, 5: 4, 6: 0})
    controller.refresh_layer_colors(feature_rows=feature_rows)
    controller.refresh_layer_features(feature_rows=feature_rows)
    original_color_keys = set(layer.colormap.color_dict)
    original_features = layer.features.copy()

    handled = controller.refresh_user_class_colormap_and_feature(UserClassAnnotationChange(instance_id=99, class_id=4))

    assert handled is False
    assert set(layer.colormap.color_dict) == original_color_keys
    pd.testing.assert_frame_equal(layer.features, original_features)
