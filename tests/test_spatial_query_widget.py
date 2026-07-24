from __future__ import annotations

import copy
from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from qtpy.QtCore import Qt
from spatialdata import SpatialData
from spatialdata.transformations import Identity, set_transformation

import napari_harpy.widgets.spatial_query.widget as widget_module
from napari_harpy.core.spatial_query import CANONICAL_OBSM_KEY, CanonicalCacheState
from napari_harpy.viewer._styling import MISSING_CATEGORICAL_COLOR
from napari_harpy.widgets.annotation.models import AnnotationContext, ShapesAnnotationTarget
from napari_harpy.widgets.spatial_query.widget import SpatialQuery


class _EventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class _Selection:
    def __init__(self) -> None:
        self.active: object | None = None

    def select_only(self, layer: object) -> None:
        self.active = layer


class _Layers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = _Selection()
        self.events = SimpleNamespace(
            inserted=_EventEmitter(),
            removed=_EventEmitter(),
            reordered=_EventEmitter(),
        )


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


def _context(
    sdata: SpatialData,
    *,
    coordinate_system: str = "global",
    dirty: bool = False,
    create_new: bool = False,
) -> AnnotationContext:
    target = (
        ShapesAnnotationTarget.create_new() if create_new else ShapesAnnotationTarget.edit_existing("blobs_circles")
    )
    return AnnotationContext(
        sdata=sdata,
        coordinate_system=coordinate_system,
        shapes_target=target,
        has_unsaved_shapes_changes=dirty,
    )


def _select_labels(widget: SpatialQuery, labels_name: str = "blobs_labels") -> None:
    index = widget.labels_combo.findData(labels_name)
    assert index >= 0
    widget.labels_combo.setCurrentIndex(index)


def _add_default_annotation_column(sdata: SpatialData) -> None:
    table = sdata.tables["table"]
    table.obs["spatial_annotation"] = pd.Categorical(
        ["A"] * table.n_obs,
        categories=["A"],
    )


def _status_text(label) -> str:
    return unescape(label.text())


def test_spatial_query_shell_starts_inactive_without_parent_context(qtbot) -> None:
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    assert widget.selected_spatialdata is None
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.labels_combo.isEnabled() is False
    assert widget.table_combo.isEnabled() is False
    assert widget.run_button.isEnabled() is False
    assert widget.status_label.objectName() == "spatial_query_status_label"
    assert not hasattr(widget, "cache_status_label")
    assert not hasattr(widget, "readiness_status_label")
    assert "No SpatialData Loaded" in _status_text(widget.status_label)


def test_spatial_query_shell_requires_an_explicit_new_column_name_and_emits_only_action_intents(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    inspection_calls = 0
    real_inspect = widget_module.inspect_canonical_cache

    def record_inspection(*args, **kwargs):
        nonlocal inspection_calls
        inspection_calls += 1
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(widget_module, "inspect_canonical_cache", record_inspection)
    obs_before = sdata_blobs.tables["table"].obs.copy(deep=True)
    uns_before = copy.deepcopy(sdata_blobs.tables["table"].uns)

    widget.apply_annotation_context(_context(sdata_blobs))

    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.cache_report is None
    assert inspection_calls == 0
    assert viewer.layers == []

    _select_labels(widget)

    assert widget.selected_labels_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.cache_report is not None
    assert widget.run_button.isEnabled() is False
    assert "Enter a new annotation column name" in _status_text(widget.status_label)
    assert inspection_calls == 1
    assert len(viewer.layers) == 1  # Explicit labels selection may claim primary-label styling.
    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)
    assert np.allclose(viewer.layers[0].colormap.map(10_000), neutral_rgba)

    widget.new_column_edit.setText("reviewed_annotation")
    assert widget.selected_column_name == "reviewed_annotation"
    assert widget.run_button.isEnabled() is True
    assert inspection_calls == 1  # Status rendering reuses the captured report.

    intents: list[str] = []
    widget.run_requested.connect(lambda: intents.append("run"))
    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)

    assert intents == ["run"]
    pd.testing.assert_frame_equal(sdata_blobs.tables["table"].obs, obs_before)
    assert sdata_blobs.tables["table"].uns == uns_before


def test_spatial_query_shell_uses_compatible_default_and_styles_only_after_explicit_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    _add_default_annotation_column(sdata_blobs)
    table = sdata_blobs.tables["table"]
    table.uns["spatial_annotation_colors"] = ["#ff0000"]
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))

    assert widget.selected_labels_name is None
    assert viewer.layers == []

    _select_labels(widget)

    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active is viewer.layers[0]
    assert viewer.layers[0].name == "blobs_labels"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))

    assert widget.new_column_edit.text() == ""
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False
    assert "Enter a new annotation column name" in _status_text(widget.status_label)
    assert np.allclose(
        viewer.layers[0].colormap.map(1),
        np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32),
    )
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_query_shell_lists_user_class_but_excludes_classifier_outputs(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["user_class"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.uns["user_class_colors"] = ["#ff0000"]
    table.obs["pred_class"] = pd.Categorical([1] * table.n_obs, categories=[1])
    table.obs["pred_confidence"] = np.ones(table.n_obs, dtype=np.float64)
    obs_before = table.obs.copy(deep=True)
    uns_before = copy.deepcopy(table.uns)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    available_columns = [
        widget.existing_column_combo.itemData(index) for index in range(widget.existing_column_combo.count())
    ]
    assert "user_class" in available_columns
    assert "pred_class" not in available_columns
    assert "pred_confidence" not in available_columns

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))
    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("user_class"))

    assert widget.selected_column_name == "user_class"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_query_shell_rejects_reserved_object_classification_new_columns(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    for column_name in ("user_class", "pred_class", "pred_confidence"):
        widget.new_column_edit.setText(column_name)
        assert widget.selected_column_name is None
        assert widget.run_button.isEnabled() is False
        status_text = _status_text(widget.status_label)
        assert f'New annotation column "{column_name}" is reserved for Object Classification' in status_text


def test_spatial_query_shell_uses_named_default_when_preferred_existing_column_disappears(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    _add_default_annotation_column(sdata_blobs)
    table = sdata_blobs.tables["table"]
    table.uns["spatial_annotation_colors"] = ["#ff0000"]
    table.obs["old_annotation"] = pd.Categorical(
        ["old"] * table.n_obs,
        categories=["old"],
    )
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    old_index = widget.existing_column_combo.findData("old_annotation")
    assert old_index >= 0
    widget.existing_column_combo.setCurrentIndex(old_index)
    assert widget.selected_column_name == "old_annotation"
    widget.new_column_edit.setText("draft_annotation")

    del table.obs["old_annotation"]
    widget._refresh_columns(
        preferred_mode="existing",
        preferred_existing_column="old_annotation",
        preferred_new_column=widget.new_column_edit.text(),
    )

    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert widget.new_column_edit.text() == "draft_annotation"

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))

    assert widget.existing_column_combo.currentIndex() == -1
    assert widget.existing_column_combo.placeholderText() == "Choose an existing column"
    assert widget.new_column_edit.text() == "draft_annotation"
    assert widget.selected_column_name == "draft_annotation"
    neutral_rgba = np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32)
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))

    assert widget.existing_column_combo.currentIndex() == -1
    assert widget.selected_column_name is None
    assert widget.new_column_edit.text() == "draft_annotation"
    assert np.allclose(viewer.layers[0].colormap.map(1), neutral_rgba)

    widget.existing_column_combo.setCurrentIndex(widget.existing_column_combo.findData("spatial_annotation"))

    assert widget.selected_column_name == "spatial_annotation"
    assert np.allclose(viewer.layers[0].colormap.map(1), np.asarray(to_rgba("#ff0000"), dtype=np.float32))


def test_spatial_query_shell_explains_incompatible_preferred_column_with_empty_new_draft(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["spatial_annotation"] = pd.Series(
        ["existing"] * table.n_obs,
        index=table.obs.index,
        dtype="string",
    )
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False
    status_text = _status_text(widget.status_label)
    assert 'Existing annotation column "spatial_annotation" cannot be used' in status_text
    assert "categorical column containing only strings or positive integers" in status_text
    assert "different New-column name" in status_text
    assert len(viewer.layers) == 1
    assert np.allclose(
        viewer.layers[0].colormap.map(1),
        np.asarray(to_rgba(MISSING_CATEGORICAL_COLOR), dtype=np.float32),
    )

    widget.new_column_edit.setText("reviewed_annotation")

    assert widget.selected_column_name == "reviewed_annotation"
    assert widget.run_button.isEnabled() is True


def test_spatial_query_shell_clears_new_column_draft_when_table_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    sdata_blobs.tables["second_table"] = sdata_blobs.tables["table"].copy()
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    widget.new_column_edit.setText("draft_annotation")
    assert widget.selected_column_name == "draft_annotation"
    assert widget.run_button.isEnabled() is True

    next_table = "table" if widget.selected_table_name == "second_table" else "second_table"
    next_table_index = widget.table_combo.findData(next_table)
    assert next_table_index >= 0
    widget.table_combo.setCurrentIndex(next_table_index)

    assert widget.selected_table_name == next_table
    assert widget.selected_column_mode == "new"
    assert widget.new_column_edit.text() == ""
    assert widget.new_column_edit.placeholderText() == "spatial_annotation"
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False


def test_spatial_query_shell_blocks_live_input_inspection_failure(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    def reject_inspection(*args, **kwargs):
        raise ValueError("invalid current table binding")

    monkeypatch.setattr(widget_module, "inspect_canonical_cache", reject_inspection)
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)

    assert widget.cache_report is None
    assert widget.run_button.isEnabled() is False
    status_text = _status_text(widget.status_label)
    assert "Labels or Table Validation Failed" in status_text
    assert "invalid current table binding" in status_text
    assert "cannot calculate centers until this issue is resolved" in status_text


def test_spatial_query_shell_keeps_invalid_cache_ready_for_recalculation(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obsm[CANONICAL_OBSM_KEY] = np.zeros((table.n_obs, 3), dtype=np.float64)
    widget = SpatialQuery(_Viewer())
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")

    report = widget.cache_report
    assert report is not None
    assert report.state is CanonicalCacheState.INVALID
    assert widget.run_button.isEnabled() is True
    status_text = _status_text(widget.status_label)
    assert "Spatial Query Ready" in status_text
    assert 'Centers for labels element "blobs_labels" will be recalculated' in status_text
    assert "Detected:" not in status_text


@pytest.mark.parametrize(
    ("dirty", "create_new", "expected_status"),
    [
        (True, False, "Save or Discard Shapes Changes"),
        (False, True, "Saved Shapes Required"),
    ],
)
def test_spatial_query_shell_shapes_context_blocks_run(
    qtbot,
    sdata_blobs: SpatialData,
    dirty: bool,
    create_new: bool,
    expected_status: str,
) -> None:
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs, dirty=dirty, create_new=create_new))
    _select_labels(widget)

    assert widget.cache_report is not None
    assert widget.run_button.isEnabled() is False
    assert expected_status in _status_text(widget.status_label)


def test_spatial_query_shell_coordinate_change_clears_valid_labels_without_reinspection(
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    inspection_calls = 0
    real_inspect = widget_module.inspect_canonical_cache

    def record_inspection(*args, **kwargs):
        nonlocal inspection_calls
        inspection_calls += 1
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(widget_module, "inspect_canonical_cache", record_inspection)
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))
    _select_labels(widget)
    assert inspection_calls == 1
    assert len(viewer.layers) == 1

    set_transformation(
        sdata_blobs.labels["blobs_labels"],
        Identity(),
        to_coordinate_system="shared",
    )
    widget.apply_annotation_context(_context(sdata_blobs, coordinate_system="shared"))

    assert widget.labels_combo.isEnabled() is True
    assert widget.labels_combo.currentIndex() == -1
    assert widget.labels_combo.currentText() == ""
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.cache_report is None
    assert widget.run_button.isEnabled() is False
    assert inspection_calls == 1


def test_spatial_query_shell_tracks_only_its_selected_primary_labels_layer(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = _Viewer()
    widget = SpatialQuery(viewer)
    qtbot.addWidget(widget)
    widget.apply_annotation_context(_context(sdata_blobs))

    unrelated_result = widget.app_state.viewer_adapter.ensure_labels_loaded(
        sdata_blobs,
        "blobs_multiscale_labels",
        "global",
    )
    assert widget.selected_labels_name is None  # Layer insertion never selects it.

    _select_labels(widget)
    widget.new_column_edit.setText("reviewed_annotation")
    selected_layer = widget.app_state.viewer_adapter.get_loaded_primary_labels_layer(
        sdata_blobs,
        "blobs_labels",
        "global",
    )
    assert selected_layer is not None
    cache_report = widget.cache_report
    assert cache_report is not None

    viewer.layers.remove(unrelated_result.layer)
    viewer.layers.events.removed.emit(unrelated_result.layer)

    assert widget.selected_labels_name == "blobs_labels"
    assert widget.cache_report is cache_report
    assert widget.run_button.isEnabled() is True

    viewer.layers.remove(selected_layer)
    viewer.layers.events.removed.emit(selected_layer)

    assert widget.labels_combo.currentIndex() == -1
    assert widget.labels_combo.placeholderText() == "Choose a labels element"
    assert widget.selected_labels_name is None
    assert widget.selected_table_name is None
    assert widget.cache_report is None
    assert widget.run_button.isEnabled() is False
