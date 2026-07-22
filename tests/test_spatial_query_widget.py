from __future__ import annotations

import copy
from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import pandas as pd
import pytest
from qtpy.QtCore import Qt
from spatialdata import SpatialData

import napari_harpy.widgets.spatial_query.widget as widget_module
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


def _context(sdata: SpatialData, *, dirty: bool = False, create_new: bool = False) -> AnnotationContext:
    target = (
        ShapesAnnotationTarget.create_new() if create_new else ShapesAnnotationTarget.edit_existing("blobs_circles")
    )
    return AnnotationContext(
        sdata=sdata,
        coordinate_system="global",
        shapes_target=target,
        has_unsaved_shapes_changes=dirty,
    )


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
    assert widget.labels_combo.isEnabled() is False
    assert widget.table_combo.isEnabled() is False
    assert widget.run_button.isEnabled() is False
    assert "No SpatialData Loaded" in _status_text(widget.readiness_status_label)


def test_spatial_query_shell_derives_ready_new_column_state_and_emits_only_action_intents(
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

    assert widget.selected_labels_name == "blobs_labels"
    assert widget.selected_table_name == "table"
    assert widget.selected_column_mode == "new"
    assert widget.selected_column_name == "spatial_annotation"
    assert widget.cache_report is not None
    assert widget.run_button.isEnabled() is True
    assert inspection_calls == 1
    assert viewer.layers == []  # Programmatic context refresh does not claim viewer styling.

    widget.new_column_edit.setText("reviewed_annotation")
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

    assert widget.selected_column_mode == "existing"
    assert widget.selected_column_name == "spatial_annotation"
    assert viewer.layers == []

    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("new"))
    widget.column_mode_combo.setCurrentIndex(widget.column_mode_combo.findData("existing"))

    assert len(viewer.layers) == 1
    assert viewer.layers.selection.active is viewer.layers[0]
    assert viewer.layers[0].name == "blobs_labels"
    pd.testing.assert_frame_equal(table.obs, obs_before)
    assert table.uns == uns_before


def test_spatial_query_shell_blocks_colliding_default_until_new_name_is_valid(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs.tables["table"]
    table.obs["spatial_annotation"] = pd.Series(
        ["existing"] * table.n_obs,
        index=table.obs.index,
        dtype="string",
    )
    widget = SpatialQuery()
    qtbot.addWidget(widget)

    widget.apply_annotation_context(_context(sdata_blobs))

    assert widget.selected_column_mode == "new"
    assert widget.selected_column_name is None
    assert widget.run_button.isEnabled() is False
    assert "already exists" in _status_text(widget.readiness_status_label)

    widget.new_column_edit.setText("reviewed_annotation")

    assert widget.selected_column_name == "reviewed_annotation"
    assert widget.run_button.isEnabled() is True


def test_spatial_query_shell_reports_cache_inspection_failure_separately(
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

    assert widget.cache_report is None
    assert widget.run_button.isEnabled() is False
    assert "Centroid Inspection Error" in _status_text(widget.cache_status_label)
    assert "invalid current table binding" in _status_text(widget.cache_status_label)


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

    assert widget.cache_report is not None
    assert widget.run_button.isEnabled() is False
    assert expected_status in _status_text(widget.readiness_status_label)
