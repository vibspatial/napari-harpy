from __future__ import annotations

from collections.abc import Callable
from html import unescape
from types import SimpleNamespace

import dask
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.layers import Image, Shapes
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QComboBox
from shapely.geometry import LineString, Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

import napari_harpy._app_state as app_state_module
import napari_harpy.widgets.viewer.widget as viewer_widget_module
from napari_harpy._app_state import (
    ClassificationTableWrittenEvent,
    FeatureMatrixWrittenEvent,
    ShapesElementWrittenEvent,
)
from napari_harpy._points_value_index import PointsValueSelection, PointsValueTable
from napari_harpy.core._color_source import ShapeColumnColorSourceSpec, TableColorSourceSpec
from napari_harpy.viewer.adapter import PointsLayerIdentity
from napari_harpy.viewer.shapes_styling import SHAPES_FACE_ALPHA
from napari_harpy.widgets.shared_styles import (
    STATUS_CARD_PALETTE,
    WIDGET_MIN_WIDTH,
    CompactComboBox,
)
from napari_harpy.widgets.viewer.disclosure import _ElidedLabel, _ElidedToolButton
from napari_harpy.widgets.viewer.image_widget import QColorDialog, _OverlayColorButton
from napari_harpy.widgets.viewer.points_controller import PointsLoadRequest
from napari_harpy.widgets.viewer.shapes_widget import ShapesLoadRequest
from napari_harpy.widgets.viewer.widget import ViewerWidget


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self) -> None:
        super().__init__()
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer


class DummyViewer:
    def __init__(self) -> None:
        self.layers = DummyLayers()


_FEEDBACK_BACKGROUND_BY_KIND = {
    kind: palette["background"] for kind, palette in STATUS_CARD_PALETTE.items()
}


def _assert_action_feedback_card(widget: ViewerWidget, *, title: str, kind: str) -> None:
    assert title in widget.global_action_feedback_label.text()
    assert f"background-color: {_FEEDBACK_BACKGROUND_BY_KIND[kind]}" in widget.global_action_feedback_label.styleSheet()
    assert not widget.global_action_feedback_label.isHidden()


def _label_text(label) -> str:
    return unescape(label.text())


def _tooltip_text(label) -> str:
    return unescape(label.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def _patch_coordinate_system_names(monkeypatch, coordinate_systems: list[str]) -> None:
    monkeypatch.setattr(
        viewer_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )
    monkeypatch.setattr(
        app_state_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: list(coordinate_systems),
    )


def _combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def _patch_viewer_widget_labels_tables(
    monkeypatch,
    *,
    labels_names: list[str],
    table_names_by_label: dict[str, list[str]],
    color_sources_by_table: dict[str, list[TableColorSourceSpec]] | None = None,
) -> None:
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: list(labels_names),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: list(table_names_by_label.get(labels_name, [])),
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: list((color_sources_by_table or {}).get(table_name, [])),
    )


def _make_shapes_sdata(geodataframe: gpd.GeoDataFrame, shapes_name: str = "cells") -> SimpleNamespace:
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SimpleNamespace(shapes={shapes_name: shapes}, tables={})


def _points_dataframe(data: dict[str, object]) -> dd.DataFrame:
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(pd.DataFrame(data), npartitions=1)


def _make_points_sdata(points_name: str = "transcripts") -> SimpleNamespace:
    return SimpleNamespace(
        points={
            points_name: _points_dataframe(
                {
                    "x": [0.0, 1.0, 2.0],
                    "y": [3.0, 4.0, 5.0],
                    "gene": ["AAMP", "AXL", "MALAT1"],
                    "target": ["T1", "T2", "T3"],
                    "score": [0.1, 0.2, 0.3],
                }
            )
        }
    )


def _make_colorable_shapes_sdata(
    *,
    shapes_name: str = "cells",
    cell_type_colors: list[str] | None = None,
    duplicate_index: bool = False,
    include_unsupported_geometry: bool = False,
) -> SimpleNamespace:
    geometries = [
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
        Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
    ]
    if include_unsupported_geometry:
        geometries[1] = LineString([(5, 0), (9, 4)])

    data: dict[str, object] = {
        "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
        "score": [0.0, 1.0],
        "free_text": ["alpha", "beta"],
    }
    if cell_type_colors is not None:
        data["cell_type_colors"] = cell_type_colors

    index = ["cell_1", "cell_1"] if duplicate_index else ["cell_1", "cell_2"]
    geodataframe = gpd.GeoDataFrame(data, geometry=geometries, index=index)
    return _make_shapes_sdata(geodataframe, shapes_name=shapes_name)


def _select_shape_column(card: object, value_key: str) -> None:
    card.color_source_kind_combo.setCurrentIndex(1)
    card.color_source_value_input.setText(value_key)


def _make_points_load_request(sdata: object) -> PointsLoadRequest:
    value_table = PointsValueTable(
        values=pd.DataFrame(
            {
                "value_id": pd.Series([0], dtype="uint32"),
                "value": ["AAMP"],
                "n_points": pd.Series([2], dtype="uint64"),
            }
        ),
        index_column="gene",
        total_count=2,
    )
    selection = PointsValueSelection(
        coordinates=np.asarray([[3.0, 0.0], [4.0, 1.0]], dtype="float32"),
        features=pd.DataFrame(
            {
                "gene": pd.Categorical(["AAMP", "AAMP"], categories=["AAMP"]),
                "value_id": pd.Series([0, 0], dtype="uint32"),
            }
        ),
        index_column="gene",
        selected_values=("AAMP",),
        selected_value_ids=(0,),
        selection_mode="values",
        total_count=2,
        render_point_budget=100_000,
        is_sampled=False,
        warning=None,
    )
    return PointsLoadRequest(
        identity=PointsLayerIdentity(
            sdata=sdata,
            points_name="transcripts",
            coordinate_system="global",
            index_column="gene",
        ),
        selection=selection,
        value_table=value_table,
    )


def test_viewer_widget_can_be_instantiated(qtbot) -> None:
    widget = ViewerWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget._logo_path.is_file()
    assert widget.app_state.sdata is None
    assert not widget.empty_state_label.isHidden()
    assert "No SpatialData Loaded" in _label_text(widget.summary_label)
    assert "No SpatialData loaded." in _label_text(widget.summary_label)
    assert widget.coordinate_system_combo.count() == 0
    assert not widget.coordinate_system_combo.isEnabled()
    assert isinstance(widget.coordinate_system_combo, CompactComboBox)
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.image_cards == []
    assert widget.labels_cards == []
    assert widget.shape_cards == []


def test_elided_label_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    label = _ElidedLabel("blobs_multiscale_image")

    qtbot.addWidget(label)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_multiscale…"

    monkeypatch.setattr(label, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(400))
    label._update_elided_text()

    assert label.toolTip() == ""

    monkeypatch.setattr(label, "contentsRect", lambda: _FakeRect(10))
    label._update_elided_text()

    tooltip = unescape(label.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_multiscale_image" in tooltip
    assert "..." in label.text() or "\u2026" in label.text()


def test_elided_tool_button_only_shows_tooltip_when_text_is_truncated(qtbot, monkeypatch) -> None:
    button = _ElidedToolButton("blobs_image_long_name_blobs_image_long_name")

    qtbot.addWidget(button)

    class _FakeRect:
        def __init__(self, width: int) -> None:
            self._width = width

        def width(self) -> int:
            return self._width

    class _FakeFontMetrics:
        def elidedText(self, text: str, mode: object, width: int) -> str:
            del mode
            return text if width >= len(text) else "blobs_image..."

    monkeypatch.setattr(button, "fontMetrics", lambda: _FakeFontMetrics())
    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(400))
    button.refresh_elision()

    assert button.toolTip() == ""

    monkeypatch.setattr(button, "contentsRect", lambda: _FakeRect(20))
    button.refresh_elision()

    tooltip = unescape(button.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "blobs_image_long_name_blobs_image_long_name" in tooltip
    assert "collapsed" not in tooltip
    assert "..." in button.text() or "\u2026" in button.text()


def test_overlay_color_button_uses_color_dialog_selection(qtbot, monkeypatch) -> None:
    button = _OverlayColorButton("#00FFFF")

    qtbot.addWidget(button)

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        lambda *args, **kwargs: QColor("#123456"),
    )

    button.choose_color()

    assert button.current_color == "#123456"
    assert "background-color: #123456" in button.styleSheet()
    assert "Current color" in button.toolTip()


def test_viewer_widget_refreshes_cards_when_shared_sdata_changes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    assert widget.app_state.sdata is sdata_blobs
    assert widget.empty_state_label.isHidden()
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert len(widget.image_cards) == 2
    assert len(widget.labels_cards) == 2
    assert len(widget.shape_cards) == 3
    assert [card.image_name for card in widget.image_cards] == ["blobs_image", "blobs_multiscale_image"]
    assert [card.labels_name for card in widget.labels_cards] == ["blobs_labels", "blobs_multiscale_labels"]
    assert [card.shapes_name for card in widget.shape_cards] == [
        "blobs_circles",
        "blobs_multipolygons",
        "blobs_polygons",
    ]
    assert widget.image_cards[0].channel_names == ["0", "1", "2"]
    assert widget.image_cards[0].stack_toggle.text() == "stack"
    assert widget.image_cards[0].stack_toggle.isChecked()
    assert widget.image_cards[0].overlay_toggle.text() == "overlay"
    assert not widget.image_cards[0].overlay_toggle.isChecked()
    assert widget.image_cards[0].channel_color_buttons[0].current_color == "#00FFFF"
    assert "background-color: #00FFFF" in widget.image_cards[0].channel_color_buttons[0].styleSheet()
    assert "Cyan" in widget.image_cards[0].channel_color_buttons[0].toolTip()
    assert len(widget.image_rows) == 2
    assert len(widget.labels_rows) == 2
    assert len(widget.shape_rows) == 3
    assert widget.images_section_toggle.text() == "Images (2)"
    assert widget.labels_section_toggle.text() == "Labels (2)"
    assert widget.shapes_section_toggle.text() == "Shapes (3)"
    assert not widget.images_group.is_expanded()
    assert not widget.labels_group.is_expanded()
    assert not widget.shapes_group.is_expanded()
    assert widget.image_rows[0].detail_widget.isHidden()
    assert widget.labels_rows[0].detail_widget.isHidden()
    assert widget.shape_rows[0].detail_widget.isHidden()
    assert widget.shape_cards[0].action_hint_label.text() == "Action: add/update primary shapes layer"
    assert widget.shape_cards[0].add_update_button.isEnabled()
    assert widget.labels_cards[0].linked_table_combo.count() == 1
    assert widget.labels_cards[0].linked_table_combo.itemText(0) == "table"
    assert widget.labels_cards[0].linked_table_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    assert widget.labels_cards[1].linked_table_combo.count() == 1
    assert widget.labels_cards[1].linked_table_combo.itemText(0) == "No linked tables"
    assert not widget.labels_cards[1].linked_table_combo.isEnabled()
    assert "Current View" in _label_text(widget.summary_label)
    assert '"global":' in _label_text(widget.summary_label)
    assert widget.summary_label.toolTip() == ""


def test_viewer_widget_summary_card_shortens_long_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs,
) -> None:
    coordinate_system = "global_long_coordinate_system_name_" + "x" * 80
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    qtbot.addWidget(widget)
    _patch_coordinate_system_names(monkeypatch, [coordinate_system])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    summary = _label_text(widget.summary_label)
    assert "Current View" in summary
    assert coordinate_system not in summary
    assert "…" in summary
    assert "0 image element(s)" in summary
    tooltip = _tooltip_text(widget.summary_label)
    assert coordinate_system in tooltip
    assert 'In coordinate system "' in tooltip


def test_viewer_widget_points_section_populates_and_starts_value_loading(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = _make_points_sdata()
    load_value_calls = 0

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: ["transcripts"]
    )

    def record_value_loading() -> bool:
        nonlocal load_value_calls
        load_value_calls += 1
        return True

    monkeypatch.setattr(widget._points_controller, "load_value_source", record_value_loading)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.points_section_toggle.text() == "Points (1)"
    assert widget.points_empty_label.isHidden()
    assert not widget.points_widget.isHidden()
    assert widget.points_widget.selected_points_name() == "transcripts"
    assert widget.points_widget.selected_index_column() == "gene"
    assert [
        widget.points_widget.index_column_combo.itemText(index)
        for index in range(widget.points_widget.index_column_combo.count())
    ] == ["gene", "target"]
    assert load_value_calls == 1


def test_viewer_widget_points_add_update_request_calls_controller(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_requests: list[tuple[object, int]] = []

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        widget._points_controller,
        "load_selection",
        lambda values, *, render_point_budget, random_state=42: (
            recorded_requests.append((values, render_point_budget)) or True
        ),
    )
    widget.points_widget.set_points_names(["transcripts"])
    widget.points_widget.set_index_columns(["gene"])
    widget.points_widget.set_value_source(
        SimpleNamespace(value_table=SimpleNamespace(values=pd.DataFrame({"value": ["AAMP", "AXL"]})))
    )
    widget.points_widget.render_controller_state(
        SimpleNamespace(
            can_load_values=True,
            can_visualize=True,
            is_loading=False,
            is_loading_values=False,
            status_message="Points: ready.",
            status_kind="success",
        )
    )
    widget.points_widget.value_input.setText("AAMP")
    widget.points_widget.add_value_button.click()
    widget.points_widget.value_input.setText("AXL")
    widget.points_widget.add_value_button.click()
    widget.points_widget.render_point_budget_input.setText("50_000")

    widget.points_widget.add_update_button.click()

    assert recorded_requests == [(("AAMP", "AXL"), 50_000)]


def test_viewer_widget_on_points_loaded_applies_layer_and_status(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    load_request = _make_points_load_request(fake_sdata)

    qtbot.addWidget(widget)

    widget._on_points_loaded(load_request)

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "transcripts: gene=AAMP"
    assert viewer.layers.selection.active is layer
    assert "Points Layer Created" in widget.global_action_feedback_label.text()
    assert "2 point" in widget.global_action_feedback_label.text()
    assert not widget.global_action_feedback_label.isHidden()


def test_viewer_widget_progressive_disclosure_expands_sections_and_elements(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_image_row = widget.image_rows[0]
    second_image_row = widget.image_rows[1]
    first_labels_row = widget.labels_rows[0]
    first_shape_row = widget.shape_rows[0]

    assert widget.images_group.content_widget.isHidden()
    assert widget.labels_group.content_widget.isHidden()
    assert widget.shapes_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()
    assert first_labels_row.detail_widget.isHidden()
    assert first_shape_row.detail_widget.isHidden()
    assert widget.images_section_toggle.arrowType() == Qt.ArrowType.NoArrow
    assert not widget.images_section_toggle.icon().isNull()

    widget.images_section_toggle.click()

    assert widget.images_group.is_expanded()
    assert not widget.images_group.content_widget.isHidden()
    assert first_image_row.detail_widget.isHidden()

    first_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert widget.image_cards[0].stack_toggle.isChecked()

    second_image_row.toggle_button.click()

    assert first_image_row.is_expanded()
    assert not first_image_row.detail_widget.isHidden()
    assert second_image_row.is_expanded()
    assert not second_image_row.detail_widget.isHidden()

    widget.labels_section_toggle.click()
    first_labels_row.toggle_button.click()

    assert widget.labels_group.is_expanded()
    assert first_labels_row.is_expanded()
    assert not first_labels_row.detail_widget.isHidden()
    assert widget.labels_cards[0].linked_table_combo.currentText() == "table"

    widget.shapes_section_toggle.click()
    first_shape_row.toggle_button.click()

    assert widget.shapes_group.is_expanded()
    assert first_shape_row.is_expanded()
    assert not first_shape_row.detail_widget.isHidden()
    assert widget.shape_cards[0].add_update_button.isEnabled()


def test_viewer_widget_expanded_detail_panels_fit_current_minimum_width(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(widget._points_controller, "load_value_source", lambda: None)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    scrollbar_width = widget.scroll_area.verticalScrollBar().sizeHint().width()
    content_margins = widget.content_layout.contentsMargins()
    content_width = WIDGET_MIN_WIDTH - scrollbar_width - content_margins.left() - content_margins.right()

    for group, row, card in (
        (widget.images_group, widget.image_rows[0], widget.image_cards[0]),
        (widget.labels_group, widget.labels_rows[0], widget.labels_cards[0]),
        (widget.shapes_group, widget.shape_rows[0], widget.shape_cards[0]),
    ):
        group_margins = group.layout().contentsMargins()
        row_margins = row.layout().contentsMargins()
        available_detail_width = (
            content_width
            - group_margins.left()
            - group_margins.right()
            - row_margins.left()
            - row_margins.right()
        )

        assert card.minimumSizeHint().width() <= available_detail_width
        assert card.sizeHint().width() <= available_detail_width

    points_group_margins = widget.points_group.layout().contentsMargins()
    available_points_width = content_width - points_group_margins.left() - points_group_margins.right()

    assert widget.points_widget.minimumSizeHint().width() <= available_points_width
    assert widget.points_widget.sizeHint().width() <= available_points_width


def test_viewer_widget_progressive_disclosure_actions_still_load_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    widget.images_section_toggle.click()
    widget.image_rows[0].toggle_button.click()
    widget.image_cards[0].add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image"

    widget.labels_section_toggle.click()
    widget.labels_rows[0].toggle_button.click()
    widget.labels_cards[0].add_update_button.click()

    assert len(viewer.layers) == 2
    assert viewer.layers[1].name == "blobs_labels"

    widget.shapes_section_toggle.click()
    widget.shape_rows[0].toggle_button.click()
    widget.shape_cards[0].add_update_button.click()

    assert len(viewer.layers) == 3
    assert viewer.layers[2].name == "blobs_circles"


def test_viewer_widget_shapes_empty_state_appears_when_no_shapes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.shapes_section_toggle.text() == "Shapes (0)"
    assert not widget.shapes_empty_label.isHidden()
    assert widget.shapes_section.isHidden()
    assert widget.shape_cards == []
    assert widget.shape_rows == []


def test_viewer_widget_preserves_expanded_shape_rows_across_refreshes(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    widget.shapes_section_toggle.click()
    widget.shape_rows[0].toggle_button.click()
    expanded_shapes_name = widget.shape_cards[0].shapes_name

    widget.refresh_from_sdata(sdata_blobs)

    refreshed_row = widget.shape_rows[0]
    refreshed_card = widget.shape_cards[0]
    assert refreshed_card.shapes_name == expanded_shapes_name
    assert refreshed_row.is_expanded()
    assert not refreshed_row.detail_widget.isHidden()


def test_viewer_widget_labels_cards_expose_table_driven_coloring_controls(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    second_card = widget.labels_cards[1]

    assert first_card.color_source_kind_combo.count() == 3
    assert [first_card.color_source_kind_combo.itemText(index) for index in range(3)] == [
        "None",
        "Observations",
        "Vars",
    ]
    assert first_card.color_source_value_input.completer() is not None
    assert not first_card.color_source_value_input.isEnabled()
    assert first_card.action_hint_label.text() == "Action: add/update primary labels layer"

    first_card.color_source_kind_combo.setCurrentIndex(1)
    assert first_card.color_source_value_input.isEnabled()
    assert first_card.color_source_value_input.completer().model().stringList() == ["instance_id"]
    assert first_card.action_hint_label.text() == 'Action: add/update colored overlay for obs["instance_id"]'

    first_card.color_source_kind_combo.setCurrentIndex(2)
    assert first_card.color_source_value_input.isEnabled()
    assert first_card.color_source_value_input.completer().model().stringList() == [
        "channel_0_sum",
        "channel_1_sum",
        "channel_2_sum",
    ]
    assert first_card.action_hint_label.text() == 'Action: add/update colored overlay for X[:, "channel_0_sum"]'

    second_card.color_source_kind_combo.setCurrentIndex(2)
    assert second_card.action_hint_label.text() == "Action: colored overlays require a linked table"


def test_viewer_widget_labels_card_repopulates_color_sources_when_linked_table_changes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: ["labels"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_annotating_table_names", lambda sdata, labels_name: ["table_a", "table_b"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: (
            [
                TableColorSourceSpec(
                    table_name=table_name, source_kind="obs_column", value_key="cell_type", value_kind="categorical"
                )
            ]
            if table_name == "table_a"
            else [
                TableColorSourceSpec(
                    table_name=table_name, source_kind="x_var", value_key="GeneA", value_kind="continuous"
                )
            ]
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]

    card.color_source_kind_combo.setCurrentIndex(1)
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == ["cell_type"]
    assert card.action_hint_label.text() == 'Action: add/update colored overlay for obs["cell_type"]'

    card.linked_table_combo.setCurrentIndex(1)
    assert not card.color_source_value_input.isEnabled()
    assert card.action_hint_label.text() == "Action: no colorable observation columns available"

    card.color_source_kind_combo.setCurrentIndex(2)
    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.completer().model().stringList() == ["GeneA"]
    assert card.action_hint_label.text() == 'Action: add/update colored overlay for X[:, "GeneA"]'


def test_viewer_widget_ignores_non_feature_matrix_write_events(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    table_names_by_label["labels"] = ["table", "new_table"]

    widget._on_feature_matrix_written(object())

    assert _combo_texts(card.linked_table_combo) == ["table"]
    assert card.selected_table_name == "table"


def test_viewer_widget_ignores_feature_matrix_writes_for_other_sdata(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    table_names_by_label["labels"] = ["table", "new_table"]

    widget.app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=other_sdata,
            table_name="new_table",
            feature_key="features",
            change_kind="created",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["table"]
    assert card.selected_table_name == "table"


def test_viewer_widget_refreshes_labels_card_linked_tables_from_feature_matrix_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    row = widget.labels_rows[0]
    row.set_expanded(True)
    table_names_by_label["labels"] = ["new_table", "table"]

    widget.app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=fake_sdata,
            table_name="new_table",
            feature_key="features",
            change_kind="created",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["new_table", "table"]
    assert card.selected_table_name == "table"
    assert row.is_expanded()
    assert len(viewer.layers) == 0


def test_viewer_widget_selects_first_linked_table_when_event_creates_first_table(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": []}

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    assert _combo_texts(card.linked_table_combo) == ["No linked tables"]
    assert card.selected_table_name is None
    assert not card.linked_table_combo.isEnabled()

    table_names_by_label["labels"] = ["new_table"]
    widget.app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=fake_sdata,
            table_name="new_table",
            feature_key="features",
            change_kind="created",
        )
    )

    assert _combo_texts(card.linked_table_combo) == ["new_table"]
    assert card.linked_table_combo.isEnabled()
    assert card.selected_table_name == "new_table"
    assert len(viewer.layers) == 0


def test_viewer_widget_preserves_labels_card_color_source_selection_after_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    card.color_source_kind_combo.setCurrentIndex(1)
    assert card.selected_color_source == color_sources_by_table["table"][0]

    table_names_by_label["labels"] = ["new_table", "table"]
    color_sources_by_table["new_table"] = [
        TableColorSourceSpec(
            table_name="new_table",
            source_kind="obs_column",
            value_key="other_type",
            value_kind="categorical",
        )
    ]
    widget.app_state.emit_feature_matrix_written(
        FeatureMatrixWrittenEvent(
            sdata=fake_sdata,
            table_name="new_table",
            feature_key="features",
            change_kind="created",
        )
    )

    assert card.selected_table_name == "table"
    assert card.selected_source_kind == "obs_column"
    assert card.selected_color_source == color_sources_by_table["table"][0]
    assert card.action_hint_label.text() == 'Action: add/update colored overlay for obs["cell_type"]'


def test_viewer_widget_refreshes_table_color_sources_from_classification_table_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    card.color_source_kind_combo.setCurrentIndex(1)
    assert card._color_source_completer_model.stringList() == ["cell_type"]

    color_sources_by_table["table"] = [
        *color_sources_by_table["table"],
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="user_class",
            value_kind="categorical",
        ),
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="pred_class",
            value_kind="categorical",
        ),
    ]

    widget.app_state.emit_classification_table_written(
        ClassificationTableWrittenEvent(
            sdata=fake_sdata,
            table_name="table",
            columns=("user_class", "pred_class"),
        )
    )

    assert card._color_source_completer_model.stringList() == ["cell_type", "user_class", "pred_class"]
    assert card.selected_table_name == "table"
    assert card.selected_source_kind == "obs_column"
    assert len(viewer.layers) == 0


def test_viewer_widget_ignores_classification_table_events_for_other_sdata(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    table_names_by_label = {"labels": ["table"]}
    color_sources_by_table = {
        "table": [
            TableColorSourceSpec(
                table_name="table",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ]
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    _patch_viewer_widget_labels_tables(
        monkeypatch,
        labels_names=["labels"],
        table_names_by_label=table_names_by_label,
        color_sources_by_table=color_sources_by_table,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    card = widget.labels_cards[0]
    card.color_source_kind_combo.setCurrentIndex(1)
    color_sources_by_table["table"] = [
        *color_sources_by_table["table"],
        TableColorSourceSpec(
            table_name="table",
            source_kind="obs_column",
            value_key="user_class",
            value_kind="categorical",
        ),
    ]

    widget.app_state.emit_classification_table_written(
        ClassificationTableWrittenEvent(
            sdata=other_sdata,
            table_name="table",
            columns=("user_class",),
        )
    )

    assert card._color_source_completer_model.stringList() == ["cell_type"]


def test_viewer_widget_refreshes_only_shapes_section_from_shapes_element_event(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    names = {
        "images": ["image"],
        "labels": ["labels"],
        "shapes": ["shape_a"],
        "points": ["points"],
    }

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: names["images"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: names["labels"])
    monkeypatch.setattr(viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: names["shapes"])
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: names["points"])
    monkeypatch.setattr(viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0"])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]
    image_rows = widget.image_rows
    labels_rows = widget.labels_rows

    def fail_if_rebuilt(*args, **kwargs):
        del args, kwargs
        raise AssertionError("non-shapes sections should not be rebuilt")

    monkeypatch.setattr(widget, "_rebuild_image_cards", fail_if_rebuilt)
    monkeypatch.setattr(widget, "_rebuild_labels_cards", fail_if_rebuilt)
    monkeypatch.setattr(widget, "_refresh_points_section", fail_if_rebuilt)
    names["shapes"] = ["shape_a", "new_regions"]

    widget.app_state.emit_shapes_element_written(
        ShapesElementWrittenEvent(
            sdata=fake_sdata,
            shapes_name="new_regions",
            coordinate_system="global",
        )
    )

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a", "new_regions"]
    assert widget.image_rows == image_rows
    assert widget.labels_rows == labels_rows
    assert widget.shapes_section_title.full_text() == "Shapes (2)"
    assert widget.shapes_empty_label.isHidden()
    assert not widget.shapes_section.isHidden()
    assert "2 shapes element(s)" in widget.summary_label.text()
    assert len(viewer.layers) == 0


def test_viewer_widget_ignores_shapes_element_events_for_other_sdata_or_coordinate_system(
    qtbot,
    monkeypatch,
) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    other_sdata = object()
    shapes_names = ["shape_a"]

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: list(shapes_names),
    )
    monkeypatch.setattr(viewer_widget_module, "_get_points_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.app_state.coordinate_system == "global"
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]

    shapes_names[:] = ["shape_a", "new_regions"]
    widget.app_state.emit_shapes_element_written(
        ShapesElementWrittenEvent(
            sdata=other_sdata,
            shapes_name="new_regions",
            coordinate_system="global",
        )
    )
    widget.app_state.emit_shapes_element_written(
        ShapesElementWrittenEvent(
            sdata=fake_sdata,
            shapes_name="new_regions",
            coordinate_system="local",
        )
    )

    assert [card.shapes_name for card in widget.shape_cards] == ["shape_a"]
    assert widget.shapes_section_title.full_text() == "Shapes (1)"
    assert "1 shapes element(s)" in widget.summary_label.text()


def test_viewer_widget_image_mode_toggles_are_mutually_exclusive(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    assert image_card.stack_toggle.isChecked()
    assert not image_card.overlay_toggle.isChecked()
    assert image_card.channel_panel.isHidden()
    assert image_card.channel_section_label.text() == "Channels"
    assert image_card.add_update_button.isEnabled()

    image_card.overlay_toggle.setChecked(True)

    assert not image_card.stack_toggle.isChecked()
    assert image_card.overlay_toggle.isChecked()
    assert not image_card.channel_panel.isHidden()
    assert image_card.add_update_button.isEnabled()

    image_card.stack_toggle.setChecked(True)

    assert image_card.stack_toggle.isChecked()
    assert not image_card.overlay_toggle.isChecked()
    assert image_card.channel_panel.isHidden()
    assert image_card.add_update_button.isEnabled()


def test_viewer_widget_overlay_channel_panel_scrolls_when_many_channels(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    many_channels = [f"c{i}" for i in range(12)]

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: many_channels
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]

    assert len(image_card.channel_checkboxes) == len(many_channels)
    assert image_card.channel_scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert image_card.channel_scroll_area.maximumHeight() > 0
    assert image_card.channel_scroll_area.maximumHeight() < image_card.channel_list_widget.sizeHint().height()


def test_viewer_widget_surfaces_duplicate_channel_names_and_disables_overlay(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: (_ for _ in ()).throw(
            ValueError(
                "Image element `image` exposes duplicate channel names (`dup`), "
                "which napari-harpy does not support. "
                "Update the channel names in the SpatialData object with "
                "`sdata.set_channel_names(...)`."
            )
        ),
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]

    assert image_card.channel_names == []
    assert image_card.channel_error is not None
    assert not image_card.overlay_toggle.isEnabled()
    assert not image_card.channel_warning_label.isHidden()
    assert "sdata.set_channel_names(...)" in image_card.channel_warning_label.text()
    assert "duplicate channel names" in image_card.channel_warning_label.toolTip()


def test_viewer_widget_filters_cards_by_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: ["labels_global"] if coordinate_system == "global" else ["labels_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table_global"] if labels_name == "labels_global" else ["table_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [],
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    assert widget.coordinate_system_combo.count() == 2
    assert widget.app_state.coordinate_system == "global"
    assert [card.image_name for card in widget.image_cards] == ["image_global"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_global"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_global"]

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed) as blocker:
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert blocker.args[0].previous_coordinate_system == "global"
    assert blocker.args[0].coordinate_system == "local"
    assert blocker.args[0].source == "viewer_widget"
    assert widget.app_state.coordinate_system == "local"
    assert [card.image_name for card in widget.image_cards] == ["image_local"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_local"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_local"]


def test_viewer_widget_refreshes_from_shared_coordinate_system_changes(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_labels_in_coordinate_system",
        lambda sdata, coordinate_system: ["labels_global"] if coordinate_system == "global" else ["labels_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, labels_name: ["table_global"] if labels_name == "labels_global" else ["table_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    changed = widget.app_state.set_coordinate_system("local", source="object_classification_widget")

    assert changed is True
    assert widget.coordinate_system_combo.currentText() == "local"
    assert [card.image_name for card in widget.image_cards] == ["image_local"]
    assert [card.labels_name for card in widget.labels_cards] == ["labels_local"]
    assert [card.shapes_name for card in widget.shape_cards] == ["shape_local"]


def test_viewer_widget_coordinate_system_switch_prunes_old_harpy_layers(qtbot, monkeypatch) -> None:
    global_layer = Image(np.zeros((2, 2), dtype=np.float32), name="global_layer")
    local_layer = Image(np.zeros((2, 2), dtype=np.float32), name="local_layer")
    external_layer = Image(np.zeros((2, 2), dtype=np.float32), name="external_layer")
    viewer = DummyViewer()
    viewer.layers.extend([global_layer, local_layer, external_layer])
    widget = ViewerWidget(viewer)
    fake_sdata = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.app_state.viewer_adapter.register_image_layer(
        global_layer,
        sdata=fake_sdata,
        image_name="global_image",
        coordinate_system="global",
    )
    widget.app_state.viewer_adapter.register_image_layer(
        local_layer,
        sdata=fake_sdata,
        image_name="local_image",
        coordinate_system="local",
    )
    widget._set_action_feedback(
        title="Labels Layer Created",
        lines=['Created labels layer for "global_image".'],
        kind="success",
    )

    with qtbot.waitSignal(widget.app_state.coordinate_system_changed):
        widget.coordinate_system_combo.setCurrentIndex(1)

    assert widget.app_state.coordinate_system == "local"
    assert widget.global_action_feedback_label.text() == ""
    assert widget.global_action_feedback_label.isHidden()
    assert list(viewer.layers) == [local_layer, external_layer]
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(global_layer) is None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(local_layer) is not None
    assert widget.app_state.viewer_adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_widget_open_spatialdata_loads_selected_store(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_paths: list[str] = []
    recorded_sdata: list[object] = []
    original_set_sdata = widget.app_state.set_sdata

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "read_zarr",
        lambda path: recorded_paths.append(path) or sdata_blobs,
    )

    def wrapped_set_sdata(sdata: object) -> None:
        recorded_sdata.append(sdata)
        original_set_sdata(sdata)

    monkeypatch.setattr(widget.app_state, "set_sdata", wrapped_set_sdata)
    widget._set_action_feedback("Old error", is_error=True)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.open_sdata_button.click()

    assert recorded_paths == ["/tmp/example.zarr"]
    assert recorded_sdata == [sdata_blobs]
    assert widget.app_state.sdata is sdata_blobs
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.global_action_feedback_label.text() == ""
    assert widget.global_action_feedback_label.isHidden()


def test_viewer_widget_open_spatialdata_shows_error_when_loading_fails(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(
        viewer_widget_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: "/tmp/example.zarr",
    )

    def raise_read_error(path: str) -> object:
        raise ValueError(f"bad store at {path}")

    monkeypatch.setattr(viewer_widget_module, "read_zarr", raise_read_error)

    widget.open_sdata_button.click()

    assert widget.app_state.sdata is None
    assert "Could not load SpatialData store" in widget.global_action_feedback_label.text()
    assert "bad store at /tmp/example.zarr" in widget.global_action_feedback_label.text()
    assert not widget.global_action_feedback_label.isHidden()


def test_viewer_widget_add_update_labels_loads_and_activates_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    _assert_action_feedback_card(widget, title="Labels Layer Created", kind="success")
    assert 'Created labels layer for "blobs_labels"' in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_labels_dispatches_to_styled_overlay_path(qtbot, monkeypatch, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    recorded_requests: list[object] = []

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    monkeypatch.setattr(widget, "_add_or_update_styled_labels_layer", lambda request: recorded_requests.append(request))

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(2)
    first_card.color_source_value_input.setText("channel_1_sum")
    first_card.add_update_button.click()

    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.labels_name == "blobs_labels"
    assert request.table_name == "table"
    assert request.selected_source_kind == "x_var"
    assert request.selected_color_source is not None
    assert request.selected_color_source.value_key == "channel_1_sum"


def test_viewer_widget_add_update_labels_creates_and_updates_styled_overlay(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.labels_role == "styled"
    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="success")
    assert 'Created colored overlay for obs["cell_type"]' in widget.global_action_feedback_label.text()
    assert "stored categorical palette" in widget.global_action_feedback_label.text()

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    _assert_action_feedback_card(widget, title="Colored Overlay Updated", kind="success")
    assert 'Updated colored overlay for obs["cell_type"]' in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_missing_palette_uses_info_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="info")
    assert "no stored palette was present" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_instance_key_uses_success_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created", kind="success")
    assert 'Created colored overlay for obs["instance_id"]' in widget.global_action_feedback_label.text()
    assert "Used instance colors." in widget.global_action_feedback_label.text()
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(viewer.layers[0])
    assert binding is not None
    assert binding.style_spec is not None
    assert binding.style_spec.value_kind == "instance"


def test_viewer_widget_styled_overlay_invalid_palette_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["cell_type"] = ["odd"] * table.n_obs
    table.obs["cell_type"] = table.obs["cell_type"].astype("category")
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("cell_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "stored categorical palette was invalid" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_string_coercion_uses_warning_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    table = sdata_blobs["table"]
    table.obs["sample_type"] = ["odd" if instance_id % 2 else "even" for instance_id in table.obs["instance_id"]]

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("sample_type")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Colored Overlay Created With Warning", kind="warning")
    assert "Coerced string values to categorical" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_overlay_precondition_error_uses_error_card(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.labels_cards[0]
    first_card.color_source_kind_combo.setCurrentIndex(1)
    first_card.color_source_value_input.setText("not_a_column")

    first_card.add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Labels Error", kind="error")
    assert "The selected observation column is not available" in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_image_loads_stack_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_image"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"
    assert viewer.layers.selection.active is layer
    _assert_action_feedback_card(widget, title="Image Layer Created", kind="success")
    assert 'Created image layer for "blobs_image" in stack mode' in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_image_reuses_existing_stack_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.image_cards[0]

    first_card.add_update_button.click()
    first_layer = viewer.layers[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is first_layer


def test_viewer_widget_add_update_image_overlay_passes_selected_channels_and_colors(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layers = [object(), object()]
    recorded_calls: list[tuple[object, str, str, str, list[int] | None, list[str] | None]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: ["image"]
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["c0", "c1", "c2"],
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_loaded",
        lambda sdata, image_name, coordinate_system, *, mode, channels=None, channel_colors=None: (
            recorded_calls.append((sdata, image_name, coordinate_system, mode, channels, channel_colors))
            or SimpleNamespace(
                layers=tuple(fake_layers),
                primary_layer=fake_layers[0],
                mode=mode,
                created=True,
                channels=tuple(channels or ()),
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    image_card = widget.image_cards[0]
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_checkboxes[0].setChecked(True)
    image_card.channel_checkboxes[2].setChecked(True)
    image_card.channel_color_buttons[0].set_color("#00FFFF")
    image_card.channel_color_buttons[2].set_color("#FFA500")

    image_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "image", "global", "overlay", [0, 2], ["#00FFFF", "#FFA500"])]
    assert activated_layers == [fake_layers[0]]


def test_viewer_widget_add_update_image_overlay_loads_reuses_and_replaces_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    image_card.overlay_toggle.setChecked(True)
    image_card.channel_checkboxes[0].setChecked(True)
    image_card.channel_checkboxes[2].setChecked(True)
    image_card.add_update_button.click()

    assert len(viewer.layers) == 2
    first_layers = list(viewer.layers)
    assert [layer.name for layer in first_layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert viewer.layers.selection.active is first_layers[0]
    assert 'Created image overlay for "blobs_image"' in widget.global_action_feedback_label.text()

    image_card.add_update_button.click()

    assert len(viewer.layers) == 2
    assert list(viewer.layers) == first_layers

    image_card.channel_checkboxes[0].setChecked(False)
    image_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image[2]"


def test_viewer_widget_empty_overlay_selection_removes_existing_image_layers(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    image_card = widget.image_cards[0]

    image_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].name == "blobs_image"

    image_card.overlay_toggle.setChecked(True)
    image_card.add_update_button.click()

    assert list(viewer.layers) == []
    assert "Overlay mode requires at least one selected channel." in widget.global_action_feedback_label.text()
    assert not widget.global_action_feedback_label.isHidden()


def test_viewer_widget_add_update_image_uses_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = Shapes([np.asarray([(0, 0), (0, 1), (1, 1), (1, 0)], dtype=float)], shape_type="polygon")
    recorded_calls: list[tuple[object, str, str, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_images_in_coordinate_system",
        lambda sdata, coordinate_system: ["image_global"] if coordinate_system == "global" else ["image_local"],
    )
    monkeypatch.setattr(
        viewer_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: ["c0", "c1"]
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_image_loaded",
        lambda sdata, image_name, coordinate_system, *, mode, channels=None, channel_colors=None: (
            recorded_calls.append((sdata, image_name, coordinate_system, mode))
            or SimpleNamespace(
                layers=(fake_layer,),
                primary_layer=fake_layer,
                mode=mode,
                created=True,
                channels=tuple(channels or ()),
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.coordinate_system_combo.setCurrentIndex(1)
    image_card = widget.image_cards[0]

    image_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "image_local", "local", "stack")]
    assert activated_layers == [fake_layer]


def test_viewer_widget_shapes_card_exposes_shape_column_controls(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert [card.color_source_kind_combo.itemText(index) for index in range(card.color_source_kind_combo.count())] == [
        "None",
        "Shapes column",
        "Observations",
        "Vars",
    ]
    assert _combo_texts(card.linked_table_combo) == ["No linked tables"]
    assert not card.linked_table_combo.isEnabled()
    assert card.color_source_value_label.text() == "Value source"
    assert not card.color_source_value_input.isEnabled()
    assert card.fill_toggle.text() == "Fill"
    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card.action_hint_label.text() == "Action: add/update primary shapes layer"

    card.color_source_kind_combo.setCurrentIndex(1)

    assert card.color_source_value_input.isEnabled()
    assert card.color_source_value_input.placeholderText() == "Search shapes columns"
    assert card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card._color_source_completer_model.stringList() == ["cell_type", "score", "free_text"]
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for column "cell_type"'

    card.color_source_kind_combo.setCurrentIndex(2)

    assert not card.color_source_value_input.isEnabled()
    assert card.action_hint_label.text() == "Action: table-backed shapes coloring requires a linked table"

    card.fill_toggle.setChecked(True)
    card.color_source_kind_combo.setCurrentIndex(1)
    card.color_source_value_input.setText("not_a_shape_column")

    assert not card.fill_toggle.isEnabled()
    assert not card.fill_toggle.isChecked()
    assert card.action_hint_label.text() == "Action: select a shapes column for a styled shapes layer"


def test_viewer_widget_shape_column_selector_hides_geometry_and_palette_columns(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    card.color_source_kind_combo.setCurrentIndex(1)

    assert "geometry" not in card._color_source_completer_model.stringList()
    assert "cell_type_colors" not in card._color_source_completer_model.stringList()


def test_viewer_widget_shapes_card_exposes_linked_table_sources(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    color_sources_by_table = {
        "table_a": [
            TableColorSourceSpec(
                table_name="table_a",
                source_kind="obs_column",
                value_key="cell_type",
                value_kind="categorical",
            )
        ],
        "table_b": [
            TableColorSourceSpec(
                table_name="table_b",
                source_kind="x_var",
                value_key="GeneA",
                value_kind="continuous",
            )
        ],
    }

    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, element_name: ["table_a", "table_b"] if element_name == "cells" else [],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: list(color_sources_by_table[table_name]),
    )
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]

    assert _combo_texts(card.linked_table_combo) == ["table_a", "table_b"]
    assert card.selected_table_name == "table_a"

    card.color_source_kind_combo.setCurrentIndex(2)
    assert card.color_source_value_label.text() == "Observation"
    assert card.color_source_value_input.isEnabled()
    assert card._color_source_completer_model.stringList() == ["cell_type"]
    assert card.selected_color_source == color_sources_by_table["table_a"][0]
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for obs["cell_type"]'

    card.linked_table_combo.setCurrentIndex(1)
    assert not card.color_source_value_input.isEnabled()
    assert card.action_hint_label.text() == "Action: no colorable observation columns available"

    card.color_source_kind_combo.setCurrentIndex(3)
    assert card.color_source_value_label.text() == "Var"
    assert card.color_source_value_input.isEnabled()
    assert card._color_source_completer_model.stringList() == ["GeneA"]
    assert card.selected_color_source == color_sources_by_table["table_b"][0]
    assert card.action_hint_label.text() == 'Action: add/update styled shapes layer for X[:, "GeneA"]'


def test_viewer_widget_add_update_shapes_with_table_source_dispatches_to_styled_path(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    table_source = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    recorded_calls: list[tuple[object, str, str, TableColorSourceSpec, bool]] = []
    result_layer = object()

    monkeypatch.setattr(
        viewer_widget_module,
        "get_annotating_table_names",
        lambda sdata, element_name: ["table"] if element_name == "cells" else [],
    )
    monkeypatch.setattr(
        viewer_widget_module,
        "get_table_color_source_options",
        lambda sdata, table_name: [table_source],
    )

    def ensure_styled_shapes_loaded(
        sdata_arg: object,
        shapes_name: str,
        coordinate_system: str,
        style_spec: TableColorSourceSpec,
        *,
        fill: bool = False,
    ) -> SimpleNamespace:
        recorded_calls.append((sdata_arg, shapes_name, coordinate_system, style_spec, fill))
        return SimpleNamespace(
            layer=result_layer,
            created=True,
            value_kind="categorical",
            palette_source="stored",
            coercion_applied=False,
            skipped_geometry_count=0,
            unannotated_source_shape_count=1,
            unannotated_rendered_shape_count=1,
            shapes_rendering_mode="points",
        )

    monkeypatch.setattr(widget.app_state.viewer_adapter, "ensure_styled_shapes_loaded", ensure_styled_shapes_loaded)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    card.color_source_kind_combo.setCurrentIndex(2)
    card.fill_toggle.setChecked(True)
    card.add_update_button.click()

    assert recorded_calls == [(sdata, "cells", "global", table_source, True)]
    assert viewer.layers.selection.active is result_layer
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="info")
    assert 'Created styled shapes layer for obs["cell_type"]' in widget.global_action_feedback_label.text()
    assert "Rendered point-radius shapes as napari points for faster display." in widget.global_action_feedback_label.text()
    assert "Used the stored categorical palette." in widget.global_action_feedback_label.text()
    assert "Rendered 1 shape transparent because it has no row in the linked table." in (
        widget.global_action_feedback_label.text()
    )


def test_viewer_widget_add_update_shapes_with_shape_column_dispatches_to_styled_path(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    recorded_requests: list[ShapesLoadRequest] = []

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    monkeypatch.setattr(widget, "_add_or_update_styled_shapes_layer", lambda request: recorded_requests.append(request))
    card = widget.shape_cards[0]
    _select_shape_column(card, "score")
    card.fill_toggle.setChecked(True)

    card.add_update_button.click()

    assert len(recorded_requests) == 1
    request = recorded_requests[0]
    assert request.shapes_name == "cells"
    assert request.table_name is None
    assert request.selected_source_kind == "shape_column"
    assert request.selected_color_source == ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )
    assert request.fill_shapes is True


def test_viewer_widget_add_update_styled_shapes_creates_and_updates_layer(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    card = widget.shape_cards[0]
    _select_shape_column(card, "cell_type")

    card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "cells[shapes_column:cell_type]"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_type == "shapes"
    assert binding.shapes_role == "styled"
    assert binding.style_spec == ShapeColumnColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    np.testing.assert_allclose(layer.face_color[:, 3], np.zeros(len(layer.data)))
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="success")
    assert 'Created styled shapes layer for column "cell_type"' in widget.global_action_feedback_label.text()
    assert "Used the stored categorical palette." in widget.global_action_feedback_label.text()

    card.fill_toggle.setChecked(True)
    card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is layer
    np.testing.assert_allclose(layer.face_color[:, 3], np.full(len(layer.data), SHAPES_FACE_ALPHA))
    _assert_action_feedback_card(widget, title="Styled Shapes Updated", kind="success")
    assert 'Updated styled shapes layer for column "cell_type"' in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_missing_palette(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=None)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="info")
    assert "no stored palette was present" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_invalid_palette(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "not-a-color"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "stored categorical palette was invalid" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_string_coercion(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "free_text")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "Coerced string values to categorical" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_allows_duplicate_source_index(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"], duplicate_index=True)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0].features["index"].to_list() == ["cell_1", "cell_1"]
    assert viewer.layers[0].features["cell_type"].to_list() == ["T", "B"]
    _assert_action_feedback_card(widget, title="Styled Shapes Created", kind="success")


def test_viewer_widget_table_backed_styled_shapes_without_linked_table_is_feedback(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name=None,
        selected_source_kind="obs_column",
        selected_color_source=None,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "has no linked table for table-driven coloring" in widget.global_action_feedback_label.text()


def test_viewer_widget_table_backed_styled_shapes_missing_source_is_feedback(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name="table",
        selected_source_kind="obs_column",
        selected_color_source=None,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "The selected observation column is not available" in widget.global_action_feedback_label.text()


def test_viewer_widget_table_backed_styled_shapes_alignment_error_is_feedback(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(cell_type_colors=["red", "blue"])
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    def raise_alignment_error(*args: object, **kwargs: object) -> None:
        raise ValueError("Every selected-region table instance must exist in the shapes instance column.")

    monkeypatch.setattr(widget.app_state.viewer_adapter, "ensure_styled_shapes_loaded", raise_alignment_error)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    request = ShapesLoadRequest(
        shapes_name="cells",
        table_name="table",
        selected_source_kind="obs_column",
        selected_color_source=style_spec,
        fill_shapes=False,
    )

    widget._add_or_update_shapes_layer(request)

    _assert_action_feedback_card(widget, title="Styled Shapes Error", kind="error")
    assert "Every selected-region table instance must exist" in widget.global_action_feedback_label.text()


def test_viewer_widget_styled_shapes_feedback_reports_skipped_geometry(qtbot) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    sdata = _make_colorable_shapes_sdata(
        cell_type_colors=["red", "blue"],
        include_unsupported_geometry=True,
    )

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata)

    _select_shape_column(widget.shape_cards[0], "cell_type")
    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Styled Shapes Created With Warning", kind="warning")
    assert "Skipped 1 empty, invalid, or unsupported geometries" in widget.global_action_feedback_label.text()


def test_viewer_widget_add_update_shapes_loads_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.shape_cards[0]
    first_card.fill_toggle.setChecked(True)

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "blobs_circles"
    binding = widget.app_state.viewer_adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_type == "shapes"
    assert binding.element_name == "blobs_circles"
    assert binding.coordinate_system == "global"
    assert viewer.layers.selection.active is layer
    np.testing.assert_allclose(layer.face_color, np.asarray([to_rgba("#00FFFF")] * len(layer.data)))
    _assert_action_feedback_card(widget, title="Shapes Layer Created", kind="success")
    assert 'Created shapes layer for "blobs_circles".' in widget.global_action_feedback_label.text()
    assert "Rendered point-radius shapes as napari points for faster display." in (
        widget.global_action_feedback_label.text()
    )


def test_viewer_widget_add_update_shapes_reuses_existing_layer(qtbot, sdata_blobs) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)

    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(sdata_blobs)

    first_card = widget.shape_cards[0]

    first_card.add_update_button.click()
    first_layer = viewer.layers[0]

    first_card.add_update_button.click()

    assert len(viewer.layers) == 1
    assert viewer.layers[0] is first_layer


def test_viewer_widget_add_update_shapes_uses_selected_coordinate_system(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()
    recorded_calls: list[tuple[object, str, str]] = []
    activated_layers: list[object] = []

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global", "local"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module,
        "_get_shapes_in_coordinate_system",
        lambda sdata, coordinate_system: ["shape_global"] if coordinate_system == "global" else ["shape_local"],
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_shapes_loaded",
        lambda sdata, shapes_name, coordinate_system: (
            recorded_calls.append((sdata, shapes_name, coordinate_system))
            or SimpleNamespace(
                layer=fake_layer,
                created=True,
                value_kind=None,
                palette_source=None,
                coercion_applied=False,
                skipped_geometry_count=0,
                shapes_rendering_mode="shapes",
            )
        ),
    )
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "activate_layer",
        lambda layer: activated_layers.append(layer) or True,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.coordinate_system_combo.setCurrentIndex(1)
    shape_card = widget.shape_cards[0]

    shape_card.add_update_button.click()

    assert recorded_calls == [(fake_sdata, "shape_local", "local")]
    assert activated_layers == [fake_layer]


def test_viewer_widget_add_update_shapes_reports_skipped_geometry_warning(qtbot, monkeypatch) -> None:
    viewer = DummyViewer()
    widget = ViewerWidget(viewer)
    fake_sdata = object()
    fake_layer = object()

    qtbot.addWidget(widget)

    _patch_coordinate_system_names(monkeypatch, ["global"])
    monkeypatch.setattr(viewer_widget_module, "_get_labels_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(viewer_widget_module, "_get_images_in_coordinate_system", lambda sdata, coordinate_system: [])
    monkeypatch.setattr(
        viewer_widget_module, "_get_shapes_in_coordinate_system", lambda sdata, coordinate_system: ["cells"]
    )
    monkeypatch.setattr(viewer_widget_module, "get_shape_column_color_source_options", lambda sdata, shapes_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_annotating_table_names", lambda sdata, element_name: [])
    monkeypatch.setattr(viewer_widget_module, "get_table_color_source_options", lambda sdata, table_name: [])
    monkeypatch.setattr(
        widget.app_state.viewer_adapter,
        "ensure_shapes_loaded",
        lambda sdata, shapes_name, coordinate_system: SimpleNamespace(
            layer=fake_layer,
            created=True,
            value_kind=None,
            palette_source=None,
            coercion_applied=False,
            skipped_geometry_count=2,
            shapes_rendering_mode="shapes",
        ),
    )
    monkeypatch.setattr(widget.app_state.viewer_adapter, "activate_layer", lambda layer: True)
    widget.app_state.viewer_adapter.register_shapes_layer(
        fake_layer,
        sdata=fake_sdata,
        shapes_name="cells",
        coordinate_system="global",
        skipped_geometry_count=2,
    )

    with qtbot.waitSignal(widget.app_state.sdata_changed):
        widget.app_state.set_sdata(fake_sdata)

    widget.shape_cards[0].add_update_button.click()

    _assert_action_feedback_card(widget, title="Shapes Layer Created With Warning", kind="warning")
    assert "point-radius shapes as napari points" not in widget.global_action_feedback_label.text()
    assert "Skipped 2 empty, invalid, or unsupported geometries" in widget.global_action_feedback_label.text()


def test_viewer_widget_shares_app_state_for_same_viewer(qtbot) -> None:
    viewer = DummyViewer()
    first = ViewerWidget(viewer)
    second = ViewerWidget(viewer)

    qtbot.addWidget(first)
    qtbot.addWidget(second)

    assert first.app_state is second.app_state
