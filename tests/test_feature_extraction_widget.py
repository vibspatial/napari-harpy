from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from napari.layers import Labels
from qtpy.QtWidgets import QCheckBox, QScrollArea
from spatialdata import SpatialData

from napari_harpy.widgets._feature_extraction_widget import FeatureExtractionWidget


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[Labels] | None = None) -> None:
        super().__init__(layers or [])
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )


class DummyViewer:
    def __init__(self, layers: list[Labels] | None = None) -> None:
        self.layers = DummyLayers(layers)


def make_blobs_labels_layer(sdata: SpatialData, label_name: str = "blobs_labels") -> Labels:
    return Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={"sdata": sdata, "name": label_name},
    )


def test_feature_extraction_widget_can_be_instantiated(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    assert widget is not None
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_coordinate_system is None
    assert widget.selected_feature_names == ()
    assert widget.selected_feature_key is None
    assert widget.overwrite_feature_key is False
    assert widget.segmentation_combo.count() == 0
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.table_combo.count() == 0
    assert widget.coordinate_system_combo.count() == 0
    assert widget.calculate_button.isEnabled() is False


def test_feature_extraction_widget_populates_selector_flow_from_spatialdata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.image_combo.count() == 3
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "blobs_image",
        "blobs_multiscale_image",
    ]
    assert widget.table_combo.count() == 1
    assert widget.table_combo.itemText(0) == "table"
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_segmentation_name == "blobs_labels"
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_image_name is None
    assert widget.selected_table_name == "table"
    assert widget.selected_coordinate_system == "global"
    assert "Selection Ready" in widget.selection_status.text()
    assert "Segmentation: blobs_labels" in widget.selection_status.text()
    assert "Table: table" in widget.selection_status.text()
    assert "Coordinate system: global" in widget.selection_status.text()
    assert widget.validation_status.isHidden()


def test_feature_extraction_widget_blocks_when_selected_segmentation_has_no_linked_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.segmentation_combo.setCurrentIndex(1)

    assert widget.selected_segmentation_name == "blobs_multiscale_labels"
    assert widget.image_combo.count() == 3
    assert widget.selected_image_name is None
    assert widget.table_combo.count() == 0
    assert widget.selected_table_name is None
    assert widget.coordinate_system_combo.count() == 1
    assert widget.selected_coordinate_system == "global"
    assert "No Table Linked" in widget.selection_status.text()
    assert "creating a new linked table" in widget.selection_status.text()
    assert widget.validation_status.isHidden()


def test_feature_extraction_widget_exposes_grouped_feature_checkboxes(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    scroll_area = widget.findChild(QScrollArea, "feature_extraction_scroll_area")
    assert scroll_area is not None
    assert scroll_area.widgetResizable()
    assert widget.intensity_features_group.title() == "Intensity Features"
    assert widget.morphology_features_group.title() == "Morphology Features"
    assert widget.findChild(QCheckBox, "feature_checkbox_mean") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_var") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_area") is not None
    assert widget.findChild(QCheckBox, "feature_checkbox_perimeter") is not None
    assert widget.findChild(QCheckBox, "feature_extraction_overwrite_feature_key_checkbox") is None
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_reads_back_feature_selection_and_output_key(qtbot) -> None:
    widget = FeatureExtractionWidget()

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_var").setChecked(True)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("object_features")

    assert widget.selected_feature_names == ("mean", "var", "area")
    assert widget.selected_feature_key == "object_features"
    assert widget.overwrite_feature_key is False
    assert not widget.intensity_features_hint.isHidden()
    assert "choose an image" in widget.intensity_features_hint.text()


def test_feature_extraction_widget_hides_intensity_warning_when_image_is_selected(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_rebinds_controller_when_inputs_change(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind(*args, **kwargs):
        bind_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind = fake_bind  # type: ignore[method-assign]

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert bind_calls
    args, kwargs = bind_calls[-1]
    assert args == (
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ("area",),
        "features",
    )
    assert kwargs == {"overwrite_feature_key": False}


def test_feature_extraction_widget_enables_calculate_button_for_runnable_selection(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.calculate_button.isEnabled() is False

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is True
    assert "Calculate the selected features" in widget.calculate_button.toolTip()


def test_feature_extraction_widget_keeps_calculate_disabled_for_intensity_features_without_image(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is False
    assert "choose an image" in widget.calculate_button.toolTip()


def test_feature_extraction_widget_blocks_when_no_coordinate_system_is_selected(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")
    widget._set_selected_coordinate_system(-1)
    widget._bind_current_selection()

    assert widget.selected_coordinate_system is None
    assert "Choose Coordinate System" in widget.selection_status.text()
    assert "choose a coordinate system" in widget.calculate_button.toolTip().lower()
    assert widget.calculate_button.isEnabled() is False


def test_feature_extraction_widget_calculate_button_click_launches_controller(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    calls: list[bool | None] = []

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        calls.append(overwrite_feature_key)
        return True

    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert calls == [False]


def test_feature_extraction_widget_calculate_without_existing_key_uses_non_overwrite_run(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("new_features")

    overwrite_calls: list[bool | None] = []

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        overwrite_calls.append(overwrite_feature_key)
        return True

    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert overwrite_calls == [False]


def test_feature_extraction_widget_prompts_before_overwriting_existing_feature_key(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("existing_features")
    sdata_blobs.tables["table"].obsm["existing_features"] = np.zeros((sdata_blobs.tables["table"].n_obs, 1))

    prompt_calls: list[tuple[str, str | None]] = []
    overwrite_calls: list[bool | None] = []

    def fake_prompt(feature_key: str, table_name: str | None) -> bool | None:
        prompt_calls.append((feature_key, table_name))
        return True

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        overwrite_calls.append(overwrite_feature_key)
        return True

    widget._prompt_overwrite_feature_key_confirmation = fake_prompt  # type: ignore[method-assign]
    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert prompt_calls == [("existing_features", "table")]
    assert overwrite_calls == [True]


def test_feature_extraction_widget_cancelled_overwrite_does_not_launch_calculation(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("existing_features")
    sdata_blobs.tables["table"].obsm["existing_features"] = np.zeros((sdata_blobs.tables["table"].n_obs, 1))

    calculate_calls: list[bool | None] = []

    def fake_calculate(*, overwrite_feature_key: bool | None = None) -> bool:
        calculate_calls.append(overwrite_feature_key)
        return True

    def fake_prompt(feature_key: str, table_name: str | None) -> bool | None:
        del feature_key, table_name
        return None

    widget._prompt_overwrite_feature_key_confirmation = fake_prompt  # type: ignore[method-assign]
    widget._feature_extraction_controller.calculate = fake_calculate  # type: ignore[method-assign]

    widget.calculate_button.click()

    assert calculate_calls == []


def test_feature_extraction_widget_refreshes_table_state_after_controller_success(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer([make_blobs_labels_layer(sdata_blobs)])
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    calls: list[str] = []

    original_refresh_table_names = widget._refresh_table_names
    original_bind_current_selection = widget._bind_current_selection

    def recording_refresh_table_names() -> None:
        calls.append("refresh_table_names")
        original_refresh_table_names()

    def recording_bind_current_selection() -> None:
        calls.append("bind_current_selection")
        original_bind_current_selection()

    widget._refresh_table_names = recording_refresh_table_names  # type: ignore[method-assign]
    widget._bind_current_selection = recording_bind_current_selection  # type: ignore[method-assign]

    widget._on_controller_table_state_changed()

    assert calls == ["refresh_table_names", "bind_current_selection"]
