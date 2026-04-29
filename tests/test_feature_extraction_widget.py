from __future__ import annotations

from html import unescape
from types import SimpleNamespace

import numpy as np
from napari.layers import Labels
from qtpy.QtWidgets import QCheckBox, QComboBox, QScrollArea
from spatialdata import SpatialData

import napari_harpy.widgets._feature_extraction_widget as feature_extraction_widget_module
from napari_harpy._app_state import FeatureMatrixWrittenEvent, get_or_create_app_state
from napari_harpy._spatialdata import (
    SpatialDataFeatureExtractionImageDiscovery,
    SpatialDataFeatureExtractionLabelDiscovery,
    SpatialDataImageOption,
    SpatialDataLabelsOption,
)
from napari_harpy.widgets._feature_extraction_widget import FeatureExtractionWidget
from napari_harpy.widgets._viewer_widget import ViewerWidget


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


def get_coordinate_system_checkbox(widget: FeatureExtractionWidget, coordinate_system: str) -> QCheckBox:
    return widget._coordinate_system_checkboxes[coordinate_system]


def check_coordinate_system(widget: FeatureExtractionWidget, coordinate_system: str) -> None:
    get_coordinate_system_checkbox(widget, coordinate_system).setChecked(True)


def select_segmentation(widget: FeatureExtractionWidget, coordinate_system: str, index: int) -> None:
    widget._triplet_card_widgets_by_coordinate_system[coordinate_system].segmentation_combo.setCurrentIndex(index)


def make_label_discovery(
    *,
    coordinate_system: str,
    options: list[SpatialDataLabelsOption],
    unavailable_label_count: int = 0,
) -> SpatialDataFeatureExtractionLabelDiscovery:
    return SpatialDataFeatureExtractionLabelDiscovery(
        coordinate_system=coordinate_system,
        eligible_label_options=options,
        coordinate_system_label_count=len(options) + unavailable_label_count,
        unavailable_label_count=unavailable_label_count,
    )


def make_image_discovery(
    *,
    coordinate_system: str,
    label_name: str,
    options: list[SpatialDataImageOption],
    unavailable_image_count: int = 0,
) -> SpatialDataFeatureExtractionImageDiscovery:
    return SpatialDataFeatureExtractionImageDiscovery(
        coordinate_system=coordinate_system,
        label_name=label_name,
        eligible_image_options=options,
        coordinate_system_image_count=len(options) + unavailable_image_count,
        unavailable_image_count=unavailable_image_count,
    )


def make_viewer_with_shared_sdata(sdata: SpatialData) -> DummyViewer:
    viewer = DummyViewer()
    get_or_create_app_state(viewer).set_sdata(sdata)
    return viewer


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
    assert widget.selected_feature_key == "features"
    assert widget.overwrite_feature_key is False
    assert widget.segmentation_combo.count() == 0
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.table_combo.count() == 0
    assert widget.coordinate_system_combo.count() == 0
    assert widget.calculate_button.isEnabled() is False
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert "shared Harpy state" in unescape(widget.selection_status.text())
    assert all(button.text() != "Rescan Viewer" for button in widget.findChildren(type(widget.calculate_button)))
    assert widget.segmentation_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.image_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.table_combo.sizeAdjustPolicy() == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    assert widget.coordinate_system_combo.sizeAdjustPolicy() == (
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )


def test_feature_extraction_widget_seeds_from_shared_sdata_on_construction(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()
    assert widget.selected_spatialdata is sdata_blobs


def test_feature_extraction_widget_refreshes_when_shared_sdata_changes(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0
    assert "No SpatialData Loaded" in widget.selection_status.text()

    app_state.set_sdata(sdata_blobs)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()
    assert widget.selected_spatialdata is sdata_blobs


def test_feature_extraction_widget_populates_selector_flow_from_spatialdata(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert widget.selected_coordinate_system is None
    assert not widget._triplet_card_widgets_by_coordinate_system
    assert "Choose Coordinate Systems" in widget.selection_status.text()

    check_coordinate_system(widget, "global")

    assert widget.segmentation_combo.count() == 2
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "blobs_labels",
        "blobs_multiscale_labels",
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert widget.segmentation_combo.placeholderText() == "Choose a segmentation mask"
    assert widget.image_combo.count() == 1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == ["No image"]
    assert widget.table_combo.count() == 0
    assert widget.coordinate_system_combo.count() == 1
    assert widget.coordinate_system_combo.itemText(0) == "global"
    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is sdata_blobs
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_coordinate_system == "global"
    assert "Selection Needed" in widget.selection_status.text()
    assert widget.selection_status.toolTip() == ""


def test_feature_extraction_widget_filters_labels_and_images_by_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    label_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}_{label_name}",
                    display_name=f"image_{coordinate_system}_{label_name}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.coordinate_system_combo.count() == 2
    assert widget.coordinate_system_combo.itemText(0) == "aligned"
    assert widget.coordinate_system_combo.currentIndex() == -1
    assert widget.selected_coordinate_system is None
    assert widget.selected_segmentation_name is None
    assert not widget._triplet_card_widgets_by_coordinate_system

    check_coordinate_system(widget, "aligned")

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name is None
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "labels_aligned"
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image"
    ]
    select_segmentation(widget, "aligned", 0)
    assert widget.selected_segmentation_name == "labels_aligned"
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "image_aligned_labels_aligned",
    ]
    assert widget.selected_image_name is None
    assert widget.table_combo.itemText(0) == "table"

    check_coordinate_system(widget, "global")

    assert widget.selected_coordinate_system == "global"
    assert widget.selected_segmentation_name is None
    assert [widget.segmentation_combo.itemText(index) for index in range(widget.segmentation_combo.count())] == [
        "labels_global"
    ]
    assert widget.segmentation_combo.currentIndex() == -1
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image"
    ]
    select_segmentation(widget, "global", 0)
    assert widget.selected_segmentation_name == "labels_global"
    assert [widget.image_combo.itemText(index) for index in range(widget.image_combo.count())] == [
        "No image",
        "image_global_labels_global",
    ]
    assert widget.selected_image_name is None
    assert widget.table_combo.itemText(0) == "table"


def test_feature_extraction_widget_renders_one_card_per_checked_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    label_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}_{label_name}",
                    display_name=f"image_{coordinate_system}_{label_name}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert list(widget._coordinate_system_checkboxes) == ["aligned", "global"]
    assert get_coordinate_system_checkbox(widget, "aligned").isChecked() is False
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    assert list(widget._triplet_card_widgets_by_coordinate_system) == []
    assert widget.selected_coordinate_system is None

    check_coordinate_system(widget, "aligned")
    assert list(widget._triplet_card_widgets_by_coordinate_system) == ["aligned"]
    assert widget.selected_coordinate_system == "aligned"

    check_coordinate_system(widget, "global")

    assert list(widget._triplet_card_widgets_by_coordinate_system) == ["aligned", "global"]
    assert widget.selected_coordinate_system == "global"
    assert widget.segmentation_combo is widget._triplet_card_widgets_by_coordinate_system["global"].segmentation_combo


def test_feature_extraction_widget_restores_explicit_triplet_when_returning_to_coordinate_system(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    aligned_label_1 = SpatialDataLabelsOption(
        label_name="labels_aligned_1",
        display_name="labels_aligned_1",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    aligned_label_2 = SpatialDataLabelsOption(
        label_name="labels_aligned_2",
        display_name="labels_aligned_2",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_label = SpatialDataLabelsOption(
        label_name="labels_global",
        display_name="labels_global",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    image_by_label = {
        "labels_aligned_1": SpatialDataImageOption(
            image_name="image_aligned_1",
            display_name="image_aligned_1",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        "labels_aligned_2": SpatialDataImageOption(
            image_name="image_aligned_2",
            display_name="image_aligned_2",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        "labels_global": SpatialDataImageOption(
            image_name="image_global",
            display_name="image_global",
            sdata=sdata_blobs,
            coordinate_systems=("global",),
        ),
    }

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[aligned_label_1, aligned_label_2] if coordinate_system == "aligned" else [global_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[image_by_label[label_name]],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.selected_coordinate_system is None
    check_coordinate_system(widget, "aligned")

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None

    aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    aligned_widgets.segmentation_combo.setCurrentIndex(1)
    aligned_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_coordinate_system == "aligned"
    assert widget.selected_segmentation_name == "labels_aligned_2"
    assert widget.selected_image_name == "image_aligned_2"

    get_coordinate_system_checkbox(widget, "global").setChecked(True)

    assert widget.selected_coordinate_system == "global"
    assert widget.selected_segmentation_name is None
    assert widget.selected_image_name is None

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    global_widgets.segmentation_combo.setCurrentIndex(0)
    global_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_image_name == "image_global"

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)
    assert "aligned" not in widget._triplet_card_widgets_by_coordinate_system

    get_coordinate_system_checkbox(widget, "aligned").setChecked(True)

    assert widget.selected_coordinate_system == "aligned"
    restored_aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert restored_aligned_widgets.segmentation_combo.currentText() == "labels_aligned_2"
    assert restored_aligned_widgets.image_combo.currentText() == "image_aligned_2"
    assert widget.selected_segmentation_name == "labels_aligned_2"
    assert widget.selected_image_name == "image_aligned_2"


def test_feature_extraction_widget_excludes_duplicate_segmentation_across_visible_cards(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    shared_label = SpatialDataLabelsOption(
        label_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    aligned_only_label = SpatialDataLabelsOption(
        label_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_only_label = SpatialDataLabelsOption(
        label_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[shared_label, aligned_only_label] if coordinate_system == "aligned" else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "aligned", 0)

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    assert [global_widgets.segmentation_combo.itemText(index) for index in range(global_widgets.segmentation_combo.count())] == [
        "global_only"
    ]
    assert "shared_labels" in global_widgets.segmentation_note_label.text()
    assert "aligned" in global_widgets.segmentation_note_label.text()


def test_feature_extraction_widget_excludes_later_selected_shared_segmentation_from_earlier_card_options(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    aligned_only_label = SpatialDataLabelsOption(
        label_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    shared_label = SpatialDataLabelsOption(
        label_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    global_only_label = SpatialDataLabelsOption(
        label_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[aligned_only_label, shared_label] if coordinate_system == "aligned" else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    check_coordinate_system(widget, "global")

    select_segmentation(widget, "aligned", 0)
    select_segmentation(widget, "global", 0)

    aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert aligned_widgets.segmentation_combo.currentText() == "aligned_only"
    assert [aligned_widgets.segmentation_combo.itemText(index) for index in range(aligned_widgets.segmentation_combo.count())] == [
        "aligned_only"
    ]
    assert "shared_labels" in aligned_widgets.segmentation_note_label.text()
    assert "global" in aligned_widgets.segmentation_note_label.text()


def test_feature_extraction_widget_clears_blocked_remembered_segmentation_and_image_on_restore(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    shared_label = SpatialDataLabelsOption(
        label_name="shared_labels",
        display_name="shared_labels",
        sdata=sdata_blobs,
        coordinate_systems=("aligned", "global"),
    )
    aligned_only_label = SpatialDataLabelsOption(
        label_name="aligned_only",
        display_name="aligned_only",
        sdata=sdata_blobs,
        coordinate_systems=("aligned",),
    )
    global_only_label = SpatialDataLabelsOption(
        label_name="global_only",
        display_name="global_only",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    image_by_selection = {
        ("aligned", "shared_labels"): SpatialDataImageOption(
            image_name="image_aligned_shared",
            display_name="image_aligned_shared",
            sdata=sdata_blobs,
            coordinate_systems=("aligned",),
        ),
        ("global", "shared_labels"): SpatialDataImageOption(
            image_name="image_global_shared",
            display_name="image_global_shared",
            sdata=sdata_blobs,
            coordinate_systems=("global",),
        ),
    }

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[shared_label, aligned_only_label] if coordinate_system == "aligned" else [shared_label, global_only_label],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[] if (coordinate_system, label_name) not in image_by_selection else [image_by_selection[(coordinate_system, label_name)]],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(True)

    global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    assert global_widgets.segmentation_combo.currentText() == "shared_labels"
    assert global_widgets.image_combo.currentText() == "image_global_shared"

    restored_aligned_widgets = widget._triplet_card_widgets_by_coordinate_system["aligned"]
    assert restored_aligned_widgets.segmentation_combo.currentIndex() == -1
    assert restored_aligned_widgets.segmentation_combo.placeholderText() == "Choose a segmentation mask"
    assert restored_aligned_widgets.image_combo.currentText() == "No image"
    assert "shared_labels" in restored_aligned_widgets.segmentation_note_label.text()
    assert "global" in restored_aligned_widgets.segmentation_note_label.text()
    assert widget.selected_segmentation_name is None
    assert widget._remembered_card_selection_by_coordinate_system["aligned"].label_identity is None
    assert widget._remembered_card_selection_by_coordinate_system["aligned"].image_identity is None


def test_feature_extraction_widget_hides_channel_selection_without_image(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.selected_image_name is None
    assert widget.selected_extraction_channel_names is None
    assert widget.selected_extraction_channel_indices is None
    assert widget.channel_selection_label.isHidden()
    assert widget.channel_selection_container.isHidden()


def test_feature_extraction_widget_shows_selected_image_channels_and_defaults_to_all_selected(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)

    assert widget.selected_image_name == "blobs_image"
    assert [checkbox.text() for checkbox in widget._batch_channel_checkboxes] == ["0", "1", "2"]
    assert widget.selected_extraction_channel_names == ("0", "1", "2")
    assert widget.selected_extraction_channel_indices == (0, 1, 2)
    assert not widget.channel_selection_label.isHidden()
    assert not widget.channel_selection_container.isHidden()


def test_feature_extraction_widget_hides_channel_selection_when_selected_image_has_no_channel_axis(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    monkeypatch.setattr(feature_extraction_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: [])

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)

    assert widget.selected_image_name == "blobs_image"
    assert widget.selected_extraction_channel_names is None
    assert widget.selected_extraction_channel_indices is None
    assert widget.channel_selection_label.isHidden()
    assert widget.channel_selection_container.isHidden()


def test_feature_extraction_widget_skips_channel_less_images_when_choosing_shared_reference_schema(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    label_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}",
                    display_name=f"image_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: [] if image_name == "image_aligned" else ["0", "1", "2"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)

    assert widget.selected_extraction_channel_names == ("0", "1", "2")
    assert [checkbox.text() for checkbox in widget._batch_channel_checkboxes] == ["0", "1", "2"]
    assert not widget.channel_selection_label.isHidden()
    assert not widget.channel_selection_container.isHidden()
    assert "image_aligned" in widget.channel_selection_note_label.text()
    assert "does not expose channels" in widget.channel_selection_note_label.text()


def test_feature_extraction_widget_keeps_shared_channel_selector_visible_for_incompatible_later_image(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    label_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}",
                    display_name=f"image_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["0", "1", "2"] if image_name == "image_aligned" else ["0", "2"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[1].setChecked(False)

    assert widget.selected_extraction_channel_names == ("0", "2")

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)

    assert widget.selected_extraction_channel_names == ("0", "2")
    assert [checkbox.text() for checkbox in widget._batch_channel_checkboxes] == ["0", "1", "2"]
    assert not widget.channel_selection_label.isHidden()
    assert not widget.channel_selection_container.isHidden()
    assert "image_global" in widget.channel_selection_note_label.text()
    assert "does not match the shared ordered channel names" in widget.channel_selection_note_label.text()


def test_feature_extraction_widget_remembers_shared_channel_selection_by_schema(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)

    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_coordinate_system_names_from_sdata",
        lambda sdata: ["aligned", "global"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_label_discovery_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: make_label_discovery(
            coordinate_system=coordinate_system,
            options=[
                SpatialDataLabelsOption(
                    label_name=f"labels_{coordinate_system}",
                    display_name=f"labels_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_spatialdata_feature_extraction_image_discovery_for_coordinate_system_and_label_from_sdata",
        lambda *, sdata, coordinate_system, label_name: make_image_discovery(
            coordinate_system=coordinate_system,
            label_name=label_name,
            options=[
                SpatialDataImageOption(
                    image_name=f"image_{coordinate_system}",
                    display_name=f"image_{coordinate_system}",
                    sdata=sdata,
                    coordinate_systems=(coordinate_system,),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_annotating_table_names",
        lambda sdata, label_name: ["table"],
    )
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: ["0", "1", "2"] if image_name == "image_global" else ["a", "b"],
    )

    widget = FeatureExtractionWidget(viewer)
    qtbot.addWidget(widget)

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget._triplet_card_widgets_by_coordinate_system["global"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[1].setChecked(False)

    assert widget.selected_extraction_channel_names == ("0", "2")

    get_coordinate_system_checkbox(widget, "global").setChecked(False)

    check_coordinate_system(widget, "aligned")
    select_segmentation(widget, "aligned", 0)
    widget._triplet_card_widgets_by_coordinate_system["aligned"].image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[0].setChecked(False)

    assert widget.selected_extraction_channel_names == ("b",)

    get_coordinate_system_checkbox(widget, "aligned").setChecked(False)
    get_coordinate_system_checkbox(widget, "global").setChecked(True)

    restored_global_widgets = widget._triplet_card_widgets_by_coordinate_system["global"]
    select_segmentation(widget, "global", 0)
    restored_global_widgets.image_combo.setCurrentIndex(1)

    assert widget.selected_extraction_channel_names == ("0", "2")


def test_feature_extraction_widget_surfaces_duplicate_channel_names_as_batch_error_for_intensity_features(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind(*args, **kwargs):
        bind_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind = fake_bind  # type: ignore[method-assign]
    monkeypatch.setattr(
        feature_extraction_widget_module,
        "get_image_channel_names_from_sdata",
        lambda sdata, image_name: (_ for _ in ()).throw(
            ValueError(
                "Image element `blobs_image` exposes duplicate channel names (`dup`), "
                "which napari-harpy does not support. "
                "Update the channel names in the SpatialData object with "
                "`sdata.set_channel_names(...)`."
            )
        ),
    )

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.channel_selection_label.isHidden()
    assert widget.channel_selection_container.isHidden()
    assert "duplicate channel names" in widget.intensity_features_hint.text()
    assert "sdata.set_channel_names(...)" in widget.intensity_features_hint.text()
    assert bind_calls
    args, kwargs = bind_calls[-1]
    assert args == (
        sdata_blobs,
        "blobs_labels",
        None,
        "table",
        "global",
        ("mean",),
        "features",
    )
    assert kwargs == {"channels": None, "overwrite_feature_key": False}


def test_feature_extraction_widget_channel_selection_is_independent_from_viewer_overlay_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    feature_widget = FeatureExtractionWidget(viewer)
    viewer_widget = ViewerWidget(viewer)

    qtbot.addWidget(feature_widget)
    qtbot.addWidget(viewer_widget)

    check_coordinate_system(feature_widget, "global")
    select_segmentation(feature_widget, "global", 0)
    feature_widget.image_combo.setCurrentIndex(1)
    feature_widget._batch_channel_checkboxes[0].setChecked(False)

    assert feature_widget.selected_extraction_channel_indices == (1, 2)

    image_card = next(card for card in viewer_widget.image_cards if card.image_name == "blobs_image")
    image_card.overlay_toggle.setChecked(True)
    image_card.channel_checkboxes[0].setChecked(True)
    image_card.channel_checkboxes[1].setChecked(False)
    image_card.channel_checkboxes[2].setChecked(False)

    assert image_card.get_selected_overlay_channels() == [0]
    assert feature_widget.selected_extraction_channel_indices == (1, 2)


def test_feature_extraction_widget_shortens_long_identifiers_in_selection_status(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    long_segmentation_name = "blobs_labels_long_name_blobs_labels_long_name_blobs_labels_long_name"
    long_image_name = "blobs_image_long_name_blobs_image_long_name_blobs_image_long_name"
    long_table_name = "table_long_name_table_long_name_table_long_name_table_long_name"
    long_coordinate_system = "global_coordinate_system_coordinate_system_coordinate_system"

    widget._selected_label_option = SpatialDataLabelsOption(
        label_name=long_segmentation_name,
        display_name=long_segmentation_name,
        sdata=sdata_blobs,
        coordinate_systems=(long_coordinate_system,),
    )
    widget._selected_image_option = SpatialDataImageOption(
        image_name=long_image_name,
        display_name=long_image_name,
        sdata=sdata_blobs,
        coordinate_systems=(long_coordinate_system,),
    )
    widget._selected_table_name = long_table_name
    widget._selected_coordinate_system = long_coordinate_system
    widget._table_binding_error = None

    widget._update_primary_status_card()

    status_text = widget.selection_status.text()
    status_tooltip = unescape(widget.selection_status.toolTip()).replace("&#8203;", "").replace("\u200b", "")

    assert "Selection Ready" in status_text
    assert "…" in status_text
    assert long_segmentation_name not in status_text
    assert long_image_name not in status_text
    assert long_table_name not in status_text
    assert long_coordinate_system not in status_text
    assert long_segmentation_name in status_tooltip
    assert long_image_name in status_tooltip
    assert long_table_name in status_tooltip
    assert long_coordinate_system in status_tooltip


def test_feature_extraction_widget_blocks_when_selected_segmentation_has_no_linked_table(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
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


def test_feature_extraction_widget_uses_table_binding_error_as_status_tooltip(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget._selected_label_option = SpatialDataLabelsOption(
        label_name="blobs_labels",
        display_name="blobs_labels",
        sdata=sdata_blobs,
        coordinate_systems=("global",),
    )
    widget._selected_table_name = "table"
    widget._selected_coordinate_system = "global"
    widget._table_binding_error = (
        "Table `table` annotates segmentation `other_labels`, not `blobs_labels`."
    )

    widget._update_primary_status_card()

    tooltip = unescape(widget.selection_status.toolTip()).replace("&#8203;", "").replace("\u200b", "")
    assert "Table Binding Issue" in widget.selection_status.text()
    assert "annotates segmentation `other_labels`" in tooltip


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
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)
    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)
    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)

    assert widget.selected_image_name == "blobs_image"
    assert widget.intensity_features_hint.isHidden()


def test_feature_extraction_widget_rebinds_controller_when_inputs_change(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind(*args, **kwargs):
        bind_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind = fake_bind  # type: ignore[method-assign]

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
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
    assert kwargs == {"channels": None, "overwrite_feature_key": False}


def test_feature_extraction_widget_binds_selected_channels_into_controller(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    bind_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_bind(*args, **kwargs):
        bind_calls.append((args, kwargs))
        return True

    widget._feature_extraction_controller.bind = fake_bind  # type: ignore[method-assign]

    check_coordinate_system(widget, "global")
    select_segmentation(widget, "global", 0)
    widget.image_combo.setCurrentIndex(1)
    widget._batch_channel_checkboxes[1].setChecked(False)

    assert bind_calls
    args, kwargs = bind_calls[-1]
    assert args == (
        sdata_blobs,
        "blobs_labels",
        "blobs_image",
        "table",
        "global",
        (),
        "features",
    )
    assert kwargs == {"channels": ("0", "2"), "overwrite_feature_key": False}


def test_feature_extraction_widget_keeps_calculate_disabled_during_slice_two_refactor(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.calculate_button.isEnabled() is False

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is False
    assert "temporarily disabled" in widget.calculate_button.toolTip()


def test_feature_extraction_widget_keeps_calculate_disabled_for_intensity_features_without_image(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_mean").setChecked(True)
    widget.output_key_line_edit.setText("features")

    assert widget.calculate_button.isEnabled() is False
    assert "temporarily disabled" in widget.calculate_button.toolTip()
    assert "choose an image" in widget.intensity_features_hint.text()


def test_feature_extraction_widget_blocks_when_no_coordinate_system_is_checked(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    widget.findChild(QCheckBox, "feature_checkbox_area").setChecked(True)
    widget.output_key_line_edit.setText("features")
    assert get_coordinate_system_checkbox(widget, "global").isChecked() is False
    widget._bind_current_selection()

    assert widget.selected_coordinate_system is None
    assert "Choose Coordinate System" in widget.selection_status.text()
    assert widget.calculate_button.isEnabled() is False
    assert "temporarily disabled" in widget.calculate_button.toolTip()


def test_feature_extraction_widget_does_not_launch_controller_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
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

    assert calls == []


def test_feature_extraction_widget_does_not_launch_non_overwrite_run_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
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

    assert overwrite_calls == []


def test_feature_extraction_widget_does_not_prompt_before_overwriting_while_calculation_is_disabled(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
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

    assert prompt_calls == []
    assert overwrite_calls == []


def test_feature_extraction_widget_cancelled_overwrite_does_not_launch_calculation(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
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
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
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


def test_feature_extraction_widget_reemits_feature_matrix_writes_to_shared_app_state(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    event = FeatureMatrixWrittenEvent(
        sdata=sdata_blobs,
        table_name="table",
        feature_key="features_new",
        change_kind="created",
    )

    with qtbot.waitSignal(app_state.feature_matrix_written) as blocker:
        widget._on_controller_feature_matrix_written(event)

    assert blocker.args == [event]
    assert app_state.is_table_dirty(sdata_blobs, "table") is True


def test_feature_extraction_widget_clears_when_shared_sdata_is_cleared(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    viewer = make_viewer_with_shared_sdata(sdata_blobs)
    app_state = get_or_create_app_state(viewer)
    widget = FeatureExtractionWidget(viewer)

    qtbot.addWidget(widget)

    assert widget.segmentation_combo.count() == 0

    app_state.clear_sdata()

    assert widget.selected_segmentation_name is None
    assert widget.selected_spatialdata is None
    assert widget.selected_image_name is None
    assert widget.selected_table_name is None
    assert widget.selected_coordinate_system is None
    assert widget.segmentation_combo.count() == 0
    assert widget.segmentation_combo.isEnabled() is False
    assert widget.image_combo.count() == 1
    assert widget.image_combo.itemText(0) == "No image"
    assert widget.image_combo.isEnabled() is False
    assert widget.table_combo.count() == 0
    assert widget.table_combo.isEnabled() is False
    assert widget.coordinate_system_combo.count() == 0
    assert widget.coordinate_system_combo.isEnabled() is False
    assert widget.calculate_button.isEnabled() is False
    assert "No SpatialData Loaded" in widget.selection_status.text()
    assert "shared Harpy state" in unescape(widget.selection_status.text())
