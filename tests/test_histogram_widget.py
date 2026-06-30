from __future__ import annotations

from html import unescape
from types import SimpleNamespace

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QComboBox, QLabel, QWidget
from spatialdata import SpatialData

import napari_harpy.widgets.histogram.widget as histogram_widget_module
from napari_harpy._app_state import get_or_create_app_state
from napari_harpy.core.histogram import HistogramResult
from napari_harpy.core.spatialdata import get_spatialdata_image_options_for_coordinate_system_from_sdata
from napari_harpy.widgets.histogram.controller import HistogramJob, HistogramJobResult
from napari_harpy.widgets.histogram.widget import HistogramWidget


class DummyViewer:
    pass


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[object] = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[object] | None = None) -> None:
        super().__init__(layers or [])
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class LayerListDummyViewer:
    def __init__(self, layers: list[object] | None = None) -> None:
        self.layers = DummyLayers(layers)

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


class _DeferredWorker(QObject):
    returned = Signal(object)
    errored = Signal(object)
    finished = Signal()

    def __init__(self, result: HistogramJobResult | None = None) -> None:
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


def combo_texts(combo: QComboBox) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def set_combo_data(combo: QComboBox, data: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"Combo data {data!r} is not available.")


def tooltip_text(widget) -> str:
    return unescape(widget.toolTip()).replace("&#8203;", "").replace("\u200b", "")


def make_widget_with_sdata(qtbot, sdata: SpatialData) -> HistogramWidget:
    viewer = DummyViewer()
    get_or_create_app_state(viewer).set_sdata(sdata)
    widget = HistogramWidget(viewer)
    qtbot.addWidget(widget)
    return widget


def make_widget_with_viewer_and_sdata(qtbot, viewer: object, sdata: SpatialData) -> HistogramWidget:
    get_or_create_app_state(viewer).set_sdata(sdata)
    widget = HistogramWidget(viewer)
    qtbot.addWidget(widget)
    return widget


def add_valid_histogram_card(widget: HistogramWidget) -> tuple[str, object]:
    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]
    set_combo_data(card.image_combo, "blobs_image")
    set_combo_data(card.channel_combo, "0")
    return card_id, card


def make_job_result(job: HistogramJob) -> HistogramJobResult:
    return HistogramJobResult(
        card_id=job.card_id,
        job_id=job.job_id,
        target=job.target,
        settings=job.settings,
        result=HistogramResult(
            target=job.target,
            settings=job.settings,
            counts=np.array([2, 1]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            data_range=(0.0, 1.0),
            percentile_values={},
            resolved_scale=job.settings.scale,
        ),
    )


def calculate_card(widget: HistogramWidget, qtbot, card) -> None:
    deferred_workers: list[_DeferredWorker] = []

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        worker = _DeferredWorker(make_job_result(job))
        deferred_workers.append(worker)
        return worker

    widget._histogram_controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]
    qtbot.mouseClick(card.calculate_button, Qt.MouseButton.LeftButton)
    deferred_workers[0].emit_returned()


def make_overlay_layer(*, name: str = "blobs_image[0]", contrast_limits: tuple[float, float] = (0.0, 1.0)) -> Image:
    layer = Image(np.zeros((4, 4), dtype=np.float32), name=name)
    layer.contrast_limits = contrast_limits
    return layer


def register_overlay_layer(
    widget: HistogramWidget,
    layer: Image,
    sdata: SpatialData,
    *,
    channel_name: str = "0",
) -> None:
    widget.app_state.viewer_adapter.register_image_layer(
        layer,
        sdata=sdata,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=int(channel_name),
        channel_name=channel_name,
    )


def test_histogram_widget_instantiates_without_viewer(qtbot) -> None:
    widget = HistogramWidget()

    qtbot.addWidget(widget)

    assert widget.app_state.sdata is None
    assert widget.card_count == 0
    assert not widget.empty_state_label.isHidden()
    assert widget.add_button.text() == "Add histogram"
    assert widget.findChild(QLabel, "histogram_header_logo") is not None
    assert widget.findChild(QLabel, "histogram_title") is None
    action_row = widget.findChild(QWidget, "histogram_header_action_row")
    assert action_row is not None
    assert action_row.layout().itemAt(0).widget() is widget.add_button
    assert action_row.layout().itemAt(1).spacerItem() is not None


def test_histogram_widget_attaches_to_shared_app_state(qtbot) -> None:
    viewer = DummyViewer()
    app_state = get_or_create_app_state(viewer)

    widget = HistogramWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.app_state is app_state


def test_histogram_cards_can_be_added_and_removed_without_mutating_sdata(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    image_names_before = tuple(sdata_blobs.images)

    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]

    assert widget.card_count == 1
    assert widget.empty_state_label.isHidden()
    assert "Remove histogram" in tooltip_text(card.remove_button)
    assert card.remove_button.accessibleName() == "Remove histogram"
    assert widget.findChild(QWidget, f"histogram_card_header_{card_id}").styleSheet() == (
        histogram_widget_module._CARD_SUBCONTAINER_STYLESHEET
    )
    assert widget.findChild(QWidget, f"histogram_action_row_{card_id}").styleSheet() == (
        histogram_widget_module._CARD_SUBCONTAINER_STYLESHEET
    )

    card.remove_button.click()

    assert widget.card_count == 0
    assert not widget.empty_state_label.isHidden()
    assert tuple(sdata_blobs.images) == image_names_before


def test_histogram_widget_populates_target_selectors_and_starts_controller_job(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    captured_jobs: list[HistogramJob] = []
    deferred_workers: list[_DeferredWorker] = []
    widget = make_widget_with_sdata(qtbot, sdata_blobs)

    card_id, card = add_valid_histogram_card(widget)

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        captured_jobs.append(job)
        worker = _DeferredWorker(make_job_result(job))
        deferred_workers.append(worker)
        return worker

    widget._histogram_controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]

    assert card.coordinate_system_combo.currentText() == "global"
    assert combo_texts(card.image_combo) == ["blobs_image", "blobs_multiscale_image"]
    assert combo_texts(card.channel_combo) == ["0", "1", "2"]
    assert card.calculate_button.isEnabled()
    assert card.calculate_button.styleSheet() == histogram_widget_module.CALCULATE_BUTTON_STYLESHEET

    qtbot.mouseClick(card.calculate_button, Qt.MouseButton.LeftButton)

    assert len(captured_jobs) == 1
    assert captured_jobs[0].card_id == card_id
    assert captured_jobs[0].sdata is sdata_blobs
    assert captured_jobs[0].target.coordinate_system == "global"
    assert captured_jobs[0].target.image_name == "blobs_image"
    assert captured_jobs[0].target.channel_name == "0"
    assert captured_jobs[0].settings.bins == 256
    assert captured_jobs[0].settings.scale == "scale0"
    assert captured_jobs[0].settings.percentiles == ()
    assert deferred_workers[0].started is True
    assert "Calculating histogram." in card.status_label.text()

    deferred_workers[0].emit_returned()

    assert "Histogram calculated." in card.status_label.text()
    assert card.plot_widget._bar_item is not None
    np.testing.assert_allclose(card.plot_widget._bar_item.opts["x"], np.array([0.25, 0.75]))
    np.testing.assert_allclose(card.plot_widget._bar_item.opts["height"], np.array([2.0, 1.0]))


def test_histogram_widget_preserves_existing_plot_while_recalculating(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    deferred_workers: list[_DeferredWorker] = []
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    def capture_worker(job: HistogramJob) -> _DeferredWorker:
        worker = _DeferredWorker(make_job_result(job))
        deferred_workers.append(worker)
        return worker

    widget._histogram_controller._create_histogram_worker = capture_worker  # type: ignore[method-assign]

    qtbot.mouseClick(card.calculate_button, Qt.MouseButton.LeftButton)
    deferred_workers[0].emit_returned()
    first_bar_item = card.plot_widget._bar_item
    assert first_bar_item is not None

    qtbot.mouseClick(card.calculate_button, Qt.MouseButton.LeftButton)

    assert "Calculating histogram." in card.status_label.text()
    assert card.plot_widget._bar_item is first_bar_item


def test_histogram_widget_refresh_preserves_valid_target_and_clears_invalid_downstream_selection(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    widget._on_sdata_changed(sdata_blobs)

    assert card.image_combo.currentData() == "blobs_image"
    assert card.channel_combo.currentData() == "0"

    monkeypatch.setattr(
        histogram_widget_module,
        "get_spatialdata_image_options_for_coordinate_system_from_sdata",
        lambda *, sdata, coordinate_system: [
            option
            for option in get_spatialdata_image_options_for_coordinate_system_from_sdata(
                sdata=sdata,
                coordinate_system=coordinate_system,
            )
            if option.image_name == "blobs_multiscale_image"
        ],
    )

    widget._on_sdata_changed(sdata_blobs)

    assert card.image_combo.currentIndex() == -1
    assert card.channel_combo.currentIndex() == -1
    assert not card.calculate_button.isEnabled()


def test_histogram_widget_rejects_images_without_channel_names(
    qtbot,
    monkeypatch,
    sdata_blobs: SpatialData,
) -> None:
    monkeypatch.setattr(histogram_widget_module, "get_image_channel_names_from_sdata", lambda sdata, image_name: [])
    widget = make_widget_with_sdata(qtbot, sdata_blobs)

    card_id = widget.add_histogram_card()
    card = widget._cards[card_id]
    set_combo_data(card.image_combo, "blobs_image")

    assert card.channel_combo.count() == 0
    assert not card.calculate_button.isEnabled()
    assert "does not expose channel names" in card.status_label.text()


def test_histogram_widget_settings_are_collapsed_optional_and_card_local(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _first_id, first_card = add_valid_histogram_card(widget)
    _second_id, second_card = add_valid_histogram_card(widget)

    assert first_card.settings_panel.isHidden()
    assert first_card.settings_toggle.text() == "Settings"
    assert "scale: scale0" in tooltip_text(first_card.settings_toggle)
    assert "bins: 256" in tooltip_text(first_card.settings_toggle)

    first_card.bins_spin.setValue(512)
    first_card.value_range_low_edit.setText("0.1")
    first_card.value_range_high_edit.setText("0.9")
    first_card.percentile_min_edit.setText("1")
    first_card.percentile_max_edit.setText("99")
    first_card.density_checkbox.setChecked(True)
    first_card.exclude_zeros_checkbox.setChecked(True)
    first_card.log_y_checkbox.setChecked(True)
    second_card.bins_spin.setValue(128)

    first_binding = widget._resolve_card_binding(first_card)
    second_binding = widget._resolve_card_binding(second_card)

    assert first_binding.settings is not None
    assert second_binding.settings is not None
    assert first_binding.settings.bins == 512
    assert first_binding.settings.value_range == (0.1, 0.9)
    assert first_binding.settings.percentiles == (1.0, 99.0)
    assert first_binding.settings.density is True
    assert first_binding.settings.exclude_zeros is True
    assert first_binding.settings.log_y is True
    assert second_binding.settings.bins == 128
    assert second_binding.settings.value_range is None
    assert second_binding.settings.percentiles == ()
    first_settings_tooltip = tooltip_text(first_card.settings_toggle)
    assert first_card.settings_toggle.text() == "Settings"
    assert "value_range: (0.1, 0.9)" in first_settings_tooltip
    assert "density: True" in first_settings_tooltip
    assert "exclude_zeros: True" in first_settings_tooltip
    assert "log_y: True" in first_settings_tooltip
    assert "percentiles: 1, 99" in first_settings_tooltip

    first_card.reset_settings_button.click()

    reset_settings = widget._resolve_card_binding(first_card).settings
    assert reset_settings is not None
    assert reset_settings.bins == 256
    assert reset_settings.value_range is None
    assert reset_settings.percentiles == ()
    assert reset_settings.density is False
    assert reset_settings.exclude_zeros is False
    assert reset_settings.log_y is False
    assert reset_settings.scale == "scale0"
    second_settings = widget._resolve_card_binding(second_card).settings
    assert second_settings is not None
    assert second_settings.bins == 128


def test_histogram_widget_settings_panel_stays_compact_at_dock_width(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    closed_icon_key = card.settings_toggle.icon().cacheKey()
    assert card.settings_toggle.arrowType() == Qt.ArrowType.NoArrow
    assert not card.settings_toggle.icon().isNull()

    card.settings_toggle.setChecked(True)
    card.settings_panel.layout().activate()

    assert card.settings_toggle.icon().cacheKey() != closed_icon_key
    assert "QToolButton:checked" in card.settings_toggle.styleSheet()
    assert "min-height: 26px" in card.settings_toggle.styleSheet()
    assert "padding: 3px 10px" in card.settings_toggle.styleSheet()
    assert "font-size: 13px" in card.settings_toggle.styleSheet()
    assert card.settings_panel.sizeHint().width() <= 340
    assert card.scale_combo.maximumWidth() == 150
    assert card.value_range_low_edit.maximumWidth() == 112
    assert card.value_range_high_edit.maximumWidth() == 112


def test_histogram_widget_invalid_optional_settings_disable_calculate(qtbot, sdata_blobs: SpatialData) -> None:
    widget = make_widget_with_sdata(qtbot, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    card.value_range_low_edit.setText("0.1")

    assert not card.calculate_button.isEnabled()
    assert "requires both low and high values" in card.status_label.text()


def test_histogram_widget_syncs_contrast_limits_with_unique_overlay_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_overlay_layer(contrast_limits=(0.1, 0.8))
    viewer = LayerListDummyViewer([layer])
    widget = make_widget_with_viewer_and_sdata(qtbot, viewer, sdata_blobs)
    register_overlay_layer(widget, layer, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    calculate_card(widget, qtbot, card)

    assert card.plot_widget._contrast_region is not None
    np.testing.assert_allclose(card.plot_widget._contrast_region.getRegion(), (0.1, 0.8))
    assert "Contrast synced to napari overlay layer." in card.status_label.text()

    card.plot_widget.contrast_limits_dragged.emit(0.2, 0.7)
    np.testing.assert_allclose(layer.contrast_limits, (0.2, 0.7))

    layer.contrast_limits = (0.3, 0.6)
    np.testing.assert_allclose(card.plot_widget._contrast_region.getRegion(), (0.3, 0.6))


def test_histogram_widget_disables_contrast_sync_for_duplicate_overlay_layers(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    first_layer = make_overlay_layer(name="first")
    second_layer = make_overlay_layer(name="second")
    viewer = LayerListDummyViewer([first_layer, second_layer])
    widget = make_widget_with_viewer_and_sdata(qtbot, viewer, sdata_blobs)
    register_overlay_layer(widget, first_layer, sdata_blobs)
    register_overlay_layer(widget, second_layer, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)

    calculate_card(widget, qtbot, card)

    assert card.plot_widget._contrast_region is None
    assert card.plot_widget._bar_item is not None
    assert "multiple matching overlay layers" in card.status_label.text()


def test_histogram_widget_allows_two_cards_to_share_one_overlay_layer(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_overlay_layer()
    viewer = LayerListDummyViewer([layer])
    widget = make_widget_with_viewer_and_sdata(qtbot, viewer, sdata_blobs)
    register_overlay_layer(widget, layer, sdata_blobs)
    _first_card_id, first_card = add_valid_histogram_card(widget)
    _second_card_id, second_card = add_valid_histogram_card(widget)

    calculate_card(widget, qtbot, first_card)
    calculate_card(widget, qtbot, second_card)

    layer.contrast_limits = (0.25, 0.75)

    assert first_card.plot_widget._contrast_region is not None
    assert second_card.plot_widget._contrast_region is not None
    np.testing.assert_allclose(first_card.plot_widget._contrast_region.getRegion(), (0.25, 0.75))
    np.testing.assert_allclose(second_card.plot_widget._contrast_region.getRegion(), (0.25, 0.75))


def test_histogram_widget_layer_removal_clears_contrast_sync_without_clearing_histogram(
    qtbot,
    sdata_blobs: SpatialData,
) -> None:
    layer = make_overlay_layer()
    viewer = LayerListDummyViewer([layer])
    widget = make_widget_with_viewer_and_sdata(qtbot, viewer, sdata_blobs)
    register_overlay_layer(widget, layer, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)
    calculate_card(widget, qtbot, card)

    viewer.layers.remove(layer)

    assert card.contrast_sync_state is None
    assert card.plot_widget._contrast_region is None
    assert card.plot_widget._bar_item is not None
    assert "open this image in overlay mode" in card.status_label.text()


def test_histogram_widget_settings_change_clears_contrast_sync(qtbot, sdata_blobs: SpatialData) -> None:
    layer = make_overlay_layer()
    viewer = LayerListDummyViewer([layer])
    widget = make_widget_with_viewer_and_sdata(qtbot, viewer, sdata_blobs)
    register_overlay_layer(widget, layer, sdata_blobs)
    _card_id, card = add_valid_histogram_card(widget)
    calculate_card(widget, qtbot, card)

    card.bins_spin.setValue(512)

    assert card.contrast_sync_state is None
    assert card.plot_widget._contrast_region is None
    assert card.plot_widget._bar_item is None
    assert "Target or settings changed. Calculate again." in card.status_label.text()
