from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QLabel

from napari_harpy.core.histogram import HistogramResult, HistogramSettings, HistogramTarget
from napari_harpy.widgets.histogram import plot_widget as plot_widget_module
from napari_harpy.widgets.histogram.plot_widget import _HistogramPlotWidget, _ScientificYAxisItem
from napari_harpy.widgets.histogram.styles import (
    HISTOGRAM_BAR_FILL_COLOR,
    HISTOGRAM_CONTRAST_LINE_COLOR,
    HISTOGRAM_CONTRAST_REGION_ALPHA,
    HISTOGRAM_PERCENTILE_LINE_COLOR,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def make_result(
    *,
    counts: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
    settings: HistogramSettings | None = None,
    percentile_values: dict[float, float] | None = None,
) -> HistogramResult:
    target = HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name="0")
    resolved_settings = settings or HistogramSettings(scale="scale0")
    return HistogramResult(
        target=target,
        settings=resolved_settings,
        counts=np.array([2, 1]) if counts is None else counts,
        bin_edges=np.array([0.0, 0.25, 1.0]) if bin_edges is None else bin_edges,
        data_range=(0.0, 1.0),
        percentile_values={} if percentile_values is None else percentile_values,
        resolved_scale=resolved_settings.scale,
    )


def _view_axis_lower_limits(plot_widget: _HistogramPlotWidget) -> tuple[float, float]:
    limits = plot_widget._plot_item.getViewBox().state["limits"]
    return float(limits["xLimits"][0]), float(limits["yLimits"][0])


def test_histogram_plot_widget_renders_bars_from_bin_edges(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(make_result())

    bar_item = plot_widget._bar_item
    assert bar_item is not None
    np.testing.assert_allclose(bar_item.opts["x"], np.array([0.125, 0.625]))
    np.testing.assert_allclose(bar_item.opts["width"], np.array([0.25, 0.75]))
    np.testing.assert_allclose(bar_item.opts["height"], np.array([2.0, 1.0]))
    assert bar_item.opts["brush"].color().name().upper() == HISTOGRAM_BAR_FILL_COLOR.upper()
    assert plot_widget.findChild(QLabel, "histogram_plot_state_label") is None


def test_histogram_plot_widget_view_limits_allow_negative_x_and_prevent_negative_linear_y(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(make_result())

    x_lower_limit, y_lower_limit = _view_axis_lower_limits(plot_widget)
    assert x_lower_limit < 0.0
    assert y_lower_limit == 0.0

    plot_widget.set_contrast_limits((-0.25, 0.8))
    assert plot_widget._contrast_region is not None
    np.testing.assert_allclose(plot_widget._contrast_region.getRegion(), (-0.25, 0.8))

    plot_widget._plot_item.setXRange(-10.0, 0.5, padding=0.0)
    plot_widget._plot_item.setYRange(-5.0, 0.5, padding=0.0)

    x_range, y_range = plot_widget._plot_item.viewRange()
    assert x_range[0] < 0.0
    assert y_range[0] >= 0.0


def test_histogram_plot_widget_replaces_previous_bars(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    plot_widget.set_histogram(make_result())
    first_bar_item = plot_widget._bar_item

    plot_widget.set_histogram(make_result(counts=np.array([4, 3, 2]), bin_edges=np.array([0.0, 1.0, 2.0, 4.0])))

    assert first_bar_item is not plot_widget._bar_item
    assert first_bar_item not in plot_widget._plot_item.items
    assert sum(isinstance(item, pg.BarGraphItem) for item in plot_widget._plot_item.items) == 1
    assert plot_widget._bar_item is not None
    np.testing.assert_allclose(plot_widget._bar_item.opts["x"], np.array([0.5, 1.5, 3.0]))
    np.testing.assert_allclose(plot_widget._bar_item.opts["width"], np.array([1.0, 1.0, 2.0]))


def test_histogram_plot_widget_draws_unlabeled_percentile_markers(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(make_result(percentile_values={1.0: 0.1234, 99.0: 0.9876}))

    marker_lines = plot_widget._percentile_marker_lines
    assert len(marker_lines) == 2
    assert all(not line.movable for line in marker_lines)
    assert all(line.pen.style() == Qt.PenStyle.DashLine for line in marker_lines)
    assert all(line.pen.color().name().upper() == HISTOGRAM_PERCENTILE_LINE_COLOR.upper() for line in marker_lines)
    assert all(not hasattr(line, "label") for line in marker_lines)
    assert [line._tooltip_text for line in marker_lines] == ["p1 = 0.1234", "p99 = 0.9876"]
    assert all(line.toolTip() == "" for line in marker_lines)
    assert all(line.acceptHoverEvents() for line in marker_lines)
    assert all(line.hoverPen.width() > line.pen.width() for line in marker_lines)
    assert all(line.hoverPen.color().name().upper() == HISTOGRAM_PERCENTILE_LINE_COLOR.upper() for line in marker_lines)


def test_histogram_plot_widget_percentile_tooltip_uses_hover_event(qtbot, monkeypatch) -> None:
    shown_tooltips: list[tuple[QPoint, str]] = []
    hidden_tooltips: list[bool] = []
    applied_palettes: list[QPalette] = []

    class FakeToolTip:
        @staticmethod
        def palette() -> QPalette:
            return QPalette()

        @staticmethod
        def setPalette(palette: QPalette) -> None:
            applied_palettes.append(palette)

        @staticmethod
        def showText(pos: QPoint, text: str) -> None:
            shown_tooltips.append((pos, text))

        @staticmethod
        def hideText() -> None:
            hidden_tooltips.append(True)

    class FakeScreenPos:
        def toQPoint(self) -> QPoint:
            return QPoint(12, 34)

    class FakeHoverEvent:
        def __init__(self, *, exit_event: bool) -> None:
            self._exit_event = exit_event

        def isExit(self) -> bool:
            return self._exit_event

        def screenPos(self) -> FakeScreenPos:
            return FakeScreenPos()

    monkeypatch.setattr(plot_widget_module, "QToolTip", FakeToolTip)
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    plot_widget.set_histogram(make_result(percentile_values={50.0: 0.5}))
    marker_line = plot_widget._percentile_marker_lines[0]

    marker_line.hoverEvent(FakeHoverEvent(exit_event=False))

    assert marker_line.mouseHovering is True
    assert len(applied_palettes) == 1
    assert len(shown_tooltips) == 1
    assert shown_tooltips[0][0] == QPoint(12, 34)
    assert "p50 = 0.5" in shown_tooltips[0][1]
    assert "background-color: #343944" in shown_tooltips[0][1]
    assert "border: 1px solid #f0c36a" in shown_tooltips[0][1]

    marker_line.hoverEvent(FakeHoverEvent(exit_event=True))

    assert marker_line.mouseHovering is False
    assert hidden_tooltips == [True]


def test_histogram_plot_widget_omits_out_of_range_percentile_markers(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(make_result(percentile_values={1.0: -1.0, 50.0: 0.5, 99.0: 2.0}))

    marker_lines = plot_widget._percentile_marker_lines
    assert len(marker_lines) == 1
    np.testing.assert_allclose(marker_lines[0].value(), 0.5)
    assert not hasattr(marker_lines[0], "label")


def test_histogram_plot_widget_clears_percentile_markers(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    plot_widget.set_histogram(make_result(percentile_values={50.0: 0.5}))
    marker_line = plot_widget._percentile_marker_lines[0]

    plot_widget.set_histogram(make_result())

    assert plot_widget._percentile_marker_lines == []
    assert marker_line not in plot_widget._plot_item.items


def test_histogram_plot_widget_reset_view_restores_fitted_range(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    result = make_result()
    plot_widget.set_histogram(result)
    fitted_range = plot_widget._plot_item.viewRange()

    plot_widget._plot_item.setXRange(0.2, 0.4, padding=0.0)
    plot_widget._plot_item.setYRange(0.0, 0.5, padding=0.0)
    plot_widget.reset_view(result)

    reset_range = plot_widget._plot_item.viewRange()
    np.testing.assert_allclose(reset_range[0], fitted_range[0])
    np.testing.assert_allclose(reset_range[1], fitted_range[1])


def test_histogram_plot_widget_applies_log_y_and_density_label(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    settings = HistogramSettings(scale="scale0", density=True, log_y=True)

    plot_widget.set_histogram(make_result(settings=settings))

    assert plot_widget._plot_item.ctrl.logYCheck.isChecked()
    assert plot_widget._plot_item.getAxis("left").labelText == "Density"


def test_histogram_plot_widget_log_y_uses_positive_log_bar_coordinates(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    settings = HistogramSettings(scale="scale0", log_y=True)

    plot_widget.set_histogram(
        make_result(
            counts=np.array([0, 10, 100_000]),
            bin_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            settings=settings,
        )
    )

    bar_item = plot_widget._bar_item
    assert bar_item is not None
    np.testing.assert_allclose(bar_item.opts["x"], np.array([1.5, 2.5]))
    np.testing.assert_allclose(bar_item.opts["width"], np.array([1.0, 1.0]))
    np.testing.assert_allclose(bar_item.opts["y0"], np.array([1.0, 1.0]))
    np.testing.assert_allclose(bar_item.opts["y1"], np.array([1.0, 5.0]))
    assert bar_item.opts["height"] is None
    np.testing.assert_allclose(plot_widget._plot_item.viewRange()[1], np.array([1.0, 5.0]))


def test_histogram_plot_widget_log_y_switch_avoids_axis_overflow_warning(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    plot_widget.set_histogram(
        make_result(
            counts=np.array([500_000]),
            bin_edges=np.array([0.0, 1.0]),
        )
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        plot_widget.set_histogram(
            make_result(
                counts=np.array([10, 100_000]),
                bin_edges=np.array([0.0, 1.0, 2.0]),
                settings=HistogramSettings(scale="scale0", log_y=True),
            )
        )

    assert not any("overflow encountered in power" in str(warning.message) for warning in caught_warnings)


def test_histogram_plot_widget_uses_scientific_y_axis_for_large_counts(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    assert isinstance(plot_widget._plot_item.getAxis("left"), _ScientificYAxisItem)

    plot_widget.set_histogram(
        make_result(
            counts=np.array([250_000, 500_000]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
        )
    )

    axis = plot_widget._plot_item.getAxis("left")
    assert axis.tickStrings([0, 100_000, 500_000], 1.0, 100_000) == ["0", "1e5", "5e5"]


def test_histogram_plot_widget_uses_scientific_ticks_for_small_density_values(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(
        make_result(
            counts=np.array([0.001, 0.00001]),
            bin_edges=np.array([0.0, 0.5, 1.0]),
            settings=HistogramSettings(scale="scale0", density=True),
        )
    )

    axis = plot_widget._plot_item.getAxis("left")
    assert axis.autoSIPrefix is False
    assert "(x" not in axis.labelString()
    assert axis.tickStrings([0.001, 0.00001], 1.0, 0.001) == ["1e-3", "1e-5"]


def test_histogram_plot_widget_uses_scientific_ticks_for_log_density_values(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(
        make_result(
            counts=np.array([0.0001, 0.000001, 0.00000001]),
            bin_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            settings=HistogramSettings(scale="scale0", density=True, log_y=True),
        )
    )

    axis = plot_widget._plot_item.getAxis("left")
    x_lower_limit, y_lower_limit = _view_axis_lower_limits(plot_widget)
    assert x_lower_limit < 0.0
    assert y_lower_limit == -8.0
    assert axis.tickStrings([-4, -6, -8], 1.0, 1.0) == ["1e-4", "1e-6", "1e-8"]


def test_histogram_plot_widget_log_y_uses_sparse_readable_tick_labels(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    axis = plot_widget._plot_item.getAxis("left")
    axis.setLogMode(True)
    tick_levels = axis.tickValues(-8.8, -7.65, 280)

    label_ticks = tick_levels[0][1]
    assert axis.style["maxTextLevel"] == 0
    assert axis.tickStrings(label_ticks, 1.0, 1.0) == ["2e-9", "5e-9", "1e-8", "2e-8"]
    assert len(tick_levels) == 2
    assert len(tick_levels[1][1]) > len(label_ticks)


def test_histogram_plot_widget_clear_does_not_render_plot_messages(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_histogram(make_result(settings=HistogramSettings(scale="scale0", log_y=True)))
    plot_widget.clear_histogram()

    assert plot_widget._bar_item is None
    assert not plot_widget._plot_item.ctrl.logYCheck.isChecked()
    assert plot_widget.findChild(QLabel, "histogram_plot_state_label") is None

    plot_widget.set_histogram(make_result(settings=HistogramSettings(scale="scale0", log_y=True)))
    plot_widget.clear_histogram()

    assert plot_widget._bar_item is None
    assert not plot_widget._plot_item.ctrl.logYCheck.isChecked()
    assert plot_widget.findChild(QLabel, "histogram_plot_state_label") is None


def test_histogram_plot_widget_set_contrast_limits_creates_and_updates_region(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)

    plot_widget.set_contrast_limits((0.8, 0.2))

    region = plot_widget._contrast_region
    assert region is not None
    assert region in plot_widget._plot_item.items
    np.testing.assert_allclose(region.getRegion(), (0.2, 0.8))
    assert region.brush.color().alpha() == HISTOGRAM_CONTRAST_REGION_ALPHA
    assert region.lines[0].pen.color().name().upper() == HISTOGRAM_CONTRAST_LINE_COLOR.upper()

    plot_widget.set_contrast_limits((0.1, 0.9))

    assert plot_widget._contrast_region is region
    np.testing.assert_allclose(region.getRegion(), (0.1, 0.9))


def test_histogram_plot_widget_clear_contrast_limits_keeps_histogram_bars(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    plot_widget.set_histogram(make_result())
    bar_item = plot_widget._bar_item
    plot_widget.set_contrast_limits((0.1, 0.9))

    plot_widget.set_contrast_limits(None)

    assert plot_widget._bar_item is bar_item
    assert plot_widget._contrast_region is None
    assert bar_item in plot_widget._plot_item.items


def test_histogram_plot_widget_programmatic_contrast_update_does_not_emit(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    emitted_limits: list[tuple[float, float]] = []
    plot_widget.contrast_limits_dragged.connect(lambda low, high: emitted_limits.append((low, high)))

    plot_widget.set_contrast_limits((0.1, 0.9))
    plot_widget.set_contrast_limits((0.2, 0.8))

    assert emitted_limits == []


def test_histogram_plot_widget_changed_contrast_region_emits_limits(qtbot) -> None:
    plot_widget = _HistogramPlotWidget()
    qtbot.addWidget(plot_widget)
    emitted_limits: list[tuple[float, float]] = []
    plot_widget.contrast_limits_dragged.connect(lambda low, high: emitted_limits.append((low, high)))
    plot_widget.set_contrast_limits((0.1, 0.9))
    assert plot_widget._contrast_region is not None

    plot_widget._contrast_region.setRegion((0.25, 0.75))

    assert emitted_limits == [(0.25, 0.75)]


def test_pyqtgraph_imports_stay_out_of_core_and_controller() -> None:
    core_source = (_REPO_ROOT / "src/napari_harpy/core/histogram.py").read_text()
    controller_source = (_REPO_ROOT / "src/napari_harpy/widgets/histogram/controller.py").read_text()

    assert "pyqtgraph" not in core_source
    assert "pyqtgraph" not in controller_source
