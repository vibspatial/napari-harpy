from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QLabel

from napari_harpy.core.histogram import HistogramResult, HistogramSettings, HistogramTarget
from napari_harpy.widgets.histogram.plot_widget import _HistogramPlotWidget, _ScientificYAxisItem
from napari_harpy.widgets.histogram.styles import HISTOGRAM_BAR_FILL_COLOR

_REPO_ROOT = Path(__file__).resolve().parents[1]


def make_result(
    *,
    counts: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
    settings: HistogramSettings | None = None,
) -> HistogramResult:
    target = HistogramTarget(coordinate_system="global", image_name="blobs_image", channel_name="0")
    resolved_settings = settings or HistogramSettings(scale="scale0")
    return HistogramResult(
        target=target,
        settings=resolved_settings,
        counts=np.array([2, 1]) if counts is None else counts,
        bin_edges=np.array([0.0, 0.25, 1.0]) if bin_edges is None else bin_edges,
        data_range=(0.0, 1.0),
        percentile_values={},
        resolved_scale=resolved_settings.scale,
    )


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


def test_pyqtgraph_imports_stay_out_of_core_and_controller() -> None:
    core_source = (_REPO_ROOT / "src/napari_harpy/core/histogram.py").read_text()
    controller_source = (_REPO_ROOT / "src/napari_harpy/widgets/histogram/controller.py").read_text()

    assert "pyqtgraph" not in core_source
    assert "pyqtgraph" not in controller_source
