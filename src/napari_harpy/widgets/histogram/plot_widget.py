from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pyqtgraph as pg
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QGridLayout, QSizePolicy, QWidget

from napari_harpy.core.histogram import HistogramResult
from napari_harpy.widgets.histogram.styles import (
    HISTOGRAM_AXIS_GRID_COLOR,
    HISTOGRAM_AXIS_TEXT_COLOR,
    HISTOGRAM_BAR_EDGE_COLOR,
    HISTOGRAM_BAR_FILL_ALPHA,
    HISTOGRAM_BAR_FILL_COLOR,
    HISTOGRAM_PLOT_BACKGROUND_COLOR,
)

_PLOT_MIN_HEIGHT = 150
_PLOT_MAX_HEIGHT = 180
_SCIENTIFIC_TICK_HIGH_ABS = 10_000
_SCIENTIFIC_TICK_LOW_ABS = 0.001
_SCIENTIFIC_TICK_MAX_DECIMALS = 2
_LOG_Y_SINGLE_DECADE_PADDING = 0.5


class _ScientificYAxisItem(pg.AxisItem):
    """Axis item that keeps large histogram counts compact."""

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if self.logMode:
            return [_format_log_tick(value, scale) for value in values]

        default_strings = super().tickStrings(values, scale, spacing)
        strings: list[str] = []
        for value, default_string in zip(values, default_strings, strict=True):
            scaled_value = value * scale
            if _should_use_scientific_tick(scaled_value):
                strings.append(_format_scientific_tick(scaled_value))
            else:
                strings.append(default_string)
        return strings


class _HistogramPlotWidget(QWidget):
    """Small pyqtgraph wrapper for card-local histogram rendering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("histogram_plot_widget")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(_PLOT_MIN_HEIGHT)
        self.setMaximumHeight(_PLOT_MAX_HEIGHT)

        self._plot_widget = pg.PlotWidget(axisItems={"left": _ScientificYAxisItem(orientation="left")})
        self._plot_widget.setObjectName("histogram_pyqtgraph_plot")
        self._plot_widget.setBackground(HISTOGRAM_PLOT_BACKGROUND_COLOR)
        self._plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._plot_item = self._plot_widget.getPlotItem()
        self._plot_item.setMenuEnabled(False)
        self._plot_item.hideButtons()
        self._plot_item.showGrid(x=True, y=True, alpha=0.18)
        self._plot_item.setLabel("bottom", "Intensity", color=HISTOGRAM_AXIS_TEXT_COLOR)
        self._plot_item.setLabel("left", "Count", color=HISTOGRAM_AXIS_TEXT_COLOR)
        for axis_name in ("bottom", "left"):
            axis = self._plot_item.getAxis(axis_name)
            if axis_name == "left":
                axis.enableAutoSIPrefix(False)
            axis.setPen(pg.mkPen(HISTOGRAM_AXIS_GRID_COLOR, width=1))
            axis.setTextPen(pg.mkPen(HISTOGRAM_AXIS_TEXT_COLOR, width=1))

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plot_widget, 0, 0)

        self._bar_item: pg.BarGraphItem | None = None
        self._last_result: HistogramResult | None = None

    def set_histogram(self, result: HistogramResult) -> None:
        """Render histogram bars from a calculated histogram result."""
        counts = np.asarray(result.counts, dtype=float)
        bin_edges = np.asarray(result.bin_edges, dtype=float)
        if counts.ndim != 1 or bin_edges.ndim != 1 or len(bin_edges) != len(counts) + 1:
            self.clear_histogram()
            return

        self._last_result = result
        self._clear_bar_item()
        self.set_log_y(result.settings.log_y)
        self._plot_item.setLabel(
            "left",
            "Density" if result.settings.density else "Count",
            color=HISTOGRAM_AXIS_TEXT_COLOR,
        )

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges)
        brush = pg.mkBrush(_qcolor(HISTOGRAM_BAR_FILL_COLOR, HISTOGRAM_BAR_FILL_ALPHA))
        pen = None if HISTOGRAM_BAR_EDGE_COLOR is None else pg.mkPen(HISTOGRAM_BAR_EDGE_COLOR, width=1)
        self._bar_item = _create_bar_item(
            centers=centers,
            widths=widths,
            counts=counts,
            log_y=result.settings.log_y,
            brush=brush,
            pen=pen,
        )
        self._plot_item.addItem(self._bar_item)
        self._fit_histogram_view(bin_edges, counts, log_y=result.settings.log_y)

    def clear_histogram(self) -> None:
        """Clear plotted histogram data; card state text lives in the status card."""
        self._last_result = None
        self._clear_bar_item()
        self.set_log_y(False)
        self._plot_item.setLabel("left", "Count", color=HISTOGRAM_AXIS_TEXT_COLOR)

    def set_log_y(self, enabled: bool) -> None:
        if enabled:
            # Avoid pyqtgraph exponentiating a stale linear y-range while switching the axis to log mode.
            self._plot_item.setYRange(0.0, 1.0, padding=0.0)
        self._plot_item.setLogMode(x=False, y=enabled)

    def set_contrast_limits(self, limits: tuple[float, float] | None) -> None:
        """Reserve a stable API hook for the contrast-sync slice."""
        _ = limits

    def set_percentile_markers(self, markers: Mapping[float, float]) -> None:
        """Reserve a stable API hook for the percentile guide-line slice."""
        _ = markers

    def _clear_bar_item(self) -> None:
        if self._bar_item is None:
            return

        self._plot_item.removeItem(self._bar_item)
        self._bar_item = None

    def _fit_histogram_view(self, bin_edges: np.ndarray, counts: np.ndarray, *, log_y: bool) -> None:
        if len(bin_edges) >= 2 and np.all(np.isfinite(bin_edges[[0, -1]])):
            self._plot_item.setXRange(float(bin_edges[0]), float(bin_edges[-1]), padding=0.02)

        finite_counts = counts[np.isfinite(counts)]
        if finite_counts.size == 0:
            self._plot_item.setYRange(0.0, 1.0, padding=0.0)
            return

        if log_y:
            positive_counts = _positive_values(finite_counts)
            if positive_counts.size == 0:
                self._plot_item.setYRange(0.0, 1.0, padding=0.0)
                return

            low, high = _log_y_range(positive_counts)
            self._plot_item.setYRange(low, high, padding=0.0)
            return

        maximum = float(np.max(finite_counts))
        self._plot_item.setYRange(0.0, maximum * 1.05 if maximum > 0 else 1.0, padding=0.0)


def _qcolor(color: str, alpha: int | None = None) -> QColor:
    qcolor = QColor(color)
    if alpha is not None:
        qcolor.setAlpha(alpha)
    return qcolor


def _create_bar_item(
    *,
    centers: np.ndarray,
    widths: np.ndarray,
    counts: np.ndarray,
    log_y: bool,
    brush,
    pen,
) -> pg.BarGraphItem:
    if not log_y:
        return pg.BarGraphItem(x=centers, height=counts, width=widths, brush=brush, pen=pen)

    positive_mask = np.isfinite(counts) & (counts > 0)
    positive_counts = counts[positive_mask]
    if positive_counts.size == 0:
        return pg.BarGraphItem(x=[], y0=[], y1=[], width=[], brush=brush, pen=pen)

    low, _high = _log_y_range(positive_counts)
    return pg.BarGraphItem(
        x=centers[positive_mask],
        y0=np.full(positive_counts.shape, low),
        y1=np.log10(positive_counts),
        width=widths[positive_mask],
        brush=brush,
        pen=pen,
    )


def _positive_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0)]


def _log_y_range(values: np.ndarray) -> tuple[float, float]:
    positive_values = _positive_values(values)
    if positive_values.size == 0:
        return 0.0, 1.0

    low = float(np.floor(np.log10(np.min(positive_values))))
    high = float(np.ceil(np.log10(np.max(positive_values))))
    if low == high:
        return low - _LOG_Y_SINGLE_DECADE_PADDING, high + _LOG_Y_SINGLE_DECADE_PADDING
    return low, high


def _format_scientific_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""

    mantissa, exponent = f"{value:.{_SCIENTIFIC_TICK_MAX_DECIMALS}e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def _format_log_tick(value: float, scale: float) -> str:
    actual_value = np.power(10.0, value) * scale
    if not np.isfinite(actual_value):
        return ""
    if _should_use_scientific_tick(actual_value):
        return _format_scientific_tick(actual_value)
    return f"{actual_value:g}"


def _should_use_scientific_tick(value: float) -> bool:
    absolute_value = abs(value)
    return absolute_value != 0 and (
        absolute_value >= _SCIENTIFIC_TICK_HIGH_ABS or absolute_value <= _SCIENTIFIC_TICK_LOW_ABS
    )
