from __future__ import annotations

from collections.abc import Mapping
from html import escape

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QGridLayout, QSizePolicy, QToolTip, QWidget

from napari_harpy.core.histogram import HistogramResult
from napari_harpy.widgets.histogram.styles import (
    HISTOGRAM_AXIS_GRID_COLOR,
    HISTOGRAM_AXIS_TEXT_COLOR,
    HISTOGRAM_BAR_EDGE_COLOR,
    HISTOGRAM_BAR_FILL_ALPHA,
    HISTOGRAM_BAR_FILL_COLOR,
    HISTOGRAM_CONTRAST_LINE_COLOR,
    HISTOGRAM_CONTRAST_REGION_ALPHA,
    HISTOGRAM_PERCENTILE_LINE_COLOR,
    HISTOGRAM_PLOT_BACKGROUND_COLOR,
)

_PLOT_MIN_HEIGHT = 200
_PLOT_MAX_HEIGHT = 240
_SCIENTIFIC_TICK_HIGH_ABS = 10_000
_SCIENTIFIC_TICK_LOW_ABS = 0.001
_SCIENTIFIC_TICK_MAX_DECIMALS = 2
_LOG_Y_SINGLE_DECADE_PADDING = 0.5
# Log labels are intentionally range-dependent: narrow views label 1/2/5 per decade,
# while wider views fall back to sparse decade labels to avoid crowded y-axis text.
_LOG_LABEL_MANTISSAS = (1, 2, 5)
_LOG_MINOR_MANTISSAS = tuple(range(1, 10))
_LOG_MAX_DECADE_LABELS = 6
# Match napari's float32 contrast safety margin: keep at least 256 distinguishable intensity steps.
_CONTRAST_MINIMUM_SHADES = 256
_CONTRAST_LINE_WIDTH = 2
_CONTRAST_HOVER_LINE_WIDTH = 10
_PERCENTILE_LINE_WIDTH = 1
_PERCENTILE_HOVER_LINE_WIDTH = 4


class _ScientificYAxisItem(pg.AxisItem):
    """Axis item that keeps large histogram counts compact."""

    def logTickValues(
        self,
        minVal: float,
        maxVal: float,
        size: float,
        stdTicks: list[tuple[float, list[float]]],
    ) -> list[tuple[float | None, list[float]]]:
        label_ticks = _log_label_ticks(minVal, maxVal)
        if not label_ticks and stdTicks:
            label_ticks = [float(value) for value in stdTicks[0][1]]
        minor_ticks = _log_minor_ticks(minVal, maxVal, labeled_ticks=label_ticks)
        return [(None, label_ticks), (None, minor_ticks)]

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


class _HoverablePercentileLine(pg.InfiniteLine):
    """Non-movable percentile line that still gives hover feedback."""

    def __init__(self, *args: object, tooltip_text: str, **kwargs: object) -> None:
        kwargs.pop("movable", None)
        self._tooltip_text = tooltip_text
        super().__init__(*args, movable=False, **kwargs)
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, ev) -> None:
        # InfiniteLine only applies hoverPen for movable lines by default. Percentile
        # guides are read-only, so we handle hover explicitly without accepting drags.
        if ev.isExit():
            self.setMouseHover(False)
            QToolTip.hideText()
            return

        self.setMouseHover(True)
        # Qt/the platform owns native tooltip lifetime, so do not pass a custom
        # duration here; it is not reliable for pyqtgraph QGraphicsItem hover.
        _apply_histogram_tooltip_palette()
        QToolTip.showText(ev.screenPos().toQPoint(), _format_histogram_tooltip(self._tooltip_text))


class _HistogramPlotWidget(QWidget):
    """Small pyqtgraph wrapper for card-local histogram rendering."""

    # Contrast-limit drag flow:
    # 1. _ensure_contrast_region creates the pyqtgraph LinearRegionItem and connects
    #    LinearRegionItem.sigRegionChanged to _on_contrast_region_changed.
    # 2. _on_contrast_region_changed reads the current low/high x-axis positions via
    #    LinearRegionItem.getRegion(), clamps collapsed handles apart, then emits this signal.
    # 3. HistogramWidget connects this signal with the owning card id and updates
    #    the matching napari Image layer.contrast_limits.
    contrast_limits_dragged = Signal(float, float)

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
                axis.setStyle(maxTextLevel=0)
            axis.setPen(pg.mkPen(HISTOGRAM_AXIS_GRID_COLOR, width=1))
            axis.setTextPen(pg.mkPen(HISTOGRAM_AXIS_TEXT_COLOR, width=1))

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._plot_widget, 0, 0)

        self._bar_item: pg.BarGraphItem | None = None
        self._contrast_region: pg.LinearRegionItem | None = None
        self._percentile_marker_lines: list[pg.InfiniteLine] = []
        self._updating_contrast_region = False
        self._last_result: HistogramResult | None = None
        self._last_contrast_limits: tuple[float, float] | None = None

    def set_histogram(self, result: HistogramResult) -> None:
        """Render histogram bars from a calculated histogram result."""
        counts = np.asarray(result.counts, dtype=float)
        bin_edges = np.asarray(result.bin_edges, dtype=float)
        if counts.ndim != 1 or bin_edges.ndim != 1 or len(bin_edges) != len(counts) + 1:
            self.clear_histogram()
            return

        self._last_result = result
        self._clear_bar_item()
        self._clear_percentile_markers()
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
        self.set_percentile_markers(result.percentile_values)

    def is_showing_result(self, result: HistogramResult) -> bool:
        """Return whether this exact controller result is already rendered."""
        return self._last_result is result

    def clear_histogram(self) -> None:
        """Clear plotted histogram data; card state text lives in the status card."""
        self._last_result = None
        self._last_contrast_limits = None
        self._clear_bar_item()
        self._clear_percentile_markers()
        self.set_contrast_limits(None)
        self.set_log_y(False)
        self._plot_item.setLabel("left", "Count", color=HISTOGRAM_AXIS_TEXT_COLOR)

    def reset_view(self, result: HistogramResult) -> None:
        """Reset pan/zoom to the fitted view for a calculated histogram.

        In the normal widget flow, `result` is the controller's cached result
        and matches the histogram already rendered as `_last_result`.
        """
        counts = np.asarray(result.counts, dtype=float)
        bin_edges = np.asarray(result.bin_edges, dtype=float)
        if counts.ndim != 1 or bin_edges.ndim != 1 or len(bin_edges) != len(counts) + 1:
            return

        self._fit_histogram_view(bin_edges, counts, log_y=result.settings.log_y)

    def set_log_y(self, enabled: bool) -> None:
        if enabled:
            # Avoid pyqtgraph exponentiating a stale linear y-range while switching the axis to log mode.
            self._plot_item.setYRange(0.0, 1.0, padding=0.0)
        self._plot_item.setLogMode(x=False, y=enabled)

    def set_contrast_limits(self, limits: tuple[float, float] | None) -> None:
        """Set or clear the draggable contrast-limit region."""
        normalized_limits = _ordered_finite_contrast_limits(limits)
        if normalized_limits is None:
            if limits is None:
                self._last_contrast_limits = None
                self._clear_contrast_region()
            return

        normalized_limits = self._with_minimum_contrast_width(normalized_limits)
        region = self._ensure_contrast_region(normalized_limits)
        self._updating_contrast_region = True
        try:
            region.setRegion(normalized_limits)
        finally:
            self._updating_contrast_region = False
        self._last_contrast_limits = normalized_limits

    def set_percentile_markers(self, markers: Mapping[float, float]) -> None:
        """Draw non-interactive percentile guide lines for in-range values."""
        self._clear_percentile_markers()
        x_range = _histogram_x_range(self._last_result)
        if x_range is None:
            return

        x_min, x_max = x_range
        pen = pg.mkPen(
            HISTOGRAM_PERCENTILE_LINE_COLOR,
            width=_PERCENTILE_LINE_WIDTH,
            style=Qt.PenStyle.DashLine,
        )
        hover_pen = pg.mkPen(
            HISTOGRAM_PERCENTILE_LINE_COLOR,
            width=_PERCENTILE_HOVER_LINE_WIDTH,
            style=Qt.PenStyle.DashLine,
        )
        visible_markers: list[tuple[float, float]] = []
        for percentile, value in sorted(markers.items()):
            percentile_value = float(value)
            if not np.isfinite(percentile_value) or percentile_value < x_min or percentile_value > x_max:
                continue
            visible_markers.append((float(percentile), percentile_value))

        for percentile, percentile_value in visible_markers:
            line = _HoverablePercentileLine(
                pos=percentile_value,
                angle=90,
                pen=pen,
                hoverPen=hover_pen,
                tooltip_text=_format_percentile_tooltip(percentile, percentile_value),
                movable=False,
            )
            line.setZValue(8)
            self._plot_item.addItem(line)
            self._percentile_marker_lines.append(line)

    def _ensure_contrast_region(self, limits: tuple[float, float]) -> pg.LinearRegionItem:
        if self._contrast_region is not None:
            return self._contrast_region

        pen = pg.mkPen(HISTOGRAM_CONTRAST_LINE_COLOR, width=_CONTRAST_LINE_WIDTH)
        hover_pen = pg.mkPen(HISTOGRAM_CONTRAST_LINE_COLOR, width=_CONTRAST_HOVER_LINE_WIDTH)
        brush = pg.mkBrush(_qcolor(HISTOGRAM_CONTRAST_LINE_COLOR, HISTOGRAM_CONTRAST_REGION_ALPHA))
        self._contrast_region = pg.LinearRegionItem(
            values=limits,
            orientation="vertical",
            brush=brush,
            pen=pen,
            hoverPen=hover_pen,
            movable=False,
            swapMode="block",
        )
        for line in self._contrast_region.lines:
            # Keep the filled region body passive so drags between the lines pan
            # the plot, while the two InfiniteLine handles remain draggable.
            line.setMovable(True)
            line.setCursor(Qt.CursorShape.SizeHorCursor)
        self._contrast_region.setZValue(10)
        self._contrast_region.sigRegionChanged.connect(self._on_contrast_region_changed)
        self._plot_item.addItem(self._contrast_region)
        return self._contrast_region

    def _clear_contrast_region(self) -> None:
        if self._contrast_region is None:
            return

        try:
            self._contrast_region.sigRegionChanged.disconnect(self._on_contrast_region_changed)
        except (TypeError, RuntimeError):
            pass
        self._plot_item.removeItem(self._contrast_region)
        self._contrast_region = None

    def _on_contrast_region_changed(self, *_args: object) -> None:
        if self._updating_contrast_region or self._contrast_region is None:
            return

        raw_limits = _ordered_finite_contrast_limits(tuple(float(value) for value in self._contrast_region.getRegion()))
        if raw_limits is None:
            return

        limits = self._with_minimum_contrast_width(raw_limits)
        if limits != raw_limits:
            self._updating_contrast_region = True
            try:
                self._contrast_region.setRegion(limits)
            finally:
                self._updating_contrast_region = False

        self._last_contrast_limits = limits
        self.contrast_limits_dragged.emit(*limits)

    def _with_minimum_contrast_width(self, limits: tuple[float, float]) -> tuple[float, float]:
        low, high = limits
        min_width = self._minimum_contrast_width(limits)
        if high - low >= min_width:
            return limits

        previous_limits = self._last_contrast_limits
        if previous_limits is not None:
            low_delta = abs(low - previous_limits[0])
            high_delta = abs(high - previous_limits[1])
            if low_delta > high_delta:
                return high - min_width, high
            return low, low + min_width

        center = (low + high) / 2
        half_width = min_width / 2
        return center - half_width, center + half_width

    def _minimum_contrast_width(self, limits: tuple[float, float]) -> float:
        display_step = _display_step_for_span(_histogram_x_span(self._last_result) or _fallback_span(limits))
        precision_width = _float32_precision_width(limits)
        return max(display_step, precision_width)

    def _clear_bar_item(self) -> None:
        if self._bar_item is None:
            return

        self._plot_item.removeItem(self._bar_item)
        self._bar_item = None

    def _clear_percentile_markers(self) -> None:
        for line in self._percentile_marker_lines:
            self._plot_item.removeItem(line)
        self._percentile_marker_lines.clear()

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


def _ordered_finite_contrast_limits(limits: tuple[float, float] | None) -> tuple[float, float] | None:
    if limits is None:
        return None

    low, high = (float(value) for value in limits)
    if not np.isfinite(low) or not np.isfinite(high):
        return None

    return tuple(sorted((low, high)))


def _histogram_x_span(result: HistogramResult | None) -> float | None:
    x_range = _histogram_x_range(result)
    if x_range is None:
        return None

    span = abs(x_range[1] - x_range[0])
    return span if span > 0 else None


def _histogram_x_range(result: HistogramResult | None) -> tuple[float, float] | None:
    if result is None:
        return None

    bin_edges = np.asarray(result.bin_edges, dtype=float)
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        return None

    first, last = float(bin_edges[0]), float(bin_edges[-1])
    if not np.isfinite(first) or not np.isfinite(last):
        return None

    return tuple(sorted((first, last)))


def _fallback_span(limits: tuple[float, float]) -> float:
    low, high = limits
    return max(abs(high - low), abs(low), abs(high), 1.0)


def _display_step_for_span(span: float) -> float:
    if not np.isfinite(span) or span <= 0:
        return 0.001

    decimals = min(64, max(int(3 - np.log10(span)), 0))
    return float(10**-decimals)


def _float32_precision_width(limits: tuple[float, float]) -> float:
    low, high = limits
    # float32 spacing grows with value magnitude, so high-intensity ranges need
    # a wider minimum span than ranges near zero.
    low_spacing = abs(float(np.spacing(np.float32(low))))
    high_spacing = abs(float(np.spacing(np.float32(high))))
    return max(low_spacing, high_spacing) * _CONTRAST_MINIMUM_SHADES


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


def _log_label_ticks(min_log: float, max_log: float) -> list[float]:
    min_log, max_log = sorted((min_log, max_log))
    span = max_log - min_log
    if span <= 2.5:
        return _log_ticks_for_mantissas(min_log, max_log, _LOG_LABEL_MANTISSAS)

    exponent_step = max(1, int(np.ceil(span / _LOG_MAX_DECADE_LABELS)))
    first_exponent = int(np.ceil(min_log / exponent_step) * exponent_step)
    last_exponent = int(np.floor(max_log))
    return [float(exponent) for exponent in range(first_exponent, last_exponent + 1, exponent_step)]


def _log_minor_ticks(min_log: float, max_log: float, *, labeled_ticks: list[float]) -> list[float]:
    if max_log - min_log > 3:
        return []

    ticks = _log_ticks_for_mantissas(min_log, max_log, _LOG_MINOR_MANTISSAS)
    return [tick for tick in ticks if not any(np.isclose(tick, labeled_tick) for labeled_tick in labeled_ticks)]


def _log_ticks_for_mantissas(min_log: float, max_log: float, mantissas: tuple[int, ...]) -> list[float]:
    first_exponent = int(np.floor(min_log))
    last_exponent = int(np.ceil(max_log))
    ticks: list[float] = []
    for exponent in range(first_exponent, last_exponent + 1):
        for mantissa in mantissas:
            tick = float(exponent + np.log10(mantissa))
            if min_log <= tick <= max_log:
                ticks.append(tick)
    return ticks


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


def _format_percentile_tooltip(percentile: float, value: float) -> str:
    return f"p{percentile:g} = {_format_compact_number(value)}"


def _apply_histogram_tooltip_palette() -> None:
    palette = QToolTip.palette()
    background = QColor(HISTOGRAM_PLOT_BACKGROUND_COLOR)
    text = QColor(HISTOGRAM_AXIS_TEXT_COLOR)
    palette.setColor(QPalette.ColorRole.ToolTipBase, background)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Window, background)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    QToolTip.setPalette(palette)


def _format_histogram_tooltip(text: str) -> str:
    escaped_text = escape(text)
    return (
        "<qt>"
        f"<div style='"
        f"color: {HISTOGRAM_AXIS_TEXT_COLOR}; "
        f"background-color: {HISTOGRAM_PLOT_BACKGROUND_COLOR}; "
        f"border: 1px solid {HISTOGRAM_PERCENTILE_LINE_COLOR}; "
        "padding: 4px 7px; "
        "white-space: nowrap;'>"
        f"{escaped_text}"
        "</div>"
        "</qt>"
    )


def _format_compact_number(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def _should_use_scientific_tick(value: float) -> bool:
    absolute_value = abs(value)
    return absolute_value != 0 and (
        absolute_value >= _SCIENTIFIC_TICK_HIGH_ABS or absolute_value <= _SCIENTIFIC_TICK_LOW_ABS
    )
