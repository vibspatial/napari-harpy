from __future__ import annotations

import math
from dataclasses import dataclass

from qtpy.QtCore import QSignalBlocker, QSize, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QWidget,
)

from napari_harpy.viewer.adapter import ImageLayerBinding
from napari_harpy.viewer.image_styling import DEFAULT_OVERLAY_COLORS
from napari_harpy.widgets.overlay_color_button import OverlayColorButton
from napari_harpy.widgets.shared_styles import (
    WIDGET_BORDER_COLOR,
    WIDGET_PANEL_MUTED_COLOR,
    WIDGET_TEXT_COLOR,
    _ElidedLabel,
    create_visibility_eye_icon,
    format_tooltip,
)

_CHANNEL_VISIBILITY_BUTTON_STYLESHEET = (
    "QToolButton {"
    "background: transparent; "
    "border: 1px solid transparent; "
    "border-radius: 5px; "
    "padding: 2px;}"
    f"QToolButton:hover {{ background-color: {WIDGET_PANEL_MUTED_COLOR}; "
    f"border-color: {WIDGET_BORDER_COLOR}; }}"
    f"QToolButton:focus {{ border-color: {WIDGET_TEXT_COLOR}; }}"
)


@dataclass(frozen=True)
class _OverlayColormapPresentation:
    name: str
    colors: tuple[str, ...]
    solid_color: str | None


class _OverlayChannelRow(QWidget):
    """Present one loaded channel and bridge intent with live napari state.

    A row keeps its construction-time binding for its entire lifetime. Its owner
    must dispose this row and construct another one when the binding changes.

    The row never mutates its bound napari layer. User eye, color, and removal
    actions emit channel-local requests for the owning widget to validate.

    In the opposite direction, this row listens directly to
    ``layer.events.visible`` and ``layer.events.colormap``. Those callbacks read
    the accepted napari property and render the corresponding control. Napari is
    therefore authoritative; presentation updates use ``QSignalBlocker`` where
    needed so reflecting napari state cannot emit another user request.
    """

    remove_requested = Signal(int)
    visibility_change_requested = Signal(int, bool)
    color_change_requested = Signal(int, str)

    def __init__(
        self,
        binding: ImageLayerBinding,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.binding = binding
        self._disposed = False
        channel_index = _binding_channel_index(binding)
        channel_name = _binding_channel_name(binding)
        self.setObjectName("overlay_channel_row")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.visibility_button = QToolButton(self)
        self.visibility_button.setObjectName("overlay_channel_visibility_button")
        self.visibility_button.setCheckable(True)
        self.visibility_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.visibility_button.setFixedSize(28, 22)
        self.visibility_button.setIconSize(QSize(16, 16))
        self.visibility_button.setStyleSheet(_CHANNEL_VISIBILITY_BUTTON_STYLESHEET)

        self.channel_label = _ElidedLabel(channel_name, self)
        self.channel_label.setObjectName("overlay_channel_label")

        self.color_button = OverlayColorButton(DEFAULT_OVERLAY_COLORS[0], self)
        self.color_button.setObjectName("overlay_channel_color_button")

        self.remove_button = QPushButton("×")
        self.remove_button.setObjectName("overlay_channel_remove_button")
        self.remove_button.setAccessibleName(f"Remove channel {channel_name} from viewer")
        self.remove_button.setToolTip(format_tooltip(f"Remove channel {channel_name} from viewer"))
        self.remove_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_button.setFixedWidth(28)
        self.remove_button.clicked.connect(
            lambda _checked=False, current_channel_index=channel_index: self.remove_requested.emit(
                current_channel_index
            )
        )

        self.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.color_button.color_selected.connect(self._on_color_selected)

        layout.addWidget(self.visibility_button)
        layout.addWidget(self.channel_label, 1)
        layout.addWidget(self.color_button)
        layout.addWidget(self.remove_button)
        self._connect_presentation_events()
        self.refresh_presentation()

    @property
    def channel_index(self) -> int:
        return _binding_channel_index(self.binding)

    @property
    def channel_name(self) -> str:
        return _binding_channel_name(self.binding)

    def refresh_presentation(self) -> None:
        """Render visibility and colormap from the authoritative live layer."""
        self._apply_visibility(self.binding.layer.visible)
        self._sync_colormap_from_layer()

    def dispose(self) -> None:
        """Idempotently disconnect this row from its napari layer."""
        if self._disposed:
            return
        self._disposed = True
        layer = self.binding.layer
        try:
            layer.events.visible.disconnect(self._on_layer_visible_changed)
        except (TypeError, RuntimeError, ValueError):
            pass
        try:
            layer.events.colormap.disconnect(self._on_layer_colormap_changed)
        except (TypeError, RuntimeError, ValueError):
            pass

    def _connect_presentation_events(self) -> None:
        layer = self.binding.layer
        layer.events.visible.connect(self._on_layer_visible_changed)
        layer.events.colormap.connect(self._on_layer_colormap_changed)

    def _on_layer_visible_changed(self, _event: object) -> None:
        self._apply_visibility(self.binding.layer.visible)

    def _on_layer_colormap_changed(self, _event: object) -> None:
        self._sync_colormap_from_layer()

    def _on_visibility_toggled(self, visible: bool) -> None:
        self.visibility_change_requested.emit(self.channel_index, visible)

    def _on_color_selected(self, color: str) -> None:
        self.color_change_requested.emit(self.channel_index, color)

    def _apply_visibility(self, visible: bool) -> None:
        # Reflect napari state without turning this programmatic eye update into
        # another owner visibility-change request.
        with QSignalBlocker(self.visibility_button):
            self.visibility_button.setChecked(visible)
        self.visibility_button.setIcon(create_visibility_eye_icon(visible=visible))
        action = "Hide" if visible else "Show"
        message = f"{action} channel {self.channel_name}"
        self.visibility_button.setAccessibleName(message)
        self.visibility_button.setToolTip(format_tooltip(message))

    def _sync_colormap_from_layer(self) -> None:
        presentation = _colormap_presentation_from_layer(self.binding.layer)
        if presentation is None:
            return
        if presentation.solid_color is not None:
            self.color_button.set_color(presentation.solid_color)
            return
        self.color_button.set_colormap_preview(
            presentation.name,
            presentation.colors,
        )


def _binding_channel_index(binding: ImageLayerBinding) -> int:
    channel_index = binding.channel_index
    if not isinstance(channel_index, int) or channel_index < 0:
        raise ValueError("Overlay bindings require a non-negative channel index.")
    return channel_index


def _binding_channel_name(binding: ImageLayerBinding) -> str:
    channel_name = binding.channel_name
    if not isinstance(channel_name, str) or not channel_name:
        raise ValueError("Overlay bindings require a non-empty channel name.")
    return channel_name


def _solid_color_from_layer(layer: object) -> str | None:
    presentation = _colormap_presentation_from_layer(layer)
    return presentation.solid_color if presentation is not None else None


def _colormap_presentation_from_layer(
    layer: object,
) -> _OverlayColormapPresentation | None:
    colormap = getattr(layer, "colormap", None)
    name = getattr(colormap, "name", None)
    display_name = name if isinstance(name, str) and name else "Custom colormap"
    colors = _colormap_colors_as_hex(getattr(colormap, "colors", None))
    if len(colors) == 1:
        return _OverlayColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=colors[0],
        )
    if len(colors) == 2 and colors[0] == "#000000":
        return _OverlayColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=colors[1],
        )
    if len(colors) >= 2:
        return _OverlayColormapPresentation(
            name=display_name,
            colors=colors,
            solid_color=None,
        )

    if isinstance(name, str):
        solid_color = _normalized_color_or_none(name)
        if solid_color is not None:
            return _OverlayColormapPresentation(
                name=name,
                colors=(solid_color,),
                solid_color=solid_color,
            )
    if isinstance(colormap, str):
        solid_color = _normalized_color_or_none(colormap)
        if solid_color is not None:
            return _OverlayColormapPresentation(
                name=colormap,
                colors=(solid_color,),
                solid_color=solid_color,
            )
    return None


def _colormap_colors_as_hex(colors: object) -> tuple[str, ...]:
    try:
        rows = tuple(colors)  # type: ignore[arg-type]
    except TypeError:
        return ()

    converted = tuple(color for row in rows if (color := _color_row_to_hex(row)) is not None)
    return converted if len(converted) == len(rows) else ()


def _color_row_to_hex(row: object) -> str | None:
    if isinstance(row, str):
        return _normalized_color_or_none(row)

    try:
        components = tuple(float(component) for component in row)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(components) < 3:
        return None

    rgb = components[:3]
    if any(not math.isfinite(component) or component < 0.0 or component > 1.0 for component in rgb):
        return None
    return "#" + "".join(f"{round(component * 255):02X}" for component in rgb)


def _normalized_color_or_none(color: str) -> str | None:
    qcolor = QColor(color)
    if not qcolor.isValid():
        return None
    return qcolor.name(QColor.NameFormat.HexRgb).upper()
