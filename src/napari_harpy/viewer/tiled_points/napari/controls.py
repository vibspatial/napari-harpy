"""Qt controls for the logical tiled-points layer."""

from __future__ import annotations

from html import escape
from typing import cast

from napari._qt.layer_controls.qt_layer_controls_base import LayerFormLayout, QtLayerControls
from napari.utils.events import Event
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDoubleSpinBox, QLabel, QSizePolicy

from napari_harpy.viewer.tiled_points.contracts import TiledPointsLayerStatus
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel


class QtTiledPointsLayerControls(QtLayerControls):
    """Expose point style and read-only cache display status."""

    layer: TiledPointsLayerModel

    def __init__(self, layer: TiledPointsLayerModel) -> None:
        super().__init__(layer)
        form = cast(LayerFormLayout, self.layout())

        self.point_diameter_spin_box = QDoubleSpinBox(self)
        self.point_diameter_spin_box.setObjectName("tiledPointsDiameterSpinBox")
        self.point_diameter_spin_box.setDecimals(1)
        self.point_diameter_spin_box.setRange(0.1, 100.0)
        self.point_diameter_spin_box.setSingleStep(0.5)
        self.point_diameter_spin_box.setSuffix(" px")
        self.point_diameter_spin_box.setValue(layer.point_diameter)
        self.point_diameter_spin_box.valueChanged.connect(self._on_point_diameter_widget_changed)
        layer.events.point_diameter.connect(self._on_point_diameter_changed)
        form.addRow("Point diameter", self.point_diameter_spin_box)

        self.level_label = QLabel(self)
        self.level_label.setObjectName("tiledPointsLevelLabel")
        form.addRow("LOD", self.level_label)
        self.rendered_label = QLabel(self)
        self.rendered_label.setObjectName("tiledPointsRenderedLabel")
        form.addRow("Rendered", self.rendered_label)
        self.status_label = QLabel(self)
        self.status_label.setObjectName("tiledPointsStatusLabel")
        form.addRow("Status", self.status_label)
        self.sampling_label = QLabel(self)
        self.sampling_label.setObjectName("tiledPointsSamplingLabel")
        form.addRow("Sampling", self.sampling_label)

        for label in (self.status_label, self.sampling_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            # Diagnostics must fit the dock, not widen it and shrink the canvas.
            # Ignore horizontal hints even for unbroken paths/identifiers, while
            # preserving the height-for-width policy enabled by word wrapping.
            policy = label.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            label.setSizePolicy(policy)
            label.setMinimumWidth(0)

        layer.events.display_status.connect(self._on_display_status_changed)
        self._apply_display_status(layer.display_status)

    def _on_point_diameter_widget_changed(self, value: float) -> None:
        self.layer.point_diameter = value

    def _on_point_diameter_changed(self, event: Event) -> None:
        previous = self.point_diameter_spin_box.blockSignals(True)
        try:
            self.point_diameter_spin_box.setValue(float(event.value))
        finally:
            self.point_diameter_spin_box.blockSignals(previous)

    def _on_display_status_changed(self, event: Event) -> None:
        self._apply_display_status(event.value)

    def _apply_display_status(self, status: TiledPointsLayerStatus) -> None:
        self.level_label.setText(status.level_label)
        self.rendered_label.setText(f"{status.rendered_point_count:,} points / {status.rendered_tile_count:,} tiles")
        self.status_label.setText(status.message)
        self.status_label.setToolTip(f"<qt>{escape(status.message)}</qt>")
        if status.omitted_value_ids:
            omitted = ", ".join(str(value_id) for value_id in status.omitted_value_ids)
            sampling = f"Sampled; omitted value IDs: {omitted}"
        elif status.sampled:
            sampling = "Sampled"
        else:
            sampling = "No sampled omission"
        self.sampling_label.setText(sampling)
        self.sampling_label.setToolTip(f"<qt>{escape(sampling)}</qt>")
