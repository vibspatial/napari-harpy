from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from napari_harpy._spatialdata import SpatialDataAdapter

if TYPE_CHECKING:
    from spatialdata import SpatialData


class AnnotationController:
    """Manage pick-based object selection for the active segmentation layer."""

    def __init__(
        self,
        spatialdata_adapter: SpatialDataAdapter,
        on_selected_instance_changed: Callable[[int | None], None] | None = None,
    ) -> None:
        self._spatialdata_adapter = spatialdata_adapter
        self._on_selected_instance_changed = on_selected_instance_changed
        self._labels_layer: Any | None = None
        self._selected_instance_id: int | None = None

    @property
    def labels_layer(self) -> Any | None:
        """Return the currently bound labels layer, if any."""
        return self._labels_layer

    @property
    def selected_instance_id(self) -> int | None:
        """Return the currently picked segmentation instance id."""
        return self._selected_instance_id

    def bind(self, sdata: SpatialData | None, label_name: str | None) -> None:
        """Bind the controller to the currently selected SpatialData labels element."""
        next_layer = None
        if sdata is not None and label_name is not None:
            next_layer = self._spatialdata_adapter.get_labels_layer(sdata, label_name)

        if next_layer is self._labels_layer:
            return

        self._disconnect_selected_label_events()
        self._labels_layer = next_layer

        selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
        if selected_label_emitter is not None:
            selected_label_emitter.connect(self._on_layer_selected_label_changed)

        # `Labels.selected_label` defaults to 1 even before the user picks an object,
        # so we only start tracking a concrete instance id after a pick event arrives.
        self._set_selected_instance_id(None)

    def activate_pick_mode(self) -> bool:
        """Put the bound labels layer into napari pick mode."""
        if self._labels_layer is None:
            return False

        if hasattr(self._labels_layer, "mode"):
            self._labels_layer.mode = "pick"

        return True

    def _disconnect_selected_label_events(self) -> None:
        selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
        disconnect = getattr(selected_label_emitter, "disconnect", None)
        if disconnect is None:
            return

        try:
            disconnect(self._on_layer_selected_label_changed)
        except (RuntimeError, TypeError, ValueError):
            return

    def _on_layer_selected_label_changed(self, event: object | None = None) -> None:
        del event
        if self._labels_layer is None:
            self._set_selected_instance_id(None)
            return

        selected_label = getattr(self._labels_layer, "selected_label", 0)
        instance_id = int(selected_label) if int(selected_label) > 0 else None
        self._set_selected_instance_id(instance_id)

    def _set_selected_instance_id(self, instance_id: int | None) -> None:
        if instance_id == self._selected_instance_id:
            return

        self._selected_instance_id = instance_id

        if self._on_selected_instance_changed is not None:
            self._on_selected_instance_changed(instance_id)
