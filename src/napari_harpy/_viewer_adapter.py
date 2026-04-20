from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from napari.layers import Image, Labels, Layer

if TYPE_CHECKING:
    from spatialdata import SpatialData


ImageDisplayMode = Literal["stack", "overlay"]


@dataclass(frozen=True)
class LayerBinding:
    """Runtime binding between a napari layer and a SpatialData element."""

    layer: Layer
    element_name: str
    element_type: str
    coordinate_system: str | None
    sdata_id: int | None = None
    image_display_mode: ImageDisplayMode | None = None
    channel_index: int | None = None
    channel_name: str | None = None


class LayerBindingRegistry:
    """In-memory mapping from napari layers to Harpy SpatialData element identity.

    The registry is Harpy's source of truth for answering questions such as:

    - which napari layer represents a given labels element
    - which napari layers belong to a given image element
    - whether an image is shown in stack mode or overlay mode
    - which overlay layer corresponds to which image channel

    This keeps Harpy's layer lookup logic in one central place instead of
    relying on napari layer metadata as the primary contract.
    """

    def __init__(self) -> None:
        self._bindings: dict[int, LayerBinding] = {}

    def register_layer(
        self,
        layer: Layer,
        *,
        element_name: str,
        element_type: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
    ) -> LayerBinding:
        """Register a layer binding and attach lightweight metadata."""
        binding = LayerBinding(
            layer=layer,
            element_name=element_name,
            element_type=element_type,
            coordinate_system=coordinate_system,
            sdata_id=None if sdata is None else id(sdata),
            image_display_mode=image_display_mode,
            channel_index=channel_index,
            channel_name=channel_name,
        )
        self._bindings[id(layer)] = binding
        _apply_minimal_layer_metadata(layer, binding)
        return binding

    def unregister_layer(self, layer: Layer) -> LayerBinding | None:
        """Remove a layer binding if present."""
        return self._bindings.pop(id(layer), None)

    def get_binding(self, layer: Layer) -> LayerBinding | None:
        """Return the binding for a specific layer."""
        return self._bindings.get(id(layer))

    def iter_bindings(self) -> tuple[LayerBinding, ...]:
        """Return all registered bindings in insertion order."""
        return tuple(self._bindings.values())

    def find_bindings(
        self,
        *,
        sdata: SpatialData | None = None,
        element_name: str | None = None,
        element_type: str | None = None,
        coordinate_system: str | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
    ) -> list[LayerBinding]:
        """Find bindings matching the provided filters."""
        sdata_id = None if sdata is None else id(sdata)
        matches: list[LayerBinding] = []
        for binding in self._bindings.values():
            if sdata_id is not None and binding.sdata_id != sdata_id:
                continue
            if element_name is not None and binding.element_name != element_name:
                continue
            if element_type is not None and binding.element_type != element_type:
                continue
            if coordinate_system is not None and binding.coordinate_system != coordinate_system:
                continue
            if image_display_mode is not None and binding.image_display_mode != image_display_mode:
                continue
            if channel_index is not None and binding.channel_index != channel_index:
                continue
            if channel_name is not None and binding.channel_name != channel_name:
                continue
            matches.append(binding)
        return matches


class ViewerAdapter:
    """Harpy-owned service for viewer-facing layer lookup and activation.

    The adapter wraps a napari viewer and provides the Harpy-level operations
    that care about what loaded layers *mean* in terms of `SpatialData`
    elements.

    It is responsible for tasks such as:

    - registering and unregistering Harpy-managed layer bindings
    - resolving which loaded napari layer corresponds to a labels element
    - resolving which loaded napari layers correspond to an image element
    - activating a requested layer in the viewer

    The adapter does not use napari layer metadata as the primary contract.
    Instead, it relies on `LayerBindingRegistry` as the authoritative in-memory
    mapping from napari layers to `SpatialData` element identity.
    """

    def __init__(self, viewer: Any | None = None, layer_bindings: LayerBindingRegistry | None = None) -> None:
        self._viewer = viewer
        self._layer_bindings = layer_bindings or LayerBindingRegistry()

    @property
    def layer_bindings(self) -> LayerBindingRegistry:
        """Return the shared layer-binding registry."""
        return self._layer_bindings

    def register_layer(
        self,
        layer: Layer,
        *,
        element_name: str,
        element_type: str,
        coordinate_system: str | None = None,
        sdata: SpatialData | None = None,
        image_display_mode: ImageDisplayMode | None = None,
        channel_index: int | None = None,
        channel_name: str | None = None,
    ) -> LayerBinding:
        """Register a layer in the shared binding registry."""
        return self._layer_bindings.register_layer(
            layer,
            element_name=element_name,
            element_type=element_type,
            coordinate_system=coordinate_system,
            sdata=sdata,
            image_display_mode=image_display_mode,
            channel_index=channel_index,
            channel_name=channel_name,
        )

    def unregister_layer(self, layer: Layer) -> LayerBinding | None:
        """Remove a layer from the shared binding registry."""
        return self._layer_bindings.unregister_layer(layer)

    def activate_layer(self, layer: Layer | None) -> bool:
        """Make a layer active in the viewer when supported."""
        if layer is None:
            return False

        layers = getattr(self._viewer, "layers", None)
        selection = getattr(layers, "selection", None)
        if selection is None:
            return False

        select_only = getattr(selection, "select_only", None)
        if callable(select_only):
            select_only(layer)
            return True

        if hasattr(selection, "active"):
            selection.active = layer
            return True

        return False

    def get_loaded_labels_layer(self, sdata: SpatialData, label_name: str) -> Labels | None:
        """Return the loaded labels layer for a SpatialData labels element."""
        for layer in self._iter_candidate_layers():
            if not _is_pickable_labels_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_binding(binding, sdata=sdata, element_name=label_name, element_type="labels"):
                return layer

        return None

    def get_loaded_image_layers(self, sdata: SpatialData, image_name: str) -> list[Image]:
        """Return loaded image layers for a SpatialData image element."""
        matches: list[Image] = []
        for layer in self._iter_candidate_layers():
            if not _is_spatialdata_image_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_binding(binding, sdata=sdata, element_name=image_name, element_type="image"):
                matches.append(layer)

        return matches

    def _iter_candidate_layers(self) -> Iterable[Layer]:
        layers = getattr(self._viewer, "layers", None)
        if layers is not None:
            return tuple(layers)
        return tuple(binding.layer for binding in self._layer_bindings.iter_bindings())


def _apply_minimal_layer_metadata(layer: Layer, binding: LayerBinding) -> None:
    """Mirror a small subset of binding info onto the layer for debugging.

    Harpy treats ``LayerBindingRegistry`` as the authoritative layer-binding
    contract. These metadata fields are only a lightweight convenience so a
    napari layer remains somewhat self-describing during debugging or manual
    inspection.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        layer.metadata = metadata

    metadata["element_name"] = binding.element_name
    metadata["element_type"] = binding.element_type
    metadata["coordinate_system"] = binding.coordinate_system
    _set_optional_metadata_value(metadata, "image_display_mode", binding.image_display_mode)
    _set_optional_metadata_value(metadata, "channel_index", binding.channel_index)
    _set_optional_metadata_value(metadata, "channel_name", binding.channel_name)


def _set_optional_metadata_value(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        metadata.pop(key, None)
        return
    metadata[key] = value


def _matches_binding(
    binding: LayerBinding | None,
    *,
    sdata: SpatialData,
    element_name: str,
    element_type: str,
) -> bool:
    if binding is None:
        return False
    return (
        binding.sdata_id == id(sdata) and binding.element_name == element_name and binding.element_type == element_type
    )


def _is_pickable_labels_layer(layer: Layer) -> bool:
    events = getattr(layer, "events", None)
    return isinstance(layer, Labels) and getattr(events, "selected_label", None) is not None


def _is_spatialdata_image_layer(layer: Layer) -> bool:
    # napari `Labels` layers are scalar-field siblings of `Image`, not subclasses.
    return isinstance(layer, Image)
