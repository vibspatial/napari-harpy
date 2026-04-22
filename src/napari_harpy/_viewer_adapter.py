from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger
from napari.layers import Image, Labels, Layer
from qtpy.QtCore import QObject, Signal
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

if TYPE_CHECKING:
    from spatialdata import SpatialData

ImageDisplayMode = Literal["stack", "overlay"]
DEFAULT_OVERLAY_COLORS = (
    "#00FFFF",  # cyan
    "#FF00FF",  # magenta
    "#FFA500",  # orange
    "#ADFF2F",  # green-yellow
    "#FF5050",  # light red
    "#7B68EE",  # medium slate blue
    "#FF1493",  # deep pink
)


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


class ViewerAdapter(QObject):
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

    # Emitted when the set/order of live pickable Labels layers changes.
    # Used by consumers that depend on live labels-layer availability,
    # currently ObjectClassificationWidget.
    labels_layers_changed = Signal()
    active_layer_changed = Signal(object)

    def __init__(self, viewer: Any | None = None, layer_bindings: LayerBindingRegistry | None = None) -> None:
        super().__init__()
        self._viewer = viewer
        self._layer_bindings = layer_bindings or LayerBindingRegistry()
        self._connect_layer_events()

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
        binding = self._layer_bindings.register_layer(
            layer,
            element_name=element_name,
            element_type=element_type,
            coordinate_system=coordinate_system,
            sdata=sdata,
            image_display_mode=image_display_mode,
            channel_index=channel_index,
            channel_name=channel_name,
        )
        _apply_legacy_spatialdata_metadata(
            layer,
            sdata=sdata,
            element_name=element_name,
            coordinate_system=coordinate_system,
        )
        return binding

    def unregister_layer(self, layer: Layer) -> LayerBinding | None:
        """Remove a layer from the shared binding registry."""
        return self._layer_bindings.unregister_layer(layer)

    def _connect_layer_events(self) -> None:
        """Keep the registry synchronized with the viewer's layer list when possible."""
        layers = getattr(self._viewer, "layers", None)
        events = getattr(layers, "events", None)
        if events is None:
            return

        for event_name, handler in (
            ("inserted", self._on_viewer_layer_inserted),
            ("removed", self._on_viewer_layer_removed),
            ("reordered", self._on_viewer_layers_reordered),
        ):
            event_emitter = getattr(events, event_name, None)
            connect = getattr(event_emitter, "connect", None)
            if callable(connect):
                connect(handler)

    def _on_viewer_layer_inserted(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        if not isinstance(layer, Layer):
            logger.warning("Ignoring viewer layer insertion event without a napari Layer payload.")
            return
        if _is_pickable_labels_layer(layer):
            self.labels_layers_changed.emit()

    def _on_viewer_layer_removed(self, event: Any) -> None:
        """Unregister Harpy-managed layers when they disappear from the viewer."""
        layer = getattr(event, "value", None)
        if not isinstance(layer, Layer):
            logger.warning("Ignoring viewer layer removal event without a napari Layer payload.")
            return
        had_labels_semantics = _is_pickable_labels_layer(layer)
        removed_binding = self.unregister_layer(layer)
        if removed_binding is None:
            logger.warning(
                "Removed napari layer `%s` had no matching Harpy layer binding.", getattr(layer, "name", layer)
            )
        if had_labels_semantics:
            self.labels_layers_changed.emit()

    def _on_viewer_layers_reordered(self, event: Any) -> None:
        del event
        if any(_is_pickable_labels_layer(layer) for layer in self._iter_candidate_layers()):
            self.labels_layers_changed.emit()

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
            self.active_layer_changed.emit(layer)
            return True

        if hasattr(selection, "active"):
            selection.active = layer
            self.active_layer_changed.emit(layer)
            return True

        return False

    def get_loaded_labels_layer(
        self,
        sdata: SpatialData,
        label_name: str,
        coordinate_system: str | None = None,
    ) -> Labels | None:
        """Return the loaded labels layer for a SpatialData labels element."""
        for layer in self._iter_candidate_layers():
            if not _is_pickable_labels_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if _matches_binding(binding, sdata=sdata, element_name=label_name, element_type="labels"):
                if coordinate_system is not None and binding.coordinate_system != coordinate_system:
                    continue
                return layer

            # TODO: remove the _matches_legacy_spatialdata_layer_metadata
            if coordinate_system is not None and _matches_legacy_spatialdata_layer_metadata(
                layer,
                sdata=sdata,
                element_name=label_name,
                coordinate_system=coordinate_system,
            ):
                return layer

        return None

    def ensure_labels_loaded(self, sdata: SpatialData, label_name: str, coordinate_system: str) -> Labels:
        """Load a labels element into napari if it is not already present."""
        existing_layer = self._get_loaded_labels_layer_for_coordinate_system(sdata, label_name, coordinate_system)
        if existing_layer is not None:
            return existing_layer

        labels = getattr(sdata, "labels", {})
        if label_name not in labels:
            raise ValueError(f"Labels element `{label_name}` is not available in the selected SpatialData object.")

        label_element = labels[label_name]
        available_coordinate_systems = set(get_transformation(label_element, get_all=True).keys())
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available for labels element `{label_name}`."
            )

        labels_data = (
            _flatten_multiscale_element(label_element) if isinstance(label_element, DataTree) else label_element
        )
        layer = Labels(
            labels_data,
            name=label_name,
            affine=_get_affine_transform(label_element, coordinate_system),
        )
        _add_layer_to_viewer(self._viewer, layer)
        self.register_layer(
            layer,
            sdata=sdata,
            element_name=label_name,
            element_type="labels",
            coordinate_system=coordinate_system,
        )
        return layer

    def remove_labels_layer(self, sdata: SpatialData, label_name: str, coordinate_system: str) -> Labels | None:
        """Remove the loaded labels layer for one labels element in one coordinate system."""
        layer = self._get_loaded_labels_layer_for_coordinate_system(sdata, label_name, coordinate_system)
        if layer is None:
            return None

        self._remove_layer_from_viewer_and_registry(layer)
        return layer

    def ensure_image_loaded(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        mode: ImageDisplayMode = "stack",
        channels: Sequence[int | str] | None = None,
        channel_colors: Sequence[str] | None = None,
    ) -> Image | list[Image]:
        """Load an image element into napari if it is not already present."""
        images = getattr(sdata, "images", {})
        if image_name not in images:
            raise ValueError(f"Image element `{image_name}` is not available in the selected SpatialData object.")

        image_element = images[image_name]
        available_coordinate_systems = set(get_transformation(image_element, get_all=True).keys())
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available for image element `{image_name}`."
            )

        if mode == "stack":
            existing_overlay_layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode="overlay",
            )
            for layer in existing_overlay_layers:
                self._remove_layer_from_viewer_and_registry(layer)

            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=mode,
            )
            existing_layer = layers[0] if layers else None
            if existing_layer is not None:
                return existing_layer

            image_data, rgb = _get_stack_image_layer_data(image_element)
            layer = Image(
                image_data,
                name=image_name,
                affine=_get_affine_transform(image_element, coordinate_system),
                rgb=rgb,
            )
            _add_layer_to_viewer(self._viewer, layer)
            self.register_layer(
                layer,
                sdata=sdata,
                element_name=image_name,
                element_type="image",
                coordinate_system=coordinate_system,
                image_display_mode=mode,
            )
            return layer

        if mode != "overlay":
            raise NotImplementedError(f"Image display mode `{mode}` is not implemented yet.")

        resolved_channels = _resolve_overlay_channels(image_element, channels)
        resolved_colors = _resolve_overlay_colors(len(resolved_channels), channel_colors)
        affine = _get_affine_transform(image_element, coordinate_system)

        loaded_overlay_layers: list[Image] = []
        desired_channel_indices = {channel_index for channel_index, _ in resolved_channels}

        layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="stack",
        )
        existing_stack_layer = layers[0] if layers else None
        if existing_stack_layer is not None:
            self._remove_layer_from_viewer_and_registry(existing_stack_layer)

        existing_overlay_layers = self._get_loaded_image_layer_for_coordinate_system(
            sdata,
            image_name,
            coordinate_system,
            image_display_mode="overlay",
        )
        for layer in existing_overlay_layers:
            binding = self._layer_bindings.get_binding(layer)
            assert binding is not None
            if binding.channel_index in desired_channel_indices:
                continue
            self._remove_layer_from_viewer_and_registry(layer)

        for (channel_index, channel_name), color in zip(resolved_channels, resolved_colors, strict=False):
            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=mode,
                channel_index=channel_index,
            )
            existing_layer = layers[0] if layers else None
            if existing_layer is None:
                layer = Image(
                    _get_overlay_channel_layer_data(image_element, channel_index),
                    name=f"{image_name}[{channel_name}]",
                    affine=affine,
                    blending="additive",
                    colormap=color,
                )
                _add_layer_to_viewer(self._viewer, layer)
                self.register_layer(
                    layer,
                    sdata=sdata,
                    element_name=image_name,
                    element_type="image",
                    coordinate_system=coordinate_system,
                    image_display_mode=mode,
                    channel_index=channel_index,
                    channel_name=channel_name,
                )
            else:
                layer = existing_layer
                layer.name = f"{image_name}[{channel_name}]"
                layer.blending = "additive"
                layer.colormap = color

            loaded_overlay_layers.append(layer)

        return loaded_overlay_layers

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

    def remove_image_layers(self, sdata: SpatialData, image_name: str, coordinate_system: str) -> list[Image]:
        """Remove loaded stack and overlay layers for one image in one coordinate system."""
        removed_layers: list[Image] = []
        for image_display_mode in ("stack", "overlay"):
            layers = self._get_loaded_image_layer_for_coordinate_system(
                sdata,
                image_name,
                coordinate_system,
                image_display_mode=image_display_mode,
            )
            for layer in layers:
                self._remove_layer_from_viewer_and_registry(layer)
                removed_layers.append(layer)

        return removed_layers

    def _iter_candidate_layers(self) -> Iterable[Layer]:
        layers = getattr(self._viewer, "layers", None)
        if layers is not None:
            return tuple(layers)
        return ()

    def _remove_layer_from_viewer_and_registry(self, layer: Layer) -> None:
        """Remove a layer from the viewer and keep the registry synchronized."""
        _remove_layer_from_viewer(self._viewer, layer)
        # On the normal path, viewer removal emits `layers.events.removed` and
        # `_on_viewer_layer_removed(...)` has already unregistered the binding.
        # Keep this fallback for viewer-like objects that remove layers without
        # exposing or emitting the napari removal event.
        if self._layer_bindings.get_binding(layer) is not None:
            self.unregister_layer(layer)

    def _get_loaded_labels_layer_for_coordinate_system(
        self,
        sdata: SpatialData,
        label_name: str,
        coordinate_system: str,
    ) -> Labels | None:
        return self.get_loaded_labels_layer(sdata, label_name, coordinate_system)

    def _get_loaded_image_layer_for_coordinate_system(
        self,
        sdata: SpatialData,
        image_name: str,
        coordinate_system: str,
        *,
        image_display_mode: ImageDisplayMode,
        channel_index: int | None = None,
    ) -> list[Image]:
        matches: list[Image] = []
        for layer in self._iter_candidate_layers():
            if not _is_spatialdata_image_layer(layer):
                continue

            binding = self._layer_bindings.get_binding(layer)
            if not _matches_binding(binding, sdata=sdata, element_name=image_name, element_type="image"):
                continue
            if binding.coordinate_system != coordinate_system:
                continue
            if binding.image_display_mode != image_display_mode:
                continue
            if channel_index is not None and binding.channel_index != channel_index:
                continue
            matches.append(layer)

        return matches


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


def _apply_legacy_spatialdata_metadata(
    layer: Layer,
    *,
    sdata: SpatialData | None,
    element_name: str,
    coordinate_system: str | None,
) -> None:
    """Mirror legacy spatialdata layer metadata during the VW-05 transition.

    ``SpatialDataViewerBinding`` still resolves live layers through the older
    ``metadata['sdata']`` / ``metadata['name']`` contract in a few places.
    Keep Harpy-managed layers discoverable there until slice 4 migrates those
    call sites fully onto ``ViewerAdapter`` / layer bindings.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        layer.metadata = metadata

    _set_optional_metadata_value(metadata, "sdata", sdata)
    metadata["name"] = element_name
    _set_optional_metadata_value(metadata, "_current_cs", coordinate_system)


def _set_optional_metadata_value(metadata: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        metadata.pop(key, None)
        return
    metadata[key] = value


def _add_layer_to_viewer(viewer: Any | None, layer: Layer) -> None:
    if viewer is None:
        raise ValueError("A napari viewer is required to load layers into the viewer.")

    add_layer = getattr(viewer, "add_layer", None)
    if callable(add_layer):
        add_layer(layer)
        return

    layers = getattr(viewer, "layers", None)
    if layers is None:
        raise ValueError("The provided viewer does not expose napari-compatible layer APIs.")

    append = getattr(layers, "append", None)
    if callable(append):
        append(layer)
        events = getattr(layers, "events", None)
        inserted = getattr(events, "inserted", None)
        if inserted is not None and hasattr(inserted, "emit"):
            inserted.emit(layer)
        return

    raise ValueError("The provided viewer does not support adding layers.")


def _remove_layer_from_viewer(viewer: Any | None, layer: Layer) -> None:
    if viewer is None:
        raise ValueError("A napari viewer is required to remove layers from the viewer.")

    layers = getattr(viewer, "layers", None)
    if layers is None:
        raise ValueError("The provided viewer does not expose napari-compatible layer APIs.")

    remove = getattr(layers, "remove", None)
    if callable(remove):
        remove(layer)
        return

    raise ValueError("The provided viewer does not support removing layers.")


def _get_stack_image_layer_data(element: DataArray | DataTree) -> tuple[DataArray | list[DataArray], bool]:
    """Prepare image data for one napari stack-mode image layer.

    The helper keeps ordinary multiplex images in their existing raster layout,
    but detects true RGB(A) images by channel names and converts those to
    channel-last layout so napari can render them with ``rgb=True``.

    For multiscale rasters, the returned image data is flattened to the list
    of per-scale arrays expected by napari layers.
    """
    axes = get_axes_names(element)

    if "c" in axes:
        assert axes.index("c") == 0

        if isinstance(element, DataArray):
            channel_coords = element.coords.indexes["c"]
        else:
            channel_coords = element["scale0"].coords.indexes["c"]
    else:
        channel_coords = []

    if len(channel_coords) != 0 and set(channel_coords) - {"r", "g", "b"} <= {"a"}:
        rgb = True
        if isinstance(element, DataArray):
            image_data: DataArray | DataTree = element.transpose("y", "x", "c").reindex(
                c=["r", "g", "b", "a"][: len(channel_coords)]
            )
        else:
            image_data = element.msi.transpose("y", "x", "c")
            image_data = image_data.msi.reindex_data_arrays({"c": ["r", "g", "b", "a"][: len(channel_coords)]})
    else:
        rgb = False
        image_data = element

    if isinstance(image_data, DataTree):
        return _flatten_multiscale_element(image_data), rgb

    return image_data, rgb


def _get_overlay_channel_layer_data(element: DataArray | DataTree, channel_index: int) -> DataArray | list[DataArray]:
    if isinstance(element, DataTree):
        channel_arrays: list[DataArray] = []
        for key in element:
            scale_array = next(iter(element[key].values()))
            channel_array = scale_array.isel(c=channel_index)
            channel_arrays.append(channel_array)
        return channel_arrays

    return element.isel(c=channel_index)


def _resolve_overlay_channels(
    element: DataArray | DataTree,
    channels: Sequence[int | str] | None,
) -> list[tuple[int, str]]:
    axes = get_axes_names(element)
    if "c" not in axes:
        channel_names: list[Any] = []
    elif isinstance(element, DataArray):
        channel_names = list(element.coords.indexes["c"])
    else:
        scale0 = next(iter(element["scale0"].values()))
        channel_names = list(scale0.coords.indexes["c"])

    if not channel_names:
        raise ValueError("Overlay mode requires an image element with a channel axis.")
    if not channels:
        raise ValueError("Overlay mode requires at least one selected channel.")

    resolved: list[tuple[int, str]] = []
    seen_indices: set[int] = set()
    for channel in channels:
        if isinstance(channel, int):
            channel_index = channel
            if channel_index < 0 or channel_index >= len(channel_names):
                raise ValueError(f"Channel index `{channel_index}` is out of range for the selected image element.")
            channel_name = str(channel_names[channel_index])
        else:
            if channel not in channel_names:
                raise ValueError(
                    f"Channel `{channel}` is not available in the selected image element. "
                    "If needed, update the channel names in the SpatialData object with "
                    "`sdata.set_channel_names(...)`."
                )
            channel_index = channel_names.index(channel)
            channel_name = str(channel)

        if channel_index in seen_indices:
            raise ValueError("Overlay mode does not accept duplicate channel selections.")

        resolved.append((channel_index, channel_name))
        seen_indices.add(channel_index)

    return resolved


def _resolve_overlay_colors(channel_count: int, channel_colors: Sequence[str] | None) -> list[str]:
    if channel_colors is None:
        return [DEFAULT_OVERLAY_COLORS[index % len(DEFAULT_OVERLAY_COLORS)] for index in range(channel_count)]

    if len(channel_colors) != channel_count:
        raise ValueError("The number of overlay channel colors must match the number of selected channels.")

    return list(channel_colors)


def _flatten_multiscale_element(element: DataTree) -> list[DataArray]:
    multiscale_data: list[DataArray] = []
    for key in element:
        values = element[key].values()
        assert len(values) == 1
        multiscale_data.append(next(iter(values)))
    return multiscale_data


def _get_affine_transform(element: DataArray | DataTree, coordinate_system: str) -> np.ndarray | None:
    transformations = get_transformation(element, get_all=True)
    transform = transformations.get(coordinate_system)
    if transform is None:
        return None

    axes_element = get_axes_names(element)
    if "z" in axes_element:
        axes = ("z", "y", "x")
    else:
        axes = ("y", "x")
    return transform.to_affine_matrix(input_axes=axes, output_axes=axes)


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


def _matches_legacy_spatialdata_layer_metadata(
    layer: Layer,
    *,
    sdata: SpatialData,
    element_name: str,
    coordinate_system: str | None,
) -> bool:
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    if metadata.get("sdata") is not sdata:
        return False
    if metadata.get("name") != element_name:
        return False
    if coordinate_system is None:
        return True

    layer_coordinate_system = metadata.get("coordinate_system", metadata.get("_current_cs"))
    return layer_coordinate_system is None or layer_coordinate_system == coordinate_system


def _is_pickable_labels_layer(layer: Layer) -> bool:
    events = getattr(layer, "events", None)
    return isinstance(layer, Labels) and getattr(events, "selected_label", None) is not None


def _is_spatialdata_image_layer(layer: Layer) -> bool:
    # napari `Labels` layers are scalar-field siblings of `Image`, not subclasses.
    return isinstance(layer, Image)
