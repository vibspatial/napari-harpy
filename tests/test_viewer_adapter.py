from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from napari.layers import Image, Labels

from napari_harpy._app_state import get_or_create_app_state
from napari_harpy._viewer_adapter import LayerBindingRegistry, ViewerAdapter


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[object] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self, layers: list[object] | None = None) -> None:
        self.layers = DummyLayers(layers)

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


class UnsupportedViewer:
    def __init__(self) -> None:
        self.layers = None


def make_labels_layer(*, sdata, label_name: str = "blobs_labels", metadata: dict[str, object] | None = None) -> Labels:
    return Labels(
        sdata.labels[label_name],
        name=label_name,
        metadata={} if metadata is None else metadata,
    )


def make_image_layer(*, name: str, metadata: dict[str, object] | None = None) -> Image:
    return Image(
        np.zeros((8, 8), dtype=np.float32),
        name=name,
        metadata={} if metadata is None else metadata,
    )


def test_app_state_initializes_viewer_services() -> None:
    viewer = DummyViewer()

    state = get_or_create_app_state(viewer)

    assert state.layer_bindings is not None
    assert state.viewer_adapter is not None
    assert state.viewer_adapter.layer_bindings is state.layer_bindings


def test_layer_binding_registry_registers_and_unregisters_layers() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="raw image")

    binding = registry.register_layer(
        layer,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
    )

    assert binding.layer is layer
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert registry.get_binding(layer) == binding
    assert registry.find_bindings(element_name="blobs_image", element_type="image") == [binding]
    assert layer.metadata["element_name"] == "blobs_image"
    assert layer.metadata["element_type"] == "image"
    assert layer.metadata["coordinate_system"] == "global"

    removed = registry.unregister_layer(layer)

    assert removed == binding
    assert registry.get_binding(layer) is None


def test_layer_binding_registry_tracks_channel_overlay_identity() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="channel_0")

    binding = registry.register_layer(
        layer,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )

    assert binding.image_display_mode == "overlay"
    assert binding.channel_index == 0
    assert binding.channel_name == "DAPI"
    assert registry.find_bindings(
        element_name="blobs_image",
        element_type="image",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    ) == [binding]
    assert layer.metadata["image_display_mode"] == "overlay"
    assert layer.metadata["channel_index"] == 0
    assert layer.metadata["channel_name"] == "DAPI"


def test_viewer_adapter_activate_layer_selects_only_matching_layer() -> None:
    image_layer = make_image_layer(name="image")
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    activated = adapter.activate_layer(image_layer)

    assert activated is True
    assert viewer.layers.selection.active is image_layer


def test_viewer_adapter_ensure_labels_loaded_rejects_unsupported_viewer(sdata_blobs) -> None:
    adapter = ViewerAdapter(UnsupportedViewer())

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    except ValueError as error:
        assert "does not expose napari-compatible layer APIs" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unsupported viewer.")


def test_viewer_adapter_finds_registered_labels_layer(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_layer(
        labels_layer,
        sdata=sdata_blobs,
        element_name="blobs_labels",
        element_type="labels",
        coordinate_system="global",
    )

    assert adapter.get_loaded_labels_layer(sdata_blobs, "blobs_labels") is labels_layer


def test_viewer_adapter_ignores_unregistered_labels_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    labels_layer = make_labels_layer(
        sdata=sdata_blobs,
        metadata={"sdata": sdata_blobs, "name": "blobs_labels", "_current_cs": "global"},
    )
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layer = adapter.get_loaded_labels_layer(sdata_blobs, "blobs_labels")

    assert loaded_layer is None
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_remove_labels_layer_removes_matching_binding(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_layer(
        labels_layer,
        sdata=sdata_blobs,
        element_name="blobs_labels",
        element_type="labels",
        coordinate_system="global",
    )

    removed_layer = adapter.remove_labels_layer(sdata_blobs, "blobs_labels", "global")

    assert removed_layer is labels_layer
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_returns_registered_image_layers_in_viewer_order(sdata_blobs) -> None:
    first_layer = make_image_layer(name="channel_0")
    second_layer = make_image_layer(name="channel_1")
    viewer = DummyViewer([second_layer, first_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_layer(
        first_layer,
        sdata=sdata_blobs,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
    )
    adapter.register_layer(
        second_layer,
        sdata=sdata_blobs,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
    )

    assert adapter.get_loaded_image_layers(sdata_blobs, "blobs_image") == [second_layer, first_layer]


def test_viewer_adapter_remove_image_layers_removes_stack_and_overlay_bindings(sdata_blobs) -> None:
    stack_layer = make_image_layer(name="blobs_image")
    overlay_layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([stack_layer, overlay_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_layer(
        stack_layer,
        sdata=sdata_blobs,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    adapter.register_layer(
        overlay_layer,
        sdata=sdata_blobs,
        element_name="blobs_image",
        element_type="image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )

    removed_layers = adapter.remove_image_layers(sdata_blobs, "blobs_image", "global")

    assert removed_layers == [stack_layer, overlay_layer]
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert adapter.layer_bindings.get_binding(overlay_layer) is None


def test_viewer_adapter_ignores_unregistered_image_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    image_layer = make_image_layer(
        name="raw image",
        metadata={"sdata": sdata_blobs, "name": "blobs_image", "_current_cs": "global"},
    )
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layers = adapter.get_loaded_image_layers(sdata_blobs, "blobs_image")

    assert loaded_layers == []
    assert adapter.layer_bindings.get_binding(image_layer) is None


def test_viewer_adapter_ensure_labels_loaded_adds_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    assert layer in viewer.layers
    assert layer.name == "blobs_labels"
    assert layer.affine is not None
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_name == "blobs_labels"
    assert binding.element_type == "labels"
    assert binding.coordinate_system == "global"
    assert "sdata" not in layer.metadata


def test_viewer_adapter_ensure_labels_loaded_reuses_matching_existing_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    second = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    assert first is second
    assert len(viewer.layers) == 1


def test_viewer_adapter_unregisters_binding_when_user_removes_labels_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    viewer.layers.remove(layer)

    assert layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_labels_loaded_supports_multiscale_labels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_multiscale_labels", "global")

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert layer in viewer.layers


def test_viewer_adapter_ensure_labels_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "not_a_coordinate_system")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_adds_stack_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert layer in viewer.layers
    assert layer.name == "blobs_image"
    assert layer.affine is not None
    assert layer.rgb is False
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert binding.image_display_mode == "stack"
    assert layer.metadata["image_display_mode"] == "stack"


def test_viewer_adapter_ensure_image_loaded_reuses_matching_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    second = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert first is second
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_image_loaded_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_multiscale_image", "global", mode="stack")

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert layer in viewer.layers
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"


def test_viewer_adapter_ensure_image_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "not_a_coordinate_system", mode="stack")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_overlay_adds_one_layer_per_selected_channel(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 2],
        channel_colors=["blue", "magenta"],
    )

    assert isinstance(layers, list)
    assert len(layers) == 2
    assert [layer.name for layer in layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert [layer.blending for layer in layers] == ["additive", "additive"]
    assert [binding.channel_index for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [0, 2]
    assert [binding.channel_name for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [
        "0",
        "2",
    ]


def test_viewer_adapter_ensure_image_loaded_overlay_reuses_existing_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    second = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["cyan", "yellow"],
    )

    assert first == second
    assert len(viewer.layers) == 2
    assert [str(layer.colormap).lower() for layer in second] != []


def test_viewer_adapter_unregisters_binding_when_user_removes_overlay_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )

    viewer.layers.remove(layers[0])

    assert layers[0] not in viewer.layers
    assert adapter.layer_bindings.get_binding(layers[0]) is None
    assert adapter.layer_bindings.get_binding(layers[1]) is not None


def test_viewer_adapter_ensure_image_loaded_overlay_removes_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )

    assert stack_layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert len(overlay_layers) == 2
    assert all(layer in viewer.layers for layer in overlay_layers)


def test_viewer_adapter_ensure_image_loaded_overlay_removes_stale_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[1],
        channel_colors=["green"],
    )

    assert len(layers) == 1
    assert len(viewer.layers) == 1
    binding = adapter.layer_bindings.get_binding(layers[0])
    assert binding is not None
    assert binding.channel_index == 1


def test_viewer_adapter_ensure_image_loaded_stack_removes_existing_overlay_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert all(layer not in viewer.layers for layer in overlay_layers)
    assert all(adapter.layer_bindings.get_binding(layer) is None for layer in overlay_layers)
    assert stack_layer in viewer.layers


def test_viewer_adapter_ensure_image_loaded_overlay_accepts_channel_names(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    sdata_blobs.images["blobs_image"] = sdata_blobs.images["blobs_image"].assign_coords(c=["DAPI", "CD3", "CD8"])
    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=["CD3", "CD8"],
        channel_colors=["green", "magenta"],
    )

    assert [layer.name for layer in layers] == ["blobs_image[CD3]", "blobs_image[CD8]"]
    assert [adapter.layer_bindings.get_binding(layer).channel_name for layer in layers] == ["CD3", "CD8"]


def test_viewer_adapter_ensure_image_loaded_overlay_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_multiscale_image",
        "global",
        mode="overlay",
        channels=[0, 2],
    )

    assert len(layers) == 2
    assert all(layer.multiscale is True for layer in layers)
    assert all(len(layer.data) == 3 for layer in layers)


def test_viewer_adapter_ensure_image_loaded_overlay_requires_selected_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=None)
    except ValueError as error:
        assert "requires at least one selected channel" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to require selected channels for overlay mode.")


def test_viewer_adapter_ensure_image_loaded_overlay_rejects_duplicate_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=[0, 0])
    except ValueError as error:
        assert "does not accept duplicate channel selections" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject duplicate overlay channels.")
