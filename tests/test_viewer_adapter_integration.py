from __future__ import annotations

from napari_harpy._viewer_adapter import ViewerAdapter


def test_viewer_adapter_real_viewer_overlay_load_activate_and_remove(
    make_napari_viewer,
    sdata_blobs,
) -> None:
    """Exercise the highest-value adapter flow against a real napari viewer.

    This stays intentionally small: one real-viewer test that verifies overlay
    channel loading, viewer-backed activation, and registry synchronization
    after a user-style layer removal from the napari layer list.
    """
    viewer = make_napari_viewer(show=False)
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )

    assert len(layers) == 2
    assert all(layer in viewer.layers for layer in layers)
    assert [layer.name for layer in layers] == ["blobs_image[0]", "blobs_image[1]"]

    assert adapter.activate_layer(layers[0]) is True
    assert viewer.layers.selection.active is layers[0]

    viewer.layers.remove(layers[0])

    assert layers[0] not in viewer.layers
    assert adapter.layer_bindings.get_binding(layers[0]) is None
    assert adapter.layer_bindings.get_binding(layers[1]) is not None
