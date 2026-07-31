from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from napari.layers import Image

from napari_harpy.viewer.adapter import ImageLayerBinding
from napari_harpy.widgets.overlay_channel_row import (
    _colormap_presentation_from_layer,
    _OverlayChannelRow,
)


def _make_binding(layer: Image) -> ImageLayerBinding:
    return ImageLayerBinding(
        layer=layer,
        element_name="image",
        coordinate_system="global",
        sdata_id=1,
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )


def test_overlay_channel_row_emits_intent_without_mutating_layer(qtbot) -> None:
    layer = Image(np.zeros((4, 4)), colormap="#00FFFF")
    row = _OverlayChannelRow(_make_binding(layer))
    visibility_requests: list[tuple[int, bool]] = []
    color_requests: list[tuple[int, str]] = []
    removal_requests: list[int] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(
        lambda channel_index, visible: visibility_requests.append((channel_index, visible))
    )
    row.color_change_requested.connect(lambda channel_index, color: color_requests.append((channel_index, color)))
    row.remove_requested.connect(removal_requests.append)
    original_colormap = layer.colormap

    row.visibility_button.click()
    row.color_button.color_selected.emit("#123456")
    row.remove_button.click()

    assert visibility_requests == [(0, False)]
    assert color_requests == [(0, "#123456")]
    assert removal_requests == [0]
    assert layer.visible is True
    assert layer.colormap is original_colormap


def test_overlay_channel_row_renders_native_events_and_disposes_idempotently(qtbot) -> None:
    layer = Image(np.zeros((4, 4)), colormap="#00FFFF")
    row = _OverlayChannelRow(_make_binding(layer))
    visibility_requests: list[tuple[int, bool]] = []
    color_requests: list[tuple[int, str]] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(
        lambda channel_index, visible: visibility_requests.append((channel_index, visible))
    )
    row.color_change_requested.connect(lambda channel_index, color: color_requests.append((channel_index, color)))

    layer.visible = False
    layer.colormap = "viridis"

    assert not row.visibility_button.isChecked()
    assert row.color_button.gradient_name == "viridis"
    assert "qlineargradient" in row.color_button.styleSheet()
    assert visibility_requests == []
    assert color_requests == []

    row.dispose()
    row.dispose()
    layer.visible = True
    layer.colormap = "#ABCDEF"

    assert not row.visibility_button.isChecked()
    assert row.color_button.gradient_name == "viridis"


def test_overlay_colormap_presentation_rejects_malformed_color_rows() -> None:
    layer = SimpleNamespace(
        colormap=SimpleNamespace(
            name="malformed",
            colors=np.asarray([[np.nan, 0.0, 0.0, 1.0]]),
        )
    )

    assert _colormap_presentation_from_layer(layer) is None
