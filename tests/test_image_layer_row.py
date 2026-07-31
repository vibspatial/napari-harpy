from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from napari.layers import Image

from napari_harpy.viewer.adapter import ImageLayerBinding
from napari_harpy.widgets.image_layer_row import (
    _colormap_presentation_from_layer,
    _ImageLayerRow,
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


def test_image_layer_row_emits_identity_free_intent_without_mutating_layer(qtbot) -> None:
    layer = Image(np.zeros((4, 4)), colormap="#00FFFF")
    row = _ImageLayerRow(
        _make_binding(layer),
        display_label="DAPI",
        accessibility_label="channel DAPI",
    )
    visibility_requests: list[bool] = []
    color_requests: list[str] = []
    removal_requests: list[str] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(visibility_requests.append)
    row.color_change_requested.connect(color_requests.append)
    row.remove_requested.connect(lambda: removal_requests.append("remove"))
    original_colormap = layer.colormap

    row.visibility_button.click()
    row.color_button.color_selected.emit("#123456")
    row.remove_button.click()

    assert row.layer_label.text() == "DAPI"
    assert "channel DAPI" in row.visibility_button.toolTip()
    assert visibility_requests == [False]
    assert color_requests == ["#123456"]
    assert removal_requests == ["remove"]
    assert layer.visible is True
    assert layer.colormap is original_colormap


def test_image_layer_row_renders_native_events_and_disposes_idempotently(qtbot) -> None:
    layer = Image(np.zeros((4, 4)), colormap="#00FFFF")
    row = _ImageLayerRow(
        _make_binding(layer),
        display_label="DAPI",
        accessibility_label="channel DAPI",
    )
    visibility_requests: list[bool] = []
    color_requests: list[str] = []
    qtbot.addWidget(row)
    row.visibility_change_requested.connect(visibility_requests.append)
    row.color_change_requested.connect(color_requests.append)

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


def test_image_layer_row_can_hide_colormap_control_for_rgb(qtbot) -> None:
    layer = Image(np.zeros((4, 4, 3)), rgb=True)
    row = _ImageLayerRow(
        ImageLayerBinding(
            layer=layer,
            element_name="image",
            coordinate_system="global",
            sdata_id=1,
            image_display_mode="stack",
        ),
        display_label="RGB stack",
        accessibility_label="RGB stack for image image",
        show_colormap=False,
    )
    qtbot.addWidget(row)

    assert row.layer_label.text() == "RGB stack"
    assert row.color_button.isHidden()
    assert not row.visibility_button.isHidden()
    assert not row.remove_button.isHidden()


def test_overlay_colormap_presentation_rejects_malformed_color_rows() -> None:
    layer = SimpleNamespace(
        colormap=SimpleNamespace(
            name="malformed",
            colors=np.asarray([[np.nan, 0.0, 0.0, 1.0]]),
        )
    )

    assert _colormap_presentation_from_layer(layer) is None
