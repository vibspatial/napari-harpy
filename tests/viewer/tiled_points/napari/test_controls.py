from __future__ import annotations

from collections.abc import Iterator
from html import escape
from uuid import uuid4

import numpy as np
import pytest
from qtpy.QtWidgets import QDockWidget

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsLayerStatus,
)
from napari_harpy.viewer.tiled_points.napari.controls import QtTiledPointsLayerControls


@pytest.fixture
def controls(qtbot) -> QtTiledPointsLayerControls:
    layer = TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=str(uuid4()),
            points_name="spots",
            value_column="feature_name",
            value_count=3,
            x_origin=0.0,
            y_origin=0.0,
            x_min=3.0,
            x_max=23.0,
            y_min=2.0,
            y_max=12.0,
        ),
        value_palette=np.full((3, 4), 255, dtype=np.uint8),
        max_vertex_payload_bytes=1_000_000,
    )
    controls = QtTiledPointsLayerControls(layer)
    qtbot.addWidget(controls)
    return controls


@pytest.fixture
def controls_dock(qtbot, controls: QtTiledPointsLayerControls) -> Iterator[QDockWidget]:
    dock = QDockWidget("Layer controls")
    qtbot.addWidget(dock)
    dock.setWidget(controls)
    yield dock
    # Let qtbot close the independently registered controls before deletion.
    controls.setParent(None)


def test_controls_update_layer_style_and_read_only_status(controls: QtTiledPointsLayerControls) -> None:
    layer = controls.layer
    controls.point_diameter_spin_box.setValue(6.5)
    layer.display_status = TiledPointsLayerStatus(
        level=1,
        level_kind="bridge",
        rendered_point_count=1234,
        rendered_tile_count=3,
        message="Ready",
        sampled=True,
        omitted_value_ids=(9,),
    )

    assert layer.point_diameter == 6.5
    assert controls.level_label.text() == "Bridge"
    assert controls.rendered_label.text() == "1,234 points / 3 tiles"
    assert controls.status_label.text() == "Ready"
    assert controls.sampling_label.text() == "Sampled; omitted value IDs: 9"
    assert not controls.transform_button.isEnabled()


@pytest.mark.parametrize(
    ("message", "omitted_value_ids"),
    [
        ("View exceeds the point budget (100,000 estimated points); retaining the previous view", ()),
        ("Could not open cache: /" + "long_path_segment_" * 100 + "/<cache>&data.zarr", ()),
        ("Ready", tuple(range(500))),
    ],
    ids=["budget-warning", "unbroken-path", "many-omitted-values"],
)
def test_long_diagnostics_do_not_widen_controls_or_dock(
    controls: QtTiledPointsLayerControls,
    controls_dock: QDockWidget,
    message: str,
    omitted_value_ids: tuple[int, ...],
) -> None:
    """Long status and sampling text must not steal horizontal space from the canvas."""
    original_controls_width = controls.minimumSizeHint().width()
    original_dock_width = controls_dock.minimumSizeHint().width()

    controls.layer.display_status = TiledPointsLayerStatus(
        message=message, sampled=bool(omitted_value_ids), omitted_value_ids=omitted_value_ids
    )
    controls.layout().activate()

    assert controls.minimumSizeHint().width() <= original_controls_width
    assert controls.sizeHint().width() <= original_controls_width
    assert controls_dock.minimumSizeHint().width() <= original_dock_width
    assert controls.status_label.text() == message
    assert controls.status_label.toolTip() == f"<qt>{escape(message)}</qt>"
    sampling = (
        "Sampled; omitted value IDs: " + ", ".join(str(value) for value in omitted_value_ids)
        if omitted_value_ids
        else "No sampled omission"
    )
    assert controls.sampling_label.text() == sampling
    assert controls.sampling_label.toolTip() == f"<qt>{escape(sampling)}</qt>"

    # Tooltips must also follow subsequent updates, not retain an old warning.
    controls.layer.display_status = TiledPointsLayerStatus(message="Ready")
    assert controls.status_label.toolTip() == "<qt>Ready</qt>"
    assert controls.sampling_label.toolTip() == "<qt>No sampled omission</qt>"


def test_wrapped_warning_uses_more_height_in_a_narrower_dock(controls: QtTiledPointsLayerControls) -> None:
    controls.layer.display_status = TiledPointsLayerStatus(
        message="View exceeds the point budget (100,000 estimated points); retaining the previous view",
        sampled=True,
        omitted_value_ids=tuple(range(20)),
    )
    for label in (controls.status_label, controls.sampling_label):
        assert label.heightForWidth(120) > label.heightForWidth(240)
    assert controls.layout().hasHeightForWidth()
