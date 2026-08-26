from __future__ import annotations

from uuid import uuid4

import numpy as np

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsLayerStatus,
)
from napari_harpy.viewer.tiled_points.napari.controls import QtTiledPointsLayerControls


def test_controls_update_layer_style_and_read_only_status(qtbot) -> None:
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
        max_gpu_tile_bytes=1_000_000,
    )
    controls = QtTiledPointsLayerControls(layer)
    qtbot.addWidget(controls)

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
