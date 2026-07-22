from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import Mode

import napari_harpy.widgets.shapes_annotation._layer_state as shapes_annotation_layer_state_module


def test_shapes_layer_baseline_capture_and_restore_round_trip(
    restore_triangulation_backend: None,
) -> None:
    """Restore copied geometry, metadata, styles, selection, mode, and caches."""
    polygon = np.asarray(
        [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        dtype=float,
    )
    path = np.asarray([[8.0, 0.0], [9.0, 2.0], [10.0, 0.0]], dtype=float)
    layer = Shapes(
        [polygon, path],
        shape_type=["polygon", "path"],
        features=pd.DataFrame(
            {
                "instance_id": ["polygon-0", "path-1"],
                "label": ["first", "second"],
            }
        ),
    )
    layer.feature_defaults = pd.DataFrame(
        {
            "instance_id": [pd.NA],
            "label": ["next-shape"],
        }
    )
    layer.edge_color = np.asarray([to_rgba("#112233"), to_rgba("#445566")])
    layer.face_color = np.asarray([to_rgba("#01020344"), to_rgba("#05060744")])
    layer.edge_width = [2, 4]
    layer.z_index = [3, 5]
    layer.opacity = 0.42
    layer.current_edge_color = "#abcdef"
    layer.current_face_color = "#12345678"
    layer.current_edge_width = 11
    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1}

    baseline = shapes_annotation_layer_state_module._capture_shapes_layer_baseline(layer)

    replacement = np.asarray(
        [[20.0, 20.0], [20.0, 22.0], [22.0, 20.0]],
        dtype=float,
    )
    layer.data = [(replacement, "polygon")]
    layer.features = pd.DataFrame({"instance_id": ["replacement"], "label": ["changed"]})
    layer.feature_defaults = pd.DataFrame({"instance_id": ["next"], "label": ["changed-default"]})
    layer.edge_color = np.asarray([to_rgba("#fedcba")])
    layer.face_color = np.asarray([to_rgba("#87654321")])
    layer.edge_width = [9]
    layer.z_index = [12]
    layer.opacity = 0.9
    layer.current_edge_color = "#010101"
    layer.current_face_color = "#02020203"
    layer.current_edge_width = 13
    layer.mode = Mode.SELECT
    layer.selected_data = set()

    emitted_events: list[str] = []
    layer.events.data.connect(lambda event: emitted_events.append("data"))
    layer.events.features.connect(lambda event: emitted_events.append("features"))

    shapes_annotation_layer_state_module._restore_shapes_layer_baseline(layer, baseline)

    assert len(layer.data) == len(baseline.data)
    for actual, expected in zip(layer.data, baseline.data, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert tuple(layer.shape_type) == baseline.shape_types
    pd.testing.assert_frame_equal(layer.features, baseline.features)
    pd.testing.assert_frame_equal(layer.feature_defaults, baseline.feature_defaults)
    np.testing.assert_allclose(layer.edge_color, baseline.style.edge_color)
    np.testing.assert_allclose(layer.face_color, baseline.style.face_color)
    assert tuple(layer.edge_width) == baseline.style.edge_width
    assert tuple(layer.z_index) == baseline.style.z_index
    assert layer.opacity == baseline.style.opacity
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(baseline.style.current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(baseline.style.current_face_color))
    assert layer.current_edge_width == baseline.style.current_edge_width
    assert layer.mode == baseline.mode
    assert frozenset(layer.selected_data) == baseline.selected_data
    assert len(layer._data_view.shapes) == len(baseline.data)
    assert emitted_events == []
