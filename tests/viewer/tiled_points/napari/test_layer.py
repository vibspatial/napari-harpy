from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsLayerStatus,
)


def _dataset_reference(**overrides: object) -> TiledPointsDatasetReference:
    values = {
        "cache_generation_id": str(uuid4()),
        "points_name": "spots",
        "value_column": "feature_name",
        "x_min": 3.0,
        "x_max": 23.0,
        "y_min": 2.0,
        "y_max": 12.0,
    }
    values.update(overrides)
    return TiledPointsDatasetReference(**values)


def test_tiled_points_layer_keeps_logical_data_and_complete_extent() -> None:
    reference = _dataset_reference()
    layer = TiledPointsLayerModel(reference, scale=(2.0, 3.0), translate=(5.0, 7.0))

    assert layer.data is reference
    assert not isinstance(layer.data, np.ndarray)
    np.testing.assert_array_equal(layer.extent.data, np.array(((2.0, 3.0), (12.0, 23.0))))
    np.testing.assert_array_equal(layer.extent.world, np.array(((9.0, 16.0), (29.0, 76.0))))
    assert layer.ndim == 2
    assert layer.axis_labels == ("y", "x")
    assert not layer.editable
    assert layer.mode == "pan_zoom"
    layer.mode = "transform"
    assert layer.mode == "pan_zoom"
    assert layer.get_value((4.0, 5.0), world=False) is None
    assert layer.thumbnail.shape == (32, 32, 4)
    assert layer.thumbnail.dtype == np.uint8


def test_tiled_points_layer_supports_model_lifecycle_without_point_rows() -> None:
    viewer = ViewerModel()
    layer = TiledPointsLayerModel(_dataset_reference())

    viewer.layers.append(layer)
    assert viewer.layers.selection.active is layer
    layer.visible = False
    layer.visible = True
    layer.affine.translate = (11.0, 13.0)
    viewer.reset_view()
    assert np.isfinite(viewer.camera.zoom)
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0


def test_tiled_points_layer_replacement_updates_extent_and_emits_data() -> None:
    layer = TiledPointsLayerModel(_dataset_reference())
    observed: list[TiledPointsDatasetReference] = []
    set_data_count = 0

    def _record_set_data(event: object) -> None:
        nonlocal set_data_count
        del event
        set_data_count += 1

    layer.events.data.connect(lambda event: observed.append(event.value))
    layer.events.set_data.connect(_record_set_data)
    replacement = _dataset_reference(x_min=-4.0, x_max=8.0, y_min=-2.0, y_max=6.0)

    layer.data = replacement

    assert observed == [replacement]
    assert set_data_count == 1
    np.testing.assert_array_equal(layer.extent.data, np.array(((-2.0, -4.0), (6.0, 8.0))))


def test_tiled_points_layer_exposes_style_and_status_events() -> None:
    layer = TiledPointsLayerModel(_dataset_reference())
    diameters: list[float] = []
    statuses: list[TiledPointsLayerStatus] = []
    layer.events.point_diameter.connect(lambda event: diameters.append(event.value))
    layer.events.display_status.connect(lambda event: statuses.append(event.value))
    status = TiledPointsLayerStatus(
        level=2,
        level_kind="spatial",
        rendered_point_count=1234,
        rendered_tile_count=4,
        message="Ready",
        sampled=True,
        omitted_value_ids=(3, 9),
    )

    layer.point_diameter = 4.5
    layer.display_status = status

    assert diameters == [4.5]
    assert statuses == [status]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TiledPointsLayerStatus(), "—"),
        (TiledPointsLayerStatus(level=0, level_kind="exact"), "Exact"),
        (TiledPointsLayerStatus(level=1, level_kind="bridge"), "Bridge"),
        (TiledPointsLayerStatus(level=3, level_kind="spatial"), "Spatial L3"),
    ],
)
def test_layer_status_derives_canonical_level_label(status: TiledPointsLayerStatus, expected: str) -> None:
    assert status.level_label == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"level": 0},
        {"level_kind": "exact"},
        {"level": True, "level_kind": "exact"},
        {"level": 1, "level_kind": "exact"},
        {"level": 2, "level_kind": "bridge"},
    ],
)
def test_layer_status_rejects_incomplete_or_inconsistent_level(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="level"):
        TiledPointsLayerStatus(**overrides)


def test_tiled_points_layer_rejects_array_serialization() -> None:
    layer = TiledPointsLayerModel(_dataset_reference())

    with pytest.raises(NotImplementedError, match="logical cache-backed layer"):
        layer.as_layer_data_tuple()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cache_generation_id": "generation"}, "UUID"),
        ({"points_name": ""}, "points_name"),
        ({"x_min": np.nan}, "x_min"),
        ({"x_min": 3.0, "x_max": 2.0}, "minima"),
    ],
)
def test_dataset_reference_rejects_invalid_identity_or_bounds(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _dataset_reference(**overrides)
