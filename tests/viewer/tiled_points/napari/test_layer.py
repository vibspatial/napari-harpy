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
        "value_count": 3,
        "x_origin": 0.0,
        "y_origin": 0.0,
        "x_min": 3.0,
        "x_max": 23.0,
        "y_min": 2.0,
        "y_max": 12.0,
    }
    values.update(overrides)
    return TiledPointsDatasetReference(**values)


def _layer(reference: TiledPointsDatasetReference | None = None, **kwargs: object) -> TiledPointsLayerModel:
    reference = _dataset_reference() if reference is None else reference
    return TiledPointsLayerModel(
        reference,
        value_palette=np.full((reference.value_count, 4), 255, dtype=np.uint8),
        max_gpu_tile_bytes=1_000_000,
        **kwargs,
    )


def test_tiled_points_layer_keeps_logical_data_and_complete_extent() -> None:
    reference = _dataset_reference()
    layer = _layer(reference, scale=(2.0, 3.0), translate=(5.0, 7.0))

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
    layer = _layer()

    viewer.layers.append(layer)
    assert viewer.layers.selection.active is layer
    layer.visible = False
    layer.visible = True
    layer.affine.translate = (11.0, 13.0)
    viewer.reset_view()
    assert np.isfinite(viewer.camera.zoom)
    viewer.layers.remove(layer)
    assert len(viewer.layers) == 0


def test_tiled_points_layer_data_reference_is_immutable() -> None:
    reference = _dataset_reference()
    layer = _layer(reference)
    observed: list[TiledPointsDatasetReference] = []
    set_data_count = 0

    def _record_set_data(event: object) -> None:
        nonlocal set_data_count
        del event
        set_data_count += 1

    layer.events.data.connect(lambda event: observed.append(event.value))
    layer.events.set_data.connect(_record_set_data)
    replacement = _dataset_reference()

    layer.data = reference
    with pytest.raises(ValueError, match="cannot be replaced; construct a new layer and cache runtime"):
        layer.data = replacement

    assert layer.data is reference
    assert observed == []
    assert set_data_count == 0


def test_tiled_points_layer_data_setter_rejects_other_types() -> None:
    layer = _layer()

    with pytest.raises(ValueError, match="TiledPointsDatasetReference"):
        layer.data = np.empty((0, 2), dtype=np.float32)  # type: ignore[assignment]


def test_tiled_points_layer_exposes_style_and_status_events() -> None:
    layer = _layer()
    diameters: list[float] = []
    palettes: list[np.ndarray] = []
    statuses: list[TiledPointsLayerStatus] = []
    layer.events.point_diameter.connect(lambda event: diameters.append(event.value))
    layer.events.value_palette.connect(lambda event: palettes.append(event.value))
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
    palette = np.arange(12, dtype=np.uint8).reshape(3, 4)
    layer.value_palette = palette
    palette[:] = 0
    layer.display_status = status

    assert diameters == [4.5]
    np.testing.assert_array_equal(layer.value_palette, np.arange(12, dtype=np.uint8).reshape(3, 4))
    assert len(palettes) == 1
    np.testing.assert_array_equal(palettes[0], layer.value_palette)
    assert layer.value_palette.flags.owndata
    assert not layer.value_palette.flags.writeable
    assert layer.max_gpu_tile_bytes == 1_000_000
    assert statuses == [status]


@pytest.mark.parametrize(
    ("palette", "gpu_bytes", "match"),
    [
        (np.zeros((2, 4), dtype=np.uint8), 100, "value_palette"),
        (np.zeros((3, 3), dtype=np.uint8), 100, "value_palette"),
        (np.zeros((3, 4), dtype=np.float32), 100, "value_palette"),
        (np.zeros((3, 4), dtype=np.uint8), 0, "max_gpu_tile_bytes"),
        (np.zeros((3, 4), dtype=np.uint8), True, "max_gpu_tile_bytes"),
    ],
)
def test_tiled_points_layer_rejects_invalid_renderer_contracts(
    palette: np.ndarray,
    gpu_bytes: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TiledPointsLayerModel(
            _dataset_reference(),
            value_palette=palette,
            max_gpu_tile_bytes=gpu_bytes,  # type: ignore[arg-type]
        )


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
    layer = _layer()

    with pytest.raises(NotImplementedError, match="logical cache-backed layer"):
        layer.as_layer_data_tuple()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cache_generation_id": "generation"}, "UUID"),
        ({"points_name": ""}, "points_name"),
        ({"value_count": 0}, "value_count"),
        ({"x_origin": 4.0}, "origins"),
        ({"x_min": np.nan}, "x_min"),
        ({"x_min": 3.0, "x_max": 2.0}, "minima"),
    ],
)
def test_dataset_reference_rejects_invalid_identity_or_bounds(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _dataset_reference(**overrides)
