from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from napari._vispy.utils.qt_font import FontInfo

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer


def _layer(*, generation: str | None = None, gpu_bytes: int = 1_000_000) -> TiledPointsLayerModel:
    return TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=str(uuid4()) if generation is None else generation,
            points_name="spots",
            value_column="feature_name",
            value_count=3,
            x_origin=100.0,
            y_origin=200.0,
            x_min=103.0,
            x_max=143.0,
            y_min=202.0,
            y_max=232.0,
        ),
        value_palette=np.array(
            (
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
            ),
            dtype=np.uint8,
        ),
        max_gpu_tile_bytes=gpu_bytes,
    )


def _tile(
    layer: TiledPointsLayerModel,
    tile_x: int,
    *,
    point_count: int = 1,
    requested_value_ids: tuple[int, ...] | None = None,
) -> TiledPointsRenderTile:
    return TiledPointsRenderTile(
        key=TileResidencyKey(
            cache_generation_id=layer.data.cache_generation_id,
            requested_value_ids=requested_value_ids,
            level=0,
            tile_x=tile_x,
            tile_y=0,
        ),
        tile_size=10,
        location=np.column_stack(
            (
                np.arange(point_count, dtype=np.float32) + 1,
                np.arange(point_count, dtype=np.float32) + 2,
            )
        ),
        value_id=np.arange(point_count, dtype=np.uint32) % layer.data.value_count,
    )


def _snapshot(
    layer: TiledPointsLayerModel,
    tiles: tuple[TiledPointsRenderTile, ...],
    *,
    generation: int,
    within_budget: bool = True,
) -> TiledPointsRenderSnapshot:
    return TiledPointsRenderSnapshot(
        cache_generation_id=layer.data.cache_generation_id,
        request_generation=generation,
        selection_generation=0,
        requested_value_ids=None,
        level=0,
        level_kind="exact",
        within_budget=within_budget,
        estimated_point_count=sum(tile.point_count for tile in tiles) if within_budget else 100,
        omitted_value_ids=(),
        tiles=tiles,
    )


def _track_tile_resource_creation(
    visual: VispyTiledPointsLayer,
    monkeypatch: pytest.MonkeyPatch,
) -> list[TileResidencyKey]:
    """Record test-local tile uploads without retaining history in the renderer."""
    created_keys: list[TileResidencyKey] = []
    create_tile_resource = visual._create_tile_resource

    def _record(tile: TiledPointsRenderTile):
        resource = create_tile_resource(tile)
        created_keys.append(tile.key)
        return resource

    monkeypatch.setattr(visual, "_create_tile_resource", _record)
    return created_keys


@pytest.fixture
def maximum_texture_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("napari._vispy.layers.base.get_max_texture_sizes", lambda: (8192, 2048))


def test_renderer_reuses_overlapping_tiles_and_style_changes_do_not_upload(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(gpu_bytes=48)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    created_keys = _track_tile_resource_creation(visual, monkeypatch)
    first, second, third = (_tile(layer, tile_x) for tile_x in range(3))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (first, second), generation=1))
        assert created_keys == [first.key, second.key]
        assert visual.resident_gpu_tile_bytes == 24

        assert visual.apply_snapshot(_snapshot(layer, (second, third), generation=2))
        assert visual.active_keys == (second.key, third.key)
        assert created_keys == [first.key, second.key, third.key]
        assert created_keys.count(second.key) == 1
        assert visual.resident_gpu_tile_bytes == 36

        layer.point_diameter = 7.0
        replacement = layer.value_palette.copy()
        replacement[[0, 1]] = replacement[[1, 0]]
        layer.value_palette = replacement

        assert created_keys == [first.key, second.key, third.key]
        assert visual.palette_update_count == 1
        assert visual.palette_gpu_bytes == 12
    finally:
        visual.close()


def test_renderer_precomposes_large_cache_origin_and_affine_without_reupload(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=str(uuid4()),
            points_name="spots",
            value_column="feature_name",
            value_count=3,
            x_origin=100_000_000.0,
            y_origin=200_000_000.0,
            x_min=100_000_003.0,
            x_max=100_000_043.0,
            y_min=200_000_002.0,
            y_max=200_000_032.0,
        ),
        value_palette=np.array(
            ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)),
            dtype=np.uint8,
        ),
        max_gpu_tile_bytes=1_000_000,
        scale=(1.3, 0.7),
        translate=(11.0, -17.0),
        rotate=31.0,
        shear=(0.2,),
    )
    visual = VispyTiledPointsLayer(layer, FontInfo())
    created_keys = _track_tile_resource_creation(visual, monkeypatch)
    tile = _tile(layer, 2)
    try:
        assert visual.apply_snapshot(_snapshot(layer, (tile,), generation=1))
        assert created_keys == [tile.key]

        relative_x = tile.key.tile_x * tile.tile_size + float(tile.location[0, 0])
        relative_y = tile.key.tile_y * tile.tile_size + float(tile.location[0, 1])
        observed_xy = np.asarray(visual.node.transform.map((relative_x, relative_y, 0.0, 1.0)))[:2]
        expected_yx = layer.data_to_world(
            (
                layer.data.y_origin + relative_y,
                layer.data.x_origin + relative_x,
            )
        )

        assert np.allclose(observed_xy, np.asarray(expected_yx)[::-1])
        assert created_keys == [tile.key]

        layer.translate = (23.0, -29.0)
        remapped_xy = np.asarray(visual.node.transform.map((relative_x, relative_y, 0.0, 1.0)))[:2]
        remapped_yx = layer.data_to_world(
            (
                layer.data.y_origin + relative_y,
                layer.data.x_origin + relative_x,
            )
        )
        assert np.allclose(remapped_xy, np.asarray(remapped_yx)[::-1])
        assert created_keys == [tile.key]

        layer.opacity = 0.25
        layer.visible = False
        assert visual.node.opacity == 0.25
        assert not visual.node.visible
        assert created_keys == [tile.key]
    finally:
        visual.close()


def test_renderer_capacity_failure_preserves_active_snapshot(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(gpu_bytes=24)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    created_keys = _track_tile_resource_creation(visual, monkeypatch)
    first, second, third = (_tile(layer, tile_x) for tile_x in range(3))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (first, second), generation=1))
        assert not visual.apply_snapshot(_snapshot(layer, (second, third), generation=2))

        assert visual.active_keys == (first.key, second.key)
        assert created_keys == [first.key, second.key]
        assert visual.resident_gpu_tile_bytes == 24
        assert len(errors) == 1
        assert "max_gpu_tile_bytes=24" in str(errors[0])
        assert visual.pending_keys == ()
    finally:
        visual.close()


def test_renderer_upload_failure_rolls_back_pending_resources(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first, second, third = (_tile(layer, tile_x) for tile_x in range(3))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (first,), generation=1))
        create_resource = visual._create_tile_resource

        def _fail_on_third(tile: TiledPointsRenderTile):
            if tile.key == third.key:
                raise RuntimeError("synthetic upload failure")
            return create_resource(tile)

        monkeypatch.setattr(visual, "_create_tile_resource", _fail_on_third)
        assert not visual.apply_snapshot(_snapshot(layer, (second, third), generation=2))

        assert visual.active_keys == (first.key,)
        assert visual.resident_tile_count == 1
        assert visual.resident_gpu_tile_bytes == first.resident_bytes
        assert visual.pending_keys == ()
        assert len(errors) == 1
        assert str(errors[0]) == "synthetic upload failure"
    finally:
        visual.close()


def test_zero_tile_snapshot_clears_active_membership_but_over_budget_does_not(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    created_keys = _track_tile_resource_creation(visual, monkeypatch)
    tile = _tile(layer, 0)
    try:
        assert visual.apply_snapshot(_snapshot(layer, (tile,), generation=1))
        assert not visual.apply_snapshot(_snapshot(layer, (), generation=2, within_budget=False))
        assert visual.active_keys == (tile.key,)

        assert visual.apply_snapshot(_snapshot(layer, (), generation=3))
        assert visual.active_keys == ()
        assert visual.resident_tile_count == 1
        assert created_keys == [tile.key]
    finally:
        visual.close()


def test_renderer_close_releases_resources_and_ignores_late_snapshot(maximum_texture_size: None) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    snapshot = _snapshot(layer, (_tile(layer, 0),), generation=1)
    assert visual.apply_snapshot(snapshot)

    visual.close()
    visual.close()

    assert visual.resident_tile_count == 0
    assert visual.active_keys == ()
    assert not visual.apply_snapshot(snapshot)
