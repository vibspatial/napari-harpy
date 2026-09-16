from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest
from napari._vispy.utils.qt_font import FontInfo

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.render_batch import pack_render_tiles
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer


def _layer(
    *,
    generation: str | None = None,
    gpu_bytes: int = 1_000_000,
    hard_render_point_budget: int = 100_000,
) -> TiledPointsLayerModel:
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
        max_vertex_payload_bytes=gpu_bytes,
        hard_render_point_budget=hard_render_point_budget,
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
    point_count = sum(tile.point_count for tile in tiles)
    render_batch = pack_render_tiles(
        tiles,
        point_count=point_count,
        value_count=layer.data.value_count,
        max_vertex_payload_bytes=1_000_000,
    )
    return TiledPointsRenderSnapshot(
        cache_generation_id=layer.data.cache_generation_id,
        request_generation=generation,
        selection_generation=0,
        requested_value_ids=None,
        level=0,
        level_kind="exact",
        within_budget=within_budget,
        estimated_point_count=point_count if within_budget else 100,
        omitted_value_ids=(),
        rendered_tile_count=len(tiles),
        render_batch=render_batch,
        budget_message=None if within_budget else "View exceeds hard rendering limits: 100 points required, limit 1",
    )


@pytest.fixture
def maximum_texture_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("napari._vispy.layers.base.get_max_texture_sizes", lambda: (8192, 2048))


def test_renderer_keeps_one_visual_and_vbo_across_full_snapshot_replacements(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(gpu_bytes=48)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    snapshot_visual = visual._snapshot_visual
    vertex_buffer = snapshot_visual.vertex_buffer
    set_data = vertex_buffer.set_data
    staged_point_counts: list[int] = []

    def _record_set_data(vertices: np.ndarray, *, copy: bool) -> None:
        staged_point_counts.append(len(vertices))
        set_data(vertices, copy=copy)

    monkeypatch.setattr(vertex_buffer, "set_data", _record_set_data)
    first, second, third = (_tile(layer, tile_x) for tile_x in range(3))
    try:
        assert visual.visual_count == 1
        assert visual.vbo_count == 1
        assert visual.payload_replacement_count == 0

        assert visual.apply_snapshot(_snapshot(layer, (first, second), generation=1))
        assert visual.active_point_count == 2
        assert visual.active_vertex_bytes == 24
        assert visual.point_draw_submission_count == 1
        assert visual.payload_replacement_count == 1
        assert staged_point_counts == [2]

        assert visual.apply_snapshot(_snapshot(layer, (second, third), generation=2))
        assert visual.active_point_count == 2
        assert visual.active_vertex_bytes == 24
        assert visual.payload_replacement_count == 2
        assert staged_point_counts == [2, 2]
        assert visual._snapshot_visual is snapshot_visual
        assert visual._snapshot_visual.vertex_buffer is vertex_buffer

        layer.point_diameter = 7.0
        layer.opacity = 0.25
        replacement = layer.value_palette.copy()
        replacement[[0, 1]] = replacement[[1, 0]]
        layer.value_palette = replacement

        assert visual.payload_replacement_count == 2
        assert staged_point_counts == [2, 2]
        assert visual._snapshot_visual.vertex_buffer is vertex_buffer
        assert visual.palette_update_count == 1
        assert visual.palette_gpu_bytes == 12
    finally:
        visual.close()


def test_generic_layer_refresh_preserves_active_snapshot(
    maximum_texture_size: None,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    tile = _tile(layer, 0)
    try:
        assert visual.apply_snapshot(_snapshot(layer, (tile,), generation=1))
        vertex_buffer = visual._snapshot_visual.vertex_buffer

        layer.refresh()

        assert visual.active_point_count == tile.point_count
        assert visual.payload_replacement_count == 1
        assert visual._snapshot_visual.vertex_buffer is vertex_buffer
    finally:
        visual.close()


def test_render_snapshot_event_emits_generation_bound_application_result(
    maximum_texture_size: None,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    results: list[TiledPointsRenderResult] = []
    layer.events.render_snapshot_result.connect(lambda event: results.append(event.value))
    snapshot = _snapshot(layer, (_tile(layer, 0),), generation=3)
    try:
        layer.events.render_snapshot(value=snapshot)

        assert results == [TiledPointsRenderResult(3, 0, True)]
    finally:
        visual.close()


def test_renderer_precomposes_large_cache_origin_and_affine_without_reupload(
    maximum_texture_size: None,
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
        max_vertex_payload_bytes=1_000_000,
        scale=(1.3, 0.7),
        translate=(11.0, -17.0),
        rotate=31.0,
        shear=(0.2,),
    )
    visual = VispyTiledPointsLayer(layer, FontInfo())
    tile = _tile(layer, 2)
    try:
        assert visual.apply_snapshot(_snapshot(layer, (tile,), generation=1))
        assert visual.payload_replacement_count == 1

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

        layer.translate = (23.0, -29.0)
        remapped_xy = np.asarray(visual.node.transform.map((relative_x, relative_y, 0.0, 1.0)))[:2]
        remapped_yx = layer.data_to_world(
            (
                layer.data.y_origin + relative_y,
                layer.data.x_origin + relative_x,
            )
        )
        assert np.allclose(remapped_xy, np.asarray(remapped_yx)[::-1])

        layer.opacity = 0.25
        layer.visible = False
        assert visual.node.opacity == 0.25
        assert not visual.node.visible
        assert visual.payload_replacement_count == 1
    finally:
        visual.close()


def test_renderer_capacity_failure_preserves_active_snapshot(
    maximum_texture_size: None,
) -> None:
    layer = _layer(gpu_bytes=12)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first, second, third = (_tile(layer, tile_x) for tile_x in range(3))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (first,), generation=1))
        assert not visual.apply_snapshot(_snapshot(layer, (second, third), generation=2))

        assert visual.active_point_count == 1
        assert visual.active_vertex_bytes == 12
        assert visual.payload_replacement_count == 1
        assert len(errors) == 1
        assert "max_vertex_payload_bytes=12" in str(errors[0])
    finally:
        visual.close()


def test_renderer_hard_point_budget_failure_does_not_stage(
    maximum_texture_size: None,
) -> None:
    layer = _layer(hard_render_point_budget=1)
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first, second = (_tile(layer, tile_x) for tile_x in range(2))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        assert not visual.apply_snapshot(_snapshot(layer, (first, second), generation=1))

        assert visual.active_point_count == 0
        assert visual.payload_replacement_count == 0
        assert len(errors) == 1
        assert "hard_render_point_budget=1" in str(errors[0])
    finally:
        visual.close()


@pytest.mark.parametrize("reuse_active_batch", [False, True], ids=["new-batch", "active-batch"])
def test_renderer_revalidates_mutated_batch_before_accepting_snapshot(
    maximum_texture_size: None,
    reuse_active_batch: bool,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first, second = (_tile(layer, tile_x) for tile_x in range(2))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        active = _snapshot(layer, (first,), generation=1)
        assert visual.apply_snapshot(active)
        candidate = (
            replace(active, request_generation=2) if reuse_active_batch else _snapshot(layer, (second,), generation=2)
        )
        candidate.render_batch.vertices.flags.writeable = True
        assert not visual.apply_snapshot(candidate)

        assert visual.active_point_count == first.point_count
        assert visual.payload_replacement_count == 1
        assert len(errors) == 1
        assert "canonical vertex-payload contract" in str(errors[0])
    finally:
        visual.close()


def test_renderer_upload_failure_does_not_count_candidate_replacement(
    maximum_texture_size: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first, second = (_tile(layer, tile_x) for tile_x in range(2))
    errors: list[Exception] = []
    layer.events.render_error.connect(lambda event: errors.append(event.value))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (first,), generation=1))

        def _fail_upload(_vertices: np.ndarray, *, copy: bool) -> None:
            del copy
            raise RuntimeError("synthetic upload failure")

        monkeypatch.setattr(visual._snapshot_visual.vertex_buffer, "set_data", _fail_upload)
        assert not visual.apply_snapshot(_snapshot(layer, (second,), generation=2))

        assert visual.active_point_count == first.point_count
        assert visual.payload_replacement_count == 1
        assert len(errors) == 1
        assert str(errors[0]) == "synthetic upload failure"
    finally:
        visual.close()


def test_identical_batch_acknowledges_snapshot_without_staging_or_redraw(
    maximum_texture_size: None,
) -> None:
    """Reused vertices need no redraw request, but fresh request metadata is accepted."""
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    snapshot = _snapshot(layer, (_tile(layer, 0, point_count=2),), generation=1)
    results: list[TiledPointsRenderResult] = []
    layer.events.render_snapshot_result.connect(lambda event: results.append(event.value))
    redraw_requests: list[object] = []
    # Observe scene-update notifications, independently of the VBO staging count.
    visual.node.events.update.connect(lambda event: redraw_requests.append(event))
    try:
        layer.events.render_snapshot(value=snapshot)
        assert len(redraw_requests) == 1
        redraw_requests.clear()
        results.clear()

        layer.events.render_snapshot(value=replace(snapshot, request_generation=2, estimated_point_count=1))

        assert results == [TiledPointsRenderResult(2, 0, True)]
        assert visual.payload_replacement_count == 1
        assert visual.active_point_count == 2
        assert redraw_requests == []
    finally:
        visual.close()


@pytest.mark.parametrize("point_count", [1, 0], ids=["nonempty", "empty"])
def test_different_batch_updates_visual_and_requests_redraw(
    maximum_texture_size: None,
    point_count: int,
) -> None:
    """Replacing point data, including clearing it, still requests a frame."""
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    redraw_requests: list[object] = []
    visual.node.events.update.connect(lambda event: redraw_requests.append(event))
    try:
        assert visual.apply_snapshot(_snapshot(layer, (_tile(layer, 0, point_count=2),), generation=1))
        redraw_requests.clear()
        tiles = (_tile(layer, 1),) if point_count else ()

        assert visual.apply_snapshot(_snapshot(layer, tiles, generation=2))

        assert visual.active_point_count == point_count
        assert visual.payload_replacement_count == (2 if point_count else 1)
        assert len(redraw_requests) == 1
    finally:
        visual.close()


def test_renderer_reuses_identical_batch_but_reuploads_after_partial_staging_failure(
    maximum_texture_size: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restage batch A after staging batch B into the shared VBO fails partway through.

    First verify that reusing batch A skips redundant staging. Then simulate
    B failing during binding, after set_data() has already run.

    ``VispyTiledPointsLayer.apply_snapshot()`` must clear ``_active_render_batch``
    before attempting to stage B: the buffer can no longer be assumed to contain A.
    Returning to the same CPU batch A must therefore stage its vertices again,
    not take the reuse shortcut.
    """
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    first = _snapshot(layer, (_tile(layer, 0, point_count=2),), generation=1)
    second = _snapshot(layer, (_tile(layer, 1),), generation=3)
    try:
        assert visual.apply_snapshot(first)
        assert visual.apply_snapshot(replace(first, request_generation=2, estimated_point_count=1))
        assert visual.payload_replacement_count == 1
        program = visual._snapshot_visual.shared_program
        original_bind = program.bind

        def fail_after_set_data(_buffer) -> None:
            raise RuntimeError("bind failed after buffer mutation")

        monkeypatch.setattr(program, "bind", fail_after_set_data)
        assert not visual.apply_snapshot(second)
        monkeypatch.setattr(program, "bind", original_bind)
        assert visual.apply_snapshot(replace(first, request_generation=4))
        assert visual.payload_replacement_count == 2
        assert visual.active_point_count == 2
        assert visual.visual_count == visual.vbo_count == 1
        # Returning to the same batch remains subject to the current full-payload
        # limits, even when the latest visible estimate is smaller.
        layer.hard_render_point_budget = 1
        assert not visual.apply_snapshot(replace(first, request_generation=5, estimated_point_count=1))
        assert visual.payload_replacement_count == 2
    finally:
        visual.close()
    assert visual._active_render_batch is None


def test_empty_snapshot_suppresses_one_visual_without_replacing_its_vbo(
    maximum_texture_size: None,
) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    tile = _tile(layer, 0)
    vertex_buffer = visual._snapshot_visual.vertex_buffer
    try:
        assert visual.apply_snapshot(_snapshot(layer, (tile,), generation=1))
        assert not visual.apply_snapshot(_snapshot(layer, (), generation=2, within_budget=False))
        assert visual.active_point_count == tile.point_count
        assert visual.payload_replacement_count == 1

        assert visual.apply_snapshot(_snapshot(layer, (), generation=3))
        assert visual.active_point_count == 0
        assert visual.active_vertex_bytes == 0
        assert visual.point_draw_submission_count == 0
        assert visual.visual_count == 1
        assert visual.vbo_count == 1
        assert visual._snapshot_visual.vertex_buffer is vertex_buffer
        assert visual.payload_replacement_count == 1
    finally:
        visual.close()


def test_renderer_close_releases_fixed_resources_and_ignores_late_snapshot(maximum_texture_size: None) -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    snapshot = _snapshot(layer, (_tile(layer, 0),), generation=1)
    assert visual.apply_snapshot(snapshot)

    visual.close()
    visual.close()

    assert visual.visual_count == 0
    assert visual.vbo_count == 0
    assert visual.active_point_count == 0
    assert not visual.apply_snapshot(snapshot)
