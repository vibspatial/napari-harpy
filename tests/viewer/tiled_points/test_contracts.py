from __future__ import annotations

import numpy as np
import pytest

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TiledPointsViewportState,
    TileResidencyKey,
    _ViewportRequest,
)

_GENERATION_ID = "12345678-1234-5678-9234-567812345678"


def _key(
    *,
    level: int = 0,
    tile_x: int = 0,
    requested_value_ids: tuple[int, ...] | None = (1,),
) -> TileResidencyKey:
    return TileResidencyKey(_GENERATION_ID, requested_value_ids, level, tile_x, 0)


def _tile(
    *,
    level: int = 0,
    tile_x: int = 0,
    point_count: int = 2,
    requested_value_ids: tuple[int, ...] | None = (1,),
) -> TiledPointsRenderTile:
    return TiledPointsRenderTile(
        key=_key(level=level, tile_x=tile_x, requested_value_ids=requested_value_ids),
        tile_size=10,
        location=np.arange(point_count * 2, dtype=np.float32).reshape(point_count, 2).copy(),
        value_id=np.ones(point_count, dtype=np.uint32),
    )


def test_render_tile_owns_read_only_aligned_payload_views() -> None:
    tile = _tile()

    assert tile.point_count == 2
    assert tile.resident_bytes == 24
    assert tile.key.logical_tile_key == (0, 0, 0)
    assert not tile.location.flags.writeable
    assert not tile.value_id.flags.writeable


def test_render_tile_rejects_nonowning_payload_views() -> None:
    location_batch = np.zeros((4, 2), dtype=np.float32)
    value_id_batch = np.zeros(4, dtype=np.uint32)

    with pytest.raises(ValueError, match="`location` must own its backing allocation"):
        TiledPointsRenderTile(
            key=_key(),
            tile_size=10,
            location=location_batch[:2, :],
            value_id=np.zeros(2, dtype=np.uint32),
        )

    with pytest.raises(ValueError, match="`value_id` must own its backing allocation"):
        TiledPointsRenderTile(
            key=_key(),
            tile_size=10,
            location=np.zeros((2, 2), dtype=np.float32),
            value_id=value_id_batch[:2],
        )


def test_render_snapshot_reconciles_complete_active_payload() -> None:
    tiles = (_tile(tile_x=0), _tile(tile_x=1, point_count=1))
    snapshot = TiledPointsRenderSnapshot(
        cache_generation_id=_GENERATION_ID,
        request_generation=4,
        selection_generation=2,
        requested_value_ids=(1,),
        level=0,
        level_kind="exact",
        within_budget=True,
        estimated_point_count=3,
        omitted_value_ids=(),
        tiles=tiles,
    )

    assert snapshot.rendered_tile_count == 2
    assert snapshot.rendered_point_count == 3

    with pytest.raises(ValueError, match="reconcile"):
        TiledPointsRenderSnapshot(
            cache_generation_id=_GENERATION_ID,
            request_generation=4,
            selection_generation=2,
            requested_value_ids=(1,),
            level=0,
            level_kind="exact",
            within_budget=True,
            estimated_point_count=4,
            omitted_value_ids=(),
            tiles=tiles,
        )


def test_render_snapshot_identifies_complete_sampled_omission() -> None:
    omitted = TiledPointsRenderSnapshot(
        cache_generation_id=_GENERATION_ID,
        request_generation=4,
        selection_generation=2,
        requested_value_ids=(1,),
        level=1,
        level_kind="bridge",
        within_budget=True,
        estimated_point_count=0,
        omitted_value_ids=(1,),
        tiles=(),
    )
    partial = TiledPointsRenderSnapshot(
        cache_generation_id=_GENERATION_ID,
        request_generation=4,
        selection_generation=2,
        requested_value_ids=(1, 2),
        level=1,
        level_kind="bridge",
        within_budget=True,
        estimated_point_count=1,
        omitted_value_ids=(2,),
        tiles=(_tile(level=1, point_count=1, requested_value_ids=(1, 2)),),
    )

    assert omitted.all_exact_present_values_omitted
    assert not partial.all_exact_present_values_omitted


def test_render_snapshot_rejects_omissions_outside_selected_values() -> None:
    common = {
        "cache_generation_id": _GENERATION_ID,
        "request_generation": 4,
        "selection_generation": 2,
        "level": 1,
        "level_kind": "bridge",
        "within_budget": True,
        "estimated_point_count": 0,
        "tiles": (),
    }

    with pytest.raises(ValueError, match="all-values snapshot"):
        TiledPointsRenderSnapshot(
            requested_value_ids=None,
            omitted_value_ids=(1,),
            **common,
        )

    with pytest.raises(ValueError, match="subset"):
        TiledPointsRenderSnapshot(
            requested_value_ids=(1,),
            omitted_value_ids=(2,),
            **common,
        )


@pytest.mark.parametrize(
    "result",
    [
        TiledPointsRenderResult(4, 2, True),
        TiledPointsRenderResult(4, 2, False),
    ],
)
def test_render_result_preserves_generation_bound_applied_state(result: TiledPointsRenderResult) -> None:
    assert result.request_generation == 4
    assert result.selection_generation == 2
    assert isinstance(result.applied, bool)


@pytest.mark.parametrize(
    "args",
    [
        (0, 2, True),
        (4, -1, True),
        (4, 2, 1),
    ],
)
def test_render_result_rejects_invalid_identity_or_applied_state(args: tuple[object, object, object]) -> None:
    with pytest.raises(ValueError):
        TiledPointsRenderResult(*args)  # type: ignore[arg-type]


def test_viewport_request_preserves_gui_generations_and_budget() -> None:
    viewport = TiledPointsViewportState(
        displayed_axes=(0, 1),
        x_min=0.0,
        y_min=1.0,
        x_max=10.0,
        y_max=11.0,
        canvas_width=100,
        canvas_height=80,
        hard_render_point_budget=1_000,
        screen_density_budget=500,
    )

    request = _ViewportRequest(3, 2, (1,), viewport)

    assert request.request_generation == 3
    assert request.selection_generation == 2
    assert request.viewport.effective_point_budget == 500


@pytest.mark.parametrize("requested_value_ids", [(), (2, 1), (1, 1), (True,)])
def test_residency_key_rejects_invalid_value_selection(requested_value_ids: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="requested_value_ids"):
        _key(requested_value_ids=requested_value_ids)
