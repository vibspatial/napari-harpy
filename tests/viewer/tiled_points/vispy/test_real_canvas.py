"""Opt-in real-OpenGL qualification for the tiled-points renderer."""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest
from napari._vispy.utils.qt_font import FontInfo
from vispy.scene import SceneCanvas, visuals
from vispy.visuals.transforms import MatrixTransform

from napari_harpy.viewer.tiled_points import (
    TiledPointsDatasetReference,
    TiledPointsLayerModel,
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.vispy.layer import VispyTiledPointsLayer

pytestmark = pytest.mark.skipif(
    os.environ.get("NAPARI_HARPY_RUN_REAL_GL_TESTS") != "1",
    reason="Set NAPARI_HARPY_RUN_REAL_GL_TESTS=1 to run real-OpenGL qualification.",
)


def _layer() -> TiledPointsLayerModel:
    generation = str(uuid4())
    return TiledPointsLayerModel(
        TiledPointsDatasetReference(
            cache_generation_id=generation,
            points_name="spots",
            value_column="feature_name",
            value_count=2,
            x_origin=100_000_000.0,
            y_origin=200_000_000.0,
            x_min=100_000_000.0,
            x_max=100_000_040.0,
            y_min=200_000_000.0,
            y_max=200_000_040.0,
        ),
        value_palette=np.array(((255, 0, 0, 255), (0, 255, 0, 255)), dtype=np.uint8),
        max_gpu_tile_bytes=1_000_000,
        opacity=1.0,
        point_diameter=7.0,
        scale=(1.3, 0.7),
        translate=(11.0, -17.0),
        rotate=31.0,
        shear=(0.2,),
    )


def _apply_two_point_snapshot(
    layer: TiledPointsLayerModel,
    visual: VispyTiledPointsLayer,
) -> np.ndarray:
    location = np.array(((1.25, 2.75), (8.5, 7.0)), dtype=np.float32)
    value_id = np.array((0, 1), dtype=np.uint32)
    tile = TiledPointsRenderTile(
        key=TileResidencyKey(
            cache_generation_id=layer.data.cache_generation_id,
            requested_value_ids=None,
            level=0,
            tile_x=2,
            tile_y=1,
        ),
        tile_size=10,
        location=location,
        value_id=value_id,
    )
    snapshot = TiledPointsRenderSnapshot(
        cache_generation_id=layer.data.cache_generation_id,
        request_generation=1,
        selection_generation=0,
        requested_value_ids=None,
        level=0,
        level_kind="exact",
        within_budget=True,
        estimated_point_count=2,
        omitted_value_ids=(),
        tiles=(tile,),
    )
    assert visual.apply_snapshot(snapshot)
    return location.astype(np.float64) + np.array((20.0, 10.0))


def _render(node: object, *, rect: tuple[float, float, float, float]) -> np.ndarray:
    canvas = SceneCanvas(show=False, size=(320, 240))
    view = canvas.central_widget.add_view()
    view.camera = "panzoom"
    view.camera.rect = rect
    view.add(node)
    try:
        return canvas.render()
    finally:
        canvas.close()


def _colour_centroids(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    for channel, other in ((0, 1), (1, 0)):
        mask = (image[..., channel] > 150) & (image[..., other] < 100)
        rows, columns = np.nonzero(mask)
        assert len(rows) > 0
        centroids.append(np.array((columns.mean(), rows.mean())))
    return centroids[0], centroids[1]


def test_compact_visual_matches_marker_reference_under_large_origin_and_affine() -> None:
    layer = _layer()
    visual = VispyTiledPointsLayer(layer, FontInfo())
    try:
        cache_relative = _apply_two_point_snapshot(layer, visual)
        mapped = np.asarray(visual.node.transform.map(np.column_stack((cache_relative, np.zeros(2), np.ones(2)))))[
            :, :2
        ]
        lower = mapped.min(axis=0) - 12.0
        upper = mapped.max(axis=0) + 12.0
        rect = (
            float(lower[0]),
            float(lower[1]),
            float(upper[0] - lower[0]),
            float(upper[1] - lower[1]),
        )
        compact = _render(visual.node, rect=rect)

        reference = visuals.Markers()
        reference.set_data(
            cache_relative.astype(np.float32),
            face_color=layer.value_palette.astype(np.float32) / 255.0,
            edge_width=0,
            size=layer.point_diameter,
        )
        transform = MatrixTransform()
        transform.matrix = visual.node.transform.matrix.copy()
        reference.transform = transform
        markers = _render(reference, rect=rect)

        for observed, expected in zip(
            _colour_centroids(compact),
            _colour_centroids(markers),
            strict=True,
        ):
            assert np.linalg.norm(observed - expected) < 1.0
        assert compact.shape == markers.shape
    finally:
        visual.close()
