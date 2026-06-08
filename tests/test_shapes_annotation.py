from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from napari.layers import Shapes
from shapely.geometry import Polygon

from napari_harpy.core.shapes_annotation import napari_shapes_layer_to_geodataframe


def _polygon_data(offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [offset + 0.0, 0.0],
            [offset + 0.0, 2.0],
            [offset + 2.0, 2.0],
            [offset + 2.0, 0.0],
        ],
        dtype=float,
    )


def _add_polygon(layer: Shapes, offset: float = 0.0) -> None:
    layer.add_polygons(_polygon_data(offset))


def test_napari_shapes_layer_to_geodataframe_converts_polygon_coordinates() -> None:
    layer = Shapes([np.asarray([[1.0, 2.0], [1.0, 5.0], [4.0, 5.0], [4.0, 2.0]])], shape_type="polygon")

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert isinstance(geodataframe, gpd.GeoDataFrame)
    assert geodataframe.index.name == "instance_id"
    assert geodataframe.index.tolist() == ["shape_0"]
    assert geodataframe.geometry.iloc[0].equals(Polygon([(2, 1), (5, 1), (5, 4), (2, 4)]))
    assert layer.features["instance_id"].tolist() == ["shape_0"]


def test_napari_shapes_layer_to_geodataframe_converts_rectangles() -> None:
    layer = Shapes(ndim=2)
    layer.add_rectangles(np.asarray([[0.0, 1.0], [3.0, 5.0]]))

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["shape_0"]
    assert geodataframe.geometry.iloc[0].equals(Polygon([(1, 0), (1, 3), (5, 3), (5, 0)]))


def test_napari_shapes_layer_to_geodataframe_polygonizes_ellipses() -> None:
    layer = Shapes(ndim=2)
    layer.add_ellipses(np.asarray([[10.0, 20.0], [5.0, 8.0]]))

    geodataframe = napari_shapes_layer_to_geodataframe(layer, ellipse_segments=16)

    geometry = geodataframe.geometry.iloc[0]
    assert isinstance(geometry, Polygon)
    assert len(geometry.exterior.coords) == 17
    assert geometry.bounds == pytest.approx((12.0, 5.0, 28.0, 15.0))
    assert geometry.centroid.x == pytest.approx(20.0)
    assert geometry.centroid.y == pytest.approx(10.0)


def test_napari_shapes_layer_to_geodataframe_rejects_line_and_path_rows() -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [1.0, 1.0]])], shape_type="line")

    with pytest.raises(ValueError, match="Lines and paths cannot be saved"):
        napari_shapes_layer_to_geodataframe(layer)


def test_napari_shapes_layer_to_geodataframe_rejects_empty_layers() -> None:
    layer = Shapes(ndim=2)

    with pytest.raises(ValueError, match="Draw at least one supported shape"):
        napari_shapes_layer_to_geodataframe(layer)


def test_napari_shapes_layer_to_geodataframe_rejects_non_finite_coordinates() -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [0.0, np.nan], [1.0, 1.0]])], shape_type="polygon")

    with pytest.raises(ValueError, match="non-finite coordinates"):
        napari_shapes_layer_to_geodataframe(layer)


def test_napari_shapes_layer_to_geodataframe_rejects_too_few_polygon_vertices() -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [1.0, 1.0]])], shape_type="polygon")

    with pytest.raises(ValueError, match="too few vertices"):
        napari_shapes_layer_to_geodataframe(layer)


def test_napari_shapes_layer_to_geodataframe_preserves_existing_instance_ids_and_assigns_next_id() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    _add_polygon(layer, offset=6)
    layer.features["instance_id"] = ["shape_3", "custom", np.nan]

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["shape_3", "custom", "shape_4"]
    assert layer.features["instance_id"].tolist() == ["shape_3", "custom", "shape_4"]


def test_napari_shapes_layer_to_geodataframe_rejects_duplicate_instance_ids() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    layer.features["instance_id"] = ["shape_0", "shape_0"]

    with pytest.raises(ValueError, match="must be unique"):
        napari_shapes_layer_to_geodataframe(layer)


def test_napari_shapes_layer_to_geodataframe_rejects_invalid_ellipse_segments() -> None:
    layer = Shapes(ndim=2)
    layer.add_ellipses(np.asarray([[10.0, 20.0], [5.0, 8.0]]))

    with pytest.raises(ValueError, match="ellipse_segments"):
        napari_shapes_layer_to_geodataframe(layer, ellipse_segments=3)


def test_napari_shapes_layer_to_geodataframe_rejects_invalid_ellipse_bounding_box() -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [2.0, 2.0]])], shape_type="ellipse")

    with pytest.raises(ValueError, match="Ellipse row `0` cannot be converted"):
        napari_shapes_layer_to_geodataframe(layer)
