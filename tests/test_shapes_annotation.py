from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from napari.layers import Shapes
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.transformations import Identity, get_transformation

import napari_harpy.core.shapes_annotation as shapes_annotation_module
from napari_harpy.core.shapes_annotation import (
    CreateShapesElementRequest,
    create_shapes_element_from_napari_shapes_layer,
    napari_shapes_layer_to_geodataframe,
)


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
    assert "instance_id" not in layer.features.columns


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


def test_napari_shapes_layer_to_geodataframe_assigns_new_ids_for_copied_generated_instance_ids() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    layer.features["instance_id"] = ["shape_0", "shape_0"]

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["shape_0", "shape_1"]
    assert layer.features["instance_id"].tolist() == ["shape_0", "shape_1"]


def test_napari_shapes_layer_to_geodataframe_rejects_duplicate_custom_instance_ids() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    layer.features["instance_id"] = ["custom", "custom"]

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


def test_create_shapes_element_from_napari_shapes_layer_writes_shapes_element(sdata_blobs: SpatialData) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)

    result = create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
        ),
        layer,
    )

    assert result.shapes_name == "new_regions"
    assert result.coordinate_system == "global"
    assert result.row_count == 1
    assert "new_regions" in sdata_blobs.shapes
    shapes = sdata_blobs.shapes["new_regions"]
    assert shapes.index.name == "instance_id"
    assert shapes.index.tolist() == ["shape_0"]
    assert shapes.geometry.iloc[0].equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert isinstance(get_transformation(shapes, get_all=True)["global"], Identity)


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"sdata": None}, "SpatialData object"),
        ({"shapes_name": "   "}, "must not be empty"),
        ({"coordinate_system": "   "}, "non-empty string"),
        ({"overwrite": "yes"}, "boolean"),
        ({"index_name": "   "}, "must not be empty"),
        ({"index_prefix": "   "}, "non-empty string"),
    ],
)
def test_create_shapes_element_from_napari_shapes_layer_rejects_invalid_request_before_layer_mutation(
    sdata_blobs: SpatialData,
    request_kwargs: dict[str, object],
    message: str,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    kwargs = {
        "sdata": sdata_blobs,
        "shapes_name": "new_regions",
        "coordinate_system": "global",
    } | request_kwargs

    with pytest.raises(ValueError, match=message):
        create_shapes_element_from_napari_shapes_layer(CreateShapesElementRequest(**kwargs), layer)

    assert "instance_id" not in layer.features.columns
    assert "new_regions" not in sdata_blobs.shapes


def test_create_shapes_element_from_napari_shapes_layer_rejects_unknown_coordinate_system_before_layer_mutation(
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)

    with pytest.raises(ValueError, match="not available"):
        create_shapes_element_from_napari_shapes_layer(
            CreateShapesElementRequest(
                sdata=sdata_blobs,
                shapes_name="new_regions",
                coordinate_system="missing",
            ),
            layer,
        )

    assert "instance_id" not in layer.features.columns
    assert "new_regions" not in sdata_blobs.shapes


def test_create_shapes_element_from_napari_shapes_layer_rejects_collision_without_overwrite_before_conversion(
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    existing_shapes_name = next(iter(sdata_blobs.shapes))

    with pytest.raises(ValueError, match="already exists"):
        create_shapes_element_from_napari_shapes_layer(
            CreateShapesElementRequest(
                sdata=sdata_blobs,
                shapes_name=existing_shapes_name,
                coordinate_system="global",
            ),
            layer,
        )

    assert "instance_id" not in layer.features.columns


def test_create_shapes_element_from_napari_shapes_layer_failed_conversion_leaves_sdata_unchanged(
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes([np.asarray([[0.0, 0.0], [1.0, 1.0]])], shape_type="line")
    initial_shapes_names = set(sdata_blobs.shapes)

    with pytest.raises(ValueError, match="Lines and paths cannot be saved"):
        create_shapes_element_from_napari_shapes_layer(
            CreateShapesElementRequest(
                sdata=sdata_blobs,
                shapes_name="new_regions",
                coordinate_system="global",
            ),
            layer,
        )

    assert set(sdata_blobs.shapes) == initial_shapes_names


def test_create_shapes_element_from_napari_shapes_layer_repeated_save_overwrites_locked_element(
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
        ),
        layer,
    )

    _add_polygon(layer, offset=3)
    result = create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            overwrite=True,
        ),
        layer,
    )

    assert result.row_count == 2
    assert sdata_blobs.shapes["new_regions"].index.tolist() == ["shape_0", "shape_1"]
    assert layer.features["instance_id"].tolist() == ["shape_0", "shape_1"]


def test_create_shapes_element_from_napari_shapes_layer_calls_harpy_with_request_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    captured_kwargs: dict[str, object] = {}

    def fake_add_shapes(sdata: SpatialData, **kwargs):
        captured_kwargs["sdata"] = sdata
        captured_kwargs.update(kwargs)
        return sdata

    monkeypatch.setattr(shapes_annotation_module.hp.sh, "add_shapes", fake_add_shapes)

    result = create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            overwrite=True,
        ),
        layer,
    )

    assert result.row_count == 1
    assert captured_kwargs["sdata"] is sdata_blobs
    assert captured_kwargs["output_shapes_name"] == "new_regions"
    assert captured_kwargs["instance_key"] == "instance_id"
    assert captured_kwargs["overwrite"] is True
    assert isinstance(captured_kwargs["input"], gpd.GeoDataFrame)
    transformations = captured_kwargs["transformations"]
    assert isinstance(transformations, dict)
    assert isinstance(transformations["global"], Identity)
