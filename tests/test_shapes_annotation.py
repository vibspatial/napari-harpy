from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from napari.layers import Shapes
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData, read_zarr, transform
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Translation, get_transformation

import napari_harpy.core.shapes_annotation as shapes_annotation_module
from napari_harpy.core.shapes_annotation import (
    CreateShapesElementRequest,
    EditShapesElementRequest,
    ExistingShapesLayerConversion,
    NewShapesLayerConversion,
    create_shapes_element_from_napari_shapes_layer,
    edit_shapes_element_from_napari_shapes_layer,
    napari_shapes_layer_to_geodataframe,
)
from napari_harpy.core.shapes_geometry import shapely_polygon_to_napari_polygon_vertices


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


def _source_polygon(offset: float = 0.0) -> Polygon:
    return Polygon(
        [
            (offset + 0.0, 0.0),
            (offset + 2.0, 0.0),
            (offset + 2.0, 2.0),
            (offset + 0.0, 2.0),
        ]
    )


def _make_shapes_sdata(
    geodataframe: gpd.GeoDataFrame,
    *,
    shapes_name: str = "regions",
    coordinate_system: str = "global",
    transformations: dict[str, object] | None = None,
) -> SpatialData:
    if transformations is None:
        transformations = {coordinate_system: Identity()}
    shapes = ShapesModel.parse(geodataframe, transformations=transformations)
    return SpatialData(shapes={shapes_name: shapes})


def test_napari_shapes_layer_to_geodataframe_converts_polygon_coordinates() -> None:
    layer = Shapes([np.asarray([[1.0, 2.0], [1.0, 5.0], [4.0, 5.0], [4.0, 2.0]])], shape_type="polygon")

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert isinstance(geodataframe, gpd.GeoDataFrame)
    assert geodataframe.index.name == "instance_id"
    assert geodataframe.index.tolist() == ["__annotation_0"]
    assert geodataframe.geometry.iloc[0].equals(Polygon([(2, 1), (5, 1), (5, 4), (2, 4)]))
    assert layer.features["instance_id"].tolist() == ["__annotation_0"]


def test_napari_shapes_layer_to_geodataframe_converts_rectangles() -> None:
    layer = Shapes(ndim=2)
    layer.add_rectangles(np.asarray([[0.0, 1.0], [3.0, 5.0]]))

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["__annotation_0"]
    assert geodataframe.geometry.iloc[0].equals(Polygon([(1, 0), (1, 3), (5, 3), (5, 0)]))


def test_napari_shapes_layer_to_geodataframe_preserves_polygon_hole() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(source)], shape_type="polygon")

    geodataframe = napari_shapes_layer_to_geodataframe(layer)
    geometry = geodataframe.geometry.iloc[0]

    assert geometry.equals(source)
    assert len(geometry.interiors) == 1
    assert geometry.area == source.area
    assert geometry.bounds == source.bounds


def test_napari_shapes_layer_to_geodataframe_preserves_multiple_polygon_holes() -> None:
    source = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2)],
            [(6, 6), (6, 8), (8, 8), (8, 6)],
        ],
    )
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(source)], shape_type="polygon")

    geodataframe = napari_shapes_layer_to_geodataframe(layer)
    geometry = geodataframe.geometry.iloc[0]

    assert geometry.equals(source)
    assert len(geometry.interiors) == 2
    assert geometry.area == source.area
    assert geometry.bounds == source.bounds


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


def test_napari_shapes_layer_to_geodataframe_rejects_invalid_hole_path_without_mutating_features() -> None:
    source = Polygon(
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        holes=[[(2, 2), (2, 4), (4, 4), (4, 2)]],
    )
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(source)[:-1]], shape_type="polygon")
    layer.features["instance_id"] = ["existing"]

    with pytest.raises(
        ValueError,
        match=r"Shape row `0` cannot be converted to a valid polygon: .*path with holes must end",
    ):
        napari_shapes_layer_to_geodataframe(layer)

    assert layer.features["instance_id"].tolist() == ["existing"]


def test_napari_shapes_layer_to_geodataframe_preserves_existing_instance_ids_and_assigns_next_id() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    _add_polygon(layer, offset=6)
    layer.features["instance_id"] = ["__annotation_3", "custom", np.nan]

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["__annotation_3", "custom", "__annotation_4"]
    assert layer.features["instance_id"].tolist() == ["__annotation_3", "custom", "__annotation_4"]


def test_napari_shapes_layer_to_geodataframe_assigns_new_ids_for_copied_generated_instance_ids() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    layer.features["instance_id"] = ["__annotation_0", "__annotation_0"]

    geodataframe = napari_shapes_layer_to_geodataframe(layer)

    assert geodataframe.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]


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


def test_napari_shapes_layer_to_geodataframe_accepts_new_conversion_context() -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        conversion=NewShapesLayerConversion(index_name="region_id", index_prefix="region"),
    )

    assert geodataframe.index.name == "region_id"
    assert geodataframe.index.tolist() == ["region_0"]
    assert layer.features["region_id"].tolist() == ["region_0"]


def test_napari_shapes_layer_to_geodataframe_edit_existing_preserves_metadata_and_new_row_dtypes() -> None:
    source = gpd.GeoDataFrame(
        {
            "class_id": [1, 2],
            "is_selected": [True, False],
            "category": pd.Categorical(["alpha", "beta"]),
            "label": ["first", "second"],
        },
        geometry=[_source_polygon(0), _source_polygon(3)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    assert str(source["class_id"].dtype) == "int64"
    assert str(source["is_selected"].dtype) == "bool"

    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=10)
    _add_polygon(layer, offset=20)
    layer.features["instance_id"] = ["cell_2", np.nan]

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        conversion=ExistingShapesLayerConversion(
            source_geodataframe=source,
            source_shapes_index_feature_name="instance_id",
        ),
    )

    assert geodataframe.index.name == "instance_id"
    assert geodataframe.index.tolist() == ["cell_2", "__annotation_0"]
    assert geodataframe["class_id"].dtype == "Int64"
    assert geodataframe["class_id"].iloc[0] == 2
    assert pd.isna(geodataframe["class_id"].iloc[1])
    assert geodataframe["is_selected"].dtype == "boolean"
    assert geodataframe["is_selected"].iloc[0] == np.False_
    assert pd.isna(geodataframe["is_selected"].iloc[1])
    assert isinstance(geodataframe["category"].dtype, pd.CategoricalDtype)
    assert geodataframe["category"].cat.categories.tolist() == ["alpha", "beta"]
    assert geodataframe["category"].iloc[0] == "beta"
    assert pd.isna(geodataframe["category"].iloc[1])
    assert geodataframe["label"].iloc[0] == "second"
    assert pd.isna(geodataframe["label"].iloc[1])
    assert layer.features["instance_id"].tolist() == ["cell_2", "__annotation_0"]


def test_napari_shapes_layer_to_geodataframe_edit_existing_preserves_unnamed_source_index() -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1]},
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"]),
    )
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    layer.features["index"] = ["cell_1"]

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        conversion=ExistingShapesLayerConversion(
            source_geodataframe=source,
            source_shapes_index_feature_name="index",
        ),
    )

    assert geodataframe.index.name is None
    assert geodataframe.index.tolist() == ["cell_1"]


def test_napari_shapes_layer_to_geodataframe_edit_existing_saves_editable_rectangles_and_ellipses() -> None:
    source = gpd.GeoDataFrame(
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    layer = Shapes(ndim=2)
    layer.add_rectangles(np.asarray([[0.0, 1.0], [3.0, 5.0]]))
    layer.add_ellipses(np.asarray([[10.0, 20.0], [5.0, 8.0]]))
    layer.features["instance_id"] = ["cell_1", np.nan]

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        conversion=ExistingShapesLayerConversion(
            source_geodataframe=source,
            source_shapes_index_feature_name="instance_id",
        ),
        ellipse_segments=16,
    )

    assert geodataframe.index.tolist() == ["cell_1", "__annotation_0"]
    assert all(isinstance(geometry, Polygon) for geometry in geodataframe.geometry)
    assert len(geodataframe.geometry.iloc[1].exterior.coords) == 17


def test_napari_shapes_layer_to_geodataframe_edit_existing_assigns_new_ids_for_copied_generated_ids() -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1]},
        geometry=[_source_polygon()],
        index=pd.Index(["__annotation_0"], name="instance_id"),
    )
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=0)
    _add_polygon(layer, offset=3)
    layer.features["instance_id"] = ["__annotation_0", "__annotation_0"]

    geodataframe = napari_shapes_layer_to_geodataframe(
        layer,
        conversion=ExistingShapesLayerConversion(
            source_geodataframe=source,
            source_shapes_index_feature_name="instance_id",
        ),
    )

    assert geodataframe.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert geodataframe["class_id"].dtype == "Int64"
    assert geodataframe["class_id"].iloc[0] == 1
    assert pd.isna(geodataframe["class_id"].iloc[1])
    assert layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]


@pytest.mark.parametrize(
    ("source_index", "message"),
    [
        (pd.Index(["cell_1", "cell_1"], name="instance_id"), "must be unique"),
        (pd.Index(["cell_1", None], name="instance_id"), "must not be missing"),
    ],
)
def test_napari_shapes_layer_to_geodataframe_edit_existing_rejects_invalid_source_index(
    source_index: pd.Index,
    message: str,
) -> None:
    source = gpd.GeoDataFrame(
        geometry=[_source_polygon(0), _source_polygon(3)],
        index=source_index,
    )
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    layer.features["instance_id"] = ["cell_1"]

    with pytest.raises(ValueError, match=message):
        napari_shapes_layer_to_geodataframe(
            layer,
            conversion=ExistingShapesLayerConversion(
                source_geodataframe=source,
                source_shapes_index_feature_name="instance_id",
            ),
        )

    assert layer.features["instance_id"].tolist() == ["cell_1"]


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (MultiPolygon([_source_polygon()]), "Polygon geometries only"),
        (Point(0, 0), "unsupported geometry `Point`"),
        (Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]), "empty or invalid"),
    ],
)
def test_napari_shapes_layer_to_geodataframe_edit_existing_rejects_unsupported_source_geometry(
    geometry: object,
    message: str,
) -> None:
    source = gpd.GeoDataFrame(
        geometry=[geometry],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    layer.features["instance_id"] = ["cell_1"]

    with pytest.raises(ValueError, match=message):
        napari_shapes_layer_to_geodataframe(
            layer,
            conversion=ExistingShapesLayerConversion(
                source_geodataframe=source,
                source_shapes_index_feature_name="instance_id",
            ),
        )

    assert layer.features["instance_id"].tolist() == ["cell_1"]


def test_napari_shapes_layer_to_geodataframe_edit_existing_rejects_missing_source_feature_column() -> None:
    source = gpd.GeoDataFrame(
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    layer = Shapes(ndim=2)
    _add_polygon(layer)

    with pytest.raises(ValueError, match="missing source index feature column"):
        napari_shapes_layer_to_geodataframe(
            layer,
            conversion=ExistingShapesLayerConversion(
                source_geodataframe=source,
                source_shapes_index_feature_name="instance_id",
            ),
        )

    assert "instance_id" not in layer.features.columns


def test_edit_shapes_element_from_napari_shapes_layer_overwrites_existing_shapes_element() -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[_source_polygon(0), _source_polygon(3)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    sdata = _make_shapes_sdata(source)
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=10)
    _add_polygon(layer, offset=20)
    layer.features["instance_id"] = ["cell_2", np.nan]

    result = edit_shapes_element_from_napari_shapes_layer(
        EditShapesElementRequest(
            sdata=sdata,
            shapes_name="regions",
            coordinate_system="global",
            source_geodataframe=source,
            source_shapes_index_feature_name="instance_id",
        ),
        layer,
    )

    edited = sdata.shapes["regions"]
    assert result.shapes_name == "regions"
    assert result.coordinate_system == "global"
    assert result.row_count == 2
    assert edited.index.name == "instance_id"
    assert edited.index.tolist() == ["cell_2", "__annotation_0"]
    assert edited["class_id"].dtype == "Int64"
    assert edited["class_id"].iloc[0] == 2
    assert pd.isna(edited["class_id"].iloc[1])
    assert layer.features["instance_id"].tolist() == ["cell_2", "__annotation_0"]
    assert isinstance(get_transformation(edited, get_all=True)["global"], Identity)


def test_edit_shapes_element_from_napari_shapes_layer_preserves_other_coordinate_systems() -> None:
    source = gpd.GeoDataFrame(
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    sdata = _make_shapes_sdata(
        source,
        transformations={
            "global": Translation([100, 50], axes=("x", "y")),
            "global_micron": Translation([10, 20], axes=("x", "y")),
        },
    )
    layer = Shapes(
        [
            np.asarray(
                [
                    [50.0, 100.0],
                    [50.0, 102.0],
                    [52.0, 102.0],
                    [52.0, 100.0],
                ]
            )
        ],
        shape_type="polygon",
    )
    layer.features["instance_id"] = ["cell_1"]

    edit_shapes_element_from_napari_shapes_layer(
        EditShapesElementRequest(
            sdata=sdata,
            shapes_name="regions",
            coordinate_system="global",
            source_geodataframe=source,
            source_shapes_index_feature_name="instance_id",
        ),
        layer,
    )

    edited = sdata.shapes["regions"]
    edited_transformations = get_transformation(edited, get_all=True)
    assert set(edited_transformations) == {"global", "global_micron"}
    assert isinstance(edited_transformations["global"], Identity)
    assert edited.geometry.iloc[0].bounds == pytest.approx((100.0, 50.0, 102.0, 52.0))
    assert transform(edited, to_coordinate_system="global_micron").geometry.iloc[0].bounds == pytest.approx(
        (10.0, 20.0, 12.0, 22.0)
    )


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"sdata": None}, "SpatialData object"),
        ({"shapes_name": "missing"}, "does not exist"),
        ({"coordinate_system": "missing"}, "not available"),
        ({"source_shapes_index_feature_name": "   "}, "non-empty string"),
        ({"index_prefix": "   "}, "non-empty string"),
    ],
)
def test_edit_shapes_element_from_napari_shapes_layer_rejects_invalid_request_before_layer_mutation(
    request_kwargs: dict[str, object],
    message: str,
) -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1]},
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"], name="instance_id"),
    )
    sdata = _make_shapes_sdata(source)
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    layer.features["instance_id"] = ["cell_1"]
    kwargs = {
        "sdata": sdata,
        "shapes_name": "regions",
        "coordinate_system": "global",
        "source_geodataframe": source,
        "source_shapes_index_feature_name": "instance_id",
    } | request_kwargs

    with pytest.raises(ValueError, match=message):
        edit_shapes_element_from_napari_shapes_layer(EditShapesElementRequest(**kwargs), layer)

    assert layer.features["instance_id"].tolist() == ["cell_1"]
    assert sdata.shapes["regions"].index.tolist() == ["cell_1"]


def test_edit_shapes_element_from_napari_shapes_layer_preserves_unnamed_source_index() -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1]},
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"]),
    )
    sdata = _make_shapes_sdata(source)
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    _add_polygon(layer, offset=3)
    layer.features["index"] = ["cell_1", np.nan]

    edit_shapes_element_from_napari_shapes_layer(
        EditShapesElementRequest(
            sdata=sdata,
            shapes_name="regions",
            coordinate_system="global",
            source_geodataframe=source,
            source_shapes_index_feature_name="index",
        ),
        layer,
    )

    edited = sdata.shapes["regions"]
    assert edited.index.name is None
    assert edited.index.tolist() == ["cell_1", "__annotation_0"]
    assert layer.features["index"].tolist() == ["cell_1", "__annotation_0"]


def test_edit_shapes_element_from_napari_shapes_layer_persists_backed_overwrite_roundtrip(tmp_path) -> None:
    source = gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[_source_polygon(0), _source_polygon(3)],
        index=pd.Index(["cell_1", "cell_2"], name="instance_id"),
    )
    path = tmp_path / "shapes.zarr"
    _make_shapes_sdata(source).write(path)
    backed_sdata = read_zarr(path)
    layer = Shapes(ndim=2)
    _add_polygon(layer, offset=10)
    _add_polygon(layer, offset=20)
    layer.features["instance_id"] = ["cell_2", np.nan]

    result = edit_shapes_element_from_napari_shapes_layer(
        EditShapesElementRequest(
            sdata=backed_sdata,
            shapes_name="regions",
            coordinate_system="global",
            source_geodataframe=backed_sdata.shapes["regions"],
            source_shapes_index_feature_name="instance_id",
        ),
        layer,
    )

    reread = read_zarr(path)
    edited = reread.shapes["regions"]
    assert result.row_count == 2
    assert edited.index.name == "instance_id"
    assert edited.index.tolist() == ["cell_2", "__annotation_0"]
    assert edited["class_id"].dtype == "Int64"
    assert edited["class_id"].iloc[0] == 2
    assert pd.isna(edited["class_id"].iloc[1])
    assert isinstance(get_transformation(edited, get_all=True)["global"], Identity)


def test_edit_shapes_element_from_napari_shapes_layer_calls_harpy_with_overwrite_and_index_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = gpd.GeoDataFrame(
        geometry=[_source_polygon()],
        index=pd.Index(["cell_1"]),
    )
    sdata = _make_shapes_sdata(source)
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    layer.features["index"] = ["cell_1"]
    captured_kwargs: dict[str, object] = {}

    def fake_add_shapes(sdata: SpatialData, **kwargs):
        captured_kwargs["sdata"] = sdata
        captured_kwargs.update(kwargs)
        return sdata

    monkeypatch.setattr(shapes_annotation_module.hp.sh, "add_shapes", fake_add_shapes)

    result = edit_shapes_element_from_napari_shapes_layer(
        EditShapesElementRequest(
            sdata=sdata,
            shapes_name="regions",
            coordinate_system="global",
            source_geodataframe=source,
            source_shapes_index_feature_name="index",
        ),
        layer,
    )

    assert result.row_count == 1
    assert captured_kwargs["sdata"] is sdata
    assert captured_kwargs["output_shapes_name"] == "regions"
    assert captured_kwargs["instance_key"] is None
    assert captured_kwargs["overwrite"] is True
    written = captured_kwargs["input"]
    assert isinstance(written, gpd.GeoDataFrame)
    assert written.index.name is None
    transformations = captured_kwargs["transformations"]
    assert isinstance(transformations, dict)
    assert isinstance(transformations["global"], Identity)


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
    assert shapes.index.tolist() == ["__annotation_0"]
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


def test_create_shapes_element_from_napari_shapes_layer_rejects_case_variant_collision_before_conversion(
    sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    existing_shapes_name = next(iter(sdata_blobs.shapes))

    with pytest.raises(ValueError, match="already exists"):
        create_shapes_element_from_napari_shapes_layer(
            CreateShapesElementRequest(
                sdata=sdata_blobs,
                shapes_name=existing_shapes_name.upper(),
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
    assert sdata_blobs.shapes["new_regions"].index.tolist() == ["__annotation_0", "__annotation_1"]
    assert layer.features["instance_id"].tolist() == ["__annotation_0", "__annotation_1"]


def test_create_shapes_element_from_napari_shapes_layer_persists_backed_shapes_roundtrip(
    backed_sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)

    result = create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=backed_sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
        ),
        layer,
    )

    reread = read_zarr(backed_sdata_blobs.path)
    shapes = reread.shapes["new_regions"]

    assert result.row_count == 1
    assert shapes.index.name == "instance_id"
    assert shapes.index.tolist() == ["__annotation_0"]
    assert shapes.geometry.iloc[0].equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert isinstance(get_transformation(shapes, get_all=True)["global"], Identity)


def test_create_shapes_element_from_napari_shapes_layer_persists_backed_overwrite_roundtrip(
    backed_sdata_blobs: SpatialData,
) -> None:
    layer = Shapes(ndim=2)
    _add_polygon(layer)
    create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=backed_sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
        ),
        layer,
    )

    _add_polygon(layer, offset=3)
    result = create_shapes_element_from_napari_shapes_layer(
        CreateShapesElementRequest(
            sdata=backed_sdata_blobs,
            shapes_name="new_regions",
            coordinate_system="global",
            overwrite=True,
        ),
        layer,
    )

    reread = read_zarr(backed_sdata_blobs.path)
    shapes = reread.shapes["new_regions"]

    assert result.row_count == 2
    assert shapes.index.name == "instance_id"
    assert shapes.index.tolist() == ["__annotation_0", "__annotation_1"]
    assert shapes.geometry.iloc[0].equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))
    assert shapes.geometry.iloc[1].equals(Polygon([(0, 3), (2, 3), (2, 5), (0, 5)]))
    assert isinstance(get_transformation(shapes, get_all=True)["global"], Identity)


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
