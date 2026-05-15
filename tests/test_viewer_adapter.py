from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.colormaps import CyclicLabelColormap, DirectLabelColormap
from shapely.geometry import LineString, Polygon
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

import napari_harpy.viewer._styling as styling_module
from napari_harpy._app_state import get_or_create_app_state
from napari_harpy._points_value_index import PointsValueSelection
from napari_harpy.core._color_source import ShapeColorSourceSpec, TableColorSourceSpec
from napari_harpy.core.class_palette import default_categorical_colors
from napari_harpy.viewer.adapter import (
    ImageLayerBinding,
    LabelsLayerBinding,
    LayerBindingRegistry,
    PointsLayerBinding,
    PointsLayerIdentity,
    ShapesLayerBinding,
    ViewerAdapter,
)
from napari_harpy.viewer.shapes_styling import SHAPES_FACE_ALPHA


class DummyEventEmitter:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[object], None]] = []

    def connect(self, callback: Callable[[object], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: object | None = None) -> None:
        event = SimpleNamespace(value=value)
        for callback in list(self._callbacks):
            callback(event)


class DummyLayers(list):
    def __init__(self, layers: list[object] | None = None) -> None:
        super().__init__(layers or [])
        self.selection = SimpleNamespace(active=None, select_only=self._select_only)
        self.events = SimpleNamespace(
            inserted=DummyEventEmitter(),
            removed=DummyEventEmitter(),
            reordered=DummyEventEmitter(),
        )

    def _select_only(self, layer: object) -> None:
        self.selection.active = layer

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.events.removed.emit(layer)


class DummyViewer:
    def __init__(self, layers: list[object] | None = None) -> None:
        self.layers = DummyLayers(layers)

    def add_layer(self, layer: object) -> object:
        self.layers.append(layer)
        self.layers.events.inserted.emit(layer)
        return layer


class UnsupportedViewer:
    def __init__(self) -> None:
        self.layers = None


def make_labels_layer(*, sdata, labels_name: str = "blobs_labels", metadata: dict[str, object] | None = None) -> Labels:
    return Labels(
        sdata.labels[labels_name],
        name=labels_name,
        metadata={} if metadata is None else metadata,
    )


def make_image_layer(*, name: str, metadata: dict[str, object] | None = None) -> Image:
    return Image(
        np.zeros((8, 8), dtype=np.float32),
        name=name,
        metadata={} if metadata is None else metadata,
    )


def make_shapes_layer(*, name: str = "cell_boundaries", metadata: dict[str, object] | None = None) -> Shapes:
    return Shapes(
        [np.asarray([(0, 0), (0, 4), (4, 4), (4, 0)], dtype=float)],
        shape_type="polygon",
        name=name,
        metadata={} if metadata is None else metadata,
    )


def make_points_selection(
    row_values: list[str],
    *,
    selected_values: tuple[str, ...] | None = None,
    selection_mode: str = "values",
    total_count: int | None = None,
    render_point_budget: int = 100_000,
    is_sampled: bool = False,
    warning: str | None = None,
) -> PointsValueSelection:
    if selected_values is None:
        selected_values = tuple(dict.fromkeys(row_values))
    value_id_by_value = {value: index for index, value in enumerate(selected_values)}
    coordinates = np.asarray(
        [[float(index), float(index + 10)] for index in range(len(row_values))],
        dtype="float32",
    )
    features = pd.DataFrame(
        {
            "gene": pd.Categorical(row_values, categories=list(selected_values)),
            "value_id": pd.Series([value_id_by_value[value] for value in row_values], dtype="uint32"),
        }
    )
    return PointsValueSelection(
        coordinates=coordinates,
        features=features,
        index_column="gene",
        selected_values=selected_values,
        selected_value_ids=tuple(value_id_by_value[value] for value in selected_values),
        selection_mode=selection_mode,  # type: ignore[arg-type]
        total_count=len(row_values) if total_count is None else total_count,
        render_point_budget=render_point_budget,
        is_sampled=is_sampled,
        warning=warning,
    )


def make_points_identity(sdata: object, *, index_column: str = "gene") -> PointsLayerIdentity:
    return PointsLayerIdentity(
        sdata=sdata,
        points_name="transcripts",
        coordinate_system="global",
        index_column=index_column,
    )


def make_shapes_sdata(geodataframe: gpd.GeoDataFrame, shapes_name: str = "cell_boundaries") -> SimpleNamespace:
    shapes = ShapesModel.parse(geodataframe, transformations={"global": Identity()})
    return SimpleNamespace(shapes={shapes_name: shapes})


def make_colorable_shapes_sdata(shapes_name: str = "cell_boundaries") -> SimpleNamespace:
    geodataframe = gpd.GeoDataFrame(
        {
            "cell_type": pd.Categorical(["T", "B"], categories=["T", "B"]),
            "cell_type_colors": ["#ff0000", "#00ff00"],
            "score": [0.0, 1.0],
        },
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
            Polygon([(5, 0), (9, 0), (9, 4), (5, 4), (5, 0)]),
        ],
        index=["cell_1", "cell_2"],
    )
    return make_shapes_sdata(geodataframe, shapes_name=shapes_name)


def get_shapes_binding(adapter: ViewerAdapter, layer: Shapes) -> ShapesLayerBinding:
    binding = adapter.layer_bindings.get_binding(layer)
    assert isinstance(binding, ShapesLayerBinding)
    return binding


def test_app_state_initializes_viewer_services() -> None:
    viewer = DummyViewer()

    state = get_or_create_app_state(viewer)

    assert state.layer_bindings is not None
    assert state.viewer_adapter is not None
    assert state.viewer_adapter.layer_bindings is state.layer_bindings


def test_layer_binding_registry_registers_and_unregisters_layers() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="raw image")

    binding = registry.register_image_layer(
        layer,
        element_name="blobs_image",
        coordinate_system="global",
    )

    assert isinstance(binding, ImageLayerBinding)
    assert binding.layer is layer
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert registry.get_binding(layer) == binding
    assert registry.find_bindings(element_name="blobs_image", element_type="image") == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata

    removed = registry.unregister_layer(layer)

    assert removed == binding
    assert registry.get_binding(layer) is None


def test_layer_binding_registry_tracks_channel_overlay_identity() -> None:
    registry = LayerBindingRegistry()
    layer = make_image_layer(name="channel_0")

    binding = registry.register_image_layer(
        layer,
        element_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    )

    assert isinstance(binding, ImageLayerBinding)
    assert binding.image_display_mode == "overlay"
    assert binding.channel_index == 0
    assert binding.channel_name == "DAPI"
    assert registry.find_bindings(
        element_name="blobs_image",
        element_type="image",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="DAPI",
    ) == [binding]
    assert "image_display_mode" not in layer.metadata
    assert "channel_index" not in layer.metadata
    assert "channel_name" not in layer.metadata


def test_layer_binding_registry_tracks_points_identity() -> None:
    registry = LayerBindingRegistry()
    layer = Points(np.asarray([[0.0, 1.0]], dtype="float32"), name="transcripts: gene=AAMP")

    binding = registry.register_points_layer(
        layer,
        element_name="transcripts",
        coordinate_system="global",
        index_column="gene",
    )

    assert isinstance(binding, PointsLayerBinding)
    assert binding.element_name == "transcripts"
    assert binding.element_type == "points"
    assert binding.coordinate_system == "global"
    assert binding.index_column == "gene"
    assert registry.find_bindings(
        element_name="transcripts",
        element_type="points",
        points_index_column="gene",
    ) == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "index_column" not in layer.metadata


def test_layer_binding_registry_tracks_shapes_identity() -> None:
    registry = LayerBindingRegistry()
    layer = make_shapes_layer()

    binding = registry.register_shapes_layer(
        layer,
        element_name="cell_boundaries",
        coordinate_system="global",
        source_shapes_index_by_row=("cell_1",),
        source_shapes_index_feature_name="cell_id",
        skipped_geometry_count=2,
    )

    assert isinstance(binding, ShapesLayerBinding)
    assert binding.element_name == "cell_boundaries"
    assert binding.element_type == "shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.style_spec is None
    assert binding.source_shapes_index_by_row == ("cell_1",)
    assert binding.source_shapes_index_feature_name == "cell_id"
    assert binding.skipped_geometry_count == 2
    assert registry.find_bindings(element_name="cell_boundaries", element_type="shapes") == [binding]
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "source_shapes_index_by_row" not in layer.metadata
    assert "source_shapes_index_feature_name" not in layer.metadata
    assert "skipped_geometry_count" not in layer.metadata


def test_layer_binding_registry_tracks_shapes_role_and_style_spec() -> None:
    registry = LayerBindingRegistry()
    layer = make_shapes_layer()
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    binding = registry.register_shapes_layer(
        layer,
        element_name="cell_boundaries",
        coordinate_system="global",
        shapes_role="styled",
        style_spec=style_spec,
        source_shapes_index_by_row=("cell_1",),
        source_shapes_index_feature_name="cell_id",
    )

    assert isinstance(binding, ShapesLayerBinding)
    assert binding.shapes_role == "styled"
    assert binding.style_spec == style_spec
    assert binding.source_shapes_index_by_row == ("cell_1",)
    assert registry.find_bindings(
        element_name="cell_boundaries",
        element_type="shapes",
        shapes_role="styled",
        style_spec=style_spec,
    ) == [binding]
    assert "shapes_role" not in layer.metadata
    assert "style_spec" not in layer.metadata
    assert "style_source_kind" not in layer.metadata
    assert "style_value_key" not in layer.metadata
    assert "style_value_kind" not in layer.metadata


def test_shapes_layer_binding_rejects_invalid_role_style_spec_combinations() -> None:
    layer = make_shapes_layer()
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    with pytest.raises(ValueError, match="Primary shapes bindings must not carry"):
        ShapesLayerBinding(
            layer=layer,
            element_name="cell_boundaries",
            coordinate_system="global",
            shapes_role="primary",
            style_spec=style_spec,
        )

    with pytest.raises(ValueError, match="Styled shapes bindings require"):
        ShapesLayerBinding(
            layer=layer,
            element_name="cell_boundaries",
            coordinate_system="global",
            shapes_role="styled",
        )


def test_labels_layer_binding_rejects_invalid_role_style_spec_combinations() -> None:
    layer = make_labels_layer(sdata=SimpleNamespace(labels={"blobs_labels": np.zeros((2, 2), dtype=np.int32)}))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    with pytest.raises(ValueError, match="Primary labels bindings must not carry"):
        LabelsLayerBinding(
            layer=layer,
            element_name="blobs_labels",
            coordinate_system="global",
            labels_role="primary",
            style_spec=style_spec,
        )

    with pytest.raises(ValueError, match="Styled labels bindings require"):
        LabelsLayerBinding(
            layer=layer,
            element_name="blobs_labels",
            coordinate_system="global",
            labels_role="styled",
        )


def test_points_layer_binding_rejects_empty_index_column() -> None:
    layer = Points(np.asarray([[0.0, 1.0]], dtype="float32"), name="points")

    with pytest.raises(ValueError, match="index column"):
        PointsLayerBinding(
            layer=layer,
            element_name="transcripts",
            coordinate_system="global",
            index_column="",
        )


def test_layer_binding_registry_tracks_labels_role_and_style_spec() -> None:
    registry = LayerBindingRegistry()
    layer = make_labels_layer(sdata=SimpleNamespace(labels={"blobs_labels": np.zeros((2, 2), dtype=np.int32)}))
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    binding = registry.register_labels_layer(
        layer,
        element_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )

    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "styled"
    assert binding.style_spec == style_spec
    assert registry.find_bindings(
        element_name="blobs_labels",
        element_type="labels",
        labels_role="styled",
        style_spec=style_spec,
    ) == [binding]
    assert "labels_role" not in layer.metadata
    assert "style_spec" not in layer.metadata
    assert "style_table_name" not in layer.metadata
    assert "style_source_kind" not in layer.metadata
    assert "style_value_key" not in layer.metadata
    assert "style_value_kind" not in layer.metadata


def test_viewer_adapter_activate_layer_selects_only_matching_layer() -> None:
    image_layer = make_image_layer(name="image")
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    activated = adapter.activate_layer(image_layer)

    assert activated is True
    assert viewer.layers.selection.active is image_layer


def test_viewer_adapter_emits_active_layer_changed_on_activation() -> None:
    image_layer = make_image_layer(name="image")
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)
    active_layers: list[object] = []

    adapter.active_layer_changed.connect(active_layers.append)

    adapter.activate_layer(image_layer)

    assert active_layers == [image_layer]


def test_viewer_adapter_ensure_labels_loaded_rejects_unsupported_viewer(sdata_blobs) -> None:
    adapter = ViewerAdapter(UnsupportedViewer())

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    except ValueError as error:
        assert "does not expose napari-compatible layer APIs" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unsupported viewer.")


def test_viewer_adapter_finds_registered_labels_layer(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_labels_layer(
        labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )

    assert adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels") is labels_layer


def test_viewer_adapter_primary_labels_lookup_ignores_styled_variants(sdata_blobs) -> None:
    styled_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([styled_layer])
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    adapter.register_labels_layer(
        styled_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )

    assert adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels", "global") is None
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_spec, "global") is styled_layer
    assert adapter.get_loaded_styled_labels_layers(sdata_blobs, "blobs_labels", "global") == [styled_layer]


def test_viewer_adapter_ignores_unregistered_labels_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    labels_layer = make_labels_layer(
        sdata=sdata_blobs,
        metadata={"sdata": sdata_blobs, "name": "blobs_labels", "_current_cs": "global"},
    )
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layer = adapter.get_loaded_primary_labels_layer(sdata_blobs, "blobs_labels", "global")

    assert loaded_layer is None
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_remove_labels_layer_removes_matching_binding(sdata_blobs) -> None:
    labels_layer = make_labels_layer(sdata=sdata_blobs)
    viewer = DummyViewer([labels_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_labels_layer(
        labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )

    removed_layer = adapter.remove_labels_layer(sdata_blobs, "blobs_labels", "global")

    assert removed_layer is labels_layer
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(labels_layer) is None


def test_viewer_adapter_returns_registered_image_layers_in_viewer_order(sdata_blobs) -> None:
    first_layer = make_image_layer(name="channel_0")
    second_layer = make_image_layer(name="channel_1")
    viewer = DummyViewer([second_layer, first_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        first_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        second_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
    )

    assert adapter.get_loaded_image_layers(sdata_blobs, "blobs_image") == [second_layer, first_layer]


def test_viewer_adapter_remove_image_layers_removes_stack_and_overlay_bindings(sdata_blobs) -> None:
    stack_layer = make_image_layer(name="blobs_image")
    overlay_layer = make_image_layer(name="blobs_image[0]")
    viewer = DummyViewer([stack_layer, overlay_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        stack_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="stack",
    )
    adapter.register_image_layer(
        overlay_layer,
        sdata=sdata_blobs,
        image_name="blobs_image",
        coordinate_system="global",
        image_display_mode="overlay",
        channel_index=0,
        channel_name="0",
    )

    removed_layers = adapter.remove_image_layers(sdata_blobs, "blobs_image", "global")

    assert removed_layers == [stack_layer, overlay_layer]
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert adapter.layer_bindings.get_binding(overlay_layer) is None


def test_viewer_adapter_remove_layers_outside_coordinate_system_removes_only_nonmatching_registered_layers(
    sdata_blobs,
) -> None:
    keep_layer = make_image_layer(name="keep")
    remove_image_layer = make_image_layer(name="remove_image")
    remove_labels_layer = make_labels_layer(sdata=sdata_blobs)
    remove_shapes_layer = make_shapes_layer(name="remove_shapes")
    external_layer = make_image_layer(name="external")
    viewer = DummyViewer([keep_layer, remove_image_layer, remove_labels_layer, remove_shapes_layer, external_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        keep_layer,
        sdata=sdata_blobs,
        image_name="keep_image",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        remove_image_layer,
        sdata=sdata_blobs,
        image_name="remove_image",
        coordinate_system="local",
    )
    adapter.register_labels_layer(
        remove_labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="local",
    )
    adapter.register_shapes_layer(
        remove_shapes_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="local",
    )

    removed_bindings = adapter.remove_layers_outside_coordinate_system(sdata=sdata_blobs, coordinate_system="global")

    assert [binding.element_name for binding in removed_bindings] == ["remove_image", "blobs_labels", "blobs_polygons"]
    assert list(viewer.layers) == [keep_layer, external_layer]
    assert adapter.layer_bindings.get_binding(keep_layer) is not None
    assert adapter.layer_bindings.get_binding(remove_image_layer) is None
    assert adapter.layer_bindings.get_binding(remove_labels_layer) is None
    assert adapter.layer_bindings.get_binding(remove_shapes_layer) is None
    assert adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_adapter_remove_layers_for_sdata_removes_only_matching_registered_layers(sdata_blobs) -> None:
    removed_image_layer = make_image_layer(name="remove_image")
    removed_labels_layer = make_labels_layer(sdata=sdata_blobs)
    removed_shapes_layer = make_shapes_layer(name="remove_shapes")
    kept_layer = make_image_layer(name="keep")
    external_layer = make_image_layer(name="external")
    other_sdata = SimpleNamespace()
    viewer = DummyViewer([removed_image_layer, removed_labels_layer, removed_shapes_layer, kept_layer, external_layer])
    adapter = ViewerAdapter(viewer)

    adapter.register_image_layer(
        removed_image_layer,
        sdata=sdata_blobs,
        image_name="remove_image",
        coordinate_system="global",
    )
    adapter.register_labels_layer(
        removed_labels_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )
    adapter.register_shapes_layer(
        removed_shapes_layer,
        sdata=sdata_blobs,
        shapes_name="blobs_polygons",
        coordinate_system="global",
    )
    adapter.register_image_layer(
        kept_layer,
        sdata=other_sdata,
        image_name="keep_image",
        coordinate_system="global",
    )

    removed_bindings = adapter.remove_layers_for_sdata(sdata_blobs)

    assert [binding.element_name for binding in removed_bindings] == ["remove_image", "blobs_labels", "blobs_polygons"]
    assert list(viewer.layers) == [kept_layer, external_layer]
    assert adapter.layer_bindings.get_binding(removed_image_layer) is None
    assert adapter.layer_bindings.get_binding(removed_labels_layer) is None
    assert adapter.layer_bindings.get_binding(removed_shapes_layer) is None
    assert adapter.layer_bindings.get_binding(kept_layer) is not None
    assert adapter.layer_bindings.get_binding(external_layer) is None


def test_viewer_adapter_ignores_unregistered_image_layer_even_with_legacy_metadata(sdata_blobs) -> None:
    image_layer = make_image_layer(
        name="raw image",
        metadata={"sdata": sdata_blobs, "name": "blobs_image", "_current_cs": "global"},
    )
    viewer = DummyViewer([image_layer])
    adapter = ViewerAdapter(viewer)

    loaded_layers = adapter.get_loaded_image_layers(sdata_blobs, "blobs_image")

    assert loaded_layers == []
    assert adapter.layer_bindings.get_binding(image_layer) is None


def test_viewer_adapter_ensure_points_layer_from_selection_creates_registered_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AAMP"], selected_values=("AAMP",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.created is True
    assert result.warnings == ()
    assert result.layer in viewer.layers
    assert result.layer.name == "transcripts: gene=AAMP"
    np.testing.assert_array_equal(result.layer.data, selection.coordinates)
    assert result.layer.features.equals(selection.features)
    assert np.all(result.layer.size == 1.0)
    assert result.layer.opacity == 0.8
    assert all(symbol.value == "disc" for symbol in result.layer.symbol)
    assert np.all(result.layer.border_width == 0)
    assert np.allclose(result.layer.face_color, np.asarray([to_rgba("#00FFFF")] * selection.loaded_count))
    binding = adapter.layer_bindings.get_binding(result.layer)
    assert isinstance(binding, PointsLayerBinding)
    assert binding.element_name == "transcripts"
    assert binding.element_type == "points"
    assert binding.coordinate_system == "global"
    assert binding.index_column == "gene"
    assert binding.sdata_id == id(sdata)


def test_viewer_adapter_ensure_points_layer_from_selection_updates_existing_layer_preserving_visibility() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    first_selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    second_selection = make_points_selection(["AXL"], selected_values=("AXL",))
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    first = adapter._ensure_points_layer_from_selection(identity, selection=first_selection)
    first.layer.visible = False

    second = adapter._ensure_points_layer_from_selection(identity, selection=second_selection)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.layer.visible is False
    assert len(viewer.layers) == 1
    assert second.layer.name == "transcripts: gene=AXL"
    np.testing.assert_array_equal(second.layer.data, second_selection.coordinates)
    assert second.layer.features.equals(second_selection.features)


def test_viewer_adapter_ensure_points_layer_from_selection_ignores_unregistered_same_name_layer() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    external_layer = Points(selection.coordinates, name="transcripts: gene=AAMP")
    viewer = DummyViewer([external_layer])
    adapter = ViewerAdapter(viewer)

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.layer is not external_layer
    assert list(viewer.layers) == [external_layer, result.layer]
    assert adapter.layer_bindings.get_binding(external_layer) is None
    assert isinstance(adapter.layer_bindings.get_binding(result.layer), PointsLayerBinding)


def test_viewer_adapter_ensure_points_layer_from_selection_rejects_identity_selection_mismatch() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata, index_column="target")
    selection = make_points_selection(["AAMP"], selected_values=("AAMP",))
    adapter = ViewerAdapter(DummyViewer())

    with pytest.raises(ValueError, match="index_column"):
        adapter._ensure_points_layer_from_selection(identity, selection=selection)


def test_viewer_adapter_ensure_points_layer_from_selection_uses_all_values_name() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"), selection_mode="all")
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.layer.name == "transcripts: all gene values"


def test_viewer_adapter_ensure_points_layer_from_selection_uses_categorical_colors_for_small_selections() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(["AAMP", "AXL"], selected_values=("AAMP", "AXL"))
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "cycle"
    expected_colors = np.asarray([to_rgba(color) for color in default_categorical_colors(2)], dtype=np.float32)
    assert np.allclose(result.layer.face_color, expected_colors)


def test_viewer_adapter_ensure_points_layer_from_selection_uses_solid_color_above_categorical_limit() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selected_values = tuple(f"gene_{index}" for index in range(103))
    selection = make_points_selection(list(selected_values), selected_values=selected_values)
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert str(result.layer.face_color_mode) == "direct"
    assert np.allclose(result.layer.face_color, np.asarray([to_rgba("#00FFFF")] * selection.loaded_count))
    assert len(result.warnings) == 1
    assert "categorical coloring is disabled" in result.warnings[0]


def test_viewer_adapter_ensure_points_layer_from_selection_returns_sampled_warning() -> None:
    sdata = SimpleNamespace()
    identity = make_points_identity(sdata)
    selection = make_points_selection(
        ["AAMP", "AXL"],
        selected_values=("AAMP", "AXL"),
        total_count=10,
        render_point_budget=2,
        is_sampled=True,
        warning="Showing 2 of 10 selected points.",
    )
    adapter = ViewerAdapter(DummyViewer())

    result = adapter._ensure_points_layer_from_selection(identity, selection=selection)

    assert result.warnings == ("Showing 2 of 10 selected points.",)


def test_viewer_adapter_ensure_labels_loaded_adds_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    assert layer in viewer.layers
    assert layer.name == "blobs_labels"
    assert layer.affine is not None
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.element_name == "blobs_labels"
    assert binding.element_type == "labels"
    assert binding.coordinate_system == "global"
    assert binding.labels_role == "primary"
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "labels_role" not in layer.metadata
    assert "sdata" not in layer.metadata
    assert "_current_cs" not in layer.metadata
    assert labels_events == ["changed"]


def test_viewer_adapter_primary_labels_signal_ignores_styled_bindings(sdata_blobs) -> None:
    primary_layer = make_labels_layer(sdata=sdata_blobs, labels_name="blobs_labels")
    styled_layer = make_labels_layer(sdata=sdata_blobs, labels_name="blobs_labels")
    styled_layer.name = "blobs_labels[cell_type]"
    viewer = DummyViewer([primary_layer, styled_layer])
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    adapter.register_labels_layer(
        styled_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
        labels_role="styled",
        style_spec=style_spec,
    )
    assert labels_events == []

    adapter.register_labels_layer(
        primary_layer,
        sdata=sdata_blobs,
        labels_name="blobs_labels",
        coordinate_system="global",
    )
    assert labels_events == ["changed"]

    viewer.layers.remove(styled_layer)
    assert labels_events == ["changed"]

    viewer.layers.remove(primary_layer)
    assert labels_events == ["changed", "changed"]


def test_viewer_adapter_ensure_styled_labels_loaded_creates_registered_overlay_with_stored_palette(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    region_rows = table.obs.loc[table.obs["region"] == "blobs_labels"].copy()
    table.obs["cell_type"] = pd.Categorical(
        np.where(region_rows["instance_id"].to_numpy() % 2 == 0, "even", "odd"),
        categories=["odd", "even"],
    )
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.layer in viewer.layers
    assert result.layer.name == "blobs_labels[obs:cell_type]"
    assert isinstance(result.layer.colormap, DirectLabelColormap)
    binding = adapter.layer_bindings.get_binding(result.layer)
    assert isinstance(binding, LabelsLayerBinding)
    assert binding.labels_role == "styled"
    assert binding.style_spec == style_spec
    assert "style_table_name" not in result.layer.metadata
    assert "style_source_kind" not in result.layer.metadata
    assert "style_value_key" not in result.layer.metadata
    assert "style_value_kind" not in result.layer.metadata
    assert list(result.layer.features.columns) == ["index", "instance_id", "cell_type"]

    odd_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 1, "instance_id"].iloc[0])
    even_instance = int(region_rows.loc[region_rows["instance_id"] % 2 == 0, "instance_id"].iloc[0])
    odd_feature_row = result.layer.features.iloc[result.layer._label_index[odd_instance]]
    assert odd_feature_row["index"] == odd_instance
    assert odd_feature_row["instance_id"] == odd_instance
    assert np.allclose(result.layer.colormap.color_dict[odd_instance], np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert np.allclose(
        result.layer.colormap.color_dict[even_instance], np.asarray(to_rgba("#00ff00"), dtype=np.float32)
    )


def test_viewer_adapter_ensure_styled_labels_loaded_reuses_matching_variant(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_0_sum",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)
    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.value_kind == "continuous"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_styled_labels_loaded_creates_distinct_variants_for_different_style_specs(
    sdata_blobs,
) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs)
    style_a = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    style_b = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_0_sum",
        value_kind="continuous",
    )
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_a)
    second = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_b)

    assert first.layer is not second.layer
    assert len(viewer.layers) == 2
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_a, "global") is first.layer
    assert adapter.get_loaded_styled_labels_layer(sdata_blobs, "blobs_labels", style_b, "global") is second.layer


def test_viewer_adapter_ensure_styled_labels_loaded_invalid_palette_falls_back(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs + [])
    table.uns["cell_type_colors"] = ["#ff0000", "#00ff00"]
    table.obs["cell_type"] = pd.Categorical(["odd"] * table.n_obs, categories=["odd"])

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.palette_source == "default_invalid"
    assert result.coercion_applied is False


def test_viewer_adapter_ensure_styled_labels_loaded_colors_bool_obs_categorically(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["is_even"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64) % 2 == 0,
        index=table.obs.index,
    )
    table.uns["is_even_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="is_even",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    odd_instance = int(table.obs.loc[table.obs["instance_id"] % 2 == 1, "instance_id"].iloc[0])
    even_instance = int(table.obs.loc[table.obs["instance_id"] % 2 == 0, "instance_id"].iloc[0])
    assert features.loc[odd_instance, "is_even"] == np.False_
    assert features.loc[even_instance, "is_even"] == np.True_
    assert np.allclose(result.layer.colormap.color_dict[odd_instance], np.asarray(to_rgba("#ff0000"), dtype=np.float32))
    assert np.allclose(
        result.layer.colormap.color_dict[even_instance], np.asarray(to_rgba("#00ff00"), dtype=np.float32)
    )


def test_viewer_adapter_ensure_styled_labels_loaded_colors_binary_int_obs_categorically(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["binary_state"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64) % 2,
        index=table.obs.index,
        dtype="int64",
    )
    table.uns["binary_state_colors"] = ["#ff0000", "#00ff00"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="binary_state",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    zero_instance = int(table.obs.loc[table.obs["binary_state"] == 0, "instance_id"].iloc[0])
    one_instance = int(table.obs.loc[table.obs["binary_state"] == 1, "instance_id"].iloc[0])
    assert int(features.loc[zero_instance, "binary_state"]) == 0
    assert int(features.loc[one_instance, "binary_state"]) == 1
    assert np.allclose(
        result.layer.colormap.color_dict[zero_instance], np.asarray(to_rgba("#ff0000"), dtype=np.float32)
    )
    assert np.allclose(result.layer.colormap.color_dict[one_instance], np.asarray(to_rgba("#00ff00"), dtype=np.float32))


def test_viewer_adapter_ensure_styled_labels_loaded_colors_non_binary_int_obs_continuously(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["object_score"] = pd.Series(
        table.obs["instance_id"].to_numpy(dtype=np.int64),
        index=table.obs.index,
        dtype="int64",
    )

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="object_score",
        value_kind="continuous",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    features = result.layer.features.set_index("index")
    min_instance = int(table.obs["instance_id"].min())
    max_instance = int(table.obs["instance_id"].max())
    assert float(features.loc[min_instance, "object_score"]) == float(min_instance)
    assert float(features.loc[max_instance, "object_score"]) == float(max_instance)
    assert not np.allclose(
        result.layer.colormap.color_dict[min_instance], result.layer.colormap.color_dict[max_instance]
    )


def test_viewer_adapter_ensure_styled_labels_loaded_colors_instance_key_as_labels(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="instance_id",
        value_kind="instance",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "instance"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(result.layer.colormap, CyclicLabelColormap)
    assert result.layer.name == "blobs_labels[obs:instance_id]"
    assert "style_value_kind" not in result.layer.metadata
    assert list(result.layer.features.columns) == ["index", "instance_id"]

    instance_ids = table.obs.loc[table.obs["region"] == "blobs_labels", "instance_id"].astype("int64").tolist()
    first_instance, second_instance = instance_ids[:2]
    mapped_colors = result.layer.colormap.map(np.asarray([first_instance, second_instance], dtype=np.int64))
    assert not np.allclose(mapped_colors[0], mapped_colors[1])
    assert np.allclose(result.layer.colormap.map(np.asarray([0], dtype=np.int64))[0], np.zeros(4))


def test_viewer_adapter_ensure_styled_labels_loaded_coerces_string_obs_to_categorical(sdata_blobs) -> None:
    table = sdata_blobs["table"]
    table.obs["sample_type"] = pd.Series(
        np.where(table.obs["instance_id"].to_numpy() % 2 == 0, "even", "odd"),
        index=table.obs.index,
        dtype="object",
    )
    table.uns["sample_type_colors"] = ["#123456", "#654321"]

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="sample_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.coercion_applied is True
    assert result.palette_source == "default_missing"
    assert isinstance(result.layer.colormap, DirectLabelColormap)


def test_viewer_adapter_ensure_styled_labels_loaded_warns_for_high_cardinality_string_obs(
    sdata_blobs,
    monkeypatch,
) -> None:
    table = sdata_blobs["table"]
    table.obs["cell_uuid"] = pd.Series(
        [f"cell-{index:04d}" for index in range(table.n_obs)],
        index=table.obs.index,
        dtype="object",
    )

    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="obs_column",
        value_key="cell_uuid",
        value_kind="categorical",
    )

    warning_messages: list[str] = []

    class DummyLogger:
        def warning(self, message: str) -> None:
            warning_messages.append(message)

    monkeypatch.setattr(styling_module, "logger", DummyLogger())

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "categorical"
    assert result.coercion_applied is True
    assert result.palette_source == "default_missing"
    assert isinstance(result.layer.colormap, DirectLabelColormap)
    assert len(warning_messages) == 1
    assert "exceeds the categorical viewer-coloring threshold" in warning_messages[0]
    assert "Harpy will render it with the default categorical palette anyway" in warning_messages[0]


def test_viewer_adapter_ensure_styled_labels_loaded_x_var_is_continuous(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = TableColorSourceSpec(
        table_name="table",
        source_kind="x_var",
        value_key="channel_1_sum",
        value_kind="continuous",
    )

    result = adapter.ensure_styled_labels_loaded(sdata_blobs, "blobs_labels", "global", style_spec)

    assert result.value_kind == "continuous"
    assert result.palette_source is None
    assert result.coercion_applied is False
    assert isinstance(result.layer.colormap, DirectLabelColormap)


def test_viewer_adapter_ensure_labels_loaded_reuses_matching_existing_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")
    second = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    assert first is second
    assert len(viewer.layers) == 1


def test_viewer_adapter_unregisters_binding_when_user_removes_labels_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "global")

    viewer.layers.remove(layer)

    assert layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_labels_loaded_supports_multiscale_labels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_labels_loaded(sdata_blobs, "blobs_multiscale_labels", "global")

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert layer in viewer.layers


def test_viewer_adapter_ensure_labels_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_labels_loaded(sdata_blobs, "blobs_labels", "not_a_coordinate_system")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_labels_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_shapes_loaded_adds_polygon_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    labels_events: list[str] = []

    adapter.primary_labels_layers_changed.connect(lambda: labels_events.append("changed"))

    layer = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert layer in viewer.layers
    assert layer.name == "blobs_polygons"
    assert layer.shape_type == ["polygon"] * len(sdata_blobs.shapes["blobs_polygons"])
    expected_source_index_by_row = tuple(sdata_blobs.shapes["blobs_polygons"].index.to_list())
    assert list(layer.features.columns) == ["index"]
    assert tuple(layer.features["index"].to_list()) == expected_source_index_by_row
    binding = get_shapes_binding(adapter, layer)
    assert binding.element_name == "blobs_polygons"
    assert binding.element_type == "shapes"
    assert binding.coordinate_system == "global"
    assert binding.shapes_role == "primary"
    assert binding.style_spec is None
    assert binding.source_shapes_index_by_row == expected_source_index_by_row
    assert binding.source_shapes_index_feature_name == "index"
    assert binding.skipped_geometry_count == 0
    assert "element_name" not in layer.metadata
    assert "element_type" not in layer.metadata
    assert "coordinate_system" not in layer.metadata
    assert "source_shapes_index_by_row" not in layer.metadata
    assert "source_shapes_index_feature_name" not in layer.metadata
    assert "skipped_geometry_count" not in layer.metadata
    assert labels_events == []


def test_viewer_adapter_ensure_shapes_loaded_expands_multipolygons_with_source_mapping(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_multipolygons", "global")

    expected_indices: list[object] = []
    for source_index, geometry in sdata_blobs.shapes["blobs_multipolygons"].geometry.items():
        expected_indices.extend([source_index] * len(geometry.geoms))

    assert len(layer.data) == len(expected_indices)
    assert layer.shape_type == ["polygon"] * len(expected_indices)
    binding = get_shapes_binding(adapter, layer)
    assert binding.source_shapes_index_by_row == tuple(expected_indices)
    assert tuple(layer.features["index"].to_list()) == tuple(expected_indices)


def test_viewer_adapter_ensure_shapes_loaded_renders_circles_as_ellipses(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_circles", "global")

    circles = sdata_blobs.shapes["blobs_circles"]
    first_circle = circles.iloc[0]
    radius = float(first_circle["radius"])
    y = float(first_circle.geometry.y)
    x = float(first_circle.geometry.x)
    expected_first_ellipse = np.asarray(
        [
            (y - radius, x - radius),
            (y + radius, x - radius),
            (y + radius, x + radius),
            (y - radius, x + radius),
        ],
        dtype=float,
    )

    assert layer.shape_type == ["ellipse"] * len(circles)
    assert np.allclose(layer.data[0], expected_first_ellipse)
    binding = get_shapes_binding(adapter, layer)
    assert binding.source_shapes_index_by_row == tuple(circles.index.to_list())
    assert tuple(layer.features["index"].to_list()) == tuple(circles.index.to_list())


def test_viewer_adapter_ensure_shapes_loaded_preserves_polygon_holes() -> None:
    polygon = Polygon(
        shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        holes=[[(2, 2), (2, 5), (5, 5), (5, 2), (2, 2)]],
    )
    geodataframe = gpd.GeoDataFrame({"name": ["polygon_with_hole"], "geometry": [polygon]}, index=["donut"])
    sdata = make_shapes_sdata(geodataframe, shapes_name="donuts")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_shapes_loaded(sdata, "donuts", "global")
    labels = layer.to_labels(labels_shape=(12, 12))

    binding = get_shapes_binding(adapter, layer)
    assert binding.source_shapes_index_by_row == ("donut",)
    assert list(layer.features.columns) == ["index"]
    assert layer.features.iloc[0]["index"] == "donut"
    assert "index: donut" in layer.get_status(position=(1, 1))["value"]
    assert labels[1, 1] == 1
    assert labels[3, 3] == 0


def test_viewer_adapter_ensure_shapes_loaded_uses_named_geodataframe_index_in_features() -> None:
    polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    geodataframe = gpd.GeoDataFrame(
        {"cell_id": ["column_value"], "name": ["boundary"]}, geometry=[polygon], index=["cell_1"]
    )
    geodataframe.index.name = "cell_id"
    sdata = make_shapes_sdata(geodataframe, shapes_name="cell_boundaries")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")

    binding = get_shapes_binding(adapter, layer)
    assert binding.source_shapes_index_by_row == ("cell_1",)
    assert binding.source_shapes_index_feature_name == "cell_id"
    assert list(layer.features.columns) == ["cell_id"]
    assert layer.features.iloc[0]["cell_id"] == "cell_1"
    assert "cell_id: cell_1" in layer.get_status(position=(1, 1))["value"]


def test_viewer_adapter_ensure_shapes_loaded_skips_empty_invalid_or_unsupported_geometries() -> None:
    valid_polygon = Polygon([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)])
    invalid_bowtie = Polygon([(4, 0), (7, 3), (7, 0), (4, 3), (4, 0)])
    empty_polygon = Polygon()
    unsupported_line = LineString([(0, 5), (5, 5)])
    geodataframe = gpd.GeoDataFrame(
        {"geometry": [valid_polygon, invalid_bowtie, empty_polygon, unsupported_line]},
        index=["valid", "bowtie", "empty", "line"],
    )
    sdata = make_shapes_sdata(geodataframe, shapes_name="mixed_shapes")
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_shapes_loaded(sdata, "mixed_shapes", "global")

    assert layer.shape_type == ["polygon", "polygon", "polygon"]
    binding = get_shapes_binding(adapter, layer)
    assert binding.source_shapes_index_by_row == ("valid", "bowtie", "bowtie")
    assert binding.skipped_geometry_count == 2


def test_viewer_adapter_ensure_shapes_loaded_reuses_matching_existing_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")
    second = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    assert first is second
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_registered_variant_with_stored_palette() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert result.created is True
    assert result.value_kind == "categorical"
    assert result.palette_source == "stored"
    assert result.coercion_applied is False
    assert result.layer in viewer.layers
    assert result.layer.name == "cell_boundaries[shape:cell_type]"
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_boundaries", "global") is None
    binding = get_shapes_binding(adapter, result.layer)
    assert binding.shapes_role == "styled"
    assert binding.style_spec == style_spec
    assert binding.source_shapes_index_by_row == ("cell_1", "cell_2")
    assert binding.source_shapes_index_feature_name == "index"
    assert list(result.layer.features.columns) == ["index", "cell_type"]
    assert result.layer.features["cell_type"].to_list() == ["T", "B"]
    np.testing.assert_allclose(result.layer.face_color[0], (*to_rgba("#ff0000")[:3], 0.0))
    np.testing.assert_allclose(result.layer.edge_color[1], (*to_rgba("#00ff00")[:3], 1.0))
    assert "shapes_role" not in result.layer.metadata
    assert "style_source_kind" not in result.layer.metadata
    assert "style_value_key" not in result.layer.metadata
    assert "style_value_kind" not in result.layer.metadata


def test_viewer_adapter_ensure_styled_shapes_loaded_updates_fill_alpha_on_existing_variant() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    np.testing.assert_allclose(first.layer.face_color[:, 3], np.zeros(len(first.layer.data)))

    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec, fill=True)

    assert second.layer is first.layer
    assert second.created is False
    assert len(viewer.layers) == 1
    np.testing.assert_allclose(first.layer.face_color[:, 3], np.full(len(first.layer.data), SHAPES_FACE_ALPHA))
    np.testing.assert_allclose(first.layer.edge_color[:, 3], np.ones(len(first.layer.data)))


def test_viewer_adapter_styled_shapes_status_includes_selected_shape_column() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    status = result.layer.get_status([1, 1], dims_displayed=[0, 1])

    assert "index: cell_1" in status["value"]
    assert "cell_type: T" in status["value"]


def test_viewer_adapter_ensure_styled_shapes_loaded_coexists_with_primary_shapes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )

    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")
    styled_result = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert primary_layer is not styled_result.layer
    assert adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global") is primary_layer
    assert adapter.get_loaded_primary_shapes_layer(sdata, "cell_boundaries", "global") is primary_layer
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", style_spec, "global") is styled_result.layer
    assert adapter.get_loaded_styled_shapes_layers(sdata, "cell_boundaries", "global") == [styled_result.layer]
    assert list(viewer.layers) == [primary_layer, styled_result.layer]
    primary_binding = get_shapes_binding(adapter, primary_layer)
    styled_binding = get_shapes_binding(adapter, styled_result.layer)
    assert primary_binding.shapes_role == "primary"
    assert primary_binding.style_spec is None
    assert styled_binding.shapes_role == "styled"
    assert styled_binding.style_spec == style_spec


def test_viewer_adapter_ensure_styled_shapes_loaded_reuses_matching_variant() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec)

    assert first.layer is second.layer
    assert first.created is True
    assert second.created is False
    assert second.value_kind == "continuous"
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_styled_shapes_loaded_creates_distinct_variants_for_different_style_specs() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    categorical_style = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    continuous_style = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="score",
        value_kind="continuous",
    )

    first = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", categorical_style)
    second = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", continuous_style)

    assert first.layer is not second.layer
    assert len(viewer.layers) == 2
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", categorical_style, "global") is first.layer
    assert adapter.get_loaded_styled_shapes_layer(sdata, "cell_boundaries", continuous_style, "global") is second.layer
    assert adapter.get_loaded_styled_shapes_layers(sdata, "cell_boundaries", "global") == [first.layer, second.layer]


def test_viewer_adapter_remove_shapes_layer_removes_only_primary_shapes_layer() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")
    styled_layer = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec).layer

    removed_layer = adapter.remove_shapes_layer(sdata, "cell_boundaries", "global")

    assert removed_layer is primary_layer
    assert list(viewer.layers) == [styled_layer]
    assert adapter.layer_bindings.get_binding(primary_layer) is None
    assert get_shapes_binding(adapter, styled_layer).shapes_role == "styled"


def test_viewer_adapter_remove_layers_outside_coordinate_system_removes_primary_and_styled_shapes() -> None:
    sdata = make_colorable_shapes_sdata()
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    style_spec = ShapeColorSourceSpec(
        source_kind="shape_column",
        value_key="cell_type",
        value_kind="categorical",
    )
    primary_layer = adapter.ensure_shapes_loaded(sdata, "cell_boundaries", "global")
    styled_layer = adapter.ensure_styled_shapes_loaded(sdata, "cell_boundaries", "global", style_spec).layer

    removed_bindings = adapter.remove_layers_outside_coordinate_system(sdata=sdata, coordinate_system="local")

    assert [binding.layer for binding in removed_bindings] == [primary_layer, styled_layer]
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(primary_layer) is None
    assert adapter.layer_bindings.get_binding(styled_layer) is None


def test_viewer_adapter_remove_shapes_layer_removes_registered_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)
    layer = adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "global")

    removed_layer = adapter.remove_shapes_layer(sdata_blobs, "blobs_polygons", "global")

    assert removed_layer is layer
    assert list(viewer.layers) == []
    assert adapter.layer_bindings.get_binding(layer) is None


def test_viewer_adapter_ensure_shapes_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_shapes_loaded(sdata_blobs, "blobs_polygons", "not_a_coordinate_system")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_shapes_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_adds_stack_layer_and_registers_binding(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert layer in viewer.layers
    assert layer.name == "blobs_image"
    assert layer.affine is not None
    assert layer.rgb is False
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.element_name == "blobs_image"
    assert binding.element_type == "image"
    assert binding.coordinate_system == "global"
    assert binding.image_display_mode == "stack"
    assert "image_display_mode" not in layer.metadata


def test_viewer_adapter_ensure_image_loaded_reuses_matching_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    second = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert first is second
    assert len(viewer.layers) == 1


def test_viewer_adapter_ensure_image_loaded_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_multiscale_image", "global", mode="stack")

    assert layer.multiscale is True
    assert len(layer.data) == 3
    assert layer in viewer.layers
    binding = adapter.layer_bindings.get_binding(layer)
    assert binding is not None
    assert binding.image_display_mode == "stack"


def test_viewer_adapter_ensure_image_loaded_rejects_unknown_coordinate_system(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "not_a_coordinate_system", mode="stack")
    except ValueError as error:
        assert "Coordinate system `not_a_coordinate_system`" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject an unknown coordinate system.")


def test_viewer_adapter_ensure_image_loaded_overlay_adds_one_layer_per_selected_channel(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 2],
        channel_colors=["blue", "magenta"],
    )

    assert isinstance(layers, list)
    assert len(layers) == 2
    assert [layer.name for layer in layers] == ["blobs_image[0]", "blobs_image[2]"]
    assert [layer.blending for layer in layers] == ["additive", "additive"]
    assert [binding.channel_index for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [
        0,
        2,
    ]
    assert [binding.channel_name for binding in (adapter.layer_bindings.get_binding(layer) for layer in layers)] == [
        "0",
        "2",
    ]


def test_viewer_adapter_ensure_image_loaded_overlay_reuses_existing_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    first = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    second = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["cyan", "yellow"],
    )

    assert first == second
    assert len(viewer.layers) == 2
    assert [str(layer.colormap).lower() for layer in second] != []


def test_viewer_adapter_unregisters_binding_when_user_removes_overlay_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )

    viewer.layers.remove(layers[0])

    assert layers[0] not in viewer.layers
    assert adapter.layer_bindings.get_binding(layers[0]) is None
    assert adapter.layer_bindings.get_binding(layers[1]) is not None


def test_viewer_adapter_ensure_image_loaded_overlay_removes_existing_stack_layer(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")
    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )

    assert stack_layer not in viewer.layers
    assert adapter.layer_bindings.get_binding(stack_layer) is None
    assert len(overlay_layers) == 2
    assert all(layer in viewer.layers for layer in overlay_layers)


def test_viewer_adapter_ensure_image_loaded_overlay_removes_stale_channel_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[1],
        channel_colors=["green"],
    )

    assert len(layers) == 1
    assert len(viewer.layers) == 1
    binding = adapter.layer_bindings.get_binding(layers[0])
    assert binding is not None
    assert binding.channel_index == 1


def test_viewer_adapter_ensure_image_loaded_stack_removes_existing_overlay_layers(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    overlay_layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=[0, 1],
        channel_colors=["blue", "red"],
    )
    stack_layer = adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="stack")

    assert all(layer not in viewer.layers for layer in overlay_layers)
    assert all(adapter.layer_bindings.get_binding(layer) is None for layer in overlay_layers)
    assert stack_layer in viewer.layers


def test_viewer_adapter_ensure_image_loaded_overlay_accepts_channel_names(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    sdata_blobs.images["blobs_image"] = sdata_blobs.images["blobs_image"].assign_coords(c=["DAPI", "CD3", "CD8"])
    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_image",
        "global",
        mode="overlay",
        channels=["CD3", "CD8"],
        channel_colors=["green", "magenta"],
    )

    assert [layer.name for layer in layers] == ["blobs_image[CD3]", "blobs_image[CD8]"]
    assert [adapter.layer_bindings.get_binding(layer).channel_name for layer in layers] == ["CD3", "CD8"]


def test_viewer_adapter_ensure_image_loaded_overlay_supports_multiscale_images(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    layers = adapter.ensure_image_loaded(
        sdata_blobs,
        "blobs_multiscale_image",
        "global",
        mode="overlay",
        channels=[0, 2],
    )

    assert len(layers) == 2
    assert all(layer.multiscale is True for layer in layers)
    assert all(len(layer.data) == 3 for layer in layers)


def test_viewer_adapter_ensure_image_loaded_overlay_requires_selected_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=None)
    except ValueError as error:
        assert "requires at least one selected channel" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to require selected channels for overlay mode.")


def test_viewer_adapter_ensure_image_loaded_overlay_rejects_duplicate_channels(sdata_blobs) -> None:
    viewer = DummyViewer()
    adapter = ViewerAdapter(viewer)

    try:
        adapter.ensure_image_loaded(sdata_blobs, "blobs_image", "global", mode="overlay", channels=[0, 0])
    except ValueError as error:
        assert "does not accept duplicate channel selections" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ensure_image_loaded to reject duplicate overlay channels.")
