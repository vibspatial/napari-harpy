from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import bermuda
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import Mode
from shapely.geometry import Polygon

import napari_harpy.widgets.shapes_annotation._create_holes as create_holes_module
from napari_harpy._shapes_triangulation import configure_shapes_triangulation_backend
from napari_harpy.core.shapes_geometry import (
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)
from napari_harpy.viewer.shapes_styling import apply_primary_shapes_layer_style


def _copy_layer_data(layer: object) -> list[np.ndarray]:
    return [np.asarray(vertices, dtype=float).copy() for vertices in layer.data]


def _assert_layer_data_unchanged(layer: object, expected_data: list[np.ndarray]) -> None:
    assert len(layer.data) == len(expected_data)
    for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
        np.testing.assert_allclose(np.asarray(actual_vertices, dtype=float), expected_vertices)


def _rgba(color: str) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=float)


def test_create_holes_plan_from_selection_selects_largest_shell_and_encodes_child_holes() -> None:
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_1 = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    child_2 = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_2),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_1),
        ],
        shape_type=["polygon", "polygon", "polygon"],
    )
    layer.selected_data = {2, 0, 1}
    original_data = _copy_layer_data(layer)

    plan = create_holes_module._create_holes_plan_from_selection(layer)

    assert plan.shell_row_index == 1
    assert plan.hole_row_indices == (0, 2)
    planned_polygon = napari_polygon_vertices_to_shapely_polygon(plan.vertices)
    assert planned_polygon.equals(
        Polygon(shell.exterior.coords, holes=[child_2.exterior.coords, child_1.exterior.coords])
    )
    assert len(planned_polygon.interiors) == 2
    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1, 2}


def test_create_holes_plan_from_selection_preserves_existing_shell_holes() -> None:
    existing_hole = [(2, 2), (2, 4), (4, 4), (4, 2)]
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[existing_hole])
    child = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child),
        ],
        shape_type=["polygon", "polygon"],
    )
    layer.selected_data = {0, 1}

    plan = create_holes_module._create_holes_plan_from_selection(layer)

    assert plan.shell_row_index == 0
    assert plan.hole_row_indices == (1,)
    planned_polygon = napari_polygon_vertices_to_shapely_polygon(plan.vertices)
    assert planned_polygon.equals(Polygon(shell.exterior.coords, holes=[existing_hole, child.exterior.coords]))
    assert len(planned_polygon.interiors) == 2


def test_create_holes_plan_from_selection_fails_for_ambiguous_largest_area_without_mutation() -> None:
    shell_1 = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    shell_2 = Polygon([(10, 10), (14, 10), (14, 14), (10, 14)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell_1),
            shapely_polygon_to_napari_polygon_vertices(shell_2),
        ],
        shape_type=["polygon", "polygon"],
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)

    with pytest.raises(ValueError, match="largest selected polygons have equal area"):
        create_holes_module._create_holes_plan_from_selection(layer)

    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1}


@pytest.mark.parametrize("selected_data", [set(), {0}])
def test_create_holes_plan_from_selection_requires_at_least_two_selected_rows(selected_data: set[int]) -> None:
    shell = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    layer = Shapes([shapely_polygon_to_napari_polygon_vertices(shell)], shape_type="polygon")
    layer.selected_data = selected_data

    with pytest.raises(ValueError, match="Select one shell polygon"):
        create_holes_module._create_holes_plan_from_selection(layer)


def test_create_holes_plan_from_selection_rejects_non_polygon_selected_row_without_mutation() -> None:
    shell = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    line_vertices = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    layer = Shapes(
        [shapely_polygon_to_napari_polygon_vertices(shell), line_vertices],
        shape_type=["polygon", "line"],
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)

    with pytest.raises(ValueError, match="requires selected polygon rows"):
        create_holes_module._create_holes_plan_from_selection(layer)

    _assert_layer_data_unchanged(layer, original_data)
    assert set(layer.selected_data) == {0, 1}


def test_apply_create_holes_plan_replaces_shell_removes_children_and_preserves_layer_state() -> None:
    child_before = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_after = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    unselected = Polygon([(20, 20), (20, 22), (22, 22), (22, 20)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_before),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_after),
            shapely_polygon_to_napari_polygon_vertices(unselected),
        ],
        shape_type=["polygon", "polygon", "polygon", "polygon"],
        features=pd.DataFrame(
            {
                "label": ["child_before", "shell", "child_after", "unselected"],
                "source_id": ["row-0", "row-1", "row-2", "row-3"],
            }
        ),
    )
    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1, 2}
    plan = create_holes_module._create_holes_plan_from_selection(layer)

    applied = create_holes_module._apply_create_holes_plan(layer, plan)

    assert applied is True
    assert len(layer.data) == 2
    assert list(layer.shape_type) == ["polygon", "polygon"]
    assert layer.features["label"].tolist() == ["shell", "unselected"]
    assert layer.features["source_id"].tolist() == ["row-1", "row-3"]
    assert layer.mode == Mode.DIRECT
    assert set(layer.selected_data) == {0}

    expected_shell = Polygon(
        shell.exterior.coords,
        holes=[child_before.exterior.coords, child_after.exterior.coords],
    )
    planned_shell = napari_polygon_vertices_to_shapely_polygon(layer.data[0])
    assert planned_shell.equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)


def test_apply_create_holes_plan_preserves_styles_with_stale_napari_color_arrays() -> None:
    child_before = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    shell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    child_after = Polygon([(6, 6), (6, 8), (8, 8), (8, 6)])
    unselected = Polygon([(20, 20), (20, 22), (22, 22), (22, 20)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(child_before),
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child_after),
            shapely_polygon_to_napari_polygon_vertices(unselected),
        ],
        shape_type=["polygon", "polygon", "polygon", "polygon"],
    )
    apply_primary_shapes_layer_style(layer)

    current_edge_color = "#aa00aa"
    current_face_color = "#bbccdd44"
    current_edge_width = 9
    layer.current_edge_color = current_edge_color
    layer.current_face_color = current_face_color
    layer.current_edge_width = current_edge_width

    edge_color = np.asarray(
        [
            _rgba("#ff0000"),
            _rgba("#00ffff"),
            _rgba("#00ff00"),
            _rgba("#123456"),
        ],
        dtype=float,
    )
    face_color = np.asarray(
        [
            _rgba("#111111ff"),
            _rgba("#00000000"),
            _rgba("#222222ff"),
            _rgba("#65432188"),
        ],
        dtype=float,
    )
    edge_width = [2, 3, 4, 5]
    z_index = [10, 11, 12, 13]
    layer.edge_color = edge_color
    layer.face_color = face_color
    layer.edge_width = edge_width
    layer.z_index = z_index
    layer.opacity = 0.37

    # Simulate the intermittent napari state behind the UI bug: the logical
    # data rows are correct, but private color arrays have stale extra rows.
    layer._data_view._edge_color = np.vstack([layer._data_view._edge_color, _rgba("#ffff00")])
    layer._data_view._face_color = np.vstack([layer._data_view._face_color, _rgba("#ffff00")])

    layer.mode = Mode.DIRECT
    layer.selected_data = {0, 1, 2}
    plan = create_holes_module._create_holes_plan_from_selection(layer)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        applied = create_holes_module._apply_create_holes_plan(layer, plan)

    assert applied is True
    style_warnings = [
        str(warning.message)
        for warning in caught_warnings
        if "edge_color" in str(warning.message) or "face_color" in str(warning.message)
    ]
    assert style_warnings == []
    assert set(layer.selected_data) == {0}
    assert layer.mode == Mode.DIRECT
    assert layer.opacity == 0.37
    assert layer.edge_width == [3, 5]
    assert layer.z_index == [11, 13]
    np.testing.assert_allclose(layer.edge_color, edge_color[[1, 3]])
    np.testing.assert_allclose(layer.face_color, face_color[[1, 3]])
    np.testing.assert_allclose(to_rgba(layer.current_edge_color), to_rgba(current_edge_color))
    np.testing.assert_allclose(to_rgba(layer.current_face_color), to_rgba(current_face_color))
    assert layer.current_edge_width == current_edge_width

    expected_shell = Polygon(
        shell.exterior.coords,
        holes=[child_before.exterior.coords, child_after.exterior.coords],
    )
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[0]).equals(expected_shell)
    assert napari_polygon_vertices_to_shapely_polygon(layer.data[1]).equals(unselected)


def test_create_holes_plan_rejects_self_inconsistent_indices() -> None:
    shell = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])

    with pytest.raises(ValueError, match="cannot remove the shell row"):
        create_holes_module._CreateHolesShapesLayerPlan(
            shell_row_index=0,
            hole_row_indices=(0,),
            vertices=shapely_polygon_to_napari_polygon_vertices(shell),
        )


def test_apply_create_holes_plan_rejects_invalid_plan_before_mutation() -> None:
    shell = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    child = Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])
    layer = Shapes(
        [
            shapely_polygon_to_napari_polygon_vertices(shell),
            shapely_polygon_to_napari_polygon_vertices(child),
        ],
        shape_type=["polygon", "polygon"],
        features=pd.DataFrame({"label": ["shell", "child"]}),
    )
    layer.selected_data = {0, 1}
    original_data = _copy_layer_data(layer)
    original_features = layer.features.copy()
    invalid_plan = create_holes_module._CreateHolesShapesLayerPlan(
        shell_row_index=0,
        hole_row_indices=(2,),
        vertices=shapely_polygon_to_napari_polygon_vertices(shell),
    )

    with pytest.raises(ValueError, match="hole row is no longer present"):
        create_holes_module._apply_create_holes_plan(layer, invalid_plan)

    _assert_layer_data_unchanged(layer, original_data)
    pd.testing.assert_frame_equal(layer.features, original_features)
    assert set(layer.selected_data) == {0, 1}


class TestCreateHolesTriangulationFailure:
    """Characterize the Bermuda failure and specify Harpy's rollback contract.

    The checked-in fixture is the exact 570-by-2 ``float32`` napari polygon
    path captured from the failed ``Create holes`` operation. Its exterior and
    two holes render independently with Bermuda 0.1.7, but their valid combined
    hole encoding causes Bermuda's face triangulator to panic. Napari converts
    that panic into ``RuntimeError`` after replacing the live layer's private
    shape view, which would leave the annotation layer empty without Harpy's
    transaction rollback.

    The first test retains the real upstream reproducer while Bermuda is
    affected. The second test injects the same failure boundary only for the
    combined candidate, so Harpy's full-layer rollback remains covered after
    Bermuda fixes this particular input. Both verify that the Create-holes
    transaction consumes the application error after restoring every captured
    layer property.
    """

    _FIXTURE_PATH = Path(__file__).parent / "fixtures" / "create_holes_triangulation_failure.txt"
    _DATA_SHA256 = "21df8806d99580c208b26e075297fdbefeac4ce8f134ac370328203af55975ec"

    @classmethod
    def _load_vertices(cls) -> np.ndarray:
        vertices = np.loadtxt(cls._FIXTURE_PATH, dtype=np.float32)
        assert vertices.shape == (570, 2)
        assert hashlib.sha256(vertices.tobytes()).hexdigest() == cls._DATA_SHA256
        return vertices

    @classmethod
    def _make_layer(cls) -> Shapes:
        failed_vertices = cls._load_vertices()
        shell = failed_vertices[:546].copy()
        hole_1 = failed_vertices[546:557].copy()
        hole_2 = failed_vertices[558:569].copy()
        unselected = shapely_polygon_to_napari_polygon_vertices(
            Polygon([(40000, 40000), (40000, 40100), (40100, 40100), (40100, 40000)])
        ).astype(np.float32)
        layer = Shapes(
            [hole_1, unselected, shell, hole_2],
            shape_type=["polygon", "polygon", "polygon", "polygon"],
            features=pd.DataFrame(
                {
                    "label": ["hole-1", "unselected", "shell", "hole-2"],
                    "source_id": ["row-0", "row-1", "row-2", "row-3"],
                }
            ),
        )
        apply_primary_shapes_layer_style(layer)
        layer.feature_defaults = pd.DataFrame(
            {
                "label": ["next-polygon"],
                "source_id": [pd.NA],
            }
        )
        layer.edge_color = np.asarray([_rgba("#112233"), _rgba("#445566"), _rgba("#778899"), _rgba("#aabbcc")])
        layer.face_color = np.asarray([_rgba("#01020344"), _rgba("#05060744"), _rgba("#090a0b44"), _rgba("#0c0d0e44")])
        layer.edge_width = [2, 4, 6, 8]
        layer.z_index = [3, 5, 7, 9]
        layer.opacity = 0.42
        layer.current_edge_color = "#abcdef"
        layer.current_face_color = "#12345678"
        layer.current_edge_width = 11
        layer.mode = Mode.DIRECT
        layer.selected_data = {0, 2, 3}
        return layer

    @staticmethod
    def _capture_complete_layer_state(layer: Shapes) -> dict[str, object]:
        return {
            "data": tuple(np.asarray(vertices).copy() for vertices in layer.data),
            "shape_type": tuple(layer.shape_type),
            "features": layer.features.copy(deep=True),
            "feature_defaults": layer.feature_defaults.copy(deep=True),
            "edge_color": np.asarray(layer.edge_color).copy(),
            "face_color": np.asarray(layer.face_color).copy(),
            "edge_width": tuple(layer.edge_width),
            "z_index": tuple(layer.z_index),
            "opacity": layer.opacity,
            "current_edge_color": np.asarray(to_rgba(layer.current_edge_color)),
            "current_face_color": np.asarray(to_rgba(layer.current_face_color)),
            "current_edge_width": layer.current_edge_width,
            "mode": layer.mode,
            "selected_data": frozenset(layer.selected_data),
        }

    @staticmethod
    def _assert_complete_layer_state_restored(layer: Shapes, baseline: dict[str, object]) -> None:
        expected_data = baseline["data"]
        assert isinstance(expected_data, tuple)
        assert len(layer.data) == len(expected_data)
        for actual_vertices, expected_vertices in zip(layer.data, expected_data, strict=True):
            np.testing.assert_array_equal(actual_vertices, expected_vertices)
        assert tuple(layer.shape_type) == baseline["shape_type"]
        pd.testing.assert_frame_equal(layer.features, baseline["features"])
        pd.testing.assert_frame_equal(layer.feature_defaults, baseline["feature_defaults"])
        np.testing.assert_allclose(layer.edge_color, baseline["edge_color"])
        np.testing.assert_allclose(layer.face_color, baseline["face_color"])
        assert tuple(layer.edge_width) == baseline["edge_width"]
        assert tuple(layer.z_index) == baseline["z_index"]
        assert layer.opacity == baseline["opacity"]
        np.testing.assert_allclose(to_rgba(layer.current_edge_color), baseline["current_edge_color"])
        np.testing.assert_allclose(to_rgba(layer.current_face_color), baseline["current_face_color"])
        assert layer.current_edge_width == baseline["current_edge_width"]
        assert layer.mode == baseline["mode"]
        assert frozenset(layer.selected_data) == baseline["selected_data"]
        assert len(layer._data_view.shapes) == len(expected_data)

    def test_full_geometry_bermuda_failure_restores_complete_layer(
        self,
        restore_triangulation_backend: None,
    ) -> None:
        configure_shapes_triangulation_backend("bermuda")
        layer = self._make_layer()
        baseline = self._capture_complete_layer_state(layer)
        plan = create_holes_module._create_holes_plan_from_selection(layer)
        failed_vertices = self._load_vertices()
        np.testing.assert_array_equal(np.asarray(plan.vertices, dtype=np.float32), failed_vertices)

        # Bermuda 0.1.7 panics for this exact valid candidate and napari wraps
        # the panic in RuntimeError. The upstream failure is tracked at
        # https://github.com/napari/bermuda/issues/194. The Create-holes
        # transaction consumes that application error after restoring the
        # baseline. When Bermuda fixes this input, this fixture-specific
        # rollback is no longer necessary; the artificial-failure test below
        # remains the permanent contract.
        applied = create_holes_module._apply_create_holes_plan(layer, plan)

        assert applied is False
        self._assert_complete_layer_state_restored(layer, baseline)

    def test_artificial_render_failure_restores_complete_layer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        restore_triangulation_backend: None,
    ) -> None:
        configure_shapes_triangulation_backend("bermuda")
        layer = self._make_layer()
        baseline = self._capture_complete_layer_state(layer)
        plan = create_holes_module._create_holes_plan_from_selection(layer)
        expected_candidate = np.asarray(plan.vertices, dtype=np.float32)
        real_triangulate = bermuda.triangulate_polygons_with_edge
        failure_count = 0

        def fail_exact_candidate(polygons: list[np.ndarray]) -> object:
            nonlocal failure_count
            if np.array_equal(polygons[0], expected_candidate):
                failure_count += 1
                raise RuntimeError("synthetic Bermuda failure for create-holes rollback test")
            return real_triangulate(polygons)

        # Reject only the combined candidate. Baseline construction and
        # rollback continue through real Bermuda, proving that recovery rebuilds
        # renderable source rows instead of patching public Python attributes.
        monkeypatch.setattr(bermuda, "triangulate_polygons_with_edge", fail_exact_candidate)

        applied = create_holes_module._apply_create_holes_plan(layer, plan)

        assert applied is False
        self._assert_complete_layer_state_restored(layer, baseline)

        # A different, valid Create-holes edit still commits through Bermuda
        # immediately after rollback, proving that the live layer remains
        # usable rather than only superficially matching public attributes.
        layer.selected_data = {2, 3}
        follow_up_plan = create_holes_module._create_holes_plan_from_selection(layer)
        assert create_holes_module._apply_create_holes_plan(layer, follow_up_plan) is True
        assert failure_count == 1
        assert len(layer.data) == 3

    def test_raises_exception_group_when_application_and_rollback_both_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
        restore_triangulation_backend: None,
    ) -> None:
        configure_shapes_triangulation_backend("bermuda")
        layer = self._make_layer()
        plan = create_holes_module._create_holes_plan_from_selection(layer)
        application_error = RuntimeError("simulated Create-holes application failure")
        restoration_error = RuntimeError("simulated Create-holes baseline restoration failure")

        def fail_application() -> None:
            raise application_error

        def fail_restoration(bound_layer: Shapes, baseline: object) -> None:
            assert bound_layer is layer
            assert baseline is not None
            raise restoration_error

        monkeypatch.setattr(
            create_holes_module,
            "ensure_shapes_triangulation_backend",
            fail_application,
        )
        monkeypatch.setattr(
            create_holes_module,
            "_restore_shapes_layer_baseline",
            fail_restoration,
        )

        with pytest.raises(ExceptionGroup) as caught:
            create_holes_module._apply_create_holes_plan(layer, plan)

        assert caught.value.exceptions == (application_error, restoration_error)
        assert caught.value.__cause__ is application_error
