from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from napari.layers import Shapes

from napari_harpy.core.shapes_geometry import (
    create_polygon_with_direct_holes,
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
)
from napari_harpy.widgets.shapes_annotation._layer_style import (
    _capture_shapes_layer_style,
    _restore_shapes_layer_current_style,
    _restore_shapes_layer_row_styles,
    _trim_stale_private_color_rows_before_rebuild,
)


@dataclass(frozen=True)
class _CreateHolesShapesLayerPlan:
    """Planned create-holes mutation for one selected napari Shapes layer.

    Attributes
    ----------
    shell_row_index
        Current napari row in ``layer.data`` that survives the operation and
        receives the new hole-bearing vertices. This is a napari layer row
        index, not a source GeoDataFrame index value.
    hole_row_indices
        Current napari rows selected as child polygons. Their exterior rings
        become new holes, and the rows are removed after the shell row is
        replaced.
    vertices
        Napari ``(y, x)`` vertex row encoding of the new hole-bearing shell
        polygon. This array is the replacement data for
        ``layer.data[shell_row_index]``.
    """

    shell_row_index: int
    hole_row_indices: tuple[int, ...]
    vertices: np.ndarray

    def __post_init__(self) -> None:
        shell_row_index = _coerce_plan_row_index(self.shell_row_index, row_kind="shell")
        hole_row_indices = tuple(_coerce_plan_row_index(index, row_kind="hole") for index in self.hole_row_indices)
        if not hole_row_indices:
            raise ValueError("Create-holes plan must include at least one hole row.")
        if len(set(hole_row_indices)) != len(hole_row_indices):
            raise ValueError("Create-holes plan hole rows must be unique.")
        if shell_row_index in hole_row_indices:
            raise ValueError("Create-holes plan cannot remove the shell row.")

        try:
            vertices = np.asarray(self.vertices, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Create-holes plan vertices must be a numeric napari vertex array.") from error
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
            raise ValueError("Create-holes plan vertices must be a two-dimensional napari vertex array.")

        object.__setattr__(self, "shell_row_index", shell_row_index)
        object.__setattr__(self, "hole_row_indices", hole_row_indices)
        object.__setattr__(self, "vertices", vertices.copy())


def _create_holes_plan_from_selection(layer: Shapes) -> _CreateHolesShapesLayerPlan:
    """Build a mutation-ready plan from selected napari shape rows.

    Napari rows provide the editable ``(y, x)`` vertices. The planner decodes
    selected rows to Shapely polygons so shell inference and direct-hole
    topology validation can use geometry APIs, then encodes the resulting
    polygon-with-holes back to one napari vertex row for later layer mutation.
    """
    selected_rows = _selected_shape_rows(layer)
    if len(selected_rows) < 2:
        raise ValueError("Select one shell polygon and one or more polygons fully inside it.")

    polygons_by_row = {}
    for row_index in selected_rows:
        if _shape_type_at(layer, row_index) != "polygon":
            raise ValueError("Create holes requires selected polygon rows.")
        try:
            polygons_by_row[row_index] = napari_polygon_vertices_to_shapely_polygon(layer.data[row_index])
        except ValueError as error:
            raise ValueError(f"Selected shape row {row_index} cannot be converted to a valid polygon.") from error

    max_area = max(polygon.area for polygon in polygons_by_row.values())
    shell_row_indices = tuple(row_index for row_index, polygon in polygons_by_row.items() if polygon.area == max_area)
    if len(shell_row_indices) != 1:
        raise ValueError("Select one unambiguous shell polygon; the largest selected polygons have equal area.")

    shell_row_index = shell_row_indices[0]
    hole_row_indices = tuple(row_index for row_index in selected_rows if row_index != shell_row_index)
    polygon = create_polygon_with_direct_holes(
        polygons_by_row[shell_row_index],
        [polygons_by_row[row_index] for row_index in hole_row_indices],
    )

    return _CreateHolesShapesLayerPlan(
        shell_row_index=shell_row_index,
        hole_row_indices=hole_row_indices,
        vertices=shapely_polygon_to_napari_polygon_vertices(polygon),
    )


def _apply_create_holes_plan(layer: Shapes, plan: _CreateHolesShapesLayerPlan) -> None:
    row_count = len(layer.data)
    shell_row_index = plan.shell_row_index
    hole_row_indices = tuple(sorted(plan.hole_row_indices))
    if shell_row_index >= row_count:
        raise ValueError("Create-holes plan shell row is no longer present in the layer.")
    if any(row_index >= row_count for row_index in hole_row_indices):
        raise ValueError("Create-holes plan hole row is no longer present in the layer.")

    current_mode = layer.mode
    style_snapshot = _capture_shapes_layer_style(layer, row_count=row_count)
    surviving_row_indices = tuple(row_index for row_index in range(row_count) if row_index not in hole_row_indices)

    rebuilt_data = list(layer.data)
    rebuilt_data[shell_row_index] = np.asarray(plan.vertices, dtype=float).copy()

    _trim_stale_private_color_rows_before_rebuild(layer, style_snapshot)

    # Assign through `layer.data`, not `_data_view.edit(...)`: create-holes can
    # change the shell row's vertex count, and the public setter rebuilds
    # napari's private rendering and hit-testing cache.
    layer.data = rebuilt_data
    layer.remove(list(hole_row_indices))
    layer.opacity = style_snapshot.opacity

    new_shell_row_index = shell_row_index - sum(row_index < shell_row_index for row_index in hole_row_indices)
    layer.mode = current_mode
    layer.selected_data = {new_shell_row_index}

    # The annotation layer connects current edge color/width changes back to all
    # row styles. Restore draw defaults without emitting those sync callbacks,
    # after selecting the final shell row and before reapplying final row styles.
    _restore_shapes_layer_current_style(layer, style_snapshot)

    # Reapply surviving row styles last. Selecting the final shell row and
    # restoring current draw defaults can touch selected row styling through
    # napari/Harpy callbacks; the final row-aligned styles should win.
    _restore_shapes_layer_row_styles(layer, style_snapshot, row_indices=surviving_row_indices)
    # Public napari layer APIs above emit the data/selection events that drive
    # viewer refresh; keep explicit refreshes for low-level edit paths.


def _coerce_plan_row_index(index: object, *, row_kind: str) -> int:
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        raise ValueError(f"Create-holes plan {row_kind} row must be a napari row index.")
    row_index = int(index)
    if row_index < 0:
        raise ValueError(f"Create-holes plan {row_kind} row must be a napari row index.")
    return row_index


def _selected_shape_rows(layer: Shapes) -> tuple[int, ...]:
    selected_rows: list[int] = []
    for index in layer.selected_data:
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
            raise ValueError("Selected shapes must be current napari row indices.")
        row_index = int(index)
        if row_index < 0 or row_index >= len(layer.data):
            raise ValueError("Selected shape row is no longer present in the layer.")
        selected_rows.append(row_index)
    return tuple(sorted(selected_rows))


def _shape_type_at(layer: Shapes, row_index: int) -> object:
    try:
        return layer.shape_type[row_index]
    except (IndexError, TypeError):
        return None

