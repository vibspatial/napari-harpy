from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from napari.layers import Shapes

from napari_harpy.core.shapes_geometry import (
    create_polygon_with_direct_holes,
    napari_polygon_vertices_to_shapely_polygon,
    shapely_polygon_to_napari_polygon_vertices,
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
