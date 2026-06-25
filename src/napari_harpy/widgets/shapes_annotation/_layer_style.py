from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from napari.layers import Shapes


@dataclass(frozen=True)
class _ShapesLayerStyleSnapshot:
    edge_color: np.ndarray
    face_color: np.ndarray
    edge_width: tuple[object, ...]
    z_index: tuple[object, ...]
    opacity: float
    current_edge_color: object
    current_face_color: object
    current_edge_width: object


def _capture_shapes_layer_style(layer: Shapes, *, row_count: int) -> _ShapesLayerStyleSnapshot:
    return _ShapesLayerStyleSnapshot(
        edge_color=_row_aligned_color_array(layer.edge_color, row_count=row_count, name="edge_color"),
        face_color=_row_aligned_color_array(layer.face_color, row_count=row_count, name="face_color"),
        edge_width=_row_aligned_sequence(layer.edge_width, row_count=row_count, name="edge_width"),
        z_index=_row_aligned_sequence(layer.z_index, row_count=row_count, name="z_index"),
        opacity=layer.opacity,
        current_edge_color=layer.current_edge_color,
        current_face_color=layer.current_face_color,
        current_edge_width=layer.current_edge_width,
    )


def _trim_stale_private_color_rows_before_rebuild(layer: Shapes, snapshot: _ShapesLayerStyleSnapshot) -> None:
    # Trim stale private napari color rows before the data setter rebuilds the
    # private shape-data cache. The current `layer.data` row count must match
    # the private color row count, otherwise napari falls back to default filled
    # polygon colors.
    # Example stale state: `len(layer.data) == 4`, while
    # `layer._data_view._edge_color.shape == (5, 4)` and
    # `layer._data_view._face_color.shape == (5, 4)`.
    #
    # Napari's public color setter updates existing private shape rows but does
    # not shrink stale private color arrays. Trim those arrays directly
    # immediately before `layer.data = ...`; napari's data setter reads them,
    # then replaces the private shape-data cache during rebuild.
    layer._data_view._edge_color = snapshot.edge_color.copy()
    layer._data_view._face_color = snapshot.face_color.copy()
    layer.edge_width = list(snapshot.edge_width)
    layer.z_index = list(snapshot.z_index)


def _restore_shapes_layer_current_style(layer: Shapes, snapshot: _ShapesLayerStyleSnapshot) -> None:
    with layer.block_update_properties():
        with layer.events.current_edge_color.blocker():
            layer.current_edge_color = snapshot.current_edge_color
        with layer.events.current_face_color.blocker():
            layer.current_face_color = snapshot.current_face_color
        with layer.events.edge_width.blocker():
            layer.current_edge_width = snapshot.current_edge_width


def _restore_shapes_layer_row_styles(
    layer: Shapes,
    snapshot: _ShapesLayerStyleSnapshot,
    *,
    row_indices: Sequence[int],
) -> None:
    row_indices_list = list(row_indices)
    layer.z_index = [snapshot.z_index[index] for index in row_indices_list]
    layer.edge_color = snapshot.edge_color[row_indices_list]
    layer.face_color = snapshot.face_color[row_indices_list]
    layer.edge_width = [snapshot.edge_width[index] for index in row_indices_list]


def _row_aligned_color_array(colors: object, *, row_count: int, name: str) -> np.ndarray:
    color_array = np.asarray(colors, dtype=float)
    if color_array.ndim != 2 or color_array.shape[1] != 4 or len(color_array) < row_count:
        raise ValueError(f"Shapes layer {name} must contain one RGBA row for each shape row.")
    return color_array[:row_count].copy()


def _row_aligned_sequence(values: object, *, row_count: int, name: str) -> tuple[object, ...]:
    try:
        sequence = tuple(values)
    except TypeError as error:
        raise ValueError(f"Shapes layer {name} must contain one value for each shape row.") from error
    if len(sequence) < row_count:
        raise ValueError(f"Shapes layer {name} must contain one value for each shape row.")
    return sequence[:row_count]
