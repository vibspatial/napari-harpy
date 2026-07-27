"""Capture and restore row-changing annotation state for a napari Shapes layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from napari.layers import Shapes

from napari_harpy._shapes_triangulation import ensure_shapes_triangulation_backend
from napari_harpy.widgets.shapes_annotation._layer_style import (
    _capture_shapes_layer_style,
    _restore_shapes_layer_current_style,
    _restore_shapes_layer_row_styles,
    _ShapesLayerStyleSnapshot,
)


@dataclass(frozen=True)
class _ShapesLayerBaseline:
    """Capture Shapes state needed to recover a failed row-changing edit.

    Row-changing edits can alter vertex counts or remove complete shape rows,
    so recovery must restore the full row-aligned layer state and rebuild
    napari's derived shape and vertex caches rather than restore only the
    affected vertices.
    """

    data: tuple[np.ndarray, ...]
    shape_types: tuple[str, ...]
    features: pd.DataFrame
    feature_defaults: pd.DataFrame
    style: _ShapesLayerStyleSnapshot
    selected_data: frozenset[int]
    mode: str


def _capture_shapes_layer_baseline(layer: Shapes) -> _ShapesLayerBaseline:
    """Return independent copies of the Shapes state required for recovery."""
    return _ShapesLayerBaseline(
        data=tuple(np.array(vertices, copy=True) for vertices in layer.data),
        shape_types=tuple(layer.shape_type),
        features=layer.features.copy(deep=True),
        feature_defaults=layer.feature_defaults.copy(deep=True),
        style=_capture_shapes_layer_style(layer, row_count=len(layer.data)),
        selected_data=frozenset(layer.selected_data),
        mode=layer.mode,
    )


def _restore_shapes_layer_baseline(
    layer: Shapes,
    baseline: _ShapesLayerBaseline,
) -> None:
    """Restore a failed row-changing edit through the public data path."""
    restored_data = [
        (np.array(vertices, copy=True), shape_type)
        for vertices, shape_type in zip(baseline.data, baseline.shape_types, strict=True)
    ]

    # Reassigning the public rows rebuilds napari's derived shape and vertex
    # caches. Restoring `features` then resets napari's defaults from a feature
    # row, so restore the separately captured defaults afterwards.
    #
    # Block the public setters' intermediate data and features events so
    # mechanical recovery is not reported as another edit or as a successful
    # completion. Guarded insertion/deletion have already emitted CHANGING, so
    # their visible transaction remains: CHANGING -> commit fails -> baseline
    # is restored silently -> no CHANGED.
    with layer.events.data.blocker(), layer.events.features.blocker():
        ensure_shapes_triangulation_backend()
        layer.data = restored_data
        layer.features = baseline.features.copy(deep=True)

    layer.mode = baseline.mode
    layer.selected_data = set(baseline.selected_data)
    layer.feature_defaults = baseline.feature_defaults.copy(deep=True)
    layer.opacity = baseline.style.opacity
    _restore_shapes_layer_current_style(layer, baseline.style)
    _restore_shapes_layer_row_styles(
        layer,
        baseline.style,
        row_indices=range(len(baseline.data)),
    )
    layer.refresh()
