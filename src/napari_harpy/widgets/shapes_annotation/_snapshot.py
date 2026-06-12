from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from napari.layers import Shapes


@dataclass(frozen=True)
class _ShapesAnnotationLayerSnapshot:
    """Save-relevant fingerprint of one napari Shapes annotation layer.

    Attributes
    ----------
    row_count
        Number of napari shape rows captured in the snapshot.
    geometry_hash
        Hash of the ordered napari shape types and vertex arrays. Geometry can
        be large, so this avoids storing a full copy of ``layer.data`` in the
        snapshot.
    features
        Normalized copy of ``layer.features``. Features are small row metadata
        in this workflow, so keeping the DataFrame makes equality checks easier
        to inspect than a second hash.
    """

    row_count: int
    geometry_hash: str
    features: pd.DataFrame


def _capture_annotation_layer_snapshot(layer: Shapes) -> _ShapesAnnotationLayerSnapshot:
    data = list(layer.data)
    shape_types = tuple(str(shape_type).lower() for shape_type in layer.shape_type)
    if len(shape_types) != len(data):
        raise ValueError("Napari shapes layer has inconsistent data and shape-type lengths.")

    row_count = len(data)
    return _ShapesAnnotationLayerSnapshot(
        row_count=row_count,
        geometry_hash=_geometry_hash(data, shape_types),
        features=_normalized_features(layer.features, row_count=row_count),
    )


def _annotation_layer_snapshots_equal(
    left: _ShapesAnnotationLayerSnapshot,
    right: _ShapesAnnotationLayerSnapshot,
) -> bool:
    return (
        left.row_count == right.row_count
        and left.geometry_hash == right.geometry_hash
        and left.features.equals(right.features)
    )


def _geometry_hash(data: list[np.ndarray], shape_types: tuple[str, ...]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"napari-harpy-shapes-annotation-geometry-v1\0")
    for vertices, shape_type in zip(data, shape_types, strict=True):
        hasher.update(b"row\0shape:")
        hasher.update(shape_type.encode("utf-8"))
        hasher.update(b"\0")
        _update_array_hash(hasher, np.asarray(vertices))
    return hasher.hexdigest()


def _update_array_hash(hasher: hashlib._Hash, array: np.ndarray) -> None:
    normalized = _normalize_geometry_array(array)
    hasher.update(b"array-shape:")
    hasher.update(repr(normalized.shape).encode("utf-8"))
    hasher.update(b"\0dtype:")
    hasher.update(str(normalized.dtype).encode("utf-8"))
    hasher.update(b"\0values:")
    hasher.update(np.ascontiguousarray(normalized).tobytes())
    hasher.update(b"\0")


def _normalize_geometry_array(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float64, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    return array.astype(np.float64, copy=False)


def _normalized_features(features: pd.DataFrame, *, row_count: int) -> pd.DataFrame:
    return features.copy().reindex(range(row_count)).reset_index(drop=True)
