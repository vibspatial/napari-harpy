from __future__ import annotations

import numpy as np
import pandas as pd
from napari.layers import Shapes

from napari_harpy.widgets.shapes_annotation._snapshot import (
    _annotation_layer_snapshots_equal,
    _capture_annotation_layer_snapshot,
)


def _polygon_data(dtype: type[np.floating] = np.float64, offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            [offset + 0.0, 0.0],
            [offset + 0.0, 2.0],
            [offset + 2.0, 2.0],
            [offset + 2.0, 0.0],
        ],
        dtype=dtype,
    )


def _make_layer(*, dtype: type[np.floating] = np.float64) -> Shapes:
    layer = Shapes([_polygon_data(dtype=dtype)], shape_type="polygon")
    layer.features = pd.DataFrame({"instance_id": ["cell_1"]}, index=[42])
    return layer


def test_capture_annotation_layer_snapshot_handles_empty_layers() -> None:
    layer = Shapes(ndim=2)

    snapshot = _capture_annotation_layer_snapshot(layer)

    assert snapshot.row_count == 0
    assert snapshot.features.empty
    assert _annotation_layer_snapshots_equal(snapshot, _capture_annotation_layer_snapshot(layer))


def test_capture_annotation_layer_snapshot_is_stable_for_unchanged_layers() -> None:
    layer = _make_layer()

    first = _capture_annotation_layer_snapshot(layer)
    second = _capture_annotation_layer_snapshot(layer)

    assert _annotation_layer_snapshots_equal(first, second)


def test_capture_annotation_layer_snapshot_normalizes_feature_row_index() -> None:
    layer = _make_layer()

    snapshot = _capture_annotation_layer_snapshot(layer)

    assert snapshot.features.index.tolist() == [0]
    assert snapshot.features["instance_id"].tolist() == ["cell_1"]


def test_capture_annotation_layer_snapshot_normalizes_float_geometry_dtype() -> None:
    float32_layer = _make_layer(dtype=np.float32)
    float64_layer = _make_layer(dtype=np.float64)

    float32_snapshot = _capture_annotation_layer_snapshot(float32_layer)
    float64_snapshot = _capture_annotation_layer_snapshot(float64_layer)

    assert float32_snapshot.geometry_hash == float64_snapshot.geometry_hash


def test_capture_annotation_layer_snapshot_detects_vertex_edits() -> None:
    layer = _make_layer()
    clean = _capture_annotation_layer_snapshot(layer)

    layer.data[0][0, 0] = 0.25
    edited = _capture_annotation_layer_snapshot(layer)

    assert clean.geometry_hash != edited.geometry_hash
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_detects_shape_type_changes() -> None:
    polygon_layer = Shapes([_polygon_data()], shape_type="polygon")
    path_layer = Shapes([_polygon_data()], shape_type="path")

    polygon_snapshot = _capture_annotation_layer_snapshot(polygon_layer)
    path_snapshot = _capture_annotation_layer_snapshot(path_layer)

    assert polygon_snapshot.geometry_hash != path_snapshot.geometry_hash
    assert not _annotation_layer_snapshots_equal(polygon_snapshot, path_snapshot)


def test_capture_annotation_layer_snapshot_detects_added_rows() -> None:
    layer = _make_layer()
    clean = _capture_annotation_layer_snapshot(layer)

    layer.add_polygons(_polygon_data(offset=10.0))
    layer.features["instance_id"] = ["cell_1", "cell_2"]
    edited = _capture_annotation_layer_snapshot(layer)

    assert edited.row_count == clean.row_count + 1
    assert edited.geometry_hash != clean.geometry_hash
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_detects_feature_value_changes() -> None:
    layer = _make_layer()
    clean = _capture_annotation_layer_snapshot(layer)

    layer.features["instance_id"] = ["cell_2"]
    edited = _capture_annotation_layer_snapshot(layer)

    assert clean.geometry_hash == edited.geometry_hash
    assert not clean.features.equals(edited.features)
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_detects_feature_column_order_changes() -> None:
    layer = _make_layer()
    layer.features = pd.DataFrame({"instance_id": ["cell_1"], "score": [1]})
    clean = _capture_annotation_layer_snapshot(layer)

    layer.features = layer.features[["score", "instance_id"]]
    edited = _capture_annotation_layer_snapshot(layer)

    assert not clean.features.equals(edited.features)
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_detects_feature_dtype_changes() -> None:
    layer = _make_layer()
    layer.features = pd.DataFrame({"instance_id": ["cell_1"], "score": pd.Series([1], dtype="int64")})
    clean = _capture_annotation_layer_snapshot(layer)

    layer.features = pd.DataFrame({"instance_id": ["cell_1"], "score": pd.Series([1.0], dtype="float64")})
    edited = _capture_annotation_layer_snapshot(layer)

    assert not clean.features.equals(edited.features)
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_detects_feature_missing_value_changes() -> None:
    layer = _make_layer()
    layer.features = pd.DataFrame({"instance_id": ["cell_1"], "label": ["region"]})
    clean = _capture_annotation_layer_snapshot(layer)

    layer.features = pd.DataFrame({"instance_id": ["cell_1"], "label": [pd.NA]})
    edited = _capture_annotation_layer_snapshot(layer)

    assert not clean.features.equals(edited.features)
    assert not _annotation_layer_snapshots_equal(clean, edited)


def test_capture_annotation_layer_snapshot_ignores_visual_only_changes() -> None:
    layer = _make_layer()
    clean = _capture_annotation_layer_snapshot(layer)

    layer.opacity = 0.25
    layer.current_edge_color = "red"
    layer.current_face_color = "yellow"
    layer.current_edge_width = 9
    edited = _capture_annotation_layer_snapshot(layer)

    assert _annotation_layer_snapshots_equal(clean, edited)
