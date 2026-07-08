from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy.viewer._styling import MISSING_CATEGORICAL_COLOR
from napari_harpy.viewer.labels_colormap import (
    compact_categorical_labels_mapping_from_values,
    direct_label_colormap_from_rgba,
)


def _rgba(red: float, green: float, blue: float, alpha: float) -> np.ndarray:
    return np.asarray([red, green, blue, alpha], dtype=np.float64)


def test_compact_categorical_labels_mapping_preserves_current_missing_semantics() -> None:
    values = pd.Series(
        pd.Categorical(["T", None, "unknown", "B"], categories=["T", "B", "unknown"]),
        index=pd.Index([9, 3, 7, 5], name="index"),
    )

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["T", "B"],
        palette=["#ff0000", "#00ff00"],
    )

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([3, 5, 7, 9], dtype=np.int64))
    assert mapping.default_texture_code == 0
    assert mapping.background_texture_code == 1
    np.testing.assert_allclose(mapping.texture_rgba[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(mapping.texture_rgba[1], np.zeros(4, dtype=np.float32))

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[9] != mapping.default_texture_code
    assert code_by_label[5] != mapping.default_texture_code
    assert mapping.missing_texture_code is not None
    assert code_by_label[3] == mapping.missing_texture_code
    assert code_by_label[7] == mapping.missing_texture_code
    np.testing.assert_allclose(mapping.texture_rgba[mapping.missing_texture_code], to_rgba(MISSING_CATEGORICAL_COLOR))


def test_compact_categorical_labels_mapping_reuses_repeated_rgba_texture_codes() -> None:
    values = pd.Series(["a", "b", "c"], index=pd.Index([1, 2, 3], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b", "c"],
        palette=["#ff0000", "#00ff00", "#ff0000"],
    )

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[1] == code_by_label[3]
    assert code_by_label[2] != code_by_label[1]
    assert len(mapping.texture_rgba) == 4  # default, background, red, green
    assert set(mapping.texture_codes.tolist()) == {2, 3}


def test_compact_categorical_labels_mapping_accepts_unused_categories() -> None:
    """Unused categories have RGBA rows but are not used by texture codes."""
    values = pd.Series(
        pd.Categorical(["T", "B", "T"], categories=["T", "B", "unused"]),
        index=pd.Index([10, 20, 30], name="index"),
    )

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["T", "B", "unused"],
        palette=["#ff0000", "#00ff00", "#0000ff"],
    )

    code_by_label = dict(zip(mapping.label_ids.tolist(), mapping.texture_codes.tolist(), strict=True))
    assert code_by_label[10] == code_by_label[30]
    assert code_by_label[20] != code_by_label[10]
    assert mapping.missing_texture_code is None
    assert set(mapping.texture_codes.tolist()) == {2, 3}


def test_compact_categorical_labels_mapping_uses_array_backed_texture_codes() -> None:
    values = pd.Series(["a", "b"], index=pd.Index([1001, 42], name="index"), dtype="object")

    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=["a", "b"],
        palette=["#ff0000", "#00ff00"],
    )

    np.testing.assert_array_equal(mapping.label_ids, np.asarray([42, 1001], dtype=np.int64))
    assert isinstance(mapping.texture_codes, np.ndarray)
    assert mapping.texture_codes.dtype == np.dtype(np.uint8)
    assert isinstance(mapping.texture_rgba, np.ndarray)
    assert mapping.texture_rgba.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (pd.Series(["a"], index=pd.Index([0], name="index")), "positive label ids"),
        (pd.Series(["a", "b"], index=pd.Index([1, 1], name="index")), "unique label ids"),
        (pd.Series(["a"], index=pd.Index(["cell-1"], name="index")), "integer label ids"),
    ],
)
def test_compact_categorical_labels_mapping_rejects_invalid_label_ids(values: pd.Series, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compact_categorical_labels_mapping_from_values(
            values,
            categories=["a"],
            palette=["#ff0000"],
        )


def test_direct_label_colormap_from_rgba_matches_normal_constructor() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        5: _rgba(1.0, 0.0, 0.0, 1.0),
        9: _rgba(0.0, 0.0, 1.0, 0.5),
    }

    fast_colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    normal_colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)

    assert isinstance(fast_colormap, DirectLabelColormap)
    assert fast_colormap.background_value == normal_colormap.background_value
    values = np.asarray([0, 5, 9, 42], dtype=np.int64)
    np.testing.assert_allclose(fast_colormap.map(values), normal_colormap.map(values))


def test_direct_label_colormap_from_rgba_uses_configured_background_value() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        99: _rgba(0.0, 0.0, 0.0, 0.0),
        5: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    fast_colormap = direct_label_colormap_from_rgba(color_dict, background_value=99)
    normal_colormap = DirectLabelColormap(color_dict=color_dict, background_value=99)

    assert fast_colormap.background_value == 99
    values = np.asarray([99, 5, 0], dtype=np.int64)
    np.testing.assert_allclose(fast_colormap.map(values), normal_colormap.map(values))


def test_direct_label_colormap_from_rgba_keeps_event_emitters_and_clear_cache() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        7: _rgba(0.25, 0.5, 0.75, 1.0),
    }

    colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    _ = colormap._num_unique_colors
    _ = colormap._label_mapping_and_color_dict

    assert hasattr(colormap.events, "color_dict")
    colormap._clear_cache()
    assert "_num_unique_colors" not in colormap.__dict__
    assert "_label_mapping_and_color_dict" not in colormap.__dict__


def test_direct_label_colormap_from_rgba_rejects_string_default_colors() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: "transparent",
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    with pytest.raises(ValueError, match="default/background color"):
        direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("color_dict", "match"),
    [
        (
            {0: _rgba(0.0, 0.0, 0.0, 0.0), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "`None` default color",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "background label",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: np.asarray([0.0, 0.0, 0.0]), 1: _rgba(1.0, 0.0, 0.0, 1.0)},
            "shape",
        ),
    ],
)
def test_direct_label_colormap_from_rgba_rejects_invalid_default_input(
    color_dict: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]


def test_direct_label_colormap_from_rgba_trusts_per_label_values() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        1.5: _rgba(1.0, 0.0, 0.0, 1.0),
        2: "trusted-by-caller",
    }

    colormap = direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]

    assert 1.5 in colormap.color_dict
    assert colormap.color_dict[2] == "trusted-by-caller"


def test_direct_label_colormap_from_rgba_uses_trusted_mapping_without_copying() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    colormap = direct_label_colormap_from_rgba(color_dict)
    color_dict[2] = _rgba(0.0, 1.0, 0.0, 1.0)

    assert colormap.color_dict is color_dict
    assert 2 in colormap.color_dict
