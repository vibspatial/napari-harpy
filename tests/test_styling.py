from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from napari_harpy.viewer._styling import (
    categorical_colors_for_values,
    categorical_rgba_for_values,
    continuous_colors_for_values,
    continuous_rgba_for_values,
)
from napari_harpy.viewer.labels_colormap import (
    CompactLabelColormap,
    compact_categorical_label_colormap_from_values,
    direct_label_colormap_from_rgba,
)
from napari_harpy.viewer.labels_styling import (
    _apply_labels_colormap,
    _build_continuous_color_dict,
)


class _FakeLabelsLayer:
    def __init__(self) -> None:
        self.colormap = None
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


def _colors_series_to_rgba(colors: pd.Series) -> np.ndarray:
    return np.asarray([to_rgba(color) for color in colors], dtype="float64")


def test_continuous_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    values = pd.Series([0.0, 1.0, np.nan, 0.5], index=["a", "b", "c", "d"])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_constant_values() -> None:
    values = pd.Series([2.0, 2.0, np.nan], index=[10, 11, 12])

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


def test_continuous_rgba_for_values_matches_existing_colors_for_all_missing_values() -> None:
    values = pd.Series([np.nan, pd.NA, None], dtype="object")

    expected = _colors_series_to_rgba(continuous_colors_for_values(values))
    actual = continuous_rgba_for_values(values)

    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_colors_with_missing_values() -> None:
    categories = ["T", "B", "NK"]
    palette = ["#ff0000", "#00ff00", "#0000ff"]
    values = pd.Series(pd.Categorical(["T", "B", None, "NK"], categories=categories))

    expected = _colors_series_to_rgba(
        categorical_colors_for_values(values, categories=categories, palette=palette)
    )
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert actual.dtype == np.dtype("float64")
    assert actual.shape == (len(values), 4)
    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_colors_for_unknown_values() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(["T", "unknown", None, "B"], dtype="object")

    expected = _colors_series_to_rgba(
        categorical_colors_for_values(values, categories=categories, palette=palette)
    )
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_categorical_rgba_for_values_matches_existing_numpy_scalar_normalization() -> None:
    categories = [np.int64(1), np.int64(2)]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series([1, np.int64(2), 3, None], dtype="object")

    expected = _colors_series_to_rgba(
        categorical_colors_for_values(values, categories=categories, palette=palette)
    )
    actual = categorical_rgba_for_values(values, categories=categories, palette=palette)

    np.testing.assert_allclose(actual, expected)


def test_build_continuous_label_color_dict_uses_vectorized_rgba_and_preserves_background() -> None:
    values = pd.Series([0.0, 10.0, np.nan], index=pd.Index([1, 2, 3], name="index"))

    color_dict = _build_continuous_color_dict(values)
    expected = continuous_rgba_for_values(values)

    np.testing.assert_allclose(color_dict[None], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(color_dict[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(color_dict[1], expected[0])
    np.testing.assert_allclose(color_dict[2], expected[1])
    np.testing.assert_allclose(color_dict[3], expected[2])


def test_build_categorical_label_colormap_uses_compact_mapping_and_preserves_colors() -> None:
    categories = ["T", "B"]
    palette = ["#ff0000", "#00ff00"]
    values = pd.Series(
        ["T", "unknown", None, "B"],
        index=pd.Index([1, 2, 3, 4], name="index"),
        dtype="object",
    )

    colormap = compact_categorical_label_colormap_from_values(values, categories=categories, palette=palette)
    expected = categorical_rgba_for_values(values, categories=categories, palette=palette)

    assert isinstance(colormap, CompactLabelColormap)
    np.testing.assert_allclose(colormap.map(0), np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(colormap.map(values.index.to_numpy(dtype=np.int64)), expected)


def test_apply_labels_colormap_uses_fast_helper_without_explicit_refresh() -> None:
    layer = _FakeLabelsLayer()
    color_dict = {
        None: np.zeros(4, dtype=np.float32),
        0: np.zeros(4, dtype=np.float32),
        1: np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)

    _apply_labels_colormap(layer, colormap)  # type: ignore[arg-type]

    assert layer.colormap is colormap
    np.testing.assert_allclose(layer.colormap.color_dict[1], color_dict[1])
    assert layer.refresh_count == 0
