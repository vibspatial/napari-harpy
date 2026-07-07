from __future__ import annotations

import numpy as np
import pytest
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy.viewer.labels_colormap import direct_label_colormap_from_rgba


def _rgba(red: float, green: float, blue: float, alpha: float) -> np.ndarray:
    return np.asarray([red, green, blue, alpha], dtype=np.float64)


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


def test_direct_label_colormap_from_rgba_rejects_string_colors() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: "transparent",
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    with pytest.raises(ValueError, match="pre-normalized numpy array"):
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
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: _rgba(0.0, 0.0, 0.0, 0.0), 1.5: _rgba(1.0, 0.0, 0.0, 1.0)},
            "integer label ids",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: _rgba(0.0, 0.0, 0.0, 0.0), 1: np.asarray([1.0, 0.0, 0.0])},
            "shape",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: _rgba(0.0, 0.0, 0.0, 0.0), 1: np.asarray([1.0, 0.0, 0.0, np.nan])},
            "finite",
        ),
        (
            {None: _rgba(0.0, 0.0, 0.0, 0.0), 0: _rgba(0.0, 0.0, 0.0, 0.0), 1: np.asarray([1.5, 0.0, 0.0, 1.0])},
            "between 0 and 1",
        ),
    ],
)
def test_direct_label_colormap_from_rgba_rejects_invalid_input(
    color_dict: dict[object, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        direct_label_colormap_from_rgba(color_dict)  # type: ignore[arg-type]


def test_direct_label_colormap_from_rgba_copies_mapping_container() -> None:
    color_dict = {
        None: _rgba(0.0, 0.0, 0.0, 0.0),
        0: _rgba(0.0, 0.0, 0.0, 0.0),
        1: _rgba(1.0, 0.0, 0.0, 1.0),
    }

    colormap = direct_label_colormap_from_rgba(color_dict)
    color_dict[2] = _rgba(0.0, 1.0, 0.0, 1.0)

    assert 2 not in colormap.color_dict
