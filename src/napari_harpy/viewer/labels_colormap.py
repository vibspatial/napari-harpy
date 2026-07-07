from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from napari.utils.colormaps import DirectLabelColormap


def direct_label_colormap_from_rgba(
    color_dict: Mapping[int | None, np.ndarray],
    *,
    background_value: int = 0,
) -> DirectLabelColormap:
    """Construct a direct labels colormap from pre-normalized RGBA arrays.

    Napari's public ``DirectLabelColormap(color_dict=...)`` constructor becomes
    expensive for large labels layers because it validates and color-normalizes
    every ``label_id -> RGBA`` entry. Harpy's styled-labels paths already build
    numeric RGBA arrays, so the large constructor pass is redundant.

    To avoid that bottleneck, construct a tiny normal ``DirectLabelColormap``
    with only the default/background colors so napari/pydantic still initializes
    the model and event emitters correctly. Then install the already-validated
    full RGBA mapping directly and clear napari's derived colormap caches. The
    mapping container is copied, but the per-label arrays are not copied; callers
    should treat the input RGBA arrays as immutable after construction.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Labels colormap background value must be an integer label id.")
    normalized_color_dict = _normalize_rgba_color_dict(color_dict, background_value=background_value)
    small_color_dict = {
        None: normalized_color_dict[None],
        background_value: normalized_color_dict[background_value],
    }
    colormap = DirectLabelColormap(color_dict=small_color_dict, background_value=background_value)
    object.__setattr__(colormap, "color_dict", normalized_color_dict)
    colormap._clear_cache()
    return colormap


def _normalize_rgba_color_dict(
    color_dict: Mapping[int | None, np.ndarray],
    *,
    background_value: int,
) -> dict[int | None, np.ndarray]:
    if None not in color_dict:
        raise ValueError("Direct labels RGBA color dictionary must include a `None` default color.")
    if background_value not in color_dict:
        raise ValueError(
            f"Direct labels RGBA color dictionary must include background label `{background_value}`."
        )

    colors: list[np.ndarray] = []
    for label_id, color in color_dict.items():
        if label_id is not None and (not isinstance(label_id, int) or isinstance(label_id, bool)):
            raise ValueError("Direct labels RGBA color dictionary keys must be integer label ids or `None`.")
        if not isinstance(color, np.ndarray):
            raise ValueError("Direct labels RGBA colors must be pre-normalized numpy arrays.")
        colors.append(color)

    try:
        color_array = np.asarray(colors)
    except ValueError as error:
        raise ValueError("Direct labels RGBA colors must have shape `(4,)`.") from error

    if color_array.ndim != 2 or color_array.shape[1] != 4:
        raise ValueError("Direct labels RGBA colors must have shape `(4,)`.")
    if not np.issubdtype(color_array.dtype, np.number):
        raise ValueError("Direct labels RGBA colors must be numeric.")
    if not np.all(np.isfinite(color_array)):
        raise ValueError("Direct labels RGBA colors must contain only finite values.")
    if np.any((color_array < 0.0) | (color_array > 1.0)):
        raise ValueError("Direct labels RGBA colors must contain values between 0 and 1.")
    return dict(color_dict)
