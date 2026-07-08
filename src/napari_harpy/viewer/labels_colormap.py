from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy.viewer._styling import MISSING_CATEGORICAL_COLOR, normalize_category_value

_TRANSPARENT_RGBA = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class CompactCategoricalLabelsMapping:
    """Compact categorical labels color state.

    `label_ids -> texture_codes` is the logical per-object mapping. RGBA values
    live once in `texture_rgba`, where each row index is the texture code. The
    color for `label_ids[i]` is therefore `texture_rgba[texture_codes[i]]`. The
    state is intentionally independent of napari colormap internals; later
    slices can derive numba typed mappings or bounded small-label lookup arrays
    from these arrays.

    Parameters
    ----------
    label_ids
        Positive label ids / instance ids represented by this compact mapping.
    texture_codes
        Texture code for each entry in `label_ids`.
    texture_rgba
        RGBA lookup table where the row index is the texture code.
    default_texture_code
        Texture code used for labels that are not present in `label_ids`.
        This corresponds to the current `DirectLabelColormap` `None` default
        color for unmapped labels, not to known table rows with missing
        categorical values.
    background_texture_code
        Texture code used for the background label.
    background_value
        Background label id.
    missing_texture_code
        Texture code used for known table rows whose categorical value is
        missing or palette-unknown. This is `None` when all known table rows
        map to explicit categories and no missing-category color row is needed.

    Examples
    --------
    If four labels are red and four labels are green:

    ```text
    label_ids:     [101, 102, 103, 104, 105, 106, 107, 108]
    texture_codes: [2,   2,   2,   2,   3,   3,   3,   3]

    texture_rgba: [transparent default, transparent background, red, green]

    texture_rgba:
      0 -> transparent default for unmapped labels
      1 -> transparent background label
      2 -> red
      3 -> green
    ```
    """

    label_ids: np.ndarray
    texture_codes: np.ndarray
    texture_rgba: np.ndarray
    default_texture_code: int = 0
    background_texture_code: int = 1
    background_value: int = 0
    missing_texture_code: int | None = None

    def __post_init__(self) -> None:
        if self.label_ids.ndim != 1:
            raise ValueError("Compact labels mapping `label_ids` must be one-dimensional.")
        if self.texture_codes.ndim != 1:
            raise ValueError("Compact labels mapping `texture_codes` must be one-dimensional.")
        if len(self.label_ids) != len(self.texture_codes):
            raise ValueError("Compact labels mapping must contain one texture code per label id.")
        if self.texture_rgba.ndim != 2 or self.texture_rgba.shape[1] != 4:
            raise ValueError("Compact labels mapping `texture_rgba` must have shape `(n, 4)`.")
        if len(self.texture_rgba) == 0:
            raise ValueError("Compact labels mapping must contain at least one texture RGBA row.")
        if self.default_texture_code != 0:
            raise ValueError("Compact labels mapping default texture code must be 0.")
        if not 0 <= self.background_texture_code < len(self.texture_rgba):
            raise ValueError("Compact labels mapping background texture code is out of range.")
        if self.missing_texture_code is not None and not 0 <= self.missing_texture_code < len(self.texture_rgba):
            raise ValueError("Compact labels mapping missing texture code is out of range.")
        if len(self.texture_codes) and int(np.max(self.texture_codes)) >= len(self.texture_rgba):
            raise ValueError("Compact labels mapping contains a texture code without an RGBA row.")


def compact_categorical_labels_mapping_from_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[str],
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
    background_value: int = 0,
) -> CompactCategoricalLabelsMapping:
    """Return compact labels coloring state for categorical table values.

    Parameters
    ----------
    values
        Table-aligned categorical column for one labels element. The index
        contains positive label ids / instance ids, and each row value is the
        categorical value used to color that label. This is not the labels image
        data itself.
    categories
        Ordered categorical values that correspond to `palette`.
    palette
        Color values for `categories`.
    missing_color
        Color for known table rows with missing or palette-unknown categorical
        values.
    background_value
        Background label id.

    Known table rows with missing or palette-unknown categorical values receive
    the missing categorical color. Label ids not present in `values` are not
    stored here and should later fall through to the transparent default color.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Compact labels mapping background value must be an integer label id.")

    label_ids = _positive_label_ids_from_index(values.index)
    normalized_categories = [normalize_category_value(category) for category in categories]

    texture_rgba_rows: list[np.ndarray] = [_TRANSPARENT_RGBA.copy(), _TRANSPARENT_RGBA.copy()]
    color_to_texture_code: dict[tuple[float, float, float, float], int] = {
        _rgba_key(texture_rgba_rows[1]): 1,
    }

    def texture_code_for_rgba(color: Any) -> int:
        rgba = np.asarray(to_rgba(color), dtype=np.float32)
        key = _rgba_key(rgba)
        texture_code = color_to_texture_code.get(key)
        if texture_code is not None:
            return texture_code
        texture_code = len(texture_rgba_rows)
        color_to_texture_code[key] = texture_code
        texture_rgba_rows.append(rgba)
        return texture_code

    category_texture_code_by_value = {
        category: texture_code_for_rgba(color) for category, color in zip(normalized_categories, palette, strict=False)
    }

    texture_codes = _categorical_texture_codes(
        values,
        category_texture_code_by_value=category_texture_code_by_value,
    )
    missing_texture_code: int | None = None
    missing_values = texture_codes < 0
    if np.any(missing_values):
        missing_texture_code = texture_code_for_rgba(missing_color)
        texture_codes[missing_values] = missing_texture_code
    order = np.argsort(label_ids)
    label_ids = label_ids[order]
    texture_codes = texture_codes[order]

    texture_codes = texture_codes.astype(_minimum_unsigned_dtype(int(np.max(texture_codes, initial=0))), copy=False)
    texture_rgba = np.asarray(texture_rgba_rows, dtype=np.float32)

    return CompactCategoricalLabelsMapping(
        label_ids=label_ids,
        texture_codes=texture_codes,
        texture_rgba=texture_rgba,
        default_texture_code=0,
        background_texture_code=1,
        background_value=background_value,
        missing_texture_code=missing_texture_code,
    )


def direct_label_colormap_from_rgba(
    color_dict: dict[int | None, np.ndarray],
    *,
    background_value: int = 0,
) -> DirectLabelColormap:
    """Construct a direct labels colormap from Harpy-generated RGBA arrays.

    Napari's public ``DirectLabelColormap(color_dict=...)`` constructor becomes
    expensive for large labels layers because it validates and color-normalizes
    every ``label_id -> RGBA`` entry. Harpy's styled-labels paths already build
    numeric RGBA arrays, so the large constructor pass is redundant.

    To avoid that bottleneck, construct a tiny normal ``DirectLabelColormap``
    with only the default/background colors so napari/pydantic still initializes
    the model and event emitters correctly. Then install the trusted full RGBA
    mapping directly and clear napari's derived colormap caches.

    This is an internal fast path for RGBA dictionaries generated by Harpy
    itself. It performs only cheap structural checks on the default/background
    entries and does not revalidate every per-label color in the mapping. The
    mapping container and per-label arrays are not copied; callers should treat
    the input mapping and RGBA arrays as immutable after construction.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Labels colormap background value must be an integer label id.")
    if None not in color_dict:
        raise ValueError("Direct labels RGBA color dictionary must include a `None` default color.")
    if background_value not in color_dict:
        raise ValueError(f"Direct labels RGBA color dictionary must include background label `{background_value}`.")

    _validate_default_color(color_dict[None], label_id=None)
    _validate_default_color(color_dict[background_value], label_id=background_value)
    small_color_dict = {
        None: color_dict[None],
        background_value: color_dict[background_value],
    }
    colormap = DirectLabelColormap(color_dict=small_color_dict, background_value=background_value)
    object.__setattr__(colormap, "color_dict", color_dict)
    colormap._clear_cache()
    return colormap


def _validate_default_color(color: np.ndarray, *, label_id: int | None) -> None:
    if not isinstance(color, np.ndarray):
        raise ValueError(f"Direct labels RGBA default/background color for label `{label_id}` must be a numpy array.")
    if color.shape != (4,):
        raise ValueError(f"Direct labels RGBA default/background color for label `{label_id}` must have shape `(4,)`.")


def _positive_label_ids_from_index(index: pd.Index) -> np.ndarray:
    raw_label_ids = index.to_numpy()
    try:
        label_ids = raw_label_ids.astype(np.int64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Compact labels mapping index must contain integer label ids.") from error
    if not np.array_equal(raw_label_ids, label_ids):
        raise ValueError("Compact labels mapping index must contain integer label ids.")
    # This is the main validation cost for large tables. If upstream alignment
    # guarantees uniqueness in the future, we can consider trusting that input.
    if len(label_ids) != len(np.unique(label_ids)):
        raise ValueError("Compact labels mapping index must contain unique label ids.")
    if np.any(label_ids <= 0):
        raise ValueError("Compact labels mapping index must contain positive label ids.")
    return label_ids


def _categorical_texture_codes(
    values: pd.Series,
    *,
    category_texture_code_by_value: dict[object, int],
) -> np.ndarray:
    """Return one texture code per row, using `-1` for missing/unknown values.

    Pandas categorical values use the fast `.cat.codes` path and map category
    codes through the palette texture-code table. Other dtypes are normalized
    value-by-value so NumPy scalar variants and missing values match Harpy's
    existing categorical-coloring semantics.
    """
    if len(values) == 0:
        return np.asarray([], dtype=np.int64)

    if isinstance(values.dtype, pd.CategoricalDtype):
        value_categories = [normalize_category_value(category) for category in values.cat.categories]
        code_by_value_code = np.asarray(
            [category_texture_code_by_value.get(category, -1) for category in value_categories],
            dtype=np.int64,
        )
        value_codes = values.cat.codes.to_numpy(copy=False)
        texture_codes = np.full(len(values), -1, dtype=np.int64)
        present = value_codes >= 0
        if np.any(present):
            texture_codes[present] = code_by_value_code[value_codes[present]]
        return texture_codes

    normalized_values = pd.Series(
        [pd.NA if pd.isna(value) else normalize_category_value(value) for value in values],
        index=values.index,
        dtype="object",
    )
    mapped = normalized_values.map(category_texture_code_by_value)
    known = mapped.notna().to_numpy(dtype=bool, copy=False)
    texture_codes = np.empty(len(values), dtype=np.int64)
    if np.any(known):
        texture_codes[known] = mapped.loc[known].to_numpy(dtype=np.int64, copy=False)
    if np.any(~known):
        texture_codes[~known] = -1
    return texture_codes


def _rgba_key(rgba: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(channel) for channel in rgba)


def _minimum_unsigned_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)
