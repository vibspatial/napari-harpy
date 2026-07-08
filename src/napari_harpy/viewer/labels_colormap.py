from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from napari.utils.colormap_backend import ColormapBackend, set_backend
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.colormaps import _accelerated_cmap as _accel_cmap
from numba import njit, typed, types

from napari_harpy.viewer._styling import MISSING_CATEGORICAL_COLOR, normalize_category_value

_TRANSPARENT_RGBA = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@njit(cache=True)
def _fill_typed_label_texture_mapping(
    typed_mapping,
    label_ids: np.ndarray,
    texture_codes: np.ndarray,
    background_value,
    background_texture_code,
):
    typed_mapping[background_value] = background_texture_code
    for i in range(label_ids.shape[0]):
        typed_mapping[label_ids[i]] = texture_codes[i]
    return typed_mapping


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
        if np.any(self.label_ids == self.background_value):
            raise ValueError("Compact labels mapping label ids must not include the background value.")


class _CompactLabelToTextureMapping(Mapping[int | None, int]):
    """Array-backed `label_id -> texture_code` mapping view.

    This wraps the `CompactCategoricalLabelsMapping` passed to `__init__` and
    exposes its `label_ids` / `texture_codes` arrays through the mapping API.
    Napari expects the first item of `_label_mapping_and_color_dict` to behave
    like a mapping. In scalar high-bit conversion, napari does roughly:

    ```python
    mapper = colormap._label_mapping_and_color_dict[0]
    texture_code = mapper.get(label_id, default)
    ```

    This view supports that without building a large Python dict. Full
    high-bit label arrays use `_get_typed_dict_mapping(...)` instead, so this
    scalar lookup path is not the hot per-pixel rendering path.
    """

    def __init__(self, compact_mapping: CompactCategoricalLabelsMapping) -> None:
        self._compact_mapping = compact_mapping

    def __getitem__(self, key: int | None) -> int:
        compact = self._compact_mapping
        if key is None:
            return compact.default_texture_code
        label_id = int(key)
        if label_id == compact.background_value:
            return compact.background_texture_code
        position = np.searchsorted(compact.label_ids, label_id)
        if position < len(compact.label_ids) and compact.label_ids[position] == label_id:
            return int(compact.texture_codes[position])
        raise KeyError(key)

    def __iter__(self) -> Iterator[int | None]:
        yield None
        yield self._compact_mapping.background_value
        yield from (int(label_id) for label_id in self._compact_mapping.label_ids)

    def __len__(self) -> int:
        return len(self._compact_mapping.label_ids) + 2


class CompactCategoricalLabelColormap(DirectLabelColormap):
    """Prototype direct labels colormap backed by compact categorical state.

    Methods that mirror names from `DirectLabelColormap` are napari-facing
    private hooks. Methods prefixed with `_compact_` are Harpy-only helpers for
    looking up RGBA values from `CompactCategoricalLabelsMapping`.

    Napari and vispy still expect direct-label internals shaped like
    `label_id -> texture_code` and `texture_code -> RGBA`. This subclass
    satisfies that contract with `_label_to_texture_mapping`
    (`label_id -> texture_code`) and `_texture_color_dict`
    (`texture_code -> RGBA`), avoiding the huge `label_id -> RGBA` dictionary
    used by ordinary `DirectLabelColormap` construction.
    """

    def __init__(self, compact_mapping: CompactCategoricalLabelsMapping) -> None:
        set_backend(ColormapBackend.numba)

        small_color_dict: dict[int | None, np.ndarray] = {
            None: compact_mapping.texture_rgba[compact_mapping.default_texture_code],
            compact_mapping.background_value: compact_mapping.texture_rgba[compact_mapping.background_texture_code],
        }
        if len(compact_mapping.label_ids):
            # Initialize `DirectLabelColormap` with one real label color:
            # passing the huge `label_id -> RGBA` mapping through
            # napari/pydantic is known to be slow for large labels layers,
            # while one real label still makes napari detect direct mode
            # instead of treating the colormap as default/background only.
            first_label_id = int(compact_mapping.label_ids[0])
            small_color_dict[first_label_id] = compact_mapping.texture_rgba[
                int(compact_mapping.texture_codes[0])
            ]

        super().__init__(
            color_dict=small_color_dict,
            background_value=compact_mapping.background_value,
        )
        object.__setattr__(self, "_compact_mapping", compact_mapping)
        object.__setattr__(self, "_label_to_texture_mapping", _CompactLabelToTextureMapping(compact_mapping))
        object.__setattr__(
            self,
            "_texture_color_dict",
            dict(enumerate(compact_mapping.texture_rgba)),
        )

    @property
    def _num_unique_colors(self) -> int:
        """Napari hook: report non-default/background texture rows."""
        return max(0, len(self._compact_mapping.texture_rgba) - 2)

    @property
    def _label_mapping_and_color_dict(self) -> tuple[Mapping[int | None, int], dict[int, np.ndarray]]:
        """Napari hook: return the direct-label compact mapping attribute.

        Napari reads `_label_mapping_and_color_dict` directly in scalar and
        high-bit conversion paths, so this cannot be only inlined into
        `_values_mapping_to_minimum_values_set(...)`. Keep it as a cheap view
        over compact arrays rather than the inherited cached property, which
        would rebuild mappings from `color_dict`.
        """
        return self._label_to_texture_mapping, self._texture_color_dict

    def _values_mapping_to_minimum_values_set(
        self,
        apply_selection: bool = True,
    ) -> tuple[Mapping[int | None, int], dict[int, np.ndarray]]:
        """Napari hook: return raw-label-to-texture mapping and color table.

        This mirrors `DirectLabelColormap` while keeping the label mapping
        backed by compact arrays. Napari may read the first tuple item as a
        `label_id -> texture_code` mapping, while vispy reads the second item
        as the `texture_code -> RGBA` table. Returning a mapping view here is
        what lets Harpy avoid building a huge `label_id -> RGBA` dictionary.
        """
        if self.use_selection and apply_selection:
            # Mirror `DirectLabelColormap` selected-label rendering: only
            # `selection` maps to a visible texture code, and all other labels
            # fall through to transparent default color.
            return {self.selection: 1, None: 0}, {
                0: _TRANSPARENT_RGBA,
                1: self._compact_rgba_for_label(self.selection, apply_selection=False),
            }
        # When `use_selection` is false, return the cheap compact mapping view
        # plus the small texture-code-to-RGBA table.
        return self._label_mapping_and_color_dict

    def _get_typed_dict_mapping(self, data_dtype: np.dtype) -> typed.Dict:
        """Napari hook: build the high-bit label-id-to-texture-code dict."""
        data_dtype = np.dtype(data_dtype)
        cache_key = f"_compact_{data_dtype.name}_typed_dict"
        if cache_key in self._cache_other:
            return self._cache_other[cache_key]

        compact = self._compact_mapping
        target_dtype = _accel_cmap.minimum_dtype_for_labels(len(compact.texture_rgba))
        iinfo = np.iinfo(data_dtype)
        representable = (compact.label_ids >= iinfo.min) & (compact.label_ids <= iinfo.max)
        label_ids = compact.label_ids[representable].astype(data_dtype, copy=False)
        texture_codes = compact.texture_codes[representable].astype(target_dtype, copy=False)

        typed_mapping = typed.Dict.empty(
            key_type=getattr(types, data_dtype.name),
            value_type=getattr(types, target_dtype.name),
        )
        typed_mapping = _fill_typed_label_texture_mapping(
            typed_mapping,
            label_ids,
            texture_codes,
            data_dtype.type(compact.background_value),
            target_dtype.type(compact.background_texture_code),
        )

        self._cache_other[cache_key] = typed_mapping
        return typed_mapping

    def map(self, values: np.ndarray | np.integer | int) -> np.ndarray:
        """Napari hook: map scalar or array label ids to RGBA colors.

        For high-bit label arrays, the call chain intentionally moves through
        napari internals while calling back into this subclass:

        ```text
        napari _accel_cmap.labels_raw_to_texture_direct(values, self)
          -> our _get_typed_dict_mapping(...)
             -> label_id -> texture_code
        inherited _map_precast(texture_codes)
          -> our _values_mapping_to_minimum_values_set(...)[1]
             -> texture_code -> RGBA
        ```
        """
        if isinstance(values, np.integer):
            values = int(values)
        if isinstance(values, int):
            return self._compact_rgba_for_label(values, apply_selection=True)
        if isinstance(values, list | tuple):
            values = np.asarray(values)
        if not isinstance(values, np.ndarray) or values.dtype.kind in "fU":
            raise TypeError("DirectLabelColormap can only be used with int")

        if values.dtype.itemsize <= 2:
            # Low-bit labels use napari's bounded dense lookup path.
            mapper = self._get_mapping_from_cache(values.dtype)
            mapped = mapper[values]
        else:
            values_cast = _accel_cmap.labels_raw_to_texture_direct(values, self)
            mapped = self._map_precast(values_cast, apply_selection=True)

        if self.use_selection:
            mapped[(values != self.selection)] = 0
        return mapped

    def _map_without_cache(self, values: np.ndarray) -> np.ndarray:
        """Napari hook: build small-label lookup textures without selection."""
        return self._compact_rgba_for_values(values, apply_selection=False)

    @property
    def _array_map(self) -> np.ndarray:
        """Napari fallback hook that should not be used by this prototype."""
        raise RuntimeError("Compact categorical label colormaps require napari's numba colormap backend.")

    def _clear_cache(self) -> None:
        """Napari hook: clear derived lookup caches."""
        super()._clear_cache()

    def _compact_rgba_for_label(self, label_id: int, *, apply_selection: bool) -> np.ndarray:
        """Harpy helper: resolve one label id through compact texture codes."""
        if apply_selection and self.use_selection and label_id != self.selection:
            return _TRANSPARENT_RGBA
        compact = self._compact_mapping
        texture_code = compact.default_texture_code
        if label_id == compact.background_value:
            texture_code = compact.background_texture_code
        else:
            position = np.searchsorted(compact.label_ids, label_id)
            if position < len(compact.label_ids) and compact.label_ids[position] == label_id:
                texture_code = int(compact.texture_codes[position])
        return compact.texture_rgba[texture_code]

    def _compact_rgba_for_values(self, values: np.ndarray, *, apply_selection: bool) -> np.ndarray:
        """Harpy helper: resolve an array of label ids through compact state."""
        compact = self._compact_mapping
        flat_values = values.ravel()
        texture_codes = np.full(flat_values.shape, compact.default_texture_code, dtype=np.int64)

        background = flat_values == compact.background_value
        texture_codes[background] = compact.background_texture_code

        positions = np.searchsorted(compact.label_ids, flat_values)
        in_bounds = positions < len(compact.label_ids)
        matched = np.zeros(flat_values.shape, dtype=bool)
        matched[in_bounds] = compact.label_ids[positions[in_bounds]] == flat_values[in_bounds]
        if np.any(matched):
            texture_codes[matched] = compact.texture_codes[positions[matched]]

        mapped = compact.texture_rgba[texture_codes].reshape(values.shape + (4,))
        if apply_selection and self.use_selection:
            mapped = mapped.copy()
            mapped[values != self.selection] = _TRANSPARENT_RGBA
        return mapped


def compact_categorical_labels_mapping_from_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[Any],
    default_color: Any = _TRANSPARENT_RGBA,
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
    default_color
        Color for labels that are not present in `values`. The default is
        transparent, matching styled-labels behavior. Object-classification
        callers can pass the unlabeled class color while keeping the
        background label transparent.
    missing_color
        Color for known table rows with missing or palette-unknown categorical
        values.
    background_value
        Background label id.

    Known table rows with missing or palette-unknown categorical values receive
    the missing categorical color. Label ids not present in `values` are not
    stored here and later fall through to `default_color`.
    """
    if not isinstance(background_value, int) or isinstance(background_value, bool):
        raise ValueError("Compact labels mapping background value must be an integer label id.")

    label_ids = _positive_label_ids_from_index(values.index)
    normalized_categories = [normalize_category_value(category) for category in categories]

    default_rgba = np.asarray(to_rgba(default_color), dtype=np.float32)
    texture_rgba_rows: list[np.ndarray] = [default_rgba, _TRANSPARENT_RGBA.copy()]
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


def compact_categorical_label_colormap_from_values(
    values: pd.Series,
    *,
    categories: Sequence[object],
    palette: Sequence[Any],
    default_color: Any = _TRANSPARENT_RGBA,
    missing_color: Any = MISSING_CATEGORICAL_COLOR,
    background_value: int = 0,
) -> CompactCategoricalLabelColormap:
    """Return a compact direct labels colormap for categorical values."""
    mapping = compact_categorical_labels_mapping_from_values(
        values,
        categories=categories,
        palette=palette,
        default_color=default_color,
        missing_color=missing_color,
        background_value=background_value,
    )
    return CompactCategoricalLabelColormap(mapping)


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
