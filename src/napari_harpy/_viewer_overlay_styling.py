from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy._class_palette import default_categorical_colors, normalize_color_sequence
from napari_harpy._spatialdata import get_table, get_table_metadata
from napari_harpy._table_color_source import ColorValueKind, TableColorSourceSpec

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

StyledLabelsPaletteSource = Literal["stored", "default_missing", "default_invalid"]

MISSING_CATEGORICAL_COLOR = "#80808099"
MISSING_CONTINUOUS_COLOR = "#80808099"
OVERLAY_CONTINUOUS_COLORMAP = "viridis"


@dataclass(frozen=True)
class StyledLabelsStyleResult:
    """Describe how one styled labels overlay was colored."""

    value_kind: ColorValueKind
    palette_source: StyledLabelsPaletteSource | None
    coercion_applied: bool


@dataclass(frozen=True)
class StyledLabelsLoadResult(StyledLabelsStyleResult):
    """Describe the styled overlay load/update result returned to the viewer."""

    layer: Labels
    created: bool


def apply_table_color_source_to_labels_layer(
    layer: Labels,
    *,
    sdata: SpatialData,
    label_name: str,
    style_spec: TableColorSourceSpec,
) -> StyledLabelsStyleResult:
    """Apply one table-backed source to a styled labels layer."""
    table = get_table(sdata, style_spec.table_name)
    table_metadata = get_table_metadata(sdata, style_spec.table_name)
    if not table_metadata.annotates(label_name):
        raise ValueError(f"Table `{style_spec.table_name}` does not annotate segmentation `{label_name}`.")

    region_rows, obs_index = _get_region_rows_by_instance(table, table_metadata, label_name)
    if style_spec.source_kind == "obs_column":
        style_result, color_dict, features = _build_obs_column_colormap(
            table=table,
            region_rows=region_rows,
            column_name=style_spec.value_key,
        )
    else:
        style_result, color_dict, features = _build_x_var_colormap(
            table=table,
            region_rows=region_rows,
            obs_index=obs_index,
            var_name=style_spec.value_key,
        )

    _apply_labels_colormap(layer, color_dict)
    layer.features = _build_labels_features(features, instance_key=table_metadata.instance_key)
    return style_result


def build_styled_labels_layer_name(label_name: str, style_spec: TableColorSourceSpec) -> str:
    """Return the user-facing layer name for one styled labels variant."""
    if style_spec.source_kind == "obs_column":
        return f"{label_name}[obs:{style_spec.value_key}]"
    return f"{label_name}[X:{style_spec.value_key}]"


def _build_obs_column_colormap(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    column_name: str,
) -> tuple[StyledLabelsStyleResult, dict[int, Any], pd.DataFrame]:
    if column_name not in table.obs:
        raise ValueError(f"Observation column `{column_name}` is not available in the selected table.")

    full_series = table.obs[column_name]
    region_series = region_rows[column_name]

    if _is_categorical_dtype(full_series):
        categories = [_normalize_category_value(value) for value in full_series.cat.categories]
        palette_source, palette = _resolve_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(region_series, categories=categories, palette=palette)
        style_result = StyledLabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_bool_dtype(full_series):
        categories = [value for value in (False, True) if value in set(full_series.dropna().tolist())]
        palette_source, palette = _resolve_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(region_series, categories=categories, palette=palette)
        style_result = StyledLabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_integer_dtype(full_series) and _has_exact_binary_zero_one_values(full_series.dropna().tolist()):
        categories = [
            value
            for value in (0, 1)
            if value in set(pd.to_numeric(full_series.dropna(), errors="coerce").astype("int64").tolist())
        ]
        numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("Int64")
        palette_source, palette = _resolve_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        color_dict = _build_categorical_color_dict(numeric_region_series, categories=categories, palette=palette)
        style_result = StyledLabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, color_dict, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)

    if _is_string_like_series(full_series):
        logger.warning(
            f"Coercing plain string/object observation column `{column_name}` to temporary categorical values for viewer coloring."
        )
        full_values = pd.Series(
            [_normalize_string_value(value) for value in full_series],
            index=full_series.index,
            name=column_name,
            dtype="object",
        )
        region_values = pd.Series(
            [_normalize_string_value(value) for value in region_series],
            index=region_rows.index,
            name=column_name,
            dtype="object",
        )
        categories = list(pd.unique(full_values.dropna()))
        palette = default_categorical_colors(len(categories))
        color_dict = _build_categorical_color_dict(region_values, categories=categories, palette=palette)
        style_result = StyledLabelsStyleResult(
            value_kind="categorical",
            palette_source="default_missing",
            coercion_applied=True,
        )
        return style_result, color_dict, pd.DataFrame({column_name: region_values}, index=region_rows.index)

    numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("float64")
    color_dict = _build_continuous_color_dict(numeric_region_series)
    style_result = StyledLabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)


def _build_x_var_colormap(
    *,
    table: AnnData,
    region_rows: pd.DataFrame,
    obs_index: pd.Index,
    var_name: str,
) -> tuple[StyledLabelsStyleResult, dict[int, Any], pd.DataFrame]:
    if var_name not in table.var_names:
        raise ValueError(f"Var `{var_name}` is not available in the selected table.")

    var_index = table.var_names.get_loc(var_name)
    if not isinstance(var_index, (int, np.integer)):
        raise ValueError(f"Var `{var_name}` does not resolve to one unique column in `table.var_names`.")

    obs_positions = table.obs.index.get_indexer(obs_index)
    if np.any(obs_positions < 0):
        raise ValueError(f"Could not align linked table rows back to `X[:, {var_name!r}]` for viewer coloring.")

    column_values = table.X[obs_positions, int(var_index)]
    if hasattr(column_values, "toarray"):
        numeric_values = column_values.toarray().reshape(-1)
    else:
        numeric_values = np.asarray(column_values).reshape(-1)
    numeric_region_series = pd.Series(
        pd.to_numeric(numeric_values, errors="coerce").astype("float64"),
        index=region_rows.index,
        name=var_name,
    )
    color_dict = _build_continuous_color_dict(numeric_region_series)
    style_result = StyledLabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({var_name: numeric_region_series}, index=region_rows.index)


def _get_region_rows_by_instance(table: AnnData, table_metadata: Any, label_name: str) -> tuple[pd.DataFrame, pd.Index]:
    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == label_name].copy()

    instance_ids = pd.to_numeric(region_rows[table_metadata.instance_key], errors="coerce")
    region_rows = region_rows.loc[instance_ids.notna()].copy()
    region_rows[table_metadata.instance_key] = instance_ids.loc[region_rows.index].astype("int64")
    region_rows = region_rows.loc[region_rows[table_metadata.instance_key] > 0].copy()
    region_rows = region_rows.drop_duplicates(subset=[table_metadata.instance_key], keep="last")
    aligned_obs_index = region_rows.index.copy()
    region_rows = region_rows.set_index(table_metadata.instance_key)
    region_rows.index = region_rows.index.astype("int64", copy=False)
    # napari Labels uses a feature column literally named `index` as the
    # hidden label-value -> feature-row mapping for hover/properties lookup.
    region_rows.index.name = "index"
    return region_rows, aligned_obs_index


def _build_labels_features(features: pd.DataFrame, *, instance_key: str) -> pd.DataFrame:
    features = features.copy()
    if instance_key not in features.columns:
        features.insert(0, instance_key, features.index.to_numpy(copy=False))
    return features.reset_index()


def _resolve_categorical_palette(
    *,
    table: AnnData,
    column_name: str,
    categories: list[object],
) -> tuple[StyledLabelsPaletteSource, list[str]]:
    colors_key = f"{column_name}_colors"
    stored_colors = normalize_color_sequence(table.uns.get(colors_key))
    if stored_colors is None:
        logger.info(
            f"No stored `{colors_key}` palette found in `table.uns`; using the default categorical palette for viewer coloring."
        )
        return "default_missing", default_categorical_colors(len(categories))

    if len(stored_colors) != len(categories):
        logger.warning(
            f"Stored `{colors_key}` palette has {len(stored_colors)} colors for {len(categories)} categories; "
            "using the default categorical palette."
        )
        return "default_invalid", default_categorical_colors(len(categories))

    if not all(_is_valid_color(color) for color in stored_colors):
        logger.warning(
            f"Stored `{colors_key}` palette contains invalid color values; using the default categorical palette."
        )
        return "default_invalid", default_categorical_colors(len(categories))

    logger.info(f"Using stored `{colors_key}` palette from `table.uns` for viewer coloring.")
    return "stored", list(stored_colors)


def _build_categorical_color_dict(
    values: pd.Series,
    *,
    categories: list[object],
    palette: list[str],
) -> dict[int | None, Any]:
    lookup = {_normalize_category_value(category): color for category, color in zip(categories, palette, strict=False)}
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    for instance_id, value in values.items():
        if pd.isna(value):
            color_dict[int(instance_id)] = MISSING_CATEGORICAL_COLOR
            continue
        color_dict[int(instance_id)] = lookup.get(_normalize_category_value(value), MISSING_CATEGORICAL_COLOR)
    return color_dict


def _build_continuous_color_dict(values: pd.Series) -> dict[int | None, Any]:
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    non_missing = values.dropna()
    if non_missing.empty:
        for instance_id in values.index:
            color_dict[int(instance_id)] = MISSING_CONTINUOUS_COLOR
        return color_dict

    cmap = colormaps[OVERLAY_CONTINUOUS_COLORMAP]
    min_value = float(non_missing.min())
    max_value = float(non_missing.max())
    if max_value == min_value:
        normalized_values = {int(instance_id): 0.5 for instance_id in non_missing.index}
    else:
        normalized_values = {
            int(instance_id): float((value - min_value) / (max_value - min_value))
            for instance_id, value in non_missing.items()
        }

    for instance_id in values.index:
        value = values.at[instance_id]
        if pd.isna(value):
            color_dict[int(instance_id)] = MISSING_CONTINUOUS_COLOR
        else:
            color_dict[int(instance_id)] = cmap(float(np.clip(normalized_values[int(instance_id)], 0.0, 1.0)))
    return color_dict


def _apply_labels_colormap(layer: Labels, layer_color_dict: dict[int | None, Any]) -> None:
    layer.colormap = DirectLabelColormap(color_dict=layer_color_dict, background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _is_string_like_series(values: pd.Series) -> bool:
    non_null = values.dropna()
    if non_null.empty:
        return False
    return all(_is_string_scalar(value) for value in non_null.tolist())


def _is_string_scalar(value: object) -> bool:
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _normalize_string_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(value)


def _normalize_category_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _has_exact_binary_zero_one_values(values: list[object]) -> bool:
    if not values:
        return False
    normalized_values = {int(value) for value in pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1} or normalized_values == {0} or normalized_values == {1}


def _is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True
