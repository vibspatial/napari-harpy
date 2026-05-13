from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap, label_colormap

from napari_harpy.core.class_palette import normalize_color_sequence
from napari_harpy.core.spatialdata import get_table, get_table_metadata
from napari_harpy.core.table_color_source import ColorValueKind, TableColorSourceSpec
from napari_harpy.viewer._styling import (
    StyledPaletteSource,
    build_string_categorical_values,
    categorical_colors_for_values,
    continuous_colors_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    is_valid_color,
    normalize_category_value,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

StyledLabelsPaletteSource = StyledPaletteSource


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
    labels_name: str,
    style_spec: TableColorSourceSpec,
) -> StyledLabelsStyleResult:
    """Apply one table-backed source to a styled labels layer."""
    table = get_table(sdata, style_spec.table_name)
    table_metadata = get_table_metadata(sdata, style_spec.table_name)
    if not table_metadata.annotates(labels_name):
        raise ValueError(f"Table `{style_spec.table_name}` does not annotate labels element `{labels_name}`.")

    region_rows, obs_index = _get_region_rows_by_instance(table, table_metadata, labels_name)
    if style_spec.value_kind == "instance" or (
        style_spec.source_kind == "obs_column" and style_spec.value_key == table_metadata.instance_key
    ):
        if style_spec.source_kind != "obs_column" or style_spec.value_key != table_metadata.instance_key:
            raise ValueError(
                "Instance ID coloring must use the table instance key "
                f"`{table_metadata.instance_key}` as an observation source."
            )
        style_result, features = _build_instance_key_colormap(region_rows, instance_key=table_metadata.instance_key)
        _apply_instance_labels_colormap(layer)
    elif style_spec.source_kind == "obs_column":
        style_result, color_dict, features = _build_obs_column_colormap(
            table=table,
            region_rows=region_rows,
            column_name=style_spec.value_key,
        )
        _apply_labels_colormap(layer, color_dict)
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


def build_styled_labels_layer_name(labels_name: str, style_spec: TableColorSourceSpec) -> str:
    """Return the user-facing layer name for one styled labels variant."""
    if style_spec.source_kind == "obs_column":
        return f"{labels_name}[obs:{style_spec.value_key}]"
    return f"{labels_name}[X:{style_spec.value_key}]"


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
        categories = [normalize_category_value(value) for value in full_series.cat.categories]
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

    if is_string_like_series(full_series):
        string_region_values, categories = build_string_categorical_values(
            full_values=full_series,
            row_values=region_series,
            column_name=column_name,
        )
        palette = default_categorical_palette_for_categories(categories)
        color_dict = _build_categorical_color_dict(
            string_region_values,
            categories=categories,
            palette=palette,
        )
        # Plain string/object columns are rendered as viewer-only categorical
        # values. They do not use stored categorical palettes, so the caller
        # should report the default palette and mark that coercion happened.
        style_result = StyledLabelsStyleResult(
            value_kind="categorical",
            palette_source="default_missing",
            coercion_applied=True,
        )
        return style_result, color_dict, pd.DataFrame({column_name: string_region_values}, index=region_rows.index)

    numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("float64")
    color_dict = _build_continuous_color_dict(numeric_region_series)
    style_result = StyledLabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, color_dict, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)


def _build_instance_key_colormap(
    region_rows: pd.DataFrame,
    *,
    instance_key: str,
) -> tuple[StyledLabelsStyleResult, pd.DataFrame]:
    instance_values = pd.Series(
        region_rows.index.to_numpy(dtype=np.int64, copy=False),
        index=region_rows.index,
        name=instance_key,
    )
    style_result = StyledLabelsStyleResult(
        value_kind="instance",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, pd.DataFrame({instance_key: instance_values}, index=region_rows.index)


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


def _get_region_rows_by_instance(
    table: AnnData, table_metadata: Any, labels_name: str
) -> tuple[pd.DataFrame, pd.Index]:
    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == labels_name].copy()

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
        return "default_missing", default_categorical_palette_for_categories(categories)

    if len(stored_colors) != len(categories):
        logger.warning(
            f"Stored `{colors_key}` palette has {len(stored_colors)} colors for {len(categories)} categories; "
            "using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    if not all(is_valid_color(color) for color in stored_colors):
        logger.warning(
            f"Stored `{colors_key}` palette contains invalid color values; using the default categorical palette."
        )
        return "default_invalid", default_categorical_palette_for_categories(categories)

    logger.info(f"Using stored `{colors_key}` palette from `table.uns` for viewer coloring.")
    return "stored", list(stored_colors)


def _build_categorical_color_dict(
    values: pd.Series,
    *,
    categories: list[object],
    palette: list[str],
) -> dict[int | None, Any]:
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    colors = categorical_colors_for_values(values, categories=categories, palette=palette)
    for instance_id, color in colors.items():
        color_dict[int(instance_id)] = color
    return color_dict


def _build_continuous_color_dict(values: pd.Series) -> dict[int | None, Any]:
    color_dict: dict[int | None, Any] = {None: "transparent", 0: "transparent"}
    colors = continuous_colors_for_values(values)
    for instance_id, color in colors.items():
        color_dict[int(instance_id)] = color
    return color_dict


def _apply_labels_colormap(layer: Labels, layer_color_dict: dict[int | None, Any]) -> None:
    layer.colormap = DirectLabelColormap(color_dict=layer_color_dict, background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _apply_instance_labels_colormap(layer: Labels) -> None:
    layer.colormap = label_colormap(background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _has_exact_binary_zero_one_values(values: list[object]) -> bool:
    if not values:
        return False
    normalized_values = {int(value) for value in pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1} or normalized_values == {0} or normalized_values == {1}
