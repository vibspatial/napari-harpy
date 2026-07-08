from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from napari.layers import Labels
from napari.utils.colormaps import DirectLabelColormap, label_colormap

from napari_harpy.core._color_source import (
    TableColorSourceSpec,
    TableColorValueKind,
    validate_table_color_value_kind,
)
from napari_harpy.core.spatialdata import get_table, validate_table_binding
from napari_harpy.viewer._styling import (
    StyledPaletteSource,
    build_string_categorical_values,
    continuous_rgba_for_values,
    default_categorical_palette_for_categories,
    is_string_like_series,
    normalize_category_value,
    resolve_table_categorical_palette,
    validate_styled_palette_source,
)
from napari_harpy.viewer.labels_colormap import (
    CompactCategoricalLabelColormap,
    compact_categorical_label_colormap_from_values,
    direct_label_colormap_from_rgba,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


_TRANSPARENT_RGBA = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
LabelsColormap = DirectLabelColormap | CompactCategoricalLabelColormap


@dataclass(frozen=True)
class LabelsStyleResult:
    """Describe labels styling metadata, if styling was applied."""

    value_kind: TableColorValueKind | None
    palette_source: StyledPaletteSource | None
    coercion_applied: bool

    def __post_init__(self) -> None:
        if self.value_kind is not None:
            validate_table_color_value_kind(self.value_kind)
        if self.palette_source is not None:
            validate_styled_palette_source(self.palette_source)


@dataclass(frozen=True)
class LabelsLoadResult(LabelsStyleResult):
    """Describe a primary or styled labels layer load/update result."""

    layer: Labels
    created: bool


def apply_table_color_source_to_labels_layer(
    layer: Labels,
    *,
    sdata: SpatialData,
    labels_name: str,
    style_spec: TableColorSourceSpec,
) -> LabelsStyleResult:
    """Apply one table-backed source to a styled labels layer."""
    table = get_table(sdata, style_spec.table_name)
    table_metadata = validate_table_binding(sdata, labels_name, style_spec.table_name)
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
        style_result, colormap, features = _build_obs_column_colormap(
            table=table,
            region_rows=region_rows,
            column_name=style_spec.value_key,
        )
        _apply_labels_colormap(layer, colormap)
    else:
        style_result, colormap, features = _build_x_var_colormap(
            table=table,
            region_rows=region_rows,
            obs_index=obs_index,
            var_name=style_spec.value_key,
        )
        _apply_labels_colormap(layer, colormap)

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
) -> tuple[LabelsStyleResult, LabelsColormap, pd.DataFrame]:
    if column_name not in table.obs:
        raise ValueError(f"Observation column `{column_name}` is not available in the selected table.")

    full_series = table.obs[column_name]
    region_series = region_rows[column_name]

    if _is_categorical_dtype(full_series):
        categories = [normalize_category_value(value) for value in full_series.cat.categories]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        colormap = compact_categorical_label_colormap_from_values(
            region_series,
            categories=categories,
            palette=palette,
        )
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, colormap, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_bool_dtype(full_series):
        categories = [value for value in (False, True) if value in set(full_series.dropna().tolist())]
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        colormap = compact_categorical_label_colormap_from_values(
            region_series,
            categories=categories,
            palette=palette,
        )
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, colormap, pd.DataFrame({column_name: region_series}, index=region_rows.index)

    if pd.api.types.is_integer_dtype(full_series) and _has_exact_binary_zero_one_values(full_series.dropna().tolist()):
        categories = [
            value
            for value in (0, 1)
            if value in set(pd.to_numeric(full_series.dropna(), errors="coerce").astype("int64").tolist())
        ]
        numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("Int64")
        palette_source, palette = resolve_table_categorical_palette(
            table=table,
            column_name=column_name,
            categories=categories,
        )
        colormap = compact_categorical_label_colormap_from_values(
            numeric_region_series,
            categories=categories,
            palette=palette,
        )
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source=palette_source,
            coercion_applied=False,
        )
        return style_result, colormap, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)

    if is_string_like_series(full_series):
        string_region_values, categories = build_string_categorical_values(
            full_values=full_series,
            row_values=region_series,
            column_name=column_name,
        )
        palette = default_categorical_palette_for_categories(categories)
        colormap = compact_categorical_label_colormap_from_values(
            string_region_values,
            categories=categories,
            palette=palette,
        )
        # Plain string/object columns are rendered as viewer-only categorical
        # values. They do not use stored categorical palettes, so the caller
        # should report the default palette and mark that coercion happened.
        style_result = LabelsStyleResult(
            value_kind="categorical",
            palette_source="default_missing",
            coercion_applied=True,
        )
        return style_result, colormap, pd.DataFrame({column_name: string_region_values}, index=region_rows.index)

    numeric_region_series = pd.to_numeric(region_series, errors="coerce").astype("float64")
    color_dict = _build_continuous_color_dict(numeric_region_series)
    colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    style_result = LabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, colormap, pd.DataFrame({column_name: numeric_region_series}, index=region_rows.index)


def _build_instance_key_colormap(
    region_rows: pd.DataFrame,
    *,
    instance_key: str,
) -> tuple[LabelsStyleResult, pd.DataFrame]:
    instance_values = pd.Series(
        region_rows.index.to_numpy(dtype=np.int64, copy=False),
        index=region_rows.index,
        name=instance_key,
    )
    style_result = LabelsStyleResult(
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
) -> tuple[LabelsStyleResult, LabelsColormap, pd.DataFrame]:
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
    colormap = direct_label_colormap_from_rgba(color_dict, background_value=0)
    style_result = LabelsStyleResult(
        value_kind="continuous",
        palette_source=None,
        coercion_applied=False,
    )
    return style_result, colormap, pd.DataFrame({var_name: numeric_region_series}, index=region_rows.index)


def _get_region_rows_by_instance(
    table: AnnData, table_metadata: Any, labels_name: str
) -> tuple[pd.DataFrame, pd.Index]:
    region_rows = table.obs.loc[table.obs[table_metadata.region_key] == labels_name].copy()

    instance_ids = pd.to_numeric(region_rows[table_metadata.instance_key], errors="coerce")
    region_rows = region_rows.loc[instance_ids.notna()].copy()
    region_rows[table_metadata.instance_key] = instance_ids.loc[region_rows.index].astype("int64")
    region_rows = region_rows.loc[region_rows[table_metadata.instance_key] > 0].copy()
    duplicate_instance_ids = region_rows.loc[
        region_rows[table_metadata.instance_key].duplicated(keep=False),
        table_metadata.instance_key,
    ]
    if not duplicate_instance_ids.empty:
        preview = _format_instance_preview(duplicate_instance_ids.drop_duplicates().tolist())
        raise ValueError(
            f"Table `{table_metadata.table_name}` cannot be aligned to labels element `{labels_name}` because "
            f"`{table_metadata.instance_key}` contains duplicate positive label IDs after labels-specific "
            f"numeric coercion: {preview}."
        )

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


def _format_instance_preview(instance_ids: list[Any]) -> str:
    preview = ", ".join(str(instance_id) for instance_id in instance_ids[:5])
    if len(instance_ids) > 5:
        preview += ", ..."
    return preview


def _build_continuous_color_dict(values: pd.Series) -> dict[int | None, np.ndarray]:
    color_dict: dict[int | None, np.ndarray] = _transparent_default_color_dict()
    colors = continuous_rgba_for_values(values)
    for instance_id, color in zip(values.index, colors, strict=True):
        color_dict[int(instance_id)] = color
    return color_dict


def _apply_labels_colormap(layer: Labels, colormap: LabelsColormap) -> None:
    layer.colormap = colormap


def _apply_instance_labels_colormap(layer: Labels) -> None:
    layer.colormap = label_colormap(background_value=0)
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _transparent_default_color_dict() -> dict[int | None, np.ndarray]:
    return {None: _TRANSPARENT_RGBA.copy(), 0: _TRANSPARENT_RGBA.copy()}


def _is_categorical_dtype(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _has_exact_binary_zero_one_values(values: list[object]) -> bool:
    if not values:
        return False
    normalized_values = {int(value) for value in pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()}
    return normalized_values == {0, 1} or normalized_values == {0} or normalized_values == {1}
