from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib import colormaps
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy._annotation import (
    UNLABELED_CLASS,
    UNLABELED_COLOR,
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
    _default_labeled_user_class_color,
    _normalize_color_sequence,
)
from napari_harpy._classifier import PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN
from napari_harpy._spatialdata import SpatialDataAdapter, SpatialDataTableMetadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

COLOR_BY_USER_CLASS = USER_CLASS_COLUMN
COLOR_BY_PRED_CLASS = PRED_CLASS_COLUMN
COLOR_BY_PRED_CONFIDENCE = PRED_CONFIDENCE_COLUMN
COLOR_BY_OPTIONS = (
    COLOR_BY_USER_CLASS,
    COLOR_BY_PRED_CLASS,
    COLOR_BY_PRED_CONFIDENCE,
)

MISSING_CONTINUOUS_COLOR = "#80808099"
PRED_CONFIDENCE_COLORMAP = "viridis"


class ViewerStylingController:
    """Manage labels-layer styling from user labels and classifier outputs."""

    def __init__(self, spatialdata_adapter: SpatialDataAdapter) -> None:
        self._spatialdata_adapter = spatialdata_adapter
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None
        self._color_by = COLOR_BY_USER_CLASS

    @property
    def color_by(self) -> str:
        """Return the current labels-layer coloring mode."""
        return self._color_by

    @property
    def labels_layer(self) -> Any | None:
        """Return the currently styled labels layer, if any."""
        return self._labels_layer

    def bind(self, sdata: SpatialData | None, label_name: str | None, table_name: str | None) -> None:
        """Bind styling to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and label_name is not None:
            next_layer = self._spatialdata_adapter.get_labels_layer(sdata, label_name)

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = self._spatialdata_adapter.get_table_metadata(sdata, table_name)

        self._labels_layer = next_layer
        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_table_name = table_name
        self._selected_table_metadata = next_table_metadata

    def set_color_by(self, color_by: str) -> None:
        """Set the active coloring mode for the bound labels layer."""
        if color_by not in COLOR_BY_OPTIONS:
            raise ValueError(f"Unsupported color mode `{color_by}`.")

        self._color_by = color_by

    def refresh(self) -> None:
        """Refresh labels-layer colors and features from the current table state."""
        self.refresh_layer_colors()
        self.refresh_layer_features()

    def refresh_layer_colors(self) -> None:
        """Apply the current `color_by` mode to the bound labels layer."""
        if self._labels_layer is None:
            return

        feature_rows = self._get_region_feature_rows()
        color_dict: dict[int | None, Any] = {
            None: UNLABELED_COLOR,
            0: "transparent",
        }
        instance_ids = feature_rows.index.to_numpy(dtype=np.int64, copy=False)

        if self._color_by == COLOR_BY_PRED_CONFIDENCE:
            cmap = colormaps[PRED_CONFIDENCE_COLORMAP]
            for instance_id in instance_ids:
                confidence = float(feature_rows.at[instance_id, PRED_CONFIDENCE_COLUMN])
                if np.isnan(confidence):
                    color_dict[instance_id] = MISSING_CONTINUOUS_COLOR
                else:
                    color_dict[instance_id] = cmap(float(np.clip(confidence, 0.0, 1.0)))
        else:
            class_by_instance = feature_rows[self._color_by]
            class_color_lookup = self._get_class_color_lookup(extra_class_values=class_by_instance)
            unlabeled_color = class_color_lookup.get(UNLABELED_CLASS, UNLABELED_COLOR)
            color_dict[None] = unlabeled_color
            for instance_id in instance_ids:
                class_id = int(class_by_instance.at[instance_id])
                color_dict[instance_id] = class_color_lookup.get(class_id, unlabeled_color)

        self._labels_layer.colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)
        refresh = getattr(self._labels_layer, "refresh", None)
        if callable(refresh):
            refresh()

    def refresh_layer_features(self) -> None:
        """Expose current label and prediction values as napari layer features."""
        if self._labels_layer is None:
            return

        self._labels_layer.features = self._get_region_feature_rows().reset_index()

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return self._spatialdata_adapter.get_table(self._selected_spatialdata, self._selected_table_name)

    def _get_region_rows_by_instance(self) -> pd.DataFrame:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None:
            return pd.DataFrame(index=pd.Index([], dtype="int64", name="index"))

        region_rows = table.obs.loc[
            table.obs[metadata.region_key] == self._selected_label_name
        ].copy()
        instance_ids = pd.to_numeric(region_rows[metadata.instance_key], errors="coerce")
        region_rows = region_rows.loc[instance_ids.notna()].copy()
        region_rows[metadata.instance_key] = instance_ids.loc[region_rows.index].astype("int64")
        region_rows = region_rows.loc[region_rows[metadata.instance_key] > 0].copy()
        region_rows = region_rows.drop_duplicates(subset=[metadata.instance_key], keep="last")
        return region_rows.set_index(metadata.instance_key)

    def _get_region_feature_rows(self) -> pd.DataFrame:
        region_rows = self._get_region_rows_by_instance()
        feature_rows = pd.DataFrame(index=region_rows.index.astype("int64", copy=False))
        feature_rows.index.name = "index"

        if USER_CLASS_COLUMN in region_rows:
            feature_rows[USER_CLASS_COLUMN] = _to_class_values(region_rows[USER_CLASS_COLUMN], USER_CLASS_COLUMN)
        else:
            feature_rows[USER_CLASS_COLUMN] = pd.Series(
                UNLABELED_CLASS,
                index=feature_rows.index,
                dtype="int64",
            )

        if PRED_CLASS_COLUMN in region_rows:
            feature_rows[PRED_CLASS_COLUMN] = _to_class_values(region_rows[PRED_CLASS_COLUMN], PRED_CLASS_COLUMN)
        else:
            feature_rows[PRED_CLASS_COLUMN] = pd.Series(
                UNLABELED_CLASS,
                index=feature_rows.index,
                dtype="int64",
            )

        if PRED_CONFIDENCE_COLUMN in region_rows:
            feature_rows[PRED_CONFIDENCE_COLUMN] = _to_numeric_values(
                region_rows[PRED_CONFIDENCE_COLUMN],
                PRED_CONFIDENCE_COLUMN,
            )
        else:
            feature_rows[PRED_CONFIDENCE_COLUMN] = pd.Series(
                np.nan,
                index=feature_rows.index,
                dtype="float64",
            )

        return feature_rows

    def _get_class_color_lookup(self, *, extra_class_values: pd.Series | None = None) -> dict[int, str]:
        table = self._get_bound_table()
        categories = {UNLABELED_CLASS}
        if table is not None and USER_CLASS_COLUMN in table.obs:
            categories.update(_to_class_values(table.obs[USER_CLASS_COLUMN], USER_CLASS_COLUMN).tolist())
        if extra_class_values is not None:
            categories.update(_to_class_values(extra_class_values, extra_class_values.name or USER_CLASS_COLUMN).tolist())

        sorted_categories = sorted(int(class_id) for class_id in categories)
        lookup = {UNLABELED_CLASS: UNLABELED_COLOR}

        if table is None or USER_CLASS_COLUMN not in table.obs:
            for class_id in sorted_categories:
                if class_id == UNLABELED_CLASS:
                    continue
                lookup[class_id] = _default_labeled_user_class_color(class_id)
            return lookup

        user_categories = _get_table_user_class_categories(table.obs[USER_CLASS_COLUMN])
        existing_colors = _normalize_color_sequence(table.uns.get(USER_CLASS_COLORS_KEY))
        if existing_colors is not None:
            for class_id, color in zip(user_categories, existing_colors[: len(user_categories)], strict=False):
                lookup[int(class_id)] = color

        for class_id in sorted_categories:
            if class_id == UNLABELED_CLASS:
                continue
            lookup.setdefault(class_id, _default_labeled_user_class_color(class_id))

        return lookup


def _to_class_values(values: pd.Series, column_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values.astype("string"), errors="coerce").fillna(UNLABELED_CLASS).astype("int64")
    numeric_values.name = column_name
    return numeric_values


def _to_numeric_values(values: pd.Series, column_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
    numeric_values.name = column_name
    return numeric_values


def _get_table_user_class_categories(values: pd.Series) -> list[int]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [int(value) for value in values.cat.categories]

    return sorted({UNLABELED_CLASS, *_to_class_values(values, USER_CLASS_COLUMN).tolist()})
