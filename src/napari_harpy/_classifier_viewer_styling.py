"""Primary-label viewer styling used by object classification."""

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
)
from napari_harpy._class_palette import (
    backfill_missing_class_colors,
    normalize_class_values,
    normalize_color_sequence,
    read_series_class_categories,
    stored_palette_to_lookup,
)
from napari_harpy._classifier import PRED_CLASS_COLORS_KEY, PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN
from napari_harpy._spatialdata import (
    SpatialDataTableMetadata,
    get_table,
    get_table_metadata,
)
from napari_harpy._viewer_adapter import ViewerAdapter
from napari_harpy._viewer_overlay_styling import _build_labels_features

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

    def __init__(self, viewer_adapter: ViewerAdapter) -> None:
        self._viewer_adapter = viewer_adapter
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_coordinate_system: str | None = None
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

    def bind(
        self,
        sdata: SpatialData | None,
        label_name: str | None,
        table_name: str | None,
        coordinate_system: str | None = None,
    ) -> None:
        """Bind styling to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and label_name is not None:
            next_layer = self._viewer_adapter.get_loaded_primary_labels_layer(
                sdata,
                label_name,
                coordinate_system,
            )

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        self._labels_layer = next_layer
        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_coordinate_system = coordinate_system
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
            category_column = USER_CLASS_COLUMN
            colors_key = USER_CLASS_COLORS_KEY
            if self._color_by == COLOR_BY_PRED_CLASS:
                category_column = PRED_CLASS_COLUMN
                colors_key = PRED_CLASS_COLORS_KEY

            class_color_lookup = self._get_class_color_lookup(
                category_column=category_column,
                colors_key=colors_key,
                extra_class_values=class_by_instance,
                unlabeled_class=UNLABELED_CLASS,
                unlabeled_color=UNLABELED_COLOR,
            )
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

        instance_key = "instance_id"  # defensive fallback for the no metadata case.
        if self._selected_table_metadata is not None:
            instance_key = self._selected_table_metadata.instance_key
        self._labels_layer.features = _build_labels_features(
            self._get_region_feature_rows(),
            instance_key=instance_key,
        )

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _get_region_rows_by_instance(self) -> pd.DataFrame:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None:
            return pd.DataFrame(index=pd.Index([], dtype="int64", name="index"))

        region_rows = table.obs.loc[table.obs[metadata.region_key] == self._selected_label_name].copy()
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
            feature_rows[USER_CLASS_COLUMN] = normalize_class_values(
                region_rows[USER_CLASS_COLUMN],
                column_name=USER_CLASS_COLUMN,
                unlabeled_class=UNLABELED_CLASS,
            )
        else:
            feature_rows[USER_CLASS_COLUMN] = pd.Series(
                UNLABELED_CLASS,
                index=feature_rows.index,
                dtype="int64",
            )

        if PRED_CLASS_COLUMN in region_rows:
            feature_rows[PRED_CLASS_COLUMN] = normalize_class_values(
                region_rows[PRED_CLASS_COLUMN],
                column_name=PRED_CLASS_COLUMN,
                unlabeled_class=UNLABELED_CLASS,
            )
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

    def _get_class_color_lookup(
        self,
        *,
        category_column: str,
        colors_key: str,
        unlabeled_class: int = UNLABELED_CLASS,
        unlabeled_color: str = UNLABELED_COLOR,
        extra_class_values: pd.Series | None = None,
    ) -> dict[int, str]:
        """Build a class-id -> color lookup for a discrete table column.

        Stored palettes in ``table.uns[colors_key]`` are interpreted in category order
        and used first. Any missing class ids are then backfilled with the deterministic
        default class palette so labels coloring stays stable when palette state is
        missing or incomplete.
        """
        table = self._get_bound_table()

        categories = {unlabeled_class}
        # Include every class id currently present in the bound table column, not just the active region.
        if table is not None and category_column in table.obs:
            categories.update(
                normalize_class_values(
                    table.obs[category_column],
                    column_name=category_column,
                    unlabeled_class=unlabeled_class,
                ).tolist()
            )
        if extra_class_values is not None:
            categories.update(
                normalize_class_values(
                    extra_class_values,
                    column_name=extra_class_values.name or category_column,
                    unlabeled_class=unlabeled_class,
                ).tolist()
            )

        sorted_categories = sorted(int(class_id) for class_id in categories)
        # Fall back to deterministic class-id colors when no table-backed palette is available.
        # This branch is just a safety net, and typically does not happen.
        if table is None or category_column not in table.obs:
            # In the happy path, `set_class_annotation_state(...)` has already kept the stored palette complete.
            return backfill_missing_class_colors(
                {unlabeled_class: unlabeled_color},
                sorted_categories,
                unlabeled_class=unlabeled_class,
                unlabeled_color=unlabeled_color,
            )

        column_categories = read_series_class_categories(
            table.obs[category_column],
            column_name=category_column,
            unlabeled_class=unlabeled_class,
        )
        existing_colors = normalize_color_sequence(table.uns.get(colors_key))
        # Convert the stored ordered palette into an explicit class-id -> color lookup.
        lookup = stored_palette_to_lookup(
            column_categories,
            existing_colors,
            unlabeled_class=unlabeled_class,
            unlabeled_color=unlabeled_color,
        )

        # Backfill any missing class ids with the deterministic default class palette.
        # In the happy path, `sync_class_palette_state(...)` has already covered every stored category.
        return backfill_missing_class_colors(
            lookup,
            sorted_categories,
            unlabeled_class=unlabeled_class,
            unlabeled_color=unlabeled_color,
        )


def _to_numeric_values(values: pd.Series, column_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
    numeric_values.name = column_name
    return numeric_values
