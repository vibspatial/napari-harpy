"""Primary-label viewer styling used by object classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib import colormaps
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy.core.annotation import (
    UNLABELED_CLASS,
    UNLABELED_COLOR,
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
)
from napari_harpy.core.class_palette import (
    backfill_missing_class_colors,
    normalize_class_values,
    normalize_color_sequence,
    read_series_class_categories,
    stored_palette_to_lookup,
)
from napari_harpy.core.spatialdata import (
    SpatialDataTableMetadata,
    get_table,
    get_table_metadata,
)
from napari_harpy.viewer.adapter import ViewerAdapter
from napari_harpy.viewer.overlay_styling import _build_labels_features
from napari_harpy.widgets.object_classification.controller import (
    PRED_CLASS_COLORS_KEY,
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

    from napari_harpy.widgets.object_classification.annotation_controller import UserClassAnnotationChange

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
        if self._labels_layer is None:
            return

        feature_rows = self._get_region_feature_rows()
        self.refresh_layer_colors(feature_rows=feature_rows)
        self.refresh_layer_features(feature_rows=feature_rows)

    def refresh_layer_colors(self, *, feature_rows: pd.DataFrame | None = None) -> None:
        """Apply the current `color_by` mode to the bound labels layer.

        Direct annotation happy paths should use row-scoped refresh helpers
        instead. Prediction color repainting should reach this full refresh path
        when the classifier actually writes predictions, via
        `ObjectClassificationWidget._on_classifier_prediction_state_changed()`.
        """
        if self._labels_layer is None:
            return

        if feature_rows is None:
            feature_rows = self._get_region_feature_rows()

        color_dict = _base_labels_color_dict(UNLABELED_COLOR)
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
            color_dict = _base_labels_color_dict(unlabeled_color)
            if self._color_by == COLOR_BY_USER_CLASS:
                class_by_instance = class_by_instance[class_by_instance != UNLABELED_CLASS]

            for instance_id, class_id_value in class_by_instance.items():
                class_id = int(class_id_value)
                if self._color_by == COLOR_BY_USER_CLASS and class_id == UNLABELED_CLASS:
                    continue
                color_dict[int(instance_id)] = class_color_lookup.get(class_id, unlabeled_color)

        self._labels_layer.colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)
        refresh = getattr(self._labels_layer, "refresh", None)
        if callable(refresh):
            refresh()

    def refresh_layer_features(self, *, feature_rows: pd.DataFrame | None = None) -> None:
        """Expose current label and prediction values as napari layer features."""
        if self._labels_layer is None:
            return

        if feature_rows is None:
            feature_rows = self._get_region_feature_rows()

        instance_key = "instance_id"  # defensive fallback for the no metadata case.
        if self._selected_table_metadata is not None:
            instance_key = self._selected_table_metadata.instance_key
        self._labels_layer.features = _build_labels_features(
            feature_rows,
            instance_key=instance_key,
        )

    def refresh_user_class_annotation(self, change: UserClassAnnotationChange) -> bool:
        """Refresh one user-class annotation in labels colors and features.

        Returns ``True`` when the row-scoped update was fully applied. Returns
        ``False`` when the caller should fall back to a normal full refresh.
        """
        if self._labels_layer is None or self._color_by != COLOR_BY_USER_CLASS:
            return False
        if change.class_id < UNLABELED_CLASS:
            return False

        # `set_user_class_for_rows(...)` has already added any new category and
        # synced `USER_CLASS_COLORS_KEY`, so newly introduced classes can use the
        # strict palette lookup below without full-column normalization.
        color_dict = self._build_user_class_annotation_color_dict(change)
        feature_rows = self._build_user_class_annotation_features(change)
        if color_dict is None or feature_rows is None:
            return False

        self._labels_layer.colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)
        self._labels_layer.features = feature_rows
        refresh = getattr(self._labels_layer, "refresh", None)
        if callable(refresh):
            refresh()

        return True

    def _build_user_class_annotation_color_dict(
        self,
        change: UserClassAnnotationChange,
    ) -> dict[int | None, Any] | None:
        colormap = getattr(self._labels_layer, "colormap", None)
        if not isinstance(colormap, DirectLabelColormap):
            return None

        class_color_lookup = self._get_valid_user_class_color_lookup()
        if class_color_lookup is None:
            return None

        color_dict = dict(colormap.color_dict)
        instance_id = int(change.instance_id)
        class_id = int(change.class_id)
        if class_id == UNLABELED_CLASS:
            color_dict.pop(instance_id, None)
            return color_dict

        class_color = class_color_lookup.get(class_id)
        if class_color is None:
            return None

        color_dict[instance_id] = class_color
        return color_dict

    def _build_user_class_annotation_features(
        self,
        change: UserClassAnnotationChange,
    ) -> pd.DataFrame | None:
        features = getattr(self._labels_layer, "features", None)
        if not isinstance(features, pd.DataFrame) or features.empty:
            return None
        if "index" not in features or USER_CLASS_COLUMN not in features:
            return None

        feature_index = pd.to_numeric(features["index"], errors="coerce")
        matching_rows = feature_index == int(change.instance_id)
        if int(matching_rows.sum()) != 1:
            return None

        updated_features = features.copy()
        updated_features.loc[matching_rows, USER_CLASS_COLUMN] = int(change.class_id)
        return updated_features

    def _get_valid_user_class_color_lookup(self) -> dict[int, str] | None:
        table = self._get_bound_table()
        if table is None or USER_CLASS_COLUMN not in table.obs:
            return None

        return _valid_categorical_class_color_lookup(
            table.obs[USER_CLASS_COLUMN],
            table.uns.get(USER_CLASS_COLORS_KEY),
            unlabeled_class=UNLABELED_CLASS,
            unlabeled_color=UNLABELED_COLOR,
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
        """Return normalized labels features for the selected segmentation region.

        The returned rows are indexed by label/instance id and include
        `user_class`, `pred_class`, and `pred_confidence`. This is scoped to the
        currently selected labels element, not necessarily the complete table.
        """
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

        if table is not None and category_column == USER_CLASS_COLUMN and category_column in table.obs:
            fast_lookup = _valid_categorical_class_color_lookup(
                table.obs[category_column],
                table.uns.get(colors_key),
                unlabeled_class=unlabeled_class,
                unlabeled_color=unlabeled_color,
            )
            if fast_lookup is not None:
                categories = set(fast_lookup)
                if extra_class_values is not None:
                    # Happy path: `feature_rows` already contains clean integer class ids,
                    # so we can include them without the expensive `normalize_class_values(...)`
                    # scan over the full table column again.
                    extra_class_ids = _read_class_values_without_normalizing(
                        extra_class_values,
                        unlabeled_class=unlabeled_class,
                    )
                    if extra_class_ids is None:
                        # Defensive fallback for unexpected dirty feature values. This preserves
                        # robust class-value normalization instead of trusting a corrupt fast path.
                        return self._get_class_color_lookup_from_normalized_values(
                            category_column=category_column,
                            colors_key=colors_key,
                            unlabeled_class=unlabeled_class,
                            unlabeled_color=unlabeled_color,
                            extra_class_values=extra_class_values,
                        )
                    categories.update(extra_class_ids)

                return backfill_missing_class_colors(
                    fast_lookup,
                    sorted(categories),
                    unlabeled_class=unlabeled_class,
                    unlabeled_color=unlabeled_color,
                )

        return self._get_class_color_lookup_from_normalized_values(
            category_column=category_column,
            colors_key=colors_key,
            unlabeled_class=unlabeled_class,
            unlabeled_color=unlabeled_color,
            extra_class_values=extra_class_values,
        )

    def _get_class_color_lookup_from_normalized_values(
        self,
        *,
        category_column: str,
        colors_key: str,
        unlabeled_class: int = UNLABELED_CLASS,
        unlabeled_color: str = UNLABELED_COLOR,
        extra_class_values: pd.Series | None = None,
    ) -> dict[int, str]:
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


def _valid_categorical_class_color_lookup(
    values: pd.Series,
    stored_colors: Any,
    *,
    unlabeled_class: int,
    unlabeled_color: str,
) -> dict[int, str] | None:
    categories = _read_valid_categorical_class_categories(values, unlabeled_class=unlabeled_class)
    existing_colors = normalize_color_sequence(stored_colors)
    if categories is None or existing_colors is None or len(existing_colors) != len(categories):
        return None

    return stored_palette_to_lookup(
        categories,
        existing_colors,
        unlabeled_class=unlabeled_class,
        unlabeled_color=unlabeled_color,
    )


def _read_valid_categorical_class_categories(values: pd.Series, *, unlabeled_class: int) -> list[int] | None:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        return None

    categories: list[int] = []
    for category in values.cat.categories:
        if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)):
            return None
        class_id = int(category)
        if class_id < unlabeled_class or category != class_id:
            return None
        categories.append(class_id)

    if categories != sorted(categories):
        return None
    if len(categories) != len(set(categories)):
        return None
    if unlabeled_class not in categories:
        return None
    if bool((values.cat.codes.to_numpy(copy=False) < 0).any()):
        return None

    return categories


def _read_class_values_without_normalizing(values: pd.Series, *, unlabeled_class: int) -> set[int] | None:
    categories: set[int] = set()
    for value in values.to_numpy(copy=False):
        if pd.isna(value):
            return None
        try:
            class_id = int(value)
        except (TypeError, ValueError):
            return None
        if class_id < unlabeled_class or value != class_id:
            return None
        categories.add(class_id)

    return categories


def _base_labels_color_dict(default_color: Any) -> dict[int | None, Any]:
    return {None: default_color, 0: "transparent"}
