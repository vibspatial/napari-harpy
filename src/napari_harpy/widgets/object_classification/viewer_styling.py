"""Primary-label viewer styling used by object classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from napari_harpy.core.annotation import (
    UNLABELED_CLASS,
    UNLABELED_COLOR,
    USER_CLASS_COLORS_KEY,
    USER_CLASS_COLUMN,
)
from napari_harpy.core.class_palette import (
    default_class_colors,
    normalize_class_values,
    normalize_color_sequence,
)
from napari_harpy.core.spatialdata import (
    SpatialDataTableMetadata,
    get_table,
    get_table_metadata,
)
from napari_harpy.viewer._styling import MISSING_CONTINUOUS_COLOR
from napari_harpy.viewer.adapter import ViewerAdapter
from napari_harpy.viewer.labels_colormap import (
    CompactLabelColormap,
    compact_categorical_label_colormap_from_values,
    compact_continuous_label_colormap_from_values,
)
from napari_harpy.viewer.labels_styling import _build_labels_features, _get_region_rows_by_instance
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

PRED_CONFIDENCE_COLORMAP = "plasma"


class ClassStateError(ValueError):
    """Raised when object-classification class state is not canonical for styling."""


class ViewerStylingController:
    """Manage labels-layer styling from user labels and classifier outputs."""

    def __init__(self, viewer_adapter: ViewerAdapter) -> None:
        self._viewer_adapter = viewer_adapter
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_labels_name: str | None = None
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
        labels_name: str | None,
        table_name: str | None,
        coordinate_system: str | None = None,
    ) -> None:
        """Bind styling to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and labels_name is not None:
            next_layer = self._viewer_adapter.get_loaded_primary_labels_layer(
                sdata,
                labels_name,
                coordinate_system,
            )

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        self._labels_layer = next_layer
        self._selected_spatialdata = sdata
        self._selected_labels_name = labels_name
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

        if self._color_by == COLOR_BY_PRED_CONFIDENCE:
            self._labels_layer.colormap = compact_continuous_label_colormap_from_values(
                feature_rows[PRED_CONFIDENCE_COLUMN],
                colormap_name=PRED_CONFIDENCE_COLORMAP,
                missing_color=MISSING_CONTINUOUS_COLOR,
                default_color=MISSING_CONTINUOUS_COLOR,
                value_range=(0.0, 1.0),
            )
        else:
            class_values_by_instance = feature_rows[self._color_by]
            category_column = USER_CLASS_COLUMN
            colors_key = USER_CLASS_COLORS_KEY
            if self._color_by == COLOR_BY_PRED_CLASS:
                category_column = PRED_CLASS_COLUMN
                colors_key = PRED_CLASS_COLORS_KEY

            # Object-classification values are class ids, not colors. Resolve
            # them through the class palette stored in table `.uns`; generic
            # styled-labels coloring intentionally uses a separate color path.
            class_color_lookup = self._get_class_color_lookup(
                category_column=category_column,
                colors_key=colors_key,
                observed_class_values=class_values_by_instance,
                unlabeled_class=UNLABELED_CLASS,
                unlabeled_color=UNLABELED_COLOR,
            )
            unlabeled_color = class_color_lookup.get(UNLABELED_CLASS)
            if unlabeled_color is None:
                raise ClassStateError(
                    f"Class palette `{colors_key}` is missing the unlabeled class `{UNLABELED_CLASS}`. "
                    "Rebind or reload the table in the Object Classification widget to regenerate Harpy class state."
                )
            if self._color_by == COLOR_BY_USER_CLASS:
                # User-class `0` means "unlabeled": leave those labels out of
                # the compact mapping so they fall through to the default
                # unlabeled color. Prediction class `0` stays explicit.
                class_values_by_instance = class_values_by_instance[class_values_by_instance != UNLABELED_CLASS]

            categories = sorted(class_color_lookup)
            class_values = pd.Series(
                pd.Categorical(
                    class_values_by_instance.to_numpy(dtype=np.int64, copy=False),
                    categories=categories,
                ),
                index=class_values_by_instance.index,
                name=class_values_by_instance.name,
            )
            self._labels_layer.colormap = compact_categorical_label_colormap_from_values(
                class_values,
                categories=categories,
                palette=[class_color_lookup[class_id] for class_id in categories],
                default_color=unlabeled_color,
                background_value=0,
            )
        self._viewer_adapter.sync_labels_display_after_colormap_change(self._labels_layer)

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

    def refresh_user_class_colormap_and_feature(self, change: UserClassAnnotationChange) -> bool:
        """Refresh one user-class annotation in labels colors and features.

        Returns ``True`` when the row-scoped update was fully applied. Returns
        ``False`` when the caller should fall back to a normal full refresh.
        """
        if self._labels_layer is None or self._color_by != COLOR_BY_USER_CLASS:
            return False

        feature_rows = self._build_user_class_annotation_features(change)
        if feature_rows is None:
            return False

        self._refresh_compact_user_class_colormap_and_feature(change, feature_rows)
        return True

    def _refresh_compact_user_class_colormap_and_feature(
        self,
        change: UserClassAnnotationChange,
        feature_rows: pd.DataFrame,
    ) -> bool:
        colormap = getattr(self._labels_layer, "colormap", None)
        if not isinstance(colormap, CompactLabelColormap):
            raise RuntimeError(
                "Cannot update user-class annotation colors row-scoped: "
                "the labels layer is not using CompactLabelColormap."
            )

        refresh = self._labels_layer.refresh

        instance_id = int(change.instance_id)
        class_id = int(change.class_id)
        if class_id == UNLABELED_CLASS:
            result = colormap.remove_label(instance_id)
        else:
            class_color_lookup = self._get_valid_user_class_color_lookup()
            if class_color_lookup is None:
                raise RuntimeError("Cannot update compact user-class coloring without valid user-class colors.")
            class_color = class_color_lookup.get(class_id)
            if class_color is None:
                raise RuntimeError(f"Cannot update compact user-class coloring: class `{class_id}` has no color.")
            result = colormap.set_label_value(instance_id, class_id, value_color=class_color)

        self._labels_layer.features = feature_rows
        if result.texture_table_changed:
            # Only brand-new classes append a new texture-code -> RGBA row.
            # Notify vispy to upload the expanded lookup texture. Existing-
            # class edits reuse an already uploaded texture row, so they only
            # need the layer refresh below.
            self._labels_layer.events.colormap()
        # The compact mapping was mutated in place; repaint the layer without
        # asking napari to recompute the layer extent.
        refresh(extent=False)
        return True

    def refresh_user_class_feature_only(self, change: UserClassAnnotationChange) -> bool:
        """Refresh one user-class feature value without repainting label colors.

        This is the direct-annotation fast path for prediction color modes:
        annotation changes `user_class`, while `pred_class`/`pred_confidence`
        colors are refreshed only when the classifier writes predictions.
        """
        if self._labels_layer is None:
            return False

        feature_rows = self._build_user_class_annotation_features(change)
        if feature_rows is None:
            return False

        self._labels_layer.features = feature_rows
        return True

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

    def _get_valid_user_class_color_lookup(self) -> dict[int, np.ndarray] | None:
        table = self._get_bound_table()
        if table is None or USER_CLASS_COLUMN not in table.obs:
            return None

        return self._get_class_color_lookup(
            category_column=USER_CLASS_COLUMN,
            colors_key=USER_CLASS_COLORS_KEY,
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
        if table is None or metadata is None or self._selected_labels_name is None:
            return pd.DataFrame(index=pd.Index([], dtype="int64", name="index"))

        region_rows, _ = _get_region_rows_by_instance(table, metadata, self._selected_labels_name)
        return region_rows

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
        observed_class_values: pd.Series | None = None,
    ) -> dict[int, np.ndarray]:
        """Read the canonical class-id -> RGBA lookup for a prepared class column.

        Object-classification binding/adoption owns normalization of external
        class columns and palettes. Once styling runs, this path is intentionally
        strict: it validates that the bound table still contains Harpy's
        canonical categorical column and default palette, and raises an
        actionable ``ClassStateError`` instead of silently repairing drift.
        This keeps styling read-only with respect to class metadata and makes
        post-bind table edits visible to the widget warning status card.
        """
        table = self._get_bound_table()
        observed_class_ids: set[int] = set()
        if observed_class_values is not None:
            observed_class_ids = _read_class_values_without_normalizing(
                observed_class_values,
                unlabeled_class=unlabeled_class,
            )
            if observed_class_ids is None:
                raise ClassStateError(
                    f"Cannot style labels by `{category_column}` because the current feature rows contain "
                    "non-canonical class values. Rebind or reload the table in the Object Classification widget "
                    "before styling."
                )

        if table is None or category_column not in table.obs:
            only_unlabeled_classes_observed = observed_class_ids.issubset({unlabeled_class})
            if category_column == USER_CLASS_COLUMN and only_unlabeled_classes_observed:
                return {unlabeled_class: _rgba_array(unlabeled_color)}
            raise ClassStateError(
                f"Cannot style labels by `{category_column}` because the bound table is not prepared. "
                "Rebind or reload the table in the Object Classification widget to canonicalize class state."
            )

        categories = _read_canonical_class_categories(
            table.obs[category_column],
            column_name=category_column,
            unlabeled_class=unlabeled_class,
        )
        unknown_observed_classes = sorted(observed_class_ids - set(categories))
        if unknown_observed_classes:
            raise ClassStateError(
                f"Cannot style labels by `{category_column}` because observed class ids "
                f"{unknown_observed_classes} are not present in the prepared categorical column. "
                "Rebind or reload the table in the Object Classification widget to canonicalize class state."
            )

        return _read_canonical_class_color_lookup(
            categories,
            table.uns.get(colors_key),
            colors_key=colors_key,
            column_name=category_column,
            unlabeled_class=unlabeled_class,
            unlabeled_color=unlabeled_color,
        )


def _read_canonical_class_categories(
    values: pd.Series,
    *,
    column_name: str,
    unlabeled_class: int,
) -> list[int]:
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ClassStateError(
            f"`{column_name}` must be a categorical integer column before labels can be styled. "
            "Rebind or reload the table in the Object Classification widget to canonicalize class state."
        )

    categories: list[int] = []
    for category in values.cat.categories:
        if isinstance(category, (bool, np.bool_)) or not isinstance(category, (int, np.integer)):
            raise ClassStateError(
                f"`{column_name}` has non-integer categories. Rebind or reload the table in the Object "
                "Classification widget to canonicalize class state."
            )
        class_id = int(category)
        if class_id < unlabeled_class or category != class_id:
            raise ClassStateError(
                f"`{column_name}` categories must be zero or positive integer class ids. Rebind or reload "
                "the table in the Object Classification widget to canonicalize class state."
            )
        categories.append(class_id)

    if categories != sorted(categories) or len(categories) != len(set(categories)):
        raise ClassStateError(
            f"`{column_name}` categories are not in canonical sorted order. Rebind or reload the table in the "
            "Object Classification widget to canonicalize class state."
        )
    if unlabeled_class not in categories:
        raise ClassStateError(
            f"`{column_name}` is missing the unlabeled class `{unlabeled_class}`. Rebind or reload the table in "
            "the Object Classification widget to canonicalize class state."
        )
    if bool((values.cat.codes.to_numpy(copy=False) < 0).any()):
        raise ClassStateError(
            f"`{column_name}` contains missing categorical values. Rebind or reload the table in the Object "
            "Classification widget to canonicalize class state."
        )

    return categories


def _read_canonical_class_color_lookup(
    categories: list[int],
    stored_colors: Any,
    *,
    column_name: str,
    colors_key: str,
    unlabeled_class: int,
    unlabeled_color: str,
) -> dict[int, np.ndarray]:
    stored_color_list = normalize_color_sequence(stored_colors)
    expected_colors = default_class_colors(
        categories,
        unlabeled_class=unlabeled_class,
        unlabeled_color=unlabeled_color,
    )
    if stored_color_list is None:
        raise ClassStateError(
            f"Missing class palette `{colors_key}` for `{column_name}`. Rebind or reload the table in the Object "
            "Classification widget to regenerate Harpy class state."
        )
    if len(stored_color_list) != len(categories):
        raise ClassStateError(
            f"Class palette `{colors_key}` has {len(stored_color_list)} colors, but `{column_name}` has "
            f"{len(categories)} categories. Rebind or reload the table in the Object Classification widget to "
            "regenerate Harpy class state."
        )
    if stored_color_list != expected_colors:
        raise ClassStateError(
            f"Class palette `{colors_key}` no longer matches Harpy default colors for `{column_name}`. "
            "Rebind or reload the table in the Object Classification widget to regenerate Harpy class state."
        )

    try:
        return {class_id: _rgba_array(color) for class_id, color in zip(categories, expected_colors, strict=True)}
    except ValueError as error:
        raise ClassStateError(
            f"Class palette `{colors_key}` contains an invalid color. Rebind or reload the table in the Object "
            "Classification widget to regenerate Harpy class state."
        ) from error


def _to_numeric_values(values: pd.Series, column_name: str) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype("float64")
    numeric_values.name = column_name
    return numeric_values


def _read_class_values_without_normalizing(values: pd.Series, *, unlabeled_class: int) -> set[int] | None:
    raw_values = values.to_numpy(copy=False)
    if np.issubdtype(raw_values.dtype, np.integer) and not np.issubdtype(raw_values.dtype, np.bool_):
        # Happy path for normalized user/prediction classes: use NumPy instead
        # of scanning hundreds of thousands of class ids in Python.
        if len(raw_values) == 0:
            return set()
        if int(np.min(raw_values)) < unlabeled_class:
            return None
        return {int(value) for value in np.unique(raw_values)}

    categories: set[int] = set()
    for value in raw_values:
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


def _rgba_color_lookup(color_lookup: dict[int, Any]) -> dict[int, np.ndarray]:
    return {class_id: _rgba_array(color) for class_id, color in color_lookup.items()}


def _rgba_array(color: Any) -> np.ndarray:
    return np.asarray(to_rgba(color), dtype=np.float32)
