from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from napari.utils.colormaps import DirectLabelColormap

from napari_harpy._spatialdata import SpatialDataAdapter, SpatialDataTableMetadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

USER_CLASS_COLUMN = "user_class"
UNLABELED_CLASS = 0
UNLABELED_COLOR = "#80808099"
CLASS_COLORS = (
    "#1f77b4ff",
    "#d62728ff",
    "#2ca02cff",
    "#ff7f0eff",
    "#9467bdff",
    "#8c564bff",
    "#e377c2ff",
    "#17becfff",
    "#bcbd22ff",
    "#7f7f7fff",
)


class AnnotationController:
    """Manage pick-based object selection for the active segmentation layer."""

    def __init__(
        self,
        spatialdata_adapter: SpatialDataAdapter,
        on_selected_instance_changed: Callable[[int | None], None] | None = None,
    ) -> None:
        self._spatialdata_adapter = spatialdata_adapter
        self._on_selected_instance_changed = on_selected_instance_changed
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None
        self._selected_instance_id: int | None = None

    @property
    def labels_layer(self) -> Any | None:
        """Return the currently bound labels layer, if any."""
        return self._labels_layer

    @property
    def selected_instance_id(self) -> int | None:
        """Return the currently picked segmentation instance id."""
        return self._selected_instance_id

    @property
    def current_user_class(self) -> int | None:
        """Return the stored user class for the currently picked instance, if any."""
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None or self._selected_instance_id is None:
            return None

        matching_rows = self._matching_rows_mask(table.obs, metadata)
        if not matching_rows.any():
            return None

        if USER_CLASS_COLUMN not in table.obs:
            return UNLABELED_CLASS

        values = pd.to_numeric(table.obs.loc[matching_rows, USER_CLASS_COLUMN], errors="coerce").fillna(UNLABELED_CLASS)
        return int(values.iloc[0])

    @property
    def can_annotate(self) -> bool:
        """Return whether the current selection can be written back to the annotation table."""
        return (
            self._labels_layer is not None
            and self._selected_spatialdata is not None
            and self._selected_table_name is not None
            and self._selected_table_metadata is not None
            and self._selected_label_name is not None
            and self._selected_instance_id is not None
        )

    def bind(self, sdata: SpatialData | None, label_name: str | None, table_name: str | None = None) -> None:
        """Bind the controller to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and label_name is not None:
            next_layer = self._spatialdata_adapter.get_labels_layer(sdata, label_name)

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = self._spatialdata_adapter.get_table_metadata(sdata, table_name)

        layer_changed = next_layer is not self._labels_layer
        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_table_name = table_name
        self._selected_table_metadata = next_table_metadata

        if layer_changed:
            self._disconnect_selected_label_events()
            self._labels_layer = next_layer

            selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
            if selected_label_emitter is not None:
                self._clear_default_selected_label()
                selected_label_emitter.connect(self._on_layer_selected_label_changed)

            # Start without an active picked instance. We clear napari's default
            # `selected_label == 1` above so a real first click on instance `1`
            # is no longer ambiguous.
            self._set_selected_instance_id(None)

        self.refresh_layer_colors()
        self.refresh_layer_features()

    def activate_pick_mode(self) -> bool:
        """Put the bound labels layer into napari pick mode."""
        if self._labels_layer is None:
            return False

        self._spatialdata_adapter.set_active_layer(self._labels_layer)

        if hasattr(self._labels_layer, "mode"):
            self._labels_layer.mode = "pick"

        return True

    def ensure_annotation_column(self, column_name: str = USER_CLASS_COLUMN) -> None:
        """Ensure the user annotation column exists and contains integer class ids."""
        table = self._require_bound_table()
        if column_name not in table.obs:
            table.obs[column_name] = pd.Series(UNLABELED_CLASS, index=table.obs.index, dtype="int64")
            return

        table.obs[column_name] = (
            pd.to_numeric(table.obs[column_name], errors="coerce").fillna(UNLABELED_CLASS).astype("int64")
        )

    def apply_class(self, class_id: int) -> None:
        """Assign the given user class to the currently picked instance."""
        self._set_current_class(class_id)

    def clear_current_class(self) -> None:
        """Reset the current object's user class back to the unlabeled state."""
        self._set_current_class(UNLABELED_CLASS)

    def refresh_layer_colors(self) -> None:
        """Recolor the bound labels layer from the current user-class assignments."""
        if self._labels_layer is None:
            return

        color_dict: dict[int | None, str] = {
            None: UNLABELED_COLOR,
            0: "transparent",
        }
        user_class_by_instance = self._get_user_class_by_instance()
        for instance_id in _get_visible_instance_ids(self._labels_layer):
            if instance_id <= 0:
                continue

            class_id = int(user_class_by_instance.get(instance_id, UNLABELED_CLASS))
            color_dict[instance_id] = _class_to_color(class_id)

        self._labels_layer.colormap = DirectLabelColormap(color_dict=color_dict, background_value=0)
        refresh = getattr(self._labels_layer, "refresh", None)
        if callable(refresh):
            refresh()

    def refresh_layer_features(self) -> None:
        """Expose the current user classes as layer features for hover/status display."""
        if self._labels_layer is None:
            return

        instance_ids = [instance_id for instance_id in _get_visible_instance_ids(self._labels_layer) if instance_id > 0]
        user_class_by_instance = self._get_user_class_by_instance()
        features = pd.DataFrame(
            {
                "index": instance_ids,
                USER_CLASS_COLUMN: [int(user_class_by_instance.get(instance_id, UNLABELED_CLASS)) for instance_id in instance_ids],
            }
        )
        self._labels_layer.features = features

    def _disconnect_selected_label_events(self) -> None:
        selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
        disconnect = getattr(selected_label_emitter, "disconnect", None)
        if disconnect is None:
            return

        try:
            disconnect(self._on_layer_selected_label_changed)
        except (RuntimeError, TypeError, ValueError):
            return

    def _on_layer_selected_label_changed(self, event: object | None = None) -> None:
        del event
        if self._labels_layer is None:
            self._set_selected_instance_id(None)
            return

        self._set_selected_instance_id(_get_positive_selected_label(self._labels_layer))

    def _set_selected_instance_id(self, instance_id: int | None) -> None:
        if instance_id == self._selected_instance_id:
            return

        self._selected_instance_id = instance_id

        if self._on_selected_instance_changed is not None:
            self._on_selected_instance_changed(instance_id)

    def _set_current_class(self, class_id: int) -> None:
        if class_id < UNLABELED_CLASS:
            raise ValueError("Class ids must be zero or positive integers.")

        table = self._require_bound_table()
        metadata = self._require_selected_table_metadata()

        if self._selected_label_name is None:
            raise ValueError("Choose a segmentation mask before annotating.")
        if self._selected_instance_id is None:
            raise ValueError("Pick an object in the viewer before annotating.")

        self.ensure_annotation_column(USER_CLASS_COLUMN)
        matching_rows = self._matching_rows_mask(table.obs, metadata)
        if not matching_rows.any():
            raise ValueError(
                f"No table row matches segmentation `{self._selected_label_name}` and instance id `{self._selected_instance_id}`."
            )

        table.obs.loc[matching_rows, USER_CLASS_COLUMN] = int(class_id)
        self.refresh_layer_colors()
        self.refresh_layer_features()

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return self._spatialdata_adapter.get_table(self._selected_spatialdata, self._selected_table_name)

    def _require_bound_table(self) -> AnnData:
        table = self._get_bound_table()
        if table is None:
            raise ValueError("Choose an annotation table before annotating.")

        return table

    def _require_selected_table_metadata(self) -> SpatialDataTableMetadata:
        if self._selected_table_metadata is None:
            raise ValueError("Choose an annotation table before annotating.")

        return self._selected_table_metadata

    def _matching_rows_mask(self, obs: pd.DataFrame, metadata: SpatialDataTableMetadata) -> pd.Series:
        if self._selected_label_name is None or self._selected_instance_id is None:
            return pd.Series(False, index=obs.index)

        return (obs[metadata.region_key] == self._selected_label_name) & (
            obs[metadata.instance_key] == self._selected_instance_id
        )

    def _get_user_class_by_instance(self) -> pd.Series:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None:
            return pd.Series(dtype="int64")

        region_rows = table.obs.loc[table.obs[metadata.region_key] == self._selected_label_name, [metadata.instance_key]].copy()
        if USER_CLASS_COLUMN in table.obs:
            region_rows[USER_CLASS_COLUMN] = (
                pd.to_numeric(
                    table.obs.loc[region_rows.index, USER_CLASS_COLUMN],
                    errors="coerce",
                )
                .fillna(UNLABELED_CLASS)
                .astype("int64")
            )
        else:
            region_rows[USER_CLASS_COLUMN] = UNLABELED_CLASS

        region_rows = region_rows.drop_duplicates(subset=[metadata.instance_key], keep="last")
        return region_rows.set_index(metadata.instance_key)[USER_CLASS_COLUMN]

    def _clear_default_selected_label(self) -> None:
        if self._labels_layer is None:
            return

        selected_label = _get_positive_selected_label(self._labels_layer)
        if selected_label is None:
            return

        try:
            self._labels_layer.selected_label = 0
        except (AttributeError, TypeError, ValueError):
            return

def _class_to_color(class_id: int) -> str:
    if class_id <= UNLABELED_CLASS:
        return UNLABELED_COLOR

    return CLASS_COLORS[(class_id - 1) % len(CLASS_COLORS)]


def _get_visible_instance_ids(layer: Any) -> list[int]:
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict) and "indices" in metadata:
        raw_indices = metadata["indices"]
    else:
        raw_indices = np.unique(np.asarray(getattr(layer, "data", np.array([], dtype=int))))

    instance_ids = {
        int(value)
        for value in np.asarray(raw_indices).ravel().tolist()
        if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer())
    }
    return sorted(instance_ids)


def _get_positive_selected_label(layer: Any) -> int | None:
    selected_label = getattr(layer, "selected_label", 0)
    instance_id = int(selected_label)
    return instance_id if instance_id > 0 else None
