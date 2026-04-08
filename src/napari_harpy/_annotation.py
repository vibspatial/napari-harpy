from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import rcParams
from scanpy.plotting.palettes import default_20, default_28, default_102

from napari_harpy._spatialdata import SpatialDataAdapter, SpatialDataTableMetadata

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

USER_CLASS_COLUMN = "user_class"
USER_CLASS_COLORS_KEY = f"{USER_CLASS_COLUMN}_colors"
UNLABELED_CLASS = 0
UNLABELED_COLOR = "#80808099"


class AnnotationController:
    """Manage pick-based object selection for the active segmentation layer."""

    def __init__(
        self,
        spatialdata_adapter: SpatialDataAdapter,
        on_selected_instance_changed: Callable[[int | None], None] | None = None,
        on_annotation_changed: Callable[[], None] | None = None,
    ) -> None:
        self._spatialdata_adapter = spatialdata_adapter
        self._on_selected_instance_changed = on_selected_instance_changed
        self._on_annotation_changed = on_annotation_changed
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None
        self._selected_instance_id: int | None = None
        self._mouse_pick_callback: Callable[[Any, Any], None] = self._on_layer_mouse_pick

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

        values = _to_user_class_values(table.obs.loc[matching_rows, USER_CLASS_COLUMN])
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
            self._disconnect_mouse_pick_events()
            self._labels_layer = next_layer

            selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
            if selected_label_emitter is not None:
                self._clear_default_selected_label()
                selected_label_emitter.connect(self._on_layer_selected_label_changed)
            # Attach our own picker for every bound labels layer because
            # napari marks multiscale labels as non-editable, which disables
            # napari's native pick-mode selection for those layers.
            self._connect_mouse_pick_events()

            # Start without an active picked instance. We clear napari's default
            # `selected_label == 1` above so a real first click on instance `1`
            # is no longer ambiguous.
            self._set_selected_instance_id(None)

        self._normalize_existing_annotation_state()

    def activate_layer(self) -> bool:
        """Activate the bound labels layer for annotation interactions."""
        if self._labels_layer is None:
            return False

        self._spatialdata_adapter.set_active_layer(self._labels_layer)

        # We always attach our own mouse picker in `bind()` because napari
        # does not support pick mode for multiscale labels layers. For
        # single-scale editable labels we still request napari's native pick
        # mode so the cursor/interaction state stays aligned with normal
        # labels UX.
        if hasattr(self._labels_layer, "mode") and bool(getattr(self._labels_layer, "editable", False)):
            self._labels_layer.mode = "pick"

        return True

    def ensure_annotation_column(self, column_name: str = USER_CLASS_COLUMN) -> None:
        """Ensure the user annotation column exists as a categorical integer label column."""
        table = self._require_bound_table()
        if column_name not in table.obs:
            values = pd.Series(UNLABELED_CLASS, index=table.obs.index, dtype="int64", name=column_name)
            _set_user_class_annotation_state(table, values)
            return

        values = _to_user_class_values(table.obs[column_name])
        _set_user_class_annotation_state(table, values)

    def apply_class(self, class_id: int) -> None:
        """Assign the given user class to the currently picked instance."""
        self._set_current_class(class_id)

    def clear_current_class(self) -> None:
        """Reset the current object's user class back to the unlabeled state."""
        self._set_current_class(UNLABELED_CLASS)

    def _disconnect_selected_label_events(self) -> None:
        selected_label_emitter = getattr(getattr(self._labels_layer, "events", None), "selected_label", None)
        disconnect = getattr(selected_label_emitter, "disconnect", None)
        if disconnect is None:
            return

        try:
            disconnect(self._on_layer_selected_label_changed)
        except (RuntimeError, TypeError, ValueError):
            return

    def _connect_mouse_pick_events(self) -> None:
        callbacks = getattr(self._labels_layer, "mouse_drag_callbacks", None)
        if callbacks is None or self._mouse_pick_callback in callbacks:
            return

        callbacks.append(self._mouse_pick_callback)

    def _disconnect_mouse_pick_events(self) -> None:
        callbacks = getattr(self._labels_layer, "mouse_drag_callbacks", None)
        if callbacks is None:
            return

        try:
            callbacks.remove(self._mouse_pick_callback)
        except ValueError:
            return

    def _on_layer_mouse_pick(self, layer: Any, event: object | None = None) -> None:
        if layer is not self._labels_layer:
            return

        instance_id = _get_positive_label_from_mouse_event(layer, event)
        if instance_id is None:
            return

        try:
            layer.selected_label = instance_id
        except (AttributeError, TypeError, ValueError):
            self._set_selected_instance_id(instance_id)

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

        user_class_values = _to_user_class_values(table.obs[USER_CLASS_COLUMN])
        user_class_values.loc[matching_rows] = int(class_id)
        _set_user_class_annotation_state(table, user_class_values)
        if self._on_annotation_changed is not None:
            self._on_annotation_changed()

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

    def _normalize_existing_annotation_state(self) -> None:
        table = self._get_bound_table()
        if table is None or USER_CLASS_COLUMN not in table.obs:
            return

        self.ensure_annotation_column(USER_CLASS_COLUMN)

def _default_user_class_colors(categories: Sequence[int]) -> list[str]:
    """Return colors for the given user classes.

    Class ``0`` is always the reserved unlabeled color. Positive class ids are
    mapped onto the default categorical palette starting at index ``0`` for
    class ``1``.
    """
    return [
        UNLABELED_COLOR if _is_unlabeled_class(class_id) else _default_labeled_user_class_color(class_id)
        for class_id in categories
    ]


def _default_labeled_user_class_color(class_id: int) -> str:
    palette_index = _user_class_palette_index(class_id)
    palette = _default_labeled_user_class_colors(palette_index + 1)
    return palette[palette_index]


def _is_unlabeled_class(class_id: int) -> bool:
    return class_id == UNLABELED_CLASS


def _user_class_palette_index(class_id: int) -> int:
    if _is_unlabeled_class(class_id):
        raise ValueError("The unlabeled class does not map to the labeled-user-class palette.")
    if class_id < UNLABELED_CLASS:
        raise ValueError("Class ids must be zero or positive integers.")

    return class_id - 1


def _default_labeled_user_class_colors(length: int) -> list[str]:
    """Return the default categorical palette used by spatialdata-plot/scanpy."""
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        color_cycle = rcParams["axes.prop_cycle"]()
        palette = [next(color_cycle)["color"] for _ in range(length)]
    elif length <= 20:
        palette = list(default_20)
    elif length <= 28:
        palette = list(default_28)
    elif length <= len(default_102):
        palette = list(default_102)
    else:
        palette = ["grey" for _ in range(length)]

    return palette[:length]


def _to_user_class_values(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values.astype("string"), errors="coerce").fillna(UNLABELED_CLASS).astype("int64")
    numeric_values.name = USER_CLASS_COLUMN
    return numeric_values


def _get_user_class_categories(values: pd.Series) -> list[int]:
    normalized_values = _to_user_class_values(values)
    return sorted({UNLABELED_CLASS, *normalized_values.tolist()})


def _set_user_class_annotation_state(table: AnnData, values: pd.Series) -> None:
    """Normalize `user_class` state and regenerate the corresponding color palette.

    For now, `napari-harpy` treats `user_class_colors` as derived state from the
    current class ids. If a different palette already exists, it is overwritten and
    a warning is logged so that this behavior is visible during development.
    """
    normalized_values = _to_user_class_values(values)
    categories = _get_user_class_categories(normalized_values)
    generated_colors = _default_user_class_colors(categories)
    existing_colors = _normalize_color_sequence(table.uns.get(USER_CLASS_COLORS_KEY))
    if existing_colors is not None and existing_colors != generated_colors:
        logger.warning(
            f"Overwriting existing `{USER_CLASS_COLORS_KEY}` palette in `table.uns`. "
            f"Current napari-harpy behavior regenerates this palette from `{USER_CLASS_COLUMN}` categories."
        )
    table.obs[USER_CLASS_COLUMN] = pd.Series(
        pd.Categorical(normalized_values, categories=categories),
        index=normalized_values.index,
        name=USER_CLASS_COLUMN,
    )
    table.uns[USER_CLASS_COLORS_KEY] = generated_colors


def _normalize_color_sequence(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    return [str(value)]


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


def _get_positive_label_from_mouse_event(layer: Any, event: object | None = None) -> int | None:
    get_value = getattr(layer, "get_value", None)
    if not callable(get_value) or event is None:
        return None

    dims_displayed = getattr(event, "dims_displayed", None)
    if dims_displayed is None:
        dims_displayed = list(getattr(getattr(layer, "_slice_input", None), "displayed", []))

    try:
        value = get_value(
            getattr(event, "position", None),
            view_direction=getattr(event, "view_direction", None),
            dims_displayed=dims_displayed,
            world=True,
        )
    except (AttributeError, TypeError, ValueError):
        return None

    if isinstance(value, tuple):
        if not value:
            return None
        value = value[-1]

    if value is None:
        return None

    try:
        instance_id = int(value)
    except (TypeError, ValueError):
        return None

    return instance_id if instance_id > 0 else None
