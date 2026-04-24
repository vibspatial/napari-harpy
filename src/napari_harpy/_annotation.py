from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from napari_harpy._class_palette import (
    DEFAULT_UNLABELED_CLASS,
    DEFAULT_UNLABELED_COLOR,
    normalize_class_values,
    set_class_annotation_state,
)
from napari_harpy._spatialdata import (
    SpatialDataTableMetadata,
    get_table,
    get_table_metadata,
)
from napari_harpy._viewer_adapter import ViewerAdapter

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

USER_CLASS_COLUMN = "user_class"
USER_CLASS_COLORS_KEY = f"{USER_CLASS_COLUMN}_colors"
UNLABELED_CLASS = DEFAULT_UNLABELED_CLASS
UNLABELED_COLOR = DEFAULT_UNLABELED_COLOR


@dataclass(frozen=True)
class _SelectionTableState:
    """Current selection state relative to the bound annotation table."""

    table: AnnData | None
    metadata: SpatialDataTableMetadata | None
    label_name: str | None
    table_name: str | None
    instance_id: int | None
    matching_rows: pd.Series | None

    @property
    def has_annotation_binding(self) -> bool:
        return (
            self.table is not None
            and self.metadata is not None
            and self.label_name is not None
            and self.table_name is not None
            and self.instance_id is not None
        )

    @property
    def has_table_row(self) -> bool:
        return self.matching_rows is not None and bool(self.matching_rows.any())

    @property
    def instance_key_name(self) -> str | None:
        return None if self.metadata is None else self.metadata.instance_key

    @property
    def missing_table_row_message(self) -> str | None:
        instance_key_name = self.instance_key_name
        if (
            self.label_name is None
            or self.table_name is None
            or self.instance_id is None
            or instance_key_name is None
            or self.has_table_row
        ):
            return None

        return (
            f"Selected {instance_key_name} {self.instance_id} is not present in annotation table "
            f"`{self.table_name}` for segmentation `{self.label_name}` and cannot receive a user class."
        )


class AnnotationController:
    """Manage pick-based object selection for the active segmentation layer."""

    def __init__(
        self,
        viewer_adapter: ViewerAdapter,
        on_selected_instance_changed: Callable[[int | None], None] | None = None,
        on_annotation_changed: Callable[[], None] | None = None,
    ) -> None:
        self._viewer_adapter = viewer_adapter
        self._on_selected_instance_changed = on_selected_instance_changed
        self._on_annotation_changed = on_annotation_changed
        self._labels_layer: Any | None = None
        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_coordinate_system: str | None = None
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
        state = self._get_selection_table_state()
        matching_rows = state.matching_rows
        if state.table is None or matching_rows is None or not state.has_table_row:
            return None

        if USER_CLASS_COLUMN not in state.table.obs:
            return UNLABELED_CLASS

        values = _to_user_class_values(state.table.obs.loc[matching_rows, USER_CLASS_COLUMN])
        return int(values.iloc[0])

    @property
    def selected_instance_has_table_row(self) -> bool:
        """Return whether the current selection is represented in the bound annotation table."""
        return self._get_selection_table_state().has_table_row

    @property
    def missing_table_row_message(self) -> str | None:
        """Return a user-facing warning when the selected object is missing from the table."""
        return self._get_selection_table_state().missing_table_row_message

    @property
    def selected_instance_key_name(self) -> str | None:
        """Return the active table's instance key name for the current selection, if any."""
        return self._get_selection_table_state().instance_key_name

    @property
    def can_annotate(self) -> bool:
        """Return whether the current selection can be written back to the annotation table."""
        state = self._get_selection_table_state()
        return self._labels_layer is not None and state.has_annotation_binding and state.has_table_row

    def bind(
        self,
        sdata: SpatialData | None,
        label_name: str | None,
        table_name: str | None = None,
        coordinate_system: str | None = None,
    ) -> None:
        """Bind the controller to the selected labels layer and annotation table."""
        next_layer = None
        if sdata is not None and label_name is not None:
            next_layer = self._viewer_adapter.get_loaded_primary_labels_layer(sdata, label_name, coordinate_system)

        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        layer_changed = next_layer is not self._labels_layer
        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_coordinate_system = coordinate_system
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
        if not self._viewer_adapter.activate_layer(self._labels_layer):
            return False

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

    def apply_class(self, class_id: int) -> str | None:
        """Assign the given user class to the currently picked instance."""
        return self._set_current_class(class_id)

    def clear_current_class(self) -> str | None:
        """Reset the current object's user class back to the unlabeled state."""
        return self._set_current_class(UNLABELED_CLASS)

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

    def _set_current_class(self, class_id: int) -> str | None:
        """Write the selected user class for the current pick.

        The current selection must be fully bound to a segmentation, annotation
        table, and picked instance id. If the selected label has a matching row
        in the table, this normalizes the `user_class` column, updates the
        matching observation, and triggers the annotation-changed callback.

        If the picked label is present in the segmentation mask but absent from
        the annotation table, no table state is changed. Instead, a warning is
        logged and the user-facing warning message is returned so the widget can
        display it in the UI.
        """
        if class_id < UNLABELED_CLASS:
            raise ValueError("Class ids must be zero or positive integers.")

        state = self._get_selection_table_state()
        if state.table is None:
            raise ValueError("Choose an annotation table before annotating.")
        if state.metadata is None:
            raise ValueError("Choose an annotation table before annotating.")
        if state.label_name is None:
            raise ValueError("Choose a segmentation mask before annotating.")
        if state.instance_id is None:
            raise ValueError("Pick an object in the viewer before annotating.")

        matching_rows = state.matching_rows
        if matching_rows is None or not state.has_table_row:
            message = state.missing_table_row_message
            if message is None:
                raise RuntimeError(
                    "Missing-table-row warning was requested for a selection that still has a table row."
                )
            logger.warning(message)
            return message

        self.ensure_annotation_column(USER_CLASS_COLUMN)

        user_class_values = _to_user_class_values(state.table.obs[USER_CLASS_COLUMN])
        user_class_values.loc[matching_rows] = int(class_id)
        _set_user_class_annotation_state(state.table, user_class_values)
        if self._on_annotation_changed is not None:
            self._on_annotation_changed()
        return None

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _require_bound_table(self) -> AnnData:
        table = self._get_bound_table()
        if table is None:
            raise ValueError("Choose an annotation table before annotating.")

        return table

    def _matching_rows_mask(self, obs: pd.DataFrame, metadata: SpatialDataTableMetadata) -> pd.Series:
        if self._selected_label_name is None or self._selected_instance_id is None:
            return pd.Series(False, index=obs.index)

        return (obs[metadata.region_key] == self._selected_label_name) & (
            obs[metadata.instance_key] == self._selected_instance_id
        )

    def _get_selection_table_state(self) -> _SelectionTableState:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        matching_rows = None
        if (
            table is not None
            and metadata is not None
            and self._selected_label_name is not None
            and self._selected_instance_id is not None
        ):
            matching_rows = self._matching_rows_mask(table.obs, metadata)

        return _SelectionTableState(
            table=table,
            metadata=metadata,
            label_name=self._selected_label_name,
            table_name=self._selected_table_name,
            instance_id=self._selected_instance_id,
            matching_rows=matching_rows,
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


def _to_user_class_values(values: pd.Series) -> pd.Series:
    return normalize_class_values(values, column_name=USER_CLASS_COLUMN, unlabeled_class=UNLABELED_CLASS)


def _set_user_class_annotation_state(table: AnnData, values: pd.Series) -> None:
    set_class_annotation_state(
        table,
        values,
        column_name=USER_CLASS_COLUMN,
        colors_key=USER_CLASS_COLORS_KEY,
        warn_on_palette_overwrite=False,
        unlabeled_class=UNLABELED_CLASS,
        unlabeled_color=UNLABELED_COLOR,
    )


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
        # Query napari for the picked label value at the cursor instead of
        # scanning `layer.data`, so large lazy labels layers stay out-of-core.
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
