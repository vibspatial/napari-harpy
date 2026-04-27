"""Shared Harpy app state bound to one napari viewer.

This module defines the per-viewer state object used across Harpy widgets.

At the top level, ``_VIEWER_APP_STATES`` is a registry that maps one napari
viewer to one ``HarpyAppState`` instance. That is how multiple widgets opened
on the same viewer end up sharing the same Harpy state.

Each ``HarpyAppState`` then holds the shared state and services for that
viewer:

- ``viewer``: the napari viewer this state belongs to
- ``sdata``: the currently loaded ``SpatialData`` object
- ``layer_bindings``: the in-memory registry that records which napari layers
  correspond to which ``SpatialData`` elements
- ``viewer_adapter``: the viewer-facing service that uses the shared registry
  to look up, activate, and later load Harpy-managed layers

So the relationship is:

- global registry: viewer -> ``HarpyAppState``
- per-viewer state: ``viewer``, ``sdata``, ``layer_bindings``,
  ``viewer_adapter``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary

from qtpy.QtCore import QObject, Signal

from napari_harpy._spatialdata import get_coordinate_system_names_from_sdata
from napari_harpy._viewer_adapter import LayerBindingRegistry, ViewerAdapter

if TYPE_CHECKING:
    from spatialdata import SpatialData


_VIEWER_APP_STATES: WeakKeyDictionary[object, HarpyAppState] = WeakKeyDictionary()
FeatureMatrixWriteChangeKind = Literal["created", "updated"]


@dataclass(frozen=True)
class FeatureMatrixWrittenEvent:
    """Describe a feature-matrix write into an in-memory table `.obsm` mapping."""

    sdata: SpatialData
    table_name: str
    feature_key: str
    change_kind: FeatureMatrixWriteChangeKind
    source: str = "feature_extraction"


@dataclass(frozen=True)
class CoordinateSystemChangedEvent:
    """Describe a shared active coordinate-system change for one viewer session."""

    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None


class HarpyAppState(QObject):
    """Shared Harpy state bound to a napari viewer.

    This object is the per-viewer event and state hub that Harpy widgets use
    to stay synchronized without depending on each other directly.

    In particular, cross-widget updates that do not replace the loaded
    ``SpatialData`` object are published here as semantic app-state events. For
    example, ``FeatureExtractionWidget`` forwards successful feature-matrix
    writes into ``feature_matrix_written``, and
    ``ObjectClassificationWidget`` listens to that signal to refresh its
    feature-matrix selector and dirty-state. The producing widget therefore
    does not need to know which other widgets are consuming the update.

    ``HarpyAppState`` also owns shared session-level dirty-table tracking, so
    in-memory table divergence from disk is modeled as shared viewer state
    rather than as widget-local state.
    """

    sdata_changed = Signal(object)
    feature_matrix_written = Signal(object)
    coordinate_system_changed = Signal(object)

    def __init__(self, viewer: object | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata: SpatialData | None = None
        self.coordinate_system: str | None = None
        self.layer_bindings = LayerBindingRegistry()
        self.viewer_adapter = ViewerAdapter(viewer=viewer, layer_bindings=self.layer_bindings)
        self._dirty_table_keys: set[tuple[int, str]] = set()

    def set_sdata(self, sdata: SpatialData | None) -> None:
        """Set the loaded SpatialData object and notify listeners."""
        old_sdata = self.sdata
        old_coordinate_system = self.coordinate_system

        if old_sdata is not None:
            self.viewer_adapter.remove_layers_for_sdata(old_sdata)

        self.sdata = sdata
        next_coordinate_system = self._resolve_coordinate_system_for_sdata(sdata, previous=old_coordinate_system)
        self._set_coordinate_system_without_pruning(next_coordinate_system, source="set_sdata")
        # Notify connected widgets/controllers that the loaded SpatialData changed
        # (controllers/widgets that listen via e.g. self._app_state.sdata_changed.connect(self._on_sdata_changed))
        self.sdata_changed.emit(sdata)

    def clear_sdata(self) -> None:
        """Clear the loaded SpatialData object and notify listeners."""
        self.set_sdata(None)

    def set_coordinate_system(
        self,
        coordinate_system: str | None,
        *,
        source: str | None = None,
    ) -> bool:
        """Set the shared active coordinate system for the current loaded SpatialData."""
        normalized_coordinate_system = self._normalize_coordinate_system(coordinate_system)
        self._validate_coordinate_system(normalized_coordinate_system)
        changed = self._set_coordinate_system_without_pruning(normalized_coordinate_system, source=source)
        if not changed:
            return False

        self.viewer_adapter.remove_layers_outside_coordinate_system(
            sdata=self.sdata,
            coordinate_system=normalized_coordinate_system,
        )
        return True

    def clear_coordinate_system(self, *, source: str | None = None) -> bool:
        """Clear the shared active coordinate system."""
        return self.set_coordinate_system(None, source=source)

    def emit_feature_matrix_written(self, event: FeatureMatrixWrittenEvent) -> None:
        """Broadcast that a feature matrix was written into a shared in-memory table."""
        self.mark_table_dirty(event.sdata, event.table_name)
        self.feature_matrix_written.emit(event)

    def is_table_dirty(self, sdata: SpatialData | None, table_name: str | None) -> bool:
        """Return whether a selected in-memory table has unsynced local changes."""
        selection_key = self._selection_key(sdata, table_name)
        return selection_key is not None and selection_key in self._dirty_table_keys

    def mark_table_dirty(self, sdata: SpatialData | None, table_name: str | None) -> None:
        """Mark a selected in-memory table as diverged from its on-disk backed state."""
        selection_key = self._selection_key(sdata, table_name)
        if selection_key is None:
            return

        self._dirty_table_keys.add(selection_key)

    def clear_table_dirty(self, sdata: SpatialData | None, table_name: str | None) -> None:
        """Clear the dirty marker for a selected in-memory table."""
        selection_key = self._selection_key(sdata, table_name)
        if selection_key is None:
            return

        self._dirty_table_keys.discard(selection_key)

    @staticmethod
    def _selection_key(sdata: SpatialData | None, table_name: str | None) -> tuple[int, str] | None:
        if sdata is None or table_name is None:
            return None

        return (id(sdata), table_name)

    @staticmethod
    def _normalize_coordinate_system(coordinate_system: str | None) -> str | None:
        if coordinate_system is None:
            return None

        normalized_coordinate_system = coordinate_system.strip()
        return normalized_coordinate_system or None

    def _validate_coordinate_system(self, coordinate_system: str | None) -> None:
        if coordinate_system is None:
            return

        if self.sdata is None:
            raise ValueError("Cannot set an active coordinate system when no SpatialData is loaded.")

        available_coordinate_systems = get_coordinate_system_names_from_sdata(self.sdata)
        if coordinate_system not in available_coordinate_systems:
            raise ValueError(
                f"Coordinate system `{coordinate_system}` is not available in the selected SpatialData object."
            )

    def _set_coordinate_system_without_pruning(
        self,
        coordinate_system: str | None,
        *,
        source: str | None = None,
    ) -> bool:
        previous_coordinate_system = self.coordinate_system
        if coordinate_system == previous_coordinate_system:
            return False

        self.coordinate_system = coordinate_system
        self.coordinate_system_changed.emit(
            CoordinateSystemChangedEvent(
                sdata=self.sdata,
                previous_coordinate_system=previous_coordinate_system,
                coordinate_system=coordinate_system,
                source=source,
            )
        )
        return True

    @staticmethod
    def _resolve_coordinate_system_for_sdata(
        sdata: SpatialData | None,
        *,
        previous: str | None,
    ) -> str | None:
        if sdata is None:
            return None

        available_coordinate_systems = get_coordinate_system_names_from_sdata(sdata)
        if previous is not None and previous in available_coordinate_systems:
            return previous
        if available_coordinate_systems:
            return available_coordinate_systems[0]
        return None


def get_or_create_app_state(napari_viewer: object | None) -> HarpyAppState:
    """Return the shared Harpy state for a napari viewer.

    When a real viewer is provided, the same ``HarpyAppState`` instance is
    returned for repeated calls with that viewer. When ``napari_viewer`` is
    ``None``, a fresh standalone state object is returned.
    """
    if napari_viewer is None:
        return HarpyAppState()

    try:
        state = _VIEWER_APP_STATES.get(napari_viewer)
    except TypeError:
        state = getattr(napari_viewer, "_harpy_app_state", None)

    if state is not None:
        return state

    state = HarpyAppState(napari_viewer)

    try:
        _VIEWER_APP_STATES[napari_viewer] = state
    except TypeError:
        napari_viewer._harpy_app_state = state

    return state
