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

    def __init__(self, viewer: object | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata: SpatialData | None = None
        self.layer_bindings = LayerBindingRegistry()
        self.viewer_adapter = ViewerAdapter(viewer=viewer, layer_bindings=self.layer_bindings)
        self._dirty_table_keys: set[tuple[int, str]] = set()

    def set_sdata(self, sdata: SpatialData | None) -> None:
        """Set the loaded SpatialData object and notify listeners."""
        self.sdata = sdata
        # Notify connected widgets/controllers that the loaded SpatialData changed
        # (controllers/widgets that listen via e.g. self._app_state.sdata_changed.connect(self._on_sdata_changed))
        self.sdata_changed.emit(sdata)

    def clear_sdata(self) -> None:
        """Clear the loaded SpatialData object and notify listeners."""
        self.set_sdata(None)

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
