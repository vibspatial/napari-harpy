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

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from qtpy.QtCore import QObject, Signal

from napari_harpy._viewer_adapter import LayerBindingRegistry, ViewerAdapter

if TYPE_CHECKING:
    from spatialdata import SpatialData


_VIEWER_APP_STATES: WeakKeyDictionary[object, HarpyAppState] = WeakKeyDictionary()


class HarpyAppState(QObject):
    """Shared Harpy state bound to a napari viewer."""

    sdata_changed = Signal(object)

    def __init__(self, viewer: object | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata: SpatialData | None = None
        self.layer_bindings = LayerBindingRegistry()
        self.viewer_adapter = ViewerAdapter(viewer=viewer, layer_bindings=self.layer_bindings)

    def set_sdata(self, sdata: SpatialData | None) -> None:
        """Set the loaded SpatialData object and notify listeners."""
        self.sdata = sdata
        self.sdata_changed.emit(sdata)

    def clear_sdata(self) -> None:
        """Clear the loaded SpatialData object and notify listeners."""
        self.set_sdata(None)


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
