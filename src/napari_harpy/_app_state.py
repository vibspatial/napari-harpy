from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from qtpy.QtCore import QObject, Signal

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
