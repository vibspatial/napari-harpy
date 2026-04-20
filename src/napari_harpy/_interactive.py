from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state

if TYPE_CHECKING:
    from spatialdata import SpatialData


class Interactive:
    """Thin programmatic launcher for napari-harpy."""

    _PLUGIN_NAME = "napari-harpy"
    _DEFAULT_WIDGETS = ("Viewer", "Feature Extraction", "Object Classification")

    def __init__(self, sdata: SpatialData, viewer: napari.Viewer | None = None, headless: bool = False) -> None:
        self._viewer = viewer or napari.current_viewer() or napari.Viewer()
        self._app_state = get_or_create_app_state(self._viewer)
        self._dock_widgets: dict[str, tuple[Any, Any]] = {}

        self._app_state.set_sdata(sdata)
        self._ensure_harpy_widgets()

        if not headless:
            self.run()

    @property
    def viewer(self) -> napari.Viewer:
        """Return the napari viewer managed by the launcher."""
        return self._viewer

    @property
    def app_state(self) -> HarpyAppState:
        """Return the shared Harpy app state for the active viewer."""
        return self._app_state

    def run(self) -> None:
        """Run the napari application."""
        napari.run()

    def _ensure_harpy_widgets(self) -> None:
        for widget_name in self._DEFAULT_WIDGETS:
            self._dock_widgets[widget_name] = self._viewer.window.add_plugin_dock_widget(
                self._PLUGIN_NAME,
                widget_name,
                tabify=True,
            )
