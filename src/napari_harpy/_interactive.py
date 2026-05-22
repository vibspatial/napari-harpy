from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import napari

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state

if TYPE_CHECKING:
    from spatialdata import SpatialData

HarpyWidgetId: TypeAlias = Literal["viewer", "feature_extraction", "object_classification"]
HarpyWidgetSelection: TypeAlias = Literal["all"] | HarpyWidgetId | Sequence[HarpyWidgetId]


class Interactive:
    """
    Thin programmatic launcher for napari-harpy.

    Parameters
    ----------
    sdata
        SpatialData object that Harpy widgets should use as their shared active
        data source.
    viewer
        Existing napari viewer to reuse. If omitted, the current viewer is used
        or a new viewer is created.
    headless
        If True, initialize state and dock widgets without starting the napari
        event loop.
    widgets
        Which Harpy dock widgets to open. Possible values are ``"all"``,
        ``"viewer"``, ``"feature_extraction"``, and
        ``"object_classification"``. Pass a tuple of widget ids to open a
        subset.
    """

    _PLUGIN_NAME = "napari-harpy"
    _WIDGET_NAMES: dict[str, str] = {
        "viewer": "Viewer",
        "feature_extraction": "Feature Extraction",
        "object_classification": "Object Classification",
    }
    _DEFAULT_WIDGET_IDS: tuple[HarpyWidgetId, ...] = (
        "viewer",
        "feature_extraction",
        "object_classification",
    )

    def __init__(
        self,
        sdata: SpatialData,
        viewer: napari.Viewer | None = None,
        headless: bool = False,
        widgets: HarpyWidgetSelection = "all",
    ) -> None:
        widget_ids = self._normalize_widget_selection(widgets)
        self._viewer = viewer or napari.current_viewer() or napari.Viewer()
        self._app_state = get_or_create_app_state(self._viewer)
        self._dock_widgets: dict[str, tuple[Any, Any]] = {}

        self._app_state.set_sdata(sdata)
        self._ensure_harpy_widgets(widget_ids)

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

    def _ensure_harpy_widgets(self, widget_ids: Sequence[HarpyWidgetId]) -> None:
        for widget_id in widget_ids:
            widget_name = self._WIDGET_NAMES[widget_id]
            self._dock_widgets[widget_name] = self._viewer.window.add_plugin_dock_widget(
                self._PLUGIN_NAME,
                widget_name,
                tabify=True,
            )

    @classmethod
    def _normalize_widget_selection(cls, widgets: HarpyWidgetSelection) -> tuple[HarpyWidgetId, ...]:
        if widgets == "all":
            return cls._DEFAULT_WIDGET_IDS

        if isinstance(widgets, str):
            cls._validate_widget_id(widgets)
            return (widgets,)

        if not isinstance(widgets, Sequence):
            raise ValueError(
                "`widgets` must be 'all', one Harpy widget id, or a sequence of Harpy widget ids."
            )

        widget_ids: list[HarpyWidgetId] = []
        seen_widget_ids: set[HarpyWidgetId] = set()
        for widget_id in widgets:
            cls._validate_widget_id(widget_id)
            if widget_id not in seen_widget_ids:
                widget_ids.append(widget_id)
                seen_widget_ids.add(widget_id)

        return tuple(widget_ids)

    @classmethod
    def _validate_widget_id(cls, widget_id: object) -> None:
        if widget_id not in cls._WIDGET_NAMES:
            valid_widget_ids = ", ".join(("all", *cls._WIDGET_NAMES))
            raise ValueError(f"Unknown Harpy widget selection {widget_id!r}. Valid options are: {valid_widget_ids}.")
