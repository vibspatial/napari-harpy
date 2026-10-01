from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import napari

from spatiato._app_state import SpatiatoAppState, get_or_create_app_state
from spatiato._shapes_triangulation import (
    ShapesTriangulationBackend,
    configure_shapes_triangulation_backend,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData

type SpatiatoWidgetId = Literal[
    "viewer", "feature_extraction", "histogram", "object_classification", "shapes_annotation"
]
type SpatiatoWidgetSelection = Literal["all"] | SpatiatoWidgetId | Sequence[SpatiatoWidgetId]


class Interactive:
    """
    Thin programmatic launcher for spatiato.

    Parameters
    ----------
    sdata
        SpatialData object that Spatiato widgets should use as their shared active
        data source.
    viewer
        Existing napari viewer to reuse. If omitted, the current viewer is used
        or a new viewer is created.
    headless
        If True, initialize state and dock widgets without starting the napari
        event loop.
    widgets
        Which Spatiato dock widgets to open. Defaults to ``"all"``. Possible
        values are ``"all"``, ``"viewer"``, ``"feature_extraction"``,
        ``"histogram"``, ``"object_classification"``, and
        ``"shapes_annotation"``. Pass a tuple of widget ids to open a subset.
        ``"all"`` opens every Spatiato widget.
    async_slicing
        If ``True`` or ``False``, explicitly enable or disable napari's
        experimental async slicing for this session. If ``None``, leave napari's
        current setting unchanged.
    triangulation_backend
        Process-wide backend used to triangulate Shapes layers. Supported
        values are ``"bermuda"`` and ``"numba"``. Defaults to ``"bermuda"``.
    """

    _PLUGIN_NAME = "spatiato"
    _WIDGET_NAMES: dict[str, str] = {
        "viewer": "Viewer",
        "feature_extraction": "Feature Extraction",
        "histogram": "Image Histogram",
        "object_classification": "Object Classification",
        "shapes_annotation": "Annotation",
    }
    _ALL_WIDGET_IDS: tuple[SpatiatoWidgetId, ...] = (
        "viewer",
        "feature_extraction",
        "histogram",
        "object_classification",
        "shapes_annotation",
    )

    def __init__(
        self,
        sdata: SpatialData,
        viewer: napari.Viewer | None = None,
        headless: bool = False,
        widgets: SpatiatoWidgetSelection = "all",
        async_slicing: bool | None = False,
        triangulation_backend: ShapesTriangulationBackend = "bermuda",
    ) -> None:
        widget_ids = self._normalize_widget_selection(widgets)
        # Napari chooses the Shapes mesh implementation while constructing
        # shapes, so configure it before creating the viewer or dock widgets.
        configure_shapes_triangulation_backend(triangulation_backend)
        if async_slicing is not None:
            _set_napari_async_slicing(async_slicing)
        self._viewer = viewer or napari.current_viewer() or napari.Viewer()
        self._app_state = get_or_create_app_state(self._viewer)
        self._dock_widgets: dict[str, tuple[Any, Any]] = {}

        # Constructing Interactive with an existing viewer is itself explicit
        # programmatic authorization to replace that viewer's current session.
        # Never open a confirmation dialog here, including in headless mode.
        self._app_state.set_sdata(sdata, discard_current=True)
        self._ensure_spatiato_widgets(widget_ids)

        if not headless:
            self.run()

    @property
    def viewer(self) -> napari.Viewer:
        """Return the napari viewer managed by the launcher."""
        return self._viewer

    @property
    def app_state(self) -> SpatiatoAppState:
        """Return the shared Spatiato app state for the active viewer."""
        return self._app_state

    def run(self) -> None:
        """Run the napari application."""
        napari.run()

    def _ensure_spatiato_widgets(self, widget_ids: Sequence[SpatiatoWidgetId]) -> None:
        for widget_id in widget_ids:
            widget_name = self._WIDGET_NAMES[widget_id]
            self._dock_widgets[widget_name] = self._viewer.window.add_plugin_dock_widget(
                self._PLUGIN_NAME,
                widget_name,
                tabify=True,
            )

    @classmethod
    def _normalize_widget_selection(cls, widgets: SpatiatoWidgetSelection) -> tuple[SpatiatoWidgetId, ...]:
        if widgets == "all":
            return cls._ALL_WIDGET_IDS

        if isinstance(widgets, str):
            cls._validate_widget_id(widgets)
            return (widgets,)

        if not isinstance(widgets, Sequence):
            raise ValueError("`widgets` must be 'all', one Spatiato widget id, or a sequence of Spatiato widget ids.")

        widget_ids: list[SpatiatoWidgetId] = []
        seen_widget_ids: set[SpatiatoWidgetId] = set()
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
            raise ValueError(f"Unknown Spatiato widget selection {widget_id!r}. Valid options are: {valid_widget_ids}.")


def _set_napari_async_slicing(enabled: bool) -> None:
    from napari.settings import get_settings

    get_settings().experimental.async_ = enabled
