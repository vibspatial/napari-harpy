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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary

from qtpy.QtCore import QObject, Signal

from napari_harpy.core.persistence import TableComponentPath
from napari_harpy.core.spatialdata import get_coordinate_system_names_from_sdata
from napari_harpy.viewer.adapter import LayerBindingRegistry, ViewerAdapter

if TYPE_CHECKING:
    from spatialdata import SpatialData


_VIEWER_APP_STATES: WeakKeyDictionary[object, HarpyAppState] = WeakKeyDictionary()
TableChangeKind = Literal["created", "updated", "removed", "rebuilt", "reloaded"]


@dataclass(frozen=True)
class TableStateChangedEvent:
    """Describe one accepted change to explicit components of an AnnData table.

    ``regions`` is the producer-declared semantic scope of any row-scoped
    change. An empty tuple means the event is genuinely metadata-only and never
    means that the affected regions are unknown. A producer that cannot prove a
    narrower row scope must report every region declared by the table.
    """

    sdata: SpatialData
    table_name: str
    paths: frozenset[TableComponentPath]
    regions: tuple[str, ...]
    change_kind: TableChangeKind
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Table-state events require a non-empty table name.")
        paths = frozenset(self.paths)
        if not paths:
            raise ValueError("Table-state events require at least one component path.")
        object.__setattr__(self, "paths", paths)
        if self.change_kind not in ("created", "updated", "removed", "rebuilt", "reloaded"):
            raise ValueError(f"Unsupported table change kind: {self.change_kind!r}.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Table-state events require a non-empty source.")
        regions = tuple(self.regions)
        if any(not isinstance(region, str) or not region for region in regions):
            raise ValueError("Table-state event regions must be non-empty strings.")
        if len(set(regions)) != len(regions):
            raise ValueError("Table-state event regions must be unique.")
        object.__setattr__(self, "regions", regions)


class _TableMutationToken:
    """Opaque identity token for one accepted in-memory table mutation."""

    __slots__ = ()


@dataclass(frozen=True)
class TableDirtySnapshot:
    """Capture one table's dirty component state before an operation starts.

    The snapshot preserves the mutation token associated with every component
    path that was dirty at capture time. Persistence acknowledgement later
    compares these captured tokens with the current dirty manifest. A path is
    eligible to become clean only when it was successfully persisted and its
    current token is still the captured token.

    The snapshot is session-only state. It does not copy the table and is never
    stored in AnnData or zarr.

    Parameters
    ----------
    sdata
        SpatialData object containing the affected table. Object identity is
        used to ensure that acknowledgement targets the same dataset.
    table_name
        Name of the affected AnnData table.
    captured_path_tokens
        Immutable pairs of dirty component paths and the mutation tokens
        captured for them. The token objects are retained by identity rather
        than copied.
    """

    sdata: SpatialData
    table_name: str
    captured_path_tokens: tuple[tuple[TableComponentPath, _TableMutationToken], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.table_name, str) or not self.table_name:
            raise ValueError("Dirty snapshots require a non-empty table name.")
        normalized = tuple(sorted(self.captured_path_tokens, key=lambda item: item[0]))
        paths = tuple(path for path, _token in normalized)
        if len(set(paths)) != len(paths):
            raise ValueError("Dirty snapshots cannot contain duplicate component paths.")
        if any(not isinstance(token, _TableMutationToken) for _, token in normalized):
            raise ValueError("Dirty snapshots require a mutation token for every component path.")
        object.__setattr__(self, "captured_path_tokens", normalized)

    @property
    def paths(self) -> frozenset[TableComponentPath]:
        """Return the component paths captured by this snapshot."""
        return frozenset(path for path, _token in self.captured_path_tokens)


@dataclass(frozen=True)
class ShapesElementWrittenEvent:
    """Describe an in-place write to a shared `SpatialData.shapes` element."""

    sdata: SpatialData
    shapes_name: str
    coordinate_system: str
    source: str = "shapes_annotation_widget"


@dataclass(frozen=True)
class CoordinateSystemChangedEvent:
    """Describe a shared active coordinate-system change for one viewer session."""

    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None


@dataclass(frozen=True)
class CoordinateSystemChangeRequest:
    """Describe a requested coordinate-system change before it is committed.

    A coordinate-system guard receives this immutable request while the old
    app-state value and viewer layers are still intact. Returning ``False``
    rejects the request without emitting ``coordinate_system_changed`` or
    removing layers.
    """

    sdata: SpatialData | None
    previous_coordinate_system: str | None
    coordinate_system: str | None
    source: str | None = None


type CoordinateSystemChangeGuard = Callable[[CoordinateSystemChangeRequest], bool]


def _path_covers(parent: TableComponentPath, child: TableComponentPath) -> bool:
    """Return whether one synchronized path also restores a dirty path.

    ``parent`` is a path accepted as persisted or reloaded; ``child`` is a
    dirty path being considered for removal from the dirty manifest. Coverage
    follows the persistence granularity of each AnnData component: any
    ``obs`` path covers the complete dataframe, an ``obsm`` path covers only
    the same named entry, and an ``uns`` path covers itself and its nested
    descendants. Paths belonging to different components never overlap.
    """
    if parent.component != child.component:
        return False
    if parent.component == "obs":
        return True
    if parent.component == "obsm":
        return parent == child
    if parent.component == "uns":
        return child.keys[: len(parent.keys)] == parent.keys
    raise AssertionError(f"Unsupported table component: {parent.component!r}.")


class HarpyAppState(QObject):
    """Shared Harpy state bound to a napari viewer.

    This object is the per-viewer event and state hub that Harpy widgets use
    to stay synchronized without depending on each other directly.

    Cross-widget table updates are published through ``table_state_changed``
    with explicit AnnData component paths. Shapes writes retain their separate
    event because they do not mutate an AnnData table. Producing widgets do not
    need to know which other widgets consume either update.

    ``HarpyAppState`` also owns shared session-level dirty-table tracking, so
    in-memory table divergence from disk is modeled as shared viewer state
    rather than as widget-local state.

    A dirty table component can become clean only through one of three
    accepted transitions:

    1. ``acknowledge_table_write()`` clears a component after the generic
       persistence service successfully writes it and its captured mutation
       token is still current.
    2. ``record_persisted_table_change()`` clears a component that an operation
       already persisted itself, again only when its captured token is still
       current.
    3. ``record_table_reload()`` clears a component whose in-memory state was
       deliberately replaced with the persisted state.

    Publishing a table-state event alone never clears dirty state, and an
    unrelated or outdated operation cannot clear a component.
    """

    sdata_changed = Signal(object)
    table_state_changed = Signal(object)
    shapes_element_written = Signal(object)
    coordinate_system_changed = Signal(object)

    def __init__(self, viewer: object | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata: SpatialData | None = None
        self.coordinate_system: str | None = None
        self.layer_bindings = LayerBindingRegistry()
        self.viewer_adapter = ViewerAdapter(viewer=viewer, layer_bindings=self.layer_bindings)
        self._dirty_table_tokens: dict[tuple[int, str], dict[TableComponentPath, _TableMutationToken]] = {}
        self._coordinate_system_change_guard: CoordinateSystemChangeGuard | None = None

    def set_sdata(self, sdata: SpatialData | None) -> None:
        """Set the loaded SpatialData object and notify listeners."""
        old_sdata = self.sdata
        old_coordinate_system = self.coordinate_system

        if old_sdata is not None and old_sdata is not sdata:
            self.viewer_adapter.remove_layers_for_sdata(old_sdata)

        self.sdata = sdata
        next_coordinate_system = self._resolve_coordinate_system_for_sdata(sdata, previous=old_coordinate_system)
        self._update_coordinate_system_state(next_coordinate_system, source="set_sdata")
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
        """Request a change to the shared coordinate system.

        When the requested coordinate system is valid and differs from the current
        one, the registered guard is called immediately before the change is applied.
        If the guard rejects the request, the coordinate system remains unchanged,
        no ``coordinate_system_changed`` event is emitted, and no viewer layers are
        removed.

        Example flow
        ------------
        set_coordinate_system("local")
            ↓
        normalize and validate "local"
            ↓
        check that "local" differs from the current "global"
            ↓
        call the guard immediately
            ├── False
            │      → stop
            │      → coordinate system remains "global"
            │      → emit no coordinate_system_changed event
            │      → remove no viewer layers
            │
            └── True
                   → change app state to "local"
                   → emit coordinate_system_changed
                   → remove layers outside "local"
        """
        normalized_coordinate_system = self._normalize_coordinate_system(coordinate_system)
        self._validate_coordinate_system(normalized_coordinate_system)
        if normalized_coordinate_system == self.coordinate_system:
            return False

        guard = self._coordinate_system_change_guard
        if guard is not None:
            request = CoordinateSystemChangeRequest(
                sdata=self.sdata,
                previous_coordinate_system=self.coordinate_system,
                coordinate_system=normalized_coordinate_system,
                source=source,
            )
            # This must run before state mutation or layer removal: otherwise
            # an external coordinate switch could silently delete the only
            # copy of unsaved Shapes Annotation edits.
            if not guard(request):
                return False

        self._update_coordinate_system_state(normalized_coordinate_system, source=source)

        self.viewer_adapter.remove_layers_outside_coordinate_system(
            sdata=self.sdata,
            coordinate_system=normalized_coordinate_system,
        )
        return True

    def set_coordinate_system_change_guard(self, guard: CoordinateSystemChangeGuard | None) -> None:
        """Install the single owner of coordinate-system change preflight.

        The Annotation widget owns this optional per-viewer guard and must
        remove it during teardown. Replacing another active guard is rejected
        so one widget cannot silently disable another widget's data-loss
        protection.
        """
        if guard is not None and not callable(guard):
            raise TypeError("Coordinate-system change guard must be callable or None.")
        if (
            guard is not None
            and self._coordinate_system_change_guard is not None
            and guard is not self._coordinate_system_change_guard
        ):
            raise RuntimeError("A coordinate-system change guard is already installed for this viewer session.")
        self._coordinate_system_change_guard = guard

    def clear_coordinate_system_change_guard(self, guard: CoordinateSystemChangeGuard) -> bool:
        """Remove ``guard`` only when it still owns coordinate preflight.

        Identity-safe removal prevents teardown of an older widget from
        clearing a guard installed by a newer owner.
        """
        if self._coordinate_system_change_guard is not guard:
            return False
        self._coordinate_system_change_guard = None
        return True

    def clear_coordinate_system(self, *, source: str | None = None) -> bool:
        """Clear the shared active coordinate system."""
        return self.set_coordinate_system(None, source=source)

    def emit_shapes_element_written(self, event: ShapesElementWrittenEvent) -> None:
        """Broadcast that a shapes element was written into the shared SpatialData."""
        self.shapes_element_written.emit(event)

    def record_table_mutation(self, event: TableStateChangedEvent) -> None:
        """Record an accepted in-memory mutation and publish its table event."""
        selection_key = self._selection_key(event.sdata, event.table_name)
        if selection_key is None:  # pragma: no cover - event validation makes this unreachable.
            return

        mutation_token = _TableMutationToken()
        manifest = self._dirty_table_tokens.setdefault(selection_key, {})
        for path in event.paths:
            manifest[path] = mutation_token
        self.table_state_changed.emit(event)

    def record_persisted_table_change(
        self,
        event: TableStateChangedEvent,
        snapshot: TableDirtySnapshot,
    ) -> None:
        """Publish an already-persisted change without clearing newer mutations."""
        self._validate_snapshot_identity(snapshot, event.sdata, event.table_name)
        self.acknowledge_table_write(snapshot, persisted_paths=event.paths)
        self.table_state_changed.emit(event)

    def record_table_reload(self, event: TableStateChangedEvent) -> None:
        """Publish a component reload and clear only dirty paths it restored."""
        if event.change_kind != "reloaded":
            raise ValueError("Table reload events must use change_kind='reloaded'.")
        selection_key = self._selection_key(event.sdata, event.table_name)
        if selection_key is None:  # pragma: no cover - event validation makes this unreachable.
            return
        manifest = self._dirty_table_tokens.get(selection_key)
        if manifest is not None:
            for dirty_path in tuple(manifest):
                if any(_path_covers(reloaded_path, dirty_path) for reloaded_path in event.paths):
                    del manifest[dirty_path]
            self._drop_empty_manifest(selection_key)
        self.table_state_changed.emit(event)

    def is_table_dirty(self, sdata: SpatialData | None, table_name: str | None) -> bool:
        """Return whether a selected in-memory table has unsynced local changes."""
        selection_key = self._selection_key(sdata, table_name)
        return selection_key is not None and bool(self._dirty_table_tokens.get(selection_key))

    def snapshot_table_dirty_state(
        self,
        sdata: SpatialData,
        table_name: str,
    ) -> TableDirtySnapshot:
        """Capture the current component mutation tokens for a table."""
        selection_key = self._selection_key(sdata, table_name)
        if selection_key is None:  # pragma: no cover - concrete arguments make this unreachable.
            raise ValueError("Dirty snapshots require a SpatialData object and table name.")
        manifest = self._dirty_table_tokens.get(selection_key, {})
        return TableDirtySnapshot(
            sdata=sdata,
            table_name=table_name,
            captured_path_tokens=tuple(manifest.items()),
        )

    def acknowledge_table_write(
        self,
        snapshot: TableDirtySnapshot,
        *,
        persisted_paths: frozenset[TableComponentPath],
    ) -> None:
        """Mark successfully persisted component paths clean when still current.

        Each dirty component path carries the identity token of its latest
        accepted mutation. The snapshot captures those tokens before
        persistence starts.

        A persisted path is cleared only when its current token is the same
        object as the captured token. If the path changed while persistence was
        in progress, it has received a fresh token and remains dirty. This
        prevents completion of an older write from marking a newer in-memory
        mutation as persisted.
        """
        selection_key = self._selection_key(snapshot.sdata, snapshot.table_name)
        if selection_key is None:  # pragma: no cover - snapshot validation makes this unreachable.
            return
        manifest = self._dirty_table_tokens.get(selection_key)
        if manifest is None:
            return

        for path, captured_token in snapshot.captured_path_tokens:
            if not any(_path_covers(persisted_path, path) for persisted_path in persisted_paths):
                continue
            if manifest.get(path) is captured_token:
                del manifest[path]
        self._drop_empty_manifest(selection_key)

    def _drop_empty_manifest(self, selection_key: tuple[int, str]) -> None:
        if not self._dirty_table_tokens.get(selection_key):
            self._dirty_table_tokens.pop(selection_key, None)

    @staticmethod
    def _validate_snapshot_identity(
        snapshot: TableDirtySnapshot,
        sdata: SpatialData,
        table_name: str,
    ) -> None:
        if snapshot.sdata is not sdata or snapshot.table_name != table_name:
            raise ValueError("Dirty snapshot does not belong to the changed table.")

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

    def _update_coordinate_system_state(
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
