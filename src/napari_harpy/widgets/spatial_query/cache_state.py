"""Shared-state integration for accepted canonical-cache mutations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from napari_harpy._app_state import HarpyAppState, TableChangeKind, TableStateChangedEvent
from napari_harpy.core.spatial_query import (
    CANONICAL_CACHE_PATHS,
    CanonicalCacheUpdateAction,
    CanonicalCentersResult,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


CANONICAL_CACHE_UPDATE_SOURCE = "spatial_query_canonical_centers"

_CACHE_UPDATE_CHANGE_KINDS: dict[CanonicalCacheUpdateAction, TableChangeKind] = {
    CanonicalCacheUpdateAction.CREATE: "created",
    CanonicalCacheUpdateAction.EXTEND: "updated",
    CanonicalCacheUpdateAction.REFRESH: "updated",
    CanonicalCacheUpdateAction.REBUILD: "rebuilt",
}


def record_canonical_cache_update(
    app_state: HarpyAppState,
    sdata: SpatialData,
    result: CanonicalCentersResult,
) -> TableStateChangedEvent | None:
    """Record one accepted canonical-cache update in shared table state.

    This function belongs at the consumer side of the controller's
    ``on_centers_ready`` callback. Reused centers have no cache update and
    therefore produce no table event or dirty-state transition.
    """
    cache_update = result.cache_update
    if cache_update is None:
        return None
    event = TableStateChangedEvent(
        sdata=sdata,
        table_name=result.table_name,
        paths=CANONICAL_CACHE_PATHS,
        regions=(result.labels_name,),
        change_kind=_CACHE_UPDATE_CHANGE_KINDS[cache_update.action],
        source=CANONICAL_CACHE_UPDATE_SOURCE,
    )
    app_state.record_table_mutation(event)
    return event
