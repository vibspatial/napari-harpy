from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from weakref import WeakKeyDictionary, ref

from napari.layers import Labels
from qtpy.QtCore import QTimer


@dataclass
class _PendingLabelsSync:
    layer_ref: ref[Labels]
    pending: bool = False
    queued: bool = False


# Keep this workaround-local instead of storing it in Harpy layer bindings:
# the state only coordinates one deferred napari/Vispy display sync, and the
# weak keys avoid keeping removed napari layers alive.
_PENDING_LABELS_SYNCS: WeakKeyDictionary[Labels, _PendingLabelsSync] = WeakKeyDictionary()


def sync_labels_display_after_colormap_change(layer: Labels) -> None:
    """Synchronize labels display after Harpy assigns a table-driven colormap.

    Napari labels rendering has two pieces of state that must agree:

    - the displayed labels image, whose values are small texture codes produced
      by `layer.colormap._data_to_texture(...)`;
    - the Vispy texture-code-to-RGBA table produced from the current labels
      colormap.

    When Harpy assigns a direct/compact labels colormap while napari async
    slicing is enabled, those two pieces can temporarily get out of sync.
    The colormap assignment can update the Vispy color table before napari has
    recomputed the displayed labels image for that new colormap. The visible
    symptom is stale or swapped labels colors until the slice is recoded and
    uploaded again.

    The repair is to force the current labels slice to be recomputed and then
    notify Vispy in this order:

    1. `layer.set_view_slice()`
    2. `layer.events.colormap()`
    3. `layer.events.set_data()`

    `layer.set_view_slice()` recomputes the layer-side slice and recodes the
    displayed labels image through the current colormap. The explicit colormap
    event is needed after that recomputation because `Labels.colormap = ...`
    may have emitted while async slicing still exposed a placeholder or
    small-dtype slice; Vispy may therefore have chosen a color-table/shader path
    for the wrong slice dtype. Re-emitting `layer.events.colormap()` after the
    real slice is installed makes Vispy rebuild the labels colormap state for
    the current dtype. Finally, `layer.events.set_data()` uploads the recomputed
    texture-code image. A plain `layer.refresh()` is not enough here because it
    can schedule more async work without guaranteeing that Vispy rebuilds the
    colormap state after the recoded slice is available.

    The immediate three-step sync is correct when `layer.loaded` is true. In
    that state napari is not waiting for an async slice response for this layer,
    so Harpy can safely recompute the current slice and rebuild/upload the
    matching Vispy state.

    When `layer.loaded` is false, napari already has an async slice in flight.
    Calling `layer.set_view_slice()` at that moment can re-enter napari's global
    Dask cache while the async worker is also using it. In practice that can
    crash inside `dask.cache.Cache._posttask` with a missing `starttimes` entry.
    The fix is not to disable async slicing or the Dask cache. Instead, Harpy
    records that a sync is pending, returns immediately, and waits for napari to
    mark the layer loaded again.

    The deferred path connects once to napari's private
    `layer._slicing_state.loaded_data` signal. When napari finishes applying the
    async slice, Harpy queues a single `QTimer.singleShot(0, ...)` callback.
    The zero-delay Qt callback lets napari finish its current slice-ready event
    before Harpy runs the forced sync. Repeated requests while the layer is
    unloaded are coalesced into one pending sync, and the queued callback
    rechecks `layer.loaded` before doing any work. If the layer became unloaded
    again, the sync stays pending until the next loaded notification.

    This keeps napari async slicing and the Dask cache enabled, avoids blocking
    the UI thread, and preserves the final display guarantee: after Harpy
    recolors labels, the displayed texture-code image and the Vispy colormap
    table are synchronized for the latest colormap.

    In a large dask-backed Xenium labels benchmark colored by a categorical
    `.obs` column, the deferred final sync was cheaper than the old immediate
    sync: about 3.25 ms versus 14.7 ms under a headless `ViewerModel`
    async-slicing benchmark. The likely reason is that napari had already
    landed the current slice and the relevant Dask/cache state was warm by the
    time Harpy ran the forced sync.

    Workaround for https://github.com/napari/napari/issues/9188.
    """
    if layer.loaded:
        _sync_labels_display_now(layer)
        _clear_pending_sync(layer)
        return

    _request_deferred_sync(layer)


def _sync_labels_display_now(layer: Labels) -> None:
    # Update napari's internal layer slice and recode labels through the current colormap.
    layer.set_view_slice()
    # Update Vispy's color lookup/shader state for the now-current slice dtype.
    layer.events.colormap()
    # Upload the recomputed texture-code image to Vispy.
    layer.events.set_data()


def _clear_pending_sync(layer: Labels) -> None:
    state = _PENDING_LABELS_SYNCS.get(layer)
    if state is not None:
        state.pending = False


def _request_deferred_sync(layer: Labels) -> None:
    state = _PENDING_LABELS_SYNCS.get(layer)
    if state is None:
        state = _PendingLabelsSync(layer_ref=ref(layer))
        _PENDING_LABELS_SYNCS[layer] = state
        layer._slicing_state.loaded_data.connect(_on_loaded_data_changed(state))

    state.pending = True
    # Edge-case guard: avoid a lost wakeup if napari finishes the async slice
    # while this deferred state is being installed:
    #
    # 1. `sync_labels_display_after_colormap_change(...)` sees
    #    `layer.loaded == False`.
    # 2. Harpy enters this deferred path.
    # 3. Napari finishes the async slice and emits `loaded_data`.
    # 4. `layer.loaded` is now true.
    # 5. Harpy connects to `loaded_data`.
    # 6. Harpy sets `pending = True`.
    # 7. No more `loaded_data` signal is guaranteed to happen.
    # 8. Without this guard, the pending sync could never run.
    if layer.loaded:
        _queue_pending_sync(state)


def _on_loaded_data_changed(state: _PendingLabelsSync) -> Callable[[], None]:
    def _callback() -> None:
        # The deferred callback must not keep a removed napari layer alive.
        # If the layer has already been garbage-collected, the weak reference
        # returns None and there is nothing left to synchronize.
        layer = state.layer_ref()
        if layer is None or not state.pending or not layer.loaded:
            return
        _queue_pending_sync(state)

    return _callback


def _queue_pending_sync(state: _PendingLabelsSync) -> None:
    if state.queued:
        return

    state.queued = True
    QTimer.singleShot(0, lambda: _run_pending_sync(state))


def _run_pending_sync(state: _PendingLabelsSync) -> None:
    state.queued = False
    # The queued Qt callback may run after the layer was removed.
    # Keep the state-to-layer link weak so delayed sync work never owns the layer.
    layer = state.layer_ref()
    if layer is None or not state.pending:
        return
    if not layer.loaded:
        return

    state.pending = False
    _sync_labels_display_now(layer)
