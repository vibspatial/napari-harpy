from __future__ import annotations

from napari.layers import Labels


def sync_labels_display_after_colormap_change(layer: Labels) -> None:
    """Synchronize labels display after Harpy assigns a table-driven colormap.

    Napari async slicing can leave the currently displayed labels texture
    encoded with the previous colormap while the new texture-code-to-RGBA table
    has already reached vispy. For explicit Harpy recoloring actions, recompute
    the current labels slice synchronously and notify vispy without re-entering
    the public async `layer.refresh(...)` path.

    Workaround for https://github.com/napari/napari/issues/9188.
    """
    layer.set_view_slice()
    # `Labels.colormap = ...` can emit while async slicing still exposes a
    # placeholder/small-dtype slice. After recomputing the real slice, notify
    # vispy to rebuild the colormap shader/table for that now-current dtype.
    layer.events.colormap()
    # Upload the recomputed texture-code image; thumbnail/highlight refreshes
    # are unrelated to this async colormap synchronization bug.
    layer.events.set_data()
