"""Tile-retaining VisPy layer for cache-backed point snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from napari._vispy.layers.base import VispyBaseLayer
from vispy.scene.visuals import Compound

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderSnapshot,
    TiledPointsRenderTile,
    TileResidencyKey,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.vispy.residency import (
    _GpuTileCapacityError,
    _GpuTileResidency,
)
from napari_harpy.viewer.tiled_points.vispy.visuals import (
    _MAX_EXACT_FLOAT32_INTEGER,
    _TiledPointsTileVisualNode,
    _ValuePaletteTexture,
)

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.utils.events import Event


class _VispyTileResource:
    """Own one concrete VisPy tile retained through ``_GpuTileResource``.

    This class structurally implements the residency protocol while also
    exposing renderer-only state such as visibility, point diameter, and
    opacity. It owns one ``_TiledPointsTileVisualNode`` generated from
    ``_TiledPointsTileVisual`` and releases that node and its vertex buffer
    through ``close()``. The node borrows the renderer-owned shared
    ``_ValuePaletteTexture``; this resource does not close that texture.
    """

    def __init__(
        self,
        tile: TiledPointsRenderTile,
        *,
        root: Compound,
        palette: _ValuePaletteTexture,
        point_diameter: float,
        opacity: float,
    ) -> None:
        self.key = tile.key
        self.point_count = tile.point_count
        self.resident_bytes = tile.resident_bytes
        self._root = root
        self._visual = _TiledPointsTileVisualNode(
            tile.location,
            tile.value_id,
            tile_offset=(
                tile.key.tile_x * tile.tile_size,
                tile.key.tile_y * tile.tile_size,
            ),
            point_diameter=point_diameter,
            opacity=opacity,
            palette=palette,
        )
        self._visual.visible = False
        self._root.add_subvisual(self._visual)
        self._closed = False

    @property
    def visible(self) -> bool:
        """Return whether this tile participates in the active snapshot."""
        return bool(self._visual.visible)

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visual.visible = value

    @property
    def point_diameter(self) -> float:
        """Return the visual's logical-pixel point diameter."""
        return self._visual.point_diameter

    @point_diameter.setter
    def point_diameter(self, value: float) -> None:
        self._visual.point_diameter = value

    @property
    def opacity(self) -> float:
        """Return the tile's layer-opacity multiplier."""
        return self._visual.opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._visual.opacity = value

    def close(self) -> None:
        """Detach and release the tile visual exactly once."""
        if self._closed:
            return
        self._closed = True
        self._root.remove_subvisual(self._visual)
        self._visual.close()


class VispyTiledPointsLayer(VispyBaseLayer[TiledPointsLayerModel]):
    """Retain palette-indexed tile visuals and activate snapshots atomically.

    Napari constructs this GUI-thread renderer through the visual mapping
    installed by ``register_tiled_points_layer()`` when a
    ``TiledPointsLayerModel`` is inserted into a GUI viewer::

        TiledPointsLayerModel added to napari
                        |
                        v
        napari create_vispy_layer(model, font_info)
                        |
                        v
        VispyTiledPointsLayer(model, font_info)

    Like napari's standard layer renderers, this adapter exposes one root
    ``node`` that napari attaches to the canvas scene. Here that node is a
    ``Compound`` containing the dynamically retained tile visuals::

        napari canvas
                |
                +-- view.scene
                        |
                        +-- VispyTiledPointsLayer.node
                                Compound root for the complete logical layer
                                |
                                +-- _TiledPointsTileVisualNode A
                                +-- _TiledPointsTileVisualNode B
                                +-- _TiledPointsTileVisualNode C

    The root-node contract and use of a compound visual follow napari's normal
    rendering architecture. The dynamic, independently retained tile
    subvisuals are specific to this cache-backed renderer.

    The model-to-renderer contract is::

        model state or event              renderer responsibility
        --------------------              -----------------------
        data geometry and origins   --->  coordinate transform
        value_palette              --->  shared palette texture
        max_gpu_tile_bytes         --->  GPU tile-residency bound
        point_diameter event       --->  point-size uniforms
        value_palette event        --->  palette-texture update
        render_snapshot event      --->  atomic tile activation
        base layer events          --->  visibility, opacity, blending,
                                          ordering, and layer transform

    The logical model never contains visible point coordinates or owns GPU
    resources. This renderer performs no cache planning, Zarr reads, or value
    selection; it accepts complete immutable snapshots produced by the worker
    boundary and mutates VisPy only on the GUI thread. It reads but does not
    choose or mutate the model's palette. Closure disconnects model events and
    releases renderer-owned buffers, texture, scene nodes, and residency state.
    """

    def __init__(self, layer: TiledPointsLayerModel, font_info: FontInfo) -> None:
        if layer.data.value_count - 1 > _MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError(
                "The tiled-points renderer cannot represent this cache vocabulary exactly through float32 value IDs."
            )
        root = Compound([])
        super().__init__(layer, root, font_info=font_info)
        self._closed = False
        self._palette = _ValuePaletteTexture(
            layer.value_palette,
            maximum_texture_size=self.MAX_TEXTURE_SIZE_2D,
        )
        self._gpu_tile_residency = _GpuTileResidency(layer.max_gpu_tile_bytes)
        self._active_keys: tuple[TileResidencyKey, ...] = ()
        self._pending_keys: tuple[TileResidencyKey, ...] = ()
        self._coordinate_uploads: dict[TileResidencyKey, int] = {}
        self._palette_update_count = 0

        layer.events.render_snapshot.connect(self._on_render_snapshot)
        layer.events.value_palette.connect(self._on_value_palette_change)
        layer.events.point_diameter.connect(self._on_point_diameter_change)
        self.reset()

    @property
    def active_keys(self) -> tuple[TileResidencyKey, ...]:
        """Return the complete active tile membership in snapshot order."""
        return self._active_keys

    @property
    def pending_keys(self) -> tuple[TileResidencyKey, ...]:
        """Return pending tile membership while synchronous preparation runs."""
        return self._pending_keys

    @property
    def resident_tile_count(self) -> int:
        """Return the number of retained GPU tile resources."""
        return self._gpu_tile_residency.tile_count

    @property
    def resident_point_count(self) -> int:
        """Return points represented by retained GPU tile resources."""
        return self._gpu_tile_residency.point_count

    @property
    def resident_gpu_tile_bytes(self) -> int:
        """Return logical bytes retained by tile-owned GPU buffers."""
        return self._gpu_tile_residency.resident_bytes

    @property
    def palette_gpu_bytes(self) -> int:
        """Return logical bytes occupied by the shared palette texture."""
        return self._palette.resident_bytes

    @property
    def coordinate_upload_count(self) -> int:
        """Return the number of successfully created tile GPU resources."""
        return sum(self._coordinate_uploads.values())

    @property
    def coordinate_uploads_by_key(self) -> dict[TileResidencyKey, int]:
        """Return a copy of per-key coordinate upload counts."""
        return dict(self._coordinate_uploads)

    @property
    def eviction_count(self) -> int:
        """Return the number of inactive resources evicted for capacity."""
        return self._gpu_tile_residency.eviction_count

    @property
    def palette_update_count(self) -> int:
        """Return successful palette texture replacement count."""
        return self._palette_update_count

    def apply_snapshot(self, snapshot: TiledPointsRenderSnapshot) -> bool:
        """Prepare and atomically activate one complete within-budget snapshot.

        An over-budget snapshot performs no GPU work and leaves the active
        visual unchanged. Capacity or upload failures also retain the active
        snapshot and are reported through ``layer.events.render_error``.
        """
        if self._closed:
            return False
        if not isinstance(snapshot, TiledPointsRenderSnapshot):
            raise ValueError("`snapshot` must be TiledPointsRenderSnapshot.")
        if snapshot.cache_generation_id != self.layer.data.cache_generation_id:
            raise ValueError("Render snapshot cache generation differs from the layer dataset.")
        if not snapshot.within_budget:
            return False

        pending_keys = tuple(tile.key for tile in snapshot.tiles)
        tile_by_key = {tile.key: tile for tile in snapshot.tiles}
        existing_keys: set[TileResidencyKey] = set()
        missing: list[TiledPointsRenderTile] = []
        for key in pending_keys:
            resource = self._gpu_tile_residency.get(key)
            if resource is None:
                missing.append(tile_by_key[key])
            else:
                existing_keys.add(key)

        self._pending_keys = pending_keys
        protected = set(self._active_keys) | existing_keys
        try:
            self._gpu_tile_residency.prepare_capacity(
                required_new_bytes=sum(tile.resident_bytes for tile in missing),
                protected_keys=protected,
            )
        except _GpuTileCapacityError as error:
            self._clear_pending()
            self.layer.events.render_error(value=error)
            return False

        created: list[TileResidencyKey] = []
        try:
            for tile in missing:
                resource = self._create_tile_resource(tile)
                try:
                    self._gpu_tile_residency.retain(resource)
                except Exception:
                    resource.close()
                    raise
                created.append(tile.key)
                self._coordinate_uploads[tile.key] = self._coordinate_uploads.get(tile.key, 0) + 1
            if missing:
                # The Compound root was empty when napari first applied its GL
                # state, so propagate the current blending mode to new children.
                self._on_blending_change()
        except Exception as error:  # noqa: BLE001
            for key in created:
                self._gpu_tile_residency.discard(key)
            self._clear_pending()
            self.layer.events.render_error(value=error)
            return False

        # VisPy visibility setters schedule ``SceneCanvas.update()``; they do
        # not draw synchronously. SceneCanvas coalesces repeated requests, and
        # this GUI-thread block never yields to Qt, so the next frame observes
        # only the final complete tile membership.
        active_set = set(pending_keys)
        for key in self._active_keys:
            resource = self._gpu_tile_residency.get(key)
            if resource is not None and key not in active_set:
                cast(_VispyTileResource, resource).visible = False
        for key in pending_keys:
            resource = self._gpu_tile_residency.get(key)
            if resource is None:
                raise RuntimeError("Prepared GPU tile resource disappeared before snapshot activation.")
            cast(_VispyTileResource, resource).visible = True
        self._active_keys = pending_keys
        self._clear_pending()
        self.node.update()
        return True

    def close(self) -> None:
        """Disconnect and release every renderer-owned resource exactly once."""
        if self._closed:
            return
        self._closed = True
        self._active_keys = ()
        self._clear_pending()
        self._gpu_tile_residency.clear()
        self._palette.close()
        super().close()

    def _create_tile_resource(self, tile: TiledPointsRenderTile) -> _VispyTileResource:
        return _VispyTileResource(
            tile,
            root=self.node,
            palette=self._palette,
            point_diameter=self.layer.point_diameter,
            opacity=self.layer.opacity,
        )

    def _clear_pending(self) -> None:
        self._pending_keys = ()

    def _on_render_snapshot(self, event: Event) -> None:
        self.apply_snapshot(event.value)

    def _on_value_palette_change(self, event: Event) -> None:
        """Upload the shared palette and schedule its use in the next frame.

        Updating the standalone texture resource does not itself notify the
        scene graph.  Request one redraw explicitly; every resident tile uses
        this same texture, so no tile coordinates need to be uploaded again.
        """
        if self._closed:
            return
        self._palette.update(event.value)
        self._palette_update_count += 1
        self.node.update()

    def _on_point_diameter_change(self, event: Event) -> None:
        if self._closed:
            return
        for key in self._gpu_tile_residency.keys:
            resource = self._gpu_tile_residency.get(key)
            if resource is not None:
                cast(_VispyTileResource, resource).point_diameter = event.value

    def _on_opacity_change(self) -> None:
        """Apply layer opacity to dynamically added tile programs."""
        super()._on_opacity_change()
        # VispyBaseLayer connects this inherited opacity callback before this
        # subclass creates its GPU residency. Keep the ``None`` branch as a
        # defensive guard against a callback during partial construction.
        gpu_tile_residency = getattr(self, "_gpu_tile_residency", None)
        if gpu_tile_residency is None:
            return
        for key in gpu_tile_residency.keys:
            resource = gpu_tile_residency.get(key)
            if resource is not None:
                cast(_VispyTileResource, resource).opacity = self.layer.opacity

    def _on_data_change(self) -> None:
        """Drop generation-bound GPU resources after logical data replacement."""
        if getattr(self, "_closed", False):
            return
        gpu_tile_residency = getattr(self, "_gpu_tile_residency", None)
        if gpu_tile_residency is not None:
            gpu_tile_residency.clear()
            self._active_keys = ()
            self._clear_pending()
        self.node.update()
        self._on_matrix_change()

    def _on_matrix_change(self) -> None:
        """Apply the napari affine to coordinates relative to the cache origin.

        Tile vertices retain their small tile-local coordinates and tile-grid
        offsets.  Precomposing the shared cache origin into napari's float64
        layer matrix avoids adding a potentially large absolute origin in the
        float32 vertex shader, where that addition could erase local detail.
        """
        super()._on_matrix_change()
        matrix = self.node.transform.matrix.copy()
        cache_origin = np.asarray((self.layer.data.x_origin, self.layer.data.y_origin), dtype=np.float64)
        matrix[-1, :2] += cache_origin @ matrix[:2, :2]
        self.node.transform.matrix = matrix
