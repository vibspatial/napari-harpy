"""Constant-resource VisPy layer for cache-backed point snapshots."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from napari._vispy.layers.base import VispyBaseLayer
from vispy.scene.visuals import Compound

from napari_harpy.viewer.tiled_points.contracts import (
    TiledPointsRenderResult,
    TiledPointsRenderSnapshot,
)
from napari_harpy.viewer.tiled_points.napari.layer import TiledPointsLayerModel
from napari_harpy.viewer.tiled_points.render_batch import (
    MAX_EXACT_FLOAT32_INTEGER,
    TILED_POINTS_VERTEX_DTYPE,
    pack_snapshot_vertices,
)
from napari_harpy.viewer.tiled_points.vispy.visuals import (
    _TiledPointsSnapshotVisualNode,
    _ValuePaletteTexture,
)

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.utils.events import Event


class _SnapshotPayloadCapacityError(RuntimeError):
    """Report that one complete candidate vertex payload cannot fit."""


class VispyTiledPointsLayer(VispyBaseLayer[TiledPointsLayerModel]):
    """Stage complete palette-indexed snapshots through one visual and VBO.

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
    ``Compound`` containing one snapshot visual::

        napari canvas
                |
                +-- view.scene
                        |
                        +-- VispyTiledPointsLayer.node
                                Compound root for the complete logical layer
                                |
                                +-- _TiledPointsSnapshotVisualNode
                                        one shader/program path
                                        one stable vertex buffer

    The root-node contract and use of a compound visual follow napari's normal
    rendering architecture. Logical cache tiles remain snapshot and CPU-
    residency units, but they are packed into one cache-relative vertex array
    before this stable visual replaces its VBO contents.

    The model-to-renderer contract is::

        model state or event              renderer responsibility
        --------------------              -----------------------
        data geometry and origins   --->  coordinate transform
        value_palette              --->  shared palette texture
        max_gpu_tile_bytes         --->  candidate vertex-payload byte bound
        point_diameter event       --->  point-size uniforms
        value_palette event        --->  palette-texture update
        render_snapshot event      --->  one packed payload replacement
        render_snapshot_result     <---  generation-bound applied acknowledgement
        base layer events          --->  visibility, opacity, blending,
                                          ordering, and layer transform

    The logical model never contains visible point coordinates or owns GPU
    resources. This renderer performs no cache planning, Zarr reads, or value
    selection; it accepts complete immutable snapshots produced by the worker
    boundary, acknowledges whether each candidate was applied, and mutates
    VisPy only on the GUI thread. It reads but does not choose or mutate the
    model's palette. Closure disconnects model events and releases
    renderer-owned buffer, texture, and scene node.
    """

    def __init__(self, layer: TiledPointsLayerModel, font_info: FontInfo) -> None:
        if layer.data.value_count - 1 > MAX_EXACT_FLOAT32_INTEGER:
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
        self._snapshot_visual = _TiledPointsSnapshotVisualNode(
            point_diameter=layer.point_diameter,
            opacity=layer.opacity,
            palette=self._palette,
        )
        self.node.add_subvisual(self._snapshot_visual)
        self._palette_update_count = 0
        self._last_pack_ms = 0.0

        layer.events.render_snapshot.connect(self._on_render_snapshot)
        layer.events.value_palette.connect(self._on_value_palette_change)
        layer.events.point_diameter.connect(self._on_point_diameter_change)
        self.reset()

    @property
    def visual_count(self) -> int:
        """Return the fixed renderer-owned snapshot visual count."""
        return 0 if self._closed else 1

    @property
    def vbo_count(self) -> int:
        """Return the fixed renderer-owned vertex-buffer count."""
        return 0 if self._closed else 1

    @property
    def active_point_count(self) -> int:
        """Return rows in the active combined vertex payload."""
        return 0 if self._closed else self._snapshot_visual.point_count

    @property
    def active_vertex_bytes(self) -> int:
        """Return logical bytes in the active combined vertex payload."""
        return 0 if self._closed else self._snapshot_visual.active_vertex_bytes

    @property
    def payload_replacement_count(self) -> int:
        """Return successful nonempty replacements of the stable VBO."""
        return self._snapshot_visual.payload_replacement_count

    @property
    def last_pack_ms(self) -> float:
        """Return GUI-thread packing time for the latest accepted snapshot."""
        return self._last_pack_ms

    @property
    def last_vertex_staging_ms(self) -> float:
        """Return synchronous staging time for the latest accepted payload."""
        return self._snapshot_visual.last_staging_ms

    @property
    def point_draw_submission_count(self) -> int:
        """Return point draw submissions made by this topology per frame."""
        return 1 if self.active_point_count else 0

    @property
    def palette_gpu_bytes(self) -> int:
        """Return logical bytes occupied by the shared palette texture."""
        return self._palette.resident_bytes

    @property
    def palette_update_count(self) -> int:
        """Return successful palette texture replacement count."""
        return self._palette_update_count

    def apply_snapshot(self, snapshot: TiledPointsRenderSnapshot) -> bool:
        """Prepare and atomically activate one complete within-budget snapshot.

        An over-budget snapshot performs no renderer work and leaves the active
        visual unchanged. Validation, capacity, and packing failures happen
        before VBO mutation. A staging failure declines the candidate activation,
        but the single-VBO design cannot promise that physical contents remain
        drawable after mutation begins because VisPy may defer the actual GPU
        upload until drawing. If stronger recovery becomes necessary, one
        visual/program can instead own two fixed VBOs: stage into the inactive
        VBO, bind it only after synchronous staging succeeds, and retain the
        previously active VBO as a fallback. Guaranteed recovery from a deferred
        upload or draw failure would additionally require detecting that failure,
        rebinding the previous VBO, and requesting another draw.
        """
        if self._closed:
            return False
        if not isinstance(snapshot, TiledPointsRenderSnapshot):
            raise ValueError("`snapshot` must be TiledPointsRenderSnapshot.")
        if snapshot.cache_generation_id != self.layer.data.cache_generation_id:
            raise ValueError("Render snapshot cache generation differs from the layer dataset.")
        if not snapshot.within_budget:
            return False

        point_count = snapshot.rendered_point_count
        required_bytes = point_count * TILED_POINTS_VERTEX_DTYPE.itemsize
        try:
            if point_count > self.layer.hard_render_point_budget:
                raise _SnapshotPayloadCapacityError(
                    f"Snapshot contains {point_count} points, exceeding "
                    f"hard_render_point_budget={self.layer.hard_render_point_budget}."
                )
            if required_bytes > self.layer.max_gpu_tile_bytes:
                raise _SnapshotPayloadCapacityError(
                    f"Snapshot vertex payload requires {required_bytes} bytes, exceeding "
                    f"max_gpu_tile_bytes={self.layer.max_gpu_tile_bytes}."
                )
            started = time.perf_counter()
            vertices = pack_snapshot_vertices(snapshot, value_count=self.layer.data.value_count)
            pack_ms = (time.perf_counter() - started) * 1_000.0
            self._validate_vertex_payload(vertices, point_count=point_count, required_bytes=required_bytes)
            self._snapshot_visual.replace_vertices(vertices)
        except Exception as error:  # noqa: BLE001
            self.layer.events.render_error(value=error)
            return False

        self._last_pack_ms = pack_ms
        self.node.update()
        return True

    def close(self) -> None:
        """Disconnect and release every renderer-owned resource exactly once."""
        if self._closed:
            return
        self._closed = True
        self.node.remove_subvisual(self._snapshot_visual)
        self._snapshot_visual.close()
        self._palette.close()
        super().close()

    @staticmethod
    def _validate_vertex_payload(vertices: np.ndarray, *, point_count: int, required_bytes: int) -> None:
        if (
            not isinstance(vertices, np.ndarray)
            or vertices.ndim != 1
            or vertices.dtype != TILED_POINTS_VERTEX_DTYPE
            or not vertices.flags.c_contiguous
            or not vertices.flags.owndata
            or len(vertices) != point_count
            or vertices.nbytes != required_bytes
        ):
            raise ValueError("Packed snapshot does not satisfy the canonical vertex-payload contract.")

    def _on_render_snapshot(self, event: Event) -> None:
        snapshot = event.value
        if not isinstance(snapshot, TiledPointsRenderSnapshot):
            raise ValueError("The render-snapshot event must carry TiledPointsRenderSnapshot.")
        applied = self.apply_snapshot(snapshot)
        self.layer.events.render_snapshot_result(
            value=TiledPointsRenderResult(
                request_generation=snapshot.request_generation,
                selection_generation=snapshot.selection_generation,
                applied=applied,
            )
        )

    def _on_value_palette_change(self, event: Event) -> None:
        """Upload the shared palette and schedule its use in the next frame.

        Updating the standalone texture resource does not itself notify the
        scene graph. Request one redraw explicitly; the snapshot visual uses
        this same texture, so no point coordinates need to be uploaded again.
        """
        if self._closed:
            return
        self._palette.update(event.value)
        self._palette_update_count += 1
        self.node.update()

    def _on_point_diameter_change(self, event: Event) -> None:
        if self._closed:
            return
        self._snapshot_visual.point_diameter = event.value

    def _on_opacity_change(self) -> None:
        """Apply layer opacity to the one snapshot program."""
        super()._on_opacity_change()
        # VispyBaseLayer connects this inherited opacity callback before this
        # subclass creates its snapshot visual. Keep the ``None`` branch as a
        # defensive guard against a callback during partial construction.
        snapshot_visual = getattr(self, "_snapshot_visual", None)
        if snapshot_visual is None:
            return
        snapshot_visual.opacity = self.layer.opacity

    def _on_data_change(self) -> None:
        """Redraw unchanged cache data after napari's generic refresh signal.

        ``VispyBaseLayer`` requires this callback and connects it to
        ``layer.events.set_data``. ``TiledPointsLayerModel.data`` is immutable,
        so this signal cannot represent an accepted cache replacement. Napari's
        generic ``layer.refresh()`` can still emit it for the existing data; the
        active VBO, logical keys, palette, and cache-origin transform must remain
        intact in that case.
        """
        if getattr(self, "_closed", False):
            return
        self.node.update()

    def _on_matrix_change(self) -> None:
        """Apply the napari affine to coordinates relative to the cache origin.

        Snapshot vertices contain cache-relative coordinates with logical tile
        offsets already folded in. Precomposing the shared cache origin into
        napari's float64 layer matrix avoids adding a potentially large absolute
        origin in the float32 vertex shader, where it could erase local detail.
        """
        super()._on_matrix_change()
        matrix = self.node.transform.matrix.copy()
        cache_origin = np.asarray((self.layer.data.x_origin, self.layer.data.y_origin), dtype=np.float64)
        matrix[-1, :2] += cache_origin @ matrix[:2, :2]
        self.node.transform.matrix = matrix
