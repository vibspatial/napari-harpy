"""Compact palette-indexed VisPy visual for complete point snapshots.

Renderer ownership is::

    VispyTiledPointsLayer
            |
            +-- owns one _ValuePaletteTexture
            |       shared with the snapshot visual
            |
            +-- owns one _TiledPointsSnapshotVisualNode
                    |
                    +-- generated from _TiledPointsSnapshotVisual
                    +-- owns one stable snapshot vertex buffer
                    +-- borrows the shared palette texture
"""

from __future__ import annotations

import math
import time
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt
from vispy.gloo import Texture2D, VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

from napari_harpy.viewer.tiled_points.render_batch import TILED_POINTS_VERTEX_DTYPE

# Interleaved snapshot-VBO attribute contract:
#
# ``a_position``
#     Float32 ``(x, y)`` position relative to the shared cache origin. Logical
#     tile offsets are already folded in. The renderer's float64 root transform
#     restores the cache origin and applies the napari affine.
#
# ``a_value_id``
#     Canonical cache-vocabulary index encoded as an exactly representable
#     float32 integer. The fragment shader uses it to select one RGBA texel from
#     the shared palette texture. It is not a source gene or string value.
#
# These names must match the fields in ``TILED_POINTS_VERTEX_DTYPE`` because
# ``Program.bind(VertexBuffer)`` binds structured fields by attribute name.
_VERTEX_SHADER: Final = """
attribute vec2 a_position;   // Cache-relative (x, y).
attribute float a_value_id;  // Exact integer palette index.

uniform float u_point_diameter;
uniform float u_pixel_scale;

varying float v_value_id;

void main() {
    gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    gl_PointSize = u_point_diameter * u_pixel_scale;
    v_value_id = a_value_id;
}
"""

_FRAGMENT_SHADER: Final = """
uniform sampler2D u_palette;
uniform float u_palette_width;
uniform float u_palette_height;
uniform float u_point_diameter;
uniform float u_pixel_scale;
uniform float u_layer_opacity;

varying float v_value_id;

void main() {
    vec2 centered = gl_PointCoord - vec2(0.5, 0.5);
    float radius = length(centered);
    float antialias_width = 1.0 / max(u_point_diameter * u_pixel_scale, 1.0);
    float coverage = 1.0 - smoothstep(0.5 - antialias_width, 0.5, radius);
    if (coverage <= 0.0) {
        discard;
    }

    float row = floor(v_value_id / u_palette_width);
    float column = v_value_id - row * u_palette_width;
    vec2 palette_coordinate = vec2(
        (column + 0.5) / u_palette_width,
        (row + 0.5) / u_palette_height
    );
    vec4 color = texture2D(u_palette, palette_coordinate);
    gl_FragColor = vec4(color.rgb, color.a * coverage * u_layer_opacity);
}
"""


class _ValuePaletteTexture:
    """Own the renderer-wide value-ID-to-RGBA lookup texture.

    ``VispyTiledPointsLayer`` owns one instance and shares its texture with
    the snapshot program. A point's canonical ``value_id`` selects one palette
    texel, so palette updates change colours without replacing snapshot vertex
    data. The snapshot visual borrows this object; only the layer renderer
    closes it.
    """

    def __init__(self, palette: npt.NDArray[np.uint8], *, maximum_texture_size: int) -> None:
        if (
            not isinstance(maximum_texture_size, Integral)
            or isinstance(maximum_texture_size, bool)
            or maximum_texture_size <= 0
        ):
            raise ValueError("`maximum_texture_size` must be a positive integer.")
        maximum_texture_size = int(maximum_texture_size)
        self._value_count = len(palette)
        self._width = min(self._value_count, maximum_texture_size)
        self._height = math.ceil(self._value_count / self._width)
        if self._height > maximum_texture_size:
            raise ValueError(
                f"The {self._value_count}-row value palette exceeds the supported "
                f"{maximum_texture_size} by {maximum_texture_size} texture capacity."
            )
        packed = self._pack(palette)
        self._texture = Texture2D(
            packed,
            format="rgba",
            interpolation="nearest",
            wrapping="clamp_to_edge",
        )
        self._closed = False

    @property
    def texture(self) -> Texture2D:
        """Return the shared gloo texture used by the snapshot program."""
        return self._texture

    @property
    def value_count(self) -> int:
        """Return the number of canonical palette rows."""
        return self._value_count

    @property
    def width(self) -> int:
        """Return the packed texture width in texels."""
        return self._width

    @property
    def height(self) -> int:
        """Return the packed texture height in texels."""
        return self._height

    @property
    def resident_bytes(self) -> int:
        """Return logical bytes occupied by the packed RGBA texels."""
        return self._width * self._height * 4

    def update(self, palette: npt.NDArray[np.uint8]) -> None:
        """Replace palette texels without replacing the texture object."""
        if self._closed:
            raise RuntimeError("Cannot update a closed value-palette texture.")
        if len(palette) != self._value_count:
            raise ValueError("A palette update must preserve the canonical value count.")
        self._texture.set_data(self._pack(palette), copy=True)

    def close(self) -> None:
        """Release the texture exactly once."""
        if self._closed:
            return
        self._closed = True
        self._texture.delete()

    def _pack(self, palette: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        packed = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        packed.reshape(-1, 4)[: self._value_count, :] = palette
        return packed


class _TiledPointsSnapshotVisual(Visual):
    """Draw one complete packed point snapshot through one stable VBO.

    ``create_visual_node()`` converts this visual class into the
    ``_TiledPointsSnapshotVisualNode`` scene-node class. One renderer owns one
    node instance for its complete lifetime. The visual owns one vertex buffer
    and shader state while borrowing the renderer's shared palette texture.
    """

    def __init__(
        self,
        *,
        point_diameter: float,
        opacity: float,
        palette: _ValuePaletteTexture,
    ) -> None:
        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self._point_count = 0
        self._point_diameter = float(point_diameter)
        self._opacity = float(opacity)
        self._payload_replacement_count = 0
        self._last_staging_ms = 0.0
        self._closed = False

        self._vertex_buffer = VertexBuffer(np.empty(0, dtype=TILED_POINTS_VERTEX_DTYPE))
        self.shared_program.bind(self._vertex_buffer)
        self.shared_program["u_point_diameter"] = self._point_diameter
        self.shared_program["u_layer_opacity"] = self._opacity
        self.shared_program["u_palette"] = palette.texture
        self.shared_program["u_palette_width"] = float(palette.width)
        self.shared_program["u_palette_height"] = float(palette.height)
        self.set_gl_state(
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        self._draw_mode = "points"
        self.freeze()

    @property
    def point_count(self) -> int:
        """Return the number of rows in the active snapshot payload."""
        return self._point_count

    @property
    def vertex_buffer(self) -> VertexBuffer:
        """Return the one stable vertex buffer owned by this visual."""
        return self._vertex_buffer

    @property
    def active_vertex_bytes(self) -> int:
        """Return logical bytes in the active snapshot payload."""
        return self._point_count * TILED_POINTS_VERTEX_DTYPE.itemsize

    @property
    def payload_replacement_count(self) -> int:
        """Return successful nonempty VBO payload replacements."""
        return self._payload_replacement_count

    @property
    def last_staging_ms(self) -> float:
        """Return synchronous staging time for the latest nonempty payload."""
        return self._last_staging_ms

    def replace_vertices(self, vertices: npt.NDArray[np.void]) -> None:
        """Stage one complete validated payload without replacing the VBO."""
        if self._closed:
            raise RuntimeError("Cannot replace vertices on a closed snapshot visual.")
        if (
            not isinstance(vertices, np.ndarray)
            or vertices.ndim != 1
            or vertices.dtype != TILED_POINTS_VERTEX_DTYPE
            or not vertices.flags.c_contiguous
        ):
            raise ValueError("`vertices` must be a C-contiguous canonical tiled-points vertex array.")
        if len(vertices) == 0:
            self._point_count = 0
            self._last_staging_ms = 0.0
            return

        started = time.perf_counter()
        # VisPy defers the physical GPU upload. This visual does not retain the
        # caller-owned array as its own immutable state, so copy it to prevent a
        # later caller mutation from changing the queued payload before drawing.
        self._vertex_buffer.set_data(vertices, copy=True)
        # Program.bind() creates field views whose sizes reflect the current
        # buffer, so refresh those views after resizing the stable VBO.
        self.shared_program.bind(self._vertex_buffer)
        staging_ms = (time.perf_counter() - started) * 1_000.0
        self._point_count = len(vertices)
        self._payload_replacement_count += 1
        self._last_staging_ms = staging_ms

    def clear_vertices(self) -> None:
        """Suppress drawing while retaining the one VBO and its allocation."""
        if self._closed:
            return
        self._point_count = 0
        self._last_staging_ms = 0.0

    @property
    def point_diameter(self) -> float:
        """Return the requested marker diameter in logical canvas pixels."""
        return self._point_diameter

    @point_diameter.setter
    def point_diameter(self, value: float) -> None:
        self._point_diameter = float(value)
        self.shared_program["u_point_diameter"] = self._point_diameter
        self.update()

    @property
    def opacity(self) -> float:
        """Return the layer-opacity multiplier applied by this snapshot."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = float(value)
        self.shared_program["u_layer_opacity"] = self._opacity
        self.update()

    def close(self) -> None:
        """Release the snapshot vertex buffer exactly once."""
        if self._closed:
            return
        self._closed = True
        self._vertex_buffer.delete()

    def _prepare_transforms(self, view: Visual) -> None:
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view: Visual | None = None) -> bool:
        if self._closed or self._point_count == 0:
            return False
        target = self if view is None else view
        target.view_program["u_pixel_scale"] = target.transforms.pixel_scale
        return True

    def _compute_bounds(self, axis: int, view: Visual) -> None:
        del axis, view
        # The logical napari layer owns the complete persistent dataset extent.
        return None


_TiledPointsSnapshotVisualNode = create_visual_node(_TiledPointsSnapshotVisual)
