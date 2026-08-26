"""Compact palette-indexed VisPy visuals for tiled point payloads.

Renderer ownership is::

    VispyTiledPointsLayer
            |
            +-- owns one _ValuePaletteTexture
            |       shared by every tile visual
            |
            +-- owns _GpuTileResidency
                    |
                    +-- retains one _VispyTileResource per logical tile
                            |
                            +-- owns one _TiledPointsTileVisualNode
                                    |
                                    +-- generated from _TiledPointsTileVisual
                                    +-- owns one tile vertex buffer
                                    +-- borrows the shared palette texture
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt
from vispy.gloo import Texture2D, VertexBuffer
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual

_MAX_EXACT_FLOAT32_INTEGER: Final = 2**24

_VERTEX_SHADER: Final = """
attribute vec2 a_position;
attribute float a_value_id;

uniform vec2 u_tile_offset;
uniform float u_point_diameter;
uniform float u_pixel_scale;

varying float v_value_id;

void main() {
    vec2 cache_relative_position = a_position + u_tile_offset;
    gl_Position = $transform(vec4(cache_relative_position, 0.0, 1.0));
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
    every tile program. A point's canonical ``value_id`` selects one palette
    texel, so palette updates change colours without replacing tile vertex
    buffers. Tile resources borrow this object; only the layer renderer closes
    it.
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
        """Return the shared gloo texture used by tile programs."""
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


class _TiledPointsTileVisual(Visual):
    """Define the low-level drawing implementation for one logical point tile.

    ``create_visual_node()`` converts this visual class into the
    ``_TiledPointsTileVisualNode`` scene-node class. Each
    ``_VispyTileResource`` owns one node instance, whose visual owns the tile
    vertex buffer and shader state while borrowing the renderer's shared
    ``_ValuePaletteTexture``.
    """

    def __init__(
        self,
        location: npt.NDArray[np.float32],
        value_id: npt.NDArray[np.uint32],
        *,
        tile_offset: tuple[float, float],
        point_diameter: float,
        opacity: float,
        palette: _ValuePaletteTexture,
    ) -> None:
        if len(location) != len(value_id) or len(value_id) == 0:
            raise ValueError("Tile locations and value IDs must be nonempty and aligned.")
        maximum_value_id = int(value_id.max())
        if maximum_value_id >= palette.value_count:
            raise ValueError("A tile value ID exceeds the complete value palette.")
        if maximum_value_id > _MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError("Tile value IDs exceed exact float32 integer representation.")

        super().__init__(vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self._point_count = len(value_id)
        self._point_diameter = float(point_diameter)
        self._opacity = float(opacity)
        self._closed = False

        data = np.empty(
            self._point_count,
            dtype=np.dtype([("a_position", np.float32, 2), ("a_value_id", np.float32)]),
        )
        data["a_position"] = location
        data["a_value_id"] = value_id
        self._vertex_buffer = VertexBuffer()
        self._vertex_buffer.set_data(data, copy=True)
        self.shared_program.bind(self._vertex_buffer)
        self.shared_program["u_tile_offset"] = tuple(float(value) for value in tile_offset)
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
        """Return the number of uploaded tile rows."""
        return self._point_count

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
        """Return the layer-opacity multiplier applied by this tile."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = float(value)
        self.shared_program["u_layer_opacity"] = self._opacity
        self.update()

    def close(self) -> None:
        """Release the tile vertex buffer exactly once."""
        if self._closed:
            return
        self._closed = True
        self._vertex_buffer.delete()

    def _prepare_transforms(self, view: _TiledPointsTileVisual) -> None:
        view.view_program.vert["transform"] = view.get_transform()

    def _prepare_draw(self, view: _TiledPointsTileVisual | None = None) -> bool:
        if self._closed:
            return False
        target = self if view is None else view
        target.view_program["u_pixel_scale"] = target.transforms.pixel_scale
        return True

    def _compute_bounds(self, axis: int, view: _TiledPointsTileVisual) -> None:
        del axis, view
        # The logical napari layer owns the complete persistent dataset extent.
        return None


_TiledPointsTileVisualNode = create_visual_node(_TiledPointsTileVisual)
