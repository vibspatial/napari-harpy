from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from napari.layers import Image

ImageDisplayMode = Literal["stack", "overlay"]

DEFAULT_OVERLAY_COLORS = (
    "#00FFFF",  # cyan
    "#FF00FF",  # magenta
    "#FFFF00",  # yellow
    "#00FF7F",  # green
    "#FF5050",  # red
    "#1E90FF",  # blue
    "#FFA500",  # orange
    "#9370DB",  # purple
)


@dataclass(frozen=True)
class ImageLoadResult:
    """Describe one image load/update result returned to the viewer."""

    layers: tuple[Image, ...]
    mode: ImageDisplayMode
    created: bool
    channels: tuple[int, ...] = ()

    @property
    def primary_layer(self) -> Image:
        """Return the layer that should become active after loading."""
        return self.layers[0]
