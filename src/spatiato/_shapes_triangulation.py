from __future__ import annotations

from typing import Literal

from napari.settings import get_settings
from napari.utils.triangulation_backend import TriangulationBackend, get_backend, set_backend

type ShapesTriangulationBackend = Literal["bermuda", "numba"]

_SUPPORTED_SHAPES_TRIANGULATION_BACKENDS: dict[str, TriangulationBackend] = {
    "bermuda": TriangulationBackend.bermuda,
    "numba": TriangulationBackend.numba,
}
_CONFIGURED_SHAPES_TRIANGULATION_BACKEND = TriangulationBackend.bermuda


def configure_shapes_triangulation_backend(backend: ShapesTriangulationBackend) -> None:
    """Configure the process-wide backend used by Harpy Shapes layers."""
    try:
        configured_backend = _SUPPORTED_SHAPES_TRIANGULATION_BACKENDS[backend]
    except (KeyError, TypeError) as error:
        valid_backends = ", ".join(_SUPPORTED_SHAPES_TRIANGULATION_BACKENDS)
        raise ValueError(
            f"Unknown Shapes triangulation backend {backend!r}. Valid options are: {valid_backends}."
        ) from error

    global _CONFIGURED_SHAPES_TRIANGULATION_BACKEND
    _CONFIGURED_SHAPES_TRIANGULATION_BACKEND = configured_backend
    ensure_shapes_triangulation_backend()


def ensure_shapes_triangulation_backend() -> None:
    """Keep napari aligned with Harpy's configured Shapes backend."""
    settings = get_settings()
    settings.experimental.triangulation_backend = _CONFIGURED_SHAPES_TRIANGULATION_BACKEND
    if get_backend() != _CONFIGURED_SHAPES_TRIANGULATION_BACKEND:
        set_backend(_CONFIGURED_SHAPES_TRIANGULATION_BACKEND)
