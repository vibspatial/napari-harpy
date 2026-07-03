from __future__ import annotations

from napari.settings import get_settings
from napari.utils.triangulation_backend import TriangulationBackend, get_backend, set_backend

_NUMBA_TRIANGULATION_BACKEND = TriangulationBackend.numba
_UNSAFE_SHAPES_TRIANGULATION_BACKENDS = {
    TriangulationBackend.fastest_available,
    TriangulationBackend.bermuda,
}


def ensure_shapes_triangulation_backend() -> None:
    """Prefer napari's Numba Shapes triangulation backend."""
    settings = get_settings()
    settings_backend = settings.experimental.triangulation_backend
    runtime_backend = get_backend()

    if (
        settings_backend in _UNSAFE_SHAPES_TRIANGULATION_BACKENDS
        or runtime_backend in _UNSAFE_SHAPES_TRIANGULATION_BACKENDS
    ):
        settings.experimental.triangulation_backend = _NUMBA_TRIANGULATION_BACKEND
        if get_backend() != _NUMBA_TRIANGULATION_BACKEND:
            set_backend(_NUMBA_TRIANGULATION_BACKEND)
