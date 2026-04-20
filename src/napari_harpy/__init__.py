"""napari-harpy package."""

from importlib.metadata import PackageNotFoundError, version

from napari_harpy._app_state import HarpyAppState, get_or_create_app_state
from napari_harpy._interactive import Interactive

try:
    __version__ = version("napari-harpy")
except PackageNotFoundError:  # pragma: no cover - fallback during local development
    __version__ = "0.0.0"

__all__ = ["HarpyAppState", "Interactive", "__version__", "get_or_create_app_state"]
