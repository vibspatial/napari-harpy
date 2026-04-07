"""napari-harpy package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-harpy")
except PackageNotFoundError:  # pragma: no cover - fallback during local development
    __version__ = "0.0.0"

