"""napari-harpy package."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from . import core, datasets, headless, viewer, widgets
    from ._app_state import HarpyAppState, get_or_create_app_state
    from ._interactive import Interactive

try:
    __version__ = version("napari-harpy")
except PackageNotFoundError:  # pragma: no cover - fallback during local development
    __version__ = "0.0.0"

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submodules=["core", "datasets", "headless", "viewer", "widgets"],
    submod_attrs={
        "_app_state": ["HarpyAppState", "get_or_create_app_state"],
        "_interactive": ["Interactive"],
    },
)

__all__ = [
    "HarpyAppState",
    "Interactive",
    "__version__",
    "core",
    "datasets",
    "get_or_create_app_state",
    "headless",
    "viewer",
    "widgets",
]
