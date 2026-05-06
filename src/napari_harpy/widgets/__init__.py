"""Widget exports for napari-harpy."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .feature_extraction.widget import FeatureExtractionWidget
    from .object_classification.widget import ObjectClassificationWidget
    from .viewer.widget import ViewerWidget

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "feature_extraction.widget": ["FeatureExtractionWidget"],
        "object_classification.widget": ["ObjectClassificationWidget"],
        "viewer.widget": ["ViewerWidget"],
    },
)

__all__ = ["FeatureExtractionWidget", "ObjectClassificationWidget", "ViewerWidget"]
