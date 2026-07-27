"""Lazy exports for the parent Annotation widget and shared UI models."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from .models import AnnotationContext, ShapesAnnotationTarget
    from .widget import AnnotationWidget

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "models": ["AnnotationContext", "ShapesAnnotationTarget"],
        "widget": ["AnnotationWidget"],
    },
)

__all__ = ["AnnotationContext", "AnnotationWidget", "ShapesAnnotationTarget"]
