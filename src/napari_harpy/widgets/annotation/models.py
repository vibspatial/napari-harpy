"""Immutable UI state shared by the Annotation parent and its children."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from spatialdata import SpatialData

type ShapesAnnotationTargetMode = Literal["create_new", "edit_existing"]


@dataclass(frozen=True)
class ShapesAnnotationTarget:
    """Unambiguously describe the parent-owned Shapes target selection.

    ``AnnotationContext.shapes_target=None`` means that no target is available.
    This value object keeps that state distinct from create-new and couples an
    existing Shapes name to the edit-existing mode without invalid mode/name
    combinations leaking across parent-child boundaries.
    """

    mode: ShapesAnnotationTargetMode
    existing_shapes_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "create_new":
            if self.existing_shapes_name is not None:
                raise ValueError("Create-new shapes targets cannot carry an existing shapes name.")
            return

        if self.mode == "edit_existing":
            if self.existing_shapes_name is None or not self.existing_shapes_name.strip():
                raise ValueError("Edit-existing shapes targets require a shapes name.")
            return

        raise ValueError(f"Unknown shapes annotation target mode: {self.mode!r}.")

    @classmethod
    def create_new(cls) -> ShapesAnnotationTarget:
        """Return the target used to create a new Shapes element."""
        return cls(mode="create_new")

    @classmethod
    def edit_existing(cls, shapes_name: str) -> ShapesAnnotationTarget:
        """Return a target for an existing Shapes element."""
        return cls(mode="edit_existing", existing_shapes_name=shapes_name)


@dataclass(frozen=True)
class AnnotationContext:
    """Committed parent selection and Shapes edit state supplied to children."""

    sdata: SpatialData | None
    coordinate_system: str | None
    shapes_target: ShapesAnnotationTarget | None
    has_unsaved_shapes_changes: bool

    @property
    def saved_shapes_name(self) -> str | None:
        """Return the selected persisted Shapes name, if one exists."""
        target = self.shapes_target
        if target is None or target.mode != "edit_existing":
            return None
        return target.existing_shapes_name
