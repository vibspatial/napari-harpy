from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData

FEATURE_EXTRACTION_IDLE_STATUS = "Feature extraction: choose a segmentation, table, and output key."


@dataclass(frozen=True)
class FeatureExtractionJob:
    """Immutable payload copied on the main thread and consumed by a worker."""

    job_id: int
    label_name: str
    image_name: str | None
    table_name: str
    coordinate_system: str | None
    feature_names: tuple[str, ...]
    feature_key: str
    overwrite_feature_key: bool


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Summary produced by a completed feature-extraction worker."""

    job_id: int
    label_name: str
    table_name: str
    feature_key: str


class FeatureExtractionController:
    """Manage widget-facing state for background feature extraction."""

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
        on_table_state_changed: Callable[[], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed
        self._on_table_state_changed = on_table_state_changed

        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_image_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_coordinate_system: str | None = None
        self._selected_feature_names: tuple[str, ...] = ()
        self._selected_feature_key: str | None = None
        self._overwrite_feature_key = False

        self._latest_requested_job_id = 0
        self._active_worker_job_id: int | None = None
        self._active_worker: Any | None = None

        self._status_message = FEATURE_EXTRACTION_IDLE_STATUS
        self._status_kind = "warning"

    @property
    def status_message(self) -> str:
        """Return the latest user-facing feature-extraction status message."""
        return self._status_message

    @property
    def status_kind(self) -> str:
        """Return the latest status level: info, warning, success, or error."""
        return self._status_kind

    @property
    def is_running(self) -> bool:
        """Return whether a feature-extraction worker is currently active."""
        return self._active_worker is not None

    @property
    def can_calculate(self) -> bool:
        """Return whether the current selection has the minimum data to run."""
        return (
            self._selected_spatialdata is not None
            and self._selected_label_name is not None
            and self._selected_table_name is not None
            and bool(self._selected_feature_names)
            and self._selected_feature_key is not None
            and bool(self._selected_feature_key.strip())
            and not self.is_running
        )

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_state_changed(self) -> None:
        if self._on_table_state_changed is not None:
            self._on_table_state_changed()
