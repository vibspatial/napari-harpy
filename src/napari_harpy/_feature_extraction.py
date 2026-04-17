from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from napari_harpy._spatialdata import SpatialDataTableMetadata, get_table, validate_table_binding

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

FEATURE_EXTRACTION_IDLE_STATUS = "Feature extraction: choose a segmentation, table, and output key."
_INTENSITY_FEATURES = frozenset({"sum", "mean", "var", "min", "max", "kurtosis", "skew"})


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
        self._effective_table_name: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None
        self._table_binding_error: str | None = None
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
            and self._effective_table_name is not None
            and bool(self._selected_feature_names)
            and self._selected_feature_key is not None
            and bool(self._selected_feature_key.strip())
            and (not _requires_image(self._selected_feature_names) or self._selected_image_name is not None)
            and not self.is_running
        )

    @property
    def table_binding_error(self) -> str | None:
        """Return the latest table-binding validation error, if any."""
        return self._table_binding_error

    def bind(
        self,
        sdata: SpatialData | None,
        label_name: str | None,
        image_name: str | None,
        table_name: str | None,
        coordinate_system: str | None,
        feature_names: Sequence[str] | str | None,
        feature_key: str | None,
        *,
        overwrite_feature_key: bool = False,
    ) -> bool:
        """Bind the controller to the currently selected SpatialData inputs."""
        normalized_feature_names = _normalize_feature_names(feature_names)
        normalized_feature_key = None if feature_key is None else feature_key.strip()

        context_changed = (
            sdata is not self._selected_spatialdata
            or label_name != self._selected_label_name
            or image_name != self._selected_image_name
            or table_name != self._selected_table_name
            or coordinate_system != self._selected_coordinate_system
            or normalized_feature_names != self._selected_feature_names
            or normalized_feature_key != self._selected_feature_key
            or bool(overwrite_feature_key) is not self._overwrite_feature_key
        )

        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_image_name = image_name
        self._selected_table_name = table_name
        self._selected_coordinate_system = coordinate_system
        self._selected_feature_names = normalized_feature_names
        self._selected_feature_key = normalized_feature_key
        self._overwrite_feature_key = bool(overwrite_feature_key)

        self._effective_table_name = None
        self._selected_table_metadata = None
        self._table_binding_error = None

        if context_changed:
            self._cancel_pending_and_active_jobs()

        if sdata is None or label_name is None:
            self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return context_changed

        if table_name is None:
            self._set_status(
                "Feature extraction: choose an annotation table linked to the selected segmentation.",
                kind="warning",
            )
            return context_changed

        try:
            table_metadata = validate_table_binding(sdata, label_name, table_name)
        except ValueError as error:
            self._table_binding_error = str(error)
            self._set_status(f"Feature extraction: {error}", kind="warning")
            return context_changed

        self._effective_table_name = table_name
        self._selected_table_metadata = table_metadata

        if not normalized_feature_names:
            self._set_status("Feature extraction: choose at least one feature to calculate.", kind="warning")
            return context_changed

        if normalized_feature_key is None or not normalized_feature_key:
            self._set_status("Feature extraction: choose an output feature key.", kind="warning")
            return context_changed

        if _requires_image(normalized_feature_names) and image_name is None:
            self._set_status(
                "Feature extraction: choose an image before calculating intensity features.",
                kind="warning",
            )
            return context_changed

        if self.is_running:
            self._set_status("Feature extraction: calculation is running.", kind="info")
            return context_changed

        self._set_status("Feature extraction: ready to calculate.", kind="success")
        return context_changed

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_state_changed(self) -> None:
        if self._on_table_state_changed is not None:
            self._on_table_state_changed()

    def _cancel_pending_and_active_jobs(self) -> None:
        self._cancel_active_worker()

    def _cancel_active_worker(self) -> None:
        if self._active_worker is None:
            return

        quit_worker = getattr(self._active_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_worker = None
        self._active_worker_job_id = None

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._effective_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._effective_table_name)


def _normalize_feature_names(feature_names: Sequence[str] | str | None) -> tuple[str, ...]:
    if feature_names is None:
        return ()
    if isinstance(feature_names, str):
        names = [feature_names]
    else:
        names = list(feature_names)

    normalized: list[str] = []
    seen: set[str] = set()
    for feature_name in names:
        normalized_name = str(feature_name).strip()
        if not normalized_name or normalized_name in seen:
            continue
        normalized.append(normalized_name)
        seen.add(normalized_name)
    return tuple(normalized)


def _requires_image(feature_names: Sequence[str]) -> bool:
    return any(feature_name in _INTENSITY_FEATURES for feature_name in feature_names)
