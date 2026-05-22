from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from harpy.utils._keys import _FEATURE_MATRICES_KEY

from napari_harpy._app_state import FeatureMatrixWriteChangeKind, FeatureMatrixWrittenEvent
from napari_harpy.core.feature_extraction import (
    FeatureExtractionChannel,
    FeatureExtractionTriplet,
    _get_triplet_channel_selection_error,
    _normalize_channels,
    _normalize_triplets,
    _requires_image,
    _resolve_harpy_channel_parameter,
    _resolve_harpy_coordinate_system_parameter,
    _resolve_harpy_image_name_parameter,
    _resolve_harpy_labels_name_parameter,
)
from napari_harpy.core.spatialdata import get_table
from napari_harpy.core.validation import normalize_spatialdata_name

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


FEATURE_EXTRACTION_IDLE_STATUS = "Feature extraction: choose a labels element, table, and output key."

thread_worker = _resolve_thread_worker()


@dataclass(frozen=True)
class FeatureExtractionRequest:
    """A validated feature-extraction request covering one or more triplets.

    `table_name` is always the napari-harpy target table name. `create_table`
    decides whether that name refers to an existing table or to a table Harpy
    should create while writing the feature matrix.
    """

    triplets: tuple[FeatureExtractionTriplet, ...]
    table_name: str
    create_table: bool
    feature_names: tuple[str, ...]
    feature_key: str
    overwrite_feature_key: bool = False

    def __post_init__(self) -> None:
        if not self.table_name.strip():
            raise ValueError("Feature extraction request requires a table name.")
        if self.create_table and self.overwrite_feature_key:
            raise ValueError("Cannot overwrite a feature key while creating a new table.")

    @property
    def harpy_table_name(self) -> str | None:
        """Return Harpy's existing-table target parameter for this request."""
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        """Return Harpy's create-table output parameter for this request."""
        return self.table_name if self.create_table else None


@dataclass(frozen=True)
class FeatureExtractionBindingState:
    """Read-only snapshot of the controller's currently bound widget state.

    The Feature Extraction widget keeps its own staged UI state while the
    controller stores the state that would be used for the next calculation.
    This value object lets the widget compare those two views directly before it
    enables calculation. If the expected widget state differs from
    `controller.binding_state`, the UI treats the request as still
    synchronizing and keeps Calculate disabled.

    Unlike `FeatureExtractionRequest`, this binding state may represent an
    incomplete form. For example, `table_name` can be `None` while the user has
    not selected a valid annotation table or entered a new output table name
    yet. When present, `table_name` is the target name for both existing-table
    and create-table modes; `create_table` disambiguates which mode applies.
    """

    sdata: SpatialData | None
    triplets: tuple[FeatureExtractionTriplet, ...]
    table_name: str | None
    create_table: bool
    feature_names: tuple[str, ...]
    feature_key: str | None

    def __post_init__(self) -> None:
        if self.table_name is not None and not self.table_name.strip():
            raise ValueError("Feature extraction binding table name cannot be empty.")

    @property
    def harpy_table_name(self) -> str | None:
        """Return Harpy's existing-table target parameter for this binding."""
        if self.table_name is None:
            return None
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        """Return Harpy's create-table output parameter for this binding."""
        if self.table_name is None:
            return None
        return self.table_name if self.create_table else None


@dataclass(frozen=True)
class FeatureExtractionJob:
    """Immutable payload copied on the main thread and consumed by a worker."""

    job_id: int
    sdata: SpatialData
    request: FeatureExtractionRequest
    change_kind: FeatureMatrixWriteChangeKind = "created"

    @property
    def labels_names(self) -> tuple[str, ...]:
        return tuple(triplet.labels_name for triplet in self.request.triplets)

    @property
    def image_name(self) -> str | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].image_name

    @property
    def channels(self) -> tuple[FeatureExtractionChannel, ...] | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].channels

    @property
    def table_name(self) -> str:
        return self.request.table_name

    @property
    def coordinate_system(self) -> str | None:
        if len(self.request.triplets) != 1:
            return None
        return self.request.triplets[0].coordinate_system

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.request.feature_names

    @property
    def feature_key(self) -> str:
        return self.request.feature_key

    @property
    def overwrite_feature_key(self) -> bool:
        return self.request.overwrite_feature_key

    @property
    def triplet_count(self) -> int:
        return len(self.request.triplets)


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Summary produced by a completed feature-extraction worker."""

    job_id: int
    # Tuple because a single feature-extraction request can cover a batch of labels elements.
    labels_names: tuple[str, ...]
    table_name: str
    feature_key: str
    change_kind: FeatureMatrixWriteChangeKind = "created"

    @property
    def triplet_count(self) -> int:
        return len(self.labels_names)


@thread_worker(start_thread=False, ignore_errors=True)
def _run_feature_extraction_job(job: FeatureExtractionJob) -> FeatureExtractionResult:
    import harpy as hp

    triplets = job.request.triplets

    hp.tb.add_feature_matrix(
        sdata=job.sdata,
        labels_name=_resolve_harpy_labels_name_parameter(triplets),
        image_name=_resolve_harpy_image_name_parameter(triplets, job.request.feature_names),
        table_name=job.request.harpy_table_name,
        output_table_name=job.request.harpy_output_table_name,
        feature_key=job.request.feature_key,
        features=list(job.request.feature_names),
        channels=_resolve_harpy_channel_parameter(triplets, job.request.feature_names),
        feature_matrices_key=_FEATURE_MATRICES_KEY,
        # Table-name collisions are blocked in the UI/controller; replacing an
        # existing table would need a broader table and .obsm replacement policy.
        overwrite_output_table=False,
        overwrite_feature_key=job.request.overwrite_feature_key,
        to_coordinate_system=_resolve_harpy_coordinate_system_parameter(triplets),
    )

    return FeatureExtractionResult(
        job_id=job.job_id,
        labels_names=job.labels_names,
        table_name=job.request.table_name,
        feature_key=job.request.feature_key,
        change_kind=job.change_kind,
    )


class FeatureExtractionController:
    """Manage widget-facing state for background feature extraction."""

    def __init__(
        self,
        *,
        on_state_changed: Callable[[], None] | None = None,
        on_table_state_changed: Callable[[FeatureExtractionResult], None] | None = None,
        on_feature_matrix_written: Callable[[FeatureMatrixWrittenEvent], None] | None = None,
    ) -> None:
        self._on_state_changed = on_state_changed
        self._on_table_state_changed = on_table_state_changed
        self._on_feature_matrix_written = on_feature_matrix_written

        self._selected_spatialdata: SpatialData | None = None
        self._selected_triplets: tuple[FeatureExtractionTriplet, ...] = ()
        self._selected_table_name: str | None = None
        self._create_table = False
        self._selected_feature_names: tuple[str, ...] = ()
        self._selected_feature_key: str | None = None
        self._selected_labels_name_hint: str | None = None
        self._selected_coordinate_system_hint: str | None = None

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
    def binding_state(self) -> FeatureExtractionBindingState:
        """Return the current bound request snapshot for widget-side comparisons."""
        return FeatureExtractionBindingState(
            sdata=self._selected_spatialdata,
            triplets=self._selected_triplets,
            table_name=self._selected_table_name,
            create_table=self._create_table,
            feature_names=self._selected_feature_names,
            feature_key=self._selected_feature_key,
        )

    @property
    def can_calculate(self) -> bool:
        """Return whether the current selection has the minimum data to run."""
        return (
            self._get_table_target_validation_error() is None
            and bool(self._selected_triplets)
            and bool(self._selected_feature_names)
            and self._selected_feature_key is not None
            and bool(self._selected_feature_key.strip())
            and _get_feature_key_validation_error(self._selected_feature_key) is None
            and _get_triplet_channel_selection_error(self._selected_triplets, self._selected_feature_names) is None
            and (
                not _requires_image(self._selected_feature_names)
                or all(triplet.image_name is not None for triplet in self._selected_triplets)
            )
            and not self.is_running
        )

    def bind(
        self,
        sdata: SpatialData | None,
        labels_name: str | None,
        image_name: str | None,
        table_name: str | None,
        coordinate_system: str | None,
        feature_names: Sequence[str] | str | None,
        feature_key: str | None,
        *,
        channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None = None,
        create_table: bool = False,
    ) -> bool:
        """Bind the controller to one visible single-triplet selection."""
        normalized_coordinate_system = None if coordinate_system is None else coordinate_system.strip() or None
        normalized_feature_names = _normalize_feature_names(feature_names)
        normalized_feature_key = None if feature_key is None else feature_key.strip()
        normalized_channels = _normalize_channels(channels)
        triplets: tuple[FeatureExtractionTriplet, ...] = ()
        if labels_name is not None and normalized_coordinate_system is not None:
            triplets = (
                FeatureExtractionTriplet(
                    coordinate_system=normalized_coordinate_system,
                    labels_name=labels_name,
                    image_name=image_name,
                    channels=normalized_channels,
                ),
            )

        return self._bind_batch_state(
            sdata=sdata,
            triplets=triplets,
            table_name=table_name,
            create_table=create_table,
            feature_names=normalized_feature_names,
            feature_key=normalized_feature_key,
            labels_name_hint=labels_name,
            coordinate_system_hint=normalized_coordinate_system,
        )

    def bind_batch(
        self,
        sdata: SpatialData | None,
        triplets: Sequence[FeatureExtractionTriplet] | FeatureExtractionTriplet | None,
        table_name: str | None,
        feature_names: Sequence[str] | str | None,
        feature_key: str | None,
        *,
        create_table: bool = False,
    ) -> bool:
        """Bind the controller to one or more explicit extraction triplets."""
        normalized_triplets = _normalize_triplets(triplets)
        normalized_feature_names = _normalize_feature_names(feature_names)
        normalized_feature_key = None if feature_key is None else feature_key.strip()
        labels_name_hint = normalized_triplets[0].labels_name if len(normalized_triplets) == 1 else None
        coordinate_system_hint = normalized_triplets[0].coordinate_system if len(normalized_triplets) == 1 else None

        return self._bind_batch_state(
            sdata=sdata,
            triplets=normalized_triplets,
            table_name=table_name,
            create_table=create_table,
            feature_names=normalized_feature_names,
            feature_key=normalized_feature_key,
            labels_name_hint=labels_name_hint,
            coordinate_system_hint=coordinate_system_hint,
        )

    def _bind_batch_state(
        self,
        *,
        sdata: SpatialData | None,
        triplets: tuple[FeatureExtractionTriplet, ...],
        table_name: str | None,
        create_table: bool,
        feature_names: tuple[str, ...],
        feature_key: str | None,
        labels_name_hint: str | None,
        coordinate_system_hint: str | None,
    ) -> bool:
        normalized_table_name = _normalize_optional_table_name(table_name)
        normalized_create_table = bool(create_table)
        context_changed = (
            sdata is not self._selected_spatialdata
            or triplets != self._selected_triplets
            or normalized_table_name != self._selected_table_name
            or normalized_create_table is not self._create_table
            or feature_names != self._selected_feature_names
            or feature_key != self._selected_feature_key
            or labels_name_hint != self._selected_labels_name_hint
            or coordinate_system_hint != self._selected_coordinate_system_hint
        )

        self._selected_spatialdata = sdata
        self._selected_triplets = triplets
        self._selected_table_name = normalized_table_name
        self._create_table = normalized_create_table
        self._selected_feature_names = feature_names
        self._selected_feature_key = feature_key
        self._selected_labels_name_hint = labels_name_hint
        self._selected_coordinate_system_hint = coordinate_system_hint

        if context_changed:
            self._cancel_pending_and_active_jobs()

        self._update_idle_status()
        return context_changed

    def calculate(self, *, overwrite_feature_key: bool = False) -> bool:
        """Launch feature extraction for the current bound inputs."""
        if self.is_running:
            self._set_status("Feature extraction: calculation is already running.", kind="info")
            return False

        next_job_id = self._latest_requested_job_id + 1
        job = self._prepare_feature_extraction_job(
            next_job_id,
            overwrite_feature_key=bool(overwrite_feature_key),
        )
        if job is None:
            return False

        self._latest_requested_job_id = job.job_id
        worker = self._create_feature_extraction_worker(job)
        self._active_worker = worker
        self._active_worker_job_id = job.job_id
        worker.returned.connect(partial(self._on_worker_returned, job.job_id))
        worker.errored.connect(partial(self._on_worker_errored, job.job_id))
        worker.finished.connect(partial(self._on_worker_finished, job.job_id))
        if job.triplet_count == 1:
            labels_name = job.labels_names[0]
            self._set_status(
                f"Feature extraction: calculating `{job.feature_key}` for labels element `{labels_name}`.",
                kind="info",
            )
        else:
            self._set_status(
                f"Feature extraction: calculating `{job.feature_key}` for {job.triplet_count} extraction targets.",
                kind="info",
            )
        worker.start()
        return True

    def _prepare_feature_extraction_job(
        self,
        job_id: int,
        *,
        overwrite_feature_key: bool,
    ) -> FeatureExtractionJob | None:
        if self._selected_spatialdata is None:
            self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return None

        if not self._selected_triplets:
            if self._selected_coordinate_system_hint is None and self._selected_labels_name_hint is not None:
                self._set_status("Feature extraction: choose a coordinate system.", kind="warning")
            elif self._selected_coordinate_system_hint is not None and self._selected_labels_name_hint is None:
                self._set_status("Feature extraction: choose a labels element.", kind="warning")
            else:
                self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return None

        table_target_validation_error = self._get_table_target_validation_error()
        if table_target_validation_error is not None:
            self._set_status(table_target_validation_error, kind="warning")
            return None

        if not self._selected_feature_names:
            self._set_status("Feature extraction: choose at least one feature to calculate.", kind="warning")
            return None

        if self._selected_feature_key is None or not self._selected_feature_key.strip():
            self._set_status("Feature extraction: choose an output feature key.", kind="warning")
            return None

        feature_key_validation_error = _get_feature_key_validation_error(self._selected_feature_key)
        if feature_key_validation_error is not None:
            self._set_status(feature_key_validation_error, kind="warning")
            return None

        channel_selection_error = _get_triplet_channel_selection_error(
            self._selected_triplets, self._selected_feature_names
        )
        if channel_selection_error is not None:
            self._set_status(channel_selection_error, kind="warning")
            return None

        if _requires_image(self._selected_feature_names) and any(
            triplet.image_name is None for triplet in self._selected_triplets
        ):
            self._set_status(
                "Feature extraction: choose an image before calculating intensity features.",
                kind="warning",
            )
            return None

        if self._create_table and overwrite_feature_key:
            self._set_status(
                "Feature extraction: cannot overwrite a feature key while creating a new table.",
                kind="warning",
            )
            return None

        selected_table_name = self._selected_table_name
        if selected_table_name is None:
            self._set_status(
                (
                    "Feature extraction: enter a new table name."
                    if self._create_table
                    else "Feature extraction: choose an annotation table linked to the selected labels element."
                ),
                kind="warning",
            )
            return None

        request = FeatureExtractionRequest(
            triplets=self._selected_triplets,
            table_name=selected_table_name,
            create_table=self._create_table,
            feature_names=self._selected_feature_names,
            feature_key=self._selected_feature_key,
            overwrite_feature_key=overwrite_feature_key,
        )
        table = self._get_bound_table()
        return FeatureExtractionJob(
            job_id=job_id,
            sdata=self._selected_spatialdata,
            request=request,
            change_kind=(
                "updated"
                if table is not None and self._selected_feature_key in table.obsm
                else "created"
            ),
        )

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_state_changed(self, result: FeatureExtractionResult) -> None:
        # Keep this local widget refresh hook separate from the shared
        # `feature_matrix_written` app-state event. The owning widget uses the
        # concrete result to refresh table choices and, after create-table
        # writes, promote the new table into normal existing-table mode before
        # downstream widgets consume the shared semantic event.
        if self._on_table_state_changed is not None:
            self._on_table_state_changed(result)

    def _notify_feature_matrix_written(self, result: FeatureExtractionResult) -> None:
        if self._on_feature_matrix_written is None or self._selected_spatialdata is None:
            return

        self._on_feature_matrix_written(
            FeatureMatrixWrittenEvent(
                sdata=self._selected_spatialdata,
                table_name=result.table_name,
                feature_key=result.feature_key,
                change_kind=result.change_kind,
            )
        )

    def _update_idle_status(self) -> None:
        if self._selected_spatialdata is None:
            self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return

        if not self._selected_triplets and self._selected_coordinate_system_hint is not None:
            self._set_status("Feature extraction: choose a labels element.", kind="warning")
            return

        if not self._selected_triplets and self._selected_coordinate_system_hint is None:
            if self._selected_labels_name_hint is not None:
                self._set_status("Feature extraction: choose a coordinate system.", kind="warning")
            else:
                self._set_status(FEATURE_EXTRACTION_IDLE_STATUS, kind="warning")
            return

        table_target_validation_error = self._get_table_target_validation_error()
        if table_target_validation_error is not None:
            self._set_status(table_target_validation_error, kind="warning")
            return

        if not self._selected_feature_names:
            self._set_status("Feature extraction: choose at least one feature to calculate.", kind="warning")
            return

        if self._selected_feature_key is None or not self._selected_feature_key.strip():
            self._set_status("Feature extraction: choose an output feature key.", kind="warning")
            return

        feature_key_validation_error = _get_feature_key_validation_error(self._selected_feature_key)
        if feature_key_validation_error is not None:
            self._set_status(feature_key_validation_error, kind="warning")
            return

        channel_selection_error = _get_triplet_channel_selection_error(
            self._selected_triplets, self._selected_feature_names
        )
        if channel_selection_error is not None:
            self._set_status(channel_selection_error, kind="warning")
            return

        if _requires_image(self._selected_feature_names) and any(
            triplet.image_name is None for triplet in self._selected_triplets
        ):
            self._set_status(
                "Feature extraction: choose an image before calculating intensity features.",
                kind="warning",
            )
            return

        if self.is_running:
            return

        self._set_status("Feature extraction: ready to calculate.", kind="success")

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

    def _create_feature_extraction_worker(self, job: FeatureExtractionJob) -> Any:
        return _run_feature_extraction_job(job)

    def _on_worker_returned(self, job_id: int, result: FeatureExtractionResult) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        self._notify_table_state_changed(result)
        self._notify_feature_matrix_written(result)
        self._set_status(
            "Feature extraction: "
            f"wrote `{result.feature_key}` into table `{result.table_name}` as "
            f"`.obsm[{result.feature_key!r}]` with metadata in "
            f"`.uns[{_FEATURE_MATRICES_KEY!r}][{result.feature_key!r}]`.",
            kind="success",
        )

    def _on_worker_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        self._set_status(f"Feature extraction: calculation failed: {error}", kind="error")

    def _on_worker_finished(self, job_id: int) -> None:
        if job_id != self._active_worker_job_id:
            return

        self._active_worker = None
        self._active_worker_job_id = None
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None or self._create_table:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _get_table_target_validation_error(self) -> str | None:
        if self._selected_spatialdata is None:
            return None

        if self._selected_table_name is None:
            if self._create_table:
                return "Feature extraction: enter a new table name."
            return "Feature extraction: choose an annotation table linked to the selected labels element."

        if not self._create_table:
            self._get_bound_table()
            return None

        try:
            normalize_spatialdata_name(self._selected_table_name, "table name")
        except ValueError as error:
            return f"Feature extraction: choose a valid table name. {error}"

        if self._selected_table_name in self._selected_spatialdata.tables:
            return (
                f"Feature extraction: table `{self._selected_table_name}` already exists. "
                "Choose a different table name."
            )

        return None


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


def _normalize_optional_table_name(table_name: str | None) -> str | None:
    if table_name is None:
        return None

    normalized_table_name = str(table_name).strip()
    return normalized_table_name or None


def _get_feature_key_validation_error(feature_key: str | None) -> str | None:
    if feature_key is None or not feature_key.strip():
        return "Feature extraction: choose an output feature key."

    try:
        normalize_spatialdata_name(feature_key, "feature matrix key")
    except ValueError as error:
        return f"Feature extraction: choose a valid feature matrix key. {error}"
    return None
