from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qtpy.QtCore import QTimer
from sklearn.ensemble import RandomForestClassifier

from napari_harpy._annotation import UNLABELED_CLASS, USER_CLASS_COLUMN
from napari_harpy._class_palette import set_class_annotation_state
from napari_harpy._spatialdata import SpatialDataTableMetadata, get_table, get_table_metadata

try:
    from scipy.sparse import issparse
except ImportError:  # pragma: no cover - scipy is expected in the plugin env

    def issparse(value: object) -> bool:
        return False


def _resolve_thread_worker() -> Any:
    try:
        from napari.qt.threading import thread_worker
    except Exception:  # pragma: no cover - fallback for sandboxed test imports  # noqa: BLE001
        from superqt.utils import thread_worker

    return thread_worker


if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

thread_worker = _resolve_thread_worker()

BoolArray = NDArray[np.bool_]
TableRowPositions = NDArray[np.int64]

PRED_CLASS_COLUMN = "pred_class"
PRED_CLASS_COLORS_KEY = f"{PRED_CLASS_COLUMN}_colors"
PRED_CONFIDENCE_COLUMN = "pred_confidence"
CLASSIFIER_CONFIG_KEY = "classifier_config"

DEFAULT_RETRAIN_DEBOUNCE_MS = 300
MIN_LABELED_SAMPLES = 2
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 0,
    "n_jobs": 1,
}

ClassifierScopeMode = Literal["selected_segmentation_only", "all"]
DEFAULT_TRAINING_SCOPE: ClassifierScopeMode = "all"
DEFAULT_PREDICTION_SCOPE: ClassifierScopeMode = "selected_segmentation_only"


@dataclass(frozen=True)
class ResolvedClassifierScope:
    """Resolved classifier scope for one training or prediction selection.

    ``regions`` is the resolved semantic scope requested by the user, expressed
    as table region names. ``n_rows_in_regions`` is the raw number of table rows
    whose region key belongs to those regions, before feature-validity filtering.
    ``table_row_positions`` contains the original table row positions that are
    both in those regions and usable for the selected feature matrix.
    """

    mode: ClassifierScopeMode
    regions: tuple[str, ...]
    table_row_positions: TableRowPositions
    n_rows_in_regions: int

    @property
    def n_eligible_rows(self) -> int:
        return int(self.table_row_positions.size)

    @property
    def n_excluded_feature_invalid_rows(self) -> int:
        return self.n_rows_in_regions - int(self.table_row_positions.size)


@dataclass(frozen=True)
class ResolvedClassifierScopes:
    """Resolved training and prediction scopes for the current controller state."""

    training: ResolvedClassifierScope
    prediction: ResolvedClassifierScope


@dataclass(frozen=True)
class ClassifierPreparationSummary:
    """Describe the current classifier preparation state without worker arrays."""

    training_scope: ResolvedClassifierScope
    prediction_scope: ResolvedClassifierScope
    eligible: bool
    reason: str
    labeled_count: int
    class_labels: tuple[int, ...]
    n_features: int | None

    @property
    def resolved_training_row_count(self) -> int:
        return self.training_scope.n_eligible_rows

    @property
    def resolved_prediction_row_count(self) -> int:
        return self.prediction_scope.n_eligible_rows

    @property
    def training_region_count(self) -> int:
        return len(self.training_scope.regions)


@dataclass(frozen=True)
class ClassifierJob:
    """Immutable payload copied on the main thread and consumed by a worker."""

    job_id: int
    feature_key: str
    label_name: str
    table_name: str
    predict_features: Any
    train_features: Any
    train_labels: np.ndarray
    summary: ClassifierPreparationSummary

    @property
    def training_scope(self) -> ResolvedClassifierScope:
        return self.summary.training_scope

    @property
    def prediction_scope(self) -> ResolvedClassifierScope:
        return self.summary.prediction_scope


@dataclass(frozen=True)
class ClassifierJobResult:
    """Predictions produced by a completed training worker."""

    job_id: int
    feature_key: str
    label_name: str
    table_name: str
    pred_classes: np.ndarray
    pred_confidences: np.ndarray
    trained_at: str
    model_params: dict[str, int]
    summary: ClassifierPreparationSummary

    @property
    def training_scope(self) -> ResolvedClassifierScope:
        return self.summary.training_scope

    @property
    def prediction_scope(self) -> ResolvedClassifierScope:
        return self.summary.prediction_scope


@thread_worker(start_thread=False, ignore_errors=True)
def _run_classifier_job(job: ClassifierJob) -> ClassifierJobResult:
    return _fit_classifier_job(job)


# Keep the synchronous fit/predict logic separate from the `@thread_worker`
# wrapper above: `_run_classifier_job(...)` returns a worker object, while this
# function directly computes and returns the eventual `ClassifierJobResult`.
def _fit_classifier_job(job: ClassifierJob) -> ClassifierJobResult:
    classifier = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    classifier.fit(job.train_features, job.train_labels)
    pred_classes = np.asarray(classifier.predict(job.predict_features), dtype=np.int64)
    pred_proba = np.asarray(classifier.predict_proba(job.predict_features), dtype=np.float64)
    pred_confidences = pred_proba.max(axis=1)

    return ClassifierJobResult(
        job_id=job.job_id,
        feature_key=job.feature_key,
        label_name=job.label_name,
        table_name=job.table_name,
        pred_classes=pred_classes,
        pred_confidences=pred_confidences,
        trained_at=datetime.now(UTC).isoformat(),
        model_params={key: int(value) for key, value in RANDOM_FOREST_PARAMS.items()},
        summary=job.summary,
    )


class ClassifierController:
    """Manage debounced background classifier retraining for the active table."""

    def __init__(
        self,
        *,
        debounce_interval_ms: int = DEFAULT_RETRAIN_DEBOUNCE_MS,
        on_state_changed: Callable[[], None] | None = None,
        on_table_state_changed: Callable[[], None] | None = None,
    ) -> None:
        self._debounce_interval_ms = max(0, int(debounce_interval_ms))
        self._on_state_changed = on_state_changed
        self._on_table_state_changed = on_table_state_changed

        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_feature_key: str | None = None
        self._selected_training_scope: ClassifierScopeMode = DEFAULT_TRAINING_SCOPE
        self._selected_prediction_scope: ClassifierScopeMode = DEFAULT_PREDICTION_SCOPE
        # TODO: clean up. There is overlap between the above class attributes and spatialdatatablemetadata.
        self._selected_table_metadata: SpatialDataTableMetadata | None = None

        self._latest_requested_job_id = 0
        self._active_worker_job_id: int | None = None
        self._active_worker: Any | None = None
        self._active_job: ClassifierJob | None = None
        self._is_dirty = False

        self._status_message = "Classifier: choose an annotation table and feature matrix."
        self._status_kind = "warning"

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(self._debounce_interval_ms)
        self._debounce_timer.timeout.connect(self._launch_scheduled_retrain)

    @property
    def status_message(self) -> str:
        """Return the latest user-facing classifier status message."""
        return self._status_message

    @property
    def status_kind(self) -> str:
        """Return the latest status level: info, warning, success, or error."""
        return self._status_kind

    @property
    def is_training(self) -> bool:
        """Return whether a classifier worker is currently active."""
        return self._active_worker is not None

    @property
    def is_dirty(self) -> bool:
        """Return whether the current classifier outputs are stale."""
        return self._is_dirty

    @property
    def can_retrain(self) -> bool:
        """Return whether the current classifier inputs support a retrain request."""
        return self._get_bound_table() is not None and self._selected_feature_key is not None and not self.is_training

    def bind(
        self,
        sdata: SpatialData | None,
        label_name: str | None,
        table_name: str | None,
        feature_key: str | None,
        training_scope: ClassifierScopeMode = DEFAULT_TRAINING_SCOPE,
        prediction_scope: ClassifierScopeMode = DEFAULT_PREDICTION_SCOPE,
    ) -> bool:
        """Bind the controller to the currently selected SpatialData inputs."""
        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = get_table_metadata(sdata, table_name)

        normalized_training_scope = _normalize_scope_mode(training_scope)
        normalized_prediction_scope = _normalize_scope_mode(prediction_scope)
        context_changed = (
            sdata is not self._selected_spatialdata
            or label_name != self._selected_label_name
            or table_name != self._selected_table_name
            or feature_key != self._selected_feature_key
            or normalized_training_scope != self._selected_training_scope
            or normalized_prediction_scope != self._selected_prediction_scope
        )

        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_table_name = table_name
        self._selected_feature_key = feature_key
        self._selected_training_scope = normalized_training_scope
        self._selected_prediction_scope = normalized_prediction_scope
        self._selected_table_metadata = next_table_metadata

        if context_changed:
            self._cancel_pending_and_active_jobs()

        table = self._get_bound_table()
        if table is None:
            self._is_dirty = False
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return context_changed

        self._ensure_prediction_columns(table)

        self._update_idle_status()
        return context_changed

    def mark_dirty(self, reason: str | None = None) -> None:
        """Mark the current classifier outputs as stale after an input change."""
        if self._get_bound_table() is None:
            self._is_dirty = False
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        self._is_dirty = True
        self._update_idle_status(reason=reason)

    def retrain_now(self) -> bool:
        """Immediately retrain the classifier for the current inputs."""
        return self.schedule_retrain(immediate=True)

    def schedule_retrain(self, *, immediate: bool = False) -> bool:
        """Schedule a background retraining job for the current classifier inputs."""
        if self._get_bound_table() is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return False
        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix before training.", kind="warning")
            return False

        self._latest_requested_job_id += 1
        self._debounce_timer.stop()
        self._cancel_active_worker()
        self._is_dirty = True

        if immediate or self._debounce_interval_ms == 0:
            self._launch_retrain_job(self._latest_requested_job_id)
        else:
            self._debounce_timer.start()
            self._set_status("Classifier: model is stale. Classifier training is scheduled.", kind="info")

        return True

    def freeze_for_reload(self) -> None:
        """Cancel pending async work so reload cannot apply stale classifier results."""
        self._invalidate_async_jobs()
        self._update_idle_status()

    def invalidate_for_feature_matrix_overwrite(self, feature_key: str) -> bool:
        """Invalidate pending work when the selected feature matrix was overwritten in place."""
        if self._get_bound_table() is None or self._selected_feature_key != feature_key:
            return False

        self._invalidate_async_jobs()
        self._is_dirty = True
        self._update_idle_status(reason=f"feature matrix `{feature_key}` was overwritten")
        return True

    def reset_after_reload(self) -> None:
        """Recompute classifier state from the reloaded table without retraining."""
        self._invalidate_async_jobs()
        table = self._get_bound_table()
        if table is None:
            self._is_dirty = False
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        self._ensure_prediction_columns(table)
        self._is_dirty = False
        self._update_status_from_reloaded_table(table)

    def describe_current_preparation(self) -> ClassifierPreparationSummary | None:
        """Return a side-effect-free summary of the currently bound classifier state."""
        return self._prepare_classifier_summary()

    def _launch_scheduled_retrain(self) -> None:
        self._launch_retrain_job(self._latest_requested_job_id)

    def _launch_retrain_job(self, job_id: int) -> None:
        if job_id != self._latest_requested_job_id:
            return

        job = self._prepare_classifier_job(job_id)
        if job is None:
            return

        summary = job.summary
        if not summary.eligible:
            self._apply_ineligible_state(job)
            return

        worker = self._create_training_worker(job)
        self._active_worker = worker
        self._active_job = job
        self._active_worker_job_id = job_id
        worker.returned.connect(partial(self._on_worker_returned, job_id))
        worker.errored.connect(partial(self._on_worker_errored, job_id))
        worker.finished.connect(partial(self._on_worker_finished, job_id))
        self._set_status(
            (
                f"Classifier: training RandomForest on {summary.labeled_count} labeled objects "
                f"across {len(summary.class_labels)} classes from {summary.resolved_training_row_count} "
                f"eligible rows in {summary.training_region_count} region(s). "
                f"Prediction scope contains {summary.resolved_prediction_row_count} row(s)."
            ),
            kind="info",
        )
        worker.start()

    def _prepare_classifier_summary(self) -> ClassifierPreparationSummary | None:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None or self._selected_table_name is None:
            return None

        # Early error paths still need region/count context for status and
        # summary reporting, even before feature-valid row positions can be
        # resolved from a concrete feature matrix.
        pre_feature_scopes = self._resolve_classifier_scopes(table, metadata, feature_valid_row_mask=None)
        if self._selected_feature_key is None:
            return ClassifierPreparationSummary(
                training_scope=pre_feature_scopes.training,
                prediction_scope=pre_feature_scopes.prediction,
                eligible=False,
                reason="Choose a feature matrix before training the classifier.",
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )

        try:
            feature_matrix = _normalize_feature_matrix(table.obsm[self._selected_feature_key], table.n_obs, copy=False)
        except KeyError:
            return ClassifierPreparationSummary(
                training_scope=pre_feature_scopes.training,
                prediction_scope=pre_feature_scopes.prediction,
                eligible=False,
                reason=f"Feature matrix `{self._selected_feature_key}` is not available in `.obsm`.",
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )
        except ValueError as error:
            return ClassifierPreparationSummary(
                training_scope=pre_feature_scopes.training,
                prediction_scope=pre_feature_scopes.prediction,
                eligible=False,
                reason=str(error),
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )

        feature_valid_row_mask = _get_finite_feature_row_mask(feature_matrix)
        scopes = self._resolve_classifier_scopes(table, metadata, feature_valid_row_mask=feature_valid_row_mask)
        prediction_scope = scopes.prediction
        training_scope = scopes.training
        n_features = int(feature_matrix.shape[1])

        if prediction_scope.n_rows_in_regions == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                eligible=False,
                reason=(
                    f"No table rows for segmentation `{self._selected_label_name}` were found in "
                    f"`{self._selected_table_name}`."
                ),
                labeled_count=0,
                class_labels=(),
                n_features=n_features,
            )

        if prediction_scope.n_eligible_rows == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                eligible=False,
                reason=(
                    f"Feature matrix `{self._selected_feature_key}` has no usable rows in the prediction scope: "
                    f"all {prediction_scope.n_rows_in_regions} row(s) have non-finite or missing values."
                ),
                labeled_count=0,
                class_labels=(),
                n_features=n_features,
            )

        if n_features == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                eligible=False,
                reason=f"Feature matrix `{self._selected_feature_key}` does not contain any columns.",
                labeled_count=0,
                class_labels=(),
                n_features=0,
            )

        if training_scope.n_rows_in_regions > 0 and training_scope.n_eligible_rows == 0:
            return ClassifierPreparationSummary(
                training_scope=training_scope,
                prediction_scope=prediction_scope,
                eligible=False,
                reason=(
                    f"Feature matrix `{self._selected_feature_key}` has no usable rows in the training scope: "
                    f"all {training_scope.n_rows_in_regions} row(s) have non-finite or missing values."
                ),
                labeled_count=0,
                class_labels=(),
                n_features=n_features,
            )

        user_class_values = _get_user_class_values(table.obs, len(table.obs))
        training_user_class_values = user_class_values[training_scope.table_row_positions]
        labeled_mask = training_user_class_values != UNLABELED_CLASS
        class_labels = tuple(sorted(int(value) for value in np.unique(training_user_class_values[labeled_mask])))
        summary = ClassifierPreparationSummary(
            training_scope=training_scope,
            prediction_scope=prediction_scope,
            eligible=True,
            reason="Ready to train.",
            labeled_count=int(labeled_mask.sum()),
            class_labels=class_labels,
            n_features=n_features,
        )

        if summary.labeled_count < MIN_LABELED_SAMPLES:
            summary = ClassifierPreparationSummary(
                eligible=False,
                reason=(
                    f"Need at least {MIN_LABELED_SAMPLES} labeled samples before training the classifier. "
                    f"Resolved {summary.resolved_training_row_count} eligible training row(s) across "
                    f"{summary.training_region_count} region(s); {summary.labeled_count} row(s) are labeled. "
                    f"Prediction scope contains {summary.resolved_prediction_row_count} row(s)."
                ),
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=summary.n_features,
            )
        elif len(summary.class_labels) < 2:
            summary = ClassifierPreparationSummary(
                eligible=False,
                reason=(
                    "Need at least two labeled classes before training the classifier. "
                    f"Resolved {summary.resolved_training_row_count} eligible training row(s) across "
                    f"{summary.training_region_count} region(s); {summary.labeled_count} row(s) are labeled "
                    f"across {len(summary.class_labels)} class(es). "
                    f"Prediction scope contains {summary.resolved_prediction_row_count} row(s)."
                ),
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=summary.n_features,
            )

        return summary

    def _prepare_classifier_job(self, job_id: int) -> ClassifierJob | None:
        table = self._get_bound_table()
        if table is None or self._selected_label_name is None or self._selected_table_name is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return None

        summary = self._prepare_classifier_summary()
        if summary is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return None

        feature_key = "" if self._selected_feature_key is None else self._selected_feature_key
        n_features = 0 if summary.n_features is None else int(summary.n_features)
        empty_features = np.empty((0, n_features), dtype=np.float64)
        if not summary.eligible or self._selected_feature_key is None:
            return ClassifierJob(
                job_id=job_id,
                feature_key=feature_key,
                label_name=self._selected_label_name,
                table_name=self._selected_table_name,
                predict_features=empty_features,
                train_features=empty_features,
                train_labels=np.array([], dtype=np.int64),
                summary=summary,
            )

        try:
            feature_matrix = _normalize_feature_matrix(table.obsm[self._selected_feature_key], table.n_obs)
        except KeyError:
            summary = ClassifierPreparationSummary(
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                eligible=False,
                reason=f"Feature matrix `{self._selected_feature_key}` is not available in `.obsm`.",
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=None,
            )
            return ClassifierJob(
                job_id=job_id,
                feature_key=feature_key,
                label_name=self._selected_label_name,
                table_name=self._selected_table_name,
                predict_features=np.empty((0, 0), dtype=np.float64),
                train_features=np.empty((0, 0), dtype=np.float64),
                train_labels=np.array([], dtype=np.int64),
                summary=summary,
            )
        except ValueError as error:
            summary = ClassifierPreparationSummary(
                training_scope=summary.training_scope,
                prediction_scope=summary.prediction_scope,
                eligible=False,
                reason=str(error),
                labeled_count=summary.labeled_count,
                class_labels=summary.class_labels,
                n_features=None,
            )
            return ClassifierJob(
                job_id=job_id,
                feature_key=feature_key,
                label_name=self._selected_label_name,
                table_name=self._selected_table_name,
                predict_features=np.empty((0, 0), dtype=np.float64),
                train_features=np.empty((0, 0), dtype=np.float64),
                train_labels=np.array([], dtype=np.int64),
                summary=summary,
            )

        predict_features = _slice_feature_rows(feature_matrix, summary.prediction_scope.table_row_positions)
        user_class_values = _get_user_class_values(table.obs, len(table.obs))
        training_user_class_values = user_class_values[summary.training_scope.table_row_positions]
        labeled_mask = training_user_class_values != UNLABELED_CLASS
        labeled_training_positions = summary.training_scope.table_row_positions[labeled_mask]
        train_features = _slice_feature_rows(feature_matrix, labeled_training_positions)
        train_labels = np.asarray(training_user_class_values[labeled_mask], dtype=np.int64)
        return ClassifierJob(
            job_id=job_id,
            feature_key=feature_key,
            label_name=self._selected_label_name,
            table_name=self._selected_table_name,
            predict_features=predict_features,
            train_features=train_features,
            train_labels=train_labels,
            summary=summary,
        )

    def _apply_ineligible_state(
        self,
        job: ClassifierJob,
    ) -> None:
        table = self._get_bound_table()
        if table is None:
            self._set_status(f"Classifier: {job.summary.reason}", kind="warning")
            return

        self._ensure_prediction_columns(table)
        self._clear_predictions_for_prediction_regions(table, job.prediction_scope.regions)

        table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
            feature_key=job.feature_key,
            table_name=job.table_name,
            summary=job.summary,
            trained=False,
            trained_at=None,
        )
        self._notify_table_state_changed()
        self._is_dirty = True
        self._set_status(f"Classifier: {job.summary.reason}", kind="warning")

    def _on_worker_returned(self, job_id: int, result: ClassifierJobResult) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        table = self._get_bound_table()
        if table is None:
            return

        self._ensure_prediction_columns(table)
        self._clear_predictions_for_prediction_regions(table, result.prediction_scope.regions)
        self._set_predictions_for_prediction_rows(
            table,
            result.prediction_scope.table_row_positions,
            result.pred_classes,
            result.pred_confidences,
        )
        table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
            feature_key=result.feature_key,
            table_name=result.table_name,
            summary=result.summary,
            trained=True,
            trained_at=result.trained_at,
        )
        self._notify_table_state_changed()
        self._is_dirty = False
        self._set_status(
            f"Classifier: model is up to date. Updated predictions for {result.summary.resolved_prediction_row_count} objects.",
            kind="success",
        )

    def _on_worker_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        table = self._get_bound_table()
        if table is not None:
            active_job = self._active_job
            if active_job is None:
                error_summary = ClassifierPreparationSummary(
                    training_scope=_empty_resolved_classifier_scope(self._selected_training_scope),
                    prediction_scope=_empty_resolved_classifier_scope(self._selected_prediction_scope),
                    eligible=False,
                    reason=str(error),
                    labeled_count=0,
                    class_labels=(),
                    n_features=None,
                )
            else:
                error_summary = replace(active_job.summary, eligible=False, reason=str(error))
            table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
                feature_key=None if active_job is None else active_job.feature_key,
                table_name=None if active_job is None else active_job.table_name,
                summary=error_summary,
                trained=False,
                trained_at=None,
            )
            self._notify_table_state_changed()
        self._is_dirty = True
        self._set_status(f"Classifier: training failed: {error}", kind="error")

    def _on_worker_finished(self, job_id: int) -> None:
        if job_id != self._active_worker_job_id:
            return

        self._active_worker = None
        self._active_job = None
        self._active_worker_job_id = None
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _cancel_pending_and_active_jobs(self) -> None:
        self._debounce_timer.stop()
        self._cancel_active_worker()

    def _invalidate_async_jobs(self) -> None:
        self._latest_requested_job_id += 1
        self._cancel_pending_and_active_jobs()

    def _cancel_active_worker(self) -> None:
        if self._active_worker is None:
            return

        quit_worker = getattr(self._active_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_worker = None
        self._active_job = None
        self._active_worker_job_id = None

    def _create_training_worker(self, job: ClassifierJob) -> Any:
        return _run_classifier_job(job)

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return get_table(self._selected_spatialdata, self._selected_table_name)

    def _ensure_prediction_columns(self, table: AnnData) -> None:
        pred_class_values = _get_pred_class_values(table.obs, len(table.obs))
        pred_confidence_values = _get_pred_confidence_values(table.obs, len(table.obs))
        _set_pred_class_annotation_state(table, pred_class_values)
        table.obs[PRED_CONFIDENCE_COLUMN] = pred_confidence_values

    def _set_predictions_for_prediction_rows(
        self,
        table: AnnData,
        prediction_table_row_positions: TableRowPositions,
        pred_classes: np.ndarray,
        pred_confidences: np.ndarray,
    ) -> None:
        pred_class_values = _get_pred_class_values(table.obs, len(table.obs))
        pred_confidence_values = _get_pred_confidence_values(table.obs, len(table.obs))
        pred_class_values.iloc[prediction_table_row_positions] = np.asarray(pred_classes, dtype=np.int64)
        pred_confidence_values.iloc[prediction_table_row_positions] = np.asarray(pred_confidences, dtype=np.float64)
        _set_pred_class_annotation_state(table, pred_class_values)
        table.obs[PRED_CONFIDENCE_COLUMN] = pred_confidence_values

    def _clear_predictions_for_prediction_regions(
        self,
        table: AnnData,
        prediction_regions: tuple[str, ...],
    ) -> None:
        if self._selected_table_metadata is None:
            return

        prediction_table_row_positions = _resolve_region_row_positions(
            table.obs,
            self._selected_table_metadata.region_key,
            prediction_regions,
        )
        self._set_predictions_for_prediction_rows(
            table,
            prediction_table_row_positions,
            np.full(prediction_table_row_positions.shape, UNLABELED_CLASS, dtype=np.int64),
            np.full(prediction_table_row_positions.shape, np.nan, dtype=np.float64),
        )

    def _build_classifier_config(
        self,
        *,
        feature_key: str | None,
        table_name: str | None,
        summary: ClassifierPreparationSummary,
        trained: bool,
        trained_at: str | None,
    ) -> dict[str, object]:
        return {
            "model_type": "RandomForestClassifier",
            "feature_key": feature_key,
            "table_name": table_name,
            "roi_mode": "none",
            "trained": trained,
            "eligible": summary.eligible,
            "reason": summary.reason,
            "training_timestamp": trained_at,
            "n_labeled_objects": summary.labeled_count,
            "n_features": summary.n_features,
            "class_labels_seen": list(summary.class_labels),
            "rf_params": dict(RANDOM_FOREST_PARAMS),
            "training_scope": summary.training_scope.mode,
            "training_regions": list(summary.training_scope.regions),
            "n_training_rows": summary.resolved_training_row_count,
            "prediction_scope": summary.prediction_scope.mode,
            "prediction_regions": list(summary.prediction_scope.regions),
            "n_predicted_rows": summary.resolved_prediction_row_count,
        }

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

    def _notify_table_state_changed(self) -> None:
        if self._on_table_state_changed is not None:
            self._on_table_state_changed()

    def _update_idle_status(self, *, reason: str | None = None) -> None:
        if self._get_bound_table() is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return

        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix to enable training.", kind="warning")
            return

        if self.is_training:
            return

        if self._is_dirty:
            if reason is None:
                self._set_status(
                    "Classifier: model is stale. Click Train Classifier to refresh predictions.",
                    kind="warning",
                )
            else:
                self._set_status(f"Classifier: model is stale because {reason}", kind="warning")
            return

        self._set_status("Classifier: model is up to date.", kind="success")

    def _update_status_from_reloaded_table(self, table: AnnData) -> None:
        if self._selected_feature_key is None:
            self._set_status("Classifier: choose a feature matrix to enable training.", kind="warning")
            return

        config = table.uns.get(CLASSIFIER_CONFIG_KEY)
        if not isinstance(config, Mapping):
            self._update_idle_status()
            return

        mismatches: list[str] = []
        config_feature_key = config.get("feature_key")
        if isinstance(config_feature_key, str) and config_feature_key != self._selected_feature_key:
            mismatches.append(
                f"loaded predictions use feature matrix `{config_feature_key}` but `{self._selected_feature_key}` is selected"
            )

        config_table_name = config.get("table_name")
        if (
            isinstance(config_table_name, str)
            and self._selected_table_name is not None
            and config_table_name != self._selected_table_name
        ):
            mismatches.append(
                f"loaded predictions target table `{config_table_name}` but `{self._selected_table_name}` is selected"
            )

        config_prediction_scope = config.get("prediction_scope")
        if not isinstance(config_prediction_scope, str):
            mismatches.append("loaded predictions do not describe their prediction scope")
        elif config_prediction_scope != self._selected_prediction_scope:
            mismatches.append(
                f"loaded predictions use prediction scope `{config_prediction_scope}` but `{self._selected_prediction_scope}` is selected"
            )

        config_prediction_regions = _normalize_prediction_regions(config.get("prediction_regions"))
        if self._selected_label_name is not None and self._selected_label_name not in config_prediction_regions:
            mismatches.append(f"loaded predictions do not cover segmentation `{self._selected_label_name}`")

        if mismatches:
            self._is_dirty = True
            self._set_status(f"Classifier: model is stale because {'; '.join(mismatches)}.", kind="warning")
            return

        trained = config.get("trained")
        if trained is True:
            predicted_count = config.get("n_predicted_rows")
            if isinstance(predicted_count, int):
                self._set_status(
                    f"Classifier: model is up to date. Loaded predictions for {predicted_count} objects from disk.",
                    kind="success",
                )
            else:
                self._set_status("Classifier: model is up to date. Loaded predictions from disk.", kind="success")
            return

        eligible = config.get("eligible")
        reason = config.get("reason")
        if eligible is False and isinstance(reason, str) and reason:
            self._set_status(f"Classifier: {reason}", kind="warning")
            return
        if trained is False and isinstance(reason, str) and reason:
            self._set_status(f"Classifier: {reason}", kind="warning")
            return

        self._update_idle_status()

    def _resolve_classifier_scopes(
        self,
        table: AnnData,
        metadata: SpatialDataTableMetadata,
        *,
        feature_valid_row_mask: BoolArray | None,
    ) -> ResolvedClassifierScopes:
        def resolve_one_scope(scope_mode: ClassifierScopeMode) -> ResolvedClassifierScope:
            if scope_mode == "selected_segmentation_only":
                regions: tuple[str, ...] = () if self._selected_label_name is None else (self._selected_label_name,)
            elif scope_mode == "all":
                regions = metadata.regions
            else:  # pragma: no cover - guarded by _normalize_scope_mode
                raise ValueError(f"Unsupported classifier scope mode: {scope_mode!r}")

            raw_table_row_positions = _resolve_region_row_positions(
                table.obs,
                metadata.region_key,
                regions,
            )
            if feature_valid_row_mask is None or raw_table_row_positions.size == 0:
                table_row_positions = np.array([], dtype=np.int64)
            else:
                valid_in_scope_mask = feature_valid_row_mask[raw_table_row_positions]
                table_row_positions = np.asarray(raw_table_row_positions[valid_in_scope_mask], dtype=np.int64)

            return ResolvedClassifierScope(
                mode=scope_mode,
                regions=regions,
                table_row_positions=table_row_positions,
                n_rows_in_regions=int(raw_table_row_positions.size),
            )

        return ResolvedClassifierScopes(
            training=resolve_one_scope(self._selected_training_scope),
            prediction=resolve_one_scope(self._selected_prediction_scope),
        )


def _normalize_feature_matrix(feature_matrix: Any, n_obs: int, *, copy: bool = True) -> Any:
    if issparse(feature_matrix):
        if feature_matrix.ndim != 2:
            raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
        if feature_matrix.shape[0] != n_obs:
            raise ValueError(
                f"Feature matrix has {feature_matrix.shape[0]} rows but the table has {n_obs} observations."
            )
        return feature_matrix.copy() if copy else feature_matrix

    array = np.asarray(feature_matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
    if array.shape[0] != n_obs:
        raise ValueError(f"Feature matrix has {array.shape[0]} rows but the table has {n_obs} observations.")
    return array.copy() if copy else array


def _slice_feature_rows(feature_matrix: Any, positions: TableRowPositions) -> Any:
    return feature_matrix[positions]


def _get_finite_feature_row_mask(feature_matrix: Any) -> BoolArray:
    if issparse(feature_matrix):
        finite_data_mask = np.isfinite(feature_matrix.data)
        if bool(finite_data_mask.all()):
            return np.ones(feature_matrix.shape[0], dtype=bool)

        invalid_rows = np.unique(feature_matrix.tocoo().row[~finite_data_mask])
        valid_row_mask = np.ones(feature_matrix.shape[0], dtype=bool)
        valid_row_mask[invalid_rows] = False
        return valid_row_mask

    finite_feature_mask = np.isfinite(np.asarray(feature_matrix, dtype=np.float64))
    return np.asarray(finite_feature_mask.all(axis=1), dtype=bool)


def _resolve_region_row_positions(obs: pd.DataFrame, region_key: str, regions: tuple[str, ...]) -> TableRowPositions:
    if not regions:
        return np.array([], dtype=np.int64)
    if len(regions) == 1:
        region_mask = (obs[region_key] == regions[0]).to_numpy(dtype=bool, copy=False)
    else:
        region_mask = obs[region_key].isin(regions).to_numpy(dtype=bool, copy=False)
    return np.asarray(np.flatnonzero(region_mask), dtype=np.int64)


def _normalize_prediction_regions(value: object) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return ()
        value = value.tolist()
    if isinstance(value, (list, tuple)) and all(isinstance(region, str) for region in value):
        return tuple(value)
    return ()


def _normalize_scope_mode(scope_mode: ClassifierScopeMode | str) -> ClassifierScopeMode:
    if scope_mode in ("selected_segmentation_only", "all"):
        return scope_mode
    raise ValueError(f"Unsupported classifier scope mode: {scope_mode!r}")


def _empty_resolved_classifier_scope(mode: ClassifierScopeMode) -> ResolvedClassifierScope:
    return ResolvedClassifierScope(
        mode=mode,
        regions=(),
        table_row_positions=np.array([], dtype=np.int64),
        n_rows_in_regions=0,
    )


def _get_user_class_values(obs: pd.DataFrame, n_obs: int) -> np.ndarray:
    if USER_CLASS_COLUMN not in obs:
        return np.full(n_obs, UNLABELED_CLASS, dtype=np.int64)

    values = pd.to_numeric(obs[USER_CLASS_COLUMN].astype("string"), errors="coerce").fillna(UNLABELED_CLASS)
    return np.asarray(values, dtype=np.int64)


def _get_pred_class_values(obs: pd.DataFrame, n_obs: int) -> pd.Series:
    if PRED_CLASS_COLUMN not in obs:
        return pd.Series(UNLABELED_CLASS, index=obs.index, dtype="int64", name=PRED_CLASS_COLUMN)

    values = pd.to_numeric(obs[PRED_CLASS_COLUMN].astype("string"), errors="coerce").fillna(UNLABELED_CLASS)
    return pd.Series(np.asarray(values, dtype=np.int64), index=obs.index, dtype="int64", name=PRED_CLASS_COLUMN)


def _set_pred_class_annotation_state(table: AnnData, values: pd.Series) -> None:
    set_class_annotation_state(
        table,
        values,
        column_name=PRED_CLASS_COLUMN,
        colors_key=PRED_CLASS_COLORS_KEY,
        warn_on_palette_overwrite=False,
    )


def _get_pred_confidence_values(obs: pd.DataFrame, n_obs: int) -> pd.Series:
    if PRED_CONFIDENCE_COLUMN not in obs:
        return pd.Series(np.full(n_obs, np.nan, dtype=np.float64), index=obs.index, name=PRED_CONFIDENCE_COLUMN)

    values = pd.to_numeric(obs[PRED_CONFIDENCE_COLUMN], errors="coerce").astype("float64")
    return pd.Series(np.asarray(values, dtype=np.float64), index=obs.index, name=PRED_CONFIDENCE_COLUMN)
