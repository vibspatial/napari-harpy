from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QTimer
from sklearn.ensemble import RandomForestClassifier

from napari_harpy._annotation import UNLABELED_CLASS, USER_CLASS_COLUMN
from napari_harpy._spatialdata import SpatialDataAdapter, SpatialDataTableMetadata

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

PRED_CLASS_COLUMN = "pred_class"
PRED_CONFIDENCE_COLUMN = "pred_confidence"
CLASSIFIER_CONFIG_KEY = "classifier_config"

DEFAULT_RETRAIN_DEBOUNCE_MS = 300
MIN_LABELED_SAMPLES = 2
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": 0,
    "n_jobs": 1,
}


@dataclass(frozen=True)
class TrainingEligibility:
    """Describe whether the current bound table can be trained."""

    eligible: bool
    reason: str
    active_row_count: int
    labeled_count: int
    class_labels: tuple[int, ...]
    n_features: int | None


@dataclass(frozen=True)
class ClassifierJob:
    """Immutable payload copied on the main thread and consumed by a worker."""

    job_id: int
    feature_key: str
    label_name: str
    table_name: str
    active_positions: np.ndarray
    predict_features: Any
    train_features: Any
    train_labels: np.ndarray
    eligibility: TrainingEligibility


@dataclass(frozen=True)
class ClassifierJobResult:
    """Predictions produced by a completed training worker."""

    job_id: int
    feature_key: str
    label_name: str
    table_name: str
    active_positions: np.ndarray
    pred_classes: np.ndarray
    pred_confidences: np.ndarray
    trained_at: str
    model_params: dict[str, int]
    eligibility: TrainingEligibility


@thread_worker(start_thread=False, ignore_errors=True)
def _run_classifier_job(job: ClassifierJob) -> ClassifierJobResult:
    return _fit_classifier_job(job)


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
        active_positions=job.active_positions,
        pred_classes=pred_classes,
        pred_confidences=pred_confidences,
        trained_at=datetime.now(UTC).isoformat(),
        model_params={key: int(value) for key, value in RANDOM_FOREST_PARAMS.items()},
        eligibility=job.eligibility,
    )


class ClassifierController:
    """Manage debounced background classifier retraining for the active table."""

    def __init__(
        self,
        spatialdata_adapter: SpatialDataAdapter | None = None,
        *,
        debounce_interval_ms: int = DEFAULT_RETRAIN_DEBOUNCE_MS,
        on_state_changed: Any | None = None,
    ) -> None:
        self._spatialdata_adapter = spatialdata_adapter or SpatialDataAdapter()
        self._debounce_interval_ms = max(0, int(debounce_interval_ms))
        self._on_state_changed = on_state_changed

        self._selected_spatialdata: SpatialData | None = None
        self._selected_label_name: str | None = None
        self._selected_table_name: str | None = None
        self._selected_feature_key: str | None = None
        self._selected_table_metadata: SpatialDataTableMetadata | None = None

        self._latest_requested_job_id = 0
        self._active_worker_job_id: int | None = None
        self._active_worker: Any | None = None
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
    ) -> bool:
        """Bind the controller to the currently selected SpatialData inputs."""
        next_table_metadata = None
        if sdata is not None and table_name is not None:
            next_table_metadata = self._spatialdata_adapter.get_table_metadata(sdata, table_name)

        context_changed = (
            sdata is not self._selected_spatialdata
            or label_name != self._selected_label_name
            or table_name != self._selected_table_name
            or feature_key != self._selected_feature_key
        )

        self._selected_spatialdata = sdata
        self._selected_label_name = label_name
        self._selected_table_name = table_name
        self._selected_feature_key = feature_key
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
            self._set_status("Classifier: choose a feature matrix before retraining.", kind="warning")
            return False

        self._latest_requested_job_id += 1
        self._debounce_timer.stop()
        self._cancel_active_worker()
        self._is_dirty = True

        if immediate or self._debounce_interval_ms == 0:
            self._launch_retrain_job(self._latest_requested_job_id)
        else:
            self._debounce_timer.start()
            self._set_status("Classifier: model is stale. Retraining is scheduled.", kind="info")

        return True

    def _launch_scheduled_retrain(self) -> None:
        self._launch_retrain_job(self._latest_requested_job_id)

    def _launch_retrain_job(self, job_id: int) -> None:
        if job_id != self._latest_requested_job_id:
            return

        prepared = self._prepare_classifier_job(job_id)
        if prepared is None:
            return

        job, eligibility = prepared
        if not eligibility.eligible:
            self._apply_ineligible_state(eligibility)
            return

        worker = self._create_training_worker(job)
        self._active_worker = worker
        self._active_worker_job_id = job_id
        worker.returned.connect(partial(self._on_worker_returned, job_id))
        worker.errored.connect(partial(self._on_worker_errored, job_id))
        worker.finished.connect(partial(self._on_worker_finished, job_id))
        self._set_status(
            (
                f"Classifier: training RandomForest on {eligibility.labeled_count} labeled objects "
                f"across {len(eligibility.class_labels)} classes."
            ),
            kind="info",
        )
        worker.start()

    def _prepare_classifier_job(self, job_id: int) -> tuple[ClassifierJob, TrainingEligibility] | None:
        table = self._get_bound_table()
        metadata = self._selected_table_metadata
        if table is None or metadata is None or self._selected_label_name is None or self._selected_table_name is None:
            self._set_status("Classifier: choose an annotation table and feature matrix.", kind="warning")
            return None

        if self._selected_feature_key is None:
            eligibility = TrainingEligibility(
                eligible=False,
                reason="Choose a feature matrix before training the classifier.",
                active_row_count=0,
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )
            return (
                ClassifierJob(
                    job_id=job_id,
                    feature_key="",
                    label_name=self._selected_label_name,
                    table_name=self._selected_table_name,
                    active_positions=np.array([], dtype=np.int64),
                    predict_features=np.empty((0, 0), dtype=np.float64),
                    train_features=np.empty((0, 0), dtype=np.float64),
                    train_labels=np.array([], dtype=np.int64),
                    eligibility=eligibility,
                ),
                eligibility,
            )

        active_mask = (table.obs[metadata.region_key] == self._selected_label_name).to_numpy(dtype=bool)
        active_positions = np.flatnonzero(active_mask)
        if active_positions.size == 0:
            eligibility = TrainingEligibility(
                eligible=False,
                reason=(
                    f"No table rows for segmentation `{self._selected_label_name}` were found in "
                    f"`{self._selected_table_name}`."
                ),
                active_row_count=0,
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )
            return (
                ClassifierJob(
                    job_id=job_id,
                    feature_key=self._selected_feature_key,
                    label_name=self._selected_label_name,
                    table_name=self._selected_table_name,
                    active_positions=active_positions,
                    predict_features=np.empty((0, 0), dtype=np.float64),
                    train_features=np.empty((0, 0), dtype=np.float64),
                    train_labels=np.array([], dtype=np.int64),
                    eligibility=eligibility,
                ),
                eligibility,
            )

        try:
            feature_matrix = _normalize_feature_matrix(table.obsm[self._selected_feature_key], table.n_obs)
        except KeyError:
            eligibility = TrainingEligibility(
                eligible=False,
                reason=f"Feature matrix `{self._selected_feature_key}` is not available in `.obsm`.",
                active_row_count=int(active_positions.size),
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )
            return (
                ClassifierJob(
                    job_id=job_id,
                    feature_key=self._selected_feature_key,
                    label_name=self._selected_label_name,
                    table_name=self._selected_table_name,
                    active_positions=active_positions,
                    predict_features=np.empty((0, 0), dtype=np.float64),
                    train_features=np.empty((0, 0), dtype=np.float64),
                    train_labels=np.array([], dtype=np.int64),
                    eligibility=eligibility,
                ),
                eligibility,
            )
        except ValueError as error:
            eligibility = TrainingEligibility(
                eligible=False,
                reason=str(error),
                active_row_count=int(active_positions.size),
                labeled_count=0,
                class_labels=(),
                n_features=None,
            )
            return (
                ClassifierJob(
                    job_id=job_id,
                    feature_key=self._selected_feature_key,
                    label_name=self._selected_label_name,
                    table_name=self._selected_table_name,
                    active_positions=active_positions,
                    predict_features=np.empty((0, 0), dtype=np.float64),
                    train_features=np.empty((0, 0), dtype=np.float64),
                    train_labels=np.array([], dtype=np.int64),
                    eligibility=eligibility,
                ),
                eligibility,
            )

        predict_features = _slice_feature_rows(feature_matrix, active_positions)
        n_features = int(predict_features.shape[1])
        if n_features == 0:
            eligibility = TrainingEligibility(
                eligible=False,
                reason=f"Feature matrix `{self._selected_feature_key}` does not contain any columns.",
                active_row_count=int(active_positions.size),
                labeled_count=0,
                class_labels=(),
                n_features=0,
            )
            return (
                ClassifierJob(
                    job_id=job_id,
                    feature_key=self._selected_feature_key,
                    label_name=self._selected_label_name,
                    table_name=self._selected_table_name,
                    active_positions=active_positions,
                    predict_features=predict_features,
                    train_features=np.empty((0, 0), dtype=np.float64),
                    train_labels=np.array([], dtype=np.int64),
                    eligibility=eligibility,
                ),
                eligibility,
            )

        user_class_values = _get_user_class_values(table.obs, len(table.obs))
        active_user_class_values = user_class_values[active_positions]
        labeled_mask = active_user_class_values != UNLABELED_CLASS
        class_labels = tuple(sorted(int(value) for value in np.unique(active_user_class_values[labeled_mask])))
        eligibility = TrainingEligibility(
            eligible=True,
            reason="Ready to train.",
            active_row_count=int(active_positions.size),
            labeled_count=int(labeled_mask.sum()),
            class_labels=class_labels,
            n_features=n_features,
        )

        if eligibility.labeled_count < MIN_LABELED_SAMPLES:
            eligibility = TrainingEligibility(
                eligible=False,
                reason=(f"Need at least {MIN_LABELED_SAMPLES} labeled samples before training the classifier."),
                active_row_count=eligibility.active_row_count,
                labeled_count=eligibility.labeled_count,
                class_labels=eligibility.class_labels,
                n_features=eligibility.n_features,
            )
        elif len(eligibility.class_labels) < 2:
            eligibility = TrainingEligibility(
                eligible=False,
                reason="Need at least two labeled classes before training the classifier.",
                active_row_count=eligibility.active_row_count,
                labeled_count=eligibility.labeled_count,
                class_labels=eligibility.class_labels,
                n_features=eligibility.n_features,
            )

        train_features = _slice_feature_rows(predict_features, np.flatnonzero(labeled_mask))
        train_labels = np.asarray(active_user_class_values[labeled_mask], dtype=np.int64)
        job = ClassifierJob(
            job_id=job_id,
            feature_key=self._selected_feature_key,
            label_name=self._selected_label_name,
            table_name=self._selected_table_name,
            active_positions=active_positions,
            predict_features=predict_features,
            train_features=train_features,
            train_labels=train_labels,
            eligibility=eligibility,
        )
        return job, eligibility

    def _apply_ineligible_state(self, eligibility: TrainingEligibility) -> None:
        table = self._get_bound_table()
        if table is None:
            self._set_status(f"Classifier: {eligibility.reason}", kind="warning")
            return

        self._ensure_prediction_columns(table)
        if self._selected_label_name is not None and self._selected_table_metadata is not None:
            active_positions = np.flatnonzero(
                (table.obs[self._selected_table_metadata.region_key] == self._selected_label_name).to_numpy(dtype=bool)
            )
            self._set_predictions_for_active_rows(
                table,
                active_positions,
                np.full(active_positions.shape, UNLABELED_CLASS, dtype=np.int64),
                np.full(active_positions.shape, np.nan, dtype=np.float64),
            )

        table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
            eligibility=eligibility,
            trained=False,
            trained_at=None,
        )
        self._is_dirty = True
        self._set_status(f"Classifier: {eligibility.reason}", kind="warning")

    def _on_worker_returned(self, job_id: int, result: ClassifierJobResult) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        table = self._get_bound_table()
        if table is None:
            return

        self._ensure_prediction_columns(table)
        self._set_predictions_for_active_rows(
            table, result.active_positions, result.pred_classes, result.pred_confidences
        )
        table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
            eligibility=result.eligibility,
            trained=True,
            trained_at=result.trained_at,
        )
        self._is_dirty = False
        self._set_status(
            f"Classifier: model is up to date. Updated predictions for {result.eligibility.active_row_count} objects.",
            kind="success",
        )

    def _on_worker_errored(self, job_id: int, error: Exception) -> None:
        if job_id != self._latest_requested_job_id or job_id != self._active_worker_job_id:
            return

        table = self._get_bound_table()
        if table is not None:
            table.uns[CLASSIFIER_CONFIG_KEY] = self._build_classifier_config(
                eligibility=TrainingEligibility(
                    eligible=False,
                    reason=str(error),
                    active_row_count=0,
                    labeled_count=0,
                    class_labels=(),
                    n_features=None,
                ),
                trained=False,
                trained_at=None,
            )
        self._is_dirty = True
        self._set_status(f"Classifier: training failed: {error}", kind="error")

    def _on_worker_finished(self, job_id: int) -> None:
        if job_id != self._active_worker_job_id:
            return

        self._active_worker = None
        self._active_worker_job_id = None

    def _cancel_pending_and_active_jobs(self) -> None:
        self._debounce_timer.stop()
        self._cancel_active_worker()

    def _cancel_active_worker(self) -> None:
        if self._active_worker is None:
            return

        quit_worker = getattr(self._active_worker, "quit", None)
        if callable(quit_worker):
            quit_worker()
        self._active_worker = None
        self._active_worker_job_id = None

    def _create_training_worker(self, job: ClassifierJob) -> Any:
        return _run_classifier_job(job)

    def _get_bound_table(self) -> AnnData | None:
        if self._selected_spatialdata is None or self._selected_table_name is None:
            return None

        return self._spatialdata_adapter.get_table(self._selected_spatialdata, self._selected_table_name)

    def _ensure_prediction_columns(self, table: AnnData) -> None:
        pred_class_values = _get_pred_class_values(table.obs, len(table.obs))
        pred_confidence_values = _get_pred_confidence_values(table.obs, len(table.obs))
        table.obs[PRED_CLASS_COLUMN] = pred_class_values
        table.obs[PRED_CONFIDENCE_COLUMN] = pred_confidence_values

    def _set_predictions_for_active_rows(
        self,
        table: AnnData,
        active_positions: np.ndarray,
        pred_classes: np.ndarray,
        pred_confidences: np.ndarray,
    ) -> None:
        pred_class_values = _get_pred_class_values(table.obs, len(table.obs))
        pred_confidence_values = _get_pred_confidence_values(table.obs, len(table.obs))
        pred_class_values.iloc[active_positions] = np.asarray(pred_classes, dtype=np.int64)
        pred_confidence_values.iloc[active_positions] = np.asarray(pred_confidences, dtype=np.float64)
        table.obs[PRED_CLASS_COLUMN] = pred_class_values
        table.obs[PRED_CONFIDENCE_COLUMN] = pred_confidence_values

    def _build_classifier_config(
        self,
        *,
        eligibility: TrainingEligibility,
        trained: bool,
        trained_at: str | None,
    ) -> dict[str, object]:
        return {
            "model_type": "RandomForestClassifier",
            "feature_key": self._selected_feature_key,
            "table_name": self._selected_table_name,
            "label_name": self._selected_label_name,
            "roi_mode": "none",
            "trained": trained,
            "eligible": eligibility.eligible,
            "reason": eligibility.reason,
            "training_timestamp": trained_at,
            "n_labeled_objects": eligibility.labeled_count,
            "n_active_objects": eligibility.active_row_count,
            "n_features": eligibility.n_features,
            "class_labels_seen": list(eligibility.class_labels),
            "rf_params": dict(RANDOM_FOREST_PARAMS),
        }

    def _set_status(self, message: str, *, kind: str) -> None:
        self._status_message = message
        self._status_kind = kind
        if self._on_state_changed is not None:
            self._on_state_changed()

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
                self._set_status("Classifier: model is stale. Click Retrain to refresh predictions.", kind="warning")
            else:
                self._set_status(f"Classifier: model is stale because {reason}", kind="warning")
            return

        self._set_status("Classifier: model is up to date.", kind="success")


def _normalize_feature_matrix(feature_matrix: Any, n_obs: int) -> Any:
    if issparse(feature_matrix):
        if feature_matrix.ndim != 2:
            raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
        if feature_matrix.shape[0] != n_obs:
            raise ValueError(
                f"Feature matrix has {feature_matrix.shape[0]} rows but the table has {n_obs} observations."
            )
        return feature_matrix.copy()

    array = np.asarray(feature_matrix, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Feature matrices stored in `.obsm` must be 2-dimensional.")
    if array.shape[0] != n_obs:
        raise ValueError(f"Feature matrix has {array.shape[0]} rows but the table has {n_obs} observations.")
    return array.copy()


def _slice_feature_rows(feature_matrix: Any, positions: np.ndarray) -> Any:
    return feature_matrix[positions]


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


def _get_pred_confidence_values(obs: pd.DataFrame, n_obs: int) -> pd.Series:
    if PRED_CONFIDENCE_COLUMN not in obs:
        return pd.Series(np.full(n_obs, np.nan, dtype=np.float64), index=obs.index, name=PRED_CONFIDENCE_COLUMN)

    values = pd.to_numeric(obs[PRED_CONFIDENCE_COLUMN], errors="coerce").astype("float64")
    return pd.Series(np.asarray(values, dtype=np.float64), index=obs.index, name=PRED_CONFIDENCE_COLUMN)
