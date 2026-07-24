from __future__ import annotations

import copy
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from napari_harpy.core.class_palette import (
    default_categorical_colors,
    extend_categorical_palette,
    normalize_color_sequence,
    resolve_table_categorical_palette,
)
from napari_harpy.core.object_classification.annotation import USER_CLASS_COLUMN
from napari_harpy.core.object_classification.classifier import (
    PRED_CLASS_COLUMN,
    PRED_CONFIDENCE_COLUMN,
)
from napari_harpy.core.spatial_query.canonical import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_OBSM_KEY,
    build_canonical_region_binding,
    inspect_canonical_cache,
)
from napari_harpy.core.spatial_query.canonical_models import (
    CanonicalCacheState,
    CanonicalRegionBinding,
    _readonly_array,
)
from napari_harpy.core.spatial_query.query_models import CanonicalCenterQueryResult
from napari_harpy.core.spatialdata import get_table_metadata
from napari_harpy.core.validation import normalize_spatialdata_dataframe_column_name

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

type SpatialAnnotationColumnMode = Literal["existing", "new"]
type SpatialAnnotationValueKind = Literal["string", "positive_integer"]
type SpatialAnnotationValue = str | int | None

_CLASSIFIER_OWNED_COLUMNS = frozenset((PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN))
_OBJECT_CLASSIFICATION_COLUMNS = frozenset((USER_CLASS_COLUMN, PRED_CLASS_COLUMN, PRED_CONFIDENCE_COLUMN))


class SpatialAnnotationColumnChangedError(ValueError):
    """The reviewed column changed and its summary must be refreshed."""


class SpatialAnnotationQueryOutdatedError(ValueError):
    """The binding or canonical-center query provenance is no longer current."""


@dataclass(frozen=True)
class SpatialAnnotationPreparation:
    """Immutable table-row and target-column snapshot prepared for review."""

    query_result: CanonicalCenterQueryResult
    column_name: str
    column_mode: SpatialAnnotationColumnMode
    row_positions: NDArray[np.intp] = field(repr=False, compare=False)
    current_values: pd.Series = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query_result, CanonicalCenterQueryResult):
            raise TypeError("Spatial annotation preparation requires a CanonicalCenterQueryResult.")
        if self.column_mode not in ("existing", "new"):
            raise ValueError("Spatial annotation column mode must be `existing` or `new`.")
        if not isinstance(self.column_name, str) or not self.column_name:
            raise ValueError("Spatial annotation column name must not be empty.")
        row_positions = _readonly_array(self.row_positions, dtype=np.intp)
        if row_positions.ndim != 1 or np.any(row_positions < 0):
            raise ValueError("Spatial annotation row positions must be a one-dimensional non-negative array.")
        if len(np.unique(row_positions)) != len(row_positions):
            raise ValueError("Spatial annotation row positions must be unique.")
        if len(row_positions) != self.query_result.matched_instance_count:
            raise ValueError("Spatial annotation row positions must match the queried instance count.")
        if not isinstance(self.current_values, pd.Series):
            raise TypeError("Spatial annotation current values must be a pandas Series.")
        if len(self.current_values) != len(row_positions):
            raise ValueError("Spatial annotation current values must match the queried instance count.")
        object.__setattr__(self, "row_positions", row_positions)
        object.__setattr__(self, "current_values", self.current_values.copy(deep=True))
        _ = self.value_kind

    @property
    def value_kind(self) -> SpatialAnnotationValueKind:
        """Return the annotation kind represented by the captured column."""
        return validate_and_resolve_spatial_annotation_value_kind(
            self.current_values,
            column_name=self.column_name,
        )

    @property
    def binding(self) -> CanonicalRegionBinding:
        """Return the selected-region binding used by the query."""
        return self.query_result.binding


@dataclass(frozen=True)
class SpatialAnnotationSummary:
    """Reviewed effect of setting or removing one annotation value.

    Parameters
    ----------
    annotation_value
        Normalized string or positive integer category for Set annotation.
        ``None`` is the domain sentinel for Remove annotation: Apply clears
        the matched categorical values with ``pd.NA``. ``None`` itself is not
        stored, and the strings ``"None"`` and ``"nan"`` remain ordinary
        annotation categories.
    matched_count
        Total number of matched rows included in the review.
    current_missing_count
        Number of matched rows whose current annotation is missing.
    current_equal_count
        Number of matched rows already equal to a Set annotation value. This is
        zero for Remove annotation.
    current_other_count
        Number of different non-missing values that Set would overwrite, or
        existing non-missing annotations that Remove would clear.
    """

    annotation_value: SpatialAnnotationValue
    matched_count: int
    current_missing_count: int
    current_equal_count: int
    current_other_count: int

    def __post_init__(self) -> None:
        annotation_value = self.annotation_value
        if annotation_value is not None:
            if isinstance(annotation_value, str):
                annotation_value = annotation_value.strip()
                if not annotation_value:
                    raise ValueError("Spatial annotation string value must not be empty.")
            elif isinstance(annotation_value, bool) or not isinstance(annotation_value, int):
                raise TypeError("Spatial annotation value must be a string, positive integer, or None.")
            elif annotation_value <= 0:
                raise ValueError("Spatial annotation integer value must be positive.")
            object.__setattr__(self, "annotation_value", annotation_value)

        counts = (
            self.matched_count,
            self.current_missing_count,
            self.current_equal_count,
            self.current_other_count,
        )
        if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0 for count in counts):
            raise ValueError("Spatial annotation summary counts must be non-negative integers.")
        if self.matched_count != sum(counts[1:]):
            raise ValueError("Spatial annotation summary counts must partition every matched row.")
        if self.is_removal and self.current_equal_count != 0:
            raise ValueError("Removal summaries must have zero equal values.")

    @property
    def is_removal(self) -> bool:
        """Return whether this summary describes annotation removal."""
        return self.annotation_value is None

    @property
    def changed_count(self) -> int:
        """Return the number of rows that would change."""
        if self.is_removal:
            return self.current_other_count
        return self.current_missing_count + self.current_other_count

    @property
    def unchanged_count(self) -> int:
        """Return the number of rows already in the requested state."""
        if self.is_removal:
            return self.current_missing_count
        return self.current_equal_count

    @property
    def overwrite_count(self) -> int:
        """Return the number of different non-missing values Set would replace."""
        return 0 if self.is_removal else self.current_other_count

    @property
    def removal_count(self) -> int:
        """Return the number of existing annotations Remove would clear."""
        return self.current_other_count if self.is_removal else 0


@dataclass(frozen=True)
class SpatialAnnotationApplyResult:
    """Report which parts of the annotation consistency unit changed."""

    annotation_changed: bool
    palette_changed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.annotation_changed, bool) or not isinstance(self.palette_changed, bool):
            raise TypeError("Spatial annotation apply flags must be booleans.")
        if self.palette_changed and not self.annotation_changed:
            raise ValueError("A spatial-annotation palette changes only with an effective annotation mutation.")


def prepare_spatial_annotation(
    sdata: SpatialData,
    *,
    query_result: CanonicalCenterQueryResult,
    column_name: str,
    column_mode: SpatialAnnotationColumnMode,
) -> SpatialAnnotationPreparation:
    """Resolve queried instance IDs to live table rows without mutation."""
    if not isinstance(query_result, CanonicalCenterQueryResult):
        raise TypeError("Spatial annotation preparation requires a CanonicalCenterQueryResult.")
    if query_result.matched_instance_count == 0:
        raise ValueError("Spatial annotation preparation requires at least one matching instance.")
    if column_mode not in ("existing", "new"):
        raise ValueError("Spatial annotation column mode must be `existing` or `new`.")
    normalized_column_name = normalize_spatialdata_dataframe_column_name(column_name, "Annotation column name")

    table, live_binding = _resolve_live_binding(sdata, query_result.binding)
    if not _bindings_match(live_binding, query_result.binding):
        raise SpatialAnnotationQueryOutdatedError(
            "The selected table-region binding changed after the canonical-center query."
        )
    if normalized_column_name in (live_binding.region_key, live_binding.instance_key):
        raise ValueError(f"Annotation column `{normalized_column_name}` cannot be a table linkage column.")
    row_positions = _resolve_query_row_positions(query_result, live_binding)

    if column_mode == "existing":
        current_column = require_compatible_spatial_annotation_column(table, normalized_column_name)
        current_values = current_column.iloc[row_positions].copy(deep=True)
    else:
        if normalized_column_name in _OBJECT_CLASSIFICATION_COLUMNS:
            raise ValueError(
                f"New annotation column `{normalized_column_name}` is reserved for Object Classification "
                "and cannot be created by Spatial Query."
            )
        if normalized_column_name in table.obs.columns:
            raise ValueError(f"New annotation column `{normalized_column_name}` already exists.")
        current_values = pd.Series(
            pd.Categorical([pd.NA] * len(row_positions), categories=[]),
            index=table.obs.index[row_positions],
            name=normalized_column_name,
        )

    return SpatialAnnotationPreparation(
        query_result=query_result,
        column_name=normalized_column_name,
        column_mode=column_mode,
        row_positions=row_positions,
        current_values=current_values,
    )


def summarize_spatial_annotation(
    preparation: SpatialAnnotationPreparation,
    annotation_value: SpatialAnnotationValue,
) -> SpatialAnnotationSummary:
    """Summarize a proposed Set or Remove action without mutation."""
    if not isinstance(preparation, SpatialAnnotationPreparation):
        raise TypeError("Spatial annotation summarization requires a SpatialAnnotationPreparation.")
    if annotation_value is None and preparation.column_mode == "new":
        raise ValueError("Remove annotation is not available for a new column.")

    normalized_value = annotation_value
    if normalized_value is not None:
        if preparation.value_kind == "string":
            if not isinstance(normalized_value, str):
                raise TypeError("String spatial annotation targets require a string value or None.")
            normalized_value = normalized_value.strip()
            if not normalized_value:
                raise ValueError("Spatial annotation string value must not be empty.")
        elif preparation.value_kind == "positive_integer":
            if isinstance(normalized_value, bool) or not isinstance(normalized_value, int):
                raise TypeError("Positive-integer spatial annotation targets require a Python int or None.")
            if normalized_value <= 0:
                raise ValueError("Spatial annotation integer value must be positive.")
        else:  # pragma: no cover - guarded by SpatialAnnotationPreparation
            raise ValueError(f"Unsupported spatial annotation value kind: {preparation.value_kind!r}.")

    missing = preparation.current_values.isna().to_numpy(dtype=bool)
    if normalized_value is None:
        equal = np.zeros(len(missing), dtype=bool)
    else:
        equal = np.asarray(preparation.current_values == normalized_value, dtype=bool) & ~missing
    other = ~missing & ~equal

    return SpatialAnnotationSummary(
        annotation_value=normalized_value,
        matched_count=len(missing),
        current_missing_count=int(missing.sum()),
        current_equal_count=int(equal.sum()),
        current_other_count=int(other.sum()),
    )


def apply_spatial_annotation(
    sdata: SpatialData,
    preparation: SpatialAnnotationPreparation,
    expected_summary: SpatialAnnotationSummary,
) -> SpatialAnnotationApplyResult:
    """Revalidate and atomically apply one reviewed annotation mutation.

    Parameters
    ----------
    sdata
        SpatialData object containing the current table and canonical-center
        cache.
    preparation
        Immutable row and target-column snapshot created before the annotation
        was reviewed.
    expected_summary
        Exact Set or Remove summary presented for review. The annotation value
        lives on this summary so the reviewed value and counts cannot disagree.
        Apply first verifies that the live binding, cache, centers, and selected
        column values still match ``preparation``. It then recalculates the
        summary from those accepted values and requires it to equal
        ``expected_summary`` before mutating the table. This final comparison
        protects the review contract; live table changes are handled by the
        preceding freshness validation.

    Returns
    -------
    SpatialAnnotationApplyResult
        Whether the annotation column and companion palette changed.
    """
    if not isinstance(preparation, SpatialAnnotationPreparation):
        raise TypeError("Spatial annotation apply requires a SpatialAnnotationPreparation.")
    if not isinstance(expected_summary, SpatialAnnotationSummary):
        raise TypeError("Spatial annotation apply requires a SpatialAnnotationSummary.")
    if expected_summary.is_removal and preparation.column_mode == "new":
        raise ValueError("Remove annotation is not available for a new column.")

    table = _require_current_query_provenance(sdata, preparation)
    current_column: pd.Series | None = None
    if preparation.column_mode == "existing":
        current_column = require_compatible_spatial_annotation_column(table, preparation.column_name)
        current_value_kind = validate_and_resolve_spatial_annotation_value_kind(
            current_column,
            column_name=preparation.column_name,
        )
        if current_value_kind != preparation.value_kind:
            raise SpatialAnnotationColumnChangedError(
                "The annotation column value kind changed while the annotation was being reviewed."
            )
        _require_unchanged_existing_column(preparation, current_column)
    else:
        if preparation.column_name in _OBJECT_CLASSIFICATION_COLUMNS:
            raise ValueError(
                f"New annotation column `{preparation.column_name}` is reserved for Object Classification "
                "and cannot be created by Spatial Query."
            )
        if preparation.column_name in table.obs.columns:
            raise ValueError(f"New annotation column `{preparation.column_name}` already exists.")

    fresh_summary = summarize_spatial_annotation(preparation, expected_summary.annotation_value)
    if fresh_summary != expected_summary:
        raise ValueError("The expected spatial-annotation summary does not match the prepared values.")
    if fresh_summary.changed_count == 0:
        return SpatialAnnotationApplyResult(annotation_changed=False, palette_changed=False)

    # Build the complete replacement column and optional companion palette
    # off-table. This phase reads the live AnnData state but does not mutate it.
    replacement, palette, palette_changed = _build_annotation_replacement(
        table,
        preparation,
        current_column,
        fresh_summary,
    )
    # This is the AnnData mutation boundary: install the prepared `.obs`
    # column and `.uns` palette as one unit, rolling both back on failure.
    _assign_annotation_column_and_palette_atomically(
        table,
        column_name=preparation.column_name,
        replacement=replacement,
        palette=palette,
    )
    return SpatialAnnotationApplyResult(annotation_changed=True, palette_changed=palette_changed)


def _resolve_live_binding(
    sdata: SpatialData,
    expected_binding: CanonicalRegionBinding,
) -> tuple[AnnData, CanonicalRegionBinding]:
    try:
        table = sdata.tables[expected_binding.table_name]
        table_metadata = get_table_metadata(sdata, expected_binding.table_name)
        binding = build_canonical_region_binding(table, table_metadata, expected_binding.labels_name)
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialAnnotationQueryOutdatedError("The selected table-region binding is no longer available.") from exc
    return table, binding


def _bindings_match(current: CanonicalRegionBinding, expected: CanonicalRegionBinding) -> bool:
    return (
        current.table_name == expected.table_name
        and current.labels_name == expected.labels_name
        and current.region_key == expected.region_key
        and current.instance_key == expected.instance_key
        and np.array_equal(current.row_positions, expected.row_positions)
        and np.array_equal(current.instance_ids, expected.instance_ids)
    )


def require_compatible_spatial_annotation_column(table: AnnData, column_name: str) -> pd.Series:
    """Return an existing column that can store spatial annotations."""
    if column_name not in table.obs.columns:
        raise ValueError(f"Existing annotation column `{column_name}` is not present.")
    values = table.obs[column_name]
    validate_and_resolve_spatial_annotation_value_kind(values, column_name=column_name)
    return values


def validate_and_resolve_spatial_annotation_value_kind(
    values: pd.Series,
    *,
    column_name: str,
) -> SpatialAnnotationValueKind:
    """Validate a categorical annotation column and return its value kind."""
    if column_name in _CLASSIFIER_OWNED_COLUMNS:
        raise ValueError(
            f"Existing annotation column `{column_name}` is classifier-owned and cannot be modified by Spatial Query."
        )
    if not isinstance(values, pd.Series):
        raise TypeError("Spatial annotation values must be a pandas Series.")
    if not isinstance(values.dtype, pd.CategoricalDtype):
        raise ValueError(f"Existing annotation column `{column_name}` must be categorical.")

    categories = values.cat.categories
    # `user_class` has a fixed positive-integer schema. Handle it before the
    # generic empty-category fallback because an empty categorical commonly
    # has object-typed categories and would otherwise be misclassified as a
    # string target. `all([])` intentionally accepts that empty vocabulary,
    # while a non-empty string or invalid integer vocabulary still fails.
    if column_name == USER_CLASS_COLUMN:
        if all(_is_positive_integer_category(category) for category in categories):
            return "positive_integer"
        raise ValueError(f"Existing annotation column `{column_name}` must contain only positive integer categories.")
    if len(categories) == 0:
        if pd.api.types.is_integer_dtype(categories.dtype) and not pd.api.types.is_bool_dtype(categories.dtype):
            return "positive_integer"
        return "string"
    if all(isinstance(category, str) for category in categories):
        return "string"
    if all(_is_positive_integer_category(category) for category in categories):
        return "positive_integer"
    raise ValueError(
        f"Existing annotation column `{column_name}` must contain only string categories "
        "or positive integer categories."
    )


def get_compatible_spatial_annotation_column_names(
    sdata: SpatialData,
    table_name: str,
) -> list[str]:
    """Return compatible existing annotation columns in table order."""
    table = sdata.tables[table_name]
    table_metadata = get_table_metadata(sdata, table_name)
    excluded_columns = {
        table_metadata.region_key,
        table_metadata.instance_key,
        *_CLASSIFIER_OWNED_COLUMNS,
    }
    compatible_columns: list[str] = []
    for column_name in table.obs.columns:
        if not isinstance(column_name, str) or column_name in excluded_columns:
            continue
        try:
            validate_and_resolve_spatial_annotation_value_kind(
                table.obs[column_name],
                column_name=column_name,
            )
        except (TypeError, ValueError):
            continue
        compatible_columns.append(column_name)
    return compatible_columns


def _is_positive_integer_category(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and int(value) > 0


def _resolve_query_row_positions(
    query_result: CanonicalCenterQueryResult,
    binding: CanonicalRegionBinding,
) -> NDArray[np.intp]:
    sort_order = np.argsort(binding.instance_ids)
    sorted_instance_ids = binding.instance_ids[sort_order]
    selected_positions = np.searchsorted(sorted_instance_ids, query_result.matched_instance_ids)
    if np.any(selected_positions >= len(sorted_instance_ids)) or not np.array_equal(
        sorted_instance_ids[selected_positions], query_result.matched_instance_ids
    ):  # pragma: no cover - guarded by the result and binding checks
        raise SpatialAnnotationQueryOutdatedError(
            "A queried instance is no longer present in the selected table region."
        )
    return np.asarray(binding.row_positions[sort_order[selected_positions]], dtype=np.intp)


def _require_current_query_provenance(
    sdata: SpatialData,
    preparation: SpatialAnnotationPreparation,
) -> AnnData:
    """Return the current table after accepting the query provenance.

    Re-inspect the selected canonical cache immediately before annotation
    mutation. The live table-region binding, resolved queried rows, labels
    source signature, selected-region metadata, and row-aligned canonical
    centers must still equal the snapshots retained by ``preparation``.
    Otherwise, raise ``SpatialAnnotationQueryOutdatedError`` and require a new
    query.

    This is necessary because the query result is calculated before the user
    reviews and confirms the annotation. While that result is pending, another
    operation can reload or replace the table, change its row-to-instance
    binding, or rebuild its canonical-center cache. Applying the old matching
    instance IDs after such a change could annotate rows using a geometric
    decision made from different centers. Requiring the complete provenance to
    remain current prevents that stale-result mutation.

    This validation reads only table state and the in-memory canonical-center
    cache. It does not read labels pixels or recalculate centers. Shapes and
    transformation freshness remain controller responsibilities.
    """
    canonical_centers = preparation.query_result.canonical_centers
    try:
        report = inspect_canonical_cache(
            sdata,
            table_name=preparation.binding.table_name,
            labels_name=preparation.binding.labels_name,
        )
        table = sdata.tables[preparation.binding.table_name]
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialAnnotationQueryOutdatedError("The canonical-center query inputs are no longer available.") from exc

    if report.state is not CanonicalCacheState.VALID:
        raise SpatialAnnotationQueryOutdatedError("The canonical-center cache is no longer valid.")
    if not _bindings_match(report.binding, preparation.binding):
        raise SpatialAnnotationQueryOutdatedError(
            "The selected table-region binding changed after annotation preparation."
        )
    current_row_positions = _resolve_query_row_positions(preparation.query_result, report.binding)
    if not np.array_equal(current_row_positions, preparation.row_positions):
        raise SpatialAnnotationQueryOutdatedError("The queried instances no longer resolve to the prepared table rows.")
    if report.source_signature != canonical_centers.source_signature:
        raise SpatialAnnotationQueryOutdatedError("The labels source changed after the canonical-center query.")

    stored_metadata = report.stored_metadata
    region_metadata = None if stored_metadata is None else stored_metadata.regions.get(preparation.binding.labels_name)
    if (
        region_metadata is None
        or region_metadata.source_signature != canonical_centers.source_signature
        or region_metadata.n_obs != preparation.binding.n_obs
        or region_metadata.instance_set_digest != preparation.binding.instance_set_digest
        or region_metadata.algorithm_version != CANONICAL_ALGORITHM_VERSION
    ):
        raise SpatialAnnotationQueryOutdatedError(
            "The selected region's canonical-center metadata changed after the query."
        )

    current_centers = np.asarray(table.obsm[CANONICAL_OBSM_KEY])[report.binding.row_positions]
    if not np.array_equal(current_centers, canonical_centers.centers):
        raise SpatialAnnotationQueryOutdatedError("The selected region's canonical centers changed after the query.")
    return table


def _require_unchanged_existing_column(
    preparation: SpatialAnnotationPreparation,
    current_column: pd.Series,
) -> None:
    prepared_values = preparation.current_values
    current_categories = tuple(current_column.cat.categories)
    prepared_categories = tuple(prepared_values.cat.categories)
    if current_categories != prepared_categories or current_column.cat.ordered != prepared_values.cat.ordered:
        raise SpatialAnnotationColumnChangedError(
            "The annotation column categories changed while the annotation was being reviewed."
        )
    current_values = current_column.iloc[preparation.row_positions].reset_index(drop=True)
    prepared_values = prepared_values.reset_index(drop=True)
    if not current_values.equals(prepared_values):
        raise SpatialAnnotationColumnChangedError(
            "The annotation values changed while the annotation was being reviewed."
        )


def _build_annotation_replacement(
    table: AnnData,
    preparation: SpatialAnnotationPreparation,
    current_column: pd.Series | None,
    summary: SpatialAnnotationSummary,
) -> tuple[pd.Series, list[str] | None, bool]:
    column_name = preparation.column_name
    annotation_value = summary.annotation_value
    if preparation.column_mode == "new":
        assert annotation_value is not None
        replacement = pd.Series(
            pd.Categorical([pd.NA] * table.n_obs, categories=[annotation_value]),
            index=table.obs.index,
            name=column_name,
        )
        replacement.iloc[preparation.row_positions] = annotation_value
        return replacement, default_categorical_colors(1), True

    assert current_column is not None
    current_categories = tuple(current_column.cat.categories)
    replacement = current_column.copy(deep=True)
    next_categories = current_categories
    if annotation_value is not None and annotation_value not in current_categories:
        next_categories = (*current_categories, annotation_value)
        replacement = replacement.cat.add_categories([annotation_value])
    replacement.iloc[preparation.row_positions] = pd.NA if annotation_value is None else annotation_value

    palette_source, current_palette = resolve_table_categorical_palette(
        table=table,
        column_name=column_name,
        categories=current_categories,
    )
    if next_categories != current_categories:
        palette = extend_categorical_palette(
            current_palette,
            current_categories=current_categories,
            next_categories=next_categories,
        )
        return replacement, palette, True
    if palette_source != "stored":
        return replacement, current_palette, True
    return replacement, None, False


def _assign_annotation_column_and_palette_atomically(
    table: AnnData,
    *,
    column_name: str,
    replacement: pd.Series,
    palette: list[str] | None,
) -> None:
    column_existed = column_name in table.obs.columns
    previous_column = table.obs[column_name].copy(deep=True) if column_existed else None
    palette_key = f"{column_name}_colors"
    palette_existed = palette_key in table.uns
    previous_palette = copy.deepcopy(table.uns[palette_key]) if palette_existed else None

    try:
        table.obs[column_name] = replacement
        if palette is not None:
            table.uns[palette_key] = list(palette)
        if not table.obs[column_name].equals(replacement):
            raise RuntimeError("Spatial annotation column assignment did not preserve the prepared values.")
        if palette is not None and normalize_color_sequence(table.uns.get(palette_key)) != palette:
            raise RuntimeError("Spatial annotation palette assignment did not preserve the prepared colors.")
    except BaseException:
        try:
            if column_existed:
                table.obs[column_name] = previous_column
            elif column_name in table.obs.columns:
                table.obs.pop(column_name)
            if palette_existed:
                table.uns[palette_key] = previous_palette
            else:
                table.uns.pop(palette_key, None)
        except BaseException as rollback_error:  # pragma: no cover - catastrophic mapping failure
            raise RuntimeError(
                "Spatial annotation assignment failed, and rollback could not restore the previous column and palette."
            ) from rollback_error
        raise
