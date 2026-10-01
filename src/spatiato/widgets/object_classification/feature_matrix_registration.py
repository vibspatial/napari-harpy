from __future__ import annotations

from dataclasses import dataclass

from napari_harpy.core.feature_matrix_metadata import (
    CUSTOM_OBSM_SOURCE_KIND,
    HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND,
    FeatureMatrixMetadataState,
)


@dataclass(frozen=True)
class _FeatureMatrixRegistrationButtonState:
    enabled: bool
    tooltip: str
    warning_message: str | None = None


def _build_feature_matrix_registration_button_state(
    metadata_state: FeatureMatrixMetadataState | None,
) -> _FeatureMatrixRegistrationButtonState:
    if metadata_state is None:
        return _FeatureMatrixRegistrationButtonState(
            enabled=False,
            tooltip="Choose an annotation table and feature matrix before registering feature metadata.",
        )

    feature_key = metadata_state.feature_key
    if metadata_state.status == "unregistered":
        return _FeatureMatrixRegistrationButtonState(
            enabled=True,
            tooltip=(
                f'Register feature-column metadata for "{feature_key}" so classifiers trained on this matrix '
                "can be exported and reused."
            ),
        )

    if metadata_state.status == "registered_valid":
        if metadata_state.source_kind == CUSTOM_OBSM_SOURCE_KIND:
            tooltip = f'Feature matrix "{feature_key}" is already registered as a custom `.obsm` feature matrix.'
        elif metadata_state.source_kind == HARPY_ADD_FEATURE_MATRIX_SOURCE_KIND:
            tooltip = f'Feature matrix "{feature_key}" is already registered from Harpy feature extraction.'
        else:
            tooltip = f'Feature matrix "{feature_key}" is already registered.'
        return _FeatureMatrixRegistrationButtonState(enabled=False, tooltip=tooltip)

    if metadata_state.status == "registered_mismatched":
        detail = metadata_state.error or "Existing feature metadata does not match the live matrix."
        warning_message = (
            f'Feature matrix "{feature_key}" has mismatched metadata. {detail} '
            "Registration from the widget is disabled to avoid overwriting existing metadata."
        )
        return _FeatureMatrixRegistrationButtonState(
            enabled=False,
            tooltip=warning_message,
            warning_message=warning_message,
        )

    if metadata_state.status == "invalid_matrix":
        detail = metadata_state.error or "The selected matrix is not a valid feature matrix."
        warning_message = f'Feature matrix "{feature_key}" cannot be registered. {detail}'
        return _FeatureMatrixRegistrationButtonState(
            enabled=False,
            tooltip=warning_message,
            warning_message=warning_message,
        )

    if metadata_state.status == "missing_matrix":
        warning_message = metadata_state.error or f'Feature matrix "{feature_key}" is not available in `.obsm`.'
        return _FeatureMatrixRegistrationButtonState(
            enabled=False,
            tooltip=warning_message,
            warning_message=warning_message,
        )

    raise ValueError(f"Unsupported feature matrix metadata status: {metadata_state.status!r}.")
