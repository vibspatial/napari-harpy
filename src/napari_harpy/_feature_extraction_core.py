from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

FeatureExtractionChannel = int | str

_INTENSITY_FEATURES = frozenset({"sum", "mean", "var", "min", "max", "kurtosis", "skew"})


@dataclass(frozen=True)
class FeatureExtractionTriplet:
    """One explicit `coordinate_system -> segmentation -> image` selection."""

    coordinate_system: str
    label_name: str
    image_name: str | None
    channels: tuple[FeatureExtractionChannel, ...] | None = None


def _normalize_channels(
    channels: Sequence[FeatureExtractionChannel] | FeatureExtractionChannel | None,
) -> tuple[FeatureExtractionChannel, ...] | None:
    if channels is None:
        return None
    if isinstance(channels, (str, int)):
        values = [channels]
    else:
        values = list(channels)

    normalized: list[FeatureExtractionChannel] = []
    seen: set[FeatureExtractionChannel] = set()
    for channel in values:
        if isinstance(channel, str):
            normalized_channel: FeatureExtractionChannel = channel.strip()
            if not normalized_channel:
                continue
        else:
            normalized_channel = channel

        if normalized_channel in seen:
            raise ValueError(f"Duplicate channel selection is not allowed: `{normalized_channel}`.")
        normalized.append(normalized_channel)
        seen.add(normalized_channel)
    return tuple(normalized)


def _normalize_triplets(
    triplets: Sequence[FeatureExtractionTriplet] | FeatureExtractionTriplet | None,
) -> tuple[FeatureExtractionTriplet, ...]:
    if triplets is None:
        return ()
    if isinstance(triplets, FeatureExtractionTriplet):
        values = [triplets]
    else:
        values = list(triplets)

    normalized: list[FeatureExtractionTriplet] = []
    seen_label_names: set[str] = set()
    for triplet in values:
        normalized_coordinate_system = str(triplet.coordinate_system).strip()
        normalized_label_name = str(triplet.label_name).strip()
        normalized_image_name = None if triplet.image_name is None else str(triplet.image_name).strip() or None
        normalized_channels = _normalize_channels(triplet.channels)

        if not normalized_coordinate_system:
            raise ValueError("Feature extraction triplets require a coordinate system.")
        if not normalized_label_name:
            raise ValueError("Feature extraction triplets require a segmentation name.")
        if normalized_label_name in seen_label_names:
            raise ValueError(
                f"Duplicate segmentation selections are not allowed in a single feature-extraction request: "
                f"`{normalized_label_name}`."
            )

        normalized.append(
            FeatureExtractionTriplet(
                coordinate_system=normalized_coordinate_system,
                label_name=normalized_label_name,
                image_name=normalized_image_name,
                channels=normalized_channels,
            )
        )
        seen_label_names.add(normalized_label_name)

    return tuple(normalized)


def _requires_image(feature_names: Sequence[str]) -> bool:
    return any(feature_name in _INTENSITY_FEATURES for feature_name in feature_names)


def _get_triplet_channel_selection_error(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> str | None:
    if not _requires_image(feature_names) or len(triplets) <= 1:
        return None

    first_channels = triplets[0].channels
    for triplet in triplets[1:]:
        if triplet.channels != first_channels:
            return "Feature extraction: all selected extraction targets must currently use the same channel selection."

    return None


def _resolve_harpy_labels_name_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
) -> str | list[str]:
    label_names = [triplet.label_name for triplet in triplets]
    return label_names[0] if len(label_names) == 1 else label_names


def _resolve_harpy_coordinate_system_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
) -> str | list[str]:
    coordinate_systems = [triplet.coordinate_system for triplet in triplets]
    return coordinate_systems[0] if len(coordinate_systems) == 1 else coordinate_systems


def _resolve_harpy_image_name_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> str | list[str] | None:
    if not _requires_image(feature_names):
        return None

    image_names = [triplet.image_name for triplet in triplets]
    if any(image_name is None for image_name in image_names):
        raise ValueError(
            "An image is required for every extraction target when intensity-derived features are selected."
        )

    resolved_image_names = [str(image_name) for image_name in image_names if image_name is not None]
    return resolved_image_names[0] if len(resolved_image_names) == 1 else resolved_image_names


def _resolve_harpy_channel_parameter(
    triplets: Sequence[FeatureExtractionTriplet],
    feature_names: Sequence[str],
) -> list[FeatureExtractionChannel] | None:
    if not _requires_image(feature_names):
        return None

    channel_selection_error = _get_triplet_channel_selection_error(triplets, feature_names)
    if channel_selection_error is not None:
        raise ValueError(channel_selection_error)

    channels = triplets[0].channels
    if channels is None:
        return None

    return list(channels)
