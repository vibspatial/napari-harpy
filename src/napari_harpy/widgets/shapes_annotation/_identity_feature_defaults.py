"""Guard napari Shapes feature defaults for annotation identity columns."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from napari.layers import Shapes


class _AnnotationIdentityFeatureDefaultGuard:
    """Keep source row identity out of napari's new-shape feature defaults.

    Napari uses ``layer.current_properties`` as the one-row feature template for
    the next drawn shape. It can seed or update that template from existing
    feature rows, for example from the last row when a layer is initialized or
    from a selected row during interaction. That is useful for normal columns
    such as ``label`` or ``class``, but it is wrong for Harpy's source identity
    column because a new unsaved row must not inherit an existing row ID.

    Example before clearing::

        layer.current_properties == {
            "instance_id": np.asarray(["__annotation_1"], dtype=object),
            "label": np.asarray(["tumor"], dtype=object),
        }

    The guard changes only the active source identity feature::

        layer.current_properties == {
            "instance_id": np.asarray([pd.NA], dtype=object),
            "label": np.asarray(["tumor"], dtype=object),
        }

    Because ``pd.NA`` is treated as a missing feature value, Harpy's status text
    omits the identity feature for a newly drawn unsaved row instead of showing
    ``instance_id: <NA>``.

    Existing ``layer.features`` rows are left untouched. Newly drawn rows receive
    a missing source ID while they are unsaved, so Harpy's status text has no
    identity feature to show for those rows. The save path later assigns fresh
    stable IDs such as ``__annotation_2``.
    """

    def __init__(self) -> None:
        self._layer: Shapes | None = None
        self._feature_name: str | None = None
        self._connected_events: list[tuple[object, Callable[..., object]]] = []
        self._is_clearing = False

    @property
    def layer(self) -> Shapes | None:
        return self._layer

    @property
    def feature_name(self) -> str | None:
        return self._feature_name

    def attach(self, layer: Shapes, *, feature_name: str) -> None:
        feature_name = str(feature_name).strip()
        if not feature_name:
            raise ValueError("Annotation identity feature name must be a non-empty string.")

        if self._layer is not layer or self._feature_name != feature_name:
            self.disconnect()
            self._layer = layer
            self._feature_name = feature_name
            self._connect_layer_event(layer, "current_properties")
            # Today `feature_defaults` writes also emit `current_properties`.
            # Listen to both public default-related events anyway so this guard
            # keeps working if napari decouples those APIs later.
            self._connect_layer_event(layer, "feature_defaults")

        self._clear_identity_feature_default()

    def disconnect(self) -> None:
        for event, callback in self._connected_events:
            disconnect = getattr(event, "disconnect", None)
            if not callable(disconnect):
                continue
            try:
                disconnect(callback)
            except (RuntimeError, ValueError):
                pass

        self._connected_events = []
        self._layer = None
        self._feature_name = None
        self._is_clearing = False

    def _connect_layer_event(self, layer: Shapes, event_name: str) -> None:
        event = getattr(getattr(layer, "events", None), event_name, None)
        connect = getattr(event, "connect", None)
        if not callable(connect):
            return

        callback = self._on_identity_feature_default_changed
        connect(callback)
        self._connected_events.append((event, callback))

    def _on_identity_feature_default_changed(self, _event: object | None = None) -> None:
        if self._is_clearing:
            return
        self._clear_identity_feature_default()

    def _clear_identity_feature_default(self) -> None:
        layer = self._layer
        feature_name = self._feature_name
        if layer is None or feature_name is None:
            return
        if self._identity_feature_default_is_missing(layer, feature_name):
            return

        current_properties = dict(layer.current_properties)
        current_properties[feature_name] = np.asarray([pd.NA], dtype=object)
        self._is_clearing = True
        try:
            # Setting `current_properties` can write through to selected rows in
            # `layer.features`; block that so only future-row defaults change.
            with layer.block_update_properties():
                layer.current_properties = current_properties
        finally:
            self._is_clearing = False

    def _identity_feature_default_is_missing(self, layer: Shapes, feature_name: str) -> bool:
        return self._current_identity_feature_default_is_missing(
            layer,
            feature_name,
        ) and self._feature_defaults_identity_value_is_missing(layer, feature_name)

    def _current_identity_feature_default_is_missing(self, layer: Shapes, feature_name: str) -> bool:
        current_properties = dict(layer.current_properties)
        if feature_name not in current_properties:
            return feature_name not in layer.features.columns
        return _is_missing_annotation_feature_value(current_properties[feature_name])

    def _feature_defaults_identity_value_is_missing(self, layer: Shapes, feature_name: str) -> bool:
        feature_defaults = layer.feature_defaults
        if feature_name not in feature_defaults.columns:
            return True
        if feature_defaults.empty:
            return True
        return _is_missing_annotation_feature_value(feature_defaults[feature_name].iloc[0])


def _is_missing_annotation_feature_value(value: object) -> bool:
    if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
        values = np.asarray(value, dtype=object).ravel()
        if len(values) == 0:
            return True
        value = values[0]
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
