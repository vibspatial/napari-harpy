# `pandas.NA` Failure During Shapes Multi-Selection

## Status

Implemented and covered by a focused regression test.

## Observed Failure

The failure was observed while annotating the SpatialData shapes element
`smooth_muscle` from:

`/Users/arne.defauw/VIB/DATA/test_data/sdata_xenium_3_6_26.zarr`

Shift-clicking a shape to add it to an existing selection raised:

```text
TypeError: boolean value of NA is ambiguous
```

The relevant traceback path was:

```text
_edit_guard.py
    yielded = next(direct_drag)

napari.layers.shapes._shapes_mouse_bindings.select(...)
    layer.selected_data.add(shape_under_cursor)

napari.layers.shapes.shapes._on_selection_changed(...)
    unique_properties[k] = _unique_element(...)

napari.layers.utils.layer_utils._unique_element(...)
    np.any(array[1:] != el)
```

The Harpy frame only advances napari's native generator for mouse-press setup.
The failure occurs inside napari's selection-change callback before Harpy
classifies the gesture or processes a move.

This issue is independent of polygon geometry and the triangulation backend.

## Root Cause

Harpy keeps a source-row identity column in `layer.features`. Existing shapes
have a stable source identity, while newly drawn unsaved shapes must not inherit
an existing identity.

`_AnnotationIdentityFeatureDefaultGuard` currently represents that missing
identity with `pd.NA`:

```python
current_properties[feature_name] = np.asarray([pd.NA], dtype=object)
```

When multiple shapes are selected, napari checks whether their property values
are identical:

```python
el = array[0]
if np.any(array[1:] != el):
    return None
```

Comparing `pd.NA` with another value produces `pd.NA`, not a Boolean. NumPy then
tries to reduce that result to a Boolean and raises `TypeError`.

The traceback reaches `selected_data.add(...)`, which is napari's Shift-add
branch when another shape is already selected. The stored `smooth_muscle`
element has one row, so reaching this branch is also consistent with the live
annotation layer containing at least one newly drawn unsaved row.

The selection set is mutated before its callback raises. In an isolated
reproduction, the selection changes from `{0}` to `{0, 1}` despite the
exception. This explains why annotation can appear to continue normally.
However, that mouse press aborts before napari's native generator reaches its
first yield, so its normal press and release contract is not completed.

## Minimal Reproduction

This reproduces the failure without a viewer or SpatialData:

```python
import numpy as np
import pandas as pd
from napari.layers import Shapes

first = np.asarray(
    [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]],
    dtype=float,
)
second = first + 4.0

layer = Shapes(
    [first, second],
    shape_type="polygon",
    features=pd.DataFrame(
        {
            "index": pd.Series([0, pd.NA], dtype=object),
        }
    ),
)

layer.selected_data = {0}
layer.selected_data.add(1)
```

Expected current result:

```text
psygnal._exceptions.EmitLoopError
caused by: TypeError: boolean value of NA is ambiguous
```

After catching the exception, `set(layer.selected_data)` is `{0, 1}`.

## Possible Fixes

### 1. Use `None` for an unsaved source identity — recommended

Replace the `pd.NA` default with an object-array containing `None`:

```python
current_properties[feature_name] = np.asarray([None], dtype=object)
```

This is the smallest fix. It preserves the required semantics:

- `None` still means that a newly drawn row has no source identity;
- Harpy's missing-value helpers and save path already treat `None` as missing;
- the save path can still assign a fresh generated identity;
- napari can compare `None` during multi-selection without an ambiguous Boolean.

The minimal reproduction succeeds when `pd.NA` is replaced by `None`.

### 2. Use `np.nan`

`np.nan` also avoids the ambiguous-Boolean failure and is recognized as
missing by the current save path. It is less explicit for an object-valued
identity column and introduces a floating-point sentinel into string or integer
identity data, so `None` is preferable.

### 3. Make napari's `_unique_element(...)` missing-aware upstream

Napari could compare nullable property values without asking for the Boolean
value of `pd.NA`. That would make napari more robust for all nullable feature
columns, but it does not provide an immediate fix for Harpy's installed napari
version and would require an upstream change.

Catching `EmitLoopError` in Harpy's drag wrapper is not an appropriate fix. The
selection has already been partially processed, and swallowing the exception
would leave the native press contract incomplete. Replacing napari's selection
machinery or moving source identity out of `layer.features` would be unnecessary
for this issue.

## Implemented Fix

The identity default sentinel now uses `None` instead of `pd.NA`. A focused
regression test creates a stored polygon and an unsaved polygon using that
default, selects both rows, and verifies that napari retains the complete
selection without raising. Existing tests continue to cover missing identities,
fresh identity assignment on save, and preservation of stored identities.

No custom selection wrapper, generalized missing-value abstraction, or napari
monkeypatch is required.

## Acceptance Criteria

- Shift-adding an unsaved annotation row to a selection does not raise.
- The complete selected set is retained.
- Existing source identities remain unchanged.
- Unsaved rows still have a missing identity and do not display one in status
  text.
- Saving assigns fresh stable identities exactly as before.
