# Create Table During Feature Extraction

## Goal

Allow the Feature Extraction widget to run when no existing table annotates all
currently staged labels elements.

The intended UI is:

- the `Table` dropdown keeps showing existing eligible annotating tables;
- the same dropdown also offers `Create table...`;
- when `Create table...` is selected, a text field appears where the user enters
  the new table name;
- calculation calls Harpy in table-creation mode, then the new table becomes the
  authoritative table in `sdata.tables`.

## Current State

The widget is currently existing-table-only, while the controller and worker now
have the explicit create-table target mode from slices 1 and 2.

- [src/napari_harpy/widgets/feature_extraction/widget.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/feature_extraction/widget.py:1661)
  computes eligible table names by intersecting
  `get_annotating_table_names(sdata, labels_name)` across the staged labels
  batch.
- If that intersection is empty, the `Table` combo is disabled,
  `selected_table_name` becomes `None`, and calculation is blocked with
  "No table annotates all staged labels elements."
- Existing table selections are validated with
  `validate_table_annotation_coverage(...)` and
  `validate_table_region_instance_ids(...)`.
- [src/napari_harpy/widgets/feature_extraction/controller.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/feature_extraction/controller.py:28)
  stores `FeatureExtractionRequest.table_name` as the napari-harpy target table
  and uses `FeatureExtractionRequest.create_table` to distinguish existing-table
  updates from create-table writes.
- The controller's `can_calculate` path now accepts either a valid existing table
  target or a valid create-table target.
- `_run_feature_extraction_job(...)` currently calls
  `hp.tb.add_feature_matrix(...)` with `table_name` and `output_table_name`
  derived from the request adapter properties, and always passes
  `overwrite_output_table=False`.

Harpy already supports the target behavior:

```python
hp.tb.add_feature_matrix(
    sdata=sdata,
    labels_name=...,
    image_name=...,
    table_name=None,
    output_table_name="new_table",
    feature_key="features",
    features=[...],
    channels=...,
    overwrite_output_table=False,
    overwrite_feature_key=False,
    to_coordinate_system=...,
)
```

Important Harpy constraints:

- `output_table_name` is required when `table_name is None`;
- `output_table_name` is not allowed when updating an existing table;
- `overwrite_feature_key=True` is only valid for existing-table updates;
- if `output_table_name` already exists and `overwrite_output_table=False`,
  Harpy raises a collision error.

## Proposed UX Rules

- Add `Create table...` as the last option in the `Table` combo whenever the
  staged labels batch is complete enough to know which labels elements would be
  annotated.
- Preserve the current behavior for existing tables: if at least one eligible
  table exists, keep auto-selecting the first eligible table unless the previous
  selection is still valid.
- If no eligible existing table exists, auto-select `Create table...` so the
  user can continue immediately.
- Show a `New table name` `QLineEdit` only while `Create table...` is selected.
- When create mode becomes active, suggest `features_table` as the default new
  table name. If `features_table` already exists in `sdata.tables`, suggest
  `features_table_{uuid.uuid4()}` instead.
- Validate the new table name before enabling calculation:
  - non-empty after trimming;
  - valid SpatialData element name, using the same naming helper as the feature
    matrix key;
  - no collision with any existing `sdata.tables` key.
- Report create-table validation failures through the existing Feature
  Extraction status-card pattern. The calculate button tooltip should mirror the
  same blocking reason, and the `New table name` field can receive a tooltip, but
  the status card should be the primary user-facing explanation.
- Do not support overwriting existing tables for this feature. A table-name
  collision remains a hard validation block with a clear status-card message.
  Supporting table overwrites would require a broader design change around how
  feature matrices are written into `.obsm`.
- Treat `Create table...` as a one-shot mode. After successful creation, the UI
  should switch to the newly created table as a normal existing-table selection.
  A second click on `Calculate` should therefore use the existing-table
  feature-key overwrite prompt instead of trying to create the table again.

For a table-name collision, use wording along these lines:

> Table `my_table` already exists. Choose a different table name.

## Implementation Slices

### Slice 1: Add Explicit Table Target Mode

Carry the table target explicitly through the controller request/binding state so
the controller can distinguish an existing table from a table that does not exist
yet.

Use one napari-harpy target name plus a mode flag. This reads more naturally in
the widget/controller than carrying Harpy's `table_name`/`output_table_name`
pair everywhere:

```python
@dataclass(frozen=True)
class FeatureExtractionRequest:
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
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        return self.table_name if self.create_table else None
```

`FeatureExtractionBindingState` should mirror the target mode, but remain able
to represent incomplete UI state:

```python
@dataclass(frozen=True)
class FeatureExtractionBindingState:
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
        if self.table_name is None:
            return None
        return None if self.create_table else self.table_name

    @property
    def harpy_output_table_name(self) -> str | None:
        if self.table_name is None:
            return None
        return self.table_name if self.create_table else None
```

Work items:

- update `FeatureExtractionRequest` to carry `table_name: str`,
  `create_table: bool`, and small Harpy adapter properties such as
  `harpy_table_name` and `harpy_output_table_name`;
- add `FeatureExtractionRequest.__post_init__` validation:
  - reject an empty `table_name`;
  - reject `create_table=True` with `overwrite_feature_key=True`;
- update `FeatureExtractionBindingState` similarly, with `table_name: str | None`
  and `create_table: bool`;
- add `FeatureExtractionBindingState.__post_init__` validation that rejects an
  empty non-`None` table name, but still allows `table_name=None` because binding
  state can represent incomplete UI state;
- do not keep overwrite-feature-key state in controller binding; overwrite is a
  per-click calculation decision after the widget confirmation prompt;
- keep `FeatureExtractionController.calculate(..., overwrite_feature_key=False)`
  as the only controller entry point for feature-key overwrite intent;
- keep `_prepare_feature_extraction_job(..., overwrite_feature_key: bool)` on a
  resolved boolean only, with no `None`/tri-state value entering job creation;
- keep `FeatureExtractionJob.table_name` returning `request.table_name` for
  event/status use;
- refactor controller table validation:
  - existing-table mode still uses `get_table(...)`;
  - create-table mode validates the requested new table name and collision state without
    reading a table;
- keep `bind(...)` and `bind_batch(...)` focused on selection binding by adding
  optional keyword-only `create_table=False`; do not pass overwrite state through
  binding.

Acceptance tests:

- existing-table controller tests still pass unchanged;
- binding with `table_name="new_table", create_table=True` can become ready;
- `FeatureExtractionRequest(table_name="", create_table=True, ...)` raises a
  `ValueError`;
- `FeatureExtractionRequest(table_name="new_table", create_table=True, overwrite_feature_key=True, ...)`
  raises a `ValueError`;
- `FeatureExtractionBindingState(table_name=None, create_table=False, ...)` and
  `FeatureExtractionBindingState(table_name=None, create_table=True, ...)`
  remain valid for incomplete UI state;
- create-table mode rejects an empty, invalid, or colliding new table name;
- create-table mode rejects `overwrite_feature_key=True`;
- existing-table mode still rejects missing/nonexistent tables.

### Slice 2: Wire The Worker To Harpy Creation Mode

Once the controller can prepare a create-table job, pass the new fields into
Harpy.

The worker should call Harpy exactly through the request adapter properties:

```python
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
    overwrite_output_table=False,
    overwrite_feature_key=job.request.overwrite_feature_key,
    to_coordinate_system=_resolve_harpy_coordinate_system_parameter(triplets),
)
```

Work items:

- update `_run_feature_extraction_job(...)` to pass:
  - `table_name=job.request.harpy_table_name`;
  - `output_table_name=job.request.harpy_output_table_name`;
  - `overwrite_output_table=False`;
- keep `FeatureExtractionResult.table_name` as the napari-harpy target table name,
  i.e. `job.request.table_name` in both modes;
- keep `change_kind="created"` for any successful write that creates a new
  feature matrix key. In create-table mode this means "created table plus
  feature matrix"; in existing-table mode it means "created feature matrix in an
  existing table";
- keep `change_kind="updated"` only for existing-table feature-key replacement.

Acceptance tests:

- fake-Harpy test for `create_table=True` captures `table_name=None` and
  `output_table_name="new_table"`;
- fake-Harpy test captures `overwrite_output_table=False`;
- result/event table name is `"new_table"`;
- create-table result emits `change_kind="created"` using the same semantics as a
  newly created feature matrix key;
- existing-table result emits `change_kind="created"` when the feature key did
  not previously exist and `change_kind="updated"` when the feature key is
  replaced;
- existing-table worker tests still capture `output_table_name is None` or omit it
  according to the chosen implementation.

### Slice 3: Add The Widget Dropdown Sentinel And Name Field

Status: implemented.

Add the UI without changing the labels/image staging behavior and without yet
making create-table mode calculable. Slice 3 should make the widget able to
represent the user's table-target choice. Slice 4 will bind that create-table
choice into the controller.

Current table-selection code assumes that the combo index maps directly into
`_table_names`. This slice should remove that assumption. Once the combo can
contain both real tables and a sentinel option, selection must be derived from
`itemData(...)`, not from the visible row index.

Suggested constants/types:

```python
import uuid
from dataclasses import dataclass
from enum import Enum

_CREATE_TABLE_OPTION_TEXT = "Create table..."
_DEFAULT_NEW_TABLE_NAME = "features_table"


class _FeatureExtractionTableComboMode(Enum):
    EXISTING = "existing"
    CREATE_TABLE = "create_table"


@dataclass(frozen=True)
class _FeatureExtractionTableComboData:
    mode: _FeatureExtractionTableComboMode
    table_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, _FeatureExtractionTableComboMode):
            raise ValueError("Unknown table combo mode.")

        if self.mode is _FeatureExtractionTableComboMode.EXISTING:
            if self.table_name is None or not self.table_name.strip():
                raise ValueError("Existing table combo data requires a table name.")
            return

        if (
            self.mode is _FeatureExtractionTableComboMode.CREATE_TABLE
            and self.table_name is not None
        ):
            raise ValueError("Create-table combo data cannot carry a table name.")

    @classmethod
    def existing(cls, table_name: str) -> "_FeatureExtractionTableComboData":
        return cls(_FeatureExtractionTableComboMode.EXISTING, table_name)

    @classmethod
    def create_table(cls) -> "_FeatureExtractionTableComboData":
        return cls(_FeatureExtractionTableComboMode.CREATE_TABLE)
```

Real table combo rows should use
`_FeatureExtractionTableComboData.existing(table_name)`, while the sentinel row
should use `_FeatureExtractionTableComboData.create_table()`. That keeps the
string tags in one enum and avoids any possible collision between a real table
name and the sentinel.

State shape:

```python
# Existing real tables that annotate every currently staged labels element.
self._eligible_existing_table_names: list[str] = []
self._create_table: bool = False
self._selected_table_name: str | None = None
self._new_table_name: str | None = None
```

Property semantics:

- `selected_table_name` remains the selected existing table name only.
- `selected_table_name is None` while `Create table...` is selected.
- add `selected_table_mode` as a derived read-only convenience:
  - return `"create"` when `_create_table` is `True`;
  - return `"existing"` when `_create_table` is `False` and
    `_selected_table_name is not None`;
  - otherwise return `None`;
- add `selected_new_table_name`, returning the trimmed new table name only in
  create mode, otherwise `None`.

Work items:

- rename `_table_names` to `_eligible_existing_table_names` to keep real table
  names separate from the sentinel option;
- add the combo sentinel constants and structured item-data model;
- add `self.new_table_name_line_edit` below the `Table` row:
  - label: `New table name`;
  - object name: `feature_extraction_new_table_name_line_edit`;
  - same input stylesheet as the feature-matrix-key field;
  - hidden by default;
  - connected to `_on_new_table_name_changed(...)`;
  - placement in the shared form:
    - `Table`;
    - `New table name`;
    - `Feature matrix key`;
- add a helper such as `_suggest_new_table_name()`:
  - return `features_table` when it is not already in `sdata.tables`;
  - otherwise return `features_table_{uuid.uuid4()}`;
  - only consult `sdata.tables`, not eligible-table filtering;
- add a helper such as `_ensure_new_table_name_suggestion()`:
  - if `_new_table_name` already has a non-empty value, preserve it;
  - otherwise set `_new_table_name` and the line edit text to the suggested
    default;
- add a helper such as `_set_create_table_name_controls_visible(is_visible)` to
  show/hide the label and line edit together;
- update `_refresh_table_names()` to populate eligible existing tables plus the
  sentinel when the staged labels batch is complete, with `Create table...` as
  the last combo item;
- when the staged labels batch is not complete enough to know labels elements,
  keep the current inactive behavior: no sentinel, combo disabled, create-name
  field hidden;
- update selection restoration in `_refresh_table_names()`:
  - if previous mode was `"create"` and the sentinel is available, keep create
    mode selected and preserve the typed new table name;
  - else, if the previous existing table is still eligible, keep it selected;
  - else, if at least one eligible existing table exists, select the first
    existing table;
  - else, if the sentinel is available, select `Create table...`;
  - else select no table;
- replace `_set_selected_table_name(index)` with a target-aware helper, for
  example `_set_selected_table_target_from_combo(index)`, that reads
  `self.table_combo.itemData(index)` and updates:
  - `_create_table`;
  - `_selected_table_name`;
  - `_new_table_name`/line-edit visibility when create mode is selected;
- update `_on_table_changed(...)` to call the target-aware helper, then
  `_bind_current_selection()`;
- add `_on_new_table_name_changed(...)`:
  - store the text in `_new_table_name`;
  - call `_bind_current_selection()` so binding and selection status refresh
    through the normal path;
  - Slice 3 may still bind no existing table while create mode is selected;
    Slice 4 will change the controller binding to `create_table=True`.

Acceptance tests:

- before a labels batch is complete, the table controls remain inactive as they
  are today: no sentinel, combo disabled, and the new-name field hidden;
- with one eligible existing table, the combo contains `table` and
  `Create table...`, with `table` selected by default;
- with no eligible existing tables, the combo contains `Create table...`, it is
  enabled, and the new-name field is visible;
- when create mode first becomes active, the new-name field is prefilled with
  `features_table` or a `features_table_<uuid>` fallback when needed;
- the UUID fallback is deterministic in tests by monkeypatching `uuid.uuid4`;
- switching away from create mode hides the new-name field without losing the
  typed value;
- switching back to create mode restores the previously typed value instead of
  replacing it with a fresh suggestion;
- selecting `Create table...` sets `selected_table_mode == "create"`;
- `selected_table_name` remains `None` in create mode, while
  `selected_new_table_name` returns the trimmed new table name;
- selecting an existing table sets `selected_table_mode == "existing"`,
  `selected_table_name == <table>`, and `selected_new_table_name is None`;
- `bind_batch(...)` still receives the existing-table-only target in this slice:
  create-table controller binding is intentionally deferred to Slice 4.

### Slice 4: Bind Create-Table Widget State Into The Controller

Status: implemented.

This is the point where the UI path becomes calculable.

The controller is already ready for this slice: `bind_batch(...)` accepts
`create_table=True`, `FeatureExtractionBindingState` mirrors that mode, and
`FeatureExtractionController.can_calculate` validates create-table names and
collisions as a final safety net. Slice 4 should make the widget pass the
create-table target into that existing controller path.

Terminology:

- existing-table mode targets `selected_table_name` with `create_table=False`;
- create-table mode targets `selected_new_table_name` with `create_table=True`;
- `selected_table_name` should remain `None` in create mode, because it means
  "selected existing table" at the widget API level.

Replace `_table_binding_error` with a stored table-target snapshot in this
slice. The old name was accurate when every target was an existing table; after
this slice, the target state can describe either an existing table binding or a
create-table target.

```python
@dataclass(frozen=True)
class _FeatureExtractionTableTargetState:
    table_name: str | None
    create_table: bool
    error_text: str | None = None

    @property
    def bound_table_name(self) -> str | None:
        return self.table_name if self.error_text is None else None

    @property
    def is_ready(self) -> bool:
        return self.bound_table_name is not None
```

Work items:

- replace widget state and call sites from `_table_binding_error` with
  `_table_target_state`;
- keep `_validate_selected_table_binding(...)` existing-table-only:
  - return `None` in create mode;
  - continue using `validate_table_annotation_coverage(...)` and
    `validate_table_region_instance_ids(...)` only for existing selected tables;
- add `_validate_create_table_target(...)` for the new table name:
  - return `"Enter a new table name."` when `selected_new_table_name is None`;
  - validate the non-empty name with `normalize_spatialdata_name(..., "table name")`;
  - reject collisions with any key in `selected_spatialdata.tables`;
  - use the primary message
    ``Table `<name>` already exists. Choose a different table name.`` for
    collisions;
  - do not check table annotation coverage or region instance ids, because the
    table does not exist yet;
- add a small table-target resolver, either as a helper or inline in
  `_bind_current_selection()`:

```python
if self._create_table:
    return _FeatureExtractionTableTargetState(
        table_name=self.selected_new_table_name,
        create_table=True,
        error_text=self._validate_create_table_target(),
    )
else:
    return _FeatureExtractionTableTargetState(
        table_name=self.selected_table_name,
        create_table=False,
        error_text=self._validate_selected_table_binding(staged_batch_state),
    )
```

- update `_bind_current_selection()` and `_expected_controller_binding_state()`
  to pass either:
  - existing mode: `table_name=<existing>, create_table=False`;
  - create mode: `table_name=<new>, create_table=True`;
- `_bind_current_selection()` should store the resolved
  `_table_target_state`, and `_expected_controller_binding_state()` should read
  that stored snapshot rather than recomputing target validation;
- only pass staged triplets to the controller when:
  - the staged batch is bindable;
  - the resolved table target is ready;
- when the create-table name changes, `_on_new_table_name_changed(...)` should
  continue calling `_bind_current_selection()`, so the controller binding and
  status cards update immediately as the user types;
- update `_selection_status_table_blocker()` and
  `_get_calculate_button_blocking_reason()` so create mode is blocked only by an
  invalid or missing new table name, not by the absence of an existing table;
- extend the status-card table blocker model in
  `widgets/feature_extraction/status_card.py` so create-table errors get their
  own wording instead of reusing `"no_eligible"`:
  - missing name: title `Table Not Ready`, line `Enter a new table name.`;
  - invalid name: title `Table Not Ready`, line from the validation error;
  - collision: title `Table Not Ready`, line
    ``Table `<name>` already exists. Choose a different table name.``;
- keep the calculate-button tooltip synchronized with the same table-target
  validation message;
- keep `_calculate_feature_matrix()` from inspecting `.obsm` in create mode.
  Since `selected_table_name is None` in create mode, the existing overwrite
  prompt path should remain existing-table-only, and the controller should be
  called with `calculate(overwrite_feature_key=False)` for the normal
  create-table click.

Acceptance tests:

- a labels-only morphology run with no linked table becomes enabled after a valid
  new table name is entered;
- the same scenario stays disabled while the new table name is empty or invalid;
- a colliding new table name stays disabled and is explained by the primary
  status card plus the calculate-button tooltip;
- in create mode, `bind_batch(...)` receives the staged triplets,
  `table_name=<selected_new_table_name>`, and `create_table=True`;
- in create mode with an invalid or missing new table name, `bind_batch(...)`
  does not receive a calculable staged request;
- `_expected_controller_binding_state()` matches the controller binding for both
  existing-table and create-table modes;
- existing-table table-binding errors still appear in the status tooltip;
- intensity feature image requirements still apply unchanged in create mode.
- clicking `Calculate` in create mode calls
  `controller.calculate(overwrite_feature_key=False)` without showing the
  existing-feature-key overwrite dialog.

### Slice 5: Keep Existing Overwrite Behavior Correct

Status: implemented.

Existing feature-key overwrite checks currently assume an existing selected
table. Keep that path isolated from create-table mode, and preserve it once a
newly created table is promoted back into existing-table mode.

This matters for the common "clicked Calculate twice" flow. The first click with
`create_table=True` creates `request.table_name` and writes the feature matrix.
The success path must then promote that new table into existing-table mode. The
second click should see `sdata.tables[new_table].obsm[feature_key]`, show the
existing feature-key overwrite confirmation, and call Harpy with
`table_name="new_table"`, `create_table=False`, and `overwrite_feature_key=True`.

Current code contains this behavior:

- `_calculate_feature_matrix()` only checks `.obsm` when
  `selected_table_name is not None`, `_table_target_state.error_text is None`,
  and `_create_table is False`;
- create-table mode calls
  `controller.calculate(overwrite_feature_key=False)` without showing the
  overwrite dialog;
- `FeatureExtractionController._prepare_feature_extraction_job(...)` still has a
  defensive guard that rejects `create_table=True` with
  `overwrite_feature_key=True`;
- table-name collisions are hard-blocked before calculation, and
  `_run_feature_extraction_job(...)` always passes
  `overwrite_output_table=False`.

Slice 6 supplies the promotion step: after a successful create-table run, the
widget is in existing-table mode before the next click, so the existing
`.obsm[feature_key]` overwrite confirmation runs normally.

Work items:

- keep `_calculate_feature_matrix()` existing-table-only for `.obsm` inspection:
  - require `_create_table is False`;
  - require a non-`None` `selected_table_name`;
  - require no table-target error;
- keep create-table clicks one-way:
  - never show `_prompt_overwrite_feature_key_confirmation(...)`;
  - call `controller.calculate(overwrite_feature_key=False)`;
- keep the controller's `create_table and overwrite_feature_key` guard as a
  defensive backstop;
- keep table-name collisions as hard blocks. Do not add an
  `overwrite_output_table=True` path;
- after Slice 6 promotes a created table to existing-table mode, ensure the next
  click uses the existing-table overwrite path:
  - inspect `sdata.tables[new_table].obsm`;
  - prompt if `.obsm[feature_key]` exists;
  - on confirmation, call
    `controller.calculate(overwrite_feature_key=True)`;
  - the controller request should then have `table_name="new_table"`,
    `create_table=False`, and `overwrite_feature_key=True`.

Acceptance tests:

- create-table mode does not call `_prompt_overwrite_feature_key_confirmation`,
  even if another existing table contains the same feature key;
- create-table mode calls `controller.calculate(overwrite_feature_key=False)`;
- existing-table mode still prompts when `.obsm[feature_key]` already exists;
- if the existing-table overwrite prompt is cancelled, calculation is not
  launched;
- if the existing-table overwrite prompt is confirmed, calculation is launched
  with `overwrite_feature_key=True`;
- the controller still rejects `calculate(overwrite_feature_key=True)` while
  bound in create-table mode;
- a new table name colliding with any existing `sdata.tables` key keeps
  Calculate disabled with a clear status-card message and matching tooltip;
- after a successful create-table run, clicking `Calculate` again follows the
  existing-table overwrite prompt path rather than the create-table collision
  path;
- after promotion, the second-click test should assert
  `create_table is False` in the controller binding before triggering the
  overwrite prompt.

### Slice 6: Refresh Selection After The New Table Is Created

Status: implemented.

After Harpy creates the table, the widget should move from create mode to the new
authoritative existing table.

Without an explicit preference, the current widget selection restoration would
try to preserve the previous create-table sentinel after refresh. That would
leave the widget in create mode with a now-colliding table name. Slice 6 should
therefore promote the successful target table into normal existing-table mode as
a one-shot post-success preference.

Use the controller's local table-refresh hook for this, not the shared
`feature_matrix_written` app-state event. That hook already exists for local
table-context refreshes, but today it has no payload:

```python
on_table_state_changed: Callable[[], None] | None
```

Change it to always carry the completed `FeatureExtractionResult`, for both
existing-table writes and create-table writes:

```python
on_table_state_changed: Callable[[FeatureExtractionResult], None] | None
```

The controller should call this before emitting the shared
`FeatureMatrixWrittenEvent`, so the owning Feature Extraction widget can refresh
its table choices, prefer the newly created table, and rebind before downstream
widgets consume the shared semantic event.
For create-table success, the widget uses `result.table_name` as a one-shot
preferred existing-table selection. For existing-table success, this remains a
targeted refresh/rebind hook and should preserve the current existing table
selection.

Work items:

- update `FeatureExtractionController._notify_table_state_changed(...)` to accept
  and forward `FeatureExtractionResult`;
- update `FeatureExtractionController.__init__(...)` type annotations for
  `on_table_state_changed`;
- update `_on_worker_returned(...)` to call
  `_notify_table_state_changed(result)` before
  `_notify_feature_matrix_written(result)`;
- update the existing controller comment there: this is no longer only a future
  "may create or relink a table" hook; it is the mechanism that promotes a
  successful create-table run into normal existing-table mode;
- update `FeatureExtractionWidget._on_controller_table_state_changed(...)` to
  accept the result and treat `result.table_name` as the one-shot preferred table
  for the next refresh;
- add a widget helper or small state field for the one-shot preference, for
  example `_preferred_existing_table_name_after_success: str | None`;
- update `_refresh_table_names(...)` so the restore priority is:
  - if a one-shot preferred existing table is present and eligible, select it;
  - else if previous mode was create and the sentinel is available, keep create
    mode selected;
  - else if the previous existing table is still eligible, keep it selected;
  - else if at least one eligible existing table exists, select the first;
  - else if the sentinel is available, select `Create table...`;
  - else select no table;
- clear the one-shot preferred table after one refresh attempt, regardless of
  whether it was found, so ordinary user selection rules resume afterwards;
- after creation, select the newly created table in the `Table` combo as an
  existing-table row;
- hide the `New table name` field once the new table is selected as an existing
  table;
- rebind the controller after that promotion with `table_name=<new_table>` and
  `create_table=False`;
- after promotion, preserve the typed `_new_table_name` value only as inactive
  create-mode memory; it must not keep the widget in create mode;
- emit the existing `FeatureMatrixWrittenEvent` with the new table name so dirty
  state and downstream widgets see the table update.

Acceptance tests:

- the controller table-state callback receives the `FeatureExtractionResult`;
- the controller still emits the local table-state callback before the shared
  `FeatureMatrixWrittenEvent`;
- after a fake successful create-table result, `_refresh_table_names()` selects
  the new table;
- after that refresh, `selected_table_name == "new_table"` and
  `selected_new_table_name is None`;
- after that refresh, `selected_table_mode == "existing"` and
  `new_table_name_line_edit` is hidden;
- after that refresh, the Feature Extraction controller binding has
  `table_name == "new_table"` and `create_table is False`;
- a second `Calculate` click after promotion checks the existing table `.obsm`
  and uses the existing feature-key overwrite confirmation path;
- `HarpyAppState.is_table_dirty(sdata, "new_table")` becomes `True`.

### Slice 7: Refresh Object Classification From Feature-Matrix Events

Status: implemented.

Object Classification should become aware that Feature Extraction may create a
new annotating table. Today
[src/napari_harpy/widgets/object_classification/widget.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/object_classification/widget.py:538)
listens to `feature_matrix_written`, but returns early when
`event.table_name != self.selected_table_name`. That means it can refresh
feature-matrix keys for the currently selected table, but it cannot discover a
newly created table.

This slice should keep the existing same-table feature-matrix behavior while
adding a same-`sdata` table-list refresh path.

Keep this as an `ObjectClassificationWidget` responsibility. The widget owns the
table and feature-matrix combo boxes, while the classifier, annotation, styling,
and persistence controllers should continue to receive the already-resolved
widget selection through `_bind_current_selection(...)`.

Implementation shape:

- extend `_refresh_table_names(...)` with an optional one-shot preference:

  ```python
  def _refresh_table_names(
      self,
      *,
      preferred_table_name: str | None = None,
  ) -> None:
      ...
  ```

- restore priority should be:
  - if the previous selected table is still present, keep it;
  - else if `preferred_table_name` is present and eligible, select it;
  - else if any eligible table exists, select the first;
  - else clear the table selection;
- the feature-matrix event handler should only pass `preferred_table_name` for
  the "new table became available and there was no prior table context" case.
  In practice, that means:

  ```python
  previous_table_name = self.selected_table_name
  previous_table_names = tuple(self._table_names)
  preferred_table_name = (
      event.table_name
      if previous_table_name is None and not previous_table_names
      else None
  )
  ```

  This keeps the first-table fallback behavior for ordinary refreshes, while
  preventing an event-created table from stealing selection when Object
  Classification already had a valid table.

Work items:

- update `ObjectClassificationWidget._on_feature_matrix_written(...)` so it
  ignores non-`FeatureMatrixWrittenEvent` objects and events for a different
  `sdata`, but no longer returns early only because
  `event.table_name != self.selected_table_name`;
- treat same-`sdata` feature-matrix events in two layers:
  - always refresh the eligible table list for the current labels element,
    allowing `_refresh_table_names(...)` to refresh feature-matrix keys for the
    effective selected table;
  - only run selected-feature overwrite invalidation when
    `event.table_name == self.selected_table_name`;
- compare the effective selected `(table_name, feature_key)` before and after
  the refresh. Call `_bind_current_selection()` only when the table or feature
  matrix selection changed;
- preserve the object-classification widget's selected table during that refresh
  whenever it is still valid. If the classifier already has a selected table,
  creating a new table elsewhere must not silently switch it to the first table
  or to the newly created table;
- auto-select the newly created table only when no table was selected before the
  refresh, no other table was available/selected, and the new table annotates the
  currently selected labels element;
- do not pass a classifier dirty reason when the feature-matrix event merely
  discovers or auto-selects a table. The classifier context may need rebinding,
  but no trained classifier output is being invalidated by a user selection
  change in this path;
- same-table feature-matrix updates should preserve the existing classifier
  invalidation behavior.

Acceptance tests:

- object-classification table choices can discover the new table in the same
  session when its selected labels element matches;
- object-classification ignores events for a different `sdata`;
- object-classification ignores non-`FeatureMatrixWrittenEvent` payloads;
- if object classification already had `selected_table_name == "old_table"` and
  `old_table` is still valid, it remains selected after the create-table event;
- object classification does not silently fall back to the first table or switch
  to `new_table` when a different valid table was already selected;
- if object classification had no selected/available table and `new_table`
  annotates the currently selected labels element, it auto-selects `new_table`
  after the event refresh;
- when `new_table` is auto-selected, feature-matrix keys are refreshed and the
  newly written `event.feature_key` can become the selected feature matrix;
- if a new table is discovered but not selected, the current feature-matrix combo
  still reflects the previously selected table;
- existing object-classification behavior for updates to the currently selected
  feature matrix remains intact: overwriting the selected feature key still
  invalidates the classifier.

### Slice 8: Refresh Viewer Linked Tables From Feature-Matrix Events

The Viewer widget also needs to become aware that feature extraction may create a
new annotating table. Today
[src/napari_harpy/widgets/viewer/widget.py](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/widgets/viewer/widget.py:255)
listens to `sdata_changed` and coordinate-system changes, but not to
`feature_matrix_written`.

The labels cards discover linked tables only when they are built:

- `_rebuild_labels_cards(...)` calls
  `get_annotating_table_names(sdata, labels_name)`;
- it then calls `get_table_color_source_options(...)` for each linked table;
- `_LabelsCardWidget` receives that static table/source snapshot at
  construction time.

That means a newly created table will not appear in the Viewer widget's linked
table controls until some unrelated viewer refresh happens.

Work items:

- import `FeatureMatrixWrittenEvent` in the Viewer widget and connect
  `self._app_state.feature_matrix_written` during widget initialization;
- ignore events that are not `FeatureMatrixWrittenEvent` instances or whose
  `event.sdata` is not the viewer widget's current `sdata`;
- on a same-`sdata` event, refresh labels-card table metadata for the current
  coordinate system so newly created annotating tables become available in the
  `Linked table` combos;
- preserve each labels card's selected linked table whenever it is still valid.
  A feature-extraction-created table must not silently switch a card away from an
  already selected valid table;
- if a labels card had no linked/selected table before the refresh and the new
  table annotates that labels element, auto-select the new table;
- preserve expanded/collapsed labels rows across the refresh, matching the
  existing row refresh behavior;
- do not automatically add or update napari layers. The listener refreshes
  Viewer-widget controls only; users still choose whether to add/update primary
  labels layers or colored overlays.

Implementation notes:

- The simplest first implementation can reuse `_refresh_coordinate_system_content()`
  if it snapshots and restores per-label card state before rebuilding.
- A more focused implementation could add an in-place update method to
  `_LabelsCardWidget` for table names and color-source options, but that is not
  required for the first pass.
- Existing feature-matrix writes to `.obsm` do not currently add Viewer color
  sources, because Viewer color sources come from table `obs` and `X`/`var`, not
  from `.obsm`. The event is still the right hook because the same event also
  represents newly created annotating tables.

Acceptance tests:

- a `feature_matrix_written` event for a different `sdata` leaves Viewer cards
  unchanged;
- a same-`sdata` event after creating `new_table` refreshes the matching labels
  card so `new_table` appears in its `Linked table` combo;
- if a labels card already selected `old_table` and `old_table` is still valid,
  it remains selected after the event refresh;
- if a labels card previously had no linked table and `new_table` annotates that
  labels element, the card becomes enabled and selects `new_table`;
- expanded labels rows remain expanded after the event refresh;
- no napari layer is added or updated merely because the event was received.

### Slice 9: Polish And Documentation

Work items:

- add short user-facing status text for:
  - create-table ready;
  - missing new table name;
  - invalid new table name;
  - table-name collision;
- update any roadmap or user docs that still say table creation is deferred;
- run the focused tests:

```bash
source .venv/bin/activate
pytest tests/test_feature_extraction.py tests/test_feature_extraction_widget.py tests/test_app_state.py tests/test_viewer_widget.py
```
