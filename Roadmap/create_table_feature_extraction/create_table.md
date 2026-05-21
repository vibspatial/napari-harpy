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

The widget is currently existing-table-only.

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
  stores `FeatureExtractionRequest.table_name` as the existing table to update.
- The controller's `can_calculate` path requires `_get_bound_table() is not None`,
  so Harpy's create-table mode can never be represented today.
- `_run_feature_extraction_job(...)` currently calls
  `hp.tb.add_feature_matrix(...)` with `table_name=job.request.table_name` only.

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
    overwrite_feature_key: bool = False

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
- keep `FeatureExtractionJob.table_name` returning `request.table_name` for
  event/status use;
- refactor controller table validation:
  - existing-table mode still uses `get_table(...)`;
  - create-table mode validates the requested new table name and collision state without
    reading a table;
- keep the old `bind(...)` and `bind_batch(...)` signatures working by adding
  optional keyword-only `create_table=False`.

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

Work items:

- update `_run_feature_extraction_job(...)` to pass:
  - `table_name=job.request.harpy_table_name`;
  - `output_table_name=job.request.harpy_output_table_name`;
  - `overwrite_output_table=False`;
- keep `FeatureExtractionResult.table_name` as the napari-harpy target table name,
  i.e. `job.request.table_name` in both modes;
- keep `change_kind="created"` for create-table jobs;
- keep `change_kind="updated"` only for existing-table feature-key replacement.

Acceptance tests:

- fake-Harpy test for `create_table=True` captures `table_name=None` and
  `output_table_name="new_table"`;
- fake-Harpy test captures `overwrite_output_table=False`;
- result/event table name is `"new_table"`;
- existing-table worker tests still capture `output_table_name is None` or omit it
  according to the chosen implementation.

### Slice 3: Add The Widget Dropdown Sentinel And Name Field

Add the UI without changing the labels/image staging behavior.

Work items:

- rename `_table_names` internally to something like
  `_eligible_existing_table_names` to avoid mixing real table names with the
  sentinel option;
- add constants for the combo sentinel, for example:
  - display text: `Create table...`;
  - item data: a private sentinel object/string;
- add `self.new_table_name_line_edit` below the `Table` row and hide it by
  default;
- add a helper that proposes the default create-table name:
  - return `features_table` when it is not already in `sdata.tables`;
  - otherwise return `features_table_{uuid.uuid4()}`;
- add state:
  - `_selected_table_mode: Literal["existing", "create"] | None`;
  - `_new_table_name: str | None`;
- keep `selected_table_name` meaning "selected existing table name";
- add a property such as `selected_new_table_name` that returns the trimmed new
  table name in create mode;
- update `_refresh_table_names()` to populate eligible existing tables plus the
  sentinel when the staged labels batch is complete, with `Create table...` as
  the last combo item;
- update `_on_table_changed(...)` to show/hide the new-name field and rebind;
- add `_on_new_table_name_changed(...)` to rebind on text edits.

Acceptance tests:

- before a labels batch is complete, the table controls remain inactive as they
  are today;
- with one eligible existing table, the combo contains `table` and
  `Create table...`, with `table` selected by default;
- with no eligible existing tables, the combo contains `Create table...`, it is
  enabled, and the new-name field is visible;
- when create mode first becomes active, the new-name field is prefilled with
  `features_table` or a `features_table_<uuid>` fallback when needed;
- switching away from create mode hides the new-name field without losing the
  typed value;
- `selected_table_name` remains `None` in create mode, while
  `selected_new_table_name` returns the trimmed new table name.

### Slice 4: Bind Create-Table Widget State Into The Controller

This is the point where the UI path becomes calculable.

Work items:

- update `_validate_selected_table_binding(...)` so it only validates existing
  table selections;
- add `_validate_create_table_target(...)` for the new table name;
- update `_bind_current_selection()` and `_expected_controller_binding_state()`
  to pass either:
  - existing mode: `table_name=<existing>, create_table=False`;
  - create mode: `table_name=<new>, create_table=True`;
- update `_selection_status_table_blocker()` and
  `_get_calculate_button_blocking_reason()` so create mode is blocked only by an
  invalid or missing new table name, not by the absence of an existing table;
- update status-card wording from "No table annotates..." to something more
  actionable when create mode is available, for example:
  - missing name: "Enter a new table name.";
  - invalid name: "Choose a valid table name. ...";
  - collision: "Table `<name>` already exists. Choose a different table name."

Acceptance tests:

- a labels-only morphology run with no linked table becomes enabled after a valid
  new table name is entered;
- the same scenario stays disabled while the new table name is empty or invalid;
- a colliding new table name stays disabled and is explained by the primary
  status card plus the calculate-button tooltip;
- existing-table table-binding errors still appear in the status tooltip;
- intensity feature image requirements still apply unchanged in create mode.

### Slice 5: Keep Existing Overwrite Behavior Correct

Existing feature-key overwrite checks currently assume an existing selected
table. Keep that path isolated from create-table mode.

This matters for the common "clicked Calculate twice" flow. The first click with
`create_table=True` creates `request.table_name` and writes the feature matrix.
The success path must then promote that new table into existing-table mode. The
second click should see `sdata.tables[new_table].obsm[feature_key]`, show the
existing feature-key overwrite confirmation, and call Harpy with
`table_name="new_table"`, `create_table=False`, and `overwrite_feature_key=True`.

Work items:

- in `_calculate_feature_matrix()`, only inspect `table.obsm` and prompt for
  feature-key overwrite in existing-table mode;
- in create-table mode, never send `overwrite_feature_key=True`;
- block table-name collisions before calculation;
- do not add an `overwrite_output_table=True` path; table collisions are a hard
  block.

Acceptance tests:

- create-table mode does not call `_prompt_overwrite_feature_key_confirmation`;
- existing-table mode still prompts when `.obsm[feature_key]` already exists;
- after a successful create-table run, clicking `Calculate` again follows the
  existing-table overwrite prompt path rather than the create-table collision
  path;
- a new table name colliding with any existing `sdata.tables` key keeps
  Calculate disabled with a clear status-card message and matching tooltip.

### Slice 6: Refresh Selection After The New Table Is Created

After Harpy creates the table, the widget should move from create mode to the new
authoritative existing table.

Use the controller's local table-refresh hook for this, not the shared
`feature_matrix_written` app-state event. That hook already exists for local
table-context refreshes, but today it has no payload:

```python
on_table_state_changed: Callable[[], None] | None
```

For create-table mode, change it to carry the completed
`FeatureExtractionResult`:

```python
on_table_state_changed: Callable[[FeatureExtractionResult], None] | None
```

The controller should call this before emitting the shared
`FeatureMatrixWrittenEvent`, so the owning Feature Extraction widget can refresh
its table choices, prefer the newly created table, and rebind before downstream
widgets consume the shared semantic event.

Work items:

- update `FeatureExtractionController._notify_table_state_changed(...)` to accept
  and forward `FeatureExtractionResult`;
- update the existing controller comment there: this is no longer only a future
  "may create or relink a table" hook; it is the mechanism that promotes a
  successful create-table run into normal existing-table mode;
- update `FeatureExtractionWidget._on_controller_table_state_changed(...)` to
  accept the result and treat `result.table_name` as the one-shot preferred table
  for the next refresh;
- make the success refresh prefer `event.table_name` or the job result table name
  when repopulating table options;
- after creation, select the newly created table in the `Table` combo;
- hide the `New table name` field once the new table is selected as an existing
  table;
- rebind the controller after that promotion with `table_name=<new_table>` and
  `create_table=False`;
- emit the existing `FeatureMatrixWrittenEvent` with the new table name so dirty
  state and downstream widgets see the table update;
- update the object-classification widget listener so any
  `feature_matrix_written` event for the same `sdata` refreshes table names, not
  only events for the currently selected table;
- preserve the object-classification widget's selected table during that refresh
  whenever it is still valid. If the classifier already has a selected table,
  creating a new table elsewhere must not silently switch it to the first table
  or to the newly created table;
- auto-select the newly created table in object classification only when no
  table was selected before the refresh, no other table was available/selected,
  and the new table annotates the currently selected labels element.

Acceptance tests:

- after a fake successful create-table result, `_refresh_table_names()` selects
  the new table;
- after that refresh, `selected_table_name == "new_table"` and
  `selected_new_table_name is None`;
- `HarpyAppState.is_table_dirty(sdata, "new_table")` becomes `True`;
- object-classification table choices can discover the new table in the same
  session when its selected labels element matches;
- if object classification already had `selected_table_name == "old_table"` and
  `old_table` is still valid, it remains selected after the create-table event;
- object classification does not silently fall back to the first table or switch
  to `new_table` when a different valid table was already selected;
- if object classification had no selected/available table and `new_table`
  annotates the currently selected labels element, it auto-selects `new_table`
  after the event refresh.

### Slice 7: Refresh Viewer Linked Tables From Feature-Matrix Events

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

### Slice 8: Polish And Documentation

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
