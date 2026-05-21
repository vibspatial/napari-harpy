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
  stores `FeatureExtractionRequest.table_name` as a required `str`.
- The controller's `can_calculate` path requires `_get_bound_table() is not None`,
  so `table_name=None` can never reach the Harpy worker today.
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
- `overwrite_output_table=True` is only valid for new-table creation;
- if `output_table_name` already exists and `overwrite_output_table=False`,
  Harpy raises a collision error.

## Proposed UX Rules

- Add `Create table...` as a sentinel option in the `Table` combo whenever the
  staged labels batch is complete enough to know which labels elements would be
  annotated.
- Preserve the current behavior for existing tables: if at least one eligible
  table exists, keep auto-selecting the first eligible table unless the previous
  selection is still valid.
- If no eligible existing table exists, auto-select `Create table...` so the
  user can continue immediately.
- Show a `New table name` `QLineEdit` only while `Create table...` is selected.
- Validate the new table name before enabling calculation:
  - non-empty after trimming;
  - valid SpatialData element name, using the same naming helper as the feature
    matrix key;
  - no collision with any existing `sdata.tables` key for the MVP.
- Report create-table validation failures through the existing Feature
  Extraction status-card pattern. The calculate button tooltip should mirror the
  same blocking reason, and the `New table name` field can receive a tooltip, but
  the status card should be the primary user-facing explanation.
- Do not expose table overwrite in the first implementation. Add it later as an
  explicit confirmation path if needed.

For a table-name collision, use wording along these lines:

> Table `my_table` already exists. Choose a different table name.

## Implementation Slices

### Slice 1: Add An Explicit Table Target Model

Introduce a small target model so the controller can distinguish an existing
table from a table that does not exist yet.

Possible shape:

```python
@dataclass(frozen=True)
class FeatureExtractionTableTarget:
    table_name: str | None
    output_table_name: str | None = None
    overwrite_output_table: bool = False

    @property
    def is_create_table(self) -> bool:
        return self.table_name is None and self.output_table_name is not None
```

Work items:

- update `FeatureExtractionRequest` to carry `table_name: str | None`,
  `output_table_name: str | None`, and `overwrite_output_table: bool`;
- update `FeatureExtractionBindingState` similarly, or replace the raw table
  fields with `FeatureExtractionTableTarget`;
- keep `FeatureExtractionJob.table_name` returning the final target table name
  for event/status use;
- refactor controller table validation:
  - existing-table mode still uses `get_table(...)`;
  - create-table mode validates the output table name and collision state without
    reading a table;
- keep the old `bind(...)` and `bind_batch(...)` signatures working by adding
  optional keyword-only `output_table_name=None` and
  `overwrite_output_table=False`.

Acceptance tests:

- existing-table controller tests still pass unchanged;
- binding with `table_name=None, output_table_name="new_table"` can become ready;
- create-table mode rejects an empty, invalid, or colliding output name;
- create-table mode rejects `overwrite_feature_key=True`;
- existing-table mode still rejects missing/nonexistent tables.

### Slice 2: Wire The Worker To Harpy Creation Mode

Once the controller can prepare a create-table job, pass the new fields into
Harpy.

Work items:

- update `_run_feature_extraction_job(...)` to pass:
  - `table_name=job.request.table_name`;
  - `output_table_name=job.request.output_table_name`;
  - `overwrite_output_table=job.request.overwrite_output_table`;
- make `FeatureExtractionResult.table_name` resolve to the existing table name or
  the new output table name;
- keep `change_kind="created"` for create-table jobs;
- keep `change_kind="updated"` only for existing-table feature-key replacement.

Acceptance tests:

- fake-Harpy test captures `table_name=None` and
  `output_table_name="new_table"`;
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
- add state:
  - `_selected_table_mode: Literal["existing", "create"] | None`;
  - `_new_table_name: str | None`;
- keep `selected_table_name` meaning "selected existing table name";
- add a new property such as `selected_output_table_name` for create mode;
- update `_refresh_table_names()` to populate eligible existing tables plus the
  sentinel when the staged labels batch is complete;
- update `_on_table_changed(...)` to show/hide the new-name field and rebind;
- add `_on_new_table_name_changed(...)` to rebind on text edits.

Acceptance tests:

- before a labels batch is complete, the table controls remain inactive as they
  are today;
- with one eligible existing table, the combo contains `table` and
  `Create table...`, with `table` selected by default;
- with no eligible existing tables, the combo contains `Create table...`, it is
  enabled, and the new-name field is visible;
- switching away from create mode hides the new-name field without losing the
  typed value;
- `selected_table_name` remains `None` in create mode, while
  `selected_output_table_name` returns the trimmed new table name.

### Slice 4: Bind Create-Table Widget State Into The Controller

This is the point where the UI path becomes calculable.

Work items:

- update `_validate_selected_table_binding(...)` so it only validates existing
  table selections;
- add `_validate_create_table_target(...)` for the new table name;
- update `_bind_current_selection()` and `_expected_controller_binding_state()`
  to pass either:
  - existing mode: `table_name=<existing>, output_table_name=None`;
  - create mode: `table_name=None, output_table_name=<new>`;
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

Work items:

- in `_calculate_feature_matrix()`, only inspect `table.obsm` and prompt for
  feature-key overwrite in existing-table mode;
- in create-table mode, never send `overwrite_feature_key=True`;
- block table-name collisions before calculation for MVP;
- leave `overwrite_output_table=True` out of the UI until we explicitly spec the
  replace-table behavior.

Acceptance tests:

- create-table mode does not call `_prompt_overwrite_feature_key_confirmation`;
- existing-table mode still prompts when `.obsm[feature_key]` already exists;
- a new table name colliding with any existing `sdata.tables` key keeps
  Calculate disabled with a clear status-card message and matching tooltip.

### Slice 6: Refresh Selection After The New Table Is Created

After Harpy creates the table, the widget should move from create mode to the new
authoritative existing table.

Work items:

- make the success refresh prefer `event.table_name` or the job result table name
  when repopulating table options;
- after creation, select the newly created table in the `Table` combo;
- hide the `New table name` field once the new table is selected as an existing
  table;
- emit the existing `FeatureMatrixWrittenEvent` with the new table name so dirty
  state and downstream widgets see the table update;
- consider updating the object-classification widget listener so any
  `feature_matrix_written` event for the same `sdata` refreshes table names, not
  only events for the currently selected table.

Acceptance tests:

- after a fake successful create-table result, `_refresh_table_names()` selects
  the new table;
- `HarpyAppState.is_table_dirty(sdata, "new_table")` becomes `True`;
- object-classification table choices can discover the new table in the same
  session when its selected labels element matches.

### Slice 7: Polish And Documentation

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
pytest tests/test_feature_extraction.py tests/test_feature_extraction_widget.py tests/test_app_state.py
```

## Open Spec Questions

- Should `Create table...` be last, preserving existing table auto-selection, or
  first for discoverability?
- Should the widget suggest a default table name, such as
  `features_<labels_name>` or `table_<labels_name>`?
- In multi-label batches, should the suggested default be based on the common
  coordinate system, the first labels element, or a neutral name like
  `feature_table`?
- Do we want an overwrite-output-table confirmation in this first pass, or should
  collisions remain a hard block?
- Should object classification auto-select a newly created table when it matches
  the current labels element, or merely refresh the available table list?
