# Viewer Color Source Field: Searchable Popup Investigation

Date: 2026-07-09
Updated: 2026-07-10

## Question

The viewer widget currently lets users type an `obs` / `vars` color source into a text field. Suggestions appear while typing, but the field does not behave like a browseable selector before typing starts.

Desired UX:

- User clicks in a `QLineEdit`.
- A popup opens immediately.
- The popup shows at most 10 visible names.
- The user can scroll to see more names.
- The user can type in the line edit.
- The popup then shows matching names, still capped to 10 visible rows and scrollable when there are more matches.

This matters especially for `vars`, where a table may have around 20,000 entries.

## Scope

First implementation slice:

- Labels color source field for table-backed `obs` and `vars` coloring.
- Shapes color source field for table-backed `obs` and `vars` coloring.
- Shapes color source field for direct shapes-column coloring.

Explicitly out of scope for the first slice:

- Points `Values` field.

The points `Values` field has the same UX problem, but it should be handled in a follow-up slice after the obs/vars coloring behavior is settled.

## Slice 1: Labels/Shapes Color Source Popup

Status: implemented on 2026-07-10.

Verification:

- `.venv/bin/pytest tests/test_viewer_widget.py`
- `.venv/bin/pytest tests/test_shared_styles.py`
- `.venv/bin/ruff check src/napari_harpy/widgets/shared_styles.py src/napari_harpy/widgets/viewer/labels_widget.py src/napari_harpy/widgets/viewer/shapes_widget.py tests/test_viewer_widget.py`

Goal: make the labels and shapes color-source fields behave like browseable searchable popups, without preselecting the first available source.

Target files:

- `src/napari_harpy/widgets/viewer/labels_widget.py`
- `src/napari_harpy/widgets/viewer/shapes_widget.py`
- shared helper location if useful, likely `src/napari_harpy/widgets/shared_styles.py`
- tests in `tests/test_viewer_widget.py`

In scope:

- labels `Observations`
- labels `Vars`
- shapes `Shapes column`
- shapes `Observations`
- shapes `Vars`

Out of scope:

- points `Values`
- primary labels/shapes loading with no color source

Required behavior:

- Selecting `Shapes column`, `Observations`, or `Vars` must not automatically put the first available value into the line edit.
- With no explicit selected source, the value line edit must remain empty and show a placeholder.
- Clicking or focusing the empty line edit must open the completer popup with an empty prefix.
- The popup must show at most 10 visible rows.
- The popup must be scrollable when there are more than 10 available matches.
- Typing in the line edit must filter the popup using the current text.
- Filtering should remain case-insensitive.
- Matching should remain substring-based with `Qt.MatchContains`.
- The visible order should preserve the existing source order: `table.obs.columns` for obs and `table.var_names` for vars.
- If the user has already selected a valid source and the widget refreshes, preserve that explicit selection.
- If the selected source kind changes, clear the field unless there is an explicit valid selection for the new kind.
- If a previously selected source disappears after refresh, clear the field and restore the empty placeholder state.

Placeholder text:

- Shapes column: `Select column`
- Obs: `Select obs column`
- Vars: `Select var`

Action hint behavior:

- Empty obs field with available options: prompt the user to select an observation column.
- Empty vars field with available options: prompt the user to select a var.
- Empty shapes-column field with available options: prompt the user to select a shapes column.
- Empty obs/vars field with no available options: keep the existing no-options messages.
- The action hint must not claim that an overlay/styled layer will be created for the first source until the user has actually selected or typed a valid source.

Expected Qt configuration:

- `QCompleter.PopupCompletion`
- `QCompleter.setMaxVisibleItems(10)`
- `Qt.CaseInsensitive`
- `Qt.MatchContains`

Suggested implementation shape:

- Introduce a small reusable `QLineEdit` subclass or event filter that opens its completer on focus/click.
- Use that helper for the labels and shapes color-source value inputs.
- On focus/click, set the completion prefix to the current text and call `complete()`.
- Because the field should be empty by default, the first click should use an empty prefix and show the first available entries.
- Keep the existing `QStringListModel`-based completer model unless profiling proves it is too slow.

Test expectations:

- Labels obs selection leaves the value field empty and uses the obs placeholder.
- Labels vars selection leaves the value field empty and uses the vars placeholder.
- Shapes obs selection leaves the value field empty and uses the obs placeholder.
- Shapes vars selection leaves the value field empty and uses the vars placeholder.
- For each labels/shapes obs/vars path, the completer has max 10 visible rows.
- Empty field means `selected_color_source is None`.
- A valid manually typed or selected value resolves to the expected `selected_color_source`.
- Existing no-options states still disable the field and report the existing no-options action hint.
- Existing explicit valid selections are preserved across refreshes where possible.

## Slice 2: Points Values Popup

Status: implemented on 2026-07-10.

Goal: make the points `Values` field use the same browseable searchable popup behavior as labels/shapes, while preserving the points multi-select workflow.

Target files:

- `src/napari_harpy/widgets/viewer/points_widget.py`
- shared helper already available in `src/napari_harpy/widgets/shared_styles.py`
- tests in `tests/test_points_widget.py`
- viewer integration tests in `tests/test_viewer_widget.py` if needed

In scope:

- `PointsValueWidget.value_input`
- Existing `QLineEdit` + `QCompleter` value search behavior
- Existing Add button / Return key value-adding workflow
- Existing selected-values summary behavior

Out of scope:

- Changing how points values are loaded.
- Changing the selected-values storage model.
- Adding values immediately on completer activation, unless this is already existing behavior.
- Changing `All values` semantics.

Required behavior:

- Use `CompleterPopupLineEdit` for the points `Values` input.
- The field should stay empty until the user types or chooses a value.
- Clicking or focusing the empty field should open the completer popup with an empty prefix.
- The popup should show at most 10 visible rows.
- The popup should be scrollable when there are more than 10 available values.
- Typing in the field should filter the popup using the current text.
- Filtering should remain case-insensitive.
- Matching should remain substring-based with `Qt.MatchContains`.
- The visible order should preserve the loaded value source order.
- Adding a value through the Add button or Return key should keep clearing the input afterward.
- After the input clears, clicking/focusing it again should show the first available values from an empty prefix.
- Enabling `All values` should continue to disable the input, Add button, and completer popup behavior.

Placeholder text:

- Points values: `Select value`

Reasoning:

- The row label already says `Values`, so the placeholder can stay short.
- `Select value` matches the shape/obs/var placeholder style from Slice 1.
- The Add button and selected-values summary make it clear that this is a multi-select workflow.

Expected Qt configuration:

- `QCompleter.PopupCompletion`
- `QCompleter.setMaxVisibleItems(10)`
- `Qt.CaseInsensitive`
- `Qt.MatchContains`

Implementation notes:

- `PointsValueWidget.value_input` uses `CompleterPopupLineEdit`.
- The completer opens on click/focus when value selection is enabled.
- The input remains disabled, including popup-on-entry behavior, when `All values` is enabled or no values are loaded.
- The Add button and Return key remain the only value-add paths; selecting or typing in the completer does not by itself add a value.

Verification:

- `tests/test_points_widget.py` covers the empty-prefix popup, 10-row cap, substring filtering, short placeholder, and existing Add / All values flows.
- `tests/test_viewer_widget.py` passes with the points popup change in place.

Suggested implementation shape:

- Replace the points value `QLineEdit` with `CompleterPopupLineEdit`.
- Enable completion popup on entry only while the value input is enabled.
- Keep the existing `QStringListModel` value completer model.
- Keep existing `_resolve_available_value`, `_add_value_from_input`, and selected-values rendering behavior.
- On focus/click, set the completion prefix to the current text and call `complete()`.
- Because the field is normally empty after adding a value, the next click should browse from the first available values again.

Test expectations:

- Points value input uses the `Select value` placeholder.
- Points value completer has max 10 visible rows.
- Empty enabled value input opens completion with an empty prefix.
- Typing filters the completion model.
- Adding a valid value still clears the input and updates the selected-values summary.
- Duplicate values are still ignored.
- Unknown values are still rejected.
- `All values` still disables the value input and Add button.

## Current Implementation

The relevant viewer controls already use `QLineEdit` plus `QCompleter`.

- Labels color source field: `src/napari_harpy/widgets/viewer/labels_widget.py`
- Shapes color source field: `src/napari_harpy/widgets/viewer/shapes_widget.py`
- Points value field now uses the shared popup-on-entry pattern from Slice 2: `src/napari_harpy/widgets/viewer/points_widget.py`

Current completer behavior:

- `QStringListModel` stores the available names.
- `QCompleter` is attached to the `QLineEdit`.
- Matching is case-insensitive.
- Matching uses `Qt.MatchContains`, so typing a substring can match names that contain it.
- The popup is styled with `COMPLETER_POPUP_STYLESHEET`.

The missing behavior is not a different data model. The missing behavior is opening the popup on focus/click and limiting the visible popup height.

Current behavior also pre-fills the value field with the first available source when a source kind is selected. That is not desired for the new UX: an auto-filled first value becomes the completion prefix, so clicking the field opens a filtered popup for that one value instead of opening a browseable list from the beginning.

## Desired Empty State

When the user selects `Observations` or `Vars`, the value `QLineEdit` should not automatically select the first available value.

Desired behavior:

- If there is no explicit previous user selection, leave `QLineEdit.text()` empty.
- Use placeholder text to explain what the user can do.
- On focus/click with an empty field, open the completer with an empty prefix.
- The popup should then show the first available names, capped to 10 visible rows and scrollable.
- Typing should replace the empty state with a normal filtered search.
- The action hint should treat an empty field as "no source selected" and ask the user to select a shapes column, observation column, or var.

Selection preservation:

- If the user explicitly selected a source and the widget refreshes, preserve that source when it is still valid.
- If the preserved source is no longer valid, clear the field and return to the placeholder empty state.
- If the source kind changes between `Shapes column`, `Observations`, and `Vars`, clear the field unless there is a valid explicit selection for the new kind.

Recommended placeholder text:

- Shapes column: `Select column`
- Obs: `Select obs column`
- Vars: `Select var`

Reasoning:

- The text must fit inside the compact viewer card.
- `Select` keeps the shape, obs, and vars placeholders visually consistent.
- `Select column` fits the direct shapes-column list while the adjacent label supplies the shapes context.
- `Select obs column` fits the smaller, column-like `obs` list without wrapping or clipping.
- `Select var` fits the potentially large `vars` list and keeps the field visually clean.
- The click-to-browse affordance should come from the field behavior and tests, not from long placeholder text.
- Avoid pre-filling a real value as instructional text; the field content should always mean an actual selected or typed value.

## Qt Findings

Qt supports this UX with `QLineEdit` + `QCompleter`.

Relevant API:

- `QCompleter.setCompletionMode(QCompleter.PopupCompletion)`
- `QCompleter.setMaxVisibleItems(10)`
- `QCompleter.setFilterMode(Qt.MatchContains)`
- `QCompleter.setCaseSensitivity(Qt.CaseInsensitive)`
- `QCompleter.complete()`

`setMaxVisibleItems(10)` limits the popup height to 10 visible rows. It does not limit the total number of possible completions, so the popup can still be scrollable.

To show suggestions before typing, the line edit can trigger the completer when it receives focus or is clicked:

- set the completion prefix to the current text, usually `""` for an empty field
- call `complete()`

This should show the first 10 visible options and allow scrolling through the rest.

## Important Mode Choice

Use `QCompleter.PopupCompletion`.

Do not use `QCompleter.UnfilteredPopupCompletion` for this field.

In a local check with 20,000 synthetic names, `UnfilteredPopupCompletion` kept the completion model unfiltered after typing. For example, typed prefixes still produced 20,000 completion rows. That is not the desired search behavior.

With normal `PopupCompletion`, an empty prefix can show all available options on click, while typed text filters the popup.

## Large `vars` Lists

A 20,000-entry `vars` list should not be visually expanded to 20,000 rows. However, it is acceptable for the completer model to contain 20,000 names if the popup is capped to 10 visible rows.

Local synthetic timing check with `QStringListModel` and 20,000 names:

- Empty prefix: 20,000 completion rows, prefix refresh was effectively instant.
- Prefix matching many rows: around 1-2 ms in the synthetic check.
- Prefix matching few rows: around 1-2 ms in the synthetic check.

This was not a full GUI rendering benchmark, but it suggests the existing `QStringListModel` + `QCompleter` approach is reasonable for 20,000 vars.

Potential future optimization if needed:

- Prefer `Qt.MatchStartsWith` for very large lists and sorted models, because Qt can optimize sorted prefix matching more effectively.
- Keep `Qt.MatchContains` for better search UX unless profiling shows it is a problem.
- Avoid calling expensive total-count APIs during typing.

## Ordering

The easiest and most predictable ordering is to preserve the existing source order.

Current source construction already preserves table order:

- `obs` color sources iterate `table.obs.columns`.
- `vars` color sources iterate `table.var_names`.

That means the popup naturally follows the AnnData/table order. Alphabetical sorting is possible, but preserving table order is less surprising and requires less code.

## Recommendation

Keep the current `QLineEdit` + `QCompleter` approach and adjust its behavior.

Implementation direction:

1. Remove automatic first-value selection for labels/shapes table-backed `obs` and `vars` fields.
2. Set `maxVisibleItems` to 10 on the labels/shapes color-source completers.
3. Keep `PopupCompletion`, `CaseInsensitive`, and `MatchContains`.
4. Add a small reusable `QLineEdit` subclass or event filter that opens the completer popup on focus/click.
5. On focus/click, call `complete()` with an empty prefix when the field is empty so the popup starts at the first available entries.
6. When text is present, call `complete()` with the current text so normal filtering still works.
7. Reuse the behavior for labels and shapes color-source fields.

This matches the desired UX without replacing the field with a `QComboBox`, and it handles large `vars` lists better than a traditional dropdown.
