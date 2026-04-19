# Feature Extraction Widget: Channel Selection and Visualization Plan

## Decision

We will implement **Select channels in the widget, visualize in napari**.

This means the feature-extraction widget becomes the **source of truth** for channel selection, while napari remains the visualization surface.

## Desired user flow

1. User chooses an image in the **Feature Extraction** widget.
2. The widget shows the channel list for that image.
3. The user checks the channels they want to use.
4. The widget updates or activates the corresponding napari image layer to display exactly those selected channels.
5. Feature extraction uses exactly the same selected channels.

This approach is preferable because it is:

- **explicit**: the selected channels are visible in the widget UI
- **reproducible**: the extraction inputs are defined by widget state, not hidden viewer state
- **easy to connect to extraction parameters**: the same state can be passed directly into the feature-extraction controller and persisted in metadata

## Design principles

### 1. The widget owns channel selection

The feature-extraction widget should be the authoritative UI for:

- selected image
- selected coordinate system
- selected channels
- selected features
- output feature key

napari should not be the source of truth for channel selection. It should reflect the widget state.

### 2. napari remains the visualization surface

The widget should control napari viewer behavior:

- load the selected image layer if needed
- activate the corresponding image layer
- show only the selected channels
- keep the displayed state synchronized with the widget state

This gives users explicit visual feedback without duplicating image-rendering logic inside the widget.

### 3. Feature extraction must use exactly the selected channels

The selected channels shown in the widget and visualized in napari must be the exact channels used in the feature-extraction call.

This is important for correctness, reproducibility, and future multi-sample support.

## Proposed UI additions to the Feature Extraction widget

After the user selects an image, show a new **Channels** section.

### Channels section contents

- channel list with checkboxes
- channel names if available
- fallback names such as `channel_0`, `channel_1`, ... if names are unavailable
- summary text such as:
  - selected image name
  - number of channels selected
  - selected coordinate system

### Suggested actions

Add one or two explicit buttons:

- `Show image`
- `Show selected channels`

Possible behavior:

- `Show image`: load or activate the chosen image layer in napari
- `Show selected channels`: update the napari image layer so only the selected channels are visualized

### Optional compact status line

A compact status line in the widget would help orientation, for example:

- `Image: raw_multiplex`
- `Channels: CD3, CD8, DAPI`
- `Selected: 3 / 28`

This keeps the UI explicit without trying to embed a full viewer in the widget.

## Required architectural changes

### New widget state

We need to introduce channel-aware image selection state.

Suggested state fields:

- `image_name`
- `coordinate_system`
- `channel_indices`
- `channel_names`
- `display_mode` (optional, future-facing)

This state should live at the widget/controller boundary and be part of the authoritative configuration for extraction.

### Controller changes

The `FeatureExtractionController` and `FeatureExtractionJob` should be extended so that channel selection is included in the calculation inputs.

At minimum, the job should carry:

- selected image name
- selected channel indices
- selected channel names (optional but useful)
- coordinate system
- feature names
- output feature key

### Viewer adapter changes

A viewer adapter or helper should be responsible for:

- ensuring the selected image is loaded in napari
- activating the correct image layer
- translating widget-selected channels into napari display state
- keeping visualization synchronized with widget state

This logic should stay outside the feature-extraction controller.

## Data and metadata requirements

The selected channels must be recorded in the output metadata for the generated feature matrix.

For each generated feature matrix, metadata should include at least:

- `source_image`
- `selected_channels`
- `selected_channel_names`
- `coordinate_system`
- `source_label`
- `features`
- `schema_version`

This metadata should live alongside the feature matrix in the feature-matrix metadata namespace, for example under:

- `adata.uns["feature_matrices"][feature_key]`

## Why this matters for multi-sample support

If we maintain one authoritative table that stores features for multiple samples, then explicit channel selection becomes even more important.

We must avoid silent schema drift, such as:

- sample A extracted using one channel subset
- sample B extracted using another channel subset
- both written under the same feature key without clear visibility

Making channel selection explicit in the widget helps ensure:

- feature definitions are transparent
- extraction inputs are comparable across samples
- future multi-sample validation rules can be enforced cleanly

## Recommended phased implementation

### Phase 1: Channel discovery and selection

Add to the Feature Extraction widget:

- a `Channels` group under the selected image
- checkbox list for channels
- compact selected-channel summary

At this phase, the widget owns the state, even if visualization sync is still basic.

### Phase 2: napari synchronization

Add viewer-facing actions:

- `Show image`
- `Show selected channels`

Implement synchronization so the selected napari image layer reflects the widget state.

### Phase 3: Extraction wiring

Extend the feature-extraction controller and job payload to include selected channels.

The extraction call must use exactly the channel subset selected in the widget.

### Phase 4: Metadata persistence

Write selected-channel metadata into the stored feature-matrix metadata.

This ensures reproducibility and prepares the path for future multi-sample workflows.

## Final recommendation

We should implement:

**widget-controlled channel selection + napari-based visualization**

This gives us:

- explicit and user-friendly channel selection
- clear viewer feedback
- reproducible extraction inputs
- a strong foundation for future multi-sample support
