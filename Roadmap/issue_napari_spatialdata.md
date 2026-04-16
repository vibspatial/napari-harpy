Roadmap: Remove napari-spatialdata Runtime Dependency in napari-harpy
Goal

Make napari-harpy independently capable of:

visualizing SpatialData images,
visualizing SpatialData labels,
coloring a labels layer from a selected AnnData table,

without depending on the napari-spatialdata plugin at runtime.

Why

napari-harpy only needs a focused subset of the functionality currently provided by napari-spatialdata:

render raster images,
render segmentation labels,
bind labels to a table,
recolor labels by a selected obs column.

Depending on the full plugin for this introduces avoidable coupling to:

plugin-specific UI/state,
internal layer management,
behavior outside harpy’s actual scope.

The intended direction is to depend on:

spatialdata for transformations and table semantics,
napari for rendering,
anndata for table data,

and own the visualization logic directly inside napari-harpy.

Scope
In scope
display SpatialData image elements in napari
display SpatialData labels elements in napari
apply element transformations correctly via napari layer affine transforms
bind a labels layer to an AnnData table
recolor labels by categorical or continuous obs columns
support palettes from adata.uns["<key>_colors"] where available
Out of scope for initial implementation
shapes and points support
full napari-spatialdata feature parity
bidirectional sync back into SpatialData
complex widget ecosystems
advanced cross-layer selection/linking
viewer state serialization
Core design principle

Do not depend on napari-spatialdata for rendering.

Instead:

read transformations from spatialdata,
convert them to a napari-compatible affine,
pass the affine directly to napari.layers.Image or napari.layers.Labels,
manage labels-to-table coloring within napari-harpy.

This keeps the implementation small, explicit, and testable.

Architecture
1. SpatialData raster rendering module

Create a thin rendering layer inside napari-harpy responsible for adding images and labels from SpatialData.

Suggested API:

show_image(viewer, sdata, image_key, coordinate_system="global")
show_labels(viewer, sdata, labels_key, coordinate_system="global")

Responsibilities:

fetch the requested element from SpatialData
inspect element axes
compute affine transform for the selected coordinate system
normalize array layout for napari
create the napari layer with affine=...

Important: do not pre-warp or resample the raster before display unless explicitly needed later. Use napari’s layer transform system.

2. Transform conversion helper

Implement a small helper dedicated to turning a SpatialData transformation into a napari-ready affine.

Suggested API:

get_napari_affine(element, coordinate_system) -> np.ndarray | Affine

Responsibilities:

retrieve the transformation from spatialdata
use SpatialData’s transformation machinery rather than manually reconstructing matrices
preserve axis semantics
support 2D and 3D cases

This helper should be the single source of truth for all raster transform application.

3. Raster axis normalization helper

Implement a helper that converts image/labels arrays into napari-compatible layout.

Suggested API:

prepare_raster_for_napari(element) -> tuple[data, kwargs]

Responsibilities:

preserve y, x and z, y, x semantics
normalize channel order when channels are present
detect RGB/RGBA images and set rgb=True
support multiscale image pyramids if needed

This logic should be separate from transform logic.

4. Labels ↔ table binding model

Introduce a small internal binding object stored in layer metadata.

Suggested structure:

from dataclasses import dataclass

@dataclass
class LabelTableBinding:
    layer_name: str
    element_name: str
    table_name: str
    region: str
    instance_key: str
    region_key: str

Store in:

labels_layer.metadata["harpy_binding"] = ...

Responsibilities:

define which labels layer is linked to which table
define which obs column contains instance ids
define which region is represented
avoid implicit plugin-specific state
5. Labels coloring engine

Implement a dedicated recoloring function for labels.

Suggested API:

color_labels_by_obs(labels_layer, adata, key, instance_key, palette=None)

Responsibilities:

map label integers to rows in adata.obs
support categorical coloring
support continuous coloring
generate a napari-compatible labels colormap / direct label color mapping
respect adata.uns[f"{key}_colors"] when available

Important rule:

Label ids must be matched explicitly through instance_key, never by row position.

Do not assume:

labels are contiguous,
labels start at 1,
row order matches label id,
obs_names are the segmentation ids.
Data model assumptions
Labels mapping

The robust mapping should be:

label_value == adata.obs[instance_key]

and not:

nth row -> label n

This should remain true even when:

the table is filtered,
ids are sparse,
ids are non-consecutive,
categories are reordered.

Investigation notes from current implementations

Environment inspected:

napari-spatialdata 0.7.0
napari 0.7.0
spatialdata 0.7.2

napari-spatialdata labels coloring

Current coloring is widget-driven and depends on plugin-managed layer metadata.

When a SpatialData labels element is loaded, napari-spatialdata stores on the layer:

metadata["adata"]
metadata["region_key"]
metadata["instance_key"]
metadata["indices"]

The stored metadata `adata` is produced from a left join between the selected spatial element
and the selected table, so the labels recoloring path is tightly coupled to napari-spatialdata’s
own metadata/cache model.

When the user selects an obs column to color by, napari-spatialdata:

reads the vector from `adata`
reindexes it by `instance_key`
merges it against `layer.metadata["indices"]`
builds a per-label mapping
assigns a napari `DirectLabelColormap`

Categorical coloring:

uses `adata.uns[f"{key}_colors"]` when available
otherwise generates colors through Scanpy helpers with a `tab20` fallback
matches labels to rows through `instance_key`, not row order

Continuous coloring:

normalizes the merged numeric vector
uses a matplotlib colormap, default `viridis`
currently fills missing merged values with `0` before colormapping, so missing values are
not visually distinct from a true zero in that path

Fallback behavior:

For labels that do not end up in the direct mapping, napari’s `DirectLabelColormap` renders them
as transparent unless an explicit default color is provided.

napari-harpy labels coloring

Current Harpy behavior is already much closer to the desired standalone design.

Harpy colors labels through its own `ViewerStylingController`, not through napari-spatialdata’s
widget state. The active table in `sdata[table_name]` is treated as the source of truth, while
`layer.metadata["adata"]` is only retained as a compatibility cache for napari-spatialdata.

Harpy currently:

filters the authoritative table by `region_key == selected_label_name`
coerces and validates `instance_key`
drops duplicate instance ids
indexes rows by `instance_key`
builds a per-label `DirectLabelColormap`
refreshes layer features from the same per-instance rows

Categorical coloring:

supports `user_class` and `pred_class`
reads `table.uns[f"{key}_colors"]` when available
preserves stored categorical order when interpreting palettes
backfills missing class ids with deterministic default colors
keeps class palettes synchronized when annotations or predictions are written

Continuous coloring:

currently supports `pred_confidence`
uses `viridis`
assigns an explicit grey fallback for missing values rather than collapsing them into zero

Fallback behavior:

Harpy explicitly makes label `0` transparent and provides a default unlabeled color for otherwise
unmapped labels, which is closer to the roadmap requirement that unmapped labels have a defined
fallback behavior.

Implications for this roadmap item

Useful behavior to preserve:

explicit matching through `instance_key`
compatibility with `adata.uns[f"{key}_colors"]`
support for both categorical and continuous coloring
use of napari `DirectLabelColormap` for final rendering

Behavior not to copy from napari-spatialdata:

reliance on `layer.metadata["adata"]` as the primary source of truth
reliance on plugin widget/model state to drive recoloring
implicit fallback of unmapped labels to transparency unless that is chosen intentionally
continuous-mode treatment of missing values as ordinary zero values

Practical conclusion:

The standalone Harpy implementation should own a small explicit recoloring engine that takes
`labels_layer`, `adata`, `key`, and `instance_key`, builds the direct label-to-color mapping itself,
and uses `adata.uns[f"{key}_colors"]` only as palette input rather than depending on napari-spatialdata
layer metadata conventions.
Milestones
Milestone 1: Minimal independent raster support

Deliver:

show_image(...)
show_labels(...)
affine extraction from SpatialData
raster axis normalization

Acceptance criteria:

2D images render at the correct translated/scaled/rotated position
2D labels render aligned with images
3D data does not regress basic rendering
no runtime dependency on napari-spatialdata
Milestone 2: Labels-table binding

Deliver:

internal labels-to-table binding object
helper to bind a labels layer to an AnnData table

Suggested API:

bind_table_to_labels(labels_layer, adata, table_name, instance_key, region, region_key)

Acceptance criteria:

a labels layer can be associated with a selected table
metadata is sufficient to reconstruct the binding later
bindings remain stable across recoloring operations
Milestone 3: Table-driven label coloring

Deliver:

color_labels_by_obs(...)
categorical palette support
continuous value support
compatibility with adata.uns["<key>_colors"]

Acceptance criteria:

categorical obs columns color labels correctly
continuous obs columns color labels correctly
unmapped labels receive a defined fallback behavior
missing values are handled gracefully
Milestone 4: UI integration

Deliver minimal widget support for:

selecting a labels layer
selecting a table
selecting an obs column
choosing a palette / colormap
resetting colors

Acceptance criteria:

user can recolor a labels layer interactively
UI remains independent from napari-spatialdata
no hidden reliance on plugin-specific state
Milestone 5: Hardening and tests

Deliver a focused test suite for:

affine correctness
axis handling
labels ↔ table id mapping
palette correctness
categorical / continuous rendering
sparse/non-contiguous ids

Acceptance criteria:

rendering logic is covered independently of UI
labels coloring behavior is deterministic
regressions are caught without relying on manual viewer inspection
Testing plan
Transform tests

Test at least these axis cases:

("y", "x")
("c", "y", "x")
("z", "y", "x")
("c", "z", "y", "x")

Verify:

translation
scaling
rotation where applicable
image and labels alignment under the same transform
Labels/table mapping tests

Test cases:

label ids are contiguous
label ids are sparse
label ids do not start at 1
adata.obs is filtered or reordered
some labels are missing from table
some table rows do not correspond to any visible label

Verify:

color assignment always follows instance_key
no positional assumptions leak into the implementation
Palette tests

Test cases:

categorical with adata.uns["<key>_colors"]
categorical without stored colors
continuous numeric column
missing values
mixed/incompatible dtypes

Verify:

category/color mapping is deterministic
category order matches pandas categorical order when applicable
fallback palette behavior is stable
Implementation notes
Keep transform logic isolated

Transform bugs are easy to introduce when mixed with channel handling or UI code. Keep the affine conversion helper separate and test it directly.

Keep array normalization isolated

Image channel handling and labels layout should not be intertwined with transformation code.

Own label coloring semantics

Do not copy table-coloring behavior wholesale from napari-spatialdata. Reimplement the mapping explicitly around instance_key and AnnData semantics.

Prefer explicit metadata contracts

Use labels_layer.metadata for harpy-owned bindings instead of relying on external plugin state.

Suggested internal API surface
show_image(viewer, sdata, image_key, coordinate_system="global")
show_labels(viewer, sdata, labels_key, coordinate_system="global")

get_napari_affine(element, coordinate_system)
prepare_raster_for_napari(element)

bind_table_to_labels(
    labels_layer,
    adata,
    table_name,
    instance_key,
    region,
    region_key,
)

color_labels_by_obs(
    labels_layer,
    adata,
    key,
    instance_key,
    palette=None,
)
First deliverable

A good first PR would implement:

show_image(...)
show_labels(...)
get_napari_affine(...)
prepare_raster_for_napari(...)
tests for affine correctness and image/labels alignment

This is the smallest meaningful step that removes the rendering dependency on napari-spatialdata.

Second deliverable

A follow-up PR would implement:

labels ↔ table binding metadata
color_labels_by_obs(...)
categorical palette support
continuous value support
tests for explicit instance_key mapping

This is the smallest meaningful step that restores the user-facing workflow of coloring segmentations from AnnData.

Definition of done

napari-harpy can:

render SpatialData images independently,
render SpatialData labels independently,
align both correctly in the same coordinate system,
bind labels to an AnnData table,
recolor labels from selected obs columns,

without importing or depending on the napari-spatialdata plugin at runtime.

Summary

The path forward is not to replicate napari-spatialdata wholesale.

The path forward is to extract and own a narrow, well-defined visualization stack inside napari-harpy:

SpatialData for transformations and table semantics,
napari for rendering,
AnnData for annotation-driven label coloring.
