# Issue: Large SpatialData Points Rendering In Napari

## Summary

`napari` is not the right primitive for directly drawing very large point clouds as a single
`Points` layer when the dataset reaches tens of millions to billions of rows.

For `napari-harpy`, the concrete rendering strategy should be:

1. keep the `SpatialData` points element as the source of truth
2. convert the currently visible points into a raster overview image
3. display that raster as a napari `Image` layer
4. only switch to a true napari `Points` layer when the visible subset is small enough

In short:

`points table -> aggregated image -> napari Image layer`

and only sometimes:

`visible subset -> napari Points layer`

This should live in a separate ticket from
`issue_napari_spatialdata.md`, because the implementation and performance constraints are very
different from image and labels rendering.

For the first implementation milestone, the problem should be framed even more narrowly:

- allow the user to select a subset of genes
- lazily derive an image layer from the selected transcript points in the current viewport
- color the selected genes by category

That narrower milestone is the concrete target for the first pass.

## Goal

Make `napari-harpy` capable of showing `SpatialData` points in napari with responsive pan and zoom,
including datasets that are far too large for direct per-point rendering.

## First Milestone

The first milestone should deliberately stop short of full points support.

Concrete deliverable:

1. user selects a small set of genes
2. `napari-harpy` filters the `SpatialData` points element lazily to those genes
3. `napari-harpy` renders the current viewport into an RGB image
4. that RGB image is shown in napari as a standard `Image` layer
5. the image is refreshed on pan and zoom in the background

This means the first version is not:

- a custom napari layer type
- a full replacement for napari `Points`
- an exact point-picking solution

It is a lazy view-dependent image renderer built on top of the points table.

## Why

For large transcript or spot datasets, the viewer needs to answer a visual question first:

- where are the dense regions?
- how does density change across space?
- what does a selected category look like spatially?

At that stage, users do not need billions of editable glyphs. They need an overview that updates
quickly and stays aligned with the rest of the `SpatialData` scene.

The rendering model therefore needs to optimize for:

- fast overview rendering
- correct spatial alignment
- optional local detail
- compatibility with the rest of the raster rendering path

## Core Decision

Do not try to show large `SpatialData` point elements by directly passing the full table to
`napari.layers.Points`.

Instead:

- render a viewport-sized raster overview from the points
- show that overview in napari as an `Image` layer
- use a real `Points` layer only for a visible subset below a configurable threshold

This turns an intractable "draw N glyphs" problem into a bounded "draw one image for the current
camera view" problem.

## Spatial Transcriptomics Implications

In the spatial transcriptomics case, each point usually represents one measured transcript and the
main category of interest is typically:

- `gene`

That changes the desired viewer behavior in two ways.

### 1. Gene coloring should be panel-based, not global

Users should be able to color different genes differently, but the implementation should assume a
small selected gene set rather than "all genes at once".

Recommended first-pass contract:

- support a selected panel of roughly `1` to `8` genes
- optionally allow up to `16` with a clear performance warning
- assign a stable color per selected gene
- aggregate only the selected genes into the overview image

Do not try to render every gene in the dataset as its own visible color channel in the general case.

Reason:

- categorical rasterization cost scales with the number of displayed categories
- visual interpretation also breaks down when too many categories are mixed into the same pixels

### 2. Selection should use a hybrid overview/detail model

Users should be able to snappily select a subset of transcript points, but that does not mean the
full dataset needs to be a real napari `Points` layer at all times.

Recommended interaction model:

- use a `datashader` overview image for fast pan and zoom
- let the user filter by gene panel and viewport
- when the visible filtered subset is small enough, show a true napari `Points` layer for exact
  point selection

This means:

- overview interaction stays fast
- exact selection is still available
- the expensive glyph-rendering path is only used when it is affordable

For the first milestone, "selection" should primarily mean:

- selecting a subset of genes to display

Exact selection of individual transcript points can remain a follow-up step.

## Concrete Rendering Pipeline

This is the intended path from `SpatialData` points to something visible in napari.

### 1. Start from the `SpatialData` points element

The source of truth remains:

- `sdata.points[points_key]`

Expected shape:

- a `dask.dataframe.DataFrame` or compatible tabular points object
- with coordinate columns such as `x`, `y`
- optionally `z`
- optionally an `id` column
- optionally one or more columns used for coloring or aggregation

The full points table should remain lazy as long as possible.

### 2. Resolve the target coordinate system

Before any rendering work:

- read the `SpatialData` transformation for the points element
- resolve the selected coordinate system, such as `"global"`
- determine which columns map to napari world axes

The rendering path must preserve the same coordinate semantics used for images and labels.

### 3. Read the current napari camera view

On viewer creation and on relevant camera changes:

- read the current visible world-space bounds
- read the canvas size in screen pixels
- derive the target raster size for this render

Important rule:

- the overview should be rendered at screen scale, not at source-data scale

Typical cap:

- longest side between `1024` and `2048` pixels for the initial implementation

### 4. Aggregate points into a raster

Use `datashader` as the dynamic overview backend.

The render worker should:

1. construct a `Canvas` from the current world bounds and target pixel size
2. aggregate the points into that canvas
3. return a 2D or multi-channel array

Supported first-pass aggregation modes:

- count points per pixel
- count points per category
- aggregate a numeric value, such as sum or mean

This is the key conversion:

- rows in a table become pixels in an image

### 5. Convert the aggregate into a napari image

The aggregated raster should then be shown in napari as an `Image` layer.

The napari layer data is:

- the aggregated array returned by `datashader`

The napari layer transform is:

- an affine that maps raster pixel coordinates back to the world bounds used for aggregation

This keeps the points overview aligned with:

- images
- labels
- shapes
- any other element shown in the same coordinate system

### 6. Update on pan and zoom

When the camera changes:

- debounce the event
- start a background render job
- cancel stale jobs
- replace the image layer data only when the latest job completes

The viewer should never block on a large points render.

### 7. Switch to a real `Points` layer only for local detail

If the visible subset becomes small enough, the controller may replace or overlay the raster
overview with a real napari `Points` layer.

Suggested first-pass threshold:

- at or below `100_000` visible points

This threshold should remain configurable and be tuned later by profiling.

At that point, the render path becomes:

- filter the source table to the current visible region
- materialize just that subset
- add or update a napari `Points` layer

This gives:

- true glyphs
- per-point hover
- per-point selection

without making overview rendering pay the same cost.

## Two Tiers Of Performance

The ticket should explicitly distinguish between two performance targets.

### Tier A: Dynamic overview for large but still interactive datasets

Use dynamic `datashader` aggregation directly from the points table.

This is a good target for:

- millions of points
- tens of millions of points
- possibly more, depending on hardware and storage layout

This is the right first implementation because it is simple, testable, and flexible.

### Tier B: Very snappy rendering for extremely large datasets

For datasets around `1e9` points and beyond, repeated dynamic aggregation from raw points may still
be too expensive if we want consistently snappy pan and zoom on a workstation.

For that scale, the preferred strategy is:

- precompute one or more multiscale aggregated images from the points
- store them as `SpatialData` images
- render them through the regular raster path
- keep the original points table only for local detail queries

In other words:

- overview comes from precomputed raster pyramids
- detail comes from the original points table

This should be the recommended production path for "1 billion points and more".

## Concrete Answer To "How Do Points Become Visible In Napari?"

The shortest correct answer is:

1. load the `SpatialData` points table
2. choose the current coordinate system and visible world bounds
3. rasterize the points in that viewport into a 2D image
4. show that raster in napari as an `Image` layer with the correct affine
5. optionally show a real `Points` layer only for a small visible subset

So the visible thing in napari is usually not the raw point table itself.

It is a derived raster image computed from the point table.

## Category Rendering For Genes

For transcriptomics, the overview renderer should support categorical aggregation by gene.

Concrete first-pass implementation:

1. filter the source table to the selected gene panel
2. aggregate with a categorical reduction such as `ds.by("gene", ds.count())`
3. shade that categorical aggregate into one RGB image
4. show that RGB image in napari as an `Image` layer

This is a good fit for:

- one gene
- a few selected genes
- quick comparisons between a small number of markers

This is not a good fit for:

- every gene in the assay shown simultaneously with its own distinct visible color

When too many genes are selected, the renderer should prefer one of these fallbacks:

- refuse and ask the user to reduce the selection
- switch to density mode
- show only the top `k` selected genes

## Suggested API Shape

### Public entry point

```python
show_points(
    viewer,
    sdata,
    points_key,
    coordinate_system="global",
    color_by=None,
    selected_categories=None,
    mode="auto",
)
```

Possible behavior:

- `mode="overview"`: always show raster overview
- `mode="detail"`: always show real points from a subset
- `mode="auto"`: switch between overview and detail based on visible count

Implementation note:

- the first milestone only needs to implement the `overview` path

### Internal helpers

```python
get_points_axes_and_transform(points_element, coordinate_system) -> PointsAxesTransform

get_view_world_bounds(viewer) -> ViewBounds

render_points_overview(
    points_element,
    coordinate_system,
    world_bounds,
    canvas_shape,
    color_by=None,
) -> RenderedOverview

render_visible_points_subset(
    points_element,
    coordinate_system,
    world_bounds,
    color_by=None,
) -> RenderedPointsSubset
```

### Layer metadata binding

```python
from dataclasses import dataclass


@dataclass
class PointsRenderBinding:
    layer_name: str
    element_name: str
    coordinate_system: str
    x_key: str
    y_key: str
    z_key: str | None
    id_key: str | None
    color_by: str | None
    detail_threshold: int
```

Store in:

```python
layer.metadata["harpy_points_binding"] = ...
```

## Interaction Model

The initial interaction contract should stay simple.

### Overview mode

In raster overview mode, the layer is primarily for visualization.

Supported:

- visibility
- opacity
- colormap changes
- camera-driven refresh
- changing the selected gene panel

Not guaranteed in the first pass:

- exact per-point hover
- exact per-point click selection
- editing points

### Detail mode

In detail mode, when a real napari `Points` layer is shown, support:

- hover
- selection
- optional point-level metadata

This keeps interaction accurate without forcing the overview mode to behave like a full point editor.

## Choosing The Raster Backend

### Dynamic rendering

Use `datashader` for dynamic viewport-sized aggregation.

Why:

- built for very large point sets
- works with pandas and Dask dataframes
- supports categorical and numeric aggregations
- matches the "rows to pixels" rendering model we need

### Precomputed rendering

Use `spatialdata.rasterize()` or a Harpy-owned preprocessing step when we want persistent raster
products that can be loaded later as normal images.

This is especially attractive for:

- very large static datasets
- repeated viewing of the same point set
- deployment scenarios where startup cost matters less than navigation speed

### Special case: regular bins or grids

When the points represent bins on a regular or inferable grid, prefer a bin-aware rasterization path
instead of generic free-point rendering.

That case should be evaluated separately, because it can often be turned into a very efficient image
representation directly.

## Performance Rules

The implementation should follow these constraints from the beginning.

1. Never materialize the full points table into a napari `Points` layer unless it is genuinely small.
2. Keep coordinate columns in `float32` when precision allows.
3. Read only the columns needed for the current render.
4. Run overview rendering in a worker thread.
5. Debounce camera updates.
6. Cancel stale renders aggressively.
7. Warm the `datashader` path once to avoid a confusing first-interaction stall.
8. Cap raster output size to screen scale.
9. Cache recent overview renders by view bounds and zoom level when useful.
10. Treat precomputed multiscale rasters as the preferred path for the largest datasets.
11. Cap the number of simultaneously rendered gene categories in overview mode.

## Scope

### In scope for initial implementation

- 2D points rendering
- dynamic raster overview in napari
- world-aligned image layer output
- count and simple category-based coloring
- small selected gene panels with stable per-gene colors
- user-driven gene subset selection
- background rendering and stale-job cancellation

### Out of scope for initial implementation

- direct rendering of the full dataset as a napari `Points` layer
- automatic handoff to a detail `Points` layer
- full point editing at all scales
- exact point picking in overview mode
- 3D point-cloud rendering
- arbitrary high-cardinality multichannel raster products
- distributed rendering orchestration

## Investigation Notes

Environment inspected in this repository:

- `napari 0.7.0`
- `napari-spatialdata 0.7.0`
- `spatialdata 0.7.2`

Available locally in the project environment:

- `datashader`
- `dask`

Small local synthetic benchmark after warm-up:

- `10,000,000` points -> `2048 x 2048` aggregate from pandas in about `0.033 s`
- `10,000,000` points -> `2048 x 2048` aggregate from Dask in about `0.299 s`

Small local synthetic categorical benchmark after warm-up with `5,000,000` points on a
`1024 x 1024` canvas:

- `1` category in about `0.015 s`
- `4` categories in about `0.191 s`
- `8` categories in about `0.233 s`
- `16` categories in about `0.248 s`

Important nuance:

- the first `datashader` call had a much larger one-time warm-up cost

These numbers are not a guarantee for real datasets, but they support the design decision that a
raster overview layer is the right first implementation target.

## Decision

The concrete plan is:

- points become visible in napari by first becoming a raster overview image
- that image is rendered as a napari `Image` layer with the correct affine
- the first implementation milestone focuses on selected genes -> lazy overview image -> category colors
- a real napari `Points` layer is reserved for later local-detail work
- if "very snappy" behavior is required at `1e9+` scale, precomputed multiscale raster products
  should be the preferred overview representation

This gives `napari-harpy` a path to points support that is realistic, performant, and compatible with
the raster rendering direction already described in `issue_napari_spatialdata.md`.
