# Issue: `layer.metadata["adata"]` Cache And `napari-spatialdata` Coloring

## Summary

`napari-harpy` now refreshes `layer.metadata["adata"]` from the authoritative in-memory table
`sdata[table_name]`. While investigating the cache refresh path, we found two separate problems in how
`napari-spatialdata` consumes that cached `AnnData` for labels coloring:

1. integer class columns such as `pred_class` are not reliably interpreted as categorical
2. categorical coloring for labels can fail when the segmentation contains ids that are not present in the
   cached table rows

These issues explain both:

- wrong colors for `pred_class` / `user_class`
- `DirectLabelColormap` validation errors involving `float` values

## Findings

### 1. `metadata["indices"]` comes from the labels layer, not from the table

For labels layers, `napari-spatialdata` computes `layer.metadata["indices"]` from the labels element
itself via `get_element_instances(...)`.

For labels this means:

- unique label ids are collected from the image
- they are sorted
- background label `0` is dropped

So `metadata["indices"]` is the set of ids present in the segmentation, not the set of ids present in the
table.

### 2. Categorical coloring merges layer ids against `adata.obs[instance_key]`

When `napari-spatialdata` colors a categorical `obs` column for a labels layer, it:

1. starts from `layer.metadata["indices"]`
2. merges those ids against the chosen vector from `adata.obs`
3. maps category values to colors
4. builds a `DirectLabelColormap`

This means the cache `adata` must be compatible with the full set of label ids in the layer, not just the
rows currently present in the table.

### 3. Missing table rows can become `NaN` colors

If some label ids are present in `metadata["indices"]` but absent from `adata.obs[instance_key]`, the
merge produces missing values.

In the categorical code path, `napari-spatialdata` adds a fallback missing color only when it generates a
palette itself.

If `*_colors` already exists in `adata.uns`, that fallback is not added. Missing values then map to `NaN`,
and `DirectLabelColormap` receives entries like:

```python
{1: "#80808099", 2: "#ff0000", 3: np.nan}
```

This reproduces the observed error:

- `cannot convert type '<class 'float'>' to a color array`

### 4. `pred_class` is especially vulnerable to wrong type inference

`napari-spatialdata` does not always treat integer `obs` columns as categorical.

Its `_ensure_dense_vector(...)` heuristic does the following for integer series:

- binary `0/1` columns are coerced to `bool`
- small integer columns are only treated as categorical in some size/distribution cases
- otherwise they are treated as numeric vectors

This is a poor fit for Harpy's class-id semantics.

At the moment:

- `user_class` is normalized to categorical in Harpy
- `pred_class` is still stored as plain `int64`

So `pred_class` can be interpreted by `napari-spatialdata` as boolean or continuous instead of
categorical.

### 5. Harpy and `napari-spatialdata` currently do not share one explicit category/palette contract

Harpy's own viewer styling already knows that:

- `user_class` is categorical class state
- `pred_class` is also categorical class state
- `pred_confidence` is continuous

But the metadata cache exposed to `napari-spatialdata` does not yet encode that contract strongly enough.

## Root Causes

There are two distinct root causes:

### Root Cause A: semantic typing is too weak in the cache

`pred_class` is stored as integer labels instead of categorical labels with an explicit palette.

Result:

- `napari-spatialdata` applies its own heuristic
- predictions can be colored as boolean or continuous instead of categorical

### Root Cause B: the labels cache is not layer-complete

The cached `adata` may contain only the rows present in the table, while the labels layer may contain more
instance ids in `metadata["indices"]`.

Result:

- merge against `metadata["indices"]` creates missing rows
- existing `*_colors` palettes then produce `NaN` color entries
- `DirectLabelColormap` raises

## Recommended Fix Direction

### 1. Make Harpy class columns explicit in the cache

When building `layer.metadata["adata"]` for labels:

- ensure `user_class` is categorical
- ensure `pred_class` is categorical
- ensure `pred_confidence` stays numeric
- ensure matching color palettes exist in `adata.uns`

This removes reliance on `napari-spatialdata`'s integer heuristic for Harpy-owned class columns.

### 2. Make the labels cache complete with respect to layer ids

For labels layers, the cache should be safe for all ids in `layer.metadata["indices"]`.

That means:

- every label id in `metadata["indices"]` should have a corresponding cache row
- if the table has no real row for that id, Harpy should synthesize a placeholder row in the cache with
  default values

This keeps the cache compatible with `napari-spatialdata`'s merge-based coloring path.

### 3. Keep the authoritative state unchanged

This issue should be fixed in the cache-building layer only.

We should keep:

- authoritative table: `sdata[table_name]`
- derived cache: `layer.metadata["adata"]`

We should not change Harpy's source-of-truth model just to satisfy `napari-spatialdata`.

## Practical Implementation Shape

The likely fix belongs in `SpatialDataAdapter.build_layer_metadata_adata(...)`.

That helper should:

1. start from the current region-specific table view
2. normalize Harpy-owned columns for cache consumption
3. expand the cache to cover all label ids in `layer.metadata["indices"]`
4. assign default values for missing ids
5. attach explicit palettes in `uns`

The result can still remain a derived cache object whose only purpose is compatibility with
`napari-spatialdata`.

## Suggested Follow-Up

Implement a Harpy-local cache normalization step that:

- casts `pred_class` to categorical
- preserves `user_class` as categorical
- creates `pred_class_colors`
- ensures all layer ids are represented in the cache before `napari-spatialdata` consumes it

## Decision

We do not want cache behavior that branches depending on whether the layer has ids missing from the table.
We also do not want to mutate `layer.metadata["indices"]`, because that metadata should continue to describe
the actual labels layer rather than the subset represented in the table.

The decided contract is:

- keep `user_class_colors` in `sdata[table_name].uns`
- keep `pred_class_colors` in `sdata[table_name].uns`
- always make `user_class` categorical in `sdata[table_name]`
- always make `pred_class` categorical in `sdata[table_name]`
- always make `user_class` categorical in `layer.metadata["adata"]`
- always make `pred_class` categorical in `layer.metadata["adata"]`
- always remove `user_class_colors` from `layer.metadata["adata"].uns`
- always remove `pred_class_colors` from `layer.metadata["adata"].uns`

This means:

- Harpy's authoritative table keeps the explicit class palettes it needs
- the metadata cache exposed to `napari-spatialdata` always takes the same predictable code path
- `napari-spatialdata` regenerates its own categorical palette for the cache and adds its missing-value
  fallback color
- we avoid the `DirectLabelColormap` failure caused by `NaN` color entries when some layer ids are not
  represented in the table

Implementation note:

- when the metadata cache is an `AnnData` view, removing cache-only colors should be done by replacing
  `cache.uns` with a filtered mapping
- the explicit working pattern is:

```python
cache.uns = {k: v for k, v in cache.uns.items() if k not in {"user_class_colors", "pred_class_colors"}}
```

- direct `del cache.uns[...]` / `pop(...)` on a view is not reliable enough for this purpose
